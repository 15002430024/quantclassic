"""
graph_builder.py - 动态图构建器

提供多种动态图构建策略，支持在训练/推理时为每个交易日实时构建邻接矩阵。

核心功能：
1. GraphBuilder 抽象基类：定义标准接口
2. CorrGraphBuilder：基于特征相似度/相关性构建
3. IndustryGraphBuilder：基于行业分类构建（支持缓存）
4. HybridGraphBuilder：混合图 = α * corr_graph + (1-α) * industry_graph

使用示例：
    from quantclassic.data_processor.graph_builder import CorrGraphBuilder, GraphBuilderFactory
    
    # 方式1：直接创建
    builder = CorrGraphBuilder(method='pearson', top_k=10)
    adj = builder(df_day)  # 返回当日邻接矩阵
    
    # 方式2：从配置创建
    config = {
        'type': 'hybrid',
        'corr_method': 'cosine',
        'top_k': 10,
        'alpha': 0.7,
        'industry_adj_path': 'output/industry_adj_matrix.pt'
    }
    builder = GraphBuilderFactory.create(config)
    adj, stock_list, stock_to_idx = builder(df_day)
"""

import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict, Union, Any
from pathlib import Path
import logging
import yaml
from dataclasses import dataclass, field


@dataclass
class GraphBuilderConfig:
    """
    图构建器配置

    管理动态图/静态图构建所需的全部参数，支持多种相似度算法、行业图、混合图以及缓存策略。

    ⚠️ 重要提示（列名兼容性）:
        stock_col 默认为 'order_book_id'，但 DataManager 默认使用 'ts_code'。
        在使用 DailyGraphDataLoader 时，请确保 stock_col 与数据中的实际列名一致。
        
        推荐做法：
        - 从 DataConfig 透传 stock_col 到 GraphBuilderConfig
        - 或在创建 GraphBuilderConfig 时显式指定 stock_col='ts_code'

    Args:
        type (str): 图类型，默认 'corr'。
            - 'corr': 基于特征相似度/相关性的动态图
            - 'industry': 基于行业分类的静态图
            - 'hybrid': 混合图 = α * corr_graph + (1-α) * industry_graph

        stock_col (str): 股票代码列名，默认 'order_book_id'。
            ⚠️ 注意：与 DataManager(ts_code) 集成时需要显式设置为 'ts_code'。

        corr_method (str): 相关性/相似度计算方法，默认 'pearson'。
            可选 'pearson'/'spearman'/'cosine'。

        top_k (int): 每只股票保留的邻居数量，默认 10。

        threshold (float): 边权重阈值，默认 0.0。
            仅保留大于阈值的边，常用于稀疏化图结构。

        feature_cols (Optional[List[str]]): 用于计算相似度的特征列。
            - None: 自动检测所有数值列
            - 指定列表: 仅使用给定列计算相似度

        industry_col (str): 行业列名，默认 'industry_name'。

        industry_adj_path (Optional[str]): 预计算的行业邻接矩阵路径。
            若提供则直接切片使用，可显著加速历史回测。

        alpha (float): 混合图权重系数，默认 0.7。
            hybrid 图中：A = α * A_corr + (1-α) * A_industry。

        add_self_loop (bool): 是否添加自环，默认 True。
            推荐开启，使节点可以聚合自身信息。

        normalize (bool): 是否对邻接矩阵进行行归一化，默认 False。
            GAT 内部包含 softmax，通常无需额外归一化。

        symmetric (bool): 是否对称化邻接矩阵，默认 True。
            设为 False 可保留方向信息。

        cache_dir (Optional[str]): 图缓存目录，默认 None。

        enable_cache (bool): 是否启用图缓存，默认 False。
            回测场景下建议开启以避免重复构图。
    """
    # 基础配置
    type: str = 'corr'  # 'corr', 'industry', 'hybrid'
    stock_col: str = 'order_book_id'  # 兼容 ts_code
    
    # 相关性图配置
    corr_method: str = 'pearson'  # 'pearson', 'spearman', 'cosine'
    top_k: int = 10  # 每只股票选择 top_k 个邻居
    threshold: float = 0.0  # 相关性阈值（仅保留 > threshold 的边）
    feature_cols: Optional[List[str]] = None  # 用于计算相似度的特征列
    
    # 行业图配置
    industry_col: str = 'industry_name'
    industry_adj_path: Optional[str] = None  # 预计算的行业邻接矩阵路径
    
    # 混合图配置
    alpha: float = 0.7  # hybrid: α * corr_graph + (1-α) * industry_graph
    
    # 通用配置
    add_self_loop: bool = True
    normalize: bool = False  # 行归一化（GAT 内部会做 softmax，通常不需要）
    symmetric: bool = True  # 是否对称化邻接矩阵
    
    # 缓存配置
    cache_dir: Optional[str] = None  # 图缓存目录
    enable_cache: bool = False  # 是否启用缓存
    
    def adapt_stock_col(self, df: 'pd.DataFrame') -> str:
        """
        根据数据自动适配 stock_col
        
        Args:
            df: 数据框
            
        Returns:
            实际使用的股票代码列名
        """
        if self.stock_col in df.columns:
            return self.stock_col
        
        # 尝试常见的列名
        candidates = ['order_book_id', 'ts_code', 'stock_code', 'symbol', 'code']
        for col in candidates:
            if col in df.columns:
                self.stock_col = col
                return col
        
        raise ValueError(f"未找到股票代码列，尝试了: {candidates}")
    
    @classmethod
    def from_data_config(cls, data_config: Any, **kwargs) -> 'GraphBuilderConfig':
        """
        从 DataConfig 创建 GraphBuilderConfig，自动透传 stock_col
        
        Args:
            data_config: DataConfig 实例（来自 data_set.config）
            **kwargs: 其他 GraphBuilderConfig 参数
            
        Returns:
            GraphBuilderConfig 实例
            
        示例:
            from data_set import DataConfig
            from data_processor.graph_builder import GraphBuilderConfig
            
            data_config = DataConfig(stock_col='ts_code')
            graph_config = GraphBuilderConfig.from_data_config(data_config, type='hybrid', top_k=10)
        """
        # 从 data_config 提取相关字段
        stock_col = getattr(data_config, 'stock_col', 'ts_code')
        
        # 合并参数
        merged_kwargs = {'stock_col': stock_col, **kwargs}
        
        return cls(**merged_kwargs)
    
    def to_dict(self) -> Dict:
        """转为字典"""
        return {
            'type': self.type,
            'stock_col': self.stock_col,
            'corr_method': self.corr_method,
            'top_k': self.top_k,
            'threshold': self.threshold,
            'feature_cols': self.feature_cols,
            'industry_col': self.industry_col,
            'industry_adj_path': self.industry_adj_path,
            'alpha': self.alpha,
            'add_self_loop': self.add_self_loop,
            'normalize': self.normalize,
            'symmetric': self.symmetric,
            'cache_dir': self.cache_dir,
            'enable_cache': self.enable_cache,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'GraphBuilderConfig':
        """从字典创建"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, yaml_path: Union[str, Path], **dump_kwargs) -> None:
        """将配置写入 YAML 文件"""
        yaml_file = Path(yaml_path)
        yaml_file.parent.mkdir(parents=True, exist_ok=True)

        with yaml_file.open('w', encoding='utf-8') as handle:
            yaml.safe_dump(
                self.to_dict(),
                handle,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                **dump_kwargs,
            )

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'GraphBuilderConfig':
        """从 YAML 文件加载配置"""
        yaml_file = Path(yaml_path)

        if not yaml_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {yaml_file}")

        with yaml_file.open('r', encoding='utf-8') as handle:
            config_dict = yaml.safe_load(handle) or {}

        return cls.from_dict(config_dict)


class GraphBuilder(ABC):
    """
    图构建器抽象基类
    
    所有图构建器必须实现 __call__ 方法，接收当日数据，返回邻接矩阵。
    
    ⚠️ 这是构建图的统一入口（canonical path）。
    model/build_industry_adj.py 等工具已标记为 deprecated，仅作为静态基准使用。
    """
    
    def __init__(self, config: Optional[GraphBuilderConfig] = None, **kwargs):
        if config is not None:
            self.config = config
        else:
            self.config = GraphBuilderConfig(**kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def __call__(
        self,
        df_day: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
        """
        构建当日邻接矩阵
        
        Args:
            df_day: 当日所有股票的 DataFrame（每行一只股票）
            feature_cols: 特征列（用于计算相似度）
            
        Returns:
            adj_matrix: [N, N] 邻接矩阵 (torch.Tensor)
            stock_list: 股票代码列表（顺序与矩阵索引对应）
            stock_to_idx: 股票代码到索引的映射字典
        """
        pass
    
    def to_edge_index(
        self, 
        adj: torch.Tensor,
        threshold: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将邻接矩阵转换为 PyTorch Geometric 的 edge_index 格式
        
        Args:
            adj: [N, N] 邻接矩阵
            threshold: 边权重阈值
            
        Returns:
            edge_index: [2, E] 边索引
            edge_weight: [E] 边权重
        """
        # 获取非零边
        mask = adj > threshold
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()  # [2, E]
        edge_weight = adj[mask]  # [E]
        
        return edge_index, edge_weight
    
    def _add_self_loop(self, adj: torch.Tensor) -> torch.Tensor:
        """添加自环"""
        adj = adj.clone()
        adj.fill_diagonal_(1.0)
        return adj
    
    def _normalize(self, adj: torch.Tensor) -> torch.Tensor:
        """行归一化"""
        row_sum = adj.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1
        return adj / row_sum
    
    def _symmetrize(self, adj: torch.Tensor) -> torch.Tensor:
        """对称化"""
        return (adj + adj.T) / 2


class CorrGraphBuilder(GraphBuilder):
    """
    基于特征相似度/相关性的动态图构建器
    
    支持的相似度计算方法：
    - pearson: 皮尔逊相关系数
    - spearman: 斯皮尔曼等级相关
    - cosine: 余弦相似度
    
    Args:
        method: 相似度计算方法
        top_k: 每只股票选择 top_k 个邻居
        threshold: 相关性阈值
        **kwargs: 传递给 GraphBuilderConfig 的其他参数
    """
    
    def __init__(
        self,
        method: str = 'pearson',
        top_k: int = 10,
        threshold: float = 0.0,
        **kwargs
    ):
        config = GraphBuilderConfig(
            type='corr',
            corr_method=method,
            top_k=top_k,
            threshold=threshold,
            **kwargs
        )
        super().__init__(config)
    
    def __call__(
        self,
        df_day: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
        """
        构建基于特征相似度的邻接矩阵
        
        Args:
            df_day: 当日所有股票数据（每行一只股票）
            feature_cols: 特征列（默认使用 config 中的设置）
            
        Returns:
            adj_matrix, stock_list, stock_to_idx
        """
        # 自适应股票代码列名（兼容 ts_code 和 order_book_id）
        stock_col = self.config.adapt_stock_col(df_day)
        feature_cols = feature_cols or self.config.feature_cols
        
        # 获取股票列表
        stock_list = sorted(df_day[stock_col].unique().tolist())
        n_stocks = len(stock_list)
        stock_to_idx = {s: i for i, s in enumerate(stock_list)}
        
        if n_stocks == 0:
            return torch.zeros(0, 0), [], {}
        
        if n_stocks == 1:
            adj = torch.ones(1, 1) if self.config.add_self_loop else torch.zeros(1, 1)
            return adj, stock_list, stock_to_idx
        
        # 提取特征矩阵
        if feature_cols is None:
            # 自动检测数值列
            numeric_cols = df_day.select_dtypes(include=[np.number]).columns.tolist()
            # 排除 ID 列
            feature_cols = [c for c in numeric_cols if c not in [stock_col, 'trade_date']]
        
        # 确保按股票排序
        df_sorted = df_day.set_index(stock_col).reindex(stock_list)
        features = df_sorted[feature_cols].values  # [N, F]
        
        # 处理 NaN
        features = np.nan_to_num(features, nan=0.0)
        
        # 计算相似度矩阵
        if self.config.corr_method == 'cosine':
            sim_matrix = self._cosine_similarity(features)
        elif self.config.corr_method == 'pearson':
            sim_matrix = self._pearson_correlation(features)
        elif self.config.corr_method == 'spearman':
            sim_matrix = self._spearman_correlation(features)
        else:
            raise ValueError(f"不支持的相似度方法: {self.config.corr_method}")
        
        # 构建 Top-K 邻接矩阵
        adj_matrix = self._build_topk_adj(sim_matrix, self.config.top_k)
        
        # 应用阈值
        adj_matrix[adj_matrix < self.config.threshold] = 0
        
        # 对称化
        if self.config.symmetric:
            adj_matrix = self._symmetrize(adj_matrix)
        
        # 添加自环
        if self.config.add_self_loop:
            adj_matrix = self._add_self_loop(adj_matrix)
        
        # 行归一化
        if self.config.normalize:
            adj_matrix = self._normalize(adj_matrix)
        
        return adj_matrix, stock_list, stock_to_idx
    
    def _cosine_similarity(self, features: np.ndarray) -> torch.Tensor:
        """计算余弦相似度矩阵"""
        # 归一化
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = features / norms
        
        # 点积即为余弦相似度
        sim = np.dot(normalized, normalized.T)
        return torch.from_numpy(sim).float()
    
    def _pearson_correlation(self, features: np.ndarray) -> torch.Tensor:
        """计算皮尔逊相关系数矩阵"""
        # 标准化
        mean = features.mean(axis=1, keepdims=True)
        std = features.std(axis=1, keepdims=True)
        std[std == 0] = 1
        normalized = (features - mean) / std
        
        # 相关系数 = 标准化后的点积 / 特征数
        n_features = features.shape[1]
        corr = np.dot(normalized, normalized.T) / n_features
        return torch.from_numpy(corr).float()
    
    def _spearman_correlation(self, features: np.ndarray) -> torch.Tensor:
        """计算斯皮尔曼等级相关系数矩阵"""
        from scipy.stats import rankdata
        
        # 转换为排名
        ranked = np.apply_along_axis(rankdata, 1, features)
        
        # 对排名计算皮尔逊相关
        return self._pearson_correlation(ranked)
    
    def _build_topk_adj(self, sim_matrix: torch.Tensor, top_k: int) -> torch.Tensor:
        """构建 Top-K 邻接矩阵"""
        n = sim_matrix.size(0)
        adj = torch.zeros_like(sim_matrix)
        
        for i in range(n):
            sim_i = sim_matrix[i].clone()
            sim_i[i] = float('-inf')  # 排除自己
            
            # 获取 top-k 索引
            k = min(top_k, n - 1)
            _, top_indices = torch.topk(sim_i, k)
            
            # 设置边权重
            for j in top_indices:
                if sim_matrix[i, j] > 0:  # 只保留正相关
                    adj[i, j] = sim_matrix[i, j]
        
        return adj


class IndustryGraphBuilder(GraphBuilder):
    """
    基于行业分类的图构建器
    
    同行业股票互相连接，不同行业不连接。
    支持从预计算的行业邻接矩阵加载（提高效率）。
    
    🆕 支持传入全局股票-行业映射 (stock_industry_mapping)，
    这样 df_day 只需要包含股票ID，不需要包含行业列。
    """
    
    def __init__(
        self,
        industry_col: str = 'industry_name',
        industry_adj_path: Optional[str] = None,
        stock_industry_mapping: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        config = GraphBuilderConfig(
            type='industry',
            industry_col=industry_col,
            industry_adj_path=industry_adj_path,
            **kwargs
        )
        super().__init__(config)
        
        # 缓存预计算的行业邻接矩阵
        self._cached_adj = None
        self._cached_stock_list = None
        self._cached_stock_to_idx = None
        
        # 🆕 全局股票-行业映射（用于动态模式）
        self._stock_industry_mapping = stock_industry_mapping or {}
        
        # 如果提供了路径，预加载
        if industry_adj_path and Path(industry_adj_path).exists():
            self._load_cached_adj(industry_adj_path)
    
    def set_stock_industry_mapping(self, mapping: Dict[str, str]):
        """设置全局股票-行业映射"""
        self._stock_industry_mapping = mapping
        self.logger.info(f"已设置股票-行业映射，共 {len(mapping)} 只股票")
    
    def _load_cached_adj(self, path: str):
        """加载预计算的行业邻接矩阵"""
        data = torch.load(path, map_location='cpu')
        self._cached_adj = data['adj_matrix']
        self._cached_stock_list = data['stock_list']
        self._cached_stock_to_idx = data['stock_to_idx']
        self.logger.info(f"加载预计算行业邻接矩阵: {path}")
    
    def __call__(
        self,
        df_day: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
        """
        构建基于行业的邻接矩阵
        
        如果已加载预计算矩阵，则根据当日股票子集进行切片。
        否则动态构建。
        """
        # 自适应股票代码列名
        stock_col = self.config.adapt_stock_col(df_day)
        industry_col = self.config.industry_col
        
        # 获取当日股票列表
        stock_list = sorted(df_day[stock_col].unique().tolist())
        n_stocks = len(stock_list)
        stock_to_idx = {s: i for i, s in enumerate(stock_list)}
        
        if n_stocks == 0:
            return torch.zeros(0, 0), [], {}
        
        # 如果有缓存，从缓存中切片
        if self._cached_adj is not None:
            return self._slice_from_cache(stock_list, stock_to_idx)
        
        # 动态构建
        return self._build_from_scratch(df_day, stock_list, stock_to_idx, industry_col)
    
    def _slice_from_cache(
        self, 
        stock_list: List[str],
        stock_to_idx: Dict[str, int]
    ) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
        """从缓存的全局矩阵中切片（向量化优化版）"""
        n = len(stock_list)
        
        # 构建索引映射（向量化）
        indices = []
        valid_mask = []
        for s in stock_list:
            if s in self._cached_stock_to_idx:
                indices.append(self._cached_stock_to_idx[s])
                valid_mask.append(True)
            else:
                indices.append(0)  # 占位
                valid_mask.append(False)
        
        indices = torch.tensor(indices, dtype=torch.long)
        valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
        
        # 【关键优化】使用高级索引一次性切片，避免双重 for 循环
        # 这是 O(N²) 的内存操作，但比 Python 循环快 100 倍
        adj = self._cached_adj[indices][:, indices]
        
        # 将无效股票的行列置零
        if not valid_mask.all():
            invalid_mask = ~valid_mask
            adj[invalid_mask, :] = 0
            adj[:, invalid_mask] = 0
        
        return adj, stock_list, stock_to_idx
    
    def _build_from_scratch(
        self,
        df_day: pd.DataFrame,
        stock_list: List[str],
        stock_to_idx: Dict[str, int],
        industry_col: str
    ) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
        """
        从头构建行业邻接矩阵（向量化优化版）
        
        🆕 优先使用全局股票-行业映射 (_stock_industry_mapping)，
        如果未设置，则回退到 df_day 中的行业列。
        """
        stock_col = self.config.stock_col
        n = len(stock_list)
        
        # 🆕 优先使用全局映射，回退到 df_day 列
        if self._stock_industry_mapping:
            # 使用预设的全局映射
            stock_to_industry = self._stock_industry_mapping
        elif industry_col in df_day.columns:
            # 回退：从 df_day 构建映射
            stock_to_industry = dict(zip(
                df_day[stock_col],
                df_day[industry_col]
            ))
        else:
            # 无行业信息，返回全连接图（仅自环）
            self.logger.warning(f"未找到行业列 '{industry_col}'，返回仅自环邻接矩阵")
            adj = torch.eye(n, dtype=torch.float32)
            return adj, stock_list, stock_to_idx
        
        # 【优化】向量化构建：将行业转为整数编码，然后用广播比较
        # 获取每只股票的行业
        industries = [stock_to_industry.get(s, None) for s in stock_list]
        
        # 编码行业为整数（None 编码为 -1）
        unique_industries = list(set(ind for ind in industries if ind is not None))
        ind_to_code = {ind: i for i, ind in enumerate(unique_industries)}
        ind_codes = torch.tensor([ind_to_code.get(ind, -1) if ind is not None else -1 for ind in industries])
        
        # 使用广播比较：adj[i,j] = 1 if ind_codes[i] == ind_codes[j] and ind_codes[i] != -1
        # 扩展为 [N, 1] 和 [1, N] 进行广播
        ind_codes_row = ind_codes.unsqueeze(1)  # [N, 1]
        ind_codes_col = ind_codes.unsqueeze(0)  # [1, N]
        
        # 同行业为 1，不同行业或无效为 0
        adj = ((ind_codes_row == ind_codes_col) & (ind_codes_row >= 0)).float()
        
        # 添加自环
        if self.config.add_self_loop:
            adj = self._add_self_loop(adj)
        
        return adj, stock_list, stock_to_idx


class HybridGraphBuilder(GraphBuilder):
    """
    混合图构建器（动态图核心实现）
    
    ⚠️ 适用场景: 同时考虑特征相似度和行业结构的股票关系建模
    
    构建策略:
        A_hybrid = α * A_corr + (1-α) * A_industry
        
        其中:
        - A_corr: 基于特征相似度(余弦/皮尔逊)的相关性邻接矩阵
        - A_industry: 基于行业分类的邻接矩阵(同行业=1, 不同行业=0)
        - α: 混合系数 (0~1), 控制相关性图和行业图的权重比例
    
    实现原理:
    1. 调用 CorrGraphBuilder 计算当日股票特征相似度矩阵
    2. 调用 IndustryGraphBuilder 加载/构建行业邻接矩阵
    3. 按权重 α 混合两种图结构
    4. 可选: 添加自环、归一化
    
    Args:
        alpha: 混合系数 (默认 0.7)
               - α=1.0: 纯相关性图
               - α=0.0: 纯行业图
               - α=0.6~0.8: 推荐值，平衡两种结构
        corr_method: 相关性计算方法 ('cosine', 'pearson', 'spearman')
        top_k: 每只股票保留的最相似邻居数量
        industry_col: 行业列名
        industry_adj_path: 预计算的行业邻接矩阵路径（可选，用于加速）
        **kwargs: 传递给 GraphBuilderConfig 的其他参数
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        corr_method: str = 'cosine',
        top_k: int = 10,
        industry_col: str = 'industry_name',
        industry_adj_path: Optional[str] = None,
        **kwargs
    ):
        config = GraphBuilderConfig(
            type='hybrid',
            alpha=alpha,
            corr_method=corr_method,
            top_k=top_k,
            industry_col=industry_col,
            industry_adj_path=industry_adj_path,
            **kwargs
        )
        super().__init__(config)
        
        # 子构建器
        self.corr_builder = CorrGraphBuilder(
            method=corr_method,
            top_k=top_k,
            stock_col=config.stock_col,
            add_self_loop=False,  # 最后统一添加
            normalize=False,
            symmetric=True
        )
        
        self.industry_builder = IndustryGraphBuilder(
            industry_col=industry_col,
            industry_adj_path=industry_adj_path,
            stock_col=config.stock_col,
            add_self_loop=False,
            normalize=False
        )
    
    def __call__(
        self,
        df_day: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
        """
        构建混合邻接矩阵（动态图每批次调用）
        
        ⚠️ 注意: 此方法在每个训练批次中被调用，确保图结构随数据动态更新
        
        执行流程:
        1. 调用 corr_builder 计算当日股票的特征相似度图
        2. 调用 industry_builder 生成/加载行业邻接矩阵
        3. 按 α 混合: A = α*A_corr + (1-α)*A_industry
        4. 后处理: 添加自环、归一化
        
        Args:
            df_day: 当日所有股票的特征数据 (每行一只股票)
                   必须包含: stock_col, feature_cols, industry_col
            feature_cols: 用于计算相似度的特征列
        
        Returns:
            adj_matrix: [N, N] 混合邻接矩阵
            stock_list: 股票代码列表（顺序与矩阵索引对应）
            stock_to_idx: 股票代码到索引的映射字典
        """
        alpha = self.config.alpha
        
        # =================================================================
        # 步骤1: 构建相关性图 (基于特征相似度)
        # =================================================================
        adj_corr, stock_list, stock_to_idx = self.corr_builder(df_day, feature_cols)
        
        # =================================================================
        # 步骤2: 构建行业图 (基于行业分类)
        # =================================================================
        adj_industry, _, _ = self.industry_builder(df_day, feature_cols)
        
        # 确保尺寸一致
        assert adj_corr.shape == adj_industry.shape, \
            f"矩阵尺寸不匹配: {adj_corr.shape} vs {adj_industry.shape}"
        
        # =================================================================
        # 步骤3: 混合两种图结构
        # 例如: α=0.7 时，70% 相关性 + 30% 行业关系
        # =================================================================
        adj = alpha * adj_corr + (1 - alpha) * adj_industry
        
        # =================================================================
        # 步骤4: 后处理
        # =================================================================
        # 添加自环 (允许节点接收自身信息)
        if self.config.add_self_loop:
            adj = self._add_self_loop(adj)
        
        # 行归一化 (可选，GAT 内部有 softmax 通常不需要)
        if self.config.normalize:
            adj = self._normalize(adj)
        
        return adj, stock_list, stock_to_idx


class CachedGraphBuilder(GraphBuilder):
    """
    带缓存的图构建器包装器
    
    将构建好的图缓存到磁盘，避免重复计算。
    适用于历史数据回测场景。
    """
    
    def __init__(
        self,
        base_builder: GraphBuilder,
        cache_dir: str,
        time_col: str = 'trade_date'
    ):
        super().__init__()
        self.base_builder = base_builder
        self.cache_dir = Path(cache_dir)
        self.time_col = time_col
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def __call__(
        self,
        df_day: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
        """优先从缓存加载，否则构建并缓存"""
        # 获取日期作为缓存键
        if self.time_col in df_day.columns:
            date = df_day[self.time_col].iloc[0]
            if hasattr(date, 'strftime'):
                date_str = date.strftime('%Y%m%d')
            else:
                date_str = str(date)
        else:
            # 无日期列，使用数据哈希
            date_str = f"hash_{hash(tuple(df_day.index.tolist()))}"
        
        cache_path = self.cache_dir / f"{date_str}.pt"
        
        # 尝试从缓存加载
        if cache_path.exists():
            data = torch.load(cache_path, map_location='cpu')
            return data['adj'], data['stock_list'], data['stock_to_idx']
        
        # 构建
        adj, stock_list, stock_to_idx = self.base_builder(df_day, feature_cols)
        
        # 缓存
        torch.save({
            'adj': adj,
            'stock_list': stock_list,
            'stock_to_idx': stock_to_idx
        }, cache_path)
        
        return adj, stock_list, stock_to_idx


class AdjMatrixUtils:
    """
    邻接矩阵工具类
    
    提供与旧版 AdjMatrixBuilder 兼容的静态方法，便于迁移。
    主要用于 rolling_trainer.py 中的 _build_adj_matrix 方法。
    
    使用示例:
        from quantclassic.data_processor.graph_builder import AdjMatrixUtils
        
        # 基于收益率相关性构建
        adj = AdjMatrixUtils.build_correlation_adj(
            returns_pivot,  # DataFrame: [dates x stocks]
            top_k=10,
            method='pearson'
        )
        
        # 基于行业构建
        adj = AdjMatrixUtils.build_industry_adj(industry_codes)
    """
    
    @staticmethod
    def build_correlation_adj(
        returns: pd.DataFrame,
        top_k: int = 10,
        method: str = 'pearson',
        self_loop: bool = True,
        threshold: Optional[float] = None
    ) -> torch.Tensor:
        """
        基于收益率相关性构建邻接矩阵
        
        兼容旧版 AdjMatrixBuilder.build_correlation_adj
        
        Args:
            returns: 收益率矩阵 DataFrame，shape=[time, stocks]
            top_k: 每只股票选取的邻居数量
            method: 相关性计算方法 ('pearson', 'spearman')
            self_loop: 是否包含自环
            threshold: 相关性阈值
            
        Returns:
            adj: [N, N] 邻接矩阵
        """
        # 计算相关性矩阵
        if method == 'pearson':
            corr = returns.corr(method='pearson')
        elif method == 'spearman':
            corr = returns.corr(method='spearman')
        else:
            raise ValueError(f"不支持的相关性方法: {method}")
        
        corr_values = corr.values
        n_stocks = corr_values.shape[0]
        
        # 初始化邻接矩阵
        adj = torch.zeros(n_stocks, n_stocks)
        
        # 对每只股票，选取相关性最高的 top_k 个邻居
        for i in range(n_stocks):
            corr_i = corr_values[i].copy()
            corr_i[i] = -np.inf  # 暂时排除自己
            
            # 选取 top_k 个最相关的股票
            k = min(top_k, n_stocks - 1)
            top_k_indices = np.argsort(corr_i)[-k:]
            
            # 设置边
            for j in top_k_indices:
                if threshold is None or corr_values[i, j] >= threshold:
                    adj[i, j] = 1.0
        
        # 对称化邻接矩阵（无向图）
        adj = torch.maximum(adj, adj.T)
        
        # 添加自环
        if self_loop:
            adj.fill_diagonal_(1)
        
        return adj
    
    @staticmethod
    def build_industry_adj(
        industry_codes: Union[List[str], pd.Series],
        self_loop: bool = True
    ) -> torch.Tensor:
        """
        基于行业分类构建邻接矩阵
        
        兼容旧版 AdjMatrixBuilder.build_industry_adj
        
        Args:
            industry_codes: 股票的行业代码列表
            self_loop: 是否包含自环
            
        Returns:
            adj: [N, N] 邻接矩阵
        """
        if isinstance(industry_codes, pd.Series):
            industry_codes = industry_codes.tolist()
        
        n_stocks = len(industry_codes)
        adj = torch.zeros(n_stocks, n_stocks)
        
        # 构建行业映射
        for i in range(n_stocks):
            for j in range(n_stocks):
                if industry_codes[i] == industry_codes[j]:
                    adj[i, j] = 1.0
        
        # 移除自环（如果需要）
        if not self_loop:
            adj.fill_diagonal_(0)
        
        return adj
    
    @staticmethod
    def build_weighted_adj(
        returns: pd.DataFrame,
        top_k: int = 10,
        method: str = 'pearson',
        self_loop: bool = True
    ) -> torch.Tensor:
        """
        构建加权邻接矩阵（边权重为相关性值）
        
        Args:
            returns: 收益率矩阵
            top_k: 每只股票选取的邻居数量
            method: 相关性计算方法
            self_loop: 是否包含自环
            
        Returns:
            adj: [N, N] 加权邻接矩阵
        """
        # 计算相关性矩阵
        if method == 'pearson':
            corr = returns.corr(method='pearson')
        elif method == 'spearman':
            corr = returns.corr(method='spearman')
        else:
            raise ValueError(f"不支持的相关性方法: {method}")
        
        corr_values = corr.values
        n_stocks = corr_values.shape[0]
        
        # 初始化加权邻接矩阵
        adj = torch.zeros(n_stocks, n_stocks)
        
        # 对每只股票，选取相关性最高的 top_k 个邻居
        for i in range(n_stocks):
            corr_i = corr_values[i].copy()
            corr_i[i] = -np.inf
            
            k = min(top_k, n_stocks - 1)
            top_k_indices = np.argsort(corr_i)[-k:]
            
            # 设置边权重为相关性绝对值
            for j in top_k_indices:
                adj[i, j] = abs(corr_values[i, j])
        
        # 对称化
        adj = torch.maximum(adj, adj.T)
        
        # 添加自环
        if self_loop:
            adj.fill_diagonal_(1)
        
        return adj
    
    @staticmethod
    def save_adj_matrix(adj: torch.Tensor, save_path: str, format: str = 'pt'):
        """保存邻接矩阵"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pt':
            torch.save(adj, save_path)
        elif format == 'npy':
            np.save(save_path, adj.numpy())
        else:
            raise ValueError(f"不支持的保存格式: {format}")
    
    @staticmethod
    def load_adj_matrix(load_path: str) -> torch.Tensor:
        """加载邻接矩阵"""
        load_path = Path(load_path)
        
        if load_path.suffix in ['.pt', '.pth']:
            return torch.load(load_path, map_location='cpu')
        elif load_path.suffix == '.npy':
            return torch.from_numpy(np.load(load_path)).float()
        else:
            raise ValueError(f"不支持的文件格式: {load_path.suffix}")


class GraphBuilderFactory:
    """
    图构建器工厂
    
    ⚠️ 这是构建动态图的统一入口（canonical path）。
    训练和评估应统一使用此工厂创建图构建器。
    model/build_industry_adj.py 等工具已标记为 deprecated。
    
    与 DataManager 集成示例:
        from data_set import DataConfig
        from data_processor.graph_builder import GraphBuilderConfig, GraphBuilderFactory
        
        # 方式1: 从 DataConfig 透传 stock_col
        data_config = DataConfig(stock_col='ts_code')
        graph_config = GraphBuilderConfig.from_data_config(data_config, type='hybrid', top_k=10)
        builder = GraphBuilderFactory.create(graph_config)
        
        # 方式2: 手动指定（确保与数据列名一致）
        builder = GraphBuilderFactory.create({
            'type': 'hybrid',
            'stock_col': 'ts_code',  # 与 DataManager 一致
            'top_k': 10
        })
    """
    
    _registry = {
        'corr': CorrGraphBuilder,
        'correlation': CorrGraphBuilder,
        'industry': IndustryGraphBuilder,
        'hybrid': HybridGraphBuilder,
    }
    
    @classmethod
    def create(
        cls,
        config: Union[Dict, GraphBuilderConfig, str],
        data_config: Any = None
    ) -> GraphBuilder:
        """
        创建图构建器
        
        Args:
            config: 配置字典、GraphBuilderConfig 或类型字符串
            data_config: 可选，DataConfig 实例，用于自动透传 stock_col
            
        Returns:
            GraphBuilder 实例
        """
        if isinstance(config, str):
            config = {'type': config}
        
        # 如果提供了 data_config，透传 stock_col
        if data_config is not None and isinstance(config, dict):
            stock_col = getattr(data_config, 'stock_col', 'ts_code')
            if 'stock_col' not in config:
                config['stock_col'] = stock_col
        
        if isinstance(config, dict):
            builder_type = config.get('type', 'corr')
        else:
            builder_type = config.type
        
        if builder_type not in cls._registry:
            raise ValueError(f"未知的图构建器类型: {builder_type}. "
                           f"可选: {list(cls._registry.keys())}")
        
        builder_class = cls._registry[builder_type]
        
        if isinstance(config, dict):
            # 从字典创建
            # 过滤掉 'type' 键
            kwargs = {k: v for k, v in config.items() if k != 'type'}
            
            if builder_type == 'corr':
                return builder_class(
                    method=kwargs.pop('corr_method', 'pearson'),
                    top_k=kwargs.pop('top_k', 10),
                    threshold=kwargs.pop('threshold', 0.0),
                    **kwargs
                )
            elif builder_type == 'industry':
                return builder_class(
                    industry_col=kwargs.pop('industry_col', 'industry_name'),
                    industry_adj_path=kwargs.pop('industry_adj_path', None),
                    stock_industry_mapping=kwargs.pop('stock_industry_mapping', None),
                    **kwargs
                )
            elif builder_type == 'hybrid':
                return builder_class(
                    alpha=kwargs.pop('alpha', 0.7),
                    corr_method=kwargs.pop('corr_method', 'cosine'),
                    top_k=kwargs.pop('top_k', 10),
                    industry_col=kwargs.pop('industry_col', 'industry_name'),
                    industry_adj_path=kwargs.pop('industry_adj_path', None),
                    **kwargs
                )
        else:
            # 从 GraphBuilderConfig 创建
            return builder_class(config=config)
    
    @classmethod
    def register(cls, name: str, builder_class: type):
        """注册新的图构建器类型"""
        cls._registry[name] = builder_class


# ==================== 单元测试 ====================

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("GraphBuilder 单元测试")
    print("=" * 80)
    
    # 创建测试数据：2天，每天5只股票
    np.random.seed(42)
    
    dates = ['2024-01-01', '2024-01-02']
    stocks = ['000001', '000002', '000003', '000004', '000005']
    industries = ['银行', '银行', '科技', '科技', '消费']
    
    rows = []
    for date in dates:
        for i, stock in enumerate(stocks):
            rows.append({
                'trade_date': date,
                'order_book_id': stock,
                'industry_name': industries[i],
                'feature1': np.random.randn(),
                'feature2': np.random.randn(),
                'feature3': np.random.randn(),
                'close': 10 + np.random.randn(),
            })
    
    df = pd.DataFrame(rows)
    print(f"\n📊 测试数据: {len(df)} 行, {len(stocks)} 只股票, {len(dates)} 天")
    
    # 测试 1: CorrGraphBuilder
    print("\n【测试 1: CorrGraphBuilder】")
    corr_builder = CorrGraphBuilder(method='cosine', top_k=2)
    
    df_day1 = df[df['trade_date'] == dates[0]]
    adj, stock_list, stock_to_idx = corr_builder(df_day1, feature_cols=['feature1', 'feature2', 'feature3'])
    
    print(f"  股票列表: {stock_list}")
    print(f"  邻接矩阵形状: {adj.shape}")
    print(f"  非零边数: {(adj > 0).sum().item()}")
    assert adj.shape == (5, 5), "邻接矩阵尺寸错误"
    print("  ✓ CorrGraphBuilder 测试通过")
    
    # 测试 2: IndustryGraphBuilder
    print("\n【测试 2: IndustryGraphBuilder】")
    industry_builder = IndustryGraphBuilder(industry_col='industry_name')
    
    adj_ind, stock_list, _ = industry_builder(df_day1)
    print(f"  邻接矩阵:\n{adj_ind}")
    
    # 验证：银行(0,1)互连，科技(2,3)互连
    assert adj_ind[0, 1] == 1.0 and adj_ind[1, 0] == 1.0, "银行股应该互连"
    assert adj_ind[2, 3] == 1.0 and adj_ind[3, 2] == 1.0, "科技股应该互连"
    assert adj_ind[0, 4] == 0.0, "银行和消费不应连接"
    print("  ✓ IndustryGraphBuilder 测试通过")
    
    # 测试 3: HybridGraphBuilder
    print("\n【测试 3: HybridGraphBuilder】")
    hybrid_builder = HybridGraphBuilder(
        alpha=0.5,
        corr_method='cosine',
        top_k=2,
        industry_col='industry_name'
    )
    
    adj_hybrid, _, _ = hybrid_builder(df_day1, feature_cols=['feature1', 'feature2', 'feature3'])
    print(f"  混合邻接矩阵形状: {adj_hybrid.shape}")
    print("  ✓ HybridGraphBuilder 测试通过")
    
    # 测试 4: GraphBuilderFactory
    print("\n【测试 4: GraphBuilderFactory】")
    config = {
        'type': 'hybrid',
        'alpha': 0.7,
        'corr_method': 'pearson',
        'top_k': 3,
    }
    builder = GraphBuilderFactory.create(config)
    adj, _, _ = builder(df_day1, feature_cols=['feature1', 'feature2', 'feature3'])
    print(f"  工厂创建的构建器类型: {type(builder).__name__}")
    print("  ✓ GraphBuilderFactory 测试通过")
    
    # 测试 5: edge_index 转换
    print("\n【测试 5: edge_index 转换】")
    edge_index, edge_weight = corr_builder.to_edge_index(adj, threshold=0.1)
    print(f"  edge_index 形状: {edge_index.shape}")
    print(f"  edge_weight 长度: {len(edge_weight)}")
    print("  ✓ edge_index 转换测试通过")
    
    # 测试 6: 每日独立构建
    print("\n【测试 6: 每日独立构建】")
    for date in dates:
        df_day = df[df['trade_date'] == date]
        adj, stocks, _ = corr_builder(df_day, feature_cols=['feature1', 'feature2'])
        print(f"  {date}: {len(stocks)} 只股票, 邻接矩阵 {adj.shape}")
    print("  ✓ 每日独立构建测试通过")
    
    print("\n" + "=" * 80)
    print("✅ 所有 GraphBuilder 测试通过！")
    print("=" * 80)
