"""
daily_graph_loader.py - 日批次数据加载器

实现按交易日组织的数据加载，每个 batch 包含当日所有股票。
支持动态图构建：在 collate_fn 中调用 GraphBuilder 生成邻接矩阵。

核心组件：
1. DailyBatchDataset: 按日组织数据，每个样本是一天的所有股票
2. collate_daily: 自定义 collate_fn，转换为模型输入格式
3. DailyGraphDataLoader: 封装 DataLoader，提供便捷接口

使用示例：
    from quantclassic.data_set.graph import (
        DailyBatchDataset, DailyGraphDataLoader
    )
    from quantclassic.data_processor.graph_builder import CorrGraphBuilder
    
    # 创建数据集
    dataset = DailyBatchDataset(
        df=df,
        feature_cols=feature_cols,
        label_col='alpha_label',
        window_size=20,
        time_col='trade_date',
        stock_col='order_book_id'
    )
    
    # 创建图构建器
    graph_builder = CorrGraphBuilder(method='cosine', top_k=10)
    
    # 创建数据加载器
    loader = DailyGraphDataLoader(
        dataset=dataset,
        graph_builder=graph_builder,
        shuffle_dates=True
    )
    
    # 训练循环
    for batch in loader:
        X, y, adj, stock_ids, date = batch
        logits = model(X, adj)
        loss = criterion(logits, y)
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path

try:
    from ...config.base_config import BaseConfig
except ImportError:
    from config.base_config import BaseConfig


@dataclass
class DailyLoaderConfig(BaseConfig):
    """日批次加载器配置"""
    # 基础配置
    window_size: int = 20
    feature_cols: Optional[List[str]] = None
    label_col: str = 'alpha_label'
    stock_col: str = 'order_book_id'
    time_col: str = 'trade_date'
    
    # 窗口变换
    enable_window_transform: bool = True
    window_price_log: bool = True
    window_volume_norm: bool = True
    price_cols: List[str] = field(default_factory=lambda: ['open', 'high', 'low', 'close', 'vwap'])
    close_col: str = 'close'
    volume_cols: List[str] = field(default_factory=lambda: ['vol', 'amount'])
    
    # 标签处理
    label_rank_normalize: bool = True
    label_rank_output_range: Tuple[float, float] = (-1, 1)
    
    # 加载器配置
    shuffle_dates: bool = True
    num_workers: int = 0  # 日批次模式建议单进程
    pin_memory: bool = True
    
    # 图构建配置（可选）
    graph_builder_config: Optional[Dict] = None


class DailyBatchDataset(Dataset):
    """
    日批次数据集
    
    将数据按交易日组织，每个样本是一天的所有股票数据。
    适用于 GNN 训练场景，需要同时看到当日所有股票。
    
    与传统 Dataset 的区别：
    - 传统: __getitem__ 返回单只股票的 (X, y)
    - 本类: __getitem__ 返回当日所有股票的 (X_day, y_day, stock_ids)
    
    Args:
        df: 包含所有股票时序数据的 DataFrame
        feature_cols: 特征列列表
        label_col: 标签列
        window_size: 时间窗口大小
        time_col: 时间列名
        stock_col: 股票代码列名
        enable_window_transform: 是否启用窗口级变换
        window_price_log: 是否对价格做对数变换
        window_volume_norm: 是否对成交量做窗口内标准化
        price_cols: 价格列列表
        close_col: 收盘价列名
        volume_cols: 成交量列列表
        label_rank_normalize: 是否对标签做截面排名标准化
        label_rank_output_range: 排名标准化输出范围
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        window_size: int = 20,
        time_col: str = 'trade_date',
        stock_col: str = 'order_book_id',
        enable_window_transform: bool = True,
        window_price_log: bool = True,
        window_volume_norm: bool = True,
        price_cols: Optional[List[str]] = None,
        close_col: str = 'close',
        volume_cols: Optional[List[str]] = None,
        label_rank_normalize: bool = True,
        label_rank_output_range: Tuple[float, float] = (-1, 1),
        valid_label_start_date: Optional[pd.Timestamp] = None
    ):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.window_size = window_size
        self.time_col = time_col
        self.stock_col = stock_col
        self.valid_label_start_date = valid_label_start_date
        
        # 窗口变换配置
        self.enable_window_transform = enable_window_transform
        self.window_price_log = window_price_log
        self.window_volume_norm = window_volume_norm
        self.price_cols = price_cols or ['open', 'high', 'low', 'close', 'vwap']
        self.close_col = close_col
        self.volume_cols = volume_cols or ['vol', 'amount']
        
        # 标签处理配置
        self.label_rank_normalize = label_rank_normalize
        self.label_rank_output_range = label_rank_output_range
        
        # 预计算特征列索引
        self._price_indices = [i for i, col in enumerate(feature_cols) if col in self.price_cols]
        self._close_index = feature_cols.index(close_col) if close_col in feature_cols else None
        self._volume_indices = [i for i, col in enumerate(feature_cols) if col in self.volume_cols]
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 构建数据索引
        self._build_index()
    
    def _build_index(self):
        """
        构建数据索引
        
        按日期组织数据，记录每天可用的股票及其时序数据位置
        """
        self.logger.info("构建日批次数据索引...")
        
        # 确保时间列是 datetime
        self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])
        
        # 获取所有唯一日期并排序（直接使用 pandas DatetimeIndex）
        all_dates = pd.DatetimeIndex(sorted(self.df[self.time_col].unique()))
        
        # 过滤：只保留有足够历史数据的日期
        # 需要至少 window_size 天的历史数据
        if len(all_dates) <= self.window_size:
            self.valid_dates = []
            self.date_to_stocks = {}
            self.stock_data = {}
            self.logger.warning(f"数据天数 ({len(all_dates)}) 不足 window_size ({self.window_size})")
            return
        
        # 有效日期是从第 window_size 天开始
        self.valid_dates = list(all_dates[self.window_size:])
        
        # 如果设置了有效标签起始日期，进一步过滤
        if self.valid_label_start_date is not None:
            self.valid_dates = [d for d in self.valid_dates if d >= self.valid_label_start_date]
        
        # 创建日期到索引的全局映射（用于快速查找）
        date_to_global_idx = {d: i for i, d in enumerate(all_dates)}
        
        # 预构建每只股票的完整时序数据（优化版：避免逐行 pd.Timestamp 转换）
        self.stock_data = {}
        self.logger.info(f"  正在构建 {self.df[self.stock_col].nunique()} 只股票的索引...")
        
        for stock, group in self.df.groupby(self.stock_col):
            group_sorted = group.sort_values(self.time_col)
            
            # 提取特征和标签数组
            features = group_sorted[self.feature_cols].values.astype(np.float32)
            labels = group_sorted[self.label_col].values.astype(np.float32)
            # 🔧 优化：直接使用 DatetimeIndex，避免逐个转换
            dates = pd.DatetimeIndex(group_sorted[self.time_col].values)
            
            self.stock_data[stock] = {
                'features': features,
                'labels': labels,
                'dates': dates,
                'date_to_idx': {d: i for i, d in enumerate(dates)}
            }
        
        # 🔧 优化：反向构建 - 先遍历股票，再分配到日期
        # 比双层循环快 10 倍以上
        self.logger.info(f"  正在构建每日可用股票列表...")
        from collections import defaultdict
        date_to_stocks_tmp = defaultdict(list)
        valid_dates_set = set(self.valid_dates)  # 预计算 set，避免重复创建
        
        for stock, data in self.stock_data.items():
            # 找出该股票有足够历史窗口的所有日期
            for i, date in enumerate(data['dates']):
                if i >= self.window_size and date in valid_dates_set:
                    date_to_stocks_tmp[date].append(stock)
        
        # 排序并转换为普通 dict
        self.date_to_stocks = {
            date: sorted(stocks) 
            for date, stocks in date_to_stocks_tmp.items() 
            if stocks
        }
        
        # 更新有效日期列表（排除没有股票的日期）
        self.valid_dates = [d for d in self.valid_dates if d in self.date_to_stocks]
        
        self.logger.info(f"  有效日期: {len(self.valid_dates)} 天")
        if self.valid_dates:
            avg_stocks = np.mean([len(self.date_to_stocks[d]) for d in self.valid_dates])
            self.logger.info(f"  平均每日股票数: {avg_stocks:.1f}")
    
    def __len__(self) -> int:
        """返回有效日期数"""
        return len(self.valid_dates)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取指定日期的所有股票数据 (日批次模式核心实现 - 优化版)
        
        ⚠️ 关键区别：与传统 Dataset 不同，这里返回的是一天中所有股票的数据
        - 传统模式: __getitem__(idx) -> (X[T, F], y) 单只股票
        - 日批次模式: __getitem__(idx) -> {features:[N,T,F], labels:[N]} N只股票
        
        性能优化：
        - 预分配 numpy 数组，避免动态列表扩展
        - 批量切片提取窗口，减少 Python 循环开销
        - 批量应用窗口变换，使用向量化操作
        
        Args:
            idx: 日期索引 (0 ~ len(valid_dates)-1)
            
        Returns:
            Dict 包含:
                - 'features': [N, T, F] 特征张量 (N=当日股票数, T=时间窗口, F=特征维度)
                - 'labels': [N] 标签张量
                - 'stock_ids': List[str] 股票代码列表（长度为N）
                - 'date': pd.Timestamp 当前日期
                - 'n_stocks': int 有效股票数量
        """
        # ======================================================================
        # 步骤1: 按日期索引查找当日的股票列表
        # ======================================================================
        date = self.valid_dates[idx]
        stocks = self.date_to_stocks[date]
        n_stocks = len(stocks)
        n_features = len(self.feature_cols)
        
        if n_stocks == 0:
            return {
                'features': torch.zeros(0, self.window_size, n_features),
                'labels': torch.zeros(0),
                'stock_ids': [],
                'date': date,
                'n_stocks': 0
            }
        
        # ======================================================================
        # 步骤2: 预分配数组，批量提取窗口（性能关键）
        # ======================================================================
        # 预分配，避免动态扩展
        features_array = np.empty((n_stocks, self.window_size, n_features), dtype=np.float32)
        labels_array = np.empty(n_stocks, dtype=np.float32)
        valid_mask = np.ones(n_stocks, dtype=bool)
        
        for i, stock in enumerate(stocks):
            data = self.stock_data[stock]
            stock_idx = data['date_to_idx'][date]
            
            # 窗口切片（numpy 切片是 O(1) 视图操作）
            start_idx = stock_idx - self.window_size
            features_array[i] = data['features'][start_idx:stock_idx]
            labels_array[i] = data['labels'][stock_idx]
            
            # 标记无效样本（标签缺失）
            if np.isnan(labels_array[i]):
                valid_mask[i] = False
        
        # ======================================================================
        # 步骤3: 过滤无效样本
        # ======================================================================
        if not valid_mask.all():
            features_array = features_array[valid_mask]
            labels_array = labels_array[valid_mask]
            valid_stocks = [s for s, m in zip(stocks, valid_mask) if m]
        else:
            valid_stocks = stocks
        
        # 处理空数据
        if len(valid_stocks) == 0:
            return {
                'features': torch.zeros(0, self.window_size, n_features),
                'labels': torch.zeros(0),
                'stock_ids': [],
                'date': date,
                'n_stocks': 0
            }
        
        # ======================================================================
        # 步骤4: 批量应用窗口变换（向量化操作，比逐个变换快 10 倍）
        # ======================================================================
        if self.enable_window_transform:
            features_array = self._apply_window_transform_batch(features_array)
        
        # ======================================================================
        # 步骤5: 转换为 Tensor
        # ======================================================================
        features = torch.from_numpy(features_array)
        labels = torch.from_numpy(labels_array)
        
        # ======================================================================
        # 步骤6: 标签截面排名标准化
        # ======================================================================
        if self.label_rank_normalize and len(labels) > 1:
            labels = self._rank_normalize(labels)
        
        return {
            'features': features,
            'labels': labels,
            'stock_ids': valid_stocks,
            'date': date,
            'n_stocks': len(valid_stocks)
        }
    
    def _apply_window_transform(self, window: np.ndarray) -> np.ndarray:
        """
        应用窗口级变换（向量化优化版）
        
        Args:
            window: [T, F] 窗口数据
            
        Returns:
            变换后的窗口数据
        """
        # 使用视图避免不必要的拷贝
        window = window.copy()
        
        # 价格对数变换: log(price / close_T) - 向量化
        if self.window_price_log and self._close_index is not None and self._price_indices:
            close_T = window[-1, self._close_index]
            if close_T > 0:
                price_cols = np.array(self._price_indices)
                prices = window[:, price_cols]
                prices = np.maximum(prices, 1e-8)
                window[:, price_cols] = np.log(prices / close_T)
        
        # 成交量窗口标准化: volume / mean(volume) - 向量化
        if self.window_volume_norm and self._volume_indices:
            vol_cols = np.array(self._volume_indices)
            volumes = window[:, vol_cols]
            mean_vols = np.mean(volumes, axis=0, keepdims=True)
            mean_vols = np.maximum(mean_vols, 1e-8)  # 避免除零
            window[:, vol_cols] = volumes / mean_vols
        
        # 处理 NaN 和 Inf
        np.nan_to_num(window, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
        return window
    
    def _apply_window_transform_batch(self, windows: np.ndarray) -> np.ndarray:
        """
        批量应用窗口级变换（向量化优化版）
        
        Args:
            windows: [N, T, F] 批量窗口数据
            
        Returns:
            变换后的窗口数据 [N, T, F]
        """
        # 价格对数变换: log(price / close_T) - 向量化
        if self.window_price_log and self._close_index is not None and self._price_indices:
            price_cols = np.array(self._price_indices)
            # close_T: [N, 1]
            close_T = windows[:, -1:, self._close_index:self._close_index+1]
            close_T = np.maximum(close_T, 1e-8)
            # prices: [N, T, len(price_cols)]
            prices = windows[:, :, price_cols]
            prices = np.maximum(prices, 1e-8)
            windows[:, :, price_cols] = np.log(prices / close_T)
        
        # 成交量窗口标准化: volume / mean(volume) - 向量化
        if self.window_volume_norm and self._volume_indices:
            vol_cols = np.array(self._volume_indices)
            volumes = windows[:, :, vol_cols]  # [N, T, len(vol_cols)]
            mean_vols = np.mean(volumes, axis=1, keepdims=True)  # [N, 1, len(vol_cols)]
            mean_vols = np.maximum(mean_vols, 1e-8)
            windows[:, :, vol_cols] = volumes / mean_vols
        
        # 处理 NaN 和 Inf
        np.nan_to_num(windows, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
        return windows
    
    def _rank_normalize(self, labels: torch.Tensor) -> torch.Tensor:
        """
        截面排名标准化
        
        将标签转换为 [low, high] 范围内的排名分数
        """
        n = len(labels)
        if n <= 1:
            return labels
        
        # 计算排名
        sorted_indices = torch.argsort(labels)
        ranks = torch.zeros_like(labels)
        ranks[sorted_indices] = torch.arange(n, dtype=labels.dtype)
        
        # 归一化到 [0, 1]
        normalized = ranks / (n - 1)
        
        # 映射到目标范围
        low, high = self.label_rank_output_range
        return normalized * (high - low) + low
    
    def get_date_list(self) -> List:
        """获取所有有效日期"""
        return self.valid_dates
    
    def get_stocks_for_date(self, date) -> List[str]:
        """获取指定日期的股票列表"""
        return self.date_to_stocks.get(date, [])


def collate_daily(
    batch: List[Dict[str, Any]],
    graph_builder: Optional[Any] = None,
    feature_cols: Optional[List[str]] = None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[str], Any]:
    """
    日批次 collate 函数（动态图构建的关键入口 - 优化版）
    
    ⚠️ 核心功能：将 DailyBatchDataset 返回的字典转换为模型输入，并动态构建邻接矩阵
    
    性能优化：
    - 对于行业图：直接传递 stock_ids，避免创建 DataFrame
    - 对于相关性图：使用轻量级字典代替 DataFrame
    
    Args:
        batch: DailyBatchDataset.__getitem__ 返回的字典列表
        graph_builder: GraphBuilder 实例（可选）
        feature_cols: 特征列名列表（传递给 GraphBuilder）
        
    Returns:
        X, y, adj, stock_ids, date
    """
    # ==========================================================================
    # 情况1: 单日批次（推荐模式）
    # ==========================================================================
    if len(batch) == 1:
        item = batch[0]
        X = item['features']      # [N, T, F]
        y = item['labels']        # [N]
        stock_ids = item['stock_ids']
        date = item['date']
        n_stocks = item['n_stocks']
        
        # ----------------------------------------------------------------------
        # 【关键步骤】动态构建邻接矩阵（优化版）
        # ----------------------------------------------------------------------
        adj = None
        if graph_builder is not None and n_stocks > 0:
            # 🆕 动态获取 stock_col，支持 ts_code/order_book_id 兼容
            stock_col = 'order_book_id'  # 默认值
            if hasattr(graph_builder, 'config') and hasattr(graph_builder.config, 'stock_col'):
                stock_col = graph_builder.config.stock_col
            
            # 检查图构建器类型，选择最优路径
            builder_type = getattr(graph_builder, 'config', None)
            builder_type = getattr(builder_type, 'type', 'corr') if builder_type else 'corr'
            
            if builder_type == 'industry':
                # 【优化】行业图：只需要 stock_ids，不需要特征
                # 创建最小化的 DataFrame（使用动态 stock_col）
                df_day = pd.DataFrame({stock_col: stock_ids})
                adj, _, _ = graph_builder(df_day)
            else:
                # 相关性图或混合图：需要特征
                # 使用 numpy 直接构建，避免逐列字典解析
                last_step_features = X[:, -1, :].numpy()  # [N, F]
                
                # 构建 DataFrame（这里仍需要 DataFrame，但用更高效的方式）
                df_day = pd.DataFrame(
                    last_step_features,
                    columns=feature_cols if feature_cols else [f'feature_{i}' for i in range(X.size(2))]
                )
                # 🆕 使用动态 stock_col 而非硬编码 order_book_id
                df_day.insert(0, stock_col, stock_ids)
                
                adj, _, _ = graph_builder(df_day)
        
        return X, y, adj, stock_ids, date
    
    # ==========================================================================
    # 情况2: 多日批次（较少使用）
    # ==========================================================================
    else:
        all_X = []
        all_y = []
        all_adj = []
        all_stocks = []
        all_dates = []
        
        # 🆕 获取动态 stock_col（多日批次）
        stock_col = 'order_book_id'  # 默认值
        if graph_builder is not None:
            if hasattr(graph_builder, 'config') and hasattr(graph_builder.config, 'stock_col'):
                stock_col = graph_builder.config.stock_col
        
        for item in batch:
            all_X.append(item['features'])
            all_y.append(item['labels'])
            all_stocks.append(item['stock_ids'])
            all_dates.append(item['date'])
            
            # 为每一天分别构建邻接矩阵
            if graph_builder is not None and item['n_stocks'] > 0:
                X = item['features']
                # 🆕 使用动态 stock_col
                df_day = pd.DataFrame({
                    stock_col: item['stock_ids'],
                    **{f'feature_{i}': X[:, -1, i].numpy() for i in range(X.size(2))}
                })
                adj, _, _ = graph_builder(df_day)
                all_adj.append(adj)
            else:
                all_adj.append(None)
        
        return all_X, all_y, all_adj, all_stocks, all_dates


class DailyGraphDataLoader:
    """
    日批次图数据加载器
    
    封装 DataLoader，提供更便捷的接口和动态图构建支持。
    
    特点：
    1. 每个 batch 是一个交易日的所有股票
    2. 支持动态图构建
    3. 日期顺序可打乱（训练）或保持（推理）
    
    Args:
        dataset: DailyBatchDataset 实例
        graph_builder: GraphBuilder 实例（可选）
        shuffle_dates: 是否打乱日期顺序
        num_workers: 工作进程数
        pin_memory: 是否使用 pinned memory
        device: 目标设备
    """
    
    def __init__(
        self,
        dataset: DailyBatchDataset,
        graph_builder: Optional[Any] = None,
        feature_cols: Optional[List[str]] = None,
        shuffle_dates: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        device: str = 'cuda'
    ):
        self.dataset = dataset
        self.graph_builder = graph_builder
        self.feature_cols = feature_cols or dataset.feature_cols
        self.device = device
        self.shuffle_dates = shuffle_dates
        
        # 创建内部 DataLoader
        # batch_size=1 确保每个批次是一天
        self._loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle_dates,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate
        )
    
    def _collate(self, batch):
        """内部 collate 函数（返回 CPU 张量，训练时再转 GPU）"""
        return collate_daily(
            batch,
            graph_builder=self.graph_builder,
            feature_cols=self.feature_cols
        )
    
    def __iter__(self):
        """迭代器（过滤空批次）"""
        for batch in self._loader:
            X, y, adj, stock_ids, date = batch
            # 🆕 跳过空批次（n_stocks=0 会导致 GAT 层 N=0 reshape 异常）
            if X.size(0) == 0:
                continue
            yield batch
    
    def __len__(self):
        """返回天数"""
        return len(self.dataset)
    
    @property
    def n_days(self) -> int:
        """有效天数"""
        return len(self.dataset)
    
    def get_date_list(self) -> List:
        """获取日期列表"""
        return self.dataset.get_date_list()


class DailyBatchSampler:
    """
    日批次采样器
    
    用于在普通 Dataset 上实现按日采样，每个 batch 包含同一天的样本。
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        time_col: str = 'trade_date',
        shuffle_dates: bool = True,
        drop_last: bool = False
    ):
        self.time_col = time_col
        self.shuffle_dates = shuffle_dates
        self.drop_last = drop_last
        
        # 构建日期到索引的映射
        self.date_to_indices = {}
        for idx, row in df.iterrows():
            date = row[time_col]
            if date not in self.date_to_indices:
                self.date_to_indices[date] = []
            self.date_to_indices[date].append(idx)
        
        self.dates = list(self.date_to_indices.keys())
    
    def __iter__(self):
        dates = self.dates.copy()
        if self.shuffle_dates:
            np.random.shuffle(dates)
        
        for date in dates:
            indices = self.date_to_indices[date]
            yield indices
    
    def __len__(self):
        return len(self.dates)


# ==================== 工厂函数 ====================

def create_daily_loader(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    window_size: int = 20,
    time_col: str = 'trade_date',
    stock_col: str = 'order_book_id',
    graph_builder_config: Optional[Dict] = None,
    shuffle_dates: bool = True,
    device: str = 'cuda',
    **kwargs
) -> DailyGraphDataLoader:
    """
    创建日批次数据加载器的便捷函数
    
    Args:
        df: 数据 DataFrame
        feature_cols: 特征列
        label_col: 标签列
        window_size: 窗口大小
        time_col: 时间列
        stock_col: 股票列
        graph_builder_config: 图构建器配置
        shuffle_dates: 是否打乱日期
        device: 设备
        **kwargs: 其他参数传递给 DailyBatchDataset
        
    Returns:
        DailyGraphDataLoader 实例
    """
    # 创建数据集
    dataset = DailyBatchDataset(
        df=df,
        feature_cols=feature_cols,
        label_col=label_col,
        window_size=window_size,
        time_col=time_col,
        stock_col=stock_col,
        **kwargs
    )
    
    # 创建图构建器
    graph_builder = None
    if graph_builder_config is not None:
        from quantclassic.data_processor.graph_builder import GraphBuilderFactory
        graph_builder = GraphBuilderFactory.create(graph_builder_config)
    
    # 创建加载器
    loader = DailyGraphDataLoader(
        dataset=dataset,
        graph_builder=graph_builder,
        feature_cols=feature_cols,
        shuffle_dates=shuffle_dates,
        device=device
    )
    
    return loader


# ==================== 单元测试 ====================

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("DailyBatchDataset 单元测试")
    print("=" * 80)
    
    # 创建测试数据：30天，5只股票
    np.random.seed(42)
    
    n_days = 30
    stocks = ['000001', '000002', '000003', '000004', '000005']
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    
    rows = []
    for date in dates:
        for stock in stocks:
            rows.append({
                'trade_date': date,
                'order_book_id': stock,
                'industry_name': 'TEST',
                'open': 10 + np.random.randn(),
                'high': 11 + np.random.randn(),
                'low': 9 + np.random.randn(),
                'close': 10 + np.random.randn(),
                'vol': 1000 + np.random.randn() * 100,
                'amount': 10000 + np.random.randn() * 1000,
                'alpha_label': np.random.randn()
            })
    
    df = pd.DataFrame(rows)
    feature_cols = ['open', 'high', 'low', 'close', 'vol', 'amount']
    
    print(f"\n📊 测试数据: {len(df)} 行, {len(stocks)} 只股票, {n_days} 天")
    
    # 测试 1: 创建数据集
    print("\n【测试 1: 创建 DailyBatchDataset】")
    window_size = 10
    dataset = DailyBatchDataset(
        df=df,
        feature_cols=feature_cols,
        label_col='alpha_label',
        window_size=window_size,
        enable_window_transform=True,
        label_rank_normalize=True
    )
    
    print(f"  有效天数: {len(dataset)}")
    print(f"  期望天数: {n_days - window_size} = 30 - 10")
    assert len(dataset) == n_days - window_size, "有效天数计算错误"
    print("  ✓ 数据集创建成功")
    
    # 测试 2: 获取单日数据
    print("\n【测试 2: 获取单日数据】")
    sample = dataset[0]
    print(f"  日期: {sample['date']}")
    print(f"  股票数: {sample['n_stocks']}")
    print(f"  特征形状: {sample['features'].shape}")  # [N, T, F]
    print(f"  标签形状: {sample['labels'].shape}")  # [N]
    
    assert sample['features'].shape[0] == sample['n_stocks'], "特征和股票数不匹配"
    assert sample['features'].shape[1] == window_size, "窗口大小错误"
    assert sample['features'].shape[2] == len(feature_cols), "特征数错误"
    print("  ✓ 单日数据获取成功")
    
    # 测试 3: 标签排名标准化
    print("\n【测试 3: 标签排名标准化】")
    labels = sample['labels']
    print(f"  标签范围: [{labels.min():.4f}, {labels.max():.4f}]")
    assert labels.min() >= -1 and labels.max() <= 1, "标签范围错误"
    print("  ✓ 标签排名标准化正确")
    
    # 测试 4: collate_daily
    print("\n【测试 4: collate_daily】")
    batch = [dataset[0]]
    X, y, adj, stock_ids, date = collate_daily(batch, graph_builder=None)
    
    print(f"  X 形状: {X.shape}")
    print(f"  y 形状: {y.shape}")
    print(f"  股票ID数: {len(stock_ids)}")
    print(f"  日期: {date}")
    print("  ✓ collate_daily 正确")
    
    # 测试 5: 配合 GraphBuilder
    print("\n【测试 5: 配合 GraphBuilder】")
    import sys
    try:
        import quantclassic
    except ImportError:
         # 尝试动态添加项目根目录
        project_root = str(Path(__file__).resolve().parents[3])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            
    from quantclassic.data_processor.graph_builder import CorrGraphBuilder
    
    graph_builder = CorrGraphBuilder(method='cosine', top_k=2)
    
    X, y, adj, stock_ids, date = collate_daily(
        batch=[dataset[0]],
        graph_builder=graph_builder,
        feature_cols=feature_cols
    )
    
    print(f"  邻接矩阵形状: {adj.shape}")
    print(f"  非零边数: {(adj > 0).sum().item()}")
    assert adj.shape[0] == len(stock_ids), "邻接矩阵尺寸错误"
    print("  ✓ 动态图构建正确")
    
    # 测试 6: DailyGraphDataLoader
    print("\n【测试 6: DailyGraphDataLoader】")
    loader = DailyGraphDataLoader(
        dataset=dataset,
        graph_builder=graph_builder,
        feature_cols=feature_cols,
        shuffle_dates=True,
        device='cpu'
    )
    
    print(f"  加载器天数: {len(loader)}")
    
    # 遍历一个 epoch
    for i, (X, y, adj, stock_ids, date) in enumerate(loader):
        if i >= 2:  # 只测试前2天
            break
        print(f"  Batch {i}: date={date}, n_stocks={len(stock_ids)}, adj={adj.shape if adj is not None else None}")
    
    print("  ✓ DailyGraphDataLoader 正确")
    
    # 测试 7: create_daily_loader 工厂函数
    print("\n【测试 7: create_daily_loader】")
    loader2 = create_daily_loader(
        df=df,
        feature_cols=feature_cols,
        label_col='alpha_label',
        window_size=10,
        graph_builder_config={'type': 'corr', 'corr_method': 'cosine', 'top_k': 2},
        device='cpu'
    )
    print(f"  工厂创建的加载器天数: {len(loader2)}")
    print("  ✓ 工厂函数正确")
    
    print("\n" + "=" * 80)
    print("✅ 所有 DailyBatchDataset 测试通过！")
    print("=" * 80)
