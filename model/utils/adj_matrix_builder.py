"""
Adjacency Matrix Builder - 邻接矩阵构建工具

为 GAT 模型生成邻接矩阵，支持两种模式:
1. 基于行业分类 (Standard GAT)
2. 基于收益率相关性 (Correlation GAT)
"""

import torch
import numpy as np
import pandas as pd
from typing import Optional, Union, List
from pathlib import Path
import logging


class AdjMatrixBuilder:
    """
    邻接矩阵构建器
    
    Example:
        # 1. 基于行业构建
        builder = AdjMatrixBuilder()
        adj = builder.build_industry_adj(industry_codes=['A01', 'A01', 'B02', 'A01'])
        
        # 2. 基于相关性构建
        returns = pd.DataFrame(...)  # [stocks x time]
        adj = builder.build_correlation_adj(returns, top_k=10)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def build_industry_adj(
        self,
        industry_codes: Union[List[str], pd.Series],
        self_loop: bool = True
    ) -> torch.Tensor:
        """
        基于行业分类构建邻接矩阵
        
        如果两只股票属于同一行业，则它们之间存在边 (A_ij = 1)。
        
        Args:
            industry_codes: 股票的行业代码列表
            self_loop: 是否包含自环（对角线为1）
            
        Returns:
            adj: [N, N] 邻接矩阵，N 为股票数量
            
        Example:
            industry_codes = ['制造业', '金融', '制造业', '科技', '金融']
            adj = builder.build_industry_adj(industry_codes)
            # 输出:
            # [[1, 0, 1, 0, 0],
            #  [0, 1, 0, 0, 1],
            #  [1, 0, 1, 0, 0],
            #  [0, 0, 0, 1, 0],
            #  [0, 1, 0, 0, 1]]
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
        
        self.logger.info(f"构建行业邻接矩阵: shape={adj.shape}, 边数={adj.sum().item()}")
        return adj
    
    def build_correlation_adj(
        self,
        returns: Union[pd.DataFrame, np.ndarray],
        top_k: int = 10,
        method: str = 'pearson',
        self_loop: bool = True,
        threshold: Optional[float] = None
    ) -> torch.Tensor:
        """
        基于收益率相关性构建邻接矩阵
        
        选取与每只股票相关性最高的 top_k 只股票作为邻居。
        
        Args:
            returns: 收益率矩阵，shape=[stocks, time] 或 DataFrame
            top_k: 每只股票选取的邻居数量
            method: 相关性计算方法 ('pearson', 'spearman', 'kendall')
            self_loop: 是否包含自环
            threshold: 相关性阈值（可选），低于此值的边会被移除
            
        Returns:
            adj: [N, N] 邻接矩阵，N 为股票数量
            
        Example:
            returns = pd.DataFrame({
                'stock1': [...],
                'stock2': [...],
                'stock3': [...]
            })
            adj = builder.build_correlation_adj(returns, top_k=2)
        """
        # 转换为 DataFrame
        if isinstance(returns, np.ndarray):
            returns = pd.DataFrame(returns)
        
        # 计算相关性矩阵
        if method == 'pearson':
            corr = returns.T.corr(method='pearson')
        elif method == 'spearman':
            corr = returns.T.corr(method='spearman')
        elif method == 'kendall':
            corr = returns.T.corr(method='kendall')
        else:
            raise ValueError(f"不支持的相关性方法: {method}")
        
        # 转换为 numpy 数组
        corr = corr.values
        n_stocks = corr.shape[0]
        
        # 初始化邻接矩阵
        adj = torch.zeros(n_stocks, n_stocks)
        
        # 对每只股票，选取相关性最高的 top_k 个邻居
        for i in range(n_stocks):
            # 获取相关性（排除自己）
            corr_i = corr[i].copy()
            corr_i[i] = -np.inf  # 暂时排除自己
            
            # 选取 top_k 个最相关的股票
            top_k_indices = np.argsort(corr_i)[-top_k:]
            
            # 设置边
            for j in top_k_indices:
                if threshold is None or corr[i, j] >= threshold:
                    adj[i, j] = 1.0
        
        # 对称化邻接矩阵（无向图）
        adj = torch.maximum(adj, adj.T)
        
        # 添加自环
        if self_loop:
            adj.fill_diagonal_(1)
        
        self.logger.info(
            f"构建相关性邻接矩阵: shape={adj.shape}, "
            f"边数={adj.sum().item()}, top_k={top_k}"
        )
        return adj
    
    def build_weighted_adj(
        self,
        returns: Union[pd.DataFrame, np.ndarray],
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
        if isinstance(returns, np.ndarray):
            returns = pd.DataFrame(returns)
        
        # 计算相关性矩阵
        if method == 'pearson':
            corr = returns.T.corr(method='pearson')
        elif method == 'spearman':
            corr = returns.T.corr(method='spearman')
        else:
            raise ValueError(f"不支持的相关性方法: {method}")
        
        corr = corr.values
        n_stocks = corr.shape[0]
        
        # 初始化加权邻接矩阵
        adj = torch.zeros(n_stocks, n_stocks)
        
        # 对每只股票，选取相关性最高的 top_k 个邻居
        for i in range(n_stocks):
            corr_i = corr[i].copy()
            corr_i[i] = -np.inf
            
            top_k_indices = np.argsort(corr_i)[-top_k:]
            
            # 设置边权重为相关性值
            for j in top_k_indices:
                # 使用绝对值或原始相关性
                adj[i, j] = abs(corr[i, j])  # 可选: corr[i, j]
        
        # 对称化
        adj = torch.maximum(adj, adj.T)
        
        # 添加自环
        if self_loop:
            adj.fill_diagonal_(1)
        
        self.logger.info(
            f"构建加权邻接矩阵: shape={adj.shape}, "
            f"平均权重={adj[adj > 0].mean():.4f}"
        )
        return adj
    
    def save_adj_matrix(
        self,
        adj: torch.Tensor,
        save_path: Union[str, Path],
        format: str = 'pt'
    ):
        """
        保存邻接矩阵到文件
        
        Args:
            adj: 邻接矩阵
            save_path: 保存路径
            format: 保存格式 ('pt', 'npy')
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pt':
            torch.save(adj, save_path)
        elif format == 'npy':
            np.save(save_path, adj.numpy())
        else:
            raise ValueError(f"不支持的保存格式: {format}")
        
        self.logger.info(f"邻接矩阵已保存: {save_path}")
    
    @staticmethod
    def load_adj_matrix(
        load_path: Union[str, Path]
    ) -> torch.Tensor:
        """
        加载邻接矩阵
        
        Args:
            load_path: 文件路径
            
        Returns:
            adj: 邻接矩阵
        """
        load_path = Path(load_path)
        
        if load_path.suffix in ['.pt', '.pth']:
            adj = torch.load(load_path)
        elif load_path.suffix == '.npy':
            adj = torch.from_numpy(np.load(load_path)).float()
        else:
            raise ValueError(f"不支持的文件格式: {load_path.suffix}")
        
        return adj
    
    def visualize_adj_matrix(
        self,
        adj: torch.Tensor,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 10),
        cmap: str = 'Blues'
    ):
        """
        可视化邻接矩阵
        
        Args:
            adj: 邻接矩阵
            save_path: 保存图像路径（可选）
            figsize: 图像大小
            cmap: 颜色映射
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=figsize)
            sns.heatmap(
                adj.numpy(),
                cmap=cmap,
                square=True,
                cbar=True,
                xticklabels=False,
                yticklabels=False
            )
            plt.title('Adjacency Matrix')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"邻接矩阵可视化已保存: {save_path}")
            
            plt.show()
        
        except ImportError:
            self.logger.warning("需要安装 matplotlib 和 seaborn 才能可视化")


if __name__ == '__main__':
    print("=" * 80)
    print("Adjacency Matrix Builder 测试")
    print("=" * 80)
    
    builder = AdjMatrixBuilder()
    
    # 测试 1: 基于行业
    print("\n1. 基于行业构建邻接矩阵:")
    industry_codes = ['A', 'A', 'B', 'C', 'A', 'B']
    adj_industry = builder.build_industry_adj(industry_codes)
    print(f"  Shape: {adj_industry.shape}")
    print(f"  边数: {adj_industry.sum().item()}")
    print(f"  邻接矩阵:\n{adj_industry}")
    
    # 测试 2: 基于相关性
    print("\n2. 基于相关性构建邻接矩阵:")
    # 生成模拟收益率数据
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.randn(50, 6),  # 50个时间点，6只股票
        columns=[f'stock_{i}' for i in range(6)]
    )
    adj_corr = builder.build_correlation_adj(returns, top_k=3)
    print(f"  Shape: {adj_corr.shape}")
    print(f"  边数: {adj_corr.sum().item()}")
    print(f"  邻接矩阵:\n{adj_corr}")
    
    # 测试 3: 加权邻接矩阵
    print("\n3. 构建加权邻接矩阵:")
    adj_weighted = builder.build_weighted_adj(returns, top_k=3)
    print(f"  Shape: {adj_weighted.shape}")
    print(f"  平均权重: {adj_weighted[adj_weighted > 0].mean():.4f}")
    
    # 测试 4: 保存和加载
    print("\n4. 保存和加载邻接矩阵:")
    save_path = '/tmp/test_adj.pt'
    builder.save_adj_matrix(adj_industry, save_path)
    adj_loaded = AdjMatrixBuilder.load_adj_matrix(save_path)
    print(f"  原始: {adj_industry.shape}, 加载: {adj_loaded.shape}")
    print(f"  数据一致性: {torch.allclose(adj_industry, adj_loaded)}")
    
    print("\n" + "=" * 80)
    print("✅ Adjacency Matrix Builder 测试完成")
    print("=" * 80)
