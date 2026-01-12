"""
build_industry_adj.py - 构建行业邻接矩阵

⚠️ DEPRECATED: 此模块为静态基准工具，建议使用 data_processor.graph_builder 作为统一入口。

推荐迁移方式:
    from quantclassic.data_processor.graph_builder import GraphBuilderFactory
    builder = GraphBuilderFactory.create({'type': 'industry', 'stock_col': 'ts_code'})
    adj = builder.build(df)

---

根据财通研报的要求，基于行业分类构建静态邻接矩阵：
- 同行业股票之间 A[i,j] = 1
- 不同行业股票之间 A[i,j] = 0

这种静态图结构避免了使用目标列（如 alpha_label）动态计算相关性，
从而防止数据泄露和训练目标耦合问题。

Usage:
    from quantclassic.model.build_industry_adj import build_industry_adjacency_matrix
    
    adj_matrix, stock_list = build_industry_adjacency_matrix(
        df=df,
        stock_col='order_book_id',
        industry_col='industry_name',
        save_path='output/industry_adj_matrix.pt'
    )
"""

import warnings
import torch
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import logging


def build_industry_adjacency_matrix(
    df: pd.DataFrame,
    stock_col: str = 'order_book_id',
    industry_col: str = 'industry_name',
    save_path: Optional[str] = None,
    add_self_loop: bool = True,
    normalize: bool = False
) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
    """
    基于行业分类构建邻接矩阵（研报 baseline）
    
    .. deprecated::
        此函数为遗产基准工具，建议使用 `data_processor.graph_builder.GraphBuilderFactory` 统一入口。
    
    同行业股票之间连接权重为 1，不同行业为 0。
    
    Args:
        df: 包含股票代码和行业信息的 DataFrame
        stock_col: 股票代码列名
        industry_col: 行业分类列名
        save_path: 保存路径（.pt 格式），None 表示不保存
        add_self_loop: 是否添加自环（对角线为 1）
        normalize: 是否对邻接矩阵进行行归一化
        
    Returns:
        adj_matrix: [N, N] 邻接矩阵
        stock_list: 股票代码列表（顺序与矩阵索引对应）
        stock_to_idx: 股票代码到索引的映射字典
    """
    warnings.warn(
        "build_industry_adjacency_matrix 已弃用，建议使用 "
        "data_processor.graph_builder.GraphBuilderFactory.create({'type': 'industry'}) 作为统一入口",
        DeprecationWarning,
        stacklevel=2
    )
    logger = logging.getLogger(__name__)
    
    # 1. 获取唯一股票列表和行业信息
    stock_industry = df[[stock_col, industry_col]].drop_duplicates(subset=[stock_col])
    stock_list = sorted(stock_industry[stock_col].unique().tolist())
    n_stocks = len(stock_list)
    
    # 创建股票到索引的映射
    stock_to_idx = {stock: i for i, stock in enumerate(stock_list)}
    
    # 2. 创建股票到行业的映射
    stock_to_industry = dict(zip(
        stock_industry[stock_col],
        stock_industry[industry_col]
    ))
    
    # 3. 构建邻接矩阵
    adj_matrix = torch.zeros(n_stocks, n_stocks)
    
    # 统计行业信息
    industries = stock_industry[industry_col].unique()
    industry_counts = stock_industry[industry_col].value_counts()
    
    logger.info(f"构建行业邻接矩阵:")
    logger.info(f"  股票数量: {n_stocks}")
    logger.info(f"  行业数量: {len(industries)}")
    
    # 按行业分组，同行业股票互相连接
    for industry in industries:
        # 获取该行业的所有股票
        industry_stocks = stock_industry[
            stock_industry[industry_col] == industry
        ][stock_col].tolist()
        
        # 获取股票索引
        indices = [stock_to_idx[s] for s in industry_stocks if s in stock_to_idx]
        
        # 同行业股票互相连接
        for i in indices:
            for j in indices:
                if i != j or add_self_loop:  # 可选择是否添加自环
                    adj_matrix[i, j] = 1.0
    
    # 4. 添加自环
    if add_self_loop:
        adj_matrix.fill_diagonal_(1.0)
    
    # 5. 可选：行归一化
    if normalize:
        row_sum = adj_matrix.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1  # 避免除零
        adj_matrix = adj_matrix / row_sum
    
    # 6. 统计连接信息
    n_edges = (adj_matrix > 0).sum().item()
    avg_neighbors = n_edges / n_stocks
    
    logger.info(f"  总边数: {n_edges:,}")
    logger.info(f"  平均邻居数: {avg_neighbors:.1f}")
    logger.info(f"  矩阵稀疏度: {1 - n_edges / (n_stocks * n_stocks):.2%}")
    
    # 7. 保存
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存邻接矩阵和元数据
        torch.save({
            'adj_matrix': adj_matrix,
            'stock_list': stock_list,
            'stock_to_idx': stock_to_idx,
            'n_stocks': n_stocks,
            'n_industries': len(industries),
            'industry_counts': industry_counts.to_dict()
        }, save_path)
        
        logger.info(f"  已保存至: {save_path}")
    
    return adj_matrix, stock_list, stock_to_idx


def build_correlation_adjacency_matrix(
    df: pd.DataFrame,
    stock_col: str = 'order_book_id',
    time_col: str = 'trade_date',
    return_col: str = 'close',  # 🔴 注意：不要使用目标列如 alpha_label
    top_k: int = 10,
    method: str = 'pearson',
    min_periods: int = 60,
    save_path: Optional[str] = None,
    add_self_loop: bool = True
) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
    """
    基于收益率相关性构建邻接矩阵（研报备选方案）
    
    🔴 重要：使用历史收益率（如 close 的 pct_change）而非训练目标列，
    避免数据泄露。
    
    Args:
        df: 包含股票时序数据的 DataFrame
        stock_col: 股票代码列名
        time_col: 时间列名
        return_col: 用于计算相关性的列（建议使用价格列，会自动计算收益率）
        top_k: 每只股票选取相关性最高的 k 个邻居
        method: 相关性计算方法 ('pearson' 或 'spearman')
        min_periods: 计算相关性所需的最小观测数
        save_path: 保存路径
        add_self_loop: 是否添加自环
        
    Returns:
        adj_matrix: [N, N] 邻接矩阵
        stock_list: 股票代码列表
        stock_to_idx: 股票代码到索引的映射
    """
    logger = logging.getLogger(__name__)
    
    # 1. 构建收益率矩阵
    logger.info(f"构建相关性邻接矩阵:")
    logger.info(f"  相关性列: {return_col}")
    logger.info(f"  Top-K 邻居: {top_k}")
    
    # Pivot 成 [时间, 股票] 格式
    pivot_df = df.pivot_table(
        index=time_col,
        columns=stock_col,
        values=return_col,
        aggfunc='last'
    )
    
    # 如果是价格列，计算收益率
    if return_col in ['close', 'open', 'high', 'low', 'vwap']:
        pivot_df = pivot_df.pct_change()
    
    stock_list = sorted(pivot_df.columns.tolist())
    n_stocks = len(stock_list)
    stock_to_idx = {stock: i for i, stock in enumerate(stock_list)}
    
    logger.info(f"  股票数量: {n_stocks}")
    logger.info(f"  时间跨度: {len(pivot_df)} 天")
    
    # 2. 计算相关性矩阵
    if method == 'pearson':
        corr_matrix = pivot_df[stock_list].corr(method='pearson', min_periods=min_periods)
    else:
        corr_matrix = pivot_df[stock_list].corr(method='spearman', min_periods=min_periods)
    
    # 转换为 numpy
    corr_values = corr_matrix.values
    corr_values = np.nan_to_num(corr_values, nan=0.0)
    
    # 3. 构建 Top-K 邻接矩阵
    adj_matrix = torch.zeros(n_stocks, n_stocks)
    
    for i in range(n_stocks):
        # 获取第 i 只股票与所有股票的相关性
        correlations = corr_values[i].copy()
        correlations[i] = -np.inf  # 排除自己
        
        # 选取 Top-K
        top_k_indices = np.argsort(correlations)[-top_k:]
        
        for j in top_k_indices:
            if correlations[j] > 0:  # 只保留正相关
                adj_matrix[i, j] = correlations[j]
    
    # 4. 对称化（可选）
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    
    # 5. 添加自环
    if add_self_loop:
        adj_matrix.fill_diagonal_(1.0)
    
    # 6. 统计
    n_edges = (adj_matrix > 0).sum().item()
    logger.info(f"  总边数: {n_edges:,}")
    logger.info(f"  平均邻居数: {n_edges / n_stocks:.1f}")
    
    # 7. 保存
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'adj_matrix': adj_matrix,
            'stock_list': stock_list,
            'stock_to_idx': stock_to_idx,
            'n_stocks': n_stocks,
            'method': method,
            'top_k': top_k
        }, save_path)
        
        logger.info(f"  已保存至: {save_path}")
    
    return adj_matrix, stock_list, stock_to_idx


def load_adjacency_matrix(
    path: str,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
    """
    加载预构建的邻接矩阵
    
    Args:
        path: .pt 文件路径
        device: 目标设备
        
    Returns:
        adj_matrix, stock_list, stock_to_idx
    """
    data = torch.load(path, map_location=device)
    
    return (
        data['adj_matrix'],
        data['stock_list'],
        data['stock_to_idx']
    )


# ==================== 命令行接口 ====================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='构建行业/相关性邻接矩阵')
    parser.add_argument('--data', type=str, required=True, help='数据文件路径 (.parquet)')
    parser.add_argument('--output', type=str, default='output/industry_adj_matrix.pt', help='输出路径')
    parser.add_argument('--type', type=str, choices=['industry', 'correlation'], default='industry',
                        help='邻接矩阵类型')
    parser.add_argument('--stock-col', type=str, default='order_book_id', help='股票代码列名')
    parser.add_argument('--industry-col', type=str, default='industry_name', help='行业列名')
    parser.add_argument('--top-k', type=int, default=10, help='相关性矩阵的 Top-K 邻居数')
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 加载数据
    print(f"加载数据: {args.data}")
    df = pd.read_parquet(args.data)
    
    # 构建邻接矩阵
    if args.type == 'industry':
        adj_matrix, stock_list, stock_to_idx = build_industry_adjacency_matrix(
            df=df,
            stock_col=args.stock_col,
            industry_col=args.industry_col,
            save_path=args.output
        )
    else:
        adj_matrix, stock_list, stock_to_idx = build_correlation_adjacency_matrix(
            df=df,
            stock_col=args.stock_col,
            return_col='close',  # 使用收盘价计算收益率
            top_k=args.top_k,
            save_path=args.output
        )
    
    print(f"\n✅ 邻接矩阵构建完成！")
    print(f"   形状: {adj_matrix.shape}")
    print(f"   保存至: {args.output}")
