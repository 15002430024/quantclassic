#!/usr/bin/env python
"""
build_adj.py - 离线邻接矩阵构建脚本

功能：
    在命令行中预计算行业/相关性/混合邻接矩阵，保存为 .pt 文件供训练时加载。
    复用 data_processor/graph_builder.py 中的 GraphBuilderFactory，确保与运行时构图逻辑一致。

使用示例:
    # 构建行业邻接矩阵
    python scripts/graph/build_adj.py --data data.parquet --type industry --output output/industry_adj.pt
    
    # 构建相关性邻接矩阵
    python scripts/graph/build_adj.py --data data.parquet --type corr --top-k 10 --output output/corr_adj.pt
    
    # 构建混合邻接矩阵
    python scripts/graph/build_adj.py --data data.parquet --type hybrid --alpha 0.7 --output output/hybrid_adj.pt

⚠️ 注意：
    本脚本是离线工具，用于预生成静态邻接矩阵。
    运行时动态构图请使用 DataManager.create_daily_loaders() + GraphBuilderFactory。
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantclassic.data_processor.graph_builder import (
    GraphBuilderFactory,
    GraphBuilderConfig,
    IndustryGraphBuilder,
    CorrGraphBuilder,
    HybridGraphBuilder,
)


def build_adjacency_matrix(
    df: pd.DataFrame,
    graph_type: str,
    stock_col: str = 'order_book_id',
    industry_col: str = 'industry_name',
    top_k: int = 10,
    alpha: float = 0.7,
    corr_method: str = 'pearson',
    add_self_loop: bool = True,
    normalize: bool = False,
):
    """
    使用 GraphBuilderFactory 构建邻接矩阵
    
    Args:
        df: 包含股票数据的 DataFrame
        graph_type: 图类型 ('industry', 'corr', 'hybrid')
        stock_col: 股票代码列名
        industry_col: 行业列名
        top_k: 相关性图的邻居数
        alpha: 混合图权重
        corr_method: 相关性计算方法
        add_self_loop: 是否添加自环
        normalize: 是否行归一化
        
    Returns:
        adj_matrix: 邻接矩阵
        stock_list: 股票列表
        stock_to_idx: 股票到索引的映射
    """
    # 构建配置
    config = {
        'type': graph_type,
        'stock_col': stock_col,
        'industry_col': industry_col,
        'top_k': top_k,
        'alpha': alpha,
        'corr_method': corr_method,
        'add_self_loop': add_self_loop,
        'normalize': normalize,
    }
    
    # 创建图构建器
    builder = GraphBuilderFactory.create(config)
    
    # 构建邻接矩阵
    adj_matrix, stock_list, stock_to_idx = builder(df)
    
    return adj_matrix, stock_list, stock_to_idx


def save_adjacency_matrix(
    adj_matrix: torch.Tensor,
    stock_list: list,
    stock_to_idx: dict,
    save_path: str,
    metadata: dict = None,
):
    """
    保存邻接矩阵及元数据
    
    Args:
        adj_matrix: 邻接矩阵
        stock_list: 股票列表
        stock_to_idx: 股票到索引的映射
        save_path: 保存路径
        metadata: 额外元数据
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'adj_matrix': adj_matrix,
        'stock_list': stock_list,
        'stock_to_idx': stock_to_idx,
        'n_stocks': len(stock_list),
    }
    
    if metadata:
        data.update(metadata)
    
    torch.save(data, save_path)
    logging.info(f"邻接矩阵已保存至: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='离线构建邻接矩阵（复用 GraphBuilderFactory）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 行业图
    python build_adj.py --data data.parquet --type industry
    
    # 相关性图
    python build_adj.py --data data.parquet --type corr --top-k 10
    
    # 混合图
    python build_adj.py --data data.parquet --type hybrid --alpha 0.7
        """
    )
    
    # 必需参数
    parser.add_argument('--data', type=str, required=True,
                        help='数据文件路径 (.parquet/.csv)')
    
    # 图类型
    parser.add_argument('--type', type=str, choices=['industry', 'corr', 'hybrid'],
                        default='industry', help='邻接矩阵类型')
    
    # 输出路径
    parser.add_argument('--output', '-o', type=str, default='output/adj_matrix.pt',
                        help='输出路径 (.pt 格式)')
    
    # 列名配置
    parser.add_argument('--stock-col', type=str, default='order_book_id',
                        help='股票代码列名')
    parser.add_argument('--industry-col', type=str, default='industry_name',
                        help='行业列名')
    
    # 相关性图参数
    parser.add_argument('--top-k', type=int, default=10,
                        help='每只股票的邻居数量（相关性图）')
    parser.add_argument('--corr-method', type=str, choices=['pearson', 'spearman', 'cosine'],
                        default='pearson', help='相关性计算方法')
    
    # 混合图参数
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='混合图权重：alpha * corr + (1-alpha) * industry')
    
    # 通用参数
    parser.add_argument('--no-self-loop', action='store_true',
                        help='不添加自环')
    parser.add_argument('--normalize', action='store_true',
                        help='行归一化')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='详细输出')
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 加载数据
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"数据文件不存在: {data_path}")
        sys.exit(1)
    
    logger.info(f"加载数据: {data_path}")
    if data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    elif data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
    else:
        logger.error(f"不支持的文件格式: {data_path.suffix}")
        sys.exit(1)
    
    logger.info(f"数据形状: {df.shape}")
    
    # 构建邻接矩阵
    logger.info(f"构建 {args.type} 邻接矩阵...")
    
    adj_matrix, stock_list, stock_to_idx = build_adjacency_matrix(
        df=df,
        graph_type=args.type,
        stock_col=args.stock_col,
        industry_col=args.industry_col,
        top_k=args.top_k,
        alpha=args.alpha,
        corr_method=args.corr_method,
        add_self_loop=not args.no_self_loop,
        normalize=args.normalize,
    )
    
    # 统计信息
    n_stocks = len(stock_list)
    n_edges = (adj_matrix > 0).sum().item()
    avg_neighbors = n_edges / n_stocks if n_stocks > 0 else 0
    sparsity = 1 - n_edges / (n_stocks * n_stocks) if n_stocks > 0 else 0
    
    logger.info(f"构建完成:")
    logger.info(f"  股票数量: {n_stocks}")
    logger.info(f"  边数: {n_edges:,}")
    logger.info(f"  平均邻居数: {avg_neighbors:.1f}")
    logger.info(f"  稀疏度: {sparsity:.2%}")
    
    # 保存
    metadata = {
        'graph_type': args.type,
        'top_k': args.top_k if args.type != 'industry' else None,
        'alpha': args.alpha if args.type == 'hybrid' else None,
        'corr_method': args.corr_method if args.type != 'industry' else None,
        'add_self_loop': not args.no_self_loop,
        'normalize': args.normalize,
    }
    
    save_adjacency_matrix(
        adj_matrix=adj_matrix,
        stock_list=stock_list,
        stock_to_idx=stock_to_idx,
        save_path=args.output,
        metadata=metadata,
    )
    
    print(f"\n✅ 邻接矩阵构建完成!")
    print(f"   类型: {args.type}")
    print(f"   形状: {adj_matrix.shape}")
    print(f"   保存至: {args.output}")


if __name__ == '__main__':
    main()
