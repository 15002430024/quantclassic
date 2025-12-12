"""
graph - 图数据加载器模块

提供按交易日组织的批次数据加载，支持动态图构建。
主要用于 GNN 模型训练场景。

核心组件：
- DailyBatchDataset: 按日组织数据集
- DailyGraphDataLoader: 日批次图数据加载器
- collate_daily: 日批次 collate 函数
- DailyLoaderConfig: 配置类
- create_daily_loader: 工厂函数
"""

from .daily_graph_loader import (
    DailyLoaderConfig,
    DailyBatchDataset,
    DailyGraphDataLoader,
    DailyBatchSampler,
    collate_daily,
    create_daily_loader,
)

__all__ = [
    'DailyLoaderConfig',
    'DailyBatchDataset',
    'DailyGraphDataLoader',
    'DailyBatchSampler',
    'collate_daily',
    'create_daily_loader',
]
