"""
daily_graph_loader.py - 兼容层 (DEPRECATED)

此模块已迁移到 quantclassic.data_set.graph.daily_graph_loader

请更新导入路径：
    # 旧路径 (已弃用)
    from quantclassic.data_fetch.daily_graph_loader import DailyBatchDataset
    
    # 新路径 (推荐)
    from quantclassic.data_set.graph import DailyBatchDataset

此兼容层将在未来版本中移除。
"""

import warnings

# 发出弃用警告
warnings.warn(
    "quantclassic.data_fetch.daily_graph_loader 已弃用，"
    "请使用 quantclassic.data_set.graph 代替。"
    "此兼容层将在 v2.0.0 中移除。",
    DeprecationWarning,
    stacklevel=2
)

# 从新位置导入所有内容
from quantclassic.data_set.graph.daily_graph_loader import (
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
