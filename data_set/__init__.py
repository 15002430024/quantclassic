"""
DataManager 模块 - 工程化数据管理封装

提供统一的数据加载、特征工程、数据划分和验证功能。

主要组件:
- DataConfig: 配置管理
- DataLoaderEngine: 数据加载器
- FeatureEngineer: 特征工程师
- DataSplitter: 数据划分器
- DataValidator: 数据验证器
- DatasetFactory: 数据集工厂
- DataManager: 主控类

快速开始:
    >>> from data_set import DataManager, DataConfig
    >>> config = DataConfig()
    >>> manager = DataManager(config)
    >>> loaders = manager.run_full_pipeline()
    >>> # 使用 loaders.train, loaders.val, loaders.test 进行训练
"""

from .config import DataConfig, ConfigTemplates
from .loader import DataLoaderEngine
from .feature_engineer import FeatureEngineer
from .splitter import (
    DataSplitter,
    TimeSeriesSplitter,
    StratifiedStockSplitter,
    RollingWindowSplitter,
    RandomSplitter,
    create_splitter
)
from .validator import DataValidator, ValidationReport
from .factory import (
    DatasetFactory,
    DatasetCollection,
    LoaderCollection,
    TimeSeriesStockDataset,
    TimeSeriesStockDatasetWithDate,
    CrossSectionalBatchSampler,
    InferenceDataset
)
from .manager import DataManager
from .rolling_trainer import RollingWindowTrainer

# 图数据加载器 (GNN 训练)
from .graph import (
    DailyLoaderConfig,
    DailyBatchDataset,
    DailyGraphDataLoader,
    DailyBatchSampler,
    collate_daily,
    create_daily_loader,
)

__version__ = "1.1.0"
__author__ = "quantclassic team"

__all__ = [
    # 配置
    'DataConfig',
    'ConfigTemplates',
    
    # 核心组件
    'DataLoaderEngine',
    'FeatureEngineer',
    'DataValidator',
    'DatasetFactory',
    'DataManager',
    'RollingWindowTrainer',
    
    # 划分器
    'DataSplitter',
    'TimeSeriesSplitter',
    'StratifiedStockSplitter',
    'RollingWindowSplitter',
    'RandomSplitter',
    'create_splitter',
    
    # 数据结构
    'ValidationReport',
    'DatasetCollection',
    'LoaderCollection',
    'TimeSeriesStockDataset',
    'TimeSeriesStockDatasetWithDate',
    'CrossSectionalBatchSampler',
    'InferenceDataset',
    
    # 图数据加载器 (NEW)
    'DailyLoaderConfig',
    'DailyBatchDataset',
    'DailyGraphDataLoader',
    'DailyBatchSampler',
    'collate_daily',
    'create_daily_loader',
]

# 版本信息
def get_version():
    """获取版本信息"""
    return __version__

# 模块初始化提示
print(f"✅ DataManager v{__version__} 已加载")
