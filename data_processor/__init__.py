"""
数据预处理模块 - Data Processor Package

提供数据预处理功能:
- 标准化/归一化
- 行业市值中性化
- SimStock中性化
- 极值处理
- 缺失值处理
- 窗口级数据处理（研报标准）
"""

from .preprocess_config import (
    ProcessMethod,
    ProcessingStep,
    NeutralizeConfig,
    PreprocessConfig,
    PreprocessTemplates
)
from .feature_processor import FeatureProcessor
from .data_preprocessor import DataPreprocessor
from .label_generator import (
    LabelGenerator,
    LabelConfig,
    generate_future_returns
)
from .window_processor import (
    WindowProcessor,
    WindowProcessConfig,
    process_price_log_transform,
    process_volume_normalize
)

__version__ = "1.1.0"

__all__ = [
    'ProcessMethod',
    'ProcessingStep',
    'NeutralizeConfig',
    'PreprocessConfig',
    'PreprocessTemplates',
    'FeatureProcessor',
    'DataPreprocessor',
    'LabelGenerator',
    'LabelConfig',
    'generate_future_returns',
    # 窗口处理
    'WindowProcessor',
    'WindowProcessConfig',
    'process_price_log_transform',
    'process_volume_normalize'
]
