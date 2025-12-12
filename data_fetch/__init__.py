"""
量化数据获取工具

工程化的数据获取、处理和特征工程工具
支持多数据源：米筐(RiceQuant)、ClickHouse 等
"""

from .config_manager import ConfigManager
from .data_fetcher import DataFetcher
from .data_processor import DataProcessor
from .data_validator import DataValidator
from .pipeline import QuantDataPipeline, DataPipeline
from .clickhouse_fetcher import ClickHouseFetcher
from .field_mapper import FieldMapper
from .unified_data_source import UnifiedDataSource

__version__ = '1.1.0'

__all__ = [
    'ConfigManager',
    'DataFetcher',
    'DataProcessor',
    'DataValidator',
    'QuantDataPipeline',
    'DataPipeline',  # 兼容旧接口
    'ClickHouseFetcher',
    'FieldMapper',
    'UnifiedDataSource',
]
