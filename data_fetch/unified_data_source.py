"""
统一数据源接口
支持多数据源（ClickHouse、RQ 等）的统一访问
"""
import logging
import os
from typing import List, Dict, Optional, Union
import pandas as pd

from .config_manager import ConfigManager
from .data_fetcher import DataFetcher
from .clickhouse_fetcher import ClickHouseFetcher
from .field_mapper import FieldMapper
from .resume_manager import ResumeManager

logger = logging.getLogger(__name__)


class UnifiedDataSource:
    """统一数据源接口"""
    
    SUPPORTED_SOURCES = ['rq', 'clickhouse']
    
    def __init__(self, config: ConfigManager):
        """
        初始化统一数据源
        
        Args:
            config: 配置管理器实例
        """
        self.config = config
        self.source_type = config.data_source.source.lower()
        self.resume_manager = ResumeManager(
            config.storage,
            enabled=getattr(config.process, 'resume_enabled', True)
        )
        
        # 验证数据源类型
        if self.source_type not in self.SUPPORTED_SOURCES:
            raise ValueError(
                f"不支持的数据源: {self.source_type}. "
                f"支持的数据源: {self.SUPPORTED_SOURCES}"
            )
        
        # 初始化相应的数据获取器
        self.fetcher = self._init_fetcher()
        self.field_mapper = FieldMapper()
        
        logger.info(f"统一数据源初始化完成: {self.source_type}")
    
    def _init_fetcher(self) -> Union[DataFetcher, ClickHouseFetcher]:
        """初始化数据获取器"""
        if self.source_type == 'rq':
            return DataFetcher(self.config, resume_manager=self.resume_manager)
        elif self.source_type == 'clickhouse':
            return ClickHouseFetcher(self.config)
        else:
            raise ValueError(f"无法初始化数据获取器: {self.source_type}")
    
    def get_stock_list(self, standardize: bool = True) -> pd.DataFrame:
        """
        获取股票/ETF 列表
        
        Args:
            standardize: 是否标准化字段
            
        Returns:
            股票列表 DataFrame
        """
        logger.info(f"=== 从 {self.source_type} 获取股票列表 ===")
        
        if self.source_type == 'rq':
            df = self.fetcher.get_stock_list()
        elif self.source_type == 'clickhouse':
            df = self.fetcher.get_etf_list()
        else:
            raise ValueError(f"不支持的数据源: {self.source_type}")
        
        # 字段标准化
        if standardize and not df.empty:
            df = self.field_mapper.map_fields(df, source=self.source_type)
            df = self.field_mapper.standardize_data_types(df)
        
        return df
    
    def get_trading_calendar(self, standardize: bool = True) -> pd.DataFrame:
        """
        获取交易日历
        
        Args:
            standardize: 是否标准化字段
            
        Returns:
            交易日历 DataFrame
        """
        logger.info(f"=== 从 {self.source_type} 获取交易日历 ===")
        
        df = self.fetcher.get_trading_calendar()
        
        # 字段标准化
        if standardize and not df.empty:
            df = self.field_mapper.map_fields(df, source=self.source_type)
            df = self.field_mapper.standardize_data_types(df)
        
        return df
    
    def get_price_data(
        self,
        order_book_ids: Optional[List[str]] = None,
        standardize: bool = True
    ) -> pd.DataFrame:
        """
        获取价格行情数据
        
        Args:
            order_book_ids: 股票代码列表
            standardize: 是否标准化字段
            
        Returns:
            价格数据 DataFrame
        """
        logger.info(f"=== 从 {self.source_type} 获取价格数据 ===")
        
        if self.source_type == 'rq':
            # 如果未提供股票列表，先获取股票列表
            if order_book_ids is None:
                df_stocks = self.get_stock_list(standardize=False)
                order_book_ids = df_stocks['order_book_id'].tolist()
            
            existing_df = self._load_existing_daily('daily_price')
            if existing_df is not None and order_book_ids is not None:
                existing_df = existing_df[existing_df['order_book_id'].isin(order_book_ids)]
            df = self.fetcher.get_price_data(order_book_ids, existing_df)
        
        elif self.source_type == 'clickhouse':
            df = self.fetcher.get_daily_data(symbols=order_book_ids)
        
        else:
            raise ValueError(f"不支持的数据源: {self.source_type}")
        
        # 字段标准化
        if standardize and not df.empty:
            df = self.field_mapper.map_fields(df, source=self.source_type)
            df = self.field_mapper.standardize_data_types(df)
        
        return df
    
    def get_valuation_data(
        self,
        order_book_ids: Optional[List[str]] = None,
        standardize: bool = True
    ) -> pd.DataFrame:
        """
        获取估值数据
        
        Args:
            order_book_ids: 股票代码列表
            standardize: 是否标准化字段
            
        Returns:
            估值数据 DataFrame
        """
        logger.info(f"=== 从 {self.source_type} 获取估值数据 ===")
        
        if self.source_type == 'rq':
            # 如果未提供股票列表，先获取股票列表
            if order_book_ids is None:
                df_stocks = self.get_stock_list(standardize=False)
                order_book_ids = df_stocks['order_book_id'].tolist()
            
            existing_df = self._load_existing_daily('daily_valuation')
            if existing_df is not None and order_book_ids is not None:
                existing_df = existing_df[existing_df['order_book_id'].isin(order_book_ids)]
            df = self.fetcher.get_valuation_data(order_book_ids, existing_df)
        
        elif self.source_type == 'clickhouse':
            # ClickHouse 的日频数据已包含估值字段
            df = self.fetcher.get_daily_data(symbols=order_book_ids)
            
            # 仅保留估值相关字段
            valuation_cols = ['order_book_id', 'trade_date', 'pe', 'pb', 'total_mv', 'circ_mv']
            existing_cols = [col for col in valuation_cols if col in df.columns]
            df = df[existing_cols]
        
        else:
            raise ValueError(f"不支持的数据源: {self.source_type}")
        
        # 字段标准化
        if standardize and not df.empty:
            df = self.field_mapper.map_fields(df, source=self.source_type)
            df = self.field_mapper.standardize_data_types(df)
        
        return df
    
    def get_industry_data(
        self,
        order_book_ids: List[str],
        standardize: bool = True
    ) -> pd.DataFrame:
        """
        获取行业分类数据
        
        Args:
            order_book_ids: 股票代码列表
            standardize: 是否标准化字段
            
        Returns:
            行业分类 DataFrame
        """
        logger.info(f"=== 从 {self.source_type} 获取行业数据 ===")
        
        df = self.fetcher.get_industry_data(order_book_ids)
        
        # 字段标准化
        if standardize and not df.empty:
            df = self.field_mapper.map_fields(df, source=self.source_type)
            df = self.field_mapper.standardize_data_types(df)
        
        return df
    
    def get_all_daily_data(
        self,
        order_book_ids: Optional[List[str]] = None,
        merge_data: bool = True,
        standardize: bool = True,
        add_derived_fields: bool = True
    ) -> pd.DataFrame:
        """
        获取完整的日频数据（价格 + 估值 + 行业）
        
        Args:
            order_book_ids: 股票代码列表
            merge_data: 是否合并多个数据表
            standardize: 是否标准化字段
            add_derived_fields: 是否添加衍生字段
            
        Returns:
            完整日频数据 DataFrame
        """
        logger.info("=== 获取完整日频数据 ===")
        
        # 获取价格数据
        df_price = self.get_price_data(order_book_ids, standardize=standardize)
        
        if df_price.empty:
            logger.warning("价格数据为空，返回空 DataFrame")
            return df_price
        
        # 对于 ClickHouse 数据源，日频数据已经很完整
        if self.source_type == 'clickhouse':
            df_final = df_price
            
        # 对于 RQ 数据源，需要合并估值数据
        elif self.source_type == 'rq' and merge_data:
            # 获取股票代码列表
            if order_book_ids is None:
                order_book_ids = df_price['order_book_id'].unique().tolist()
            
            # 获取估值数据
            df_valuation = self.get_valuation_data(order_book_ids, standardize=standardize)
            
            # 合并数据
            if not df_valuation.empty:
                df_final = pd.merge(
                    df_price,
                    df_valuation,
                    on=['order_book_id', 'trade_date'],
                    how='left'
                )
                logger.info("  ✓ 价格数据与估值数据已合并")
            else:
                df_final = df_price
        
        else:
            df_final = df_price
        
        # 添加衍生字段
        if add_derived_fields:
            df_final = self.field_mapper.add_derived_fields(df_final)
        
        logger.info(f"完整数据获取完成: {len(df_final)} 条记录, {len(df_final.columns)} 个字段")
        
        return df_final

    def _load_existing_daily(self, filename: str) -> Optional[pd.DataFrame]:
        if not getattr(self.config.process, 'resume_enabled', True):
            return None
        full_path = self.config.storage.get_full_path(
            self.config.storage.daily_data_dir,
            f"{filename}.{self.config.storage.file_format}"
        )
        if not os.path.exists(full_path):
            return None
        try:
            if self.config.storage.file_format == 'parquet':
                df = pd.read_parquet(full_path)
            elif self.config.storage.file_format == 'csv':
                df = pd.read_csv(full_path)
            elif self.config.storage.file_format == 'hdf5':
                df = pd.read_hdf(full_path, key='data')
            else:
                logger.warning("不支持的文件格式, 无法加载已有数据: %s", self.config.storage.file_format)
                return None
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
                df = df.dropna(subset=['trade_date'])
            logger.info("断点续传: 统一数据源加载已有文件 %s", full_path)
            return df
        except Exception as exc:
            logger.warning("统一数据源加载已有数据失败, 忽略 %s: %s", full_path, exc)
            return None
    
    def get_basic_info(
        self,
        order_book_ids: Optional[List[str]] = None,
        standardize: bool = True
    ) -> pd.DataFrame:
        """
        获取基本信息
        
        Args:
            order_book_ids: 股票代码列表
            standardize: 是否标准化字段
            
        Returns:
            基本信息 DataFrame
        """
        logger.info(f"=== 从 {self.source_type} 获取基本信息 ===")
        
        if self.source_type == 'clickhouse':
            df = self.fetcher.get_basic_info(order_book_ids)
        elif self.source_type == 'rq':
            # RQ 的基本信息在 get_stock_list 中
            df = self.get_stock_list(standardize=False)
            if order_book_ids is not None:
                df = df[df['order_book_id'].isin(order_book_ids)]
        else:
            raise ValueError(f"不支持的数据源: {self.source_type}")
        
        # 字段标准化
        if standardize and not df.empty:
            df = self.field_mapper.map_fields(df, source=self.source_type)
            df = self.field_mapper.standardize_data_types(df)
        
        return df
    
    def save_data(self, df: pd.DataFrame, subdir: str, filename: str):
        """
        保存数据
        
        Args:
            df: 待保存的 DataFrame
            subdir: 子目录
            filename: 文件名
        """
        self.fetcher.save_data(df, subdir, filename)
    
    def test_connection(self) -> bool:
        """
        测试数据源连接
        
        Returns:
            连接是否成功
        """
        try:
            if self.source_type == 'clickhouse':
                return self.fetcher.test_connection()
            else:
                # RQ 在初始化时已经测试连接
                return True
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False
    
    def get_data_info(self) -> Dict:
        """
        获取数据源信息
        
        Returns:
            数据源信息字典
        """
        info = {
            'source_type': self.source_type,
            'start_date': self.config.time.start_date,
            'end_date': self.config.time.end_date,
            'frequency': self.config.time.frequency,
        }
        
        if self.source_type == 'clickhouse':
            info['database'] = self.config.data_source.clickhouse_config.get('database')
            info['host'] = self.config.data_source.clickhouse_config.get('host')
        
        return info
