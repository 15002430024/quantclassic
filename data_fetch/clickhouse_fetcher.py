"""
ClickHouse 数据获取器
基于 quantchdb 封装，用于获取 ETF 数据
"""
import logging
from typing import List, Dict, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta

try:
    from quantchdb import ClickHouseDatabase
except ImportError:
    ClickHouseDatabase = None

from .config_manager import ConfigManager

logger = logging.getLogger(__name__)


class ClickHouseFetcher:
    """ClickHouse 数据获取器"""
    
    def __init__(self, config: ConfigManager):
        """
        初始化 ClickHouse 数据获取器
        
        Args:
            config: 配置管理器实例
        """
        self.config = config
        self._init_connection()
    
    def _init_connection(self):
        """初始化 ClickHouse 连接"""
        if ClickHouseDatabase is None:
            raise ImportError("未安装 quantchdb, 请运行: pip install quantchdb")
        
        try:
            # 从配置中获取 ClickHouse 连接参数
            ch_config = self.config.data_source.clickhouse_config
            
            self.db = ClickHouseDatabase(
                config=ch_config,
                terminal_log=False,
                file_log=False
            )
            
            logger.info("ClickHouse 连接初始化成功")
            logger.info(f"数据库: {ch_config.get('database', 'etf')}")
            
        except Exception as e:
            logger.error(f"ClickHouse 连接初始化失败: {e}")
            raise
    
    def get_etf_list(self) -> pd.DataFrame:
        """
        获取 ETF 列表
        
        Returns:
            包含 ETF 基本信息的 DataFrame
        """
        logger.info("=== 获取 ETF 列表 ===")
        
        try:
            sql = """
            SELECT 
                Symbol as order_book_id,
                FullName as symbol,
                TradingDate as listed_date,
                ListingMarket as market
            FROM etf.etf_info
            WHERE StateCode = 0  -- 正常交易状态
            """
            
            df_etf = self.db.fetch(sql)
            
            # 数据清洗
            if not df_etf.empty:
                # 处理日期格式
                if 'listed_date' in df_etf.columns:
                    df_etf['listed_date'] = pd.to_datetime(df_etf['listed_date'])
                
                # 根据配置应用筛选规则
                df_etf = self._apply_filters(df_etf)
            
            logger.info(f"获取到 {len(df_etf)} 只 ETF")
            
            return df_etf
            
        except Exception as e:
            logger.error(f"获取 ETF 列表失败: {e}")
            raise
    
    def _apply_filters(self, df_etf: pd.DataFrame) -> pd.DataFrame:
        """应用筛选规则"""
        original_count = len(df_etf)
        
        # 筛选上市时间
        end_date = pd.to_datetime(self.config.time.end_date)
        if 'listed_date' in df_etf.columns:
            df_etf = df_etf[df_etf['listed_date'] <= end_date]
            logger.info(f"日期筛选: {original_count} -> {len(df_etf)}")
        
        return df_etf
    
    def get_trading_calendar(self) -> pd.DataFrame:
        """
        获取交易日历
        
        Returns:
            交易日历 DataFrame
        """
        logger.info("=== 获取交易日历 ===")
        
        try:
            sql = f"""
            SELECT DISTINCT TradingDate as trade_date
            FROM etf.etf_daily
            WHERE TradingDate >= '{self.config.time.start_date}'
              AND TradingDate <= '{self.config.time.end_date}'
            ORDER BY TradingDate
            """
            
            df_calendar = self.db.fetch(sql)
            df_calendar['is_trading_day'] = 1
            
            logger.info(f"获取到 {len(df_calendar)} 个交易日")
            
            return df_calendar
            
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            raise
    
    def get_daily_data(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取 ETF 日频数据
        
        Args:
            symbols: ETF 代码列表，如果为 None 则获取所有 ETF
            
        Returns:
            日频数据 DataFrame
        """
        logger.info("=== 获取 ETF 日频数据 ===")
        
        try:
            # 构建 SQL 查询
            # 使用 SELECT * 获取所有字段，避免遗漏数据
            # 字段映射将在 FieldMapper 中处理
            sql = f"""
            SELECT *
            FROM etf.etf_daily
            WHERE TradingDate >= '{self.config.time.start_date}'
              AND TradingDate <= '{self.config.time.end_date}'
              AND StateCode = 0
            """
            
            # 如果指定了股票池，添加过滤条件
            if symbols is not None and len(symbols) > 0:
                symbols_str = "', '".join(symbols)
                sql += f" AND Symbol IN ('{symbols_str}')"
            
            sql += " ORDER BY TradingDate, Symbol"
            
            logger.info(f"开始提取数据...")
            df_daily = self.db.fetch(sql)
            
            # 数据类型转换
            if not df_daily.empty:
                df_daily = self._convert_data_types(df_daily)
            
            logger.info(f"提取完成！共 {len(df_daily)} 条记录")
            if not df_daily.empty:
                # 尝试使用标准字段名或原始字段名
                symbol_col = 'order_book_id' if 'order_book_id' in df_daily.columns else 'Symbol'
                date_col = 'trade_date' if 'trade_date' in df_daily.columns else 'TradingDate'
                
                if symbol_col in df_daily.columns:
                    logger.info(f"包含 {df_daily[symbol_col].nunique()} 个不同的 ETF")
                if date_col in df_daily.columns:
                    logger.info(f"日期范围: {df_daily[date_col].min()} 到 {df_daily[date_col].max()}")
            
            return df_daily
            
        except Exception as e:
            logger.error(f"获取日频数据失败: {e}")
            raise
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换数据类型"""
        # 日期列 (包括原始字段名和标准字段名)
        date_cols = ['trade_date', 'TradingDate']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # 数值列 (包括原始字段名和标准字段名)
        # 这里只列出部分关键字段，pandas 的 to_numeric 通常不需要显式转换所有列，
        # 除非原始数据是字符串格式。ClickHouse 驱动通常会返回正确的数值类型。
        # 为了保险起见，我们对可能包含非数值的列进行处理
        numeric_cols = [
            'open', 'high', 'low', 'close', 'vol', 'amount',
            'OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice', 'Volume', 'Amount',
            'change', 'change_ratio', 'turnover_rate', 'amplitude',
            'Change', 'ChangeRatio', 'TurnoverRate', 'Amplitude',
            'adj_factor', 'pe', 'pb', 'total_mv', 'circ_mv',
            'AdjustedFactor', 'PE', 'PB', 'TotalMarketValue', 'CirculationMarketValue',
            'nav', 'acc_nav', 'ret', 'return_daily', 'premium_rate',
            'NetAssetValue', 'AccumulatedNetValue', 'DailyReturn', 'ReturnDaily', 'PremiumRate'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def get_industry_data(self, order_book_ids: List[str]) -> pd.DataFrame:
        """
        获取行业分类数据
        注：ETF 通常没有传统的行业分类，这里返回 ETF 类型分类
        
        Args:
            order_book_ids: ETF 代码列表
            
        Returns:
            行业分类 DataFrame（ETF 类型分类）
        """
        logger.info("=== 获取 ETF 类型分类 ===")
        
        try:
            sql = """
            SELECT 
                Symbol as order_book_id,
                ETFType as industry_code,
                ETFType as industry_name
            FROM etf.etf_info
            WHERE StateCode = 0
            """
            
            if order_book_ids:
                symbols_str = "', '".join(order_book_ids)
                sql += f" AND Symbol IN ('{symbols_str}')"
            
            df_industry = self.db.fetch(sql)
            
            logger.info(f"成功获取 {len(df_industry)} 只 ETF 的类型分类")
            if not df_industry.empty:
                logger.info(f"\nETF 类型分布:\n{df_industry['industry_name'].value_counts()}")
            
            return df_industry
            
        except Exception as e:
            logger.warning(f"获取 ETF 类型分类失败: {e}")
            # 返回空 DataFrame
            return pd.DataFrame(columns=['order_book_id', 'industry_code', 'industry_name'])
    
    def get_basic_info(self, order_book_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取 ETF 基本信息
        
        Args:
            order_book_ids: ETF 代码列表
            
        Returns:
            基本信息 DataFrame
        """
        logger.info("=== 获取 ETF 基本信息 ===")
        
        try:
            sql = """
            SELECT 
                Symbol as order_book_id,
                FullName as name,
                ShortName as short_name,
                ETFType as etf_type,
                InvestmentType as investment_type,
                IndexCode as underlying_index,
                ManagementCompany as fund_company,
                ListingDate as listing_date,
                ListingMarket as market,
                ManagementFee as management_fee,
                CustodyFee as custody_fee,
                StateCode as state_code
            FROM etf.etf_info
            WHERE StateCode = 0
            """
            
            if order_book_ids:
                symbols_str = "', '".join(order_book_ids)
                sql += f" AND Symbol IN ('{symbols_str}')"
            
            df_info = self.db.fetch(sql)
            
            # 日期转换
            if not df_info.empty and 'listing_date' in df_info.columns:
                df_info['listing_date'] = pd.to_datetime(df_info['listing_date'])
            
            logger.info(f"获取到 {len(df_info)} 只 ETF 的基本信息")
            
            return df_info
            
        except Exception as e:
            logger.error(f"获取 ETF 基本信息失败: {e}")
            raise
    
    def save_data(self, df: pd.DataFrame, subdir: str, filename: str):
        """
        保存数据
        
        Args:
            df: 待保存的 DataFrame
            subdir: 子目录（basic_data/daily_data）
            filename: 文件名（不含扩展名）
        """
        file_format = self.config.storage.file_format
        full_filename = f"{filename}.{file_format}"
        filepath = self.config.storage.get_full_path(subdir, full_filename)
        
        if file_format == 'parquet':
            df.to_parquet(filepath, index=False, compression=self.config.storage.compression)
        elif file_format == 'csv':
            df.to_csv(filepath, index=False)
        elif file_format == 'hdf5':
            df.to_hdf(filepath, key='data', mode='w')
        else:
            raise ValueError(f"不支持的文件格式: {file_format}")
        
        logger.info(f"数据已保存: {filepath}")
    
    def test_connection(self) -> bool:
        """
        测试数据库连接
        
        Returns:
            连接是否成功
        """
        try:
            sql = "SELECT 1 as test"
            result = self.db.fetch(sql)
            logger.info("ClickHouse 连接测试成功")
            return True
        except Exception as e:
            logger.error(f"ClickHouse 连接测试失败: {e}")
            return False
