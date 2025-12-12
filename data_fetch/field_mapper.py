"""
字段映射模块
将不同数据源的字段统一映射到标准字段名
"""
import logging
import pandas as pd
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class FieldMapper:
    """字段映射器"""
    
    # ClickHouse ETF 字段到标准字段的映射
    CLICKHOUSE_TO_STANDARD = {
        # 基础标识
        'Symbol': 'order_book_id',
        'TradingDate': 'trade_date',
        'FullName': 'symbol',
        'ShortName': 'short_name',
        
        # 价格数据
        'OpenPrice': 'open',
        'HighPrice': 'high',
        'LowPrice': 'low',
        'ClosePrice': 'close',
        'Volume': 'vol',
        'Amount': 'amount',
        
        # 涨跌数据
        'Change': 'change',
        'ChangeRatio': 'change_ratio',
        
        # 市场指标
        'TurnoverRate': 'turnover_rate',
        'Amplitude': 'amplitude',
        'AdjustedFactor': 'adj_factor',
        
        # 估值数据
        'PE': 'pe',
        'PB': 'pb',
        'TotalMarketValue': 'total_mv',
        'CirculationMarketValue': 'circ_mv',
        
        # ETF 特有数据
        'NetAssetValue': 'nav',
        'AccumulatedNetValue': 'acc_nav',
        'DailyReturn': 'daily_return',
        'ReturnDaily': 'return_daily',
        'PremiumRate': 'premium_rate',
        'IOPV': 'iopv',
        'FundShares': 'fund_shares',
        'DiscountRate': 'discount_rate',
        
        # 更多常用字段
        'PreClosePrice': 'prev_close',
        'LimitUp': 'limit_up',
        'LimitDown': 'limit_down',
        'AveragePrice': 'avg_price',
        
        # 基本信息
        'ETFType': 'etf_type',
        'InvestmentType': 'investment_type',
        'IndexCode': 'underlying_index',
        'ManagementCompany': 'fund_company',
        'ListingDate': 'listing_date',
        'ListingMarket': 'market',
        'ManagementFee': 'management_fee',
        'CustodyFee': 'custody_fee',
        'StateCode': 'state_code',
    }
    
    # 标准字段到 ClickHouse 字段的反向映射
    STANDARD_TO_CLICKHOUSE = {v: k for k, v in CLICKHOUSE_TO_STANDARD.items()}
    
    # 米筐（RQ）字段到标准字段的映射
    RQ_TO_STANDARD = {
        'order_book_id': 'order_book_id',
        'date': 'trade_date',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'vol',
        'total_turnover': 'amount',
        'pe_ratio': 'pe',
        'pe_ratio_ttm': 'pe_ttm',
        'pb_ratio': 'pb',
        'ps_ratio': 'ps',
        'ps_ratio_ttm': 'ps_ttm',
        'market_cap': 'total_mv',
        'a_share_market_val_in_circulation': 'circ_mv',
    }
    
    # 标准字段的必需字段列表
    REQUIRED_FIELDS = ['order_book_id', 'trade_date']
    
    # 标准字段的价格字段
    PRICE_FIELDS = ['open', 'high', 'low', 'close', 'vol', 'amount']
    
    # 标准字段的估值字段
    VALUATION_FIELDS = ['pe', 'pb', 'total_mv', 'circ_mv']
    
    # ETF 特有字段
    ETF_SPECIFIC_FIELDS = [
        'nav', 'acc_nav', 'premium_rate', 'etf_type', 'underlying_index',
        'iopv', 'fund_shares', 'discount_rate', 'prev_close', 'limit_up', 'limit_down'
    ]
    
    @classmethod
    def map_fields(
        cls,
        df: pd.DataFrame,
        source: str = 'clickhouse',
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        将数据源字段映射到标准字段
        
        Args:
            df: 待映射的 DataFrame
            source: 数据源类型 ('clickhouse', 'rq')
            inplace: 是否就地修改
            
        Returns:
            映射后的 DataFrame
        """
        if not inplace:
            df = df.copy()
        
        # 选择映射字典
        if source.lower() == 'clickhouse':
            mapping = cls.CLICKHOUSE_TO_STANDARD
        elif source.lower() == 'rq':
            mapping = cls.RQ_TO_STANDARD
        else:
            logger.warning(f"未知的数据源类型: {source}, 不进行字段映射")
            return df
        
        # 执行映射
        existing_cols = [col for col in df.columns if col in mapping]
        rename_dict = {col: mapping[col] for col in existing_cols}
        
        df = df.rename(columns=rename_dict)
        
        logger.info(f"字段映射完成: {len(rename_dict)} 个字段被重命名")
        logger.debug(f"映射详情: {rename_dict}")
        
        return df
    
    @classmethod
    def reverse_map_fields(
        cls,
        df: pd.DataFrame,
        target: str = 'clickhouse',
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        将标准字段映射回数据源字段
        
        Args:
            df: 待映射的 DataFrame
            target: 目标数据源类型 ('clickhouse', 'rq')
            inplace: 是否就地修改
            
        Returns:
            映射后的 DataFrame
        """
        if not inplace:
            df = df.copy()
        
        # 选择反向映射字典
        if target.lower() == 'clickhouse':
            mapping = cls.STANDARD_TO_CLICKHOUSE
        elif target.lower() == 'rq':
            mapping = {v: k for k, v in cls.RQ_TO_STANDARD.items()}
        else:
            logger.warning(f"未知的目标数据源类型: {target}, 不进行字段映射")
            return df
        
        # 执行映射
        existing_cols = [col for col in df.columns if col in mapping]
        rename_dict = {col: mapping[col] for col in existing_cols}
        
        df = df.rename(columns=rename_dict)
        
        logger.info(f"反向字段映射完成: {len(rename_dict)} 个字段被重命名")
        
        return df
    
    @classmethod
    def validate_standard_fields(cls, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        验证标准字段
        
        Args:
            df: 待验证的 DataFrame
            
        Returns:
            验证结果字典，包含 'missing' 和 'extra' 字段
        """
        current_fields = set(df.columns)
        
        # 检查必需字段
        required_fields = set(cls.REQUIRED_FIELDS)
        missing_required = required_fields - current_fields
        
        # 检查标准字段
        all_standard_fields = set(cls.CLICKHOUSE_TO_STANDARD.values())
        missing_optional = all_standard_fields - current_fields - required_fields
        extra_fields = current_fields - all_standard_fields
        
        result = {
            'missing_required': list(missing_required),
            'missing_optional': list(missing_optional),
            'extra': list(extra_fields),
            'is_valid': len(missing_required) == 0
        }
        
        if result['missing_required']:
            logger.warning(f"缺少必需字段: {result['missing_required']}")
        
        logger.info(f"字段验证完成: {len(current_fields)} 个字段")
        
        return result
    
    @classmethod
    def standardize_data_types(cls, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        标准化数据类型
        
        Args:
            df: 待标准化的 DataFrame
            inplace: 是否就地修改
            
        Returns:
            标准化后的 DataFrame
        """
        if not inplace:
            df = df.copy()
        
        # 日期列
        date_cols = ['trade_date', 'listing_date', 'listed_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                logger.debug(f"  {col} -> datetime64")
        
        # 浮点数列（价格、估值等）
        float_cols = [
            'open', 'high', 'low', 'close', 'amount',
            'change', 'change_ratio', 'turnover_rate', 'amplitude',
            'adj_factor', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
            'total_mv', 'circ_mv', 'nav', 'acc_nav',
            'daily_return', 'return_daily', 'premium_rate',
            'management_fee', 'custody_fee', 'vwap',
            'iopv', 'discount_rate', 'prev_close', 'limit_up', 'limit_down', 'avg_price'
        ]
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
        
        # 整数列（成交量）
        int_cols = ['vol', 'fund_shares']
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        
        # 字符串列
        str_cols = [
            'order_book_id', 'symbol', 'short_name', 'name',
            'etf_type', 'investment_type', 'underlying_index',
            'fund_company', 'market', 'industry_code', 'industry_name'
        ]
        for col in str_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].astype('string')
        
        logger.info("数据类型标准化完成")
        
        return df
    
    @classmethod
    def add_derived_fields(cls, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        添加衍生字段
        
        Args:
            df: 待处理的 DataFrame
            inplace: 是否就地修改
            
        Returns:
            添加衍生字段后的 DataFrame
        """
        if not inplace:
            df = df.copy()
        
        # 计算日收益率（如果有价格数据）
        if 'close' in df.columns and 'order_book_id' in df.columns:
            df = df.sort_values(['order_book_id', 'trade_date'])
            df['return'] = df.groupby('order_book_id')['close'].pct_change()
            logger.info("  ✓ 添加字段: return (日收益率)")
        
        # 计算对数收益率
        if 'return' in df.columns:
            import numpy as np
            df['log_return'] = np.log1p(df['return'])
            logger.info("  ✓ 添加字段: log_return (对数收益率)")
        
        # 添加市场标识（根据代码）
        if 'order_book_id' in df.columns and 'market' not in df.columns:
            df['market'] = df['order_book_id'].apply(cls._extract_market)
            logger.info("  ✓ 添加字段: market (市场标识)")
        
        # 添加交易日期相关字段
        if 'trade_date' in df.columns:
            df['year'] = df['trade_date'].dt.year
            df['month'] = df['trade_date'].dt.month
            df['quarter'] = df['trade_date'].dt.quarter
            df['weekday'] = df['trade_date'].dt.weekday
            logger.info("  ✓ 添加字段: year, month, quarter, weekday")
        
        return df
    
    @staticmethod
    def _extract_market(code: str) -> str:
        """从股票代码提取市场标识"""
        if pd.isna(code):
            return 'unknown'
        
        code = str(code).upper()
        if '.XSHG' in code or '.SH' in code:
            return 'SH'
        elif '.XSHE' in code or '.SZ' in code:
            return 'SZ'
        else:
            return 'unknown'
    
    @classmethod
    def get_standard_field_list(cls, include_etf_specific: bool = True) -> List[str]:
        """
        获取标准字段列表
        
        Args:
            include_etf_specific: 是否包含 ETF 特有字段
            
        Returns:
            标准字段列表
        """
        fields = (
            cls.REQUIRED_FIELDS +
            cls.PRICE_FIELDS +
            cls.VALUATION_FIELDS
        )
        
        if include_etf_specific:
            fields += cls.ETF_SPECIFIC_FIELDS
        
        return list(set(fields))
    
    @classmethod
    def select_standard_fields(
        cls,
        df: pd.DataFrame,
        include_etf_specific: bool = True,
        keep_extra: bool = False
    ) -> pd.DataFrame:
        """
        选择标准字段
        
        Args:
            df: 待处理的 DataFrame
            include_etf_specific: 是否包含 ETF 特有字段
            keep_extra: 是否保留额外字段
            
        Returns:
            仅包含标准字段的 DataFrame
        """
        standard_fields = cls.get_standard_field_list(include_etf_specific)
        
        if keep_extra:
            # 保留所有字段
            return df
        else:
            # 仅保留存在的标准字段
            available_fields = [f for f in standard_fields if f in df.columns]
            return df[available_fields]
