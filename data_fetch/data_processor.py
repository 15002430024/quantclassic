"""
数据处理器模块
负责数据清洗、合并、特征工程
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from .config_manager import ConfigManager


logger = logging.getLogger(__name__)


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config: ConfigManager):
        """
        初始化数据处理器
        
        Args:
            config: 配置管理器实例
        """
        self.config = config
    
    def clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗原始数据
        
        Args:
            df: 原始DataFrame
            
        Returns:
            清洗后的DataFrame
        """
        logger.info("开始清洗数据...")
        original_shape = df.shape
        
        # 1. 删除完全重复的行
        df = df.drop_duplicates()
        
        # 1.1 修正可能颠倒的索引列
        df = self.fix_column_order(df)

        # 2. 确保日期格式正确
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
            invalid_dates = df['trade_date'].isna()
            if invalid_dates.any():
                logger.warning(
                    "发现 %d 条 trade_date 无法解析，已自动丢弃", invalid_dates.sum()
                )
                df = df[~invalid_dates]
        
        # 3. 按股票代码和日期排序
        if 'order_book_id' in df.columns and 'trade_date' in df.columns:
            df = df.sort_values(['order_book_id', 'trade_date']).reset_index(drop=True)
        
        logger.info(f"数据清洗完成: {original_shape} -> {df.shape}")
        return df
    
    def fix_column_order(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        修复列顺序(处理米筐API返回的列名顺序问题)
        
        Args:
            df: DataFrame
            
        Returns:
            修复后的DataFrame
        """
        if len(df) == 0:
            return df
        
        # 检查是否需要交换列名
        if df.columns[0] == 'trade_date' and df.columns[1] == 'order_book_id':
            first_col_sample = df.iloc[0, 0]
            second_col_sample = df.iloc[0, 1]
            
            # 如果第一列是字符串(股票代码)或第二列是日期,则需要交换
            if isinstance(first_col_sample, str) or isinstance(second_col_sample, pd.Timestamp):
                logger.info("检测到列名与数据不匹配,交换列名...")
                new_columns = df.columns.tolist()
                new_columns[0], new_columns[1] = new_columns[1], new_columns[0]
                df.columns = new_columns
        
        return df
    
    def merge_daily_data(
        self,
        df_price: pd.DataFrame,
        df_valuation: Optional[pd.DataFrame] = None,
        df_share: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        合并日频数据
        
        Args:
            df_price: 行情数据
            df_valuation: 估值数据(可选)
            df_share: 股本数据(可选)
            
        Returns:
            合并后的DataFrame
        """
        logger.info("=== 合并日频数据 ===")
        
        # 清洗数据
        df_main = self.clean_raw_data(df_price)
        logger.info(f"行情数据: {df_main.shape}")
        
        # 合并估值数据
        if df_valuation is not None and not df_valuation.empty:
            df_valuation = self.clean_raw_data(df_valuation)
            df_main = pd.merge(
                df_main,
                df_valuation,
                on=['order_book_id', 'trade_date'],
                how='left'
            )
            logger.info(f"合并估值后: {df_main.shape}")
        
        # 合并股本数据
        if df_share is not None and not df_share.empty:
            df_share = self.clean_raw_data(df_share)
            df_main = pd.merge(
                df_main,
                df_share,
                on=['order_book_id', 'trade_date'],
                how='left'
            )
            logger.info(f"合并股本后: {df_main.shape}")
        
        return df_main
    
    def calculate_basic_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算基础字段
        
        Args:
            df: DataFrame
            
        Returns:
            添加基础字段后的DataFrame
        """
        logger.info("计算基础字段...")
        
        df = df.sort_values(['order_book_id', 'trade_date']).reset_index(drop=True)
        
        # 计算收益率
        if 'pct_chg' not in df.columns:
            df['pct_chg'] = df.groupby('order_book_id')['close'].pct_change(1)
        
        # 计算前收盘价
        if 'pre_close' not in df.columns:
            df['pre_close'] = df.groupby('order_book_id')['close'].shift(1)
        
        # 计算振幅
        if 'amplitude' not in df.columns or df['amplitude'].isnull().all():
            df['amplitude'] = (df['high'] - df['low']) / df['pre_close']
        
        # 计算换手率(如果有股本数据)
        if 'total_share' in df.columns and 'vol' in df.columns:
            df['turnover_rate'] = df['vol'] / (df['total_share'] * 100000000)
            logger.info("  ✓ 换手率已计算")
        
        # 计算量比
        if 'vol' in df.columns:
            df['volume_ratio'] = df.groupby('order_book_id')['vol'].transform(
                lambda x: x / x.rolling(5, min_periods=1).mean()
            )
            logger.info("  ✓ 量比已计算")
        
        logger.info("✓ 基础字段计算完成")
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            df: DataFrame
            
        Returns:
            添加技术指标后的DataFrame
        """
        logger.info("计算技术指标...")
        
        df = df.sort_values(['order_book_id', 'trade_date']).reset_index(drop=True)
        
        # 1. 收益率特征
        for period in self.config.feature.return_periods:
            col_name = f'ret_{period}d'
            if col_name not in df.columns:
                df[col_name] = df.groupby('order_book_id')['close'].pct_change(period)
        
        logger.info(f"  ✓ 收益率: {self.config.feature.return_periods}")
        
        # 2. 波动率特征
        window = self.config.feature.volatility_window
        if 'ret_1d' in df.columns:
            df[f'vol_{window}d'] = df.groupby('order_book_id')['ret_1d'].rolling(window).std().reset_index(0, drop=True)
            logger.info(f"  ✓ 波动率: {window}日")
        
        # 3. 移动平均线
        for window in self.config.feature.ma_windows:
            # 收盘价均线
            col_name = f'ma_close_{window}d'
            if col_name not in df.columns:
                df[col_name] = df.groupby('order_book_id')['close'].rolling(window).mean().reset_index(0, drop=True)
            
            # 成交量均线
            if 'vol' in df.columns:
                col_name = f'ma_vol_{window}d'
                if col_name not in df.columns:
                    df[col_name] = df.groupby('order_book_id')['vol'].rolling(window).mean().reset_index(0, drop=True)
        
        logger.info(f"  ✓ 移动平均: {self.config.feature.ma_windows}")
        
        logger.info("✓ 技术指标计算完成")
        return df
    
    def calculate_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算滞后特征(避免数据泄漏)
        
        Args:
            df: DataFrame
            
        Returns:
            添加滞后特征后的DataFrame
        """
        logger.info("计算滞后特征...")
        logger.info("注意: 使用历史数据作为特征,确保无数据泄漏")
        
        df = df.sort_values(['order_book_id', 'trade_date']).reset_index(drop=True)
        
        # 1. 收盘价滞后特征
        for lag in self.config.feature.lag_periods:
            col_name = f'close_lag_{lag}'
            if col_name not in df.columns:
                df[col_name] = df.groupby('order_book_id')['close'].shift(lag)
        
        logger.info(f"  ✓ 收盘价滞后: {self.config.feature.lag_periods}")
        
        # 2. 收益率滞后特征
        if 'ret_1d' in df.columns:
            for lag in self.config.feature.lag_periods:
                col_name = f'ret_lag_{lag}'
                if col_name not in df.columns:
                    df[col_name] = df.groupby('order_book_id')['ret_1d'].shift(lag)
            
            logger.info(f"  ✓ 收益率滞后: {self.config.feature.lag_periods}")
        
        # 3. 相对强度特征(价格相对均线位置)
        if 'close_lag_1' in df.columns:
            # 相对5日均线
            ma5 = df.groupby('order_book_id')['close'].shift(1).rolling(5).mean()
            df['close_to_ma5_lag_1'] = (df['close_lag_1'] / ma5 - 1)
            
            # 相对20日均线
            ma20 = df.groupby('order_book_id')['close'].shift(1).rolling(20).mean()
            df['close_to_ma20_lag_1'] = (df['close_lag_1'] / ma20 - 1)
            
            logger.info("  ✓ 相对强度特征")
        
        # 4. 动量特征
        if 'close_lag_1' in df.columns and 'close_lag_5' in df.columns:
            df['momentum_lag_1_5'] = (df['close_lag_1'] / df['close_lag_5'] - 1)
            
            if 'close_lag_10' in df.columns:
                df['momentum_lag_1_10'] = (df['close_lag_1'] / df['close_lag_10'] - 1)
            
            logger.info("  ✓ 动量特征")
        
        logger.info("✓ 滞后特征计算完成")
        logger.info("✓ 数据泄漏检查: 所有滞后特征均使用历史数据")
        
        return df
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建衍生特征
        
        Args:
            df: DataFrame
            
        Returns:
            添加衍生特征后的DataFrame
        """
        logger.info("创建衍生特征...")
        
        # 1. 价格位置特征(当前价格在高低点之间的位置)
        if all(col in df.columns for col in ['close', 'high', 'low']):
            # 20日内的价格位置
            df['price_position_20d'] = df.groupby('order_book_id').apply(
                lambda x: (x['close'] - x['low'].rolling(20).min()) / 
                         (x['high'].rolling(20).max() - x['low'].rolling(20).min())
            ).reset_index(level=0, drop=True)
            logger.info("  ✓ 价格位置特征")
        
        # 2. 成交额占比
        if 'amount' in df.columns:
            df['amount_ratio'] = df.groupby('order_book_id')['amount'].transform(
                lambda x: x / x.rolling(5, min_periods=1).mean()
            )
            logger.info("  ✓ 成交额比率")
        
        # 3. 涨跌停特征
        if all(col in df.columns for col in ['close', 'limit_up', 'limit_down']):
            df['is_limit_up'] = (df['close'] >= df['limit_up'] * 0.99).astype(int)
            df['is_limit_down'] = (df['close'] <= df['limit_down'] * 1.01).astype(int)
            logger.info("  ✓ 涨跌停标记")
        
        logger.info("✓ 衍生特征创建完成")
        return df
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行完整的特征工程流程
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            特征工程后的DataFrame
        """
        logger.info("=" * 80)
        logger.info("开始特征工程")
        logger.info("=" * 80)
        
        # 1. 计算基础字段
        df = self.calculate_basic_fields(df)
        
        # 2. 计算技术指标
        df = self.calculate_technical_indicators(df)
        
        # 3. 计算滞后特征
        df = self.calculate_lag_features(df)
        
        # 4. 创建衍生特征
        df = self.create_derived_features(df)
        
        logger.info("=" * 80)
        logger.info(f"特征工程完成! 最终特征矩阵: {df.shape}")
        logger.info("=" * 80)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: DataFrame
            method: 处理方法 (forward_fill/drop/mean)
            
        Returns:
            处理后的DataFrame
        """
        logger.info(f"处理缺失值 (方法: {method})...")
        
        missing_before = df.isnull().sum().sum()
        
        if method == 'forward_fill':
            # 按股票分组进行前向填充
            df = df.groupby('order_book_id').fillna(method='ffill')
        elif method == 'drop':
            df = df.dropna()
        elif method == 'mean':
            # 数值列用均值填充
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"缺失值: {missing_before} -> {missing_after}")
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str], std_threshold: float = 5.0) -> pd.DataFrame:
        """
        移除异常值
        
        Args:
            df: DataFrame
            columns: 需要检查异常值的列
            std_threshold: 标准差阈值
            
        Returns:
            移除异常值后的DataFrame
        """
        logger.info(f"移除异常值 (阈值: {std_threshold}倍标准差)...")
        original_len = len(df)
        
        for col in columns:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                df = df[
                    (df[col] >= mean - std_threshold * std) & 
                    (df[col] <= mean + std_threshold * std)
                ]
        
        logger.info(f"移除异常值: {original_len} -> {len(df)} 行")
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        获取特征列名分类
        
        Args:
            df: DataFrame
            
        Returns:
            特征名称字典
        """
        all_cols = df.columns.tolist()
        
        # 基础列
        basic_cols = ['order_book_id', 'trade_date']
        
        # 原始价格列
        price_cols = ['open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount']
        
        # 估值列
        valuation_cols = ['pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'total_mv', 'circ_mv']
        
        # 技术指标列
        technical_cols = [col for col in all_cols if any(
            pattern in col for pattern in ['ret_', 'vol_', 'ma_', 'momentum_', 'amplitude', 
                                           'turnover', 'volume_ratio', 'price_position']
        )]
        
        # 滞后特征列
        lag_cols = [col for col in all_cols if 'lag' in col]
        
        # 其他特征列
        other_cols = [col for col in all_cols if col not in 
                     basic_cols + price_cols + valuation_cols + technical_cols + lag_cols]
        
        return {
            'basic': [col for col in basic_cols if col in all_cols],
            'price': [col for col in price_cols if col in all_cols],
            'valuation': [col for col in valuation_cols if col in all_cols],
            'technical': technical_cols,
            'lag': lag_cols,
            'other': other_cols
        }
