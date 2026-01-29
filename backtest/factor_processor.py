"""
因子处理器模块
对原始因子进行标准化处理：去极值、标准化、中性化等
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Any
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

from .backtest_config import BacktestConfig


class FactorProcessor:
    """因子处理器 - 实现因子标准化处理流程"""
    
    def __init__(self, config: BacktestConfig):
        """
        初始化因子处理器
        
        Args:
            config: 回测配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process(self, 
                factor_df: pd.DataFrame,
                factor_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        完整的因子处理流程
        
        Args:
            factor_df: 原始因子DataFrame
            factor_cols: 需要处理的因子列（None时自动识别）
            
        Returns:
            处理后的因子DataFrame
        """
        self.logger.info("开始因子处理流程...")
        
        df = factor_df.copy()
        
        # 自动识别因子列
        if factor_cols is None:
            factor_cols = self._auto_detect_factor_cols(df)
        
        # 确保必要的列存在
        self._validate_dataframe(df)
        
        for factor_col in factor_cols:
            self.logger.info(f"处理因子: {factor_col}")
            
            # 1. 去极值
            df[f'{factor_col}_winsorized'] = self.winsorize(
                df, factor_col, method=self.config.winsorize_method
            )
            
            # 2. 缺失值填充
            df[f'{factor_col}_filled'] = self.fill_missing(
                df, f'{factor_col}_winsorized'
            )
            
            # 3. 标准化
            df[f'{factor_col}_std'] = self.standardize(
                df, f'{factor_col}_filled', method=self.config.standardize_method
            )
            
            # 4. 中性化（可选）
            if self.config.industry_neutral or self.config.market_value_neutral:
                df[f'{factor_col}_neutral'] = self.neutralize(
                    df, f'{factor_col}_std'
                )
        
        self.logger.info("因子处理完成")
        return df
    
    def winsorize(self, 
                  df: pd.DataFrame, 
                  factor_col: str,
                  method: str = 'quantile') -> pd.Series:
        """
        去极值处理
        
        Args:
            df: DataFrame
            factor_col: 因子列名
            method: 去极值方法 ('quantile', 'mad', 'std')
            
        Returns:
            处理后的Series
        """
        result = []
        
        # 按日期截面处理
        for date, group in df.groupby('trade_date'):
            values = group[factor_col].copy()
            
            if method == 'quantile':
                # 分位数法
                lower = values.quantile(self.config.winsorize_quantiles[0])
                upper = values.quantile(self.config.winsorize_quantiles[1])
                values = values.clip(lower, upper)
            
            elif method == 'mad':
                # MAD法（中位数绝对偏差）
                median = values.median()
                mad = np.median(np.abs(values - median))
                lower = median - self.config.mad_threshold * mad
                upper = median + self.config.mad_threshold * mad
                values = values.clip(lower, upper)
            
            elif method == 'std':
                # 标准差法
                mean = values.mean()
                std = values.std()
                lower = mean - self.config.std_threshold * std
                upper = mean + self.config.std_threshold * std
                values = values.clip(lower, upper)
            
            else:
                raise ValueError(f"未知的去极值方法: {method}")
            
            result.append(values)
        
        return pd.concat(result)
    
    def fill_missing(self,
                     df: pd.DataFrame,
                     factor_col: str) -> pd.Series:
        """
        缺失值填充
        
        Args:
            df: DataFrame
            factor_col: 因子列名
            
        Returns:
            填充后的Series
        """
        result = []
        
        for date, group in df.groupby('trade_date'):
            values = group[factor_col].copy()
            
            if self.config.fillna_method == 'mean':
                values = values.fillna(values.mean())
            elif self.config.fillna_method == 'median':
                values = values.fillna(values.median())
            elif self.config.fillna_method == 'forward':
                values = values.fillna(method='ffill')
            elif self.config.fillna_method == 'zero':
                values = values.fillna(0)
            else:
                raise ValueError(f"未知的填充方法: {self.config.fillna_method}")
            
            # 如果还有缺失值，填充为0
            values = values.fillna(0)
            
            result.append(values)
        
        return pd.concat(result)
    
    def standardize(self,
                    df: pd.DataFrame,
                    factor_col: str,
                    method: str = 'zscore') -> pd.Series:
        """
        标准化处理（按日期截面）
        
        Args:
            df: DataFrame
            factor_col: 因子列名
            method: 标准化方法 ('zscore', 'minmax', 'rank')
            
        Returns:
            标准化后的Series
        """
        result = []
        
        for date, group in df.groupby('trade_date'):
            values = group[factor_col].copy()
            
            if method == 'zscore':
                # Z-score标准化
                mean = values.mean()
                std = values.std()
                if std > 0:
                    values = (values - mean) / std
                else:
                    values = values - mean
            
            elif method == 'minmax':
                # MinMax标准化到[-1, 1]
                min_val = values.min()
                max_val = values.max()
                if max_val > min_val:
                    values = 2 * (values - min_val) / (max_val - min_val) - 1
                else:
                    values = values * 0
            
            elif method == 'rank':
                # 排序标准化到[-1, 1]
                values = values.rank(pct=True) * 2 - 1
            
            else:
                raise ValueError(f"未知的标准化方法: {method}")
            
            result.append(values)
        
        return pd.concat(result)
    
    def neutralize(self,
                   df: pd.DataFrame,
                   factor_col: str) -> pd.Series:
        """
        中性化处理（行业和市值）
        
        Args:
            df: DataFrame
            factor_col: 因子列名
            
        Returns:
            中性化后的Series
        """
        result = []
        
        for date, group in df.groupby('trade_date'):
            values = group[factor_col].copy().values
            
            # 构建回归变量
            X_list = []
            
            # 行业中性化
            if self.config.industry_neutral and self.config.industry_col in group.columns:
                # 行业哑变量
                industry_dummies = pd.get_dummies(
                    group[self.config.industry_col], 
                    prefix='industry',
                    drop_first=True
                )
                X_list.append(industry_dummies.values)
            
            # 市值中性化
            if self.config.market_value_neutral and self.config.market_value_col in group.columns:
                # 对数市值
                market_value = np.log(group[self.config.market_value_col].values + 1)
                X_list.append(market_value.reshape(-1, 1))
            
            # 如果有回归变量，进行回归取残差
            if len(X_list) > 0:
                X = np.hstack(X_list)
                
                # 添加截距项
                X = np.column_stack([np.ones(len(X)), X])
                
                # 线性回归
                try:
                    # 使用最小二乘法
                    beta = np.linalg.lstsq(X, values, rcond=None)[0]
                    fitted = X @ beta
                    residuals = values - fitted
                    values = residuals
                except np.linalg.LinAlgError:
                    self.logger.warning(f"日期 {date} 中性化失败，跳过")
            
            result.append(pd.Series(values, index=group.index))
        
        return pd.concat(result)
    
    def _auto_detect_factor_cols(self, df: pd.DataFrame, prefixes: Optional[List[str]] = None) -> List[str]:
        """
        自动识别因子列
        
        Args:
            df: DataFrame
            prefixes: 允许的列名前缀列表，默认 ['factor_', 'pred_', 'latent_']
        
        Returns:
            识别到的因子列列表
        """
        if prefixes is None:
            prefixes = ['factor_', 'pred_', 'latent_']
        
        exclude_suffixes = ['_winsorized', '_filled', '_std', '_neutral']
        
        factor_cols = [
            col for col in df.columns 
            if any(col.startswith(prefix) for prefix in prefixes)
            and not any(suffix in col for suffix in exclude_suffixes)
        ]
        
        if len(factor_cols) == 0:
            raise ValueError(f"未找到因子列，因子列应以 {prefixes} 之一开头")
        
        self.logger.info(f"识别到 {len(factor_cols)} 个因子列: {factor_cols}")
        return factor_cols
    
    def _validate_dataframe(self, df: pd.DataFrame):
        """验证DataFrame格式"""
        if 'trade_date' not in df.columns:
            raise ValueError("DataFrame必须包含trade_date列")
        
        if self.config.industry_neutral and self.config.industry_col not in df.columns:
            self.logger.warning(f"未找到行业列 {self.config.industry_col}，将跳过行业中性化")
        
        if self.config.market_value_neutral and self.config.market_value_col not in df.columns:
            self.logger.warning(f"未找到市值列 {self.config.market_value_col}，将跳过市值中性化")
    
    def get_processing_stats(self, 
                            original_df: pd.DataFrame,
                            processed_df: pd.DataFrame,
                            factor_col: str) -> Dict[str, Any]:
        """
        获取处理前后的统计信息
        
        Args:
            original_df: 原始DataFrame
            processed_df: 处理后DataFrame
            factor_col: 因子列名
            
        Returns:
            统计信息字典
        """
        stat_dict = {}
        
        # 原始因子统计
        original_values = original_df[factor_col].values
        stat_dict['original'] = {
            'mean': np.mean(original_values),
            'std': np.std(original_values),
            'min': np.min(original_values),
            'max': np.max(original_values),
            'skew': stats.skew(original_values),
            'kurt': stats.kurtosis(original_values)
        }
        
        # 处理后统计
        processed_col = f'{factor_col}_std'
        if processed_col in processed_df.columns:
            processed_values = processed_df[processed_col].values
            stat_dict['processed'] = {
                'mean': np.mean(processed_values),
                'std': np.std(processed_values),
                'min': np.min(processed_values),
                'max': np.max(processed_values),
                'skew': stats.skew(processed_values),
                'kurt': stats.kurtosis(processed_values)
            }
        
        return stat_dict
    
    def cross_sectional_rank(self, 
                            df: pd.DataFrame,
                            factor_col: str) -> pd.Series:
        """
        截面排序（返回百分位排名）
        
        Args:
            df: DataFrame
            factor_col: 因子列名
            
        Returns:
            排名Series（0-1之间）
        """
        result = []
        
        for date, group in df.groupby('trade_date'):
            rank = group[factor_col].rank(pct=True)
            result.append(rank)
        
        return pd.concat(result)
    
    def rolling_standardize(self,
                           df: pd.DataFrame,
                           factor_col: str,
                           window: int = 252) -> pd.Series:
        """
        滚动标准化（时间序列维度）
        
        Args:
            df: DataFrame
            factor_col: 因子列名
            window: 滚动窗口大小
            
        Returns:
            标准化后的Series
        """
        result = []
        
        for ts_code, group in df.groupby('ts_code'):
            group = group.sort_values('trade_date')
            values = group[factor_col]
            
            # 计算滚动均值和标准差
            rolling_mean = values.rolling(window=window, min_periods=1).mean()
            rolling_std = values.rolling(window=window, min_periods=1).std()
            
            # 标准化
            standardized = (values - rolling_mean) / rolling_std.replace(0, 1)
            
            result.append(standardized)
        
        return pd.concat(result)
