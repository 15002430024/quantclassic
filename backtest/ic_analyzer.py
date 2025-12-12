"""
IC分析器模块
计算因子与收益的相关性指标
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Any
from scipy import stats
import logging

from .backtest_config import BacktestConfig


class ICAnalyzer:
    """IC分析器 - 分析因子与收益的关系"""
    
    def __init__(self, config: BacktestConfig):
        """
        初始化IC分析器
        
        Args:
            config: 回测配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_ic(self,
                    factor_df: pd.DataFrame,
                    factor_col: str = 'factor_raw_std',
                    return_col: str = 'y_true') -> pd.DataFrame:
        """
        计算日度IC
        
        Args:
            factor_df: 因子DataFrame
            factor_col: 因子列名
            return_col: 收益列名
            
        Returns:
            IC时间序列DataFrame
        """
        self.logger.info("计算日度IC...")
        
        ic_list = []
        
        for date, group in factor_df.groupby('trade_date'):
            # 去除缺失值
            valid_data = group[[factor_col, return_col]].dropna()
            
            if len(valid_data) < 10:  # 样本太少跳过
                continue
            
            # 强制转为一维浮点数组，避免 spearmanr/pearsonr 返回矩阵
            factor_values = np.asarray(valid_data[factor_col].values, dtype=float).ravel()
            return_values = np.asarray(valid_data[return_col].values, dtype=float).ravel()
            
            # 再次检查长度（ravel 后）
            if len(factor_values) < 10 or len(return_values) < 10:
                continue
            
            # 计算不同类型的IC
            if self.config.ic_method == 'pearson':
                result = stats.pearsonr(factor_values, return_values)
            else:  # spearman
                result = stats.spearmanr(factor_values, return_values)
            
            # 兼容处理：如果返回的是矩阵则取 [0,1] 元素
            if hasattr(result, 'statistic'):
                # scipy >= 1.9 返回 namedtuple
                ic = float(result.statistic) if np.isscalar(result.statistic) else float(result.statistic)
                p_value = float(result.pvalue) if np.isscalar(result.pvalue) else float(result.pvalue)
            else:
                ic, p_value = result[0], result[1]
            
            # 如果 ic 仍是数组（极端情况），取标量
            if hasattr(ic, '__len__') and not isinstance(ic, str):
                ic = float(np.asarray(ic).ravel()[0]) if len(np.asarray(ic).ravel()) > 0 else 0.0
            if hasattr(p_value, '__len__') and not isinstance(p_value, str):
                p_value = float(np.asarray(p_value).ravel()[0]) if len(np.asarray(p_value).ravel()) > 0 else 1.0
            
            ic = float(ic) if not np.isnan(ic) else 0.0
            p_value = float(p_value) if not np.isnan(p_value) else 1.0
            
            # Rank IC (始终计算)
            rank_result = stats.spearmanr(factor_values, return_values)
            if hasattr(rank_result, 'statistic'):
                rank_ic = float(rank_result.statistic)
                rank_p = float(rank_result.pvalue)
            else:
                rank_ic, rank_p = float(rank_result[0]), float(rank_result[1])
            
            # 同样处理 rank_ic
            if hasattr(rank_ic, '__len__') and not isinstance(rank_ic, str):
                rank_ic = float(np.asarray(rank_ic).ravel()[0]) if len(np.asarray(rank_ic).ravel()) > 0 else 0.0
            rank_ic = float(rank_ic) if not np.isnan(rank_ic) else 0.0
            
            ic_list.append({
                'trade_date': date,
                'ic': ic,
                'rank_ic': rank_ic,
                'abs_ic': abs(ic),
                'p_value': p_value,
                'n_samples': len(valid_data),
                'significant': p_value < self.config.ic_significance_level
            })
        
        ic_df = pd.DataFrame(ic_list)
        ic_df = ic_df.sort_values('trade_date').reset_index(drop=True)
        
        # 计算累计IC
        ic_df['cum_ic'] = ic_df['ic'].cumsum()
        ic_df['cum_rank_ic'] = ic_df['rank_ic'].cumsum()
        
        self.logger.info(f"IC计算完成: {len(ic_df)} 个交易日")
        
        return ic_df
    
    def calculate_multi_period_ic(self,
                                 factor_df: pd.DataFrame,
                                 factor_col: str = 'factor_raw_std',
                                 return_col: str = 'y_true') -> Dict[int, pd.DataFrame]:
        """
        计算多期IC
        
        Args:
            factor_df: 因子DataFrame
            factor_col: 因子列名
            return_col: 收益列名
            
        Returns:
            多期IC字典 {持有期: IC DataFrame}
        """
        self.logger.info("计算多期IC...")
        
        multi_period_ic = {}
        
        for period in self.config.holding_periods:
            # 计算未来N期收益
            future_returns = []
            
            for ts_code, stock_df in factor_df.groupby('ts_code'):
                stock_df = stock_df.sort_values('trade_date').reset_index(drop=True)
                
                # 计算未来N期累计收益
                stock_df[f'return_{period}d'] = stock_df[return_col].rolling(
                    window=period
                ).sum().shift(-period)
                
                future_returns.append(stock_df)
            
            # 合并
            df_with_future = pd.concat(future_returns, ignore_index=True)
            
            # 计算IC
            ic_df = self.calculate_ic(
                df_with_future, 
                factor_col, 
                f'return_{period}d'
            )
            
            multi_period_ic[period] = ic_df
            
            self.logger.info(f"  {period}日IC计算完成")
        
        return multi_period_ic
    
    def analyze_ic_statistics(self, ic_df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析IC统计指标
        
        Args:
            ic_df: IC DataFrame
            
        Returns:
            统计指标字典
        """
        # 强制转为一维浮点数组，防止混入数组元素
        ic_values = np.asarray(ic_df['ic'].tolist(), dtype=float).ravel()
        rank_ic_values = np.asarray(ic_df['rank_ic'].tolist(), dtype=float).ravel()
        
        # 过滤掉 NaN
        ic_values = ic_values[~np.isnan(ic_values)]
        rank_ic_values = rank_ic_values[~np.isnan(rank_ic_values)]
        
        # 计算标准差（防御性处理）
        ic_std = float(np.std(ic_values)) if len(ic_values) > 0 else 0.0
        rank_ic_std = float(np.std(rank_ic_values)) if len(rank_ic_values) > 0 else 0.0
        
        stats_dict = {
            # IC均值
            'ic_mean': float(np.mean(ic_values)) if len(ic_values) > 0 else 0.0,
            'rank_ic_mean': float(np.mean(rank_ic_values)) if len(rank_ic_values) > 0 else 0.0,
            
            # IC标准差
            'ic_std': ic_std,
            'rank_ic_std': rank_ic_std,
            
            # ICIR (IC均值 / IC标准差)
            'icir': float(np.mean(ic_values) / ic_std) if ic_std > 0 else 0.0,
            'rank_icir': float(np.mean(rank_ic_values) / rank_ic_std) if rank_ic_std > 0 else 0.0,
            
            # IC胜率
            'ic_win_rate': float(np.mean(ic_values > 0)) if len(ic_values) > 0 else 0.0,
            'rank_ic_win_rate': float(np.mean(rank_ic_values > 0)) if len(rank_ic_values) > 0 else 0.0,
            
            # 绝对IC均值
            'abs_ic_mean': float(np.mean(np.abs(ic_values))) if len(ic_values) > 0 else 0.0,
            'abs_rank_ic_mean': float(np.mean(np.abs(rank_ic_values))) if len(rank_ic_values) > 0 else 0.0,
            
            # IC最大值和最小值
            'ic_max': float(np.max(ic_values)) if len(ic_values) > 0 else 0.0,
            'ic_min': float(np.min(ic_values)) if len(ic_values) > 0 else 0.0,
            
            # 显著性比例
            'significant_ratio': float(ic_df['significant'].mean()) if len(ic_df) > 0 else 0.0,
            
            # t统计量
            't_stat': float(np.mean(ic_values) / (ic_std / np.sqrt(len(ic_values)))) if ic_std > 0 and len(ic_values) > 0 else 0.0,
            
            # 样本数
            'n_periods': len(ic_df)
        }
        
        return stats_dict
    
    def calculate_ic_decay(self,
                          factor_df: pd.DataFrame,
                          factor_col: str = 'factor_raw_std',
                          return_col: str = 'y_true',
                          max_period: int = 20) -> pd.DataFrame:
        """
        计算IC衰减曲线
        
        Args:
            factor_df: 因子DataFrame
            factor_col: 因子列名
            return_col: 收益列名
            max_period: 最大持有期
            
        Returns:
            IC衰减DataFrame
        """
        self.logger.info("计算IC衰减...")
        
        decay_list = []
        
        for period in range(1, max_period + 1):
            # 计算该期IC
            future_returns = []
            
            for ts_code, stock_df in factor_df.groupby('ts_code'):
                stock_df = stock_df.sort_values('trade_date').reset_index(drop=True)
                stock_df[f'return_{period}d'] = stock_df[return_col].rolling(
                    window=period
                ).sum().shift(-period)
                future_returns.append(stock_df)
            
            df_with_future = pd.concat(future_returns, ignore_index=True)
            
            # 计算整体IC
            valid_data = df_with_future[[factor_col, f'return_{period}d']].dropna()
            
            if len(valid_data) > 10:
                # 强制一维化
                f_vals = np.asarray(valid_data[factor_col].values, dtype=float).ravel()
                r_vals = np.asarray(valid_data[f'return_{period}d'].values, dtype=float).ravel()
                
                result = stats.spearmanr(f_vals, r_vals)
                if hasattr(result, 'statistic'):
                    ic = float(result.statistic)
                else:
                    ic = float(result[0])
                
                # 兼容处理
                if hasattr(ic, '__len__') and not isinstance(ic, str):
                    ic = float(np.asarray(ic).ravel()[0]) if len(np.asarray(ic).ravel()) > 0 else 0.0
                ic = float(ic) if not np.isnan(ic) else 0.0
                
                decay_list.append({
                    'period': period,
                    'ic': ic,
                    'abs_ic': abs(ic)
                })
        
        decay_df = pd.DataFrame(decay_list)
        
        self.logger.info("IC衰减计算完成")
        
        return decay_df
    
    def calculate_ic_by_group(self,
                             factor_df: pd.DataFrame,
                             factor_col: str = 'factor_raw_std',
                             return_col: str = 'y_true',
                             group_col: str = 'industry_name') -> pd.DataFrame:
        """
        分组计算IC（如按行业）
        
        Args:
            factor_df: 因子DataFrame
            factor_col: 因子列名
            return_col: 收益列名
            group_col: 分组列名
            
        Returns:
            分组IC DataFrame
        """
        if group_col not in factor_df.columns:
            self.logger.warning(f"未找到分组列 {group_col}")
            return pd.DataFrame()
        
        self.logger.info(f"按 {group_col} 分组计算IC...")
        
        group_ic_list = []
        
        for group_name, group_data in factor_df.groupby(group_col):
            # 计算该组的IC
            ic_df = self.calculate_ic(group_data, factor_col, return_col)
            
            if len(ic_df) > 0:
                stats_dict = self.analyze_ic_statistics(ic_df)
                stats_dict['group'] = group_name
                stats_dict['n_stocks'] = group_data['ts_code'].nunique()
                
                group_ic_list.append(stats_dict)
        
        group_ic_df = pd.DataFrame(group_ic_list)
        
        return group_ic_df
    
    def rolling_ic_analysis(self,
                           ic_df: pd.DataFrame,
                           window: int = 60) -> pd.DataFrame:
        """
        滚动IC分析
        
        Args:
            ic_df: IC DataFrame
            window: 滚动窗口
            
        Returns:
            滚动IC统计DataFrame
        """
        ic_df = ic_df.sort_values('trade_date').reset_index(drop=True)
        
        # 计算滚动统计
        ic_df['rolling_ic_mean'] = ic_df['ic'].rolling(window=window).mean()
        ic_df['rolling_ic_std'] = ic_df['ic'].rolling(window=window).std()
        ic_df['rolling_icir'] = ic_df['rolling_ic_mean'] / ic_df['rolling_ic_std']
        ic_df['rolling_win_rate'] = ic_df['ic'].rolling(window=window).apply(
            lambda x: np.mean(x > 0)
        )
        
        return ic_df
    
    def ic_time_series_regression(self, 
                                  ic_df: pd.DataFrame) -> Dict[str, float]:
        """
        IC时间序列回归分析（检验IC趋势）
        
        Args:
            ic_df: IC DataFrame
            
        Returns:
            回归结果字典
        """
        ic_df = ic_df.sort_values('trade_date').reset_index(drop=True)
        
        # 时间变量（标准化）
        time_idx = np.arange(len(ic_df))
        time_idx = (time_idx - time_idx.mean()) / time_idx.std()
        
        # IC值
        ic_values = ic_df['ic'].values
        
        # 线性回归
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(time_idx, ic_values)
        
        return {
            'trend_slope': slope,
            'trend_p_value': p_value,
            'trend_r_squared': r_value ** 2,
            'is_significant': p_value < 0.05,
            'direction': 'increasing' if slope > 0 else 'decreasing'
        }
    
    def monthly_ic_analysis(self, ic_df: pd.DataFrame) -> pd.DataFrame:
        """
        月度IC分析
        
        Args:
            ic_df: IC DataFrame
            
        Returns:
            月度IC统计DataFrame
        """
        ic_df = ic_df.copy()
        ic_df['trade_date'] = pd.to_datetime(ic_df['trade_date'])
        ic_df['year_month'] = ic_df['trade_date'].dt.to_period('M')
        
        monthly_stats = ic_df.groupby('year_month').agg({
            'ic': ['mean', 'std', 'count'],
            'rank_ic': 'mean',
            'significant': 'mean'
        }).reset_index()
        
        monthly_stats.columns = ['year_month', 'ic_mean', 'ic_std', 'n_days', 
                                'rank_ic_mean', 'significant_ratio']
        
        # 计算月度ICIR
        monthly_stats['monthly_icir'] = monthly_stats['ic_mean'] / monthly_stats['ic_std']
        
        return monthly_stats
