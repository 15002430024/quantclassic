"""
绩效评估器模块
计算组合收益和风险指标
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
import logging

from .backtest_config import BacktestConfig


class PerformanceEvaluator:
    """绩效评估器 - 计算组合绩效指标"""
    
    def __init__(self, config: BacktestConfig):
        """
        初始化绩效评估器
        
        Args:
            config: 回测配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def evaluate_portfolio(self,
                          portfolio_df: pd.DataFrame,
                          return_col: str = 'portfolio_return',
                          benchmark_col: Optional[str] = None) -> Dict[str, float]:
        """
        评估组合绩效
        
        Args:
            portfolio_df: 组合DataFrame
            return_col: 收益列名
            benchmark_col: 基准收益列名
            
        Returns:
            绩效指标字典
        """
        self.logger.info("计算绩效指标...")
        
        returns = portfolio_df[return_col].values
        
        # 基础收益指标
        metrics = self._calculate_return_metrics(returns)
        
        # 风险指标
        risk_metrics = self._calculate_risk_metrics(returns)
        metrics.update(risk_metrics)
        
        # 风险调整收益
        risk_adj_metrics = self._calculate_risk_adjusted_returns(returns)
        metrics.update(risk_adj_metrics)
        
        # 如果有基准，计算相对指标
        if benchmark_col and benchmark_col in portfolio_df.columns:
            benchmark_returns = portfolio_df[benchmark_col].values
            relative_metrics = self._calculate_relative_metrics(returns, benchmark_returns)
            metrics.update(relative_metrics)
        
        # 统计指标
        stat_metrics = self._calculate_statistical_metrics(returns)
        metrics.update(stat_metrics)
        
        self.logger.info("绩效指标计算完成")
        
        return metrics
    
    def _calculate_return_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """计算收益指标"""
        cum_returns = (1 + returns).cumprod() - 1
        
        return {
            # 累计收益
            'total_return': cum_returns[-1] if len(cum_returns) > 0 else 0,
            
            # 年化收益
            'annual_return': self._annualize_return(returns),
            
            # 平均日收益
            'mean_return': np.mean(returns),
            
            # 日收益中位数
            'median_return': np.median(returns),
        }
    
    def _calculate_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """计算风险指标"""
        cum_returns = (1 + returns).cumprod()
        
        # 最大回撤
        max_dd, max_dd_duration = self._calculate_max_drawdown(cum_returns)
        
        # 下行风险
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        return {
            # 波动率
            'volatility': np.std(returns),
            
            # 年化波动率
            'annual_volatility': np.std(returns) * np.sqrt(self.config.annual_factor),
            
            # 最大回撤
            'max_drawdown': max_dd,
            
            # 最大回撤持续期
            'max_drawdown_duration': max_dd_duration,
            
            # 下行风险
            'downside_risk': downside_std,
            
            # 年化下行风险
            'annual_downside_risk': downside_std * np.sqrt(self.config.annual_factor),
        }
    
    def _calculate_risk_adjusted_returns(self, returns: np.ndarray) -> Dict[str, float]:
        """计算风险调整收益"""
        annual_return = self._annualize_return(returns)
        annual_vol = np.std(returns) * np.sqrt(self.config.annual_factor)
        
        # 夏普比率
        sharpe = (annual_return - self.config.risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        # 卡玛比率
        cum_returns = (1 + returns).cumprod()
        max_dd, _ = self._calculate_max_drawdown(cum_returns)
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        # 索提诺比率
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        annual_downside = downside_std * np.sqrt(self.config.annual_factor)
        sortino = (annual_return - self.config.risk_free_rate) / annual_downside if annual_downside > 0 else 0
        
        return {
            'sharpe_ratio': sharpe,
            'calmar_ratio': calmar,
            'sortino_ratio': sortino,
        }
    
    def _calculate_relative_metrics(self, 
                                    returns: np.ndarray,
                                    benchmark_returns: np.ndarray) -> Dict[str, float]:
        """计算相对指标"""
        # 超额收益
        excess_returns = returns - benchmark_returns
        
        # 跟踪误差
        tracking_error = np.std(excess_returns) * np.sqrt(self.config.annual_factor)
        
        # 信息比率
        annual_excess = self._annualize_return(excess_returns)
        information_ratio = annual_excess / tracking_error if tracking_error > 0 else 0
        
        # Beta
        if np.std(benchmark_returns) > 0:
            beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        else:
            beta = 0
        
        # Alpha (CAPM)
        annual_return = self._annualize_return(returns)
        annual_benchmark = self._annualize_return(benchmark_returns)
        alpha = annual_return - (self.config.risk_free_rate + beta * (annual_benchmark - self.config.risk_free_rate))
        
        return {
            'excess_return': self._annualize_return(excess_returns),
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha,
        }
    
    def _calculate_statistical_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """计算统计指标"""
        from scipy import stats
        
        return {
            # 胜率
            'win_rate': np.mean(returns > 0),
            
            # 盈亏比
            'profit_loss_ratio': self._calculate_profit_loss_ratio(returns),
            
            # 偏度
            'skewness': stats.skew(returns),
            
            # 峰度
            'kurtosis': stats.kurtosis(returns),
            
            # VaR (95%)
            'var_95': np.percentile(returns, 5),
            
            # CVaR (95%)
            'cvar_95': np.mean(returns[returns <= np.percentile(returns, 5)]),
        }
    
    def _annualize_return(self, returns: np.ndarray) -> float:
        """年化收益率"""
        if len(returns) == 0:
            return 0
        
        cum_return = (1 + returns).prod()
        n_periods = len(returns)
        annual_return = cum_return ** (self.config.annual_factor / n_periods) - 1
        
        return annual_return
    
    def _calculate_max_drawdown(self, cum_returns: np.ndarray) -> Tuple[float, int]:
        """
        计算最大回撤
        
        Returns:
            (最大回撤, 最大回撤持续期)
        """
        if len(cum_returns) == 0:
            return 0, 0
        
        # 计算累计最大值
        running_max = np.maximum.accumulate(cum_returns)
        
        # 计算回撤
        drawdown = (cum_returns - running_max) / running_max
        
        # 最大回撤
        max_dd = np.min(drawdown)
        
        # 最大回撤持续期
        max_dd_duration = 0
        current_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                max_dd_duration = max(max_dd_duration, current_duration)
            else:
                current_duration = 0
        
        return max_dd, max_dd_duration
    
    def _calculate_profit_loss_ratio(self, returns: np.ndarray) -> float:
        """计算盈亏比"""
        profits = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(profits) == 0 or len(losses) == 0:
            return 0
        
        avg_profit = np.mean(profits)
        avg_loss = np.mean(np.abs(losses))
        
        return avg_profit / avg_loss if avg_loss > 0 else 0
    
    def rolling_performance(self,
                           portfolio_df: pd.DataFrame,
                           return_col: str = 'portfolio_return',
                           window: int = 60) -> pd.DataFrame:
        """
        滚动绩效分析
        
        Args:
            portfolio_df: 组合DataFrame
            return_col: 收益列名
            window: 滚动窗口
            
        Returns:
            滚动绩效DataFrame
        """
        df = portfolio_df.copy()
        returns = df[return_col].values
        
        # 滚动年化收益
        df['rolling_annual_return'] = pd.Series(returns).rolling(window).apply(
            lambda x: self._annualize_return(x.values)
        )
        
        # 滚动波动率
        df['rolling_volatility'] = pd.Series(returns).rolling(window).std() * np.sqrt(self.config.annual_factor)
        
        # 滚动夏普比率
        df['rolling_sharpe'] = (df['rolling_annual_return'] - self.config.risk_free_rate) / df['rolling_volatility']
        
        # 滚动最大回撤
        def rolling_max_dd(x):
            cum = (1 + x).cumprod()
            running_max = np.maximum.accumulate(cum)
            dd = (cum - running_max) / running_max
            return np.min(dd)
        
        df['rolling_max_drawdown'] = pd.Series(returns).rolling(window).apply(rolling_max_dd)
        
        return df
    
    def monthly_performance(self, portfolio_df: pd.DataFrame,
                           return_col: str = 'portfolio_return') -> pd.DataFrame:
        """
        月度绩效分析
        
        Args:
            portfolio_df: 组合DataFrame
            return_col: 收益列名
            
        Returns:
            月度绩效DataFrame
        """
        df = portfolio_df.copy()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df['year_month'] = df['trade_date'].dt.to_period('M')
        
        # 按月聚合
        monthly = df.groupby('year_month').agg({
            return_col: ['sum', 'std', 'count']
        }).reset_index()
        
        monthly.columns = ['year_month', 'monthly_return', 'monthly_std', 'n_days']
        
        # 月度胜率
        monthly['win'] = monthly['monthly_return'] > 0
        
        return monthly
    
    def yearly_performance(self, portfolio_df: pd.DataFrame,
                          return_col: str = 'portfolio_return') -> pd.DataFrame:
        """
        年度绩效分析
        
        Args:
            portfolio_df: 组合DataFrame
            return_col: 收益列名
            
        Returns:
            年度绩效DataFrame
        """
        df = portfolio_df.copy()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df['year'] = df['trade_date'].dt.year
        
        yearly_stats = []
        
        for year, year_data in df.groupby('year'):
            returns = year_data[return_col].values
            
            metrics = {
                'year': year,
                'total_return': (1 + returns).prod() - 1,
                'volatility': np.std(returns) * np.sqrt(self.config.annual_factor),
                'max_drawdown': self._calculate_max_drawdown((1 + returns).cumprod())[0],
                'sharpe_ratio': self._calculate_risk_adjusted_returns(returns)['sharpe_ratio'],
                'win_rate': np.mean(returns > 0),
                'n_days': len(returns)
            }
            
            yearly_stats.append(metrics)
        
        return pd.DataFrame(yearly_stats)
