"""
组合构建器模块
构建多空投资组合
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
import logging
from datetime import datetime, timedelta

from .backtest_config import BacktestConfig


class PortfolioBuilder:
    """组合构建器 - 构建多空投资组合"""
    
    def __init__(self, config: BacktestConfig):
        """
        初始化组合构建器
        
        Args:
            config: 回测配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def build_portfolios(self,
                        factor_df: pd.DataFrame,
                        factor_col: str = 'factor_raw_std',
                        return_col: str = 'y_true') -> Dict[str, pd.DataFrame]:
        """
        构建投资组合
        
        Args:
            factor_df: 因子DataFrame
            factor_col: 因子列名
            return_col: 收益列名
            
        Returns:
            组合字典 {'long': df, 'short': df, 'long_short': df, 'groups': df}
        """
        self.logger.info("开始构建投资组合...")
        
        # 1. 分组
        group_df = self.create_factor_groups(factor_df, factor_col, return_col)
        
        # 2. 构建多头组合（Top组）
        long_df = self.create_long_portfolio(factor_df, factor_col, return_col)
        
        # 3. 构建空头组合（Bottom组）
        short_df = self.create_short_portfolio(factor_df, factor_col, return_col)
        
        # 4. 构建多空组合
        long_short_df = self.create_long_short_portfolio(long_df, short_df)
        
        self.logger.info("投资组合构建完成")
        
        return {
            'long': long_df,
            'short': short_df,
            'long_short': long_short_df,
            'groups': group_df
        }
    
    def create_factor_groups(self,
                            factor_df: pd.DataFrame,
                            factor_col: str,
                            return_col: str) -> pd.DataFrame:
        """
        创建因子分组
        
        Args:
            factor_df: 因子DataFrame
            factor_col: 因子列名
            return_col: 收益列名
            
        Returns:
            分组结果DataFrame
        """
        result = []
        
        for date, group in factor_df.groupby('trade_date'):
            # 计算分位数
            group = group.copy()
            group['factor_group'] = pd.qcut(
                group[factor_col], 
                q=self.config.n_groups, 
                labels=False,
                duplicates='drop'
            ) + 1  # 从1开始编号
            
            # 计算每组的平均收益
            group_stats = group.groupby('factor_group').agg({
                return_col: ['mean', 'std', 'count'],
                factor_col: 'mean'
            }).reset_index()
            
            group_stats.columns = ['group', 'return_mean', 'return_std', 
                                  'stock_count', 'factor_mean']
            group_stats['trade_date'] = date
            
            result.append(group_stats)
        
        group_df = pd.concat(result, ignore_index=True)
        
        self.logger.info(f"分组完成: {self.config.n_groups} 组")
        return group_df
    
    def create_long_portfolio(self,
                             factor_df: pd.DataFrame,
                             factor_col: str,
                             return_col: str) -> pd.DataFrame:
        """
        创建多头组合（做多高因子值股票）
        
        Args:
            factor_df: 因子DataFrame
            factor_col: 因子列名
            return_col: 收益列名
            
        Returns:
            多头组合DataFrame
        """
        result = []
        
        for date, group in factor_df.groupby('trade_date'):
            # 选择top股票
            n_stocks = int(len(group) * self.config.long_ratio)
            top_stocks = group.nlargest(n_stocks, factor_col)
            
            # 计算权重
            weights = self._calculate_weights(top_stocks, factor_col)
            
            # 计算组合收益
            portfolio_return = (top_stocks[return_col] * weights).sum()
            
            result.append({
                'trade_date': date,
                'portfolio_return': portfolio_return,
                'n_stocks': len(top_stocks),
                'avg_factor': top_stocks[factor_col].mean()
            })
        
        long_df = pd.DataFrame(result)
        long_df = long_df.sort_values('trade_date').reset_index(drop=True)
        
        # 计算累计收益
        long_df['cum_return'] = (1 + long_df['portfolio_return']).cumprod() - 1
        
        return long_df
    
    def create_short_portfolio(self,
                              factor_df: pd.DataFrame,
                              factor_col: str,
                              return_col: str) -> pd.DataFrame:
        """
        创建空头组合（做空低因子值股票）
        
        Args:
            factor_df: 因子DataFrame
            factor_col: 因子列名
            return_col: 收益列名
            
        Returns:
            空头组合DataFrame
        """
        result = []
        
        for date, group in factor_df.groupby('trade_date'):
            # 选择bottom股票
            n_stocks = int(len(group) * self.config.short_ratio)
            bottom_stocks = group.nsmallest(n_stocks, factor_col)
            
            # 计算权重
            weights = self._calculate_weights(bottom_stocks, factor_col)
            
            # 计算组合收益（做空，所以取负）
            portfolio_return = -(bottom_stocks[return_col] * weights).sum()
            
            result.append({
                'trade_date': date,
                'portfolio_return': portfolio_return,
                'n_stocks': len(bottom_stocks),
                'avg_factor': bottom_stocks[factor_col].mean()
            })
        
        short_df = pd.DataFrame(result)
        short_df = short_df.sort_values('trade_date').reset_index(drop=True)
        
        # 计算累计收益
        short_df['cum_return'] = (1 + short_df['portfolio_return']).cumprod() - 1
        
        return short_df
    
    def create_long_short_portfolio(self,
                                   long_df: pd.DataFrame,
                                   short_df: pd.DataFrame) -> pd.DataFrame:
        """
        创建多空组合
        
        Args:
            long_df: 多头组合DataFrame
            short_df: 空头组合DataFrame
            
        Returns:
            多空组合DataFrame
        """
        # 合并多头和空头
        ls_df = pd.merge(
            long_df[['trade_date', 'portfolio_return']], 
            short_df[['trade_date', 'portfolio_return']], 
            on='trade_date',
            suffixes=('_long', '_short')
        )
        
        # 计算多空收益
        ls_df['portfolio_return'] = ls_df['portfolio_return_long'] + ls_df['portfolio_return_short']
        
        # 计算累计收益
        ls_df['cum_return'] = (1 + ls_df['portfolio_return']).cumprod() - 1
        ls_df['cum_return_long'] = (1 + ls_df['portfolio_return_long']).cumprod() - 1
        ls_df['cum_return_short'] = (1 + ls_df['portfolio_return_short']).cumprod() - 1
        
        return ls_df
    
    def _calculate_weights(self,
                          stocks_df: pd.DataFrame,
                          factor_col: str) -> np.ndarray:
        """
        计算股票权重
        
        Args:
            stocks_df: 股票DataFrame
            factor_col: 因子列名
            
        Returns:
            权重数组
        """
        n_stocks = len(stocks_df)
        
        if self.config.weight_method == 'equal':
            # 等权
            weights = np.ones(n_stocks) / n_stocks
        
        elif self.config.weight_method == 'factor_weight':
            # 因子值加权
            factor_values = stocks_df[factor_col].values
            # 转换为正值
            factor_values = factor_values - factor_values.min() + 1e-6
            weights = factor_values / factor_values.sum()
        
        elif self.config.weight_method == 'value_weight':
            # 市值加权
            if self.config.market_value_col in stocks_df.columns:
                market_values = stocks_df[self.config.market_value_col].values
                weights = market_values / market_values.sum()
            else:
                self.logger.warning("未找到市值列，使用等权")
                weights = np.ones(n_stocks) / n_stocks
        
        else:
            weights = np.ones(n_stocks) / n_stocks
        
        # 应用权重约束
        weights = self._apply_weight_constraints(weights)
        
        return weights
    
    def _apply_weight_constraints(self, weights: np.ndarray) -> np.ndarray:
        """
        应用权重约束
        
        Args:
            weights: 原始权重
            
        Returns:
            约束后的权重
        """
        # 单股权重约束
        weights = np.clip(weights, self.config.min_stock_weight, self.config.max_stock_weight)
        
        # 重新归一化
        weights = weights / weights.sum()
        
        return weights
    
    def get_rebalance_dates(self, 
                           dates: List[str]) -> List[str]:
        """
        获取换仓日期
        
        Args:
            dates: 所有交易日期
            
        Returns:
            换仓日期列表
        """
        dates = pd.to_datetime(dates).sort_values()
        
        if self.config.rebalance_freq == 'daily':
            return dates.tolist()
        
        elif self.config.rebalance_freq == 'weekly':
            # 每周最后一个交易日
            weekly_dates = dates.to_series().groupby(
                [dates.year, dates.isocalendar().week]
            ).last()
            return weekly_dates.tolist()
        
        elif self.config.rebalance_freq == 'biweekly':
            # 双周调仓：每两周最后一个交易日
            # 使用周数除以2作为分组依据
            biweekly_dates = dates.to_series().groupby(
                [dates.year, dates.isocalendar().week // 2]
            ).last()
            return biweekly_dates.tolist()
        
        elif self.config.rebalance_freq == 'monthly':
            # 每月最后/第一个交易日
            if self.config.rebalance_day == 'last':
                monthly_dates = dates.to_series().groupby(
                    [dates.year, dates.month]
                ).last()
            else:  # first
                monthly_dates = dates.to_series().groupby(
                    [dates.year, dates.month]
                ).first()
            return monthly_dates.tolist()
        
        return dates.tolist()
    
    def calculate_turnover(self,
                          current_positions: Dict[str, float],
                          target_positions: Dict[str, float]) -> float:
        """
        计算换手率
        
        Args:
            current_positions: 当前持仓 {股票代码: 权重}
            target_positions: 目标持仓 {股票代码: 权重}
            
        Returns:
            换手率
        """
        all_stocks = set(current_positions.keys()) | set(target_positions.keys())
        
        turnover = 0
        for stock in all_stocks:
            current_weight = current_positions.get(stock, 0)
            target_weight = target_positions.get(stock, 0)
            turnover += abs(target_weight - current_weight)
        
        return turnover / 2  # 单边换手率
    
    def backtest_with_rebalance(self,
                               factor_df: pd.DataFrame,
                               factor_col: str = 'factor_raw_std',
                               return_col: str = 'y_true') -> pd.DataFrame:
        """
        考虑换仓的回测
        
        Args:
            factor_df: 因子DataFrame
            factor_col: 因子列名
            return_col: 收益列名
            
        Returns:
            回测结果DataFrame
        """
        all_dates = sorted(factor_df['trade_date'].unique())
        rebalance_dates = self.get_rebalance_dates(all_dates)
        
        results = []
        current_positions = {}
        
        for i, date in enumerate(all_dates):
            date_data = factor_df[factor_df['trade_date'] == date]
            
            # 如果是换仓日，重新构建组合
            if date in rebalance_dates:
                n_stocks = int(len(date_data) * self.config.long_ratio)
                top_stocks = date_data.nlargest(n_stocks, factor_col)
                
                # 计算目标权重
                target_positions = {}
                weights = self._calculate_weights(top_stocks, factor_col)
                for idx, (_, row) in enumerate(top_stocks.iterrows()):
                    target_positions[row['ts_code']] = weights[idx]
                
                # 计算换手率
                if len(current_positions) > 0:
                    turnover = self.calculate_turnover(current_positions, target_positions)
                else:
                    turnover = 1.0
                
                current_positions = target_positions
            else:
                turnover = 0
            
            # 计算组合收益
            portfolio_return = 0
            for ts_code, weight in current_positions.items():
                stock_data = date_data[date_data['ts_code'] == ts_code]
                if len(stock_data) > 0:
                    portfolio_return += weight * stock_data[return_col].iloc[0]
            
            # 考虑交易成本
            if self.config.consider_cost:
                transaction_cost = turnover * (
                    self.config.commission_rate + 
                    self.config.slippage_rate +
                    self.config.stamp_tax_rate
                )
                portfolio_return -= transaction_cost
            
            results.append({
                'trade_date': date,
                'portfolio_return': portfolio_return,
                'turnover': turnover,
                'n_positions': len(current_positions)
            })
        
        result_df = pd.DataFrame(results)
        result_df['cum_return'] = (1 + result_df['portfolio_return']).cumprod() - 1
        
        return result_df
