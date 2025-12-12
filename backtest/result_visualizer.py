"""
结果可视化器模块
生成各种分析图表（matplotlib版本，增强基准对比）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict
import logging
import os
import platform

from .backtest_config import BacktestConfig
from .benchmark_manager import BenchmarkManager


class ResultVisualizer:
    """结果可视化器 - 生成专业图表"""
    
    def __init__(self, config: BacktestConfig):
        """
        初始化可视化器
        
        Args:
            config: 回测配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.benchmark_manager = BenchmarkManager()
        
        # 设置绘图样式
        if self.config.plot_style == 'seaborn':
            sns.set_style("whitegrid")
            try:
                plt.style.use('seaborn-v0_8-darkgrid')
            except:
                plt.style.use('seaborn-darkgrid')
        elif self.config.plot_style == 'ggplot':
            plt.style.use('ggplot')
        
        # 跨平台中文字体配置
        self._setup_chinese_font()
        
        # 专业配色方案
        self.colors = {
            'strategy': '#2E86DE',
            'benchmark': '#EE5A6F',
            'long': '#10AC84',
            'short': '#EE5A6F',
            'ic': '#5f27cd',
            'neutral': '#95a5a6'
        }
    
    def _setup_chinese_font(self):
        """配置跨平台中文字体"""
        os_name = platform.system()
        
        if os_name == 'Darwin':  # macOS
            plt.rcParams['font.sans-serif'] = [
                'Arial Unicode MS', 'PingFang SC', 'STHeiti', 
                'Songti SC', 'DejaVu Sans'
            ]
            self.logger.debug("使用 macOS 中文字体")
        
        elif os_name == 'Linux':
            # Linux: 尝试使用文泉驿或 Noto 字体
            plt.rcParams['font.sans-serif'] = [
                'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 
                'Noto Sans CJK SC', 'Noto Sans CJK TC',
                'DejaVu Sans'
            ]
            self.logger.debug("使用 Linux 中文字体")
        
        elif os_name == 'Windows':
            plt.rcParams['font.sans-serif'] = [
                'Microsoft YaHei', 'SimHei', 'SimSun', 
                'KaiTi', 'DejaVu Sans'
            ]
            self.logger.debug("使用 Windows 中文字体")
        
        else:
            # 其他系统，使用默认字体
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            self.logger.warning(f"未知操作系统 {os_name}，使用默认字体")
        
        # 用来正常显示负号
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_cumulative_returns(self,
                               portfolio_df: pd.DataFrame,
                               return_col: str = 'portfolio_return',
                               benchmark_col: Optional[str] = None,
                               benchmark_name: Optional[str] = None,
                               title: str = '累计收益曲线',
                               save_path: Optional[str] = None):
        """
        绘制累计收益曲线（增强版，支持自动获取基准）
        
        Args:
            portfolio_df: 组合DataFrame
            return_col: 收益列名
            benchmark_col: 基准收益列名（如果已在DataFrame中）
            benchmark_name: 基准名称（如 'hs300', 'zz800'，用于自动获取）
            title: 图表标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # 确保有日期列
        df = portfolio_df.copy()
        if 'trade_date' not in df.columns:
            df['trade_date'] = df.index
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 计算策略累计收益
        cum_returns = (1 + df[return_col]).cumprod()
        
        # 绘制策略收益
        ax.plot(df['trade_date'], cum_returns, label='策略收益', 
               linewidth=2.5, color=self.colors['strategy'], alpha=0.9)
        
        # 获取或使用基准收益
        benchmark_cumret = None
        if benchmark_col and benchmark_col in df.columns:
            # 使用已有的基准列
            benchmark_cumret = (1 + df[benchmark_col]).cumprod()
            benchmark_label = '基准收益'
        elif benchmark_name:
            # 自动获取基准数据
            try:
                start_date = df['trade_date'].min().strftime('%Y-%m-%d')
                end_date = df['trade_date'].max().strftime('%Y-%m-%d')
                
                benchmark_returns = self.benchmark_manager.get_benchmark_returns(
                    benchmark_name,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # 对齐日期
                benchmark_df = pd.DataFrame({
                    'trade_date': benchmark_returns.index,
                    'benchmark_return': benchmark_returns.values
                })
                benchmark_df['trade_date'] = pd.to_datetime(benchmark_df['trade_date'])
                
                df = pd.merge(df, benchmark_df, on='trade_date', how='left')
                df['benchmark_return'] = df['benchmark_return'].fillna(0)
                benchmark_cumret = (1 + df['benchmark_return']).cumprod()
                benchmark_label = f'基准 ({benchmark_name.upper()})'
                
                self.logger.info(f"成功加载基准数据: {benchmark_name}")
            except Exception as e:
                self.logger.warning(f"获取基准数据失败: {e}")
                benchmark_cumret = None
        
        # 绘制基准收益
        if benchmark_cumret is not None:
            ax.plot(df['trade_date'], benchmark_cumret, label=benchmark_label, 
                   linewidth=2, linestyle='--', color=self.colors['benchmark'], alpha=0.8)
            
            # 添加超额收益填充区域
            ax.fill_between(df['trade_date'], cum_returns, benchmark_cumret,
                           where=(cum_returns >= benchmark_cumret),
                           alpha=0.2, color=self.colors['long'], label='正超额')
            ax.fill_between(df['trade_date'], cum_returns, benchmark_cumret,
                           where=(cum_returns < benchmark_cumret),
                           alpha=0.2, color=self.colors['short'], label='负超额')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel('累计净值', fontsize=12)
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=1, color=self.colors['neutral'], linestyle=':', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            self.logger.info(f"图表已保存: {save_path}")
        
        plt.close()
    
    def plot_drawdown(self,
                     portfolio_df: pd.DataFrame,
                     return_col: str = 'portfolio_return',
                     title: str = '回撤曲线',
                     save_path: Optional[str] = None):
        """绘制回撤曲线"""
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # 计算累计净值
        cum_returns = (1 + portfolio_df[return_col]).cumprod()
        
        # 计算回撤
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        
        ax.fill_between(portfolio_df.index, drawdown, 0, alpha=0.3, color='red')
        ax.plot(portfolio_df.index, drawdown, color='red', linewidth=1.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel('回撤', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 标注最大回撤
        max_dd_idx = np.argmin(drawdown)
        max_dd = drawdown[max_dd_idx]
        ax.annotate(f'最大回撤: {max_dd:.2%}', 
                   xy=(max_dd_idx, max_dd),
                   xytext=(max_dd_idx, max_dd * 0.5),
                   arrowprops=dict(arrowstyle='->', color='black'),
                   fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.close()
    
    def plot_ic_series(self,
                      ic_df: pd.DataFrame,
                      title: str = 'IC时间序列',
                      save_path: Optional[str] = None):
        """绘制IC时间序列"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=self.config.dpi)
        
        # IC时间序列
        ax1.plot(ic_df.index, ic_df['ic'], label='IC', linewidth=1, alpha=0.7)
        ax1.plot(ic_df.index, ic_df['rank_ic'], label='Rank IC', linewidth=1, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax1.set_title('IC时间序列', fontsize=12, fontweight='bold')
        ax1.set_ylabel('IC', fontsize=10)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 累计IC
        ax2.plot(ic_df.index, ic_df['cum_ic'], label='累计IC', linewidth=2)
        ax2.plot(ic_df.index, ic_df['cum_rank_ic'], label='累计Rank IC', linewidth=2)
        ax2.set_title('累计IC', fontsize=12, fontweight='bold')
        ax2.set_xlabel('时间', fontsize=10)
        ax2.set_ylabel('累计IC', fontsize=10)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.close()
    
    def plot_ic_distribution(self,
                            ic_df: pd.DataFrame,
                            title: str = 'IC分布',
                            save_path: Optional[str] = None):
        """绘制IC分布直方图"""
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        ax.hist(ic_df['ic'], bins=50, alpha=0.7, label='IC', edgecolor='black')
        ax.axvline(x=ic_df['ic'].mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'均值: {ic_df["ic"].mean():.4f}')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('IC', fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.close()
    
    def plot_group_returns(self,
                          group_df: pd.DataFrame,
                          title: str = '分组收益',
                          save_path: Optional[str] = None):
        """绘制分组收益柱状图"""
        # 计算每组的平均收益
        avg_returns = group_df.groupby('group')['return_mean'].mean().sort_index()
        
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        bars = ax.bar(avg_returns.index, avg_returns.values, 
                     color=['red' if x < 0 else 'green' for x in avg_returns.values],
                     alpha=0.7, edgecolor='black')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('分组', fontsize=12)
        ax.set_ylabel('平均收益率', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.close()
    
    def plot_long_short_performance(self,
                                   ls_df: pd.DataFrame,
                                   title: str = '多空组合表现',
                                   save_path: Optional[str] = None):
        """绘制多空组合表现"""
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        ax.plot(ls_df.index, ls_df['cum_return_long'], 
               label='多头组合', linewidth=2, color='green')
        ax.plot(ls_df.index, ls_df['cum_return_short'], 
               label='空头组合', linewidth=2, color='red')
        ax.plot(ls_df.index, ls_df['cum_return'], 
               label='多空组合', linewidth=2, color='blue', linestyle='--')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel('累计收益率', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.close()
    
    def plot_monthly_returns_heatmap(self,
                                    portfolio_df: pd.DataFrame,
                                    return_col: str = 'portfolio_return',
                                    title: str = '月度收益热力图',
                                    save_path: Optional[str] = None):
        """绘制月度收益热力图"""
        df = portfolio_df.copy()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df['year'] = df['trade_date'].dt.year
        df['month'] = df['trade_date'].dt.month
        
        # 计算月度收益
        monthly = df.groupby(['year', 'month'])[return_col].sum().unstack()
        
        fig, ax = plt.subplots(figsize=(14, 8), dpi=self.config.dpi)
        
        sns.heatmap(monthly, annot=True, fmt='.2%', cmap='RdYlGn', 
                   center=0, ax=ax, cbar_kws={'label': '月度收益率'})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('月份', fontsize=12)
        ax.set_ylabel('年份', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.close()
    
    def plot_excess_returns(self,
                           portfolio_df: pd.DataFrame,
                           return_col: str = 'portfolio_return',
                           benchmark_name: str = 'zz800',
                           title: str = '超额收益分析',
                           save_path: Optional[str] = None):
        """绘制超额收益曲线"""
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        df = portfolio_df.copy()
        if 'trade_date' not in df.columns:
            df['trade_date'] = df.index
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        try:
            # 获取基准收益
            start_date = df['trade_date'].min().strftime('%Y-%m-%d')
            end_date = df['trade_date'].max().strftime('%Y-%m-%d')
            
            benchmark_returns = self.benchmark_manager.get_benchmark_returns(
                benchmark_name, start_date=start_date, end_date=end_date
            )
            
            benchmark_df = pd.DataFrame({
                'trade_date': benchmark_returns.index,
                'benchmark_return': benchmark_returns.values
            })
            benchmark_df['trade_date'] = pd.to_datetime(benchmark_df['trade_date'])
            
            df = pd.merge(df, benchmark_df, on='trade_date', how='left')
            df['benchmark_return'] = df['benchmark_return'].fillna(0)
            
            # 计算超额收益
            df['excess_return'] = df[return_col] - df['benchmark_return']
            df['excess_cumret'] = (1 + df['excess_return']).cumprod()
            
            # 绘制
            ax.plot(df['trade_date'], df['excess_cumret'], 
                   linewidth=2.5, color=self.colors['long'], label='累计超额净值')
            ax.fill_between(df['trade_date'], df['excess_cumret'], 1,
                           where=(df['excess_cumret'] >= 1),
                           alpha=0.3, color=self.colors['long'])
            ax.fill_between(df['trade_date'], df['excess_cumret'], 1,
                           where=(df['excess_cumret'] < 1),
                           alpha=0.3, color=self.colors['short'])
            
            ax.axhline(y=1, color=self.colors['neutral'], linestyle='--', linewidth=1, label='基准线')
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('时间', fontsize=12)
            ax.set_ylabel('累计超额净值', fontsize=12)
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
                self.logger.info(f"图表已保存: {save_path}")
        
        except Exception as e:
            self.logger.error(f"超额收益图生成失败: {e}")
        
        plt.close()
    
    def plot_drawdown_comparison(self,
                                portfolio_df: pd.DataFrame,
                                return_col: str = 'portfolio_return',
                                benchmark_name: Optional[str] = 'zz800',
                                title: str = '回撤对比分析',
                                save_path: Optional[str] = None):
        """绘制回撤对比图（策略 vs 基准）"""
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        df = portfolio_df.copy()
        if 'trade_date' not in df.columns:
            df['trade_date'] = df.index
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 计算策略回撤
        cum_returns = (1 + df[return_col]).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        
        ax.fill_between(df['trade_date'], drawdown * 100, 0, 
                       alpha=0.5, color=self.colors['strategy'], label='策略回撤')
        ax.plot(df['trade_date'], drawdown * 100, 
               color=self.colors['strategy'], linewidth=2)
        
        # 获取基准回撤
        if benchmark_name:
            try:
                start_date = df['trade_date'].min().strftime('%Y-%m-%d')
                end_date = df['trade_date'].max().strftime('%Y-%m-%d')
                
                benchmark_returns = self.benchmark_manager.get_benchmark_returns(
                    benchmark_name, start_date=start_date, end_date=end_date
                )
                
                benchmark_df = pd.DataFrame({
                    'trade_date': benchmark_returns.index,
                    'benchmark_return': benchmark_returns.values
                })
                benchmark_df['trade_date'] = pd.to_datetime(benchmark_df['trade_date'])
                
                df = pd.merge(df, benchmark_df, on='trade_date', how='left')
                df['benchmark_return'] = df['benchmark_return'].fillna(0)
                
                bench_cum = (1 + df['benchmark_return']).cumprod()
                bench_max = np.maximum.accumulate(bench_cum)
                bench_dd = (bench_cum - bench_max) / bench_max
                
                ax.fill_between(df['trade_date'], bench_dd * 100, 0,
                               alpha=0.3, color=self.colors['benchmark'], 
                               label=f'基准回撤 ({benchmark_name.upper()})')
                ax.plot(df['trade_date'], bench_dd * 100,
                       color=self.colors['benchmark'], linewidth=2, linestyle='--')
            except Exception as e:
                self.logger.warning(f"获取基准回撤失败: {e}")
        
        # 标注最大回撤
        max_dd_idx = np.argmin(drawdown)
        max_dd = drawdown.iloc[max_dd_idx] if hasattr(drawdown, 'iloc') else drawdown[max_dd_idx]
        max_dd_date = df['trade_date'].iloc[max_dd_idx] if hasattr(df['trade_date'], 'iloc') else df['trade_date'][max_dd_idx]
        
        ax.annotate(f'最大回撤: {max_dd*100:.2f}%',
                   xy=(max_dd_date, max_dd*100),
                   xytext=(max_dd_date, max_dd*100 * 0.5),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2),
                   fontsize=11, fontweight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel('回撤 (%)', fontsize=12)
        ax.legend(loc='lower left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.close()
    
    def create_comprehensive_report(self,
                                   portfolios: Dict,
                                   ic_df: pd.DataFrame,
                                   metrics: Dict,
                                   output_dir: str,
                                   benchmark_name: str = 'zz800'):
        """
        创建综合报告（生成所有图表，增强版）
        
        Args:
            portfolios: 组合字典
            ic_df: IC DataFrame
            metrics: 绩效指标
            output_dir: 输出目录
            benchmark_name: 基准名称
        """
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info("生成综合报告图表...")
        
        # 1. 累计收益曲线（含基准）
        if 'long_short' in portfolios:
            self.logger.info("  [1/7] 累计收益曲线")
            self.plot_cumulative_returns(
                portfolios['long_short'],
                benchmark_name=benchmark_name,
                save_path=os.path.join(output_dir, 'cumulative_returns.png')
            )
        
        # 2. 超额收益
        if 'long_short' in portfolios:
            self.logger.info("  [2/7] 超额收益分析")
            self.plot_excess_returns(
                portfolios['long_short'],
                benchmark_name=benchmark_name,
                save_path=os.path.join(output_dir, 'excess_returns.png')
            )
        
        # 3. 回撤对比
        if 'long_short' in portfolios:
            self.logger.info("  [3/7] 回撤对比")
            self.plot_drawdown_comparison(
                portfolios['long_short'],
                benchmark_name=benchmark_name,
                save_path=os.path.join(output_dir, 'drawdown_comparison.png')
            )
        
        # 4. 原始回撤曲线
        if 'long_short' in portfolios:
            self.logger.info("  [4/7] 回撤曲线")
            self.plot_drawdown(
                portfolios['long_short'],
                save_path=os.path.join(output_dir, 'drawdown.png')
            )
        
        # 5. IC时间序列
        self.logger.info("  [5/7] IC时间序列")
        self.plot_ic_series(
            ic_df,
            save_path=os.path.join(output_dir, 'ic_series.png')
        )
        
        # 6. IC分布
        self.logger.info("  [6/7] IC分布")
        self.plot_ic_distribution(
            ic_df,
            save_path=os.path.join(output_dir, 'ic_distribution.png')
        )
        
        # 7. 分组收益
        if 'groups' in portfolios:
            self.logger.info("  [7/7] 分组收益")
            self.plot_group_returns(
                portfolios['groups'],
                save_path=os.path.join(output_dir, 'group_returns.png')
            )
        
        # 8. 多空表现
        if 'long_short' in portfolios:
            self.plot_long_short_performance(
                portfolios['long_short'],
                save_path=os.path.join(output_dir, 'long_short_performance.png')
            )
        
        self.logger.info(f"✅ 报告图表已保存到: {output_dir}")
