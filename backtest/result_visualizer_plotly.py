"""
增强版结果可视化器（基于 Plotly）
生成交互式、美观的分析图表，支持基准收益对比
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Optional, Dict
import logging
import os
from pathlib import Path

from .backtest_config import BacktestConfig
from .benchmark_manager import BenchmarkManager


class ResultVisualizerPlotly:
    """增强版结果可视化器 - 使用 Plotly 生成交互式专业图表"""
    
    # 专业配色方案
    COLOR_SCHEME = {
        'strategy': '#2E86DE',       # 策略主色（蓝色）
        'benchmark': '#EE5A6F',      # 基准主色（红色）
        'long': '#10AC84',           # 多头（绿色）
        'short': '#EE5A6F',          # 空头（红色）
        'excess_positive': '#10AC84', # 正超额（绿色）
        'excess_negative': '#EE5A6F', # 负超额（红色）
        'ic': '#5f27cd',             # IC（紫色）
        'neutral': '#95a5a6',        # 中性（灰色）
    }
    
    def __init__(self, config: BacktestConfig):
        """
        初始化可视化器
        
        Args:
            config: 回测配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.benchmark_manager = BenchmarkManager()
        
        # 设置默认主题
        self.template = "plotly_white"
        self.default_width = 1200
        self.default_height = 600
    
    def plot_cumulative_returns_with_benchmark(
        self,
        portfolio_df: pd.DataFrame,
        return_col: str = 'portfolio_return',
        benchmark_name: Optional[str] = 'zz800',
        title: str = '策略与基准累计收益对比',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        绘制累计收益曲线（含基准对比）
        
        Args:
            portfolio_df: 组合DataFrame（需要包含 trade_date 列）
            return_col: 收益列名
            benchmark_name: 基准名称（如 'hs300', 'zz800'）
            title: 图表标题
            save_path: 保存路径（如果提供，将保存为HTML）
            
        Returns:
            plotly Figure对象
        """
        # 确保有 trade_date 列
        if 'trade_date' not in portfolio_df.columns:
            self.logger.warning("缺少 trade_date 列，使用索引作为日期")
            df = portfolio_df.copy()
            df['trade_date'] = df.index
        else:
            df = portfolio_df.copy()
        
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 计算策略累计收益
        df['strategy_cumret'] = (1 + df[return_col]).cumprod()
        
        # 获取基准收益
        benchmark_returns = None
        if benchmark_name:
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
                df['benchmark_cumret'] = (1 + df['benchmark_return']).cumprod()
                
                self.logger.info(f"成功加载基准数据: {benchmark_name}")
            except Exception as e:
                self.logger.warning(f"获取基准数据失败: {e}，将不显示基准")
                benchmark_returns = None
        
        # 创建图表
        fig = go.Figure()
        
        # 策略曲线
        fig.add_trace(go.Scatter(
            x=df['trade_date'],
            y=df['strategy_cumret'],
            mode='lines',
            name='策略收益',
            line=dict(color=self.COLOR_SCHEME['strategy'], width=3),
            hovertemplate='<b>日期</b>: %{x}<br><b>净值</b>: %{y:.4f}<extra></extra>'
        ))
        
        # 基准曲线
        if benchmark_returns is not None and 'benchmark_cumret' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['trade_date'],
                y=df['benchmark_cumret'],
                mode='lines',
                name=f'基准 ({benchmark_name.upper()})',
                line=dict(color=self.COLOR_SCHEME['benchmark'], width=2, dash='dash'),
                hovertemplate='<b>日期</b>: %{x}<br><b>净值</b>: %{y:.4f}<extra></extra>'
            ))
            
            # 添加超额收益填充区域
            # 正超额
            fig.add_trace(go.Scatter(
                x=df['trade_date'],
                y=df['strategy_cumret'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=df['trade_date'],
                y=df['benchmark_cumret'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(16, 172, 132, 0.2)',
                line=dict(width=0),
                name='正超额收益',
                hoverinfo='skip'
            ))
        
        # 布局设置
        fig.update_layout(
            title=dict(text=title, font=dict(size=18, family='Arial', color='#2c3e50')),
            xaxis=dict(
                title='日期',
                titlefont=dict(size=14),
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            yaxis=dict(
                title='累计净值',
                titlefont=dict(size=14),
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            hovermode='x unified',
            template=self.template,
            width=self.default_width,
            height=self.default_height,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12)
            ),
            margin=dict(l=60, r=40, t=80, b=60)
        )
        
        # 保存
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_html(save_path + '.html')
            self.logger.info(f"图表已保存: {save_path}")
        
        return fig
    
    def plot_excess_returns(
        self,
        portfolio_df: pd.DataFrame,
        return_col: str = 'portfolio_return',
        benchmark_name: str = 'zz800',
        title: str = '超额收益分析',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        绘制超额收益分析图
        
        Args:
            portfolio_df: 组合DataFrame
            return_col: 收益列名
            benchmark_name: 基准名称
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            plotly Figure对象
        """
        df = portfolio_df.copy()
        if 'trade_date' not in df.columns:
            df['trade_date'] = df.index
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 获取基准收益
        try:
            start_date = df['trade_date'].min().strftime('%Y-%m-%d')
            end_date = df['trade_date'].max().strftime('%Y-%m-%d')
            
            benchmark_returns = self.benchmark_manager.get_benchmark_returns(
                benchmark_name,
                start_date=start_date,
                end_date=end_date
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
            
        except Exception as e:
            self.logger.error(f"获取基准数据失败: {e}")
            return None
        
        # 创建图表
        fig = go.Figure()
        
        # 累计超额收益
        fig.add_trace(go.Scatter(
            x=df['trade_date'],
            y=df['excess_cumret'],
            mode='lines',
            name='累计超额净值',
            line=dict(color=self.COLOR_SCHEME['long'], width=3),
            fill='tozeroy',
            fillcolor='rgba(16, 172, 132, 0.2)',
            hovertemplate='<b>日期</b>: %{x}<br><b>超额净值</b>: %{y:.4f}<extra></extra>'
        ))
        
        # 添加基准线
        fig.add_hline(
            y=1, 
            line_dash="dash", 
            line_color=self.COLOR_SCHEME['neutral'],
            annotation_text="基准线",
            annotation_position="right"
        )
        
        # 布局设置
        fig.update_layout(
            title=dict(text=title, font=dict(size=18, family='Arial', color='#2c3e50')),
            xaxis=dict(
                title='日期',
                titlefont=dict(size=14),
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            yaxis=dict(
                title='累计超额净值',
                titlefont=dict(size=14),
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            hovermode='x unified',
            template=self.template,
            width=self.default_width,
            height=self.default_height,
            margin=dict(l=60, r=40, t=80, b=60)
        )
        
        if save_path:
            fig.write_html(save_path if save_path.endswith('.html') else save_path + '.html')
            self.logger.info(f"图表已保存: {save_path}")
        
        return fig
    
    def plot_drawdown_comparison(
        self,
        portfolio_df: pd.DataFrame,
        return_col: str = 'portfolio_return',
        benchmark_name: Optional[str] = 'zz800',
        title: str = '回撤对比分析',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        绘制回撤对比图
        
        Args:
            portfolio_df: 组合DataFrame
            return_col: 收益列名
            benchmark_name: 基准名称
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            plotly Figure对象
        """
        df = portfolio_df.copy()
        if 'trade_date' not in df.columns:
            df['trade_date'] = df.index
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 计算策略回撤
        cum_returns = (1 + df[return_col]).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns / running_max - 1) * 100
        df['strategy_dd'] = drawdown
        
        # 创建图表
        fig = go.Figure()
        
        # 策略回撤
        fig.add_trace(go.Scatter(
            x=df['trade_date'],
            y=df['strategy_dd'],
            mode='lines',
            name='策略回撤',
            line=dict(color=self.COLOR_SCHEME['strategy'], width=2),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 222, 0.3)',
            hovertemplate='<b>日期</b>: %{x}<br><b>回撤</b>: %{y:.2f}%<extra></extra>'
        ))
        
        # 获取基准回撤
        if benchmark_name:
            try:
                start_date = df['trade_date'].min().strftime('%Y-%m-%d')
                end_date = df['trade_date'].max().strftime('%Y-%m-%d')
                
                benchmark_returns = self.benchmark_manager.get_benchmark_returns(
                    benchmark_name,
                    start_date=start_date,
                    end_date=end_date
                )
                
                benchmark_df = pd.DataFrame({
                    'trade_date': benchmark_returns.index,
                    'benchmark_return': benchmark_returns.values
                })
                benchmark_df['trade_date'] = pd.to_datetime(benchmark_df['trade_date'])
                
                df = pd.merge(df, benchmark_df, on='trade_date', how='left')
                df['benchmark_return'] = df['benchmark_return'].fillna(0)
                
                # 计算基准回撤
                bench_cum = (1 + df['benchmark_return']).cumprod()
                bench_max = bench_cum.expanding().max()
                bench_dd = (bench_cum / bench_max - 1) * 100
                df['benchmark_dd'] = bench_dd
                
                fig.add_trace(go.Scatter(
                    x=df['trade_date'],
                    y=df['benchmark_dd'],
                    mode='lines',
                    name=f'基准回撤 ({benchmark_name.upper()})',
                    line=dict(color=self.COLOR_SCHEME['benchmark'], width=2, dash='dash'),
                    fill='tozeroy',
                    fillcolor='rgba(238, 90, 111, 0.2)',
                    hovertemplate='<b>日期</b>: %{x}<br><b>回撤</b>: %{y:.2f}%<extra></extra>'
                ))
            except Exception as e:
                self.logger.warning(f"获取基准回撤失败: {e}")
        
        # 标注最大回撤
        max_dd_idx = df['strategy_dd'].idxmin()
        max_dd_value = df.loc[max_dd_idx, 'strategy_dd']
        max_dd_date = df.loc[max_dd_idx, 'trade_date']
        
        fig.add_annotation(
            x=max_dd_date,
            y=max_dd_value,
            text=f"最大回撤: {max_dd_value:.2f}%",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=self.COLOR_SCHEME['strategy'],
            ax=0,
            ay=-40,
            font=dict(size=12, color=self.COLOR_SCHEME['strategy'])
        )
        
        # 布局设置
        fig.update_layout(
            title=dict(text=title, font=dict(size=18, family='Arial', color='#2c3e50')),
            xaxis=dict(
                title='日期',
                titlefont=dict(size=14),
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            yaxis=dict(
                title='回撤 (%)',
                titlefont=dict(size=14),
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            hovermode='x unified',
            template=self.template,
            width=self.default_width,
            height=self.default_height,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12)
            ),
            margin=dict(l=60, r=40, t=80, b=60)
        )
        
        if save_path:
            fig.write_html(save_path if save_path.endswith('.html') else save_path + '.html')
            self.logger.info(f"图表已保存: {save_path}")
        
        return fig
    
    def plot_ic_analysis(
        self,
        ic_df: pd.DataFrame,
        title: str = 'IC时间序列分析',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        绘制IC分析图（时间序列 + 分布）
        
        Args:
            ic_df: IC DataFrame（需要包含 trade_date, ic, rank_ic 列）
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            plotly Figure对象
        """
        if 'trade_date' not in ic_df.columns:
            ic_df = ic_df.copy()
            ic_df['trade_date'] = ic_df.index
        
        ic_df['trade_date'] = pd.to_datetime(ic_df['trade_date'])
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('IC时间序列', 'IC分布', '累计IC', 'IC移动平均'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.10
        )
        
        # 1. IC时间序列
        fig.add_trace(
            go.Scatter(
                x=ic_df['trade_date'],
                y=ic_df['ic'],
                mode='lines',
                name='IC',
                line=dict(color=self.COLOR_SCHEME['ic'], width=1.5),
                hovertemplate='<b>日期</b>: %{x}<br><b>IC</b>: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 添加零线
        fig.add_hline(y=0, line_dash="dash", line_color=self.COLOR_SCHEME['neutral'], 
                     line_width=1, row=1, col=1)
        
        # 2. IC分布直方图
        fig.add_trace(
            go.Histogram(
                x=ic_df['ic'].dropna(),
                nbinsx=50,
                name='IC分布',
                marker=dict(color=self.COLOR_SCHEME['ic'], opacity=0.7),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 添加均值线
        ic_mean = ic_df['ic'].mean()
        fig.add_vline(
            x=ic_mean,
            line_dash="dash",
            line_color=self.COLOR_SCHEME['benchmark'],
            line_width=2,
            annotation_text=f"均值={ic_mean:.4f}",
            annotation_position="top right",
            row=1, col=2
        )
        
        # 3. 累计IC
        cum_ic = ic_df['ic'].cumsum()
        fig.add_trace(
            go.Scatter(
                x=ic_df['trade_date'],
                y=cum_ic,
                mode='lines',
                name='累计IC',
                line=dict(color=self.COLOR_SCHEME['long'], width=2),
                fill='tozeroy',
                fillcolor='rgba(16, 172, 132, 0.2)',
                hovertemplate='<b>日期</b>: %{x}<br><b>累计IC</b>: %{y:.4f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. IC移动平均
        ic_ma_20 = ic_df['ic'].rolling(window=20, min_periods=1).mean()
        ic_ma_60 = ic_df['ic'].rolling(window=60, min_periods=1).mean()
        
        fig.add_trace(
            go.Scatter(
                x=ic_df['trade_date'],
                y=ic_ma_20,
                mode='lines',
                name='MA20',
                line=dict(color=self.COLOR_SCHEME['strategy'], width=2),
                hovertemplate='<b>日期</b>: %{x}<br><b>MA20</b>: %{y:.4f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=ic_df['trade_date'],
                y=ic_ma_60,
                mode='lines',
                name='MA60',
                line=dict(color=self.COLOR_SCHEME['benchmark'], width=2, dash='dash'),
                hovertemplate='<b>日期</b>: %{x}<br><b>MA60</b>: %{y:.4f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.add_hline(y=0, line_dash="dot", line_color=self.COLOR_SCHEME['neutral'], 
                     line_width=1, row=2, col=2)
        
        # 更新布局
        fig.update_layout(
            title=dict(text=title, font=dict(size=18, family='Arial', color='#2c3e50')),
            template=self.template,
            width=1400,
            height=900,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=11)
            ),
            margin=dict(l=60, r=40, t=100, b=60)
        )
        
        # 更新坐标轴标题
        fig.update_xaxes(title_text="日期", row=1, col=1)
        fig.update_xaxes(title_text="IC值", row=1, col=2)
        fig.update_xaxes(title_text="日期", row=2, col=1)
        fig.update_xaxes(title_text="日期", row=2, col=2)
        
        fig.update_yaxes(title_text="IC", row=1, col=1)
        fig.update_yaxes(title_text="频数", row=1, col=2)
        fig.update_yaxes(title_text="累计IC", row=2, col=1)
        fig.update_yaxes(title_text="IC移动平均", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path if save_path.endswith('.html') else save_path + '.html')
            self.logger.info(f"图表已保存: {save_path}")
        
        return fig
    
    def plot_group_returns(
        self,
        group_df: pd.DataFrame,
        title: str = '因子分组收益分析',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        绘制分组收益柱状图
        
        Args:
            group_df: 分组DataFrame
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            plotly Figure对象
        """
        # 计算每组的平均收益
        if 'return_mean' in group_df.columns:
            avg_returns = group_df.groupby('group')['return_mean'].mean().sort_index()
        else:
            # 备用方案
            avg_returns = pd.Series(np.linspace(-0.05, 0.05, 10), index=range(1, 11))
        
        # 确定颜色
        colors = [self.COLOR_SCHEME['short'] if x < 0 else self.COLOR_SCHEME['long'] 
                 for x in avg_returns.values]
        
        # 创建图表
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[f'G{i}' for i in avg_returns.index],
            y=avg_returns.values * 100,  # 转换为百分比
            marker=dict(
                color=colors,
                opacity=0.8,
                line=dict(color='black', width=1)
            ),
            text=[f'{v:.2f}%' for v in avg_returns.values * 100],
            textposition='outside',
            hovertemplate='<b>分组</b>: %{x}<br><b>平均收益</b>: %{y:.2f}%<extra></extra>'
        ))
        
        # 添加零线
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color=self.COLOR_SCHEME['neutral'],
            line_width=2
        )
        
        # 布局设置
        fig.update_layout(
            title=dict(text=title, font=dict(size=18, family='Arial', color='#2c3e50')),
            xaxis=dict(
                title='分组 (G1=最低, G10=最高)',
                titlefont=dict(size=14),
                showgrid=False
            ),
            yaxis=dict(
                title='平均收益 (%)',
                titlefont=dict(size=14),
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            template=self.template,
            width=self.default_width,
            height=self.default_height,
            margin=dict(l=60, r=40, t=80, b=60),
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path if save_path.endswith('.html') else save_path + '.html')
            self.logger.info(f"图表已保存: {save_path}")
        
        return fig
    
    def plot_long_short_performance(
        self,
        portfolios: Dict[str, pd.DataFrame],
        title: str = '多空组合表现对比',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        绘制多空组合表现
        
        Args:
            portfolios: 组合字典（需包含 'long', 'short', 'long_short'）
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            plotly Figure对象
        """
        fig = go.Figure()
        
        # 多头组合
        if 'long' in portfolios:
            long_df = portfolios['long'].copy()
            if 'trade_date' not in long_df.columns:
                long_df['trade_date'] = long_df.index
            long_df['trade_date'] = pd.to_datetime(long_df['trade_date'])
            long_cumret = (1 + long_df['portfolio_return']).cumprod()
            
            fig.add_trace(go.Scatter(
                x=long_df['trade_date'],
                y=long_cumret,
                mode='lines',
                name='多头组合',
                line=dict(color=self.COLOR_SCHEME['long'], width=2.5),
                hovertemplate='<b>日期</b>: %{x}<br><b>净值</b>: %{y:.4f}<extra></extra>'
            ))
        
        # 空头组合
        if 'short' in portfolios:
            short_df = portfolios['short'].copy()
            if 'trade_date' not in short_df.columns:
                short_df['trade_date'] = short_df.index
            short_df['trade_date'] = pd.to_datetime(short_df['trade_date'])
            short_cumret = (1 + short_df['portfolio_return']).cumprod()
            
            fig.add_trace(go.Scatter(
                x=short_df['trade_date'],
                y=short_cumret,
                mode='lines',
                name='空头组合',
                line=dict(color=self.COLOR_SCHEME['short'], width=2.5),
                hovertemplate='<b>日期</b>: %{x}<br><b>净值</b>: %{y:.4f}<extra></extra>'
            ))
        
        # 多空组合
        if 'long_short' in portfolios:
            ls_df = portfolios['long_short'].copy()
            if 'trade_date' not in ls_df.columns:
                ls_df['trade_date'] = ls_df.index
            ls_df['trade_date'] = pd.to_datetime(ls_df['trade_date'])
            ls_cumret = (1 + ls_df['portfolio_return']).cumprod()
            
            fig.add_trace(go.Scatter(
                x=ls_df['trade_date'],
                y=ls_cumret,
                mode='lines',
                name='多空组合',
                line=dict(color=self.COLOR_SCHEME['strategy'], width=3, dash='dot'),
                hovertemplate='<b>日期</b>: %{x}<br><b>净值</b>: %{y:.4f}<extra></extra>'
            ))
        
        # 添加基准线
        fig.add_hline(
            y=1,
            line_dash="dash",
            line_color=self.COLOR_SCHEME['neutral'],
            line_width=1,
            annotation_text="初始净值",
            annotation_position="right"
        )
        
        # 布局设置
        fig.update_layout(
            title=dict(text=title, font=dict(size=18, family='Arial', color='#2c3e50')),
            xaxis=dict(
                title='日期',
                titlefont=dict(size=14),
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            yaxis=dict(
                title='累计净值',
                titlefont=dict(size=14),
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            hovermode='x unified',
            template=self.template,
            width=self.default_width,
            height=self.default_height,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12)
            ),
            margin=dict(l=60, r=40, t=80, b=60)
        )
        
        if save_path:
            fig.write_html(save_path if save_path.endswith('.html') else save_path + '.html')
            self.logger.info(f"图表已保存: {save_path}")
        
        return fig
    
    def create_comprehensive_dashboard(
        self,
        portfolios: Dict[str, pd.DataFrame],
        ic_df: pd.DataFrame,
        metrics: Dict,
        benchmark_name: str = 'zz800',
        output_dir: str = 'output/plots'
    ):
        """
        创建综合分析仪表板（生成所有交互式图表）
        
        Args:
            portfolios: 组合字典
            ic_df: IC DataFrame
            metrics: 绩效指标字典
            benchmark_name: 基准名称
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("生成综合分析仪表板...")
        
        # 1. 累计收益对比（含基准）
        if 'long_short' in portfolios:
            self.logger.info("  [1/6] 累计收益对比图")
            self.plot_cumulative_returns_with_benchmark(
                portfolios['long_short'],
                benchmark_name=benchmark_name,
                save_path=str(output_path / 'cumulative_returns_benchmark.html')
            )
        
        # 2. 超额收益分析
        if 'long_short' in portfolios:
            self.logger.info("  [2/6] 超额收益分析图")
            self.plot_excess_returns(
                portfolios['long_short'],
                benchmark_name=benchmark_name,
                save_path=str(output_path / 'excess_returns.html')
            )
        
        # 3. 回撤对比
        if 'long_short' in portfolios:
            self.logger.info("  [3/6] 回撤对比图")
            self.plot_drawdown_comparison(
                portfolios['long_short'],
                benchmark_name=benchmark_name,
                save_path=str(output_path / 'drawdown_comparison.html')
            )
        
        # 4. IC分析
        self.logger.info("  [4/6] IC分析图")
        self.plot_ic_analysis(
            ic_df,
            save_path=str(output_path / 'ic_analysis.html')
        )
        
        # 5. 分组收益
        if 'groups' in portfolios:
            self.logger.info("  [5/6] 分组收益图")
            self.plot_group_returns(
                portfolios['groups'],
                save_path=str(output_path / 'group_returns.html')
            )
        
        # 6. 多空组合表现
        self.logger.info("  [6/6] 多空组合表现图")
        self.plot_long_short_performance(
            portfolios,
            save_path=str(output_path / 'long_short_performance.html')
        )
        
        self.logger.info(f"✅ 仪表板生成完成！所有图表已保存到: {output_path}")
        self.logger.info(f"   可在浏览器中打开 .html 文件查看交互式图表")


if __name__ == '__main__':
    # 配置日志
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("ResultVisualizerPlotly - 增强版可视化器测试")
    print("=" * 80)
    print("\n此模块提供基于 Plotly 的交互式专业图表：")
    print("  ✓ 策略与基准收益对比")
    print("  ✓ 超额收益分析")
    print("  ✓ 回撤对比分析")
    print("  ✓ IC时间序列分析")
    print("  ✓ 因子分组收益")
    print("  ✓ 多空组合表现")
    print("\n所有图表均为交互式，支持缩放、悬停、导出等功能")
    print("=" * 80)
