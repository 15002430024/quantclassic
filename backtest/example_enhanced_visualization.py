"""
增强版可视化功能使用示例
展示如何使用新的 ResultVisualizer 和 ResultVisualizerPlotly
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from quantclassic.backtest import (
    BacktestConfig,
    ResultVisualizer,
    ResultVisualizerPlotly,
    BenchmarkManager
)


def generate_sample_data(n_days=252, start_date='2023-01-01'):
    """
    生成示例数据用于测试
    
    Args:
        n_days: 天数
        start_date: 起始日期
        
    Returns:
        portfolio_df, ic_df, portfolios, metrics
    """
    # 生成日期
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # 生成策略收益（带一些趋势和波动）
    np.random.seed(42)
    trend = np.linspace(0, 0.0001, n_days)
    noise = np.random.normal(0, 0.01, n_days)
    returns = trend + noise
    
    # 组合DataFrame
    portfolio_df = pd.DataFrame({
        'trade_date': dates,
        'portfolio_return': returns
    })
    
    # IC DataFrame
    ic_values = np.random.normal(0.05, 0.03, n_days)
    ic_df = pd.DataFrame({
        'trade_date': dates,
        'ic': ic_values,
        'rank_ic': ic_values * 0.9,
        'cum_ic': np.cumsum(ic_values),
        'cum_rank_ic': np.cumsum(ic_values * 0.9)
    })
    
    # 多空组合
    portfolios = {
        'long': portfolio_df.copy(),
        'short': portfolio_df.copy(),
        'long_short': portfolio_df.copy(),
        'groups': pd.DataFrame({
            'group': np.repeat(range(1, 11), 25),
            'return_mean': np.linspace(-0.02, 0.05, 250)
        })
    }
    
    # 调整多头和空头收益
    portfolios['long']['portfolio_return'] = returns * 1.2
    portfolios['short']['portfolio_return'] = -returns * 0.8
    
    # 绩效指标
    metrics = {
        'long_short': {
            'annual_return': 0.15,
            'annual_volatility': 0.20,
            'sharpe_ratio': 0.75,
            'max_drawdown': -0.12,
            'calmar_ratio': 1.25,
            'win_rate': 0.55
        }
    }
    
    return portfolio_df, ic_df, portfolios, metrics


def example_matplotlib_visualizer():
    """示例1: 使用增强版的 matplotlib 可视化器"""
    print("\n" + "=" * 80)
    print("示例1: 增强版 ResultVisualizer (matplotlib)")
    print("=" * 80)
    
    # 1. 创建配置
    config = BacktestConfig()
    config.figure_size = (12, 6)
    config.dpi = 150
    
    # 2. 生成示例数据
    portfolio_df, ic_df, portfolios, metrics = generate_sample_data()
    
    # 3. 创建可视化器
    visualizer = ResultVisualizer(config)
    
    # 4. 输出目录
    output_dir = Path('output/test_visualization_matplotlib')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n生成图表...")
    
    # 5. 累计收益曲线（含基准对比）
    print("  [1/4] 累计收益曲线（含基准）")
    visualizer.plot_cumulative_returns(
        portfolio_df,
        benchmark_name='zz800',  # 自动获取中证800作为基准
        title='策略 vs 中证800累计收益对比',
        save_path=str(output_dir / 'cumulative_returns_benchmark.png')
    )
    
    # 6. 超额收益分析
    print("  [2/4] 超额收益分析")
    visualizer.plot_excess_returns(
        portfolio_df,
        benchmark_name='zz800',
        save_path=str(output_dir / 'excess_returns.png')
    )
    
    # 7. 回撤对比
    print("  [3/4] 回撤对比分析")
    visualizer.plot_drawdown_comparison(
        portfolio_df,
        benchmark_name='zz800',
        save_path=str(output_dir / 'drawdown_comparison.png')
    )
    
    # 8. 生成完整报告
    print("  [4/4] 综合报告")
    visualizer.create_comprehensive_report(
        portfolios=portfolios,
        ic_df=ic_df,
        metrics=metrics,
        output_dir=str(output_dir),
        benchmark_name='zz800'
    )
    
    print(f"\n✅ 所有图表已保存到: {output_dir}")
    print(f"   请查看生成的 .png 文件")


def example_plotly_visualizer():
    """示例2: 使用新的 plotly 交互式可视化器"""
    print("\n" + "=" * 80)
    print("示例2: ResultVisualizerPlotly (交互式)")
    print("=" * 80)
    
    # 1. 创建配置
    config = BacktestConfig()
    
    # 2. 生成示例数据
    portfolio_df, ic_df, portfolios, metrics = generate_sample_data()
    
    # 3. 创建可视化器
    visualizer = ResultVisualizerPlotly(config)
    
    # 4. 输出目录
    output_dir = Path('output/test_visualization_plotly')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n生成交互式图表...")
    
    # 5. 累计收益曲线（含基准对比）
    print("  [1/6] 累计收益曲线（含基准）")
    fig1 = visualizer.plot_cumulative_returns_with_benchmark(
        portfolio_df,
        benchmark_name='zz800',
        title='策略 vs 中证800累计收益对比',
        save_path=str(output_dir / 'cumulative_returns_benchmark.html')
    )
    
    # 6. 超额收益分析
    print("  [2/6] 超额收益分析")
    fig2 = visualizer.plot_excess_returns(
        portfolio_df,
        benchmark_name='zz800',
        save_path=str(output_dir / 'excess_returns.html')
    )
    
    # 7. 回撤对比
    print("  [3/6] 回撤对比分析")
    fig3 = visualizer.plot_drawdown_comparison(
        portfolio_df,
        benchmark_name='zz800',
        save_path=str(output_dir / 'drawdown_comparison.html')
    )
    
    # 8. IC分析
    print("  [4/6] IC分析")
    fig4 = visualizer.plot_ic_analysis(
        ic_df,
        save_path=str(output_dir / 'ic_analysis.html')
    )
    
    # 9. 分组收益
    print("  [5/6] 分组收益")
    fig5 = visualizer.plot_group_returns(
        portfolios['groups'],
        save_path=str(output_dir / 'group_returns.html')
    )
    
    # 10. 多空组合表现
    print("  [6/6] 多空组合表现")
    fig6 = visualizer.plot_long_short_performance(
        portfolios,
        save_path=str(output_dir / 'long_short_performance.html')
    )
    
    print(f"\n✅ 所有交互式图表已保存到: {output_dir}")
    print(f"   请在浏览器中打开 .html 文件查看交互式图表")
    print(f"   支持缩放、悬停、导出等功能")


def example_comprehensive_dashboard():
    """示例3: 生成完整的交互式仪表板"""
    print("\n" + "=" * 80)
    print("示例3: 完整交互式仪表板")
    print("=" * 80)
    
    # 1. 创建配置
    config = BacktestConfig()
    
    # 2. 生成示例数据
    portfolio_df, ic_df, portfolios, metrics = generate_sample_data()
    
    # 3. 创建可视化器
    visualizer = ResultVisualizerPlotly(config)
    
    # 4. 输出目录
    output_dir = Path('output/comprehensive_dashboard')
    
    # 5. 生成完整仪表板
    print("\n生成完整仪表板...")
    visualizer.create_comprehensive_dashboard(
        portfolios=portfolios,
        ic_df=ic_df,
        metrics=metrics,
        benchmark_name='zz800',
        output_dir=str(output_dir)
    )
    
    print(f"\n✅ 完整仪表板已生成: {output_dir}")


def example_benchmark_comparison():
    """示例4: 对比多个基准"""
    print("\n" + "=" * 80)
    print("示例4: 对比多个基准指数")
    print("=" * 80)
    
    # 1. 创建配置和数据
    config = BacktestConfig()
    portfolio_df, _, _, _ = generate_sample_data()
    
    # 2. 创建可视化器
    visualizer_plotly = ResultVisualizerPlotly(config)
    
    # 3. 输出目录
    output_dir = Path('output/benchmark_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n对比不同基准...")
    
    # 4. 对比沪深300
    print("  [1/3] vs 沪深300")
    visualizer_plotly.plot_cumulative_returns_with_benchmark(
        portfolio_df,
        benchmark_name='hs300',
        title='策略 vs 沪深300',
        save_path=str(output_dir / 'vs_hs300.html')
    )
    
    # 5. 对比中证500
    print("  [2/3] vs 中证500")
    visualizer_plotly.plot_cumulative_returns_with_benchmark(
        portfolio_df,
        benchmark_name='zz500',
        title='策略 vs 中证500',
        save_path=str(output_dir / 'vs_zz500.html')
    )
    
    # 6. 对比中证800
    print("  [3/3] vs 中证800")
    visualizer_plotly.plot_cumulative_returns_with_benchmark(
        portfolio_df,
        benchmark_name='zz800',
        title='策略 vs 中证800',
        save_path=str(output_dir / 'vs_zz800.html')
    )
    
    print(f"\n✅ 基准对比图已保存到: {output_dir}")


def main():
    """主函数：运行所有示例"""
    print("\n" + "=" * 80)
    print("增强版可视化功能使用示例")
    print("=" * 80)
    print("\n本脚本展示如何使用 quantclassic 的增强可视化功能：")
    print("  1. ResultVisualizer (matplotlib) - 静态高质量图表")
    print("  2. ResultVisualizerPlotly - 交互式专业图表")
    print("  3. 基准收益对比功能")
    print("  4. 综合分析仪表板")
    print("\n" + "=" * 80)
    
    try:
        # 示例1: matplotlib版本
        example_matplotlib_visualizer()
        
        # 示例2: plotly版本
        example_plotly_visualizer()
        
        # 示例3: 完整仪表板
        example_comprehensive_dashboard()
        
        # 示例4: 基准对比
        example_benchmark_comparison()
        
        print("\n" + "=" * 80)
        print("✅ 所有示例运行完成！")
        print("=" * 80)
        print("\n📊 查看生成的图表：")
        print("  • output/test_visualization_matplotlib/  - matplotlib静态图")
        print("  • output/test_visualization_plotly/      - plotly交互式图")
        print("  • output/comprehensive_dashboard/        - 完整仪表板")
        print("  • output/benchmark_comparison/           - 基准对比")
        print("\n💡 提示：")
        print("  • .png 文件可直接查看")
        print("  • .html 文件在浏览器中打开，支持交互操作")
        print("  • 图表支持缩放、悬停查看数据、导出等功能")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
