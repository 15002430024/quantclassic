"""
在当前 gru_ZSCORE11220111.ipynb 中集成增强可视化功能的代码片段

直接复制这段代码到你的 notebook 中，替换原有的可视化部分
"""

# =============================================================================
# 使用增强版可视化功能（含基准对比）
# =============================================================================

from quantclassic.backtest import ResultVisualizer

# 创建增强版可视化器
visualizer = ResultVisualizer(backtest_config)

# 创建输出目录
plot_dir = Path('output/backtest_rolling_gru/plots')
plot_dir.mkdir(parents=True, exist_ok=True)

print("\n【生成增强版可视化图表】")
print("=" * 80)

# 1. 累计收益曲线（含中证800基准对比）
print("  [1/7] 累计收益对比图（策略 vs 中证800）")
visualizer.plot_cumulative_returns(
    portfolios['long_short'],
    return_col='portfolio_return',
    benchmark_name='zz800',  # 使用中证800作为基准
    title='多空策略 vs 中证800累计收益对比',
    save_path=str(plot_dir / 'cumulative_returns_benchmark.png')
)

# 2. 超额收益分析
print("  [2/7] 超额收益分析图")
visualizer.plot_excess_returns(
    portfolios['long_short'],
    return_col='portfolio_return',
    benchmark_name='zz800',
    title='相对中证800的超额收益',
    save_path=str(plot_dir / 'excess_returns.png')
)

# 3. 回撤对比（策略 vs 基准）
print("  [3/7] 回撤对比图")
visualizer.plot_drawdown_comparison(
    portfolios['long_short'],
    return_col='portfolio_return',
    benchmark_name='zz800',
    title='策略与基准回撤对比',
    save_path=str(plot_dir / 'drawdown_comparison.png')
)

# 4. 传统回撤曲线
print("  [4/7] 策略回撤曲线")
visualizer.plot_drawdown(
    portfolios['long_short'],
    return_col='portfolio_return',
    title='策略回撤曲线',
    save_path=str(plot_dir / 'drawdown.png')
)

# 5. IC时间序列
print("  [5/7] IC时间序列")
visualizer.plot_ic_series(
    ic_df,
    save_path=str(plot_dir / 'ic_series.png')
)

# 6. IC分布
print("  [6/7] IC分布直方图")
visualizer.plot_ic_distribution(
    ic_df,
    save_path=str(plot_dir / 'ic_distribution.png')
)

# 7. 分组收益
if 'groups' in portfolios:
    print("  [7/7] 分组收益柱状图")
    visualizer.plot_group_returns(
        portfolios['groups'],
        save_path=str(plot_dir / 'group_returns.png')
    )

print(f"\n✅ 所有图表已生成！")
print(f"   保存位置: {plot_dir}")
print(f"\n   生成的图表:")
print(f"   • cumulative_returns_benchmark.png - 累计收益对比（含基准）")
print(f"   • excess_returns.png              - 超额收益分析")
print(f"   • drawdown_comparison.png         - 回撤对比")
print(f"   • drawdown.png                    - 回撤曲线")
print(f"   • ic_series.png                   - IC时间序列")
print(f"   • ic_distribution.png             - IC分布")
print(f"   • group_returns.png               - 分组收益")

print("\n" + "=" * 80)

# =============================================================================
# 或者使用一键生成完整报告
# =============================================================================

# 如果你想一键生成所有图表，可以使用：
"""
visualizer.create_comprehensive_report(
    portfolios=portfolios,
    ic_df=ic_df,
    metrics=all_metrics,
    output_dir=str(plot_dir),
    benchmark_name='zz800'  # 指定基准
)
"""

# =============================================================================
# 可选：使用交互式 Plotly 版本（需要先安装 plotly）
# =============================================================================
"""
# 安装命令: pip install plotly

from quantclassic.backtest import ResultVisualizerPlotly

# 创建交互式可视化器
visualizer_plotly = ResultVisualizerPlotly(backtest_config)

# 生成交互式仪表板
visualizer_plotly.create_comprehensive_dashboard(
    portfolios=portfolios,
    ic_df=ic_df,
    metrics=all_metrics,
    benchmark_name='zz800',
    output_dir='output/backtest_rolling_gru/dashboard'
)

# 交互式图表会保存为 .html 文件，可以在浏览器中打开
# 支持缩放、悬停查看数据、导出等功能
"""
