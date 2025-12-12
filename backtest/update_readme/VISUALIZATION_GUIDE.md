# 增强版可视化功能使用指南

## 概述

quantclassic 项目的可视化功能已全面升级，新增了**基准收益对比**功能，并提供两种美观的可视化方案：

1. **ResultVisualizer** (matplotlib) - 静态高质量图表
2. **ResultVisualizerPlotly** (plotly) - 交互式专业图表

## 主要特性

✅ **基准收益对比** - 自动获取并对比沪深300、中证500、中证800等基准指数  
✅ **超额收益分析** - 可视化策略相对基准的超额收益  
✅ **回撤对比** - 对比策略和基准的回撤情况  
✅ **美观配色** - 专业的配色方案，清晰易读  
✅ **交互式图表** - plotly版本支持缩放、悬停、导出等功能  
✅ **智能缓存** - 基准数据自动缓存，加速后续使用  

## 快速开始

### 1. matplotlib 版本（静态图）

```python
from quantclassic.Factorsystem import BacktestConfig, ResultVisualizer

# 创建配置
config = BacktestConfig()

# 创建可视化器
visualizer = ResultVisualizer(config)

# 绘制累计收益曲线（含基准对比）
visualizer.plot_cumulative_returns(
    portfolio_df,
    benchmark_name='zz800',  # 使用中证800作为基准
    title='策略 vs 中证800累计收益对比',
    save_path='output/cumulative_returns.png'
)

# 绘制超额收益
visualizer.plot_excess_returns(
    portfolio_df,
    benchmark_name='zz800',
    save_path='output/excess_returns.png'
)

# 绘制回撤对比
visualizer.plot_drawdown_comparison(
    portfolio_df,
    benchmark_name='zz800',
    save_path='output/drawdown_comparison.png'
)

# 生成完整报告（所有图表）
visualizer.create_comprehensive_report(
    portfolios=portfolios,
    ic_df=ic_df,
    metrics=metrics,
    output_dir='output/plots',
    benchmark_name='zz800'
)
```

### 2. Plotly 版本（交互式图）

```python
from quantclassic.Factorsystem import BacktestConfig, ResultVisualizerPlotly

# 创建配置
config = BacktestConfig()

# 创建可视化器
visualizer = ResultVisualizerPlotly(config)

# 绘制累计收益曲线（交互式）
fig = visualizer.plot_cumulative_returns_with_benchmark(
    portfolio_df,
    benchmark_name='zz800',
    title='策略 vs 中证800累计收益对比',
    save_path='output/cumulative_returns.html'
)

# 生成完整交互式仪表板
visualizer.create_comprehensive_dashboard(
    portfolios=portfolios,
    ic_df=ic_df,
    metrics=metrics,
    benchmark_name='zz800',
    output_dir='output/dashboard'
)
```

## 支持的基准指数

| 参数值 | 指数名称 | 代码 |
|--------|----------|------|
| `'hs300'` | 沪深300 | 000300.XSHG |
| `'zz500'` | 中证500 | 000905.XSHG |
| `'zz800'` | 中证800 | 000906.XSHG |
| `'sz50'` | 上证50 | 000016.XSHG |
| `'zz1000'` | 中证1000 | 000852.XSHG |
| `'cybz'` | 创业板指 | 399006.XSHE |

## 可用的图表类型

### matplotlib 版本

1. **plot_cumulative_returns()** - 累计收益曲线（含基准）
2. **plot_excess_returns()** - 超额收益分析
3. **plot_drawdown_comparison()** - 回撤对比
4. **plot_drawdown()** - 回撤曲线
5. **plot_ic_series()** - IC时间序列
6. **plot_ic_distribution()** - IC分布
7. **plot_group_returns()** - 分组收益
8. **plot_long_short_performance()** - 多空组合表现

### Plotly 版本

1. **plot_cumulative_returns_with_benchmark()** - 累计收益对比（交互式）
2. **plot_excess_returns()** - 超额收益分析（交互式）
3. **plot_drawdown_comparison()** - 回撤对比（交互式）
4. **plot_ic_analysis()** - IC综合分析（4个子图）
5. **plot_group_returns()** - 分组收益（交互式）
6. **plot_long_short_performance()** - 多空组合表现（交互式）

## 完整示例

运行示例脚本查看所有功能：

```bash
cd /home/u2025210237/jupyterlab/quantclassic/Factorsystem
python example_enhanced_visualization.py
```

这将生成：
- `output/test_visualization_matplotlib/` - matplotlib静态图
- `output/test_visualization_plotly/` - plotly交互式图
- `output/comprehensive_dashboard/` - 完整仪表板
- `output/benchmark_comparison/` - 基准对比

## 在 Notebook 中使用

### Jupyter Notebook

```python
from quantclassic.Factorsystem import ResultVisualizerPlotly

visualizer = ResultVisualizerPlotly(config)

# 直接显示交互式图表
fig = visualizer.plot_cumulative_returns_with_benchmark(
    portfolio_df,
    benchmark_name='zz800'
)
fig.show()  # 在 notebook 中直接显示
```

### 集成到现有回测流程

只需在原有代码中替换可视化器的调用：

```python
# 原有代码
from quantclassic.Factorsystem import ResultVisualizer
visualizer = ResultVisualizer(backtest_config)

# 新增基准参数即可
visualizer.create_comprehensive_report(
    portfolios=portfolios,
    ic_df=ic_df,
    metrics=all_metrics,
    output_dir='output/plots',
    benchmark_name='zz800'  # 新增这个参数
)
```

## 配色方案

新的可视化器使用专业配色：

- **策略** - 蓝色 (#2E86DE)
- **基准** - 红色 (#EE5A6F)
- **多头** - 绿色 (#10AC84)
- **空头** - 红色 (#EE5A6F)
- **IC** - 紫色 (#5f27cd)

## 注意事项

1. **首次运行**：首次获取基准数据时会从 API 下载，需要联网
2. **缓存机制**：基准数据会自动缓存到 `cache/benchmark/` 目录
3. **增量更新**：再次使用时会自动使用缓存，只下载缺失的数据
4. **数据源**：默认使用米筐（rqdatac），如需使用其他数据源请参考 BenchmarkManager 文档

## 常见问题

**Q: 如何清除基准数据缓存？**

```python
from quantclassic.Factorsystem import BenchmarkManager
manager = BenchmarkManager()
manager.clear_cache()  # 清除所有缓存
```

**Q: 如何查看缓存信息？**

```python
manager = BenchmarkManager()
cache_info = manager.get_cache_info()
print(cache_info)
```

**Q: 图表不显示中文怎么办？**

可视化器会自动配置跨平台中文字体，如果仍有问题，请安装：
- Linux: `WenQuanYi Micro Hei` 或 `Noto Sans CJK`
- macOS: 系统自带
- Windows: 系统自带

## 更新日志

**v1.1.0** (2024-11-24)
- ✨ 新增 ResultVisualizerPlotly 交互式可视化器
- ✨ 增强 ResultVisualizer 支持基准对比
- ✨ 新增超额收益分析图
- ✨ 新增回撤对比图
- 🎨 优化配色方案
- 📦 集成 BenchmarkManager 智能缓存

---

如有问题或建议，请提交 issue 或联系开发团队。
