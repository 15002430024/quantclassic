# 简化后的回测流程使用指南

## 🎯 核心改进

### 之前（复杂版本）

需要手动调用 6 个步骤，编写 ~350 行代码：

```python
# 步骤 1: 因子预处理
processor = FactorProcessor(config)
processed_df = processor.process(...)

# 步骤 2: IC 分析
ic_analyzer = ICAnalyzer(config)
ic_df = ic_analyzer.calculate_ic(...)
ic_stats = ic_analyzer.analyze_ic_statistics(...)

# 步骤 3: 构建组合
builder = PortfolioBuilder(config)
portfolios = builder.build_portfolios(...)

# 步骤 4: 绩效评估
evaluator = PerformanceEvaluator(config)
metrics = {}
for name in ['long', 'short', 'long_short']:
    metrics[name] = evaluator.evaluate_portfolio(...)

# 步骤 5: 生成图表
visualizer = ResultVisualizer(config)
visualizer.plot_cumulative_returns(...)
visualizer.plot_drawdown(...)
visualizer.plot_ic_series(...)
# ... 更多手动作图代码

# 步骤 6: 手动保存数据和图表
# ... 大量手动保存代码
```

### 现在（简化版本）

**一行代码完成所有回测！** 只需 ~10 行代码：

```python
# 🎯 一键运行完整回测流程
from quantclassic.Factorsystem import BacktestRunner

runner = BacktestRunner(backtest_config)

results = runner.run_backtest(
    factor_df=backtest_df,
    factor_col='factor_value',
    return_col='y_processed',
    output_dir='output/backtest',
    save_plots=True,
    verbose=True
)

# 完成！自动包含：
# ✓ 因子预处理
# ✓ IC 分析
# ✓ 组合构建
# ✓ 绩效评估
# ✓ 生成所有图表
# ✓ 保存所有数据
```

## 📦 BacktestRunner - 一键回测工具

### 功能特性

`BacktestRunner` 封装了完整的回测流程：

1. **因子预处理** - 自动去极值、标准化
2. **IC 分析** - 计算 IC、Rank IC、统计检验
3. **组合构建** - 多头、空头、多空组合
4. **绩效评估** - 20+ 绩效指标
5. **可视化** - 自动生成 6+ 张图表
6. **数据保存** - 自动保存所有中间结果

### 基础用法

```python
from quantclassic.Factorsystem import BacktestRunner, BacktestConfig

# 创建配置
config = BacktestConfig(
    output_dir='output/my_backtest',
    n_groups=10,
    rebalance_freq='weekly',
    long_ratio=0.2,
    short_ratio=0.2
)

# 创建运行器
runner = BacktestRunner(config)

# 运行回测
results = runner.run_backtest(
    factor_df=my_factor_df,  # 必须包含: order_book_id, trade_date, factor_value, y_processed
    factor_col='factor_value',
    return_col='y_processed',
    save_plots=True,
    verbose=True
)
```

### 返回结果

`results` 字典包含所有结果：

```python
{
    'processed_df': DataFrame,      # 处理后的因子数据
    'ic_df': DataFrame,              # IC 分析结果
    'ic_stats': Dict,                # IC 统计指标
    'portfolios': Dict[str, DataFrame],  # 组合数据
    'metrics': Dict[str, Dict],      # 绩效指标
    'plots_dir': str,                # 图表保存路径
    'output_dir': str                # 输出目录
}
```

### 访问结果

```python
# IC 分析
ic_stats = results['ic_stats']
print(f"IC 均值: {ic_stats['ic_mean']:.4f}")
print(f"ICIR: {ic_stats['icir']:.4f}")

# 组合数据
portfolios = results['portfolios']
long_short = portfolios['long_short']

# 绩效指标
metrics = results['metrics']
ls_metrics = metrics['long_short']
print(f"夏普比率: {ls_metrics['sharpe_ratio']:.4f}")
print(f"最大回撤: {ls_metrics['max_drawdown']:.2%}")

# 图表路径
print(f"图表保存在: {results['plots_dir']}")
```

### 生成文本报告

```python
# 生成并保存报告
report = runner.generate_report_text(
    ic_stats=results['ic_stats'],
    metrics=results['metrics'],
    output_path='output/my_backtest/report.txt'
)

print(report)
```

## 🔄 与 Workflow 集成

### 自动保存到 Workflow

```python
from quantclassic.workflow import R
from quantclassic.Factorsystem import BacktestRunner

# 运行回测
runner = BacktestRunner(config)
results = runner.run_backtest(factor_df, save_plots=True)

# 保存到 Workflow
with R.start(experiment_name='my_experiment'):
    # 保存配置
    R.save_objects(
        data_config=data_config,
        model_config=model_config,
        backtest_config=config
    )
    
    # 记录指标
    R.log_metrics(**results['ic_stats'])
    R.log_metrics(**results['metrics']['long_short'])
    
    # 保存数据
    R.save_objects(
        processed_df=results['processed_df'],
        ic_df=results['ic_df'],
        portfolios=results['portfolios'],
        ic_stats=results['ic_stats'],
        metrics=results['metrics']
    )
    
    # 保存图表
    import shutil
    from pathlib import Path
    
    if R.current_recorder:
        artifacts_dir = Path(R.current_recorder.recorder_dir) / "artifacts" / "plots"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        for plot_file in Path(results['plots_dir']).glob('*.png'):
            shutil.copy2(plot_file, artifacts_dir)
```

## 📊 自动生成的图表

BacktestRunner 自动生成以下图表：

1. **cumulative_returns.png** - 累计收益曲线（含基准）
2. **drawdown.png** - 回撤曲线
3. **ic_series.png** - IC 时间序列（含移动平均）
4. **ic_distribution.png** - IC 分布直方图
5. **group_returns.png** - 分组收益柱状图
6. **long_short_performance.png** - 多空组合表现对比

所有图表自动保存到 `output_dir/plots/` 目录。

## 💾 自动保存的数据

BacktestRunner 自动保存以下数据：

1. **ic_analysis.csv** - IC 分析结果
2. **portfolio_long.csv** - 多头组合数据
3. **portfolio_short.csv** - 空头组合数据
4. **portfolio_long_short.csv** - 多空组合数据
5. **metrics.json** - 绩效指标
6. **ic_stats.json** - IC 统计指标

所有数据自动保存到 `output_dir/` 目录。

## 🎨 自定义配置

### 配置回测参数

```python
from quantclassic.Factorsystem import BacktestConfig

config = BacktestConfig(
    # 输出设置
    output_dir='output/my_backtest',
    
    # 组合构建
    n_groups=10,                # 分组数量
    rebalance_freq='weekly',    # 调仓频率: daily, weekly, biweekly, monthly
    long_ratio=0.2,             # 多头比例（前 20%）
    short_ratio=0.2,            # 空头比例（后 20%）
    weight_method='equal',      # 权重方法: equal, value_weight
    
    # 交易成本
    commission_rate=0.0003,     # 佣金率
    stamp_tax_rate=0.001,       # 印花税率
    slippage_rate=0.001,        # 滑点率
    
    # IC 分析
    ic_method='spearman',       # IC 方法: pearson, spearman
    holding_periods=[1, 5, 10, 20],  # 持有期
    
    # 图表设置
    plot_style='seaborn',       # 绘图风格
    figure_size=(16, 10),       # 图表大小
    dpi=150,                    # 图表分辨率
    
    # 其他
    annual_factor=252,          # 年化因子
    risk_free_rate=0.03         # 无风险利率
)
```

### 不同调仓频率

```python
# 日度调仓
config_daily = BacktestConfig(rebalance_freq='daily')

# 周度调仓
config_weekly = BacktestConfig(rebalance_freq='weekly')

# 双周调仓
config_biweekly = BacktestConfig(rebalance_freq='biweekly')

# 月度调仓
config_monthly = BacktestConfig(rebalance_freq='monthly')
```

### 不同多空比例

```python
# 标准多空（各 20%）
config_standard = BacktestConfig(long_ratio=0.2, short_ratio=0.2)

# 极端多空（各 10%）
config_extreme = BacktestConfig(long_ratio=0.1, short_ratio=0.1)

# 只做多
config_long_only = BacktestConfig(long_ratio=0.2, short_ratio=0.0)

# 只做空
config_short_only = BacktestConfig(long_ratio=0.0, short_ratio=0.2)
```

## 🔧 高级用法

### 批量回测

```python
# 测试不同参数组合
configs = [
    BacktestConfig(n_groups=5, rebalance_freq='weekly'),
    BacktestConfig(n_groups=10, rebalance_freq='weekly'),
    BacktestConfig(n_groups=10, rebalance_freq='biweekly'),
]

results_list = []
for i, config in enumerate(configs):
    runner = BacktestRunner(config)
    results = runner.run_backtest(
        factor_df=my_factor_df,
        output_dir=f'output/backtest_{i}',
        verbose=False  # 关闭详细输出
    )
    results_list.append(results)

# 对比结果
for i, results in enumerate(results_list):
    sharpe = results['metrics']['long_short']['sharpe_ratio']
    ic = results['ic_stats']['ic_mean']
    print(f"配置 {i}: Sharpe={sharpe:.4f}, IC={ic:.4f}")
```

### 自定义因子列

```python
# 使用不同的因子列名
results = runner.run_backtest(
    factor_df=my_df,
    factor_col='my_custom_factor',  # 自定义因子列
    return_col='future_return',     # 自定义收益列
    save_plots=True
)
```

### 禁用图表生成

```python
# 只运行回测，不生成图表（节省时间）
results = runner.run_backtest(
    factor_df=my_factor_df,
    save_plots=False,  # 不生成图表
    verbose=False      # 不打印进度
)
```

## 📈 完整示例

### Notebook 中的完整流程

```python
# 步骤 1: 准备数据
# ... 数据加载和预处理代码 ...

# 步骤 2: 模型训练
# ... LSTM/GRU 训练代码 ...

# 步骤 3: 生成因子
alpha_factors = predictions_df[['order_book_id', 'trade_date', 'y_pred']].copy()
alpha_factors.rename(columns={'y_pred': 'factor_value'}, inplace=True)

# 步骤 4: 合并收益数据
backtest_df = alpha_factors.merge(returns_df, on=['order_book_id', 'trade_date'])

# 步骤 5: 一键回测（替代原来的 300+ 行代码）
from quantclassic.Factorsystem import BacktestRunner, BacktestConfig

config = BacktestConfig(
    output_dir='output/backtest',
    n_groups=10,
    rebalance_freq='biweekly'
)

runner = BacktestRunner(config)
results = runner.run_backtest(
    factor_df=backtest_df,
    factor_col='factor_value',
    return_col='y_processed',
    save_plots=True,
    verbose=True
)

# 步骤 6: 保存到 Workflow（10 行代码）
from quantclassic.workflow import R

with R.start(experiment_name='my_lstm_alpha'):
    R.save_objects(config=config)
    R.log_metrics(**results['ic_stats'])
    R.log_metrics(**results['metrics']['long_short'])
    R.save_objects(
        processed_df=results['processed_df'],
        portfolios=results['portfolios'],
        metrics=results['metrics']
    )
    
    # 复制图表
    if R.current_recorder:
        import shutil
        from pathlib import Path
        
        artifacts_dir = Path(R.current_recorder.recorder_dir) / "artifacts" / "plots"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        for plot_file in Path(results['plots_dir']).glob('*.png'):
            shutil.copy2(plot_file, artifacts_dir)

# 完成！
```

## 🎯 代码对比

### 原来的方式（~350 行）

```python
# 手动实现所有步骤...
processor = FactorProcessor(config)
ic_analyzer = ICAnalyzer(config)
builder = PortfolioBuilder(config)
evaluator = PerformanceEvaluator(config)
visualizer = ResultVisualizer(config)

# 大量手动调用和数据处理...
# ... 300+ 行代码 ...
```

### 现在的方式（~10 行）

```python
runner = BacktestRunner(config)
results = runner.run_backtest(
    factor_df=backtest_df,
    save_plots=True
)
```

**代码减少 97%！** 🎉

## 📚 相关文档

- [BacktestConfig 配置指南](./BACKTEST_GUIDE.md)
- [Workflow 数据管理](../workflow/DATA_MANAGEMENT_GUIDE.md)
- [完整 API 文档](./README.md)
