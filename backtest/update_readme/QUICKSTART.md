# Factorsystem 快速入门

## 5分钟快速上手

### 步骤1: 准备数据和模型

确保你有:
- ✅ 训练好的模型文件 (如 `output/best_model.pth`)
- ✅ 包含特征和收益的数据文件 (如 `output/train_data_final_01.parquet`)

数据必须包含的列:
- `ts_code` 或 `order_book_id`: 股票代码
- `trade_date`: 交易日期
- 若干特征列 (数值型)
- `y_processed` 或其他收益列

### 步骤2: 运行基础回测

```python
# 导入库
from Factorsystem import BacktestConfig, FactorBacktestSystem
import pandas as pd

# 1. 创建配置
config = BacktestConfig(
    output_dir='output/my_backtest',  # 输出目录
    save_plots=True                    # 保存图表
)

# 2. 初始化回测系统
system = FactorBacktestSystem(config)

# 3. 加载模型
model = system.load_model('output/best_model.pth')

# 4. 加载数据
df = pd.read_parquet('output/train_data_final_01.parquet')

# 5. 运行回测 (一行代码!)
results = system.run_backtest(df)

# 6. 查看关键指标
print(f"IC均值: {results['ic_stats']['ic_mean']:.4f}")
print(f"ICIR: {results['ic_stats']['icir']:.4f}")
print(f"夏普比率: {results['performance_metrics']['long_short']['sharpe_ratio']:.4f}")
print(f"年化收益: {results['performance_metrics']['long_short']['annual_return']:.2%}")
print(f"最大回撤: {results['performance_metrics']['long_short']['max_drawdown']:.2%}")
```

### 步骤3: 查看结果

运行后，在 `output/my_backtest/` 目录下会生成:

```
output/my_backtest/
├── factors.csv              # 因子数据
├── ic_analysis.csv          # IC分析
├── portfolio_*.csv          # 组合收益
├── performance_metrics.xlsx # 绩效指标
└── plots/                   # 6张专业图表
    ├── cumulative_returns.png
    ├── drawdown.png
    ├── ic_series.png
    ├── ic_distribution.png
    ├── group_returns.png
    └── long_short_performance.png
```

---

## 3个常用场景

### 场景1: 快速测试 (性能优先)

```python
from Factorsystem import ConfigTemplates, FactorBacktestSystem

config = ConfigTemplates.fast_test()
config.output_dir = 'output/fast_test'

system = FactorBacktestSystem(config)
model = system.load_model('output/best_model.pth')
df = pd.read_parquet('output/data.parquet')

results = system.run_backtest(df)
```

特点: 5分组, 少量持有期, 不生成PDF, 速度快

### 场景2: 详细分析 (全面性优先)

```python
from Factorsystem import ConfigTemplates, FactorBacktestSystem

config = ConfigTemplates.detailed_analysis()
config.output_dir = 'output/detailed'

system = FactorBacktestSystem(config)
# ... 同上
```

特点: 10分组, 多持有期, 行业/市值中性化, 考虑成本, 全面分析

### 场景3: 自定义配置

```python
from Factorsystem import BacktestConfig, FactorBacktestSystem

config = BacktestConfig(
    # 因子处理
    winsorize_method='mad',        # MAD去极值
    standardize_method='rank',     # 排序标准化
    industry_neutral=True,         # 行业中性化
    
    # 组合构建
    n_groups=5,                    # 5分组
    rebalance_freq='weekly',       # 每周换仓
    weight_method='factor_weight', # 因子值加权
    long_ratio=0.3,                # 做多前30%
    short_ratio=0.3,               # 做空后30%
    
    # 交易成本
    consider_cost=True,
    commission_rate=0.0003,
    
    output_dir='output/custom'
)

system = FactorBacktestSystem(config)
# ... 同上
```

---

## 分步骤使用 (高级)

如果你想更细粒度地控制流程:

```python
from Factorsystem import (
    BacktestConfig,
    FactorGenerator,
    FactorProcessor,
    ICAnalyzer,
    PortfolioBuilder,
    PerformanceEvaluator,
    ResultVisualizer
)
import pandas as pd
import torch

# 配置
config = BacktestConfig()

# 加载模型和数据
model = torch.load('output/best_model.pth')
model.eval()
df = pd.read_parquet('output/data.parquet')

# 步骤1: 生成因子
generator = FactorGenerator(model, config)
factor_df = generator.generate_factors(df)

# 步骤2: 处理因子
processor = FactorProcessor(config)
processed_df = processor.process(factor_df)

# 添加收益列
processed_df = pd.merge(
    processed_df, 
    df[['ts_code', 'trade_date', 'y_processed']],
    on=['ts_code', 'trade_date'],
    how='left'
)

# 步骤3: IC分析
ic_analyzer = ICAnalyzer(config)
ic_df = ic_analyzer.calculate_ic(processed_df, 'factor_raw_std', 'y_processed')
ic_stats = ic_analyzer.analyze_ic_statistics(ic_df)

# 步骤4: 构建组合
builder = PortfolioBuilder(config)
portfolios = builder.build_portfolios(processed_df, 'factor_raw_std', 'y_processed')

# 步骤5: 绩效评估
evaluator = PerformanceEvaluator(config)
metrics = evaluator.evaluate_portfolio(portfolios['long_short'])

# 步骤6: 可视化
visualizer = ResultVisualizer(config)
visualizer.create_comprehensive_report(
    portfolios, ic_df, metrics, 'output/custom/plots'
)

print("IC统计:", ic_stats)
print("绩效指标:", metrics)
```

---

## 理解输出指标

### IC指标 (因子预测能力)
- **IC均值**: 平均预测准确度, 一般 >0.03 为好因子
- **ICIR**: IC信息比率, >0.5 表示稳定, >1.0 表示优秀
- **IC胜率**: IC>0的比例, >55% 较好

### 绩效指标 (组合表现)
- **年化收益**: 越高越好, >15% 较好
- **夏普比率**: 风险调整后收益, >1.0 较好, >2.0 优秀
- **最大回撤**: 越小越好, <20% 可接受
- **卡玛比率**: 年化收益/最大回撤, >1.0 较好

### 判断标准
```
优秀因子: IC>0.05, ICIR>0.5, Sharpe>1.5, MaxDD<15%
良好因子: IC>0.03, ICIR>0.3, Sharpe>1.0, MaxDD<20%
一般因子: IC>0.01, ICIR>0.1, Sharpe>0.5, MaxDD<30%
无效因子: IC≈0,   ICIR≈0,   Sharpe<0,   MaxDD>35%
```

---

## 常见问题速查

### Q: 提示找不到ts_code列?
```python
# 手动添加
df['ts_code'] = df['stock_code']  # 或其他股票代码列
```

### Q: 运行很慢?
```python
# 使用快速配置
config = ConfigTemplates.fast_test()

# 或减小批次
config.batch_size = 512
```

### Q: 内存不足?
```python
# 减小批次大小
config.batch_size = 128

# 减少持有期
config.holding_periods = [1, 5]
```

### Q: 想只测试特定因子?
```python
# 指定因子列
config.feature_cols = ['feature_1', 'feature_5', 'feature_10']
```

### Q: 想看分行业IC?
```python
from Factorsystem import ICAnalyzer

ic_analyzer = ICAnalyzer(config)
group_ic = ic_analyzer.calculate_ic_by_group(
    factor_df, 
    group_col='industry_name'
)
print(group_ic.sort_values('ic_mean', ascending=False))
```

---

## 下一步

1. **阅读完整文档**: `BACKTEST_GUIDE.md`
2. **查看示例代码**: `example_backtest.py`
3. **了解预期输出**: `EXPECTED_OUTPUT.md`
4. **自定义配置**: 根据需求调整 `BacktestConfig`

---

## 技术支持

- 详细文档: `BACKTEST_GUIDE.md`
- 示例代码: `example_backtest.py`
- 系统概览: `README.md`

Happy Backtesting! 🚀
