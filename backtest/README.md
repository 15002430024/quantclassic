# Factorsystem - 因子回测系统

工程化的量化因子回测框架，提供从因子生成到绩效评估的完整流程。

## 🚀 快速开始

```python
from Factorsystem import BacktestConfig, FactorBacktestSystem

# 创建配置
config = BacktestConfig(output_dir='output/backtest', save_plots=True)

# 初始化系统
system = FactorBacktestSystem(config)

# 加载模型和数据
model = system.load_model('output/best_model.pth')
df = pd.read_parquet('output/data.parquet')

# 运行回测
results = system.run_backtest(df)
```

## 📦 核心组件

```
FactorBacktestSystem (主控制器)
    ├── FactorGenerator      # 因子生成器
    ├── FactorProcessor      # 因子处理器  
    ├── PortfolioBuilder     # 组合构建器
    ├── ICAnalyzer          # IC分析器
    ├── PerformanceEvaluator # 绩效评估器
    └── ResultVisualizer    # 结果可视化器
```

## ✨ 核心特性

- ✅ 模块化设计，职责清晰
- ✅ 配置驱动，灵活可扩展
- ✅ 完整的回测流程
- ✅ 丰富的绩效指标 (IC/ICIR/夏普/回撤等)
- ✅ 专业图表自动生成
- ✅ 支持多因子/多策略回测
- ✅ 工程化日志和异常处理

## 📊 输出内容

### 绩效指标
- **收益**: 累计收益、年化收益
- **风险**: 波动率、最大回撤、下行风险
- **风险调整**: 夏普比率、卡玛比率、索提诺比率
- **IC指标**: IC均值、ICIR、IC胜率
- **统计**: 胜率、盈亏比、VaR、CVaR

### 可视化图表
- 累计收益曲线
- 回撤曲线
- IC时间序列
- IC分布直方图
- 分组收益柱状图
- 多空组合表现
- 月度收益热力图

## 📖 文档

- **BACKTEST_GUIDE.md** - 完整使用指南
- **example_backtest.py** - 5个详细示例

## 🔧 配置模板

```python
from Factorsystem import ConfigTemplates

# 快速测试
config = ConfigTemplates.fast_test()

# 详细分析
config = ConfigTemplates.detailed_analysis()

# 生产环境
config = ConfigTemplates.production()
```

## 📁 文件结构

```
Factorsystem/
├── __init__.py              # 包初始化
├── backtest_config.py       # 配置管理
├── factor_generator.py      # 因子生成
├── factor_processor.py      # 因子处理
├── portfolio_builder.py     # 组合构建
├── ic_analyzer.py          # IC分析
├── performance_evaluator.py # 绩效评估
├── result_visualizer.py    # 结果可视化
├── backtest_system.py      # 主控制器
├── example_backtest.py     # 使用示例
├── BACKTEST_GUIDE.md       # 使用指南
└── README.md              # 本文件
```

## 🎯 使用场景

1. **单因子回测**: 测试单个因子的有效性
2. **多因子比较**: 对比多个因子的表现
3. **策略优化**: 测试不同参数配置
4. **绩效归因**: 分析收益来源
5. **风险监控**: 评估策略风险

## ⚙️ 依赖

```bash
pip install pandas numpy scipy scikit-learn torch matplotlib seaborn tqdm
```

## 💡 示例

### 基础回测
```python
results = system.run_backtest(df)
print(f"IC均值: {results['ic_stats']['ic_mean']:.4f}")
print(f"夏普比率: {results['performance_metrics']['long_short']['sharpe_ratio']:.4f}")
```

### 自定义配置
```python
config = BacktestConfig(
    n_groups=10,
    rebalance_freq='monthly',
    weight_method='equal',
    industry_neutral=True,
    consider_cost=True
)
```

### 分步骤执行
```python
# 1. 生成因子
factor_df = factor_generator.generate_factors(df)

# 2. 处理因子  
processed_df = factor_processor.process(factor_df)

# 3. IC分析
ic_df = ic_analyzer.calculate_ic(processed_df)

# 4. 构建组合
portfolios = portfolio_builder.build_portfolios(processed_df)

# 5. 评估绩效
metrics = performance_evaluator.evaluate_portfolio(portfolios['long_short'])
```

## 📝 版本

- **v1.0.0** - 初始版本
- 日期: 2025-11-19

## 🔗 相关模块

- **data_loader**: 数据加载（独立）
- **data_processor**: 数据预处理（互补）
- **factor.py**: 原始实现参考

---

详细文档请参考 **BACKTEST_GUIDE.md**
