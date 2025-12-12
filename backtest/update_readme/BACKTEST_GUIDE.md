# 因子回测系统使用指南

## 目录
- [系统简介](#系统简介)
- [快速开始](#快速开始)
- [核心组件](#核心组件)
- [配置说明](#配置说明)
- [使用示例](#使用示例)
- [预期输出](#预期输出)
- [高级功能](#高级功能)
- [常见问题](#常见问题)

---

## 系统简介

因子回测系统（Factorsystem）是一个工程化的量化因子测试框架，提供从因子生成到绩效评估的完整流程。

### 核心特性

- ✅ **模块化设计**: 7大核心组件，职责清晰，易于扩展
- ✅ **配置驱动**: 丰富的配置参数,支持多种回测策略
- ✅ **完整流程**: 因子生成 → 处理 → IC分析 → 组合构建 → 绩效评估 → 可视化
- ✅ **专业指标**: IC/ICIR、夏普比率、最大回撤、信息比率等
- ✅ **可视化**: 自动生成累计收益、IC时序、分组收益等专业图表
- ✅ **工程化**: 异常处理、日志系统、结果保存完备

### 系统架构

```
FactorBacktestSystem (主控制器)
    ├── FactorGenerator (因子生成器)
    ├── FactorProcessor (因子处理器)
    ├── PortfolioBuilder (组合构建器)
    ├── ICAnalyzer (IC分析器)
    ├── PerformanceEvaluator (绩效评估器)
    └── ResultVisualizer (结果可视化器)
```

---

## 快速开始

### 安装依赖

```bash
pip install pandas numpy scipy scikit-learn torch matplotlib seaborn tqdm
```

### 最简示例

```python
from Factorsystem import BacktestConfig, FactorBacktestSystem
import pandas as pd
import torch

# 1. 创建配置
config = BacktestConfig(
    data_dir='output',
    output_dir='output/backtest_results',
    save_plots=True
)

# 2. 初始化系统
backtest_system = FactorBacktestSystem(config)

# 3. 加载模型和数据
model = backtest_system.load_model('output/best_model.pth')
df = pd.read_parquet('output/train_data_final_01.parquet')

# 4. 运行回测
results = backtest_system.run_backtest(
    data_df=df,
    factor_col='factor_raw_std',
    return_col='y_processed'
)

# 5. 查看结果
print(f"IC均值: {results['ic_stats']['ic_mean']:.4f}")
print(f"夏普比率: {results['performance_metrics']['long_short']['sharpe_ratio']:.4f}")
```

---

## 核心组件

### 1. BacktestConfig (配置管理)

**职责**: 管理所有回测参数

**核心配置项**:

```python
config = BacktestConfig(
    # 数据路径
    data_dir='output',
    output_dir='output/backtest',
    
    # 因子生成
    batch_size=256,
    window_size=40,
    
    # 因子处理
    winsorize_method='quantile',     # 去极值: quantile/mad/std
    standardize_method='zscore',     # 标准化: zscore/minmax/rank
    industry_neutral=False,          # 行业中性化
    market_value_neutral=False,      # 市值中性化
    
    # 组合构建
    n_groups=10,                     # 分组数
    rebalance_freq='monthly',        # 换仓频率: daily/weekly/monthly
    weight_method='equal',           # 权重: equal/value_weight/factor_weight
    long_ratio=0.2,                  # 多头比例
    short_ratio=0.2,                 # 空头比例
    
    # IC分析
    ic_method='spearman',            # IC方法: pearson/spearman
    holding_periods=[1, 5, 10, 20],  # 多期IC
    
    # 交易成本
    consider_cost=False,
    commission_rate=0.0003,
    
    # 输出
    save_plots=True,
    generate_excel=True
)
```

**预设模板**:

```python
from Factorsystem import ConfigTemplates

# 快速测试配置
config = ConfigTemplates.fast_test()

# 详细分析配置
config = ConfigTemplates.detailed_analysis()

# 生产环境配置
config = ConfigTemplates.production()
```

### 2. FactorGenerator (因子生成器)

**职责**: 基于训练好的模型生成原始因子

**核心方法**:

```python
from Factorsystem import FactorGenerator

# 初始化
factor_gen = FactorGenerator(model, config)

# 生成多维因子
factor_df = factor_gen.generate_factors(df)
# 输出: DataFrame包含factor_0, factor_1, ..., factor_n列

# 生成单一因子(聚合多维)
single_factor_df = factor_gen.generate_single_factor(
    df, 
    aggregation='mean'  # mean/sum/first/pca
)
# 输出: DataFrame包含factor_raw列

# 批量生成
factor_dict = factor_gen.batch_generate_factors({
    'train': train_df,
    'test': test_df
})
```

### 3. FactorProcessor (因子处理器)

**职责**: 对原始因子进行标准化处理

**处理流程**:

```
原始因子 → 去极值 → 缺失值填充 → 标准化 → 中性化 → 最终因子
```

**核心方法**:

```python
from Factorsystem import FactorProcessor

processor = FactorProcessor(config)

# 完整处理流程
processed_df = processor.process(factor_df)
# 自动生成: factor_raw_winsorized, factor_raw_filled, factor_raw_std等列

# 单独处理
# 去极值
winsorized = processor.winsorize(df, 'factor_raw', method='quantile')

# 标准化
standardized = processor.standardize(df, 'factor_raw', method='zscore')

# 中性化
neutralized = processor.neutralize(df, 'factor_raw')
```

### 4. PortfolioBuilder (组合构建器)

**职责**: 构建多空投资组合

**核心方法**:

```python
from Factorsystem import PortfolioBuilder

builder = PortfolioBuilder(config)

# 构建组合
portfolios = builder.build_portfolios(
    factor_df, 
    factor_col='factor_raw_std',
    return_col='y_true'
)

# 返回字典:
# {
#     'long': 多头组合DataFrame,
#     'short': 空头组合DataFrame,
#     'long_short': 多空组合DataFrame,
#     'groups': 分组统计DataFrame
# }

# 考虑换仓的回测
backtest_df = builder.backtest_with_rebalance(
    factor_df,
    factor_col='factor_raw_std',
    return_col='y_true'
)
```

### 5. ICAnalyzer (IC分析器)

**职责**: 分析因子与收益的相关性

**核心方法**:

```python
from Factorsystem import ICAnalyzer

ic_analyzer = ICAnalyzer(config)

# 日度IC
ic_df = ic_analyzer.calculate_ic(
    factor_df,
    factor_col='factor_raw_std',
    return_col='y_true'
)
# 包含: ic, rank_ic, p_value, cum_ic等列

# IC统计
ic_stats = ic_analyzer.analyze_ic_statistics(ic_df)
# 包含: ic_mean, ic_std, icir, ic_win_rate等

# 多期IC
multi_ic = ic_analyzer.calculate_multi_period_ic(factor_df)
# {1: ic_df_1d, 5: ic_df_5d, ...}

# IC衰减分析
decay_df = ic_analyzer.calculate_ic_decay(factor_df, max_period=20)

# 分组IC
group_ic = ic_analyzer.calculate_ic_by_group(
    factor_df, 
    group_col='industry_name'
)

# 月度IC
monthly_ic = ic_analyzer.monthly_ic_analysis(ic_df)
```

### 6. PerformanceEvaluator (绩效评估器)

**职责**: 计算组合绩效指标

**核心方法**:

```python
from Factorsystem import PerformanceEvaluator

evaluator = PerformanceEvaluator(config)

# 评估组合
metrics = evaluator.evaluate_portfolio(
    portfolio_df,
    return_col='portfolio_return',
    benchmark_col=None  # 可选基准
)

# 返回指标:
{
    # 收益指标
    'total_return': 累计收益,
    'annual_return': 年化收益,
    
    # 风险指标
    'volatility': 波动率,
    'annual_volatility': 年化波动率,
    'max_drawdown': 最大回撤,
    'downside_risk': 下行风险,
    
    # 风险调整收益
    'sharpe_ratio': 夏普比率,
    'calmar_ratio': 卡玛比率,
    'sortino_ratio': 索提诺比率,
    
    # 统计指标
    'win_rate': 胜率,
    'profit_loss_ratio': 盈亏比,
    'skewness': 偏度,
    'kurtosis': 峰度,
    'var_95': VaR,
    'cvar_95': CVaR
}

# 滚动绩效
rolling_df = evaluator.rolling_performance(portfolio_df, window=60)

# 月度绩效
monthly_df = evaluator.monthly_performance(portfolio_df)

# 年度绩效
yearly_df = evaluator.yearly_performance(portfolio_df)
```

### 7. ResultVisualizer (结果可视化器)

**职责**: 生成专业图表

**核心方法**:

```python
from Factorsystem import ResultVisualizer

visualizer = ResultVisualizer(config)

# 累计收益曲线
visualizer.plot_cumulative_returns(
    portfolio_df,
    save_path='output/cumulative_returns.png'
)

# 回撤曲线
visualizer.plot_drawdown(
    portfolio_df,
    save_path='output/drawdown.png'
)

# IC时间序列
visualizer.plot_ic_series(
    ic_df,
    save_path='output/ic_series.png'
)

# IC分布
visualizer.plot_ic_distribution(
    ic_df,
    save_path='output/ic_distribution.png'
)

# 分组收益
visualizer.plot_group_returns(
    group_df,
    save_path='output/group_returns.png'
)

# 多空表现
visualizer.plot_long_short_performance(
    ls_df,
    save_path='output/long_short.png'
)

# 月度收益热力图
visualizer.plot_monthly_returns_heatmap(
    portfolio_df,
    save_path='output/monthly_heatmap.png'
)

# 综合报告(生成所有图表)
visualizer.create_comprehensive_report(
    portfolios, ic_df, metrics, 'output/plots'
)
```

---

## 配置说明

### 完整配置参数

```python
BacktestConfig(
    # ========== 数据路径 ==========
    data_dir='output',
    output_dir='output/backtest',
    model_path=None,
    
    # ========== 因子生成 ==========
    batch_size=256,
    window_size=40,
    device='cuda',  # 或'cpu'
    feature_cols=None,  # None时自动识别
    label_col='y_processed',
    
    # ========== 因子处理 ==========
    # 去极值
    winsorize_method='quantile',  # quantile/mad/std
    winsorize_quantiles=(0.025, 0.975),
    mad_threshold=3.0,
    std_threshold=3.0,
    
    # 标准化
    standardize_method='zscore',  # zscore/minmax/rank
    
    # 中性化
    industry_neutral=False,
    industry_col='industry_name',
    market_value_neutral=False,
    market_value_col='market_value',
    
    # 缺失值
    fillna_method='median',  # mean/median/forward/zero
    
    # ========== 组合构建 ==========
    n_groups=10,
    rebalance_freq='monthly',  # daily/weekly/monthly
    rebalance_day='last',  # first/last/middle
    weight_method='equal',  # equal/value_weight/factor_weight
    long_ratio=0.2,
    short_ratio=0.2,
    max_stock_weight=0.1,
    min_stock_weight=0.0,
    max_industry_weight=0.3,
    
    # ========== IC分析 ==========
    ic_method='spearman',  # pearson/spearman
    holding_periods=[1, 5, 10, 20],
    ic_significance_level=0.05,
    
    # ========== 绩效评估 ==========
    annual_factor=252,
    risk_free_rate=0.03,
    benchmark_col=None,
    rolling_window=60,
    
    # ========== 交易成本 ==========
    consider_cost=False,
    commission_rate=0.0003,
    stamp_tax_rate=0.001,
    slippage_rate=0.001,
    
    # ========== 可视化 ==========
    plot_style='seaborn',  # seaborn/ggplot/default
    figure_size=(12, 6),
    dpi=100,
    save_plots=True,
    plot_format='png',
    
    # ========== 报告 ==========
    generate_pdf=False,
    generate_excel=True,
    report_title="因子回测报告",
    report_author="QuantClassic",
    
    # ========== 性能优化 ==========
    n_jobs=0,
    use_cache=False,
    cache_dir='cache/backtest',
    
    # ========== 日志 ==========
    log_level='INFO',
    log_file=None,
    console_log=True
)
```

---

## 使用示例

详见 `example_backtest.py`，包含:
- 示例1: 基础回测流程
- 示例2: 使用预设配置模板
- 示例3: 自定义因子处理流程
- 示例4: 分步骤运行回测
- 示例5: 多因子回测

---

## 预期输出

### 1. 控制台输出

```
================================================================================
开始因子回测
================================================================================

步骤1: 生成因子
  生成样本数: 125,432
  生成因子数: 125,432 条记录, 16 个因子维度
  因子统计信息:
    factor_0: 均值=0.0023, 标准差=1.2456, ...

步骤2: 处理因子
  处理因子: factor_0
  处理因子: factor_1
  ...
  因子处理完成

步骤3: IC分析
  IC计算完成: 245 个交易日
  IC统计指标:
    IC均值: 0.0523
    Rank IC均值: 0.0487
    IC标准差: 0.1234
    ICIR: 0.4237
    Rank ICIR: 0.3946
    IC胜率: 58.37%
    绝对IC均值: 0.0892
    t统计量: 6.6342

步骤4: 构建投资组合
  分组完成: 10 组
  投资组合构建完成

步骤5: 绩效评估
  评估 long 组合:
    累计收益: 45.23%
    年化收益: 18.34%
    年化波动率: 15.67%
    夏普比率: 0.9782
    最大回撤: -12.34%
    卡玛比率: 1.4865
    胜率: 54.29%

  评估 short 组合:
    ...

  评估 long_short 组合:
    累计收益: 67.89%
    年化收益: 25.43%
    年化波动率: 18.23%
    夏普比率: 1.2304
    最大回撤: -15.67%
    卡玛比率: 1.6221
    胜率: 56.12%

步骤6: 生成可视化图表
  报告图表已保存到: output/backtest/plots

步骤7: 保存结果
  因子已保存: output/backtest/factors.csv
  IC分析已保存: output/backtest/ic_analysis.csv
  long组合已保存: output/backtest/portfolio_long.csv
  short组合已保存: output/backtest/portfolio_short.csv
  long_short组合已保存: output/backtest/portfolio_long_short.csv
  绩效指标已保存: output/backtest/performance_metrics.xlsx

================================================================================
回测完成!
================================================================================
```

### 2. 文件输出

```
output/backtest/
├── factors.csv                      # 因子数据
├── ic_analysis.csv                  # IC分析结果
├── portfolio_long.csv               # 多头组合
├── portfolio_short.csv              # 空头组合
├── portfolio_long_short.csv         # 多空组合
├── performance_metrics.xlsx         # 绩效指标
└── plots/                           # 图表目录
    ├── cumulative_returns.png       # 累计收益曲线
    ├── drawdown.png                 # 回撤曲线
    ├── ic_series.png                # IC时间序列
    ├── ic_distribution.png          # IC分布
    ├── group_returns.png            # 分组收益
    └── long_short_performance.png   # 多空表现
```

### 3. 返回结果字典

```python
results = {
    'raw_factors': DataFrame,          # 原始因子
    'processed_factors': DataFrame,    # 处理后因子
    'ic_df': DataFrame,                # IC时间序列
    'ic_stats': {                      # IC统计
        'ic_mean': 0.0523,
        'icir': 0.4237,
        ...
    },
    'portfolios': {                    # 组合
        'long': DataFrame,
        'short': DataFrame,
        'long_short': DataFrame,
        'groups': DataFrame
    },
    'performance_metrics': {           # 绩效指标
        'long': {...},
        'short': {...},
        'long_short': {...}
    }
}
```

---

## 高级功能

### 1. 多期IC分析

```python
multi_ic = ic_analyzer.calculate_multi_period_ic(
    factor_df,
    factor_col='factor_raw_std',
    return_col='y_true'
)

# 查看不同持有期的IC
for period, ic_df in multi_ic.items():
    stats = ic_analyzer.analyze_ic_statistics(ic_df)
    print(f"{period}日IC: {stats['ic_mean']:.4f}")
```

### 2. IC衰减分析

```python
decay_df = ic_analyzer.calculate_ic_decay(
    factor_df,
    max_period=20
)

# 绘制衰减曲线
import matplotlib.pyplot as plt
plt.plot(decay_df['period'], decay_df['ic'])
plt.xlabel('持有期')
plt.ylabel('IC')
plt.title('IC衰减曲线')
plt.show()
```

### 3. 分行业回测

```python
# 按行业分组计算IC
group_ic = ic_analyzer.calculate_ic_by_group(
    factor_df,
    factor_col='factor_raw_std',
    return_col='y_true',
    group_col='industry_name'
)

print(group_ic.sort_values('ic_mean', ascending=False))
```

### 4. 滚动窗口分析

```python
# 滚动绩效
rolling_df = evaluator.rolling_performance(
    portfolio_df,
    window=60
)

# 绘制滚动夏普比率
plt.plot(rolling_df['rolling_sharpe'])
plt.title('60日滚动夏普比率')
plt.show()
```

### 5. 考虑交易成本

```python
config = BacktestConfig(
    consider_cost=True,
    commission_rate=0.0003,
    stamp_tax_rate=0.001,
    slippage_rate=0.001
)

# 使用换仓回测
backtest_df = portfolio_builder.backtest_with_rebalance(
    factor_df,
    factor_col='factor_raw_std',
    return_col='y_true'
)
```

---

## 常见问题

### Q1: 如何处理缺失的ts_code列?

A: 系统会自动尝试从`order_book_id`, `code`, `symbol`等列推断，或手动添加:
```python
df['ts_code'] = df['stock_code']
```

### Q2: 如何只使用特定因子列?

A: 在配置中指定:
```python
config.feature_cols = ['factor_0', 'factor_5', 'factor_10']
```

### Q3: 如何自定义分组数量?

A: 
```python
config.n_groups = 5  # 五分组
```

### Q4: 如何修改换仓频率?

A:
```python
config.rebalance_freq = 'weekly'  # 每周换仓
config.rebalance_day = 'last'     # 周最后一天
```

### Q5: 生成的图表不显示中文怎么办?

A: 修改matplotlib配置:
```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

### Q6: 如何只运行IC分析不构建组合?

A: 分步骤运行,参考示例4

### Q7: 内存不足怎么办?

A: 
- 减小`batch_size`
- 减少`holding_periods`
- 分批处理数据

### Q8: 如何导出Excel格式的报告?

A:
```python
config.generate_excel = True
```

---

## 与其他模块的关系

- **与data_loader的关系**: data_loader负责原始数据获取，Factorsystem负责因子回测，两者独立
- **与data_processor的关系**: data_processor负责数据预处理，Factorsystem的FactorProcessor负责因子处理，功能互补
- **与factor.py的关系**: factor.py包含完整流程，Factorsystem将其模块化封装

---

## 版本信息

- 版本: v1.0.0
- 作者: QuantClassic
- 更新日期: 2025-11-19

---

## 联系支持

如有问题，请查看:
1. `example_backtest.py` - 详细示例
2. 各模块的docstring - 函数说明
3. `factor.py` - 原始实现参考
