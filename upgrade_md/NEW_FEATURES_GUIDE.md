# QuantClassic 新功能使用指南

## 📋 目录

1. [标签生成器 (LabelGenerator)](#1-标签生成器)
2. [回测系统增强功能](#2-回测系统增强功能)
3. [可视化跨平台支持](#3-可视化跨平台支持)

---

## 1. 标签生成器

### 1.1 概述

`LabelGenerator` 现已集成到 `quantclassic.data_processor` 模块，提供灵活的标签生成功能。

### 1.2 支持的标签类型

- **未来收益率标签** (`return`): 用于回归任务
- **分类标签** (`classification`): 涨/跌/平，适合分类模型
- **排名标签** (`rank`): 截面分位数，适合排序任务

### 1.3 快速开始

#### 方式1: 使用配置对象

```python
from quantclassic.data_processor import LabelGenerator, LabelConfig

# 创建配置
config = LabelConfig(
    stock_col='order_book_id',
    time_col='trade_date',
    price_col='close',
    label_type='return',           # 收益率标签
    return_periods=[1, 5, 10, 20], # 多周期
    return_method='simple',        # simple 或 log
    neutralize=True,               # 是否中性化
    neutralize_method='market'     # market, industry, simstock
)

# 创建生成器
generator = LabelGenerator(config)

# 生成标签
df_with_labels = generator.generate_labels(df, label_name='ret')

# 输出列: ret_1d, ret_5d, ret_10d, ret_20d
```

#### 方式2: 使用便捷函数

```python
from quantclassic.data_processor import generate_future_returns

# 快速生成未来收益率
df = generate_future_returns(
    df,
    stock_col='order_book_id',
    time_col='trade_date',
    price_col='close',
    periods=[1, 5, 10],
    method='simple'
)
```

### 1.4 高级用法

#### 生成分类标签

```python
config = LabelConfig(
    label_type='classification',
    n_classes=3,                   # 涨/平/跌
    class_method='quantile',       # 或 'threshold'
    thresholds=[-0.02, 0.02]       # 阈值法使用
)

generator = LabelGenerator(config)
df = generator.generate_labels(df, label_name='class_label')
```

#### 生成排名标签

```python
config = LabelConfig(
    label_type='rank',
    n_quantiles=10,                # 十分位
    rank_method='quantile'         # 或 'percentile'
)

generator = LabelGenerator(config)
df = generator.generate_labels(df, label_name='rank_label')
```

#### 标签中性化

```python
# 市场中性化（减去市场平均收益）
config = LabelConfig(
    label_type='return',
    return_periods=[1, 5],
    neutralize=True,
    neutralize_method='market'
)

# 行业中性化（需要 industry_name 列）
config = LabelConfig(
    label_type='return',
    return_periods=[1, 5],
    neutralize=True,
    neutralize_method='industry'
)

# SimStock中性化（需要使用FeatureProcessor）
from quantclassic.data_processor import DataPreprocessor, PreprocessConfig, ProcessMethod

preprocess_config = PreprocessConfig()
preprocess_config.add_step(
    name='SimStock标签中性化',
    method=ProcessMethod.SIMSTOCK_LABEL_NEUTRALIZE,
    label_column='ret_1d',
    output_column='alpha_label',
    similarity_threshold=0.7,
    lookback_window=120
)

preprocessor = DataPreprocessor(preprocess_config)
df = preprocessor.fit_transform(df, target_column='ret_1d')
```

### 1.5 标签统计分析

```python
# 获取标签统计信息
stats = generator.get_label_statistics(
    df, 
    label_cols=['ret_1d', 'ret_5d', 'ret_10d']
)
print(stats)
```

---

## 2. 回测系统增强功能

### 2.1 双周调仓频率

现在支持 `biweekly`（双周）调仓频率。

```python
from quantclassic.Factorsystem import BacktestConfig

config = BacktestConfig(
    rebalance_freq='biweekly',  # daily, weekly, biweekly, monthly
    rebalance_day='last'        # last, first
)
```

**支持的调仓频率：**
- `daily`: 每日调仓
- `weekly`: 每周调仓
- `biweekly`: 每两周调仓 ⭐ **新增**
- `monthly`: 每月调仓

### 2.2 基准指数配置

#### 2.2.1 配置基准指数

```python
config = BacktestConfig(
    # 基准指数选择
    benchmark_index='hs300',  # hs300, zz500, zz800, custom
    
    # 股票池限制（可选）
    stock_universe='hs300'    # 限制回测股票池为沪深300成分股
)
```

**支持的基准指数：**
- `hs300`: 沪深300指数
- `zz500`: 中证500指数
- `zz800`: 中证800指数
- `custom`: 自定义基准（需要提供 `custom_benchmark_col`）

#### 2.2.2 获取基准收益率

```python
from quantclassic.Factorsystem import BenchmarkManager

# 方式1: 使用管理器
manager = BenchmarkManager()
hs300_returns = manager.get_benchmark_returns(
    'hs300',
    start_date='2020-01-01',
    end_date='2024-12-31'
)

# 方式2: 使用便捷函数
from quantclassic.Factorsystem import get_benchmark_returns

zz500_returns = get_benchmark_returns(
    'zz500',
    start_date='2020-01-01',
    end_date='2024-12-31'
)
```

#### 2.2.3 计算超额收益

```python
# 计算策略相对基准的超额收益
excess_returns = manager.calculate_excess_returns(
    portfolio_returns,
    benchmark_returns
)
```

#### 2.2.4 获取指数成分股

```python
# 获取沪深300成分股列表
hs300_stocks = manager.get_universe_stocks(
    'hs300',
    date='2024-01-01'  # 可选，默认最新
)
```

### 2.3 完整回测示例

```python
from quantclassic.Factorsystem import BacktestConfig, FactorBacktestSystem

# 创建配置
config = BacktestConfig(
    # 调仓配置
    rebalance_freq='biweekly',
    rebalance_day='last',
    
    # 基准配置
    benchmark_index='zz500',
    stock_universe='zz500',
    
    # 组合配置
    n_groups=10,
    long_ratio=0.2,
    short_ratio=0.2,
    weight_method='equal',
    
    # 其他配置
    consider_cost=True,
    save_plots=True
)

# 运行回测
backtest = FactorBacktestSystem(config)
results = backtest.run_backtest(factor_df, factor_col='factor', return_col='y_processed')
```

---

## 3. 可视化跨平台支持

### 3.1 自动适配中文字体

`ResultVisualizer` 现已自动适配不同操作系统的中文字体。

**支持的操作系统：**

| 操作系统 | 默认字体 | 备选字体 |
|---------|---------|---------|
| macOS | Arial Unicode MS | PingFang SC, STHeiti |
| Linux | WenQuanYi Micro Hei | Noto Sans CJK SC |
| Windows | Microsoft YaHei | SimHei, SimSun |

### 3.2 Linux系统安装中文字体

如果Linux系统中文显示异常，请安装中文字体：

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei

# 或安装 Noto 字体
sudo apt-get install fonts-noto-cjk

# 安装后清除matplotlib缓存
rm -rf ~/.cache/matplotlib
```

### 3.3 使用示例

```python
from quantclassic.Factorsystem import ResultVisualizer, BacktestConfig

config = BacktestConfig(
    plot_style='seaborn',
    figure_size=(12, 6),
    dpi=100
)

visualizer = ResultVisualizer(config)

# 绘制累计收益曲线
visualizer.plot_cumulative_returns(
    portfolio_df,
    return_col='portfolio_return',
    benchmark_col='benchmark_return',  # 可选
    title='策略累计收益 vs 基准',
    save_path='output/cumulative_returns.png'
)
```

---

## 4. 完整工作流示例

### 4.1 从数据加载到回测

```python
import pandas as pd
from quantclassic.data_processor import LabelGenerator, LabelConfig
from quantclassic.Factorsystem import (
    BacktestConfig, FactorBacktestSystem, BenchmarkManager
)

# ========== 步骤1: 加载数据 ==========
df = pd.read_parquet('data/stock_data.parquet')

# ========== 步骤2: 生成标签 ==========
label_config = LabelConfig(
    label_type='return',
    return_periods=[1, 5, 10],
    neutralize=True,
    neutralize_method='market'
)

label_gen = LabelGenerator(label_config)
df = label_gen.generate_labels(df, label_name='ret')

# ========== 步骤3: 训练模型并生成因子 ==========
# ... 你的模型训练代码 ...
# factor_df 应包含: order_book_id, trade_date, factor_value, ret_1d

# ========== 步骤4: 配置回测 ==========
backtest_config = BacktestConfig(
    # 调仓策略
    rebalance_freq='biweekly',
    rebalance_day='last',
    
    # 基准
    benchmark_index='zz500',
    stock_universe='zz500',
    
    # 组合
    n_groups=10,
    long_ratio=0.2,
    short_ratio=0.2,
    
    # 成本
    consider_cost=True,
    commission_rate=0.0003,
    slippage_rate=0.001,
    
    # 输出
    save_plots=True,
    generate_excel=True
)

# ========== 步骤5: 运行回测 ==========
backtest = FactorBacktestSystem(backtest_config)
results = backtest.run_backtest(
    factor_df,
    factor_col='factor_value',
    return_col='ret_1d'
)

# ========== 步骤6: 查看结果 ==========
print("回测结果:")
print(f"年化收益率: {results['annual_return']:.2%}")
print(f"年化波动率: {results['annual_volatility']:.2%}")
print(f"夏普比率: {results['sharpe_ratio']:.2f}")
print(f"最大回撤: {results['max_drawdown']:.2%}")
print(f"信息比率: {results['information_ratio']:.2f}")
```

---

## 5. API参考

### 5.1 LabelConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| stock_col | str | 'order_book_id' | 股票代码列 |
| time_col | str | 'trade_date' | 时间列 |
| price_col | str | 'close' | 价格列 |
| label_type | str | 'return' | 标签类型 |
| return_periods | List[int] | [1,5,10,20] | 收益率周期 |
| return_method | str | 'simple' | 收益率计算方法 |
| n_classes | int | 3 | 分类数量 |
| n_quantiles | int | 10 | 分位数数量 |
| neutralize | bool | False | 是否中性化 |

### 5.2 BacktestConfig 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| rebalance_freq | str | 'monthly' | 调仓频率 |
| benchmark_index | str | None | 基准指数 |
| stock_universe | str | None | 股票池限制 |
| custom_benchmark_col | str | None | 自定义基准列 |

### 5.3 BenchmarkManager 方法

- `get_benchmark_returns(name, start, end)`: 获取基准收益率
- `calculate_excess_returns(portfolio, benchmark)`: 计算超额收益
- `get_universe_stocks(name, date)`: 获取成分股列表

---

## 6. 注意事项

1. **标签生成**: 
   - 未来收益率会产生缺失值（最后N个周期），需要在训练前删除
   - SimStock中性化需要足够的历史数据（默认252天）

2. **基准数据**:
   - 需要配置数据源（米筐/Tushare/AkShare）或提供本地文件
   - 自动尝试多个数据源，如果都失败会返回零收益率（有警告）

3. **跨平台字体**:
   - Linux系统首次使用需要安装中文字体
   - 清除matplotlib缓存后重启Python才能生效

4. **双周调仓**:
   - 使用ISO周数除以2实现，可能在年初有边界问题
   - 建议配合 `rebalance_day='last'` 使用

---

## 7. 更新日志

### v1.1.0 (2025-11-20)

**新增功能：**
- ✅ 标签生成器 (`LabelGenerator`)
- ✅ 双周调仓支持 (`biweekly`)
- ✅ 基准指数管理 (`BenchmarkManager`)
- ✅ 跨平台中文字体自动适配

**改进：**
- ✅ `BacktestConfig` 新增基准和股票池配置
- ✅ `ResultVisualizer` 跨平台支持
- ✅ 文档和示例更新

---

## 8. 常见问题

### Q1: 如何使用自定义基准？

```python
config = BacktestConfig(
    benchmark_index='custom',
    custom_benchmark_col='my_benchmark_return'
)

# 在 factor_df 中添加 my_benchmark_return 列
```

### Q2: 如何限制回测只使用某个指数的成分股？

```python
config = BacktestConfig(
    stock_universe='zz500'  # 只使用中证500成分股
)

# 或在数据准备时手动过滤
from quantclassic.Factorsystem import BenchmarkManager

manager = BenchmarkManager()
zz500_stocks = manager.get_universe_stocks('zz500')
df = df[df['order_book_id'].isin(zz500_stocks)]
```

### Q3: Linux中文显示乱码怎么办？

```bash
# 安装字体
sudo apt-get install fonts-wqy-microhei

# 清除缓存
rm -rf ~/.cache/matplotlib

# 重启Python
```

---

## 9. 联系与反馈

如有问题或建议，请提交 Issue 或 PR。

**文档更新**: 2025-11-20
