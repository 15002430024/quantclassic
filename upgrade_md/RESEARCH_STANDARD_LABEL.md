# 研报标准标签生成方案

## 📋 概述

根据**财通证券**和**海通证券**的深度学习因子挖掘研报，本框架实现了业界标准的价格使用逻辑，严格区分三个阶段的价格类型。

---

## 🎯 三阶段价格使用标准

### 1. 特征计算 (Model Input Features)

**目标**: 提取历史市场状态，作为模型的输入 $X$

**使用的价格**: **复权后的 OHLC + VWAP**

```python
# 历史收益率特征（向后看）
ret_1d = close_t / close_{t-1} - 1
ret_5d = close_t / close_{t-5} - 1
```

**关键点**:
- 使用 `pct_change(period)` 或 `shift(period)` 计算历史收益率
- 通常对价格进行对数变换或截面标准化
- 这些是模型的输入特征 (X)，不是预测目标

---

### 2. 标签生成 (Training Label / Target)

**目标**: 定义模型需要预测的"正确答案" $Y$

**使用的价格**: **T+1 日价格作为基准（分母）**

#### 📐 研报标准公式

$$
label_t = \frac{price_{t+n}}{price_{t+1}} - 1
$$

**解读**:
- **分子**: $price_{t+n}$ - 未来第 n 天的价格（持有期结束）
- **分母**: $price_{t+1}$ - 次日价格（入场价格）
- **含义**: 假设在 T 日收盘预测，T+1 日开盘交易，持有 n-1 天

#### 🔍 与传统方式的区别

| 方式 | 公式 | 含义 | 问题 |
|------|------|------|------|
| **传统** | $\frac{price_{t+n}}{price_t} - 1$ | T日收盘预测，T日收盘交易 | ❌ T日收盘价无法成交 |
| **研报** | $\frac{price_{t+n}}{price_{t+1}} - 1$ | T日收盘预测，T+1日开盘交易 | ✅ 符合真实交易逻辑 |

#### 💻 代码实现

```python
from quantclassic.data_processor.label_generator import LabelConfig, LabelGenerator

# 研报标准配置
label_config = LabelConfig(
    stock_col='order_book_id',
    time_col='trade_date',
    price_col='close',          # 分子：未来价格（T+n日）
    base_price_col='close',     # 分母：基准价格（T+1日）⭐ 关键！
    label_type='return',
    return_periods=[1, 5, 10],  # 生成多周期标签
    return_method='simple'
)

label_gen = LabelGenerator(label_config)
df = label_gen.generate_labels(df, label_name='y_ret')
# 生成: y_ret_1d, y_ret_5d, y_ret_10d
```

#### 📊 实际计算逻辑

```python
# 研报标准（base_price_col='close'）
future_price = df.groupby('stock')['close'].shift(-10)  # T+10日价格
base_price = df.groupby('stock')['close'].shift(-1)    # T+1日价格（基准）
y_ret_10d = (future_price - base_price) / base_price

# 相当于：
y_ret_10d = close[t+10] / close[t+1] - 1
```

---

### 3. 回测/因子绩效计算 (Backtest & Execution)

**目标**: 计算策略的实际资金收益曲线（P&L）

**使用的价格**: **次日 VWAP（成交量加权平均价格）**

#### 📐 回测收益计算

$$
return_t = weight_t \times \frac{vwap_{t+1}}{vwap_t} - 1
$$

**关键点**:
- 避免使用收盘价（无法成交）
- 使用 VWAP 模拟机构大资金的真实成交成本
- 财通证券研报：周调仓使用"周一均价"撮合
- 海通证券研报：明确假定以"次日均价"调仓

#### 💻 数据提取配置

```python
from quantclassic.data_loader.config_manager import DataFieldsConfig

# 添加 VWAP 字段
fields_config = DataFieldsConfig(
    price_fields=[
        'open', 'high', 'low', 'close', 'volume', 'total_turnover',
        'limit_up', 'limit_down', 'num_trades'
    ],
    include_vwap=True  # ⭐ 启用 VWAP 获取
)
```

---

## 🔧 完整配置示例

### Notebook 配置

```python
# 步骤1: 数据提取（包含 VWAP）
from quantclassic.data_loader.config_manager import (
    TimeConfig, DataSourceConfig, UniverseConfig, 
    DataFieldsConfig, StorageConfig, FeatureConfig
)

# 字段配置
fields_config = DataFieldsConfig(
    price_fields=['open', 'high', 'low', 'close', 'volume', 'total_turnover'],
    include_vwap=True  # 获取 VWAP
)

# 步骤2: 标签生成（研报标准）
from quantclassic.data_processor.label_generator import LabelConfig, LabelGenerator

label_config = LabelConfig(
    stock_col='order_book_id',
    time_col='trade_date',
    price_col='close',       # 未来价格（分子）
    base_price_col='close',  # T+1 日基准价格（分母）
    label_type='return',
    return_periods=[1, 5, 10, 20],
    return_method='simple'
)

label_gen = LabelGenerator(label_config)
df = label_gen.generate_labels(df, label_name='y_ret')
# 生成: y_ret_1d, y_ret_5d, y_ret_10d, y_ret_20d
```

---

## 📊 数据字段对照表

| 用途 | 字段名 | 计算方式 | 说明 |
|------|--------|----------|------|
| **特征** | `ret_1d` | `close_t / close_{t-1} - 1` | 历史1日收益率（输入） |
| **特征** | `ret_5d` | `close_t / close_{t-5} - 1` | 历史5日收益率（输入） |
| **标签** | `y_ret_1d` | `close_{t+1} / close_{t+1} - 1` | 当日持有收益（目标） |
| **标签** | `y_ret_5d` | `close_{t+5} / close_{t+1} - 1` | 持有5天收益（目标） |
| **标签** | `y_ret_10d` | `close_{t+10} / close_{t+1} - 1` | 持有10天收益（目标） |
| **回测** | `vwap` | 米筐API `get_vwap()` | 成交量加权平均价格 |

---

## ⚠️ 常见错误

### ❌ 错误示例1：标签使用T日价格

```python
# ❌ 错误：分母使用T日价格
label_config = LabelConfig(
    price_col='close',
    base_price_col=None,  # None 表示使用T日价格
    return_periods=[10]
)
# 生成: y_ret_10d = close[t+10] / close[t] - 1
# 问题：假设T日收盘就能交易，不符合实际
```

### ✅ 正确示例：标签使用T+1日价格

```python
# ✅ 正确：分母使用T+1日价格
label_config = LabelConfig(
    price_col='close',
    base_price_col='close',  # 使用T+1日收盘价
    return_periods=[10]
)
# 生成: y_ret_10d = close[t+10] / close[t+1] - 1
# 含义：T日预测，T+1日开盘买入，T+10日卖出
```

---

### ❌ 错误示例2：回测使用收盘价

```python
# ❌ 错误：回测使用收盘价
portfolio_return = weights * (close[t+1] / close[t] - 1)
# 问题：大资金无法按收盘价成交
```

### ✅ 正确示例：回测使用VWAP

```python
# ✅ 正确：回测使用次日VWAP
portfolio_return = weights * (vwap[t+1] / vwap[t] - 1)
# 含义：T日收盘获得信号，T+1日按VWAP成交
```

---

## 📚 参考文献

1. **财通证券** - 深度学习因子挖掘系统
   - 标签定义: $label_t = neutralize(price_{t+11}/price_{t+1} - 1)$
   - 回测撮合: 按"周一均价 (VWAP)"成交

2. **海通证券** - 深度学习因子挖掘框架
   - 特征预处理: 截面标准化效果最优
   - 回测假设: 以"次日均价 (VWAP)"调仓

---

## 🔄 向后兼容性

为了向后兼容，`base_price_col` 参数默认为 `None`：

```python
# 默认配置（传统方式）
label_config = LabelConfig(
    price_col='close',
    base_price_col=None  # 默认值，使用T日价格
)
# 生成: y_ret_10d = close[t+10] / close[t] - 1

# 研报标准（推荐）
label_config = LabelConfig(
    price_col='close',
    base_price_col='close'  # 显式指定使用T+1日价格
)
# 生成: y_ret_10d = close[t+10] / close[t+1] - 1
```

---

## 📞 技术支持

如有疑问，请参考：
- `quantclassic/data_processor/label_generator.py` - 标签生成实现
- `quantclassic/data_loader/data_fetcher.py` - VWAP 数据获取
- `jupyterlab/vae.ipynb` - 完整示例

---

**最后更新**: 2025-11-20
**版本**: v2.0 - 研报标准实现
