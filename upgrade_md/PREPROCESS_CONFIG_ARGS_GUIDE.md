# 预处理配置 - 完整 Args 参考指南

## 概述

本指南详细说明了 quantclassic 预处理配置系统中所有配置类的参数（Args）及其用法。

**相关文件：**
- `data_processor/preprocess_config.py` - 配置类定义
- `data_processor/data_preprocessor.py` - 预处理器实现

---

## 快速索引

| 配置类 | 用途 | 参数数量 | 说明 |
|--------|------|---------|------|
| `LabelGeneratorConfig` | 标签生成 | 9 个 | 配置多周期收益率标签 |
| `NeutralizeConfig` | 中性化 | 8 个 | 配置中性化处理参数 |
| `ProcessingStep` | 处理步骤 | 5 个 | 定义单个处理步骤 |
| `ProcessMethod` | 处理方法 | 12 种 | 定义所有处理方法 |
| `PreprocessConfig` | 总配置 | 8 个 | 管理整个预处理流程 |

---

## 1. LabelGeneratorConfig - 标签生成配置

**用途：** 配置多周期收益率标签生成

### 参数详表

| 参数名 | 类型 | 默认值 | 必需 | 说明 |
|--------|------|--------|------|------|
| `enabled` | bool | True | ❌ | 是否启用标签生成 |
| `stock_col` | str | 'order_book_id' | ❌ | 股票代码列名 |
| `time_col` | str | 'trade_date' | ❌ | 时间列名 |
| `price_col` | str | 'close' | ❌ | 价格列名（分子） |
| `base_price_col` | Optional[str] | None | ❌ | 基准价格列名（分母） |
| `label_type` | str | 'return' | ❌ | 标签类型 |
| `return_periods` | List[int] | [1, 5, 10] | ❌ | 收益率周期 |
| `return_method` | str | 'simple' | ❌ | 收益率计算方法 |
| `label_prefix` | str | 'y_ret' | ❌ | 标签列名前缀 |

### 参数详解

#### `enabled: bool = True`
**说明：** 是否启用标签生成
- **True**: 在预处理管道中生成标签
- **False**: 跳过标签生成
- **推荐**: True（除非不需要标签）

```python
config.label_config.enabled = True
```

#### `stock_col: str = 'order_book_id'`
**说明：** 股票代码列名
- **用途**: 按股票分组计算标签
- **常见值**: 'order_book_id', 'stock_code', 'symbol'
- **推荐**: 与数据源一致

```python
config.label_config.stock_col = 'order_book_id'  # 米筐数据
```

#### `time_col: str = 'trade_date'`
**说明：** 时间列名
- **用途**: 时间序列排序和价格偏移
- **常见值**: 'trade_date', 'date', 'datetime'
- **推荐**: 与数据源一致

```python
config.label_config.time_col = 'trade_date'
```

#### `price_col: str = 'close'`
**说明：** 价格列名（分子，未来价格）
- **常见值**: 'close', 'vwap', 'open', 'high', 'low'
- **close**: 收盘价
- **vwap**: 成交量加权平均价（更接近真实交易）
- **推荐**: 'close' 或 'vwap'

```python
# 使用收盘价
config.label_config.price_col = 'close'

# 使用 VWAP
config.label_config.price_col = 'vwap'
```

#### `base_price_col: Optional[str] = None`
**说明：** 基准价格列名（分母）- **最重要的参数**

| 值 | 方式 | 公式 | 使用场景 |
|----|------|------|---------|
| None | 传统方式 | label_t = (price_{t+n} / price_t) - 1 | 理论分析 |
| 'close' | 研报标准 | label_t = (price_{t+n} / price_{t+1}) - 1 | 真实交易（推荐） |

**推荐**: 使用 'close'（研报标准）

```python
# 研报标准（推荐）
config.label_config.base_price_col = 'close'  # T+1 基准

# 传统标准
config.label_config.base_price_col = None  # T 基准

# VWAP 标准
config.label_config.base_price_col = 'vwap'
```

#### `label_type: str = 'return'`
**说明：** 标签类型
- **'return'**: 生成收益率标签（当前支持）
- **'class'**: 生成分类标签（未来支持）
- **推荐**: 'return'

```python
config.label_config.label_type = 'return'
```

#### `return_periods: List[int] = [1, 5, 10]`
**说明：** 收益率周期列表（单位：交易日）
- **[1]**: 1日收益率 → y_ret_1d
- **[1, 5, 10]**: 三个周期 → y_ret_1d, y_ret_5d, y_ret_10d
- **[1, 2, 3, 5, 10, 20]**: 多个周期

**推荐范围**: [1, 5, 10, 20] 或根据需要自定义

```python
# 仅1日
config.label_config.return_periods = [1]

# 标准三个周期
config.label_config.return_periods = [1, 5, 10]

# 完整范围
config.label_config.return_periods = [1, 2, 3, 5, 10, 20]
```

#### `return_method: str = 'simple'`
**说明：** 收益率计算方法

| 值 | 公式 | 特点 | 推荐场景 |
|----|------|------|---------|
| 'simple' | r = (P/Base) - 1 | 直观易懂 | 初始分析 |
| 'log' | r = ln(P/Base) | 数学严谨 | 统计建模 |

**推荐**: 'simple'

```python
config.label_config.return_method = 'simple'
```

#### `label_prefix: str = 'y_ret'`
**说明：** 标签列名前缀
- **格式**: {label_prefix}_{period}d
- **示例**:
  - 'y_ret' → y_ret_1d, y_ret_5d, y_ret_10d
  - 'ret' → ret_1d, ret_5d, ret_10d
  - 'future_ret' → future_ret_1d, future_ret_5d

**建议**: 使用 'y_ret' 或 'y_' 前缀区分标签和特征

```python
# 推荐
config.label_config.label_prefix = 'y_ret'

# 区分不同类型
config.label_config.label_prefix = 'y_ret'      # 简单收益率标签
config.label_config.label_prefix = 'y_log_ret'  # 对数收益率标签
config.label_config.label_prefix = 'y_vwap_ret' # VWAP 收益率标签
```

#### `neutralize: bool = False`
**说明：** 是否在生成时进行中性化
- **False**: 不进行（推荐，在后续步骤处理）
- **True**: 立即进行中性化

**推荐**: False

```python
config.label_config.neutralize = False
```

### 使用示例

#### 研报标准配置（推荐）
```python
from quantclassic.data_processor.preprocess_config import LabelGeneratorConfig

config = LabelGeneratorConfig(
    enabled=True,
    stock_col='order_book_id',
    time_col='trade_date',
    price_col='close',
    base_price_col='close',  # 研报标准：T+1 基准
    label_type='return',
    return_periods=[1, 5, 10],
    return_method='simple',
    label_prefix='y_ret',
    neutralize=False
)
```

#### VWAP 标准配置
```python
config = LabelGeneratorConfig(
    price_col='vwap',
    base_price_col='vwap',
    label_prefix='y_vwap_ret'
)
```

#### 传统标准配置
```python
config = LabelGeneratorConfig(
    base_price_col=None,  # 使用 T 日基准
    label_prefix='ret'
)
```

---

## 2. NeutralizeConfig - 中性化配置

**用途：** 配置 OLS 中性化和 SimStock 中性化参数

### 参数详表

| 参数名 | 类型 | 默认值 | 必需 | 说明 |
|--------|------|--------|------|------|
| `industry_column` | str | 'industry_name' | ❌ | 行业列名 |
| `market_cap_column` | str | 'total_mv' | ❌ | 市值列名 |
| `min_samples` | int | 10 | ❌ | OLS 最小样本数 |
| `label_column` | str | 'y_ret_1d' | ❌ | 输入标签列名 |
| `similarity_threshold` | float | 0.7 | ❌ | 相似度阈值 |
| `lookback_window` | int | 252 | ❌ | 历史回溯窗口 |
| `min_similar_stocks` | int | 5 | ❌ | 最少相似股票数 |
| `correlation_method` | str | 'pearson' | ❌ | 相关性计算方法 |

### 关键参数详解

#### `similarity_threshold: float = 0.7`
**说明：** SimStock 相似度阈值（0-1）

| 值 | 效果 | 场景 |
|----|------|------|
| 0.5 | 范围广，相似度要求低 | 快速迭代 |
| 0.6 | 较宽松 | 数据量充足 |
| 0.7 | 平衡（推荐） | 标准配置 |
| 0.8 | 严格，选择高相似股票 | 严谨因子 |
| 0.9 | 非常严格 | 极端严格 |

**推荐**: 0.7 或 0.8

```python
# 平衡配置
config.neutralize_config.similarity_threshold = 0.7

# 严格配置
config.neutralize_config.similarity_threshold = 0.8

# 宽松配置
config.neutralize_config.similarity_threshold = 0.6
```

#### `lookback_window: int = 252`
**说明：** 历史回溯窗口（交易日数）

| 值 | 时间 | 场景 |
|----|------|------|
| 60 | 约3个月 | 短期分析 |
| 120 | 约6个月 | 中期分析 |
| 252 | 约1年 | 标准配置（推荐） |
| 504 | 约2年 | 长期分析 |

**推荐**: 252（标准年度）

```python
config.neutralize_config.lookback_window = 252
```

#### `correlation_method: str = 'pearson'`
**说明：** 相关性计算方法

| 值 | 特点 | 适用场景 |
|----|------|---------|
| 'pearson' | 线性相关，对异常值敏感 | 标准数据 |
| 'spearman' | 等级相关，对异常值不敏感 | 包含异常值 |

**推荐**: 'pearson'

```python
config.neutralize_config.correlation_method = 'pearson'
```

### 参数组合推荐

#### 场景1：基础因子工程
```python
config.neutralize_config = NeutralizeConfig(
    similarity_threshold=0.7,
    lookback_window=252,
    min_similar_stocks=5,
    correlation_method='pearson'
)
```

#### 场景2：严格因子工程
```python
config.neutralize_config = NeutralizeConfig(
    similarity_threshold=0.8,
    lookback_window=252,
    min_similar_stocks=10,
    correlation_method='pearson'
)
```

#### 场景3：鲁棒因子工程
```python
config.neutralize_config = NeutralizeConfig(
    similarity_threshold=0.7,
    lookback_window=252,
    correlation_method='spearman'  # 使用等级相关
)
```

---

## 3. ProcessMethod - 处理方法枚举

**用途：** 定义所有支持的处理方法

### 完整方法列表

| 分类 | 方法 | 值 | 用途 |
|------|------|-----|------|
| **标签生成** | GENERATE_LABELS | "generate_labels" | 生成多周期标签 |
| **标准化** | Z_SCORE | "z_score" | 标准正态标准化 |
| | MINMAX | "minmax" | 最小最大标准化 |
| | RANK | "rank" | 排名标准化 |
| **中性化** | OLS_NEUTRALIZE | "ols_neutralize" | OLS 中性化 |
| | MEAN_NEUTRALIZE | "mean_neutralize" | 均值中性化 |
| | SIMSTOCK_LABEL_NEUTRALIZE | "simstock_label_neutralize" | SimStock 中性化 |
| **极值处理** | WINSORIZE | "winsorize" | 百分位截断 |
| | CLIP | "clip" | 固定值截断 |
| **缺失值处理** | FILLNA_MEDIAN | "fillna_median" | 中位数填充 |
| | FILLNA_MEAN | "fillna_mean" | 均值填充 |
| | FILLNA_FORWARD | "fillna_forward" | 向前填充 |
| | FILLNA_ZERO | "fillna_zero" | 零值填充 |

### 推荐使用顺序

```python
# 基础流程
1. GENERATE_LABELS       # 第一步：生成标签
2. WINSORIZE            # 第二步：去极值
3. Z_SCORE              # 第三步：标准化
4. FILLNA_MEDIAN        # 第四步：填充缺失

# 完整流程（含中性化）
1. GENERATE_LABELS                # 第一步：生成标签
2. WINSORIZE                       # 第二步：去极值
3. Z_SCORE                         # 第三步：标准化
4. OLS_NEUTRALIZE                 # 第四步：特征中性化
5. SIMSTOCK_LABEL_NEUTRALIZE      # 第五步：标签中性化
6. FILLNA_MEDIAN                  # 第六步：填充缺失
```

---

## 4. ProcessingStep - 处理步骤

**用途：** 定义单个处理步骤的配置

### 参数详表

| 参数名 | 类型 | 默认值 | 必需 | 说明 |
|--------|------|--------|------|------|
| `name` | str | - | ✅ | 步骤名称 |
| `method` | ProcessMethod | - | ✅ | 处理方法 |
| `features` | Union[str, List[str], None] | None | ❌ | 要处理的列 |
| `enabled` | bool | True | ❌ | 是否启用 |
| `params` | Dict[str, Any] | {} | ❌ | 方法参数 |

### 使用示例

```python
# 添加处理步骤
config.add_step(
    name='标签生成',
    method=ProcessMethod.GENERATE_LABELS,
    features=None,
    enabled=True,
    params={}
)

config.add_step(
    name='去极值',
    method=ProcessMethod.WINSORIZE,
    features=None,
    params={'limits': [0.025, 0.025]}
)

config.add_step(
    name='标准化',
    method=ProcessMethod.Z_SCORE,
    features=['close', 'volume'],
    params={'normalize_mode': 'cross_section'}
)

# 暂时禁用某个步骤
config.add_step(
    name='特征中性化',
    method=ProcessMethod.OLS_NEUTRALIZE,
    enabled=False  # 临时禁用
)
```

---

## 5. PreprocessConfig - 总配置

**用途：** 管理整个预处理流程

### 参数详表

| 参数名 | 类型 | 默认值 | 必需 | 说明 |
|--------|------|--------|------|------|
| `pipeline_steps` | List[ProcessingStep] | [] | ❌ | 处理步骤列表 |
| `column_mapping` | Dict[str, str] | {} | ❌ | 列名映射 |
| `groupby_columns` | List[str] | ['trade_date'] | ❌ | 分组列 |
| `id_columns` | List[str] | ['order_book_id', 'trade_date'] | ❌ | ID列 |
| `label_config` | LabelGeneratorConfig | LabelGeneratorConfig() | ❌ | 标签配置 |
| `neutralize_config` | NeutralizeConfig | NeutralizeConfig() | ❌ | 中性化配置 |
| `save_intermediate` | bool | False | ❌ | 保存中间结果 |
| `intermediate_dir` | str | 'intermediate_results' | ❌ | 中间结果目录 |

### 完整使用示例

```python
from quantclassic.data_processor.preprocess_config import (
    PreprocessConfig, ProcessMethod
)
from quantclassic.data_processor.data_preprocessor import DataPreprocessor

# 创建配置
config = PreprocessConfig()

# 配置标签生成
config.label_config.enabled = True
config.label_config.base_price_col = 'close'  # 研报标准
config.label_config.return_periods = [1, 5, 10]
config.label_config.label_prefix = 'y_ret'

# 配置中性化
config.neutralize_config.label_column = 'y_ret_1d'
config.neutralize_config.similarity_threshold = 0.7
config.neutralize_config.lookback_window = 252

# 配置处理步骤
config.add_step('生成标签', ProcessMethod.GENERATE_LABELS)
config.add_step('去极值', ProcessMethod.WINSORIZE, 
               params={'limits': [0.025, 0.025]})
config.add_step('标准化', ProcessMethod.Z_SCORE,
               params={'normalize_mode': 'cross_section'})
config.add_step('特征中性化', ProcessMethod.OLS_NEUTRALIZE,
               params={'industry_column': 'industry_name'})
config.add_step('标签中性化', ProcessMethod.SIMSTOCK_LABEL_NEUTRALIZE,
               params={'label_column': 'y_ret_1d',
                       'output_column': 'alpha_label'})
config.add_step('填充缺失', ProcessMethod.FILLNA_MEDIAN)

# 保存配置
config.to_yaml('preprocess_config.yaml')

# 执行预处理
processor = DataPreprocessor(config)
df_processed = processor.fit_transform(df_raw)
```

---

## 常见问题 (FAQ)

### Q1: 标签应该用哪种基准（T 还是 T+1）？
**A**: 推荐使用 T+1（研报标准），因为：
- T 日基准在实际交易中无法成交
- T+1 基准模拟真实交易情景
- 与财通证券、海通证券研报标准一致

```python
config.label_config.base_price_col = 'close'  # 使用 T+1 基准
```

### Q2: 为什么标签生成应该是第一步？
**A**: 因为：
- 标签需要对原始价格数据计算
- 后续步骤（去极值、标准化）应作用于生成后的标签
- 避免修改原始数据影响标签生成

```python
config.add_step('生成标签', ProcessMethod.GENERATE_LABELS)  # 必须第一步
config.add_step('去极值', ProcessMethod.WINSORIZE)
```

### Q3: 相似度阈值怎么选？
**A**: 根据场景选择：
- **0.6-0.7**: 快速迭代、数据充足
- **0.7-0.8**: 标准配置（推荐）
- **0.8-0.9**: 严谨因子工程

```python
# 推荐
config.neutralize_config.similarity_threshold = 0.7
```

### Q4: 标签列名前缀为什么用 'y_ret'？
**A**: 为了：
- 区分标签（y_ret_*）和特征（ret_*）
- 避免列名冲突
- 遵循命名规范（y 表示目标变量）

```python
config.label_config.label_prefix = 'y_ret'  # 推荐
```

---

## 总结

| 配置类 | 关键参数 | 推荐值 |
|--------|---------|--------|
| LabelGeneratorConfig | base_price_col | 'close' |
| | label_prefix | 'y_ret' |
| | return_periods | [1, 5, 10] |
| NeutralizeConfig | similarity_threshold | 0.7 |
| | lookback_window | 252 |
| | correlation_method | 'pearson' |
| PreprocessConfig | pipeline_steps | [生成→去极值→标准化→中性化→填充] |

更详细的使用说明，请参考：
- `LABEL_GENERATION_CONFIG_GUIDE.md` - 标签生成指南
- `RESEARCH_STANDARD_LABEL.md` - 研报标准说明
- `LABEL_GENERATION_INTEGRATION_SUMMARY.md` - 集成总结
