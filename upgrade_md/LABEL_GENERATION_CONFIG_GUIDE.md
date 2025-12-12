# 标签生成配置指南

## 概述

标签生成已集成到 quantclassic 的预处理配置系统中，无需手动调用 `LabelGenerator`，可以通过配置自动生成多周期收益率标签。

## 快速开始

### 方法1: 使用配置对象

```python
from quantclassic.data_processor.preprocess_config import PreprocessConfig, ProcessMethod
from quantclassic.data_processor.data_preprocessor import DataPreprocessor
import pandas as pd

# 1. 创建预处理配置
config = PreprocessConfig()

# 2. 配置标签生成（研报标准）
config.label_config.enabled = True
config.label_config.price_col = 'close'
config.label_config.base_price_col = 'close'  # 使用T+1作为基准（研报标准）
config.label_config.return_periods = [1, 5, 10]  # 生成1/5/10日标签
config.label_config.label_prefix = 'y_ret'  # 生成 y_ret_1d, y_ret_5d, y_ret_10d

# 3. 添加标签生成步骤到管道
config.add_step(
    name='生成多周期标签',
    method=ProcessMethod.GENERATE_LABELS
)

# 4. 添加其他预处理步骤
config.add_step('去极值', ProcessMethod.WINSORIZE, params={'limits': [0.025, 0.025]})
config.add_step('标准化', ProcessMethod.Z_SCORE)

# 5. 执行预处理
processor = DataPreprocessor(config)
df_processed = processor.fit_transform(df_raw)

# 生成的标签列: y_ret_1d, y_ret_5d, y_ret_10d
```

### 方法2: 使用YAML配置文件

```yaml
# preprocess_config.yaml

# 标签生成配置
label_config:
  enabled: true
  stock_col: order_book_id
  time_col: trade_date
  price_col: close
  base_price_col: close  # 研报标准：使用T+1作为基准
  label_type: return
  return_periods: [1, 5, 10]
  return_method: simple
  label_prefix: y_ret  # 生成 y_ret_1d, y_ret_5d, y_ret_10d
  neutralize: false

# 处理步骤
pipeline_steps:
  - name: 生成多周期标签
    method: generate_labels
    features: null
    enabled: true
    params: {}
  
  - name: 去极值处理
    method: winsorize
    features: null
    enabled: true
    params:
      limits: [0.025, 0.025]
  
  - name: 截面标准化
    method: z_score
    features: null
    enabled: true
    params:
      normalize_mode: cross_section
  
  - name: 标签中性化
    method: simstock_label_neutralize
    features: null
    enabled: true
    params:
      label_column: y_ret_1d
      output_column: alpha_label

# 中性化配置
neutralize_config:
  label_column: y_ret_1d
  output_column: alpha_label
  similarity_threshold: 0.7
  lookback_window: 252

# 其他配置
groupby_columns: [trade_date]
id_columns: [order_book_id, trade_date]
```

```python
# 加载并使用配置
from quantclassic.data_processor.preprocess_config import PreprocessConfig
from quantclassic.data_processor.data_preprocessor import DataPreprocessor

config = PreprocessConfig.from_yaml('preprocess_config.yaml')
processor = DataPreprocessor(config)
df_processed = processor.fit_transform(df_raw)
```

## 配置参数详解

### LabelGeneratorConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | True | 是否启用标签生成 |
| `stock_col` | str | 'order_book_id' | 股票代码列名 |
| `time_col` | str | 'trade_date' | 时间列名 |
| `price_col` | str | 'close' | 价格列名（分子） |
| `base_price_col` | Optional[str] | None | 基准价格列名（分母） |
| `label_type` | str | 'return' | 标签类型（目前仅支持'return'） |
| `return_periods` | List[int] | [1, 5, 10] | 收益率周期列表 |
| `return_method` | str | 'simple' | 收益率计算方法 |
| `label_prefix` | str | 'y_ret' | 标签列名前缀 |
| `neutralize` | bool | False | 是否自动中性化（保留用于未来） |

### base_price_col 说明

**研报标准（推荐）：**
```python
config.label_config.base_price_col = 'close'  # 使用T+1作为基准
# 公式: label_t = (price_{t+n} / price_{t+1}) - 1
# 含义: 在T日收盘预测，T+1日交易，持有到T+n日的收益率
```

**传统标准：**
```python
config.label_config.base_price_col = None  # 使用T作为基准
# 公式: label_t = (price_{t+n} / price_t) - 1
# 含义: 在T日收盘预测并交易（不可能），持有到T+n日的收益率
```

### 标签列命名规则

生成的标签列名格式：`{label_prefix}_{period}d`

示例：
- `label_prefix='y_ret'` + `return_periods=[1, 5, 10]` → `y_ret_1d`, `y_ret_5d`, `y_ret_10d`
- `label_prefix='ret'` + `return_periods=[1, 3, 5]` → `ret_1d`, `ret_3d`, `ret_5d`

**命名建议：**
- 使用 `y_ret` 或 `y_` 前缀表示标签（未来收益率）
- 避免与特征列冲突（特征中的历史收益率通常用 `ret_*`）

## 完整示例

### 研报标准流程（推荐）

```python
from quantclassic.data_processor.preprocess_config import PreprocessConfig, ProcessMethod
from quantclassic.data_processor.data_preprocessor import DataPreprocessor
import pandas as pd

# 加载原始数据
df_raw = pd.read_parquet('output/train_data_raw.parquet')

# 创建配置
config = PreprocessConfig()

# 配置标签生成（研报标准）
config.label_config.enabled = True
config.label_config.stock_col = 'order_book_id'
config.label_config.time_col = 'trade_date'
config.label_config.price_col = 'close'
config.label_config.base_price_col = 'close'  # 研报标准：T+1基准
config.label_config.return_periods = [1, 5, 10]
config.label_config.label_prefix = 'y_ret'

# 配置中性化参数
config.neutralize_config.label_column = 'y_ret_1d'
config.neutralize_config.output_column = 'alpha_label'

# 添加处理步骤
config.add_step('生成标签', ProcessMethod.GENERATE_LABELS)
config.add_step('去极值', ProcessMethod.WINSORIZE, params={'limits': [0.025, 0.025]})
config.add_step('标准化', ProcessMethod.Z_SCORE, params={'normalize_mode': 'cross_section'})
config.add_step('特征中性化', ProcessMethod.OLS_NEUTRALIZE, 
               params={'industry_column': 'industry_name', 'market_cap_column': 'market_cap'})
config.add_step('标签中性化', ProcessMethod.SIMSTOCK_LABEL_NEUTRALIZE,
               params={'label_column': 'y_ret_1d', 'output_column': 'alpha_label'})
config.add_step('填充缺失', ProcessMethod.FILLNA_MEDIAN)

# 保存配置
config.to_yaml('output/configs/preprocess_config.yaml')

# 执行预处理
processor = DataPreprocessor(config)
df_processed = processor.fit_transform(df_raw)

# 保存结果
df_processed.to_parquet('output/train_data_final.parquet')

# 查看生成的标签
print("生成的标签列:")
label_cols = [col for col in df_processed.columns if col.startswith('y_ret_')]
for col in label_cols:
    print(f"  {col}: {df_processed[col].count()} 有效值")
```

## 与手动方式的对比

### 旧方式（手动调用）

```python
from quantclassic.data_processor.label_generator import LabelConfig, LabelGenerator

# 手动创建配置和生成器
label_config = LabelConfig(
    stock_col='order_book_id',
    time_col='trade_date',
    price_col='close',
    base_price_col='close',
    return_periods=[1, 5, 10]
)
label_gen = LabelGenerator(label_config)

# 手动生成标签
df_with_labels = label_gen.generate_labels(df_raw, label_name='label')

# 手动重命名
rename_map = {'label_1d': 'y_ret_1d', 'label_5d': 'y_ret_5d', 'label_10d': 'y_ret_10d'}
df_with_labels = df_with_labels.rename(columns=rename_map)

# 继续其他预处理...
```

### 新方式（集成配置）

```python
from quantclassic.data_processor.preprocess_config import PreprocessConfig, ProcessMethod
from quantclassic.data_processor.data_preprocessor import DataPreprocessor

# 创建配置
config = PreprocessConfig()
config.label_config.base_price_col = 'close'  # 研报标准
config.label_config.return_periods = [1, 5, 10]
config.label_config.label_prefix = 'y_ret'

# 添加到管道
config.add_step('生成标签', ProcessMethod.GENERATE_LABELS)

# 一键执行
processor = DataPreprocessor(config)
df_processed = processor.fit_transform(df_raw)

# 标签已自动生成并重命名为 y_ret_1d, y_ret_5d, y_ret_10d
```

## 优势

1. **统一管理**: 标签生成与其他预处理步骤统一管理
2. **配置驱动**: 通过配置文件或对象配置，无需手动编码
3. **可复用**: 配置可保存为YAML文件，方便复用和版本控制
4. **易于切换**: 轻松切换研报标准和传统标准
5. **自动命名**: 自动处理列名重命名，避免命名冲突

## 常见问题

### Q1: 如何切换传统标准和研报标准？

**传统标准（T日基准）：**
```python
config.label_config.base_price_col = None
```

**研报标准（T+1日基准）：**
```python
config.label_config.base_price_col = 'close'
```

### Q2: 如何生成不同周期的标签？

```python
# 仅生成1日标签
config.label_config.return_periods = [1]

# 生成1/3/5/10/20日标签
config.label_config.return_periods = [1, 3, 5, 10, 20]
```

### Q3: 如何使用VWAP价格生成标签？

```python
config.label_config.price_col = 'vwap'
config.label_config.base_price_col = 'vwap'  # 使用T+1日VWAP作为基准
```

### Q4: 生成的标签在哪个步骤？

标签生成应该是管道的**第一步**，在其他预处理步骤之前：

```python
config.add_step('生成标签', ProcessMethod.GENERATE_LABELS)  # 第一步
config.add_step('去极值', ProcessMethod.WINSORIZE)         # 第二步
config.add_step('标准化', ProcessMethod.Z_SCORE)           # 第三步
```

### Q5: 如何禁用标签生成？

```python
# 方法1: 禁用label_config
config.label_config.enabled = False

# 方法2: 不添加GENERATE_LABELS步骤
# （不调用 config.add_step('生成标签', ProcessMethod.GENERATE_LABELS)）
```

### Q6: 标签列和特征列如何区分？

**建议命名规范：**
- **标签列**（未来收益率）: `y_ret_1d`, `y_ret_5d` 等（`y_` 前缀）
- **特征列**（历史收益率）: `ret_1d`, `ret_5d` 等（无 `y_` 前缀）
- **原始价格**: `close`, `vwap`, `open` 等

这样可以清晰区分特征（X）和标签（Y）。

## 参考文档

- [RESEARCH_STANDARD_LABEL.md](./RESEARCH_STANDARD_LABEL.md) - 研报标准详细说明
- [CONFIG_README.md](./CONFIG_README.md) - 配置系统总览
- [preprocess_config.py](./data_processor/preprocess_config.py) - 配置类定义
- [label_generator.py](./data_processor/label_generator.py) - 标签生成器实现

## 总结

标签生成集成到配置系统后，可以：

1. ✅ 通过配置对象或YAML文件统一管理
2. ✅ 与其他预处理步骤无缝衔接
3. ✅ 轻松切换研报标准和传统标准
4. ✅ 自动处理列名重命名
5. ✅ 配置可保存、加载、复用

推荐使用集成配置方式，提高代码复用性和可维护性。
