# 标签生成集成完成总结

## 完成时间
2024年11月20日

## 概述

标签生成功能已成功集成到 quantclassic 的预处理配置系统中。用户无需手动调用 `LabelGenerator`，可以通过统一的配置系统自动生成多周期收益率标签。

## 新增功能

### 1. 配置类扩展

#### LabelGeneratorConfig
在 `preprocess_config.py` 中新增 `LabelGeneratorConfig` 类：

```python
@dataclass
class LabelGeneratorConfig(BaseConfig):
    enabled: bool = True
    stock_col: str = 'order_book_id'
    time_col: str = 'trade_date'
    price_col: str = 'close'
    base_price_col: Optional[str] = None  # 研报标准：设置为'close'使用T+1基准
    label_type: str = 'return'
    return_periods: List[int] = field(default_factory=lambda: [1, 5, 10])
    return_method: str = 'simple'
    label_prefix: str = 'y_ret'  # 生成 y_ret_1d, y_ret_5d, y_ret_10d
    neutralize: bool = False
```

#### ProcessMethod.GENERATE_LABELS
在 `ProcessMethod` 枚举中新增 `GENERATE_LABELS` 方法：

```python
class ProcessMethod(Enum):
    GENERATE_LABELS = "generate_labels"  # 新增
    Z_SCORE = "z_score"
    # ...其他方法
```

### 2. PreprocessConfig 集成

`PreprocessConfig` 类新增 `label_config` 字段：

```python
@dataclass
class PreprocessConfig(BaseConfig):
    label_config: LabelGeneratorConfig = field(default_factory=LabelGeneratorConfig)
    # ...其他字段
```

### 3. DataPreprocessor 处理逻辑

在 `data_preprocessor.py` 中实现标签生成处理：

```python
if method == ProcessMethod.GENERATE_LABELS:
    # 从config获取参数
    label_config = LabelConfig(
        stock_col=self.config.label_config.stock_col,
        time_col=self.config.label_config.time_col,
        price_col=self.config.label_config.price_col,
        base_price_col=self.config.label_config.base_price_col,
        # ...
    )
    label_generator = LabelGenerator(label_config)
    df = label_generator.generate_labels(df, label_name='label')
    
    # 自动重命名为 y_ret_* 格式
    rename_map = {'label_1d': 'y_ret_1d', ...}
    df = df.rename(columns=rename_map)
```

## 使用方法

### 方法1: 配置对象

```python
from quantclassic.data_processor.preprocess_config import PreprocessConfig, ProcessMethod
from quantclassic.data_processor.data_preprocessor import DataPreprocessor

# 创建配置
config = PreprocessConfig()

# 配置标签生成（研报标准）
config.label_config.enabled = True
config.label_config.base_price_col = 'close'  # T+1基准
config.label_config.return_periods = [1, 5, 10]
config.label_config.label_prefix = 'y_ret'

# 添加到管道
config.add_step('生成标签', ProcessMethod.GENERATE_LABELS)
config.add_step('去极值', ProcessMethod.WINSORIZE, params={'limits': [0.025, 0.025]})
config.add_step('标准化', ProcessMethod.Z_SCORE)

# 执行
processor = DataPreprocessor(config)
df_processed = processor.fit_transform(df_raw)
```

### 方法2: YAML配置文件

```yaml
label_config:
  enabled: true
  base_price_col: close  # 研报标准
  return_periods: [1, 5, 10]
  label_prefix: y_ret

pipeline_steps:
  - name: 生成标签
    method: generate_labels
    enabled: true
```

```python
config = PreprocessConfig.from_yaml('preprocess_config.yaml')
processor = DataPreprocessor(config)
df_processed = processor.fit_transform(df_raw)
```

## 测试验证

所有测试用例通过：

```
✅ 测试1: 基本标签生成（研报标准）
  - 生成 y_ret_1d, y_ret_5d, y_ret_10d 列
  - 数据完整性验证通过

✅ 测试2: 完整预处理流程
  - 标签生成 + 去极值 + 标准化
  - 标准化效果验证通过

✅ 测试3: 配置保存和加载
  - YAML序列化/反序列化正常
  - 配置参数完整保留
```

## 优势

### 1. 统一管理
标签生成与其他预处理步骤统一在一个配置系统中，避免了分散管理。

### 2. 配置驱动
通过配置文件或对象配置，无需编写重复代码。

### 3. 易于切换
轻松切换研报标准和传统标准：

```python
# 研报标准（T+1基准）
config.label_config.base_price_col = 'close'

# 传统标准（T基准）
config.label_config.base_price_col = None
```

### 4. 自动重命名
自动处理列名重命名，生成 `y_ret_*` 格式的标签列，避免与特征列 `ret_*` 冲突。

### 5. 可复用
配置可保存为YAML文件，方便版本控制和团队协作。

## 文档

新增文档：

1. **LABEL_GENERATION_CONFIG_GUIDE.md** - 标签生成配置完整指南
2. **test_label_generation_integration.py** - 集成测试脚本

更新文档：

1. **preprocess_config.py** - 新增 `LabelGeneratorConfig` 类和参数注释
2. **data_preprocessor.py** - 新增标签生成处理逻辑

## 兼容性

### 向后兼容
- 旧代码仍可继续使用 `LabelGenerator` 手动生成标签
- 不影响现有预处理流程

### 向前兼容
- 配置系统支持扩展新的标签类型（如分类标签）
- 支持自定义标签生成逻辑

## 示例对比

### 旧方式（手动）

```python
# 手动创建标签生成器
from quantclassic.data_processor.label_generator import LabelConfig, LabelGenerator

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

# 继续其他预处理
# ...
```

### 新方式（集成）

```python
from quantclassic.data_processor.preprocess_config import PreprocessConfig, ProcessMethod
from quantclassic.data_processor.data_preprocessor import DataPreprocessor

# 创建配置
config = PreprocessConfig()
config.label_config.base_price_col = 'close'
config.label_config.return_periods = [1, 5, 10]
config.label_config.label_prefix = 'y_ret'

# 添加到管道
config.add_step('生成标签', ProcessMethod.GENERATE_LABELS)
config.add_step('去极值', ProcessMethod.WINSORIZE, params={'limits': [0.025, 0.025]})
config.add_step('标准化', ProcessMethod.Z_SCORE)

# 一键执行
processor = DataPreprocessor(config)
df_processed = processor.fit_transform(df_raw)
# 标签已自动生成并重命名
```

**代码行数对比：**
- 旧方式：约15行
- 新方式：约10行
- 使用YAML配置：仅3行Python代码

## Notebook 更新

已更新 `jupyterlab/vae.ipynb` 中的 Cell 5，使用集成配置方式：

```python
# 步骤 2B: 数据预处理配置（标签生成已集成）
config = PreprocessConfig()
config.label_config.base_price_col = 'close'  # 研报标准
config.add_step('生成标签', ProcessMethod.GENERATE_LABELS)
# ...
```

## 未来改进

1. **支持分类标签**: 扩展 `label_type` 支持分类任务
2. **支持对数收益率**: 扩展 `return_method` 支持 log returns
3. **支持滚动窗口**: 支持滚动窗口收益率计算
4. **支持多价格类型**: 同时使用 close 和 vwap 生成标签

## 相关文件

修改的文件：
- `quantclassic/data_processor/preprocess_config.py`
- `quantclassic/data_processor/data_preprocessor.py`
- `quantclassic/data_processor/label_generator.py`
- `jupyterlab/vae.ipynb` (Cell 5)

新增的文件：
- `quantclassic/LABEL_GENERATION_CONFIG_GUIDE.md`
- `quantclassic/test_label_generation_integration.py`

## 总结

标签生成功能已成功集成到 quantclassic 配置系统，实现了：

✅ 统一的配置管理接口  
✅ 与预处理管道无缝集成  
✅ 支持研报标准（T+1基准）  
✅ 自动列名重命名  
✅ YAML配置支持  
✅ 完整的文档和测试  

用户可以通过简单的配置完成标签生成，无需编写重复代码，提高了开发效率和代码可维护性。
