# QuantClassic 面向对象配置 - 快速参考

## 配置类速查表

### 模型配置 (`model.model_config`)

| 配置类 | 用途 | 关键参数 |
|--------|------|----------|
| `VAEConfig` | VAE 模型 | `hidden_dim`, `latent_dim`, `alpha_recon`, `beta_kl` |
| `LSTMConfig` | LSTM 模型 | `hidden_size`, `num_layers`, `bidirectional` |
| `GRUConfig` | GRU 模型 | `hidden_size`, `num_layers`, `bidirectional` |
| `TransformerConfig` | Transformer 模型 | `d_model`, `nhead`, `num_layers` |
| `MLPConfig` | MLP 模型 | `hidden_sizes`, `activation` |

### 数据配置 (`data_manager.config`)

| 配置类 | 用途 | 关键参数 |
|--------|------|----------|
| `DataConfig` | 数据管理 | `window_size`, `batch_size`, `train_ratio` |

### 预处理配置 (`data_processor.preprocess_config`)

| 配置类 | 用途 | 关键参数 |
|--------|------|----------|
| `PreprocessConfig` | 数据预处理 | `pipeline_steps` |
| `ProcessingStep` | 单个处理步骤 | `name`, `method`, `params` |
| `NeutralizeConfig` | 中性化 | `industry_column`, `market_cap_column` |

### 工作流配置 (`workflow.workflow_config`)

| 配置类 | 用途 | 关键参数 |
|--------|------|----------|
| `WorkflowConfig` | 工作流管理 | `enabled`, `recorder`, `checkpoint` |
| `RecorderConfig` | 实验记录 | `experiment_name`, `log_params` |
| `CheckpointConfig` | 检查点 | `save_frequency`, `keep_last_n` |

## 常用代码片段

### 创建 VAE 配置

```python
from quantclassic.model.model_config import VAEConfig

config = VAEConfig(
    hidden_dim=128,
    latent_dim=16,
    n_epochs=100,
    learning_rate=0.001,
    device='cuda'
)
```

### 使用工厂创建

```python
from quantclassic.model.model_config import ModelConfigFactory

# 创建配置
config = ModelConfigFactory.create('vae', hidden_dim=256)

# 使用模板
small = ModelConfigFactory.get_template('vae', 'small')
large = ModelConfigFactory.get_template('vae', 'large')
```

### 加载 YAML 配置

```python
from quantclassic.model.model_config import VAEConfig

# 直接加载
config = VAEConfig.from_yaml('config.yaml')

# 或使用加载器
from quantclassic.config.loader import ConfigLoader
config = ConfigLoader.load('config.yaml', VAEConfig)
```

### 保存配置

```python
# 保存为 YAML
config.to_yaml('output/config.yaml')

# 保存为 JSON
config.to_json('output/config.json')

# 转换为字典
config_dict = config.to_dict()
```

### 更新配置

```python
# 更新单个参数
config.update(hidden_dim=256)

# 更新多个参数
config.update(
    hidden_dim=256,
    latent_dim=32,
    learning_rate=0.002
)
```

### 合并配置

```python
config1 = VAEConfig(hidden_dim=128)
config2 = VAEConfig(latent_dim=32)

# config2 的值覆盖 config1
merged = config1.merge(config2)
```

### 配置验证

```python
try:
    config.validate()
    print("配置有效")
except ValueError as e:
    print(f"配置错误: {e}")
```

### 创建数据配置

```python
from quantclassic.data_manager.config import DataConfig

config = DataConfig(
    base_dir='rq_data_parquet',
    window_size=40,
    batch_size=512,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2
)
```

### 创建预处理配置

```python
from quantclassic.data_processor.preprocess_config import (
    PreprocessConfig,
    ProcessMethod
)

config = PreprocessConfig()
config.add_step('填充缺失值', ProcessMethod.FILLNA_MEDIAN)
config.add_step('去极值', ProcessMethod.WINSORIZE, params={'limits': [0.025, 0.025]})
config.add_step('标准化', ProcessMethod.Z_SCORE)
```

### 创建工作流配置

```python
from quantclassic.workflow.workflow_config import (
    WorkflowConfig,
    RecorderConfig,
    CheckpointConfig
)

config = WorkflowConfig(
    enabled=True,
    recorder=RecorderConfig(
        experiment_name='my_exp',
        log_params=True,
        log_metrics=True
    ),
    checkpoint=CheckpointConfig(
        save_frequency=10,
        keep_last_n=3
    )
)
```

### 完整工作流示例

```python
from quantclassic.config.loader import ConfigLoader
from quantclassic.model.model_config import VAEConfig
from quantclassic.data_manager.config import DataConfig

# 加载完整配置
full_config = ConfigLoader.load('config.yaml', return_dict=True)

# 提取各部分
model_config = VAEConfig.from_dict(full_config['model'])
data_config = DataConfig.from_dict(full_config['data'])

# 验证
model_config.validate()
data_config.validate()

# 使用配置
print(f"模型: hidden_dim={model_config.hidden_dim}")
print(f"数据: window_size={data_config.window_size}")
```

## ProcessMethod 枚举

预处理方法快速参考：

```python
from quantclassic.data_processor.preprocess_config import ProcessMethod

# 标准化/归一化
ProcessMethod.Z_SCORE          # Z-Score 标准化
ProcessMethod.MINMAX           # MinMax 归一化
ProcessMethod.RANK             # 秩归一化

# 中性化
ProcessMethod.OLS_NEUTRALIZE   # OLS 市值行业中性化
ProcessMethod.MEAN_NEUTRALIZE  # 均值中性化
ProcessMethod.SIMSTOCK_NEUTRALIZE  # 相似股票中性化

# 极值处理
ProcessMethod.WINSORIZE        # Winsorize 去极值
ProcessMethod.CLIP             # 裁剪

# 缺失值处理
ProcessMethod.FILLNA_MEDIAN    # 填充中位数
ProcessMethod.FILLNA_MEAN      # 填充均值
ProcessMethod.FILLNA_FORWARD   # 前向填充
ProcessMethod.FILLNA_ZERO      # 填充零
```

## 配置模板

### 模型模板

```python
from quantclassic.model.model_config import ModelConfigFactory

# 小型 VAE（快速原型）
small_vae = ModelConfigFactory.get_template('vae', 'small')
# hidden_dim=64, latent_dim=8, num_layers=1, n_epochs=50

# 默认 VAE
default_vae = ModelConfigFactory.get_template('vae', 'default')
# hidden_dim=128, latent_dim=16, num_layers=2, n_epochs=100

# 大型 VAE（生产环境）
large_vae = ModelConfigFactory.get_template('vae', 'large')
# hidden_dim=256, latent_dim=32, num_layers=3, n_epochs=200
```

### 数据模板

```python
from quantclassic.data_manager.config import ConfigTemplates

# 默认配置
default = ConfigTemplates.default()

# 快速测试
quick_test = ConfigTemplates.quick_test()
# window_size=20, batch_size=128, enable_cache=False

# 生产环境
production = ConfigTemplates.production()
# batch_size=512, num_workers=4, pin_memory=True

# 回测配置
backtest = ConfigTemplates.backtest()
# split_strategy='rolling', rolling_window_size=252
```

### 预处理模板

```python
from quantclassic.data_processor.preprocess_config import PreprocessTemplates

# 基础流程
basic = PreprocessTemplates.basic_pipeline()
# 裁剪 -> 填充 -> 去极值 -> 标准化

# 高级流程（含中性化）
advanced = PreprocessTemplates.advanced_pipeline()
# 裁剪 -> 填充 -> 去极值 -> 中性化 -> 秩归一化

# Alpha 因子流程
alpha = PreprocessTemplates.alpha_pipeline()
# 裁剪 -> 填充 -> 去极值 -> SimStock中性化 -> 秩归一化
```

### 工作流模板

```python
from quantclassic.workflow.workflow_config import WorkflowTemplates

# 最小化配置
minimal = WorkflowTemplates.minimal()
# 仅记录参数和指标，不记录工件

# 完整配置
full = WorkflowTemplates.full()
# 记录所有内容，频繁保存检查点

# 生产环境
production = WorkflowTemplates.production()
# 完整记录 + 环境标签
```

## BaseConfig 方法

所有配置类继承的通用方法：

| 方法 | 说明 | 示例 |
|------|------|------|
| `validate()` | 验证配置 | `config.validate()` |
| `to_dict()` | 转换为字典 | `d = config.to_dict()` |
| `from_dict()` | 从字典创建 | `c = Config.from_dict(d)` |
| `to_yaml()` | 保存为 YAML | `config.to_yaml('file.yaml')` |
| `from_yaml()` | 从 YAML 加载 | `c = Config.from_yaml('file.yaml')` |
| `to_json()` | 保存为 JSON | `config.to_json('file.json')` |
| `from_json()` | 从 JSON 加载 | `c = Config.from_json('file.json')` |
| `update()` | 更新参数 | `config.update(key=value)` |
| `merge()` | 合并配置 | `c3 = c1.merge(c2)` |

## 快速调试

### 打印配置

```python
# 详细信息
print(repr(config))

# 简洁信息
print(str(config))

# 字典形式
print(config.to_dict())
```

### 检查配置差异

```python
# 比较两个配置
config1_dict = config1.to_dict()
config2_dict = config2.to_dict()

for key in config1_dict:
    if config1_dict[key] != config2_dict.get(key):
        print(f"{key}: {config1_dict[key]} -> {config2_dict[key]}")
```

### 配置快照

```python
# 保存配置快照
import datetime

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
config.to_yaml(f'snapshots/config_{timestamp}.yaml')
```

## 常见错误处理

### 验证错误

```python
try:
    config = VAEConfig(hidden_dim=-10)
except ValueError as e:
    print(f"配置错误: {e}")
    # 配置错误: hidden_dim 必须大于 0
```

### 未知参数

```python
try:
    config.update(unknown_param=100)
except ValueError as e:
    print(f"参数错误: {e}")
    # 参数错误: 未知配置项: unknown_param
```

### 类型错误

```python
from quantclassic.model.model_config import VAEConfig

try:
    # 错误：尝试合并不同类型的配置
    vae_config = VAEConfig()
    lstm_config = LSTMConfig()
    merged = vae_config.merge(lstm_config)
except TypeError as e:
    print(f"类型错误: {e}")
```

## 最佳实践总结

1. ✅ **使用类型提示**：提高代码可读性
2. ✅ **使用工厂模式**：简化配置创建
3. ✅ **使用模板快速开始**：减少重复配置
4. ✅ **保存配置快照**：便于版本控制和复现
5. ✅ **先验证再使用**：防止运行时错误
6. ✅ **利用 IDE 自动完成**：提高开发效率

## 更多信息

详细文档请参考：
- [迁移指南](MIGRATION_GUIDE.md)
- [配置基类](config/base_config.py)
- [模型配置](model/model_config.py)
- [数据配置](data_manager/config.py)
- [预处理配置](data_processor/preprocess_config.py)
- [工作流配置](workflow/workflow_config.py)
