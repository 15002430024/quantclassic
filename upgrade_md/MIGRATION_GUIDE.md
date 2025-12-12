# QuantClassic 配置重构迁移指南

## 概述

QuantClassic 已完成配置系统的重大重构，从**字典配置**迁移到**面向对象配置**。新系统提供更好的类型检查、验证和维护性。

## 重构亮点

### ✨ 核心改进

1. **类型安全**：使用 dataclass 提供编译时类型检查
2. **自动验证**：配置参数自动验证，防止无效配置
3. **更好的 IDE 支持**：自动完成、类型提示
4. **统一接口**：所有配置类继承自 `BaseConfig`
5. **向后兼容**：保留字典配置支持

### 📦 新增模块

```
quantclassic/
├── config/
│   ├── base_config.py          # 配置基类
│   ├── loader.py               # 升级的配置加载器
│   └── templates/
│       └── vae_oop.yaml        # 新配置模板
├── model/
│   └── model_config.py         # 模型配置类
├── data_manager/
│   └── config.py               # 数据配置类（已升级）
├── data_processor/
│   └── preprocess_config.py    # 预处理配置类（已升级）
└── workflow/
    └── workflow_config.py      # 工作流配置类
```

## 快速开始

### 方式 1：使用配置对象（推荐）

```python
from quantclassic.model.model_config import VAEConfig, ModelConfigFactory
from quantclassic.data_manager.config import DataConfig
from quantclassic.config.loader import ConfigLoader

# 创建配置对象
model_config = VAEConfig(
    hidden_dim=128,
    latent_dim=16,
    n_epochs=100,
    learning_rate=0.001
)

# 自动验证
model_config.validate()  # 抛出异常如果参数无效

# 保存到 YAML
model_config.to_yaml('my_config.yaml')

# 从 YAML 加载
loaded_config = VAEConfig.from_yaml('my_config.yaml')

# 更新配置
model_config.update(hidden_dim=256, latent_dim=32)
```

### 方式 2：使用配置工厂

```python
from quantclassic.model.model_config import ModelConfigFactory

# 使用工厂创建
config = ModelConfigFactory.create('vae', hidden_dim=256, latent_dim=32)

# 使用预定义模板
small_config = ModelConfigFactory.get_template('vae', 'small')
large_config = ModelConfigFactory.get_template('vae', 'large')
```

### 方式 3：字典配置（向后兼容）

```python
from quantclassic.config.loader import ConfigLoader

# 加载为字典（旧方式仍然支持）
config_dict = ConfigLoader.load('config.yaml', return_dict=True)

# 或者自动检测
config_dict = ConfigLoader.load('config.yaml')  # 无 config_class 参数
```

## 迁移步骤

### Step 1: 识别旧配置

**旧方式（字典）：**
```python
# 旧代码
config = {
    'hidden_dim': 128,
    'latent_dim': 16,
    'n_epochs': 100,
    'lr': 0.001
}

# 手动验证
if config['hidden_dim'] <= 0:
    raise ValueError("hidden_dim must be positive")
```

**新方式（对象）：**
```python
# 新代码
from quantclassic.model.model_config import VAEConfig

config = VAEConfig(
    hidden_dim=128,
    latent_dim=16,
    n_epochs=100,
    learning_rate=0.001  # 注意：lr -> learning_rate
)

# 自动验证（在 __post_init__ 中）
# 无需手动检查
```

### Step 2: 更新 YAML 配置文件

**旧格式：**
```yaml
task:
  model:
    class: "quantclassic.model.TimeSeriesVAE"
    kwargs:
      hidden_dim: 128
      latent_dim: 16
      n_epochs: 100
      lr: 0.001
```

**新格式：**
```yaml
model:
  model_type: "vae"  # 用于工厂创建
  hidden_dim: 128
  latent_dim: 16
  n_epochs: 100
  learning_rate: 0.001  # 统一参数名
  device: "cuda"
  optimizer: "adam"
```

### Step 3: 更新代码

#### 模型配置

**Before:**
```python
model_config = {
    'class': 'TimeSeriesVAE',
    'module_path': 'quantclassic.model.pytorch_models',
    'kwargs': {
        'hidden_dim': 128,
        'latent_dim': 16,
        'n_epochs': 100,
        'lr': 0.001
    }
}
```

**After:**
```python
from quantclassic.model.model_config import VAEConfig

model_config = VAEConfig(
    hidden_dim=128,
    latent_dim=16,
    n_epochs=100,
    learning_rate=0.001
)
```

#### 数据配置

**Before:**
```python
data_config = {
    'base_dir': 'rq_data_parquet',
    'window_size': 40,
    'batch_size': 512,
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15
}
```

**After:**
```python
from quantclassic.data_manager.config import DataConfig

data_config = DataConfig(
    base_dir='rq_data_parquet',
    window_size=40,
    batch_size=512,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# 自动验证比例总和 = 1.0
```

#### 预处理配置

**Before:**
```python
preprocess_config = {
    'pipeline_steps': [
        {
            'name': '填充缺失值',
            'method': 'fillna_median',
            'features': None,
            'enabled': True,
            'params': {}
        }
    ]
}
```

**After:**
```python
from quantclassic.data_processor.preprocess_config import PreprocessConfig, ProcessMethod

config = PreprocessConfig()
config.add_step(
    name='填充缺失值',
    method=ProcessMethod.FILLNA_MEDIAN,
    features=None,
    enabled=True
)
```

#### 工作流配置

**Before:**
```python
workflow_config = {
    'enabled': True,
    'recorder': {
        'experiment_name': 'my_exp',
        'log_params': True,
        'log_metrics': True
    }
}
```

**After:**
```python
from quantclassic.workflow.workflow_config import WorkflowConfig, RecorderConfig

workflow_config = WorkflowConfig(
    enabled=True,
    recorder=RecorderConfig(
        experiment_name='my_exp',
        log_params=True,
        log_metrics=True
    )
)
```

### Step 4: 使用配置加载器

**加载单个配置：**
```python
from quantclassic.config.loader import ConfigLoader
from quantclassic.model.model_config import VAEConfig

# 加载为对象
config = ConfigLoader.load('vae_config.yaml', VAEConfig)

# 或直接使用类方法
config = VAEConfig.from_yaml('vae_config.yaml')
```

**加载完整配置（多个部分）：**
```python
from quantclassic.config.loader import ConfigLoader
from quantclassic.model.model_config import VAEConfig
from quantclassic.data_manager.config import DataConfig
from quantclassic.workflow.workflow_config import WorkflowConfig

# 加载整个 YAML
full_config = ConfigLoader.load('full_config.yaml', return_dict=True)

# 提取各部分
model_config = VAEConfig.from_dict(full_config['model'])
data_config = DataConfig.from_dict(full_config['data'])
workflow_config = WorkflowConfig.from_dict(full_config['workflow'])
```

## 配置类参考

### BaseConfig

所有配置类的基类，提供统一接口：

```python
class BaseConfig:
    def validate(self) -> bool:
        """验证配置有效性"""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建"""
    
    def to_yaml(self, yaml_path: str):
        """保存到 YAML"""
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """从 YAML 加载"""
    
    def to_json(self, json_path: str):
        """保存到 JSON"""
    
    @classmethod
    def from_json(cls, json_path: str):
        """从 JSON 加载"""
    
    def update(self, **kwargs):
        """更新配置参数"""
    
    def merge(self, other):
        """合并另一个配置"""
```

### 模型配置类

- `BaseModelConfig`：所有模型的基础配置
- `VAEConfig`：VAE 模型配置
- `LSTMConfig`：LSTM 模型配置
- `GRUConfig`：GRU 模型配置
- `TransformerConfig`：Transformer 模型配置
- `MLPConfig`：MLP 模型配置

**工厂方法：**
```python
from quantclassic.model.model_config import ModelConfigFactory

# 创建配置
config = ModelConfigFactory.create('vae', hidden_dim=256)

# 使用模板
small_vae = ModelConfigFactory.get_template('vae', 'small')
large_vae = ModelConfigFactory.get_template('vae', 'large')
```

### 数据配置类

- `DataConfig`：数据管理配置（已升级为 BaseConfig 子类）

### 预处理配置类

- `ProcessingStep`：单个处理步骤
- `NeutralizeConfig`：中性化配置
- `PreprocessConfig`：预处理总配置

**模板方法：**
```python
from quantclassic.data_processor.preprocess_config import PreprocessTemplates

# 使用模板
basic = PreprocessTemplates.basic_pipeline()
advanced = PreprocessTemplates.advanced_pipeline()
alpha = PreprocessTemplates.alpha_pipeline()
```

### 工作流配置类

- `RecorderConfig`：记录器配置
- `CheckpointConfig`：检查点配置
- `ArtifactConfig`：工件配置
- `WorkflowConfig`：工作流总配置

**模板方法：**
```python
from quantclassic.workflow.workflow_config import WorkflowTemplates

# 使用模板
minimal = WorkflowTemplates.minimal()
full = WorkflowTemplates.full()
production = WorkflowTemplates.production()
```

## 常见问题

### Q1: 如何处理嵌套配置？

**A:** 使用嵌套的配置对象：

```python
from quantclassic.workflow.workflow_config import WorkflowConfig, RecorderConfig

config = WorkflowConfig(
    recorder=RecorderConfig(
        experiment_name='my_exp',
        tags={'version': 'v1.0'}
    )
)

# 转换为字典时自动递归处理
config_dict = config.to_dict()
# {
#     'recorder': {
#         'experiment_name': 'my_exp',
#         'tags': {'version': 'v1.0'}
#     }
# }
```

### Q2: 如何向后兼容旧代码？

**A:** 使用 `return_dict=True` 参数：

```python
# 旧代码仍然可以工作
config = ConfigLoader.load('config.yaml', return_dict=True)

# 或者不指定 config_class
config = ConfigLoader.load('config.yaml')
```

### Q3: 如何自定义验证逻辑？

**A:** 重写 `validate()` 方法：

```python
from dataclasses import dataclass
from quantclassic.config.base_config import BaseConfig

@dataclass
class MyConfig(BaseConfig):
    value: int = 10
    
    def validate(self) -> bool:
        if self.value < 0 or self.value > 100:
            raise ValueError("value 必须在 [0, 100] 范围内")
        return True
```

### Q4: 如何处理配置继承？

**A:** YAML 仍然支持 `BASE_CONFIG_PATH`：

```yaml
# base_config.yaml
model:
  hidden_dim: 128
  latent_dim: 16

# my_config.yaml
BASE_CONFIG_PATH: "base_config.yaml"
model:
  latent_dim: 32  # 覆盖基础配置
```

### Q5: 如何合并两个配置对象？

**A:** 使用 `merge()` 方法：

```python
config1 = VAEConfig(hidden_dim=128)
config2 = VAEConfig(latent_dim=32)

# config2 的非 None 值覆盖 config1
merged = config1.merge(config2)
```

## 最佳实践

### 1. 使用类型提示

```python
from quantclassic.model.model_config import VAEConfig

def train_model(config: VAEConfig):
    """使用类型提示提高代码可读性"""
    print(f"Training with hidden_dim={config.hidden_dim}")
```

### 2. 使用工厂模式

```python
from quantclassic.model.model_config import ModelConfigFactory

def create_model_config(model_type: str, **kwargs):
    """使用工厂简化配置创建"""
    return ModelConfigFactory.create(model_type, **kwargs)
```

### 3. 使用模板快速开始

```python
from quantclassic.model.model_config import ModelConfigFactory

# 快速原型
config = ModelConfigFactory.get_template('vae', 'small')
config.update(n_epochs=50)  # 微调参数

# 生产环境
config = ModelConfigFactory.get_template('vae', 'large')
```

### 4. 配置版本控制

```python
# 保存配置到 Git
config.to_yaml('configs/experiment_v1.0.yaml')

# 加载历史配置
old_config = VAEConfig.from_yaml('configs/experiment_v1.0.yaml')
```

### 5. 配置继承和复用

```yaml
# templates/vae_base.yaml
model:
  model_type: "vae"
  encoder_type: "gru"
  decoder_type: "gru"
  optimizer: "adam"

# experiments/exp_001.yaml
BASE_CONFIG_PATH: "../templates/vae_base.yaml"
model:
  hidden_dim: 128
  latent_dim: 16
```

## 示例：完整工作流

```python
from quantclassic.config.loader import ConfigLoader
from quantclassic.model.model_config import VAEConfig
from quantclassic.data_manager.config import DataConfig
from quantclassic.workflow.workflow_config import WorkflowConfig

# 1. 加载配置
full_config = ConfigLoader.load('config/vae_oop.yaml', return_dict=True)

# 2. 创建配置对象
model_config = VAEConfig.from_dict(full_config['model'])
data_config = DataConfig.from_dict(full_config['data'])
workflow_config = WorkflowConfig.from_dict(full_config['workflow'])

# 3. 验证配置
model_config.validate()
data_config.validate()
workflow_config.validate()

# 4. 修改配置（如果需要）
model_config.update(n_epochs=150, learning_rate=0.002)

# 5. 保存修改后的配置
model_config.to_yaml('output/updated_model_config.yaml')

# 6. 使用配置训练模型
# ... 训练代码 ...

print("配置加载和验证完成！")
```

## 总结

新的面向对象配置系统提供了：

- ✅ **更好的类型安全**：编译时类型检查
- ✅ **自动验证**：防止无效配置
- ✅ **更易维护**：清晰的结构和接口
- ✅ **向后兼容**：支持旧的字典配置
- ✅ **灵活性**：工厂模式、模板、继承

建议所有新项目使用面向对象配置，旧项目可以逐步迁移。

## 参考链接

- 配置基类：`quantclassic/config/base_config.py`
- 模型配置：`quantclassic/model/model_config.py`
- 数据配置：`quantclassic/data_manager/config.py`
- 预处理配置：`quantclassic/data_processor/preprocess_config.py`
- 工作流配置：`quantclassic/workflow/workflow_config.py`
- 配置加载器：`quantclassic/config/loader.py`
- 新配置模板：`quantclassic/config/templates/vae_oop.yaml`
