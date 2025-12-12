# QuantClassic Config System - YAML配置系统

参照 Qlib 设计的配置驱动工作流系统，实现一键运行端到端量化流程。

## 🎯 核心特性

### vs 手写代码

**之前 (手写代码)**:
```python
# 需要50-100行代码
from data_manager import DataManager, DataConfig
from model import LSTM
from Factorsystem import BacktestSystem

data_config = DataConfig(
    base_dir='rq_data_parquet',
    window_size=20,
    train_ratio=0.6,
    # ... 更多参数
)

manager = DataManager()
loaders = manager.run_full_pipeline(data_config)

model = LSTM(
    d_feat=20,
    hidden_size=64,
    # ... 更多参数
)

model.fit(loaders.train, loaders.val)
predictions = model.predict(loaders.test)

# ... 更多代码
```

**现在 (YAML配置)**:
```yaml
# config.yaml
task:
  model:
    class: LSTM
    kwargs:
      d_feat: 20
      hidden_size: 64
  
  dataset:
    class: DataManager
    kwargs:
      config:
        base_dir: rq_data_parquet
```

```bash
# 一键运行
qcrun config.yaml
```

---

## 📦 组件说明

### 1. ConfigLoader (config/loader.py)

YAML配置文件加载器

**功能**:
- ✅ YAML文件解析
- ✅ 配置继承 (`BASE_CONFIG_PATH`)
- ✅ 环境变量替换 (`${VAR}`)
- ✅ 配置验证

**示例**:
```python
from quantclassic.config import ConfigLoader

config = ConfigLoader.load('config.yaml')
```

### 2. TaskRunner (config/runner.py)

任务运行器，执行端到端流程

**功能**:
- ✅ 自动初始化数据集
- ✅ 自动初始化模型
- ✅ 自动训练
- ✅ 自动回测
- ✅ 自动记录到 workflow

**示例**:
```python
from quantclassic.config import TaskRunner

runner = TaskRunner()
results = runner.run(config, experiment_name='my_exp')
```

### 3. CLI (config/cli.py)

命令行入口

```bash
python -m quantclassic.config.cli config.yaml
# 或简写为
qcrun config.yaml
```

---

## 📝 配置文件格式

### 完整配置示例

```yaml
# 基础配置继承 (可选)
BASE_CONFIG_PATH: "base_config.yaml"

# 实验名称 (可选)
experiment_name: lstm_experiment

# QuantClassic初始化参数
quantclassic_init:
  log_level: INFO

# 任务配置
task:
  # 模型配置
  model:
    class: LSTM  # 模型类名
    module_path: quantclassic.model.pytorch_models  # 模块路径
    kwargs:  # 模型参数
      d_feat: 20
      hidden_size: 64
      num_layers: 2
      dropout: 0.3
      n_epochs: 100
      lr: 0.001
      early_stop: 10
      batch_size: 256
  
  # 数据集配置
  dataset:
    class: DataManager
    module_path: quantclassic.data_manager.manager
    kwargs:
      config:
        base_dir: rq_data_parquet
        window_size: 20
        train_ratio: 0.6
        val_ratio: 0.2
        batch_size: 256
        shuffle: true
  
  # 回测配置 (可选)
  backtest:
    n_groups: 10
    save_plots: true
    output_dir: output/backtest
```

### 简化配置 (使用default_module)

```yaml
task:
  model:
    class: LSTM  # 会从 quantclassic.model 加载
    kwargs:
      d_feat: 20
      hidden_size: 64
```

### 配置继承示例

**base_config.yaml**:
```yaml
quantclassic_init:
  log_level: INFO

task:
  dataset:
    class: DataManager
    kwargs:
      config:
        base_dir: rq_data_parquet
        window_size: 20
```

**my_config.yaml**:
```yaml
BASE_CONFIG_PATH: "base_config.yaml"

task:
  model:
    class: LSTM
    kwargs:
      d_feat: 20
      hidden_size: 64
  
  # dataset会从base_config继承
```

### 环境变量替换

```yaml
task:
  dataset:
    class: DataManager
    kwargs:
      config:
        base_dir: ${DATA_DIR}/parquet  # 使用环境变量
        user: ${USER:default_user}     # 带默认值
```

---

## 🚀 使用指南

### 方式1: CLI运行

```bash
# 基础使用
qcrun config/templates/lstm_basic.yaml

# 或使用Python模块
python -m quantclassic.config.cli config.yaml
```

### 方式2: Python代码

```python
from quantclassic.config import ConfigLoader, TaskRunner

# 加载配置
config = ConfigLoader.load('config.yaml')

# 运行任务
runner = TaskRunner()
results = runner.run(config, experiment_name='my_experiment')

# 获取结果
model = results['model']
dataset = results['dataset']
train_results = results['train_results']
```

### 方式3: 与Workflow集成

```python
from quantclassic.config import ConfigLoader, TaskRunner
from quantclassic.workflow import R

# 加载配置
config = ConfigLoader.load('config.yaml')

# TaskRunner会自动使用R记录实验
runner = TaskRunner()
results = runner.run(config, experiment_name='auto_recorded')

# 实验自动记录到 output/experiments/
```

---

## 📂 配置模板

### LSTM模板 (config/templates/lstm_basic.yaml)

```yaml
task:
  model:
    class: LSTM
    module_path: quantclassic.model.pytorch_models
    kwargs:
      d_feat: 20
      hidden_size: 64
      num_layers: 2
      dropout: 0.3
      n_epochs: 100
      lr: 0.001
  
  dataset:
    class: DataManager
    module_path: quantclassic.data_manager.manager
    kwargs:
      config:
        base_dir: rq_data_parquet
        window_size: 20
```

### GRU模板 (config/templates/gru_basic.yaml)

```yaml
task:
  model:
    class: GRU
    module_path: quantclassic.model.pytorch_models
    kwargs:
      d_feat: 20
      hidden_size: 64
      num_layers: 2
      dropout: 0.3
      n_epochs: 100
```

### Transformer模板

```yaml
task:
  model:
    class: Transformer
    module_path: quantclassic.model.pytorch_models
    kwargs:
      d_feat: 20
      d_model: 64
      nhead: 4
      num_layers: 2
      dropout: 0.3
```

---

## 🔧 高级功能

### 1. 自定义模块路径

```yaml
task:
  model:
    class: MyCustomModel
    module_path: my_package.models  # 自定义路径
    kwargs:
      param1: value1
```

### 2. 配置验证

```python
from quantclassic.config import ConfigLoader

config = ConfigLoader.load('config.yaml')

# 自动验证
try:
    ConfigLoader.validate(config)
    print("✅ 配置有效")
except ValueError as e:
    print(f"❌ 配置无效: {e}")
```

### 3. 动态配置

```python
from quantclassic.config import ConfigLoader

# 加载基础配置
base_config = ConfigLoader.load('base.yaml')

# 修改配置
base_config['task']['model']['kwargs']['hidden_size'] = 128

# 运行
runner = TaskRunner()
results = runner.run(base_config)
```

### 4. 保存配置

```python
from quantclassic.config import ConfigLoader

config = {
    'task': {
        'model': {'class': 'LSTM', 'kwargs': {'d_feat': 20}}
    }
}

ConfigLoader.save(config, 'output/saved_config.yaml')
```

---

## 📊 与Qlib对比

| 功能 | Qlib | QuantClassic |
|------|------|--------------|
| YAML配置 | ✅ | ✅ |
| 配置继承 | ✅ | ✅ |
| 环境变量 | ✅ | ✅ |
| CLI运行 | ✅ `qrun` | ✅ `qcrun` |
| 自动记录 | ✅ | ✅ (workflow) |
| 配置验证 | ✅ | ✅ |
| 模板系统 | ✅ | ✅ |

---

## 🎯 实战案例

### 案例1: 快速实验

```bash
# 1. 复制模板
cp config/templates/lstm_basic.yaml my_experiment.yaml

# 2. 修改参数
vim my_experiment.yaml  # 修改hidden_size等

# 3. 运行
qcrun my_experiment.yaml
```

### 案例2: 超参数搜索

创建多个配置文件:

**lstm_h64.yaml**:
```yaml
experiment_name: lstm_h64
task:
  model:
    class: LSTM
    kwargs:
      hidden_size: 64
```

**lstm_h128.yaml**:
```yaml
experiment_name: lstm_h128
task:
  model:
    class: LSTM
    kwargs:
      hidden_size: 128
```

批量运行:
```bash
for config in lstm_h*.yaml; do
    qcrun $config
done
```

### 案例3: 生产部署

**production.yaml**:
```yaml
BASE_CONFIG_PATH: "base_config.yaml"

experiment_name: production_model

task:
  model:
    class: LSTM
    kwargs:
      n_epochs: 200  # 更多epochs
      early_stop: 20
  
  backtest:
    n_groups: 20  # 更详细的回测
    save_plots: true
```

```bash
qcrun production.yaml
```

---

## 🐛 故障排除

### 问题1: 配置文件找不到

```bash
❌ 错误: 配置文件不存在: config.yaml
```

**解决**: 使用绝对路径或检查当前目录

```bash
qcrun /absolute/path/to/config.yaml
# 或
cd /path/to/configs && qcrun config.yaml
```

### 问题2: 模块导入失败

```
ModuleNotFoundError: No module named 'xxx'
```

**解决**: 检查module_path是否正确

```yaml
model:
  class: LSTM
  module_path: quantclassic.model.pytorch_models  # 确保正确
```

### 问题3: 配置验证失败

```
ValueError: 配置中缺少 'task' 字段
```

**解决**: 确保配置包含必需字段

```yaml
task:  # 必需
  model:  # 至少需要model或dataset之一
    ...
```

---

## 📚 更多资源

- [配置模板目录](./templates/)
- [Workflow文档](../workflow/README.md)
- [Model文档](../model/README.md)
- [Data Manager文档](../data_manager/README.md)

---

## 🎉 总结

QuantClassic Config System 提供了:

1. ✅ **简化配置**: YAML替代手写代码
2. ✅ **一键运行**: `qcrun config.yaml`
3. ✅ **自动记录**: 集成workflow系统
4. ✅ **配置复用**: 继承和模板
5. ✅ **Qlib兼容**: 相似的API和用法

现在可以像使用Qlib一样，用配置文件驱动整个量化研究流程！🚀
