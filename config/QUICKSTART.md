# QuantClassic Config - 快速开始

## 🚀 5分钟上手

### 1. 查看可用模板

```bash
ls quantclassic/config/templates/
# lstm_basic.yaml  gru_basic.yaml
```

### 2. 复制并修改模板

```bash
cp quantclassic/config/templates/lstm_basic.yaml my_experiment.yaml
```

编辑 `my_experiment.yaml`:
```yaml
task:
  model:
    class: LSTM
    module_path: quantclassic.model.pytorch_models
    kwargs:
      d_feat: 20
      hidden_size: 128  # 修改这里
      num_layers: 2
      n_epochs: 50      # 修改这里
```

### 3. 运行实验

```bash
# 方式1: 使用Python模块
cd quantclassic
python -m config.cli my_experiment.yaml

# 方式2: 使用Python代码
python << EOF
from config import ConfigLoader, TaskRunner

config = ConfigLoader.load('my_experiment.yaml')
runner = TaskRunner()
results = runner.run(config, experiment_name='my_exp')
print(f"完成! 模型: {type(results['model']).__name__}")
EOF
```

### 4. 查看结果

```bash
# 实验自动记录在
ls output/experiments/my_exp_*/

# 包含:
# - meta.json         # 元数据
# - artifacts/        # 保存的对象（模型等）
```

---

## 📝 常用配置

### 修改数据路径

```yaml
task:
  dataset:
    kwargs:
      config:
        base_dir: /path/to/your/data  # 修改这里
```

### 修改模型参数

```yaml
task:
  model:
    kwargs:
      hidden_size: 128    # 隐藏层大小
      num_layers: 3       # 层数
      dropout: 0.5        # Dropout
      n_epochs: 200       # 训练轮数
      lr: 0.0001          # 学习率
```

### 修改训练/验证/测试划分

```yaml
task:
  dataset:
    kwargs:
      config:
        train_ratio: 0.7  # 70% 训练
        val_ratio: 0.15   # 15% 验证
        # 剩余15%自动用于测试
```

---

## 🎯 实用示例

### 示例1: 快速测试（少量epoch）

```yaml
experiment_name: quick_test

task:
  model:
    class: LSTM
    kwargs:
      d_feat: 20
      hidden_size: 64
      n_epochs: 5        # 只训练5个epoch
      early_stop: 3
```

```bash
python -m config.cli quick_test.yaml
```

### 示例2: 对比不同hidden_size

创建3个配置:

**h32.yaml**:
```yaml
experiment_name: lstm_h32
task:
  model:
    kwargs:
      hidden_size: 32
```

**h64.yaml**:
```yaml
experiment_name: lstm_h64
task:
  model:
    kwargs:
      hidden_size: 64
```

**h128.yaml**:
```yaml
experiment_name: lstm_h128
task:
  model:
    kwargs:
      hidden_size: 128
```

批量运行:
```bash
for config in h*.yaml; do
    python -m config.cli $config
done
```

查看对比:
```python
from workflow import R

# 查看所有实验
experiments = R.list_experiments()
for exp in experiments:
    if exp['name'].startswith('lstm_h'):
        recs = R.list_recorders(exp['name'])
        print(f"{exp['name']}: {len(recs)} runs")
```

### 示例3: 使用配置继承

**base.yaml** (基础配置):
```yaml
task:
  dataset:
    class: DataManager
    kwargs:
      config:
        base_dir: rq_data_parquet
        window_size: 20
        train_ratio: 0.6
        val_ratio: 0.2
```

**experiment1.yaml** (继承base):
```yaml
BASE_CONFIG_PATH: "base.yaml"

experiment_name: exp1

task:
  model:
    class: LSTM
    kwargs:
      hidden_size: 64
```

**experiment2.yaml** (继承base):
```yaml
BASE_CONFIG_PATH: "base.yaml"

experiment_name: exp2

task:
  model:
    class: GRU
    kwargs:
      hidden_size: 64
```

---

## 🔧 命令行快捷方式

### 创建别名 (可选)

在 `~/.bashrc` 或 `~/.zshrc` 中添加:

```bash
# 添加到配置文件
echo 'alias qcrun="python -m quantclassic.config.cli"' >> ~/.bashrc
source ~/.bashrc

# 现在可以使用
qcrun my_config.yaml
```

---

## 📊 与手写代码对比

### 手写代码 (❌ 繁琐)

```python
# 需要50-100行
from quantclassic.data_manager import DataManager, DataConfig
from quantclassic.model import LSTM
from quantclassic.workflow import R

# 配置数据
data_config = DataConfig(
    base_dir='rq_data_parquet',
    window_size=20,
    train_ratio=0.6,
    val_ratio=0.2,
    batch_size=256,
    shuffle=True
)

# 准备数据
manager = DataManager()
loaders = manager.run_full_pipeline(data_config)

# 配置模型
model = LSTM(
    d_feat=20,
    hidden_size=64,
    num_layers=2,
    dropout=0.3,
    n_epochs=100,
    lr=0.001,
    early_stop=10,
    batch_size=256
)

# 训练
with R.start(experiment_name='manual_exp'):
    R.log_params(
        hidden_size=64,
        num_layers=2,
        # ... 更多参数
    )
    
    model.fit(loaders.train, loaders.val)
    
    R.log_metrics(**model.best_metrics)
    R.save_objects(model=model)
```

### 使用Config (✅ 简洁)

**config.yaml**:
```yaml
task:
  model:
    class: LSTM
    kwargs:
      d_feat: 20
      hidden_size: 64
      num_layers: 2
      dropout: 0.3
      n_epochs: 100
      lr: 0.001
  
  dataset:
    class: DataManager
    kwargs:
      config:
        base_dir: rq_data_parquet
        window_size: 20
```

```bash
python -m config.cli config.yaml
```

---

## ⚡ 效率提升

| 任务 | 手写代码 | 使用Config | 提升 |
|------|---------|-----------|------|
| **代码行数** | 50-100行 | 10-20行 | **5倍** |
| **配置时间** | 5-10分钟 | 1-2分钟 | **5倍** |
| **出错概率** | 高 | 低 | **10倍** |
| **复用性** | 低 | 高 | **完美** |

---

## 📚 下一步

1. 阅读 [完整文档](./README.md)
2. 查看 [配置模板](./templates/)
3. 学习 [Workflow系统](../workflow/README.md)
4. 探索 [高级功能](./README.md#高级功能)

---

**开始你的第一个配置驱动实验吧！** 🚀
