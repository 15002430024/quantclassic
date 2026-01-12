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
---

## 🆕 训练架构 (2026-01 重构)

### 推荐训练器

| 训练器 | 用途 | 适用场景 |
|--------|------|----------|
| `SimpleTrainer` | 常规单窗口训练 | 常规训练、验证 |
| `RollingWindowTrainer` | 滚动窗口训练 | Walk-Forward 验证 |
| `RollingDailyTrainer` | 日级滚动训练 | 高频模型切换、动态图 |

### 示例: 滚动窗口训练

```yaml
experiment_name: rolling_training

task:
  # 数据配置
  dataset:
    class: DataManager
    kwargs:
      config:
        base_dir: rq_data_parquet
        split_strategy: rolling  # 启用滚动窗口
        rolling_window_days: 60
        rolling_test_days: 5
  
  # 模型配置
  model:
    class: HybridGraphModel
    kwargs:
      d_feat: 20
      hidden_size: 64
  
  # 🆕 训练器配置
  trainer_class: RollingDailyTrainer
  use_rolling_loaders: true
  trainer_kwargs:
    n_epochs: 20
    weight_inheritance: true   # 继承上一窗口权重
    save_each_window: true     # 保存每个窗口模型
    gc_interval: 5             # 显存清理间隔
```

### 示例: 动态图训练

```yaml
task:
  dataset:
    kwargs:
      config:
        graph_builder_config:
          type: hybrid
          alpha: 0.7
          top_k: 10
  
  trainer_class: SimpleTrainer  # 或 RollingDailyTrainer
  use_daily_loaders: true       # 启用日批次加载器
  trainer_kwargs:
    loss_fn: ic_corr            # IC损失 + 相关性正则
    lambda_corr: 0.01
```

### Python API 示例

```python
from quantclassic.model.train import RollingDailyTrainer, RollingTrainerConfig
from quantclassic.data_set import DataManager

# 1. 准备数据
dm = DataManager(config=data_config)
dm.run_full_pipeline()
rolling_loaders = dm.create_rolling_daily_loaders()

# 2. 定义模型工厂
def model_factory():
    return MyModel(d_feat=len(dm.feature_cols))

# 3. 创建训练器并训练
config = RollingTrainerConfig(n_epochs=20, weight_inheritance=True)
trainer = RollingDailyTrainer(model_factory, config)
trainer.fit(rolling_loaders, save_dir='output/models')

# 4. 获取预测
predictions = trainer.get_all_predictions()
```

### ⚠️ 废弃 API

以下 API 已废弃，请迁移到新训练架构:

| 废弃 API | 替代方案 |
|----------|----------|
| `DataManager.create_rolling_window_trainer()` | `model.train.RollingWindowTrainer` |
| `model.rolling_daily_trainer.RollingDailyTrainer` | `model.train.RollingDailyTrainer` |
| `trainer_class='DynamicGraphTrainer'` | `trainer_class='SimpleTrainer'` |