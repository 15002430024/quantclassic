# QuantClassic 配置系统运行指南

## 🚀 快速开始

### 方法1: 从 quantclassic 目录运行（推荐）

```bash
cd /home/u2025210237/jupyterlab/quantclassic
python -m config.cli config/examples/vae_full_pipeline_example.yaml
```

### 方法2: 从 jupyterlab 目录运行

```bash
cd /home/u2025210237/jupyterlab
python -m quantclassic.config.cli quantclassic/config/examples/vae_full_pipeline_example.yaml
```

### 方法3: 在 Python 代码中使用

```python
import sys
sys.path.insert(0, '/home/u2025210237/jupyterlab')

from quantclassic.config import ConfigLoader, TaskRunner

# 加载配置
config = ConfigLoader.load('path/to/config.yaml')

# 运行任务
runner = TaskRunner()
results = runner.run(config, experiment_name='my_experiment')
```

---

## ⚠️ 常见问题

### 问题1: ModuleNotFoundError: No module named 'quantclassic'

**原因**: Python 找不到 `quantclassic` 模块

**解决方案**:
1. 确保从正确的目录运行
2. 或者添加路径到 Python：
   ```bash
   export PYTHONPATH=/home/u2025210237/jupyterlab:$PYTHONPATH
   python -m quantclassic.config.cli config.yaml
   ```

### 问题2: No module named 'torch'

**原因**: 缺少 PyTorch 依赖

**解决方案**:
```bash
# 安装 PyTorch (CPU 版本)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 或安装 CUDA 版本
pip install torch
```

### 问题3: 配置验证失败

**原因**: YAML 配置格式不正确

**检查清单**:
- ✅ `task` 中必须包含 `dataset` 和 `model` 字段
- ✅ `class` 字段使用完整路径: `"quantclassic.data_manager.DataManager"`
- ✅ 不要使用 `module` 字段（应该用 `module_path` 或完整路径）
- ✅ YAML 缩进正确（使用2个空格）

### 问题4: 'dict' object has no attribute 'log_level'

**原因**: DataManager 期望 DataConfig 对象,但收到了 dict

**状态**: ✅ **已在 DataManager v1.1 修复!** 现在自动支持 dict 参数

**正确配置格式**:
```yaml
task:
  dataset:
    class: "quantclassic.data_manager.DataManager"
    kwargs:
      # 直接传递参数,不要嵌套在 config 字典中
      base_dir: "output"
      data_file: "train_data_final_01.parquet"
      window_size: 40
      batch_size: 512
      # ... 其他参数
```

**❌ 错误格式** (不要使用):
```yaml
task:
  dataset:
    kwargs:
      config:  # ❌ 不要嵌套在 config 中
        base_dir: "output"
        # ...
```

---

## 📋 配置文件要求

### 必需字段

```yaml
experiment_name: "my_experiment"  # 实验名称

task:
  # 数据集配置（必需）
  dataset:
    class: "quantclassic.data_manager.DataManager"  # 完整类路径
    kwargs:
      config:
        base_dir: "output"
        data_file: "data.parquet"
  
  # 模型配置（必需）
  model:
    class: "quantclassic.model.TimeSeriesVAE"  # 完整类路径
    kwargs:
      hidden_dim: 128
      latent_dim: 16
```

### 可选字段

```yaml
# 工作流管理（可选）
workflow:
  enabled: true
  recorder:
    experiment_name: "my_experiment"

# 回测配置（可选）
task:
  backtest:
    enabled: true
    output_dir: "output/backtest"
```

---

## 🔧 配置示例

### 最小配置

```yaml
experiment_name: "minimal_test"

task:
  dataset:
    class: "quantclassic.data_manager.DataManager"
    kwargs:
      config:
        base_dir: "output"
        data_file: "data.parquet"
        window_size: 20
        batch_size: 128
  
  model:
    class: "quantclassic.model.TimeSeriesVAE"
    kwargs:
      hidden_dim: 64
      latent_dim: 8
      n_epochs: 10
```

### 完整配置

参考 `vae_full_pipeline_example.yaml`，包含：
- ✅ 数据提取
- ✅ 数据预处理
- ✅ 数据管理
- ✅ 模型训练
- ✅ 因子回测
- ✅ 工作流管理

---

## 💡 使用技巧

### 1. 快速测试配置是否正确

```bash
python -c "
import yaml
from pathlib import Path
config = yaml.safe_load(Path('config.yaml').read_text())
assert 'task' in config
assert 'dataset' in config['task']
assert 'model' in config['task']
print('✅ 配置格式正确')
"
```

### 2. 查看配置内容

```bash
python -c "
import yaml
from pathlib import Path
config = yaml.safe_load(Path('config.yaml').read_text())
import json
print(json.dumps(config, indent=2, ensure_ascii=False))
"
```

### 3. 批量运行多个配置

```bash
for config in config/*.yaml; do
    echo "运行: $config"
    python -m config.cli "$config"
done
```

---

## 📚 相关文档

- [配置文件示例](./examples/)
- [完整流程指南](./examples/FULL_PIPELINE_GUIDE.md)
- [YAML 配置说明](./templates/YAML_USAGE_GUIDE.md)
- [快速开始](./QUICKSTART.md)
---

## 🆕 训练架构 (2026-01 重构)

### 新训练器层次

```
model/train/
├── base_trainer.py          # BaseTrainer 基类
├── simple_trainer.py        # SimpleTrainer 常规训练
├── rolling_window_trainer.py # RollingWindowTrainer 滚动窗口
└── rolling_daily_trainer.py  # RollingDailyTrainer 日级滚动
```

### TaskConfig 训练器选项

```yaml
task:
  # 选择训练器
  trainer_class: RollingDailyTrainer  # 可选: SimpleTrainer, RollingWindowTrainer, RollingDailyTrainer
  
  # 数据加载器选项
  use_rolling_loaders: true   # 启用滚动窗口加载器
  use_daily_loaders: false    # 启用日批次加载器
  
  # 训练器参数
  trainer_kwargs:
    n_epochs: 20
    lr: 0.001
    early_stop: 10
    loss_fn: mse              # 可选: mse, mae, huber, ic, ic_corr
    lambda_corr: 0.01         # 相关性正则化权重
    weight_inheritance: true  # 滚动训练时继承权重
    save_each_window: true    # 保存每个窗口模型
```

### 训练器对比

| 特性 | SimpleTrainer | RollingWindowTrainer | RollingDailyTrainer |
|------|---------------|---------------------|---------------------|
| 单窗口训练 | ✅ | - | - |
| 滚动窗口 | - | ✅ | ✅ |
| 权重继承 | - | ✅ | ✅ |
| 显存管理 | - | - | ✅ |
| 日级预测 | - | - | ✅ |

### ⚠️ 废弃通知

1. **`DynamicGraphTrainer`** 已废弃，改用 `SimpleTrainer` + `use_daily_loaders`
2. **`DataManager.create_rolling_window_trainer()`** 已移除，请使用 `model.train.RollingDailyTrainer`
3. **`model.rolling_daily_trainer`** 模块已改为 shim，请改用 `model.train`