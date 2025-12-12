# QuantClassic 配置文件示例

本目录包含各种 YAML 配置文件示例，展示如何使用 QuantClassic 的配置驱动功能。

---

## 📂 文件列表

### 基础模板

| 文件 | 说明 | 适用场景 |
|------|------|----------|
| `../templates/vae_basic.yaml` | 基础 VAE 配置 | 快速开始、学习使用 |
| `../templates/vae_advanced.yaml` | 完整 VAE 配置（含注释） | 生产环境、完整功能 |

### 实战示例

| 文件 | 说明 | 适用场景 |
|------|------|----------|
| `vae_custom_preprocessing.yaml` | 自定义数据预处理 | 不同特征需要不同标准化方法 |
| `vae_large_batch.yaml` | 大批次训练 | 大GPU显存环境（16GB+） |
| `vae_grid_search.yaml` | 超参数网格搜索 | 批量测试不同参数 |

### Python 脚本

| 文件 | 说明 |
|------|------|
| `run_vae_from_config.py` | 命令行工具，支持参数覆盖 |

---

## 🚀 快速开始

### 1. 使用基础模板

```bash
cd quantclassic
python -m config.cli config/templates/vae_basic.yaml
```

### 2. 使用高级模板（推荐）

```bash
python -m config.cli config/templates/vae_advanced.yaml
```

### 3. 使用自定义配置

```bash
python -m config.cli config/examples/vae_custom_preprocessing.yaml
```

---

## 📖 使用指南

### 方式1: 直接运行配置文件

```bash
# 进入 quantclassic 目录
cd /path/to/quantclassic

# 运行配置
python -m config.cli config/examples/vae_custom_preprocessing.yaml
```

### 方式2: 使用 Python 脚本（支持参数覆盖）

```bash
# 基本用法
python config/examples/run_vae_from_config.py --config templates/vae_basic.yaml

# 覆盖实验名称
python config/examples/run_vae_from_config.py \
    --config templates/vae_basic.yaml \
    --exp my_experiment

# 覆盖超参数
python config/examples/run_vae_from_config.py \
    --config templates/vae_basic.yaml \
    --latent-dim 32 \
    --batch-size 1024 \
    --lr 0.002 \
    --n-epochs 100

# 指定设备
python config/examples/run_vae_from_config.py \
    --config templates/vae_basic.yaml \
    --device cuda
```

### 方式3: Python 代码

```python
from quantclassic.config import ConfigLoader, TaskRunner

# 加载配置
config = ConfigLoader.load('quantclassic/config/templates/vae_advanced.yaml')

# 运行任务
runner = TaskRunner()
results = runner.run(config, experiment_name='my_vae_exp')

# 查看结果
print(f"IC均值: {results['metrics']['ic_mean']:.4f}")
print(f"模型路径: {results['model_path']}")
```

---

## 🔧 配置文件详解

### 1. 自定义数据预处理 (`vae_custom_preprocessing.yaml`)

**特点**: 针对不同类型的特征使用不同的标准化方法

```yaml
preprocessor:
  config:
    pipeline_steps:
      # 价格类 → Z-Score
      - name: "价格类Z-Score"
        method: "z_score"
        features: ["close", "open", "high", "low"]
      
      # 技术指标 → MinMax [0,1]
      - name: "技术指标MinMax"
        method: "minmax"
        features: ["rsi", "kdj_k", "macd"]
        params:
          feature_range: [0, 1]
      
      # 成交量 → 秩归一化 [-1,1]
      - name: "成交量Rank"
        method: "rank"
        features: ["volume", "amount"]
        params:
          output_range: [-1, 1]
```

**使用场景**:
- ✅ 特征类型多样（价格、技术指标、成交量等）
- ✅ 需要精细控制每类特征的处理方式
- ✅ 希望保留特征的原始分布特性

**运行**:
```bash
python -m quantclassic.config.cli config/examples/vae_custom_preprocessing.yaml
```

---

### 2. 大批次训练 (`vae_large_batch.yaml`)

**特点**: 针对大GPU显存优化，使用大批次和更大模型

```yaml
dataset:
  config:
    batch_size: 2048      # 大批次
    num_workers: 8        # 多进程加载
    window_size: 60       # 更长窗口

model:
  kwargs:
    hidden_dim: 256       # 更大隐藏层
    latent_dim: 32        # 更大潜在空间
    lr: 0.002             # 大批次用更大学习率
    
    # 学习率调度器
    scheduler: "ReduceLROnPlateau"
    scheduler_params:
      factor: 0.5
      patience: 5
```

**使用场景**:
- ✅ GPU显存 ≥ 16GB
- ✅ 系统内存 ≥ 32GB
- ✅ 需要更快的训练速度
- ✅ 需要更大的模型容量

**优势**:
- 🚀 更稳定的梯度估计
- 🚀 更快的训练速度（每个epoch）
- 🚀 可以使用更大的模型

**运行**:
```bash
python -m quantclassic.config.cli config/examples/vae_large_batch.yaml
```

---

### 3. 超参数网格搜索 (`vae_grid_search.yaml`)

**特点**: 批量测试不同的超参数配置

**使用方法**:

#### 方式1: 手动复制修改

```bash
# 复制模板
cp vae_grid_search.yaml vae_latent8.yaml
cp vae_grid_search.yaml vae_latent16.yaml
cp vae_grid_search.yaml vae_latent32.yaml

# 修改 latent_dim（手动或使用 sed）
# vae_latent8.yaml → latent_dim: 8
# vae_latent16.yaml → latent_dim: 16
# vae_latent32.yaml → latent_dim: 32

# 批量运行
for config in vae_latent*.yaml; do
    python -m quantclassic.config.cli $config
done
```

#### 方式2: 使用脚本参数覆盖

```bash
for latent in 8 16 32; do
    python config/examples/run_vae_from_config.py \
        --config templates/vae_basic.yaml \
        --exp vae_latent${latent} \
        --latent-dim $latent
done
```

#### 方式3: 完整的网格搜索脚本

```bash
#!/bin/bash

# 参数网格
LATENT_DIMS=(8 16 32)
BATCH_SIZES=(256 512 1024)
LEARNING_RATES=(0.0005 0.001 0.002)

# 遍历所有组合
for latent in "${LATENT_DIMS[@]}"; do
    for batch in "${BATCH_SIZES[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            exp_name="vae_l${latent}_b${batch}_lr${lr}"
            
            echo "========================================="
            echo "Running: ${exp_name}"
            echo "========================================="
            
            python config/examples/run_vae_from_config.py \
                --config templates/vae_basic.yaml \
                --exp ${exp_name} \
                --latent-dim ${latent} \
                --batch-size ${batch} \
                --lr ${lr}
        done
    done
done
```

---

## 📊 配置文件结构

所有配置文件遵循统一结构:

```yaml
experiment_name: <实验名称>

task:
  # 1. 数据预处理 (可选)
  preprocessor:
    class: DataPreprocessor
    kwargs:
      config:
        pipeline_steps: [...]
  
  # 2. 数据集 (必需)
  dataset:
    class: DataManager
    kwargs:
      config:
        base_dir: ...
        batch_size: ...
  
  # 3. 模型 (必需)
  model:
    class: VAE
    kwargs:
      latent_dim: ...
      n_epochs: ...
  
  # 4. 回测 (可选)
  backtest:
    class: FactorBacktestSystem
    kwargs:
      config:
        n_groups: ...
```

---

## 🎯 常见配置任务

### 任务1: 修改批次大小

```yaml
dataset:
  kwargs:
    config:
      batch_size: 512  # ← 修改这里

model:
  kwargs:
    batch_size: 512    # ← 保持一致
```

### 任务2: 修改数据预处理

```yaml
preprocessor:
  kwargs:
    config:
      pipeline_steps:
        # 添加新的处理步骤
        - name: "我的预处理"
          method: "z_score"  # 或 minmax, rank, winsorize
          features: ["feature1", "feature2"]
          enabled: true
          params: {}
```

### 任务3: 修改训练参数

```yaml
model:
  kwargs:
    n_epochs: 100      # 训练轮数
    lr: 0.001          # 学习率
    early_stop: 15     # 早停patience
    latent_dim: 16     # 潜在维度
```

### 任务4: 启用/禁用某个步骤

```yaml
pipeline_steps:
  - name: "市值行业中性化"
    method: "ols_neutralize"
    enabled: false     # ← 设为 false 禁用
```

---

## 📚 更多资源

- **详细文档**: `../templates/YAML_USAGE_GUIDE.md`
- **完整模板**: `../templates/vae_advanced.yaml`
- **Notebook 示例**: `/jupyterlab/vae.ipynb`

---

## ❓ 常见问题

### Q1: 如何只修改部分参数？

A: 使用脚本的参数覆盖功能:
```bash
python config/examples/run_vae_from_config.py \
    --config templates/vae_basic.yaml \
    --latent-dim 32 \
    --lr 0.002
```

### Q2: 如何批量运行多个配置？

A: 使用 Bash 循环:
```bash
for config in config/examples/*.yaml; do
    python -m quantclassic.config.cli $config
done
```

### Q3: 配置文件报错怎么办？

A: 检查:
1. YAML 语法是否正确（缩进、冒号）
2. 文件路径是否存在
3. 参数类型是否匹配（整数、浮点数、字符串）

### Q4: 如何复现实验？

A: 
1. 保存配置文件到版本控制
2. 设置固定随机种子: `seed: 42`
3. 使用 workflow 自动记录所有参数

---

**🎉 祝你使用愉快！**
