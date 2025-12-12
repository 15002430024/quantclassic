# VAE 全流程配置示例使用指南

## 📋 概述

`vae_full_pipeline_example.yaml` 是一个完整的、开箱即用的配置文件模板，包含从数据提取到回测的所有环节。你只需修改一个 YAML 文件就可以完成全部自定义。

## 🎯 配置文件结构

```yaml
vae_full_pipeline_example.yaml
├── experiment_name          # 实验名称
├── data_extraction          # 第1步: 数据提取（可选）
├── data_preprocessing       # 第2步: 数据预处理
├── data_management          # 第3步: 数据管理
├── model_training           # 第4步: 模型训练
├── factor_backtest          # 第5步: 因子回测
├── task                     # 第6步: 任务配置（必需！）
│   ├── dataset              # 数据集配置
│   ├── model                # 模型配置
│   └── backtest             # 回测配置
└── workflow                 # 实验管理配置
```

## 🚀 快速开始

### 方法1: 使用命令行工具

```bash
cd quantclassic
python -m config.cli config/examples/vae_full_pipeline_example.yaml
```

### 方法2: 复制并自定义

```bash
# 复制模板
cp config/examples/vae_full_pipeline_example.yaml my_vae_experiment.yaml

# 编辑配置（修改 🔧 标记的参数）
vim my_vae_experiment.yaml

# 运行
python -m config.cli my_vae_experiment.yaml
```

### 方法3: 在代码中使用

```python
from quantclassic.config import ConfigLoader, TaskRunner

# 加载配置
loader = ConfigLoader()
config = loader.load('config/examples/vae_full_pipeline_example.yaml')

# 运行任务
runner = TaskRunner()
results = runner.run(config, experiment_name='my_experiment')
```

---

## 🔧 核心配置说明

### ⚠️ 重要：task 配置（必需）

`task` 配置段是运行的核心，**必须包含 `dataset` 和 `model` 字段**：

```yaml
task:
  # 数据集配置（必需）
  dataset:
    class: "quantclassic.data_manager.DataManager"  # 完整类路径
    kwargs:
      config:
        base_dir: "output"
        data_file: "train_data_final_01.parquet"
        window_size: 40
        batch_size: 512
  
  # 模型配置（必需）
  model:
    class: "quantclassic.model.TimeSeriesVAE"  # 完整类路径
    kwargs:
      hidden_dim: 128
      latent_dim: 16
      n_epochs: 100
  
  # 回测配置（可选）
  backtest:
    enabled: false
```

**配置说明**：
- `class`: 使用完整的类路径格式 `模块路径.类名`
- 或者使用 `module_path` 和 `class` 分开指定（不推荐）
- `kwargs`: 传递给类构造函数的参数

### 1️⃣ 数据提取配置

**位置**: `data_extraction`

```yaml
data_extraction:
  enabled: false  # 🔧 改为 true 启用数据提取
  
  kwargs:
    config:
      # 股票池
      universe:
        method: "index_components"  # 🔧 可选: index_components, custom_list
        params:
          index_code: "000300.XSHG"  # 🔧 修改为其他指数
      
      # 时间范围
      start_date: "2020-01-01"  # 🔧 修改开始日期
      end_date: "2023-12-31"    # 🔧 修改结束日期
      
      # 特征
      features:
        price_features: ["close", "open", "high", "low"]  # 🔧 自定义
        technical_indicators: ["rsi", "macd", "kdj_k"]    # 🔧 自定义
```

**常用配置**:

| 场景 | 配置 |
|------|------|
| 沪深300 | `index_code: "000300.XSHG"` |
| 中证500 | `index_code: "000905.XSHG"` |
| 创业板 | `index_code: "399006.XSHE"` |
| 自定义股票 | `method: "custom_list"`, `custom_codes: [...]` |

---

### 2️⃣ 数据预处理配置

**位置**: `data_preprocessing.pipeline_steps`

```yaml
pipeline_steps:
  # 价格类 → Z-Score
  - name: "价格类Z-Score"
    method: "z_score"
    features: ["close", "open", "high", "low"]  # 🔧 自定义
    enabled: true
  
  # 技术指标 → MinMax
  - name: "技术指标MinMax"
    method: "minmax"
    features: ["rsi", "kdj_k", "macd"]  # 🔧 自定义
    params:
      feature_range: [0, 1]  # 🔧 修改范围
  
  # 成交量 → Rank
  - name: "成交量秩归一化"
    method: "rank"
    features: ["volume", "amount"]  # 🔧 自定义
    params:
      output_range: [-1, 1]  # 🔧 修改范围
```

**预处理方法速查**:

| 方法 | YAML值 | 适用场景 | 参数示例 |
|------|--------|----------|----------|
| Z-Score | `z_score` | 正态分布特征 | `{}` |
| MinMax | `minmax` | 有界特征 | `feature_range: [0,1]` |
| 秩归一化 | `rank` | 不规则分布 | `output_range: [-1,1]` |
| 去极值 | `winsorize` | 有异常值 | `limits: [0.025, 0.025]` |
| 裁剪 | `clip` | 处理无穷值 | `lower: -1e10, upper: 1e10` |
| 填充缺失值 | `fillna_median` | 有缺失值 | `{}` |
| 市值行业中性化 | `ols_neutralize` | Alpha因子 | `{}` |
| 相似股票中性化 | `simstock_neutralize` | 去除相关性 | `{}` |

**混合策略示例**:

```yaml
# 策略1: 全部 Z-Score
pipeline_steps:
  - name: "Z-Score"
    method: "z_score"
    features: null

# 策略2: 分组处理（推荐）
pipeline_steps:
  - name: "价格Z-Score"
    method: "z_score"
    features: ["close", "open"]
  
  - name: "技术MinMax"
    method: "minmax"
    features: ["rsi", "kdj_k"]
  
  - name: "成交量Rank"
    method: "rank"
    features: ["volume"]

# 策略3: 加中性化
pipeline_steps:
  - name: "Z-Score"
    method: "z_score"
    features: null
  
  - name: "中性化"
    method: "ols_neutralize"
    enabled: true
```

---

### 3️⃣ 数据管理配置

**位置**: `data_management`

```yaml
data_management:
  kwargs:
    config:
      # 数据文件
      base_dir: "output"  # 🔧 数据目录
      data_file: "train_data_final_01.parquet"  # 🔧 文件名
      
      # 核心参数
      window_size: 40   # 🔧 时间窗口（20-60）
      batch_size: 512   # 🔧 批次大小（128-2048）
      num_workers: 4    # 🔧 工作进程（2-8）
      
      # 数据划分
      train_ratio: 0.6  # 🔧 训练集 60%
      val_ratio: 0.2    # 🔧 验证集 20%
      test_ratio: 0.2   # 🔧 测试集 20%
      
      # 特征选择
      auto_filter_features: true  # 🔧 自动过滤
      filter_config:
        na_threshold: 0.3         # 🔧 缺失值阈值
        variance_threshold: 0.01  # 🔧 方差阈值
```

**参数调优建议**:

| 参数 | 小数据集 | 中等数据集 | 大数据集 |
|------|----------|------------|----------|
| `batch_size` | 128-256 | 512-1024 | 1024-2048 |
| `window_size` | 20 | 40 | 60 |
| `num_workers` | 2 | 4 | 8 |

---

### 4️⃣ 模型训练配置

**位置**: `model_training`

```yaml
model_training:
  kwargs:
    # 模型架构
    hidden_dim: 128   # 🔧 GRU隐藏层（64-256）
    latent_dim: 16    # 🔧 潜在维度/因子数（8-32）
    num_layers: 2     # 🔧 GRU层数（1-3）
    dropout: 0.3      # 🔧 Dropout率（0.1-0.5）
    
    # VAE损失权重（关键！）
    alpha_recon: 0.1  # 🔧 重构损失（0.01-1.0）
    beta_kl: 0.001    # 🔧 KL散度（0.0001-0.01）
    gamma_pred: 1.0   # 🔧 预测损失（0.1-1.0）
    
    # 训练参数
    n_epochs: 100     # 🔧 训练轮数（30-200）
    lr: 0.001         # 🔧 学习率（0.0001-0.01）
    early_stop: 15    # 🔧 早停patience（5-20）
```

**VAE损失权重调优**:

| 目标 | alpha_recon | beta_kl | gamma_pred | 说明 |
|------|-------------|---------|------------|------|
| 重视重构 | 1.0 | 0.0001 | 0.1 | 更好的特征重构质量 |
| 重视预测 | 0.1 | 0.001 | 1.0 | 更准确的收益率预测 |
| 规则潜在空间 | 0.1 | 0.01 | 1.0 | 更符合正态分布的因子 |

---

### 5️⃣ 因子回测配置

**位置**: `factor_backtest`

```yaml
factor_backtest:
  kwargs:
    config:
      # 输出
      output_dir: "output/vae_backtest"  # 🔧 输出目录
      save_plots: true      # 🔧 保存图表
      generate_excel: true  # 🔧 生成Excel
      
      # 分组回测
      n_groups: 10  # 🔧 分组数（5/10/20）
      
      # IC分析
      ic_method: "spearman"  # 🔧 pearson/spearman
      
      # 多空组合
      long_short:
        top_quantile: 0.1     # 🔧 做多前10%
        bottom_quantile: 0.1  # 🔧 做空后10%
        commission: 0.0003    # 🔧 手续费万3
```

---

## 📚 更多资源

- **配置文件**: `vae_full_pipeline_example.yaml` - 完整配置模板（600+行）
- **YAML通用指南**: `YAML_USAGE_GUIDE.md` - YAML配置详解  
- **示例总览**: `examples/README.md` - 所有示例说明

**🎉 祝你使用愉快！**
