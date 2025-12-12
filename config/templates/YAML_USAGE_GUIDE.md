# QuantClassic YAML 配置文件使用指南

## 📖 目录

1. [快速开始](#快速开始)
2. [数据预处理配置](#数据预处理配置)
3. [数据管理配置](#数据管理配置)
4. [模型配置](#模型配置)
5. [回测配置](#回测配置)
6. [实战案例](#实战案例)
7. [常见问题](#常见问题)

---

## 🚀 快速开始

### 方式1: 命令行运行

```bash
# 使用基础模板
cd quantclassic
python -m config.cli config/templates/vae_basic.yaml

# 使用高级模板
python -m config.cli config/templates/vae_advanced.yaml

# 指定自定义配置
python -m config.cli /path/to/your_config.yaml
```

### 方式2: Python代码

```python
from quantclassic.config import ConfigLoader, TaskRunner

# 加载配置
config = ConfigLoader.load('quantclassic/config/templates/vae_advanced.yaml')

# 运行任务
runner = TaskRunner()
results = runner.run(config, experiment_name='my_vae_exp')

# 查看结果
print(results['metrics'])
print(results['model_path'])
```

---

## 🔧 数据预处理配置

### 基本结构

```yaml
preprocessor:
  class: DataPreprocessor
  module_path: quantclassic.data_processor.data_preprocessor
  kwargs:
    config:
      pipeline_steps: [...]  # 处理步骤列表
      column_mapping: {...}   # 列名映射
      groupby_columns: [...]  # 分组列
      id_columns: [...]       # ID列
      neutralize_config: {...} # 中性化配置
```

### 处理方法大全

#### 1. 标准化方法

##### Z-Score 标准化
```yaml
- name: "Z-Score标准化"
  method: "z_score"
  features:  # 可以指定特征列表，或用 null 表示所有特征
    - "close"
    - "volume"
    - "turnover_rate"
  enabled: true
  params: {}  # Z-Score不需要额外参数
```

**适用场景**: 特征符合正态分布，需要均值为0、标准差为1

##### MinMax 归一化
```yaml
- name: "MinMax归一化"
  method: "minmax"
  features:
    - "rsi"      # 技术指标
    - "kdj_k"
    - "macd"
  enabled: true
  params:
    feature_range: [0, 1]  # 归一化范围，也可以是 [-1, 1]
```

**适用场景**: 特征有明确的上下界，如技术指标

##### 秩归一化
```yaml
- name: "秩归一化"
  method: "rank"
  features:
    - "volume"
    - "amount"
  enabled: true
  params:
    output_range: [-1, 1]  # 输出范围
```

**适用场景**: 特征分布不规则，有极端值，只关心相对排名

#### 2. 极值处理

##### Winsorization (缩尾)
```yaml
- name: "去极值"
  method: "winsorize"
  features: null  # 对所有特征
  enabled: true
  params:
    limits: [0.025, 0.025]  # 上下各去2.5%极值
    # limits: [0.01, 0.01]  # 更激进: 上下各去1%
```

**适用场景**: 数据有异常值，但不想完全删除

##### Clip (裁剪)
```yaml
- name: "处理无穷值"
  method: "clip"
  features: null
  enabled: true
  params:
    lower: -1e10
    upper: 1e10
```

**适用场景**: 处理无穷值或超大数值

#### 3. 缺失值处理

```yaml
# 方法1: 中位数填充
- name: "填充缺失值_中位数"
  method: "fillna_median"
  features: null
  enabled: true
  params: {}

# 方法2: 均值填充
- name: "填充缺失值_均值"
  method: "fillna_mean"
  features: ["close", "volume"]
  enabled: true
  params: {}

# 方法3: 前向填充
- name: "填充缺失值_前向"
  method: "fillna_forward"
  features: ["close"]
  enabled: true
  params: {}

# 方法4: 填充为0
- name: "填充缺失值_零"
  method: "fillna_zero"
  features: ["special_feature"]
  enabled: true
  params: {}
```

#### 4. 中性化方法

##### OLS 市值行业中性化
```yaml
- name: "市值行业中性化"
  method: "ols_neutralize"
  features: null  # 对所有特征
  enabled: true  # 设为 false 可禁用
  params: {}

# 配合中性化配置
neutralize_config:
  industry_column: "industry_name"  # 行业列名
  market_cap_column: "total_mv"     # 市值列名
  min_samples: 10                   # 最小样本数
```

**适用场景**: 去除行业和市值因素的影响

##### SimStock 相似股票中性化
```yaml
- name: "SimStock中性化"
  method: "simstock_neutralize"
  features: null
  enabled: false  # 默认关闭
  params: {}

neutralize_config:
  target_column: "ret_1d"           # 收益率列
  similarity_threshold: 0.7         # 相似度阈值
  lookback_window: 252              # 回看窗口
  min_similar_stocks: 5             # 最小相似股票数
  correlation_method: "pearson"     # 相关性方法
```

**适用场景**: 去除相似股票的共同因素影响

### 实战案例: 混合预处理策略

```yaml
pipeline_steps:
  # 步骤1: 统一处理无穷值和缺失值
  - name: "处理无穷值"
    method: "clip"
    features: null
    enabled: true
    params:
      lower: -1e10
      upper: 1e10
  
  - name: "填充缺失值"
    method: "fillna_median"
    features: null
    enabled: true
    params: {}
  
  # 步骤2: 统一去极值
  - name: "去极值"
    method: "winsorize"
    features: null
    enabled: true
    params:
      limits: [0.025, 0.025]
  
  # 步骤3: 分组处理 - 价格类特征用 Z-Score
  - name: "价格特征标准化"
    method: "z_score"
    features:
      - "close"
      - "open"
      - "high"
      - "low"
      - "vwap"
    enabled: true
    params: {}
  
  # 步骤4: 技术指标用 MinMax
  - name: "技术指标归一化"
    method: "minmax"
    features:
      - "rsi"
      - "kdj_k"
      - "kdj_d"
      - "cci"
      - "macd_dif"
      - "macd_dea"
    enabled: true
    params:
      feature_range: [0, 1]
  
  # 步骤5: 成交量类用秩归一化
  - name: "成交量秩归一化"
    method: "rank"
    features:
      - "volume"
      - "amount"
      - "turnover_rate"
    enabled: true
    params:
      output_range: [-1, 1]
  
  # 步骤6: 市值行业中性化 (可选)
  - name: "市值行业中性化"
    method: "ols_neutralize"
    features: null
    enabled: false  # 需要时开启
    params: {}
```

**💡 要点说明**:
- `features: null` 表示应用到所有特征
- `enabled: false` 可以临时禁用某个步骤
- 处理步骤**按顺序执行**
- 不同特征组可以使用不同的标准化方法

---

## 📊 数据管理配置

### 批次大小设置

```yaml
dataset:
  kwargs:
    config:
      # 批次大小 (根据GPU内存调整)
      batch_size: 256   # 小GPU: 128-256
                        # 中GPU: 512-1024
                        # 大GPU: 2048+
      
      # 数据加载器工作进程数
      num_workers: 4    # CPU核心数的1/2 - 1倍
      
      # 是否打乱数据
      shuffle: true     # 训练集建议 true，测试集 false
```

**调优建议**:
- `batch_size` 越大，训练越稳定，但显存占用越高
- `num_workers` 太大可能导致内存不足，太小影响加载速度
- 多卡训练时，`batch_size` 是每卡的大小

### 数据划分策略

```yaml
dataset:
  kwargs:
    config:
      # 划分策略
      split_strategy: "time_series"  # 时间序列划分
      
      # 比例设置
      train_ratio: 0.6   # 60% 训练
      val_ratio: 0.2     # 20% 验证
      test_ratio: 0.2    # 20% 测试
      
      # 时间窗口
      window_size: 40    # 40个交易日
```

**其他策略** (需要在代码中实现):
```yaml
split_strategy: "random"        # 随机划分
split_strategy: "stratified"    # 分层划分
split_strategy: "custom_date"   # 自定义日期划分
```

### 特征选择

#### 方式1: 自动过滤
```yaml
dataset:
  kwargs:
    config:
      auto_filter_features: true
      filter_config:
        na_threshold: 0.3          # 缺失值>30%的特征删除
        variance_threshold: 0.01   # 方差<0.01的特征删除
        correlation_threshold: 0.95 # 相关性>0.95的特征删除一个
```

#### 方式2: 手动指定
```yaml
dataset:
  kwargs:
    config:
      auto_filter_features: false
      feature_columns:
        # 价格类
        - "close"
        - "open"
        - "high"
        - "low"
        - "vwap"
        # 成交量类
        - "volume"
        - "amount"
        - "turnover_rate"
        # 技术指标
        - "rsi"
        - "macd"
        - "kdj_k"
        # ... 更多特征
```

---

## 🤖 模型配置

### VAE 模型参数

```yaml
model:
  class: VAE
  module_path: quantclassic.model.pytorch_models
  kwargs:
    # ===== 架构参数 =====
    d_feat: 20          # 输入特征维度 (自动推断)
    hidden_dim: 128     # GRU隐藏层维度
    latent_dim: 16      # 潜在空间维度 (提取的因子数)
    window_size: 40     # 时间窗口
    dropout: 0.3        # Dropout率
    num_layers: 2       # GRU层数
    
    # ===== VAE损失权重 (关键超参数!) =====
    alpha_recon: 0.1    # 重构损失权重
    beta_kl: 0.001      # KL散度权重
    gamma_pred: 1.0     # 预测损失权重
    
    # ===== 训练参数 =====
    n_epochs: 100       # 最大训练轮数
    lr: 0.001           # 学习率
    early_stop: 15      # 早停patience
    batch_size: 256     # 批大小
    
    # ===== 设备 =====
    device: "cuda"      # cuda 或 cpu
    seed: 42            # 随机种子
```

### 超参数调优建议

#### 1. latent_dim (潜在维度)
```yaml
# 小模型 (更快，可能欠拟合)
latent_dim: 8

# 中等模型 (平衡)
latent_dim: 16

# 大模型 (更强表达能力，可能过拟合)
latent_dim: 32
```

#### 2. 损失权重调优

```yaml
# 场景1: 重视重构质量
alpha_recon: 1.0
beta_kl: 0.0001
gamma_pred: 0.1

# 场景2: 重视预测能力
alpha_recon: 0.1
beta_kl: 0.001
gamma_pred: 1.0

# 场景3: 更规则的潜在空间
alpha_recon: 0.1
beta_kl: 0.01   # 增大 beta_kl
gamma_pred: 1.0
```

#### 3. 学习率调度器

```yaml
optimizer: "AdamW"
optimizer_params:
  weight_decay: 1e-4
  betas: [0.9, 0.999]

# 方案1: ReduceLROnPlateau (验证集不下降时降低)
scheduler: "ReduceLROnPlateau"
scheduler_params:
  mode: "min"
  factor: 0.5       # 降低到原来的50%
  patience: 5       # 等待5个epoch
  min_lr: 1e-6

# 方案2: StepLR (固定步数降低)
scheduler: "StepLR"
scheduler_params:
  step_size: 20     # 每20个epoch
  gamma: 0.5        # 降低到原来的50%

# 方案3: CosineAnnealingLR (余弦退火)
scheduler: "CosineAnnealingLR"
scheduler_params:
  T_max: 50         # 周期长度
  eta_min: 1e-6     # 最小学习率
```

---

## 📈 回测配置

```yaml
backtest:
  class: FactorBacktestSystem
  module_path: quantclassic.Factorsystem.backtest_system
  kwargs:
    config:
      # 输出设置
      output_dir: "output/vae_backtest"
      save_plots: true
      generate_excel: true
      
      # 分组回测
      n_groups: 10      # 10分位
      
      # IC分析
      ic_method: "spearman"  # 或 "pearson"
      
      # 多空组合
      long_short:
        top_quantile: 0.1     # 做多前10%
        bottom_quantile: 0.1   # 做空后10%
        commission: 0.0003     # 万3手续费
```

---

## 💼 实战案例

### 案例1: 快速原型 (最小配置)

```yaml
experiment_name: quick_test

task:
  dataset:
    class: DataManager
    module_path: quantclassic.data_manager.manager
    kwargs:
      config:
        base_dir: "output"
        data_file: "data.parquet"
        window_size: 20
        batch_size: 128
        train_ratio: 0.7
        val_ratio: 0.15
        test_ratio: 0.15
  
  model:
    class: VAE
    module_path: quantclassic.model.pytorch_models
    kwargs:
      d_feat: 20
      latent_dim: 8
      n_epochs: 30
      device: "cuda"
```

### 案例2: 生产级配置 (完整流程)

使用 `vae_advanced.yaml` 模板，包含:
- ✅ 完整的数据预处理流水线
- ✅ 自动特征筛选
- ✅ 超参数优化
- ✅ 学习率调度
- ✅ 完整的回测分析

### 案例3: 超参数网格搜索

创建多个配置文件:

**vae_latent8.yaml**:
```yaml
experiment_name: vae_latent8
task:
  model:
    kwargs:
      latent_dim: 8
```

**vae_latent16.yaml**:
```yaml
experiment_name: vae_latent16
task:
  model:
    kwargs:
      latent_dim: 16
```

**vae_latent32.yaml**:
```yaml
experiment_name: vae_latent32
task:
  model:
    kwargs:
      latent_dim: 32
```

批量运行:
```bash
for config in vae_latent*.yaml; do
    python -m quantclassic.config.cli $config
done
```

### 案例4: 不同数据处理策略对比

**策略1: 全部 Z-Score**
```yaml
pipeline_steps:
  - name: "Z-Score"
    method: "z_score"
    features: null
```

**策略2: 分组处理**
```yaml
pipeline_steps:
  - name: "价格Z-Score"
    method: "z_score"
    features: ["close", "open", "high", "low"]
  
  - name: "技术指标MinMax"
    method: "minmax"
    features: ["rsi", "kdj_k", "macd"]
  
  - name: "成交量Rank"
    method: "rank"
    features: ["volume", "amount"]
```

**策略3: 加中性化**
```yaml
pipeline_steps:
  - name: "Z-Score"
    method: "z_score"
    features: null
  
  - name: "市值行业中性化"
    method: "ols_neutralize"
    features: null
    enabled: true
```

---

## ❓ 常见问题

### Q1: 批次大小如何设置?

**A**: 根据GPU显存:
- 4GB: batch_size=128
- 8GB: batch_size=256-512
- 16GB+: batch_size=1024+

出现 OOM (Out of Memory) 错误时，减小 batch_size。

### Q2: 不同特征需要不同的标准化方法怎么办?

**A**: 在 `pipeline_steps` 中创建多个步骤，每个步骤指定不同的 `features` 列表:

```yaml
pipeline_steps:
  - name: "价格Z-Score"
    method: "z_score"
    features: ["close", "open"]
  
  - name: "技术指标MinMax"
    method: "minmax"
    features: ["rsi", "kdj"]
```

### Q3: 如何临时禁用某个预处理步骤?

**A**: 设置 `enabled: false`:

```yaml
- name: "市值行业中性化"
  method: "ols_neutralize"
  enabled: false  # 禁用
```

### Q4: window_size 如何选择?

**A**: 
- 日频数据: 20-60个交易日 (1-3个月)
- 分钟数据: 更长的窗口 (如240分钟 = 1天)
- 权衡: 窗口越长，信息越多，但数据越少

### Q5: 如何查看实验结果?

**A**: 
```python
from quantclassic.config import ConfigLoader, TaskRunner

config = ConfigLoader.load('config.yaml')
results = TaskRunner().run(config)

# 查看指标
print(results['metrics'])

# 模型路径
print(results['model_path'])

# 因子数据
print(results['factors'].head())
```

### Q6: 配置文件太长怎么办?

**A**: 使用 YAML 的锚点和别名功能:

```yaml
# 定义锚点
common_params: &common
  batch_size: 256
  device: "cuda"

task:
  dataset:
    kwargs:
      config:
        <<: *common  # 引用锚点
        window_size: 40
  
  model:
    kwargs:
      <<: *common  # 复用
      latent_dim: 16
```

### Q7: 如何复现实验?

**A**: 
1. 保存配置文件到版本控制 (Git)
2. 设置固定的随机种子:
```yaml
model:
  kwargs:
    seed: 42
```
3. 使用 workflow 自动记录所有参数和结果

---

## 📚 更多资源

- **模板文件**:
  - `vae_basic.yaml` - 基础模板
  - `vae_advanced.yaml` - 高级完整模板
  
- **文档**:
  - `quantclassic/data_processor/README.md` - 数据预处理文档
  - `quantclassic/model/README.md` - 模型文档
  - `quantclassic/Factorsystem/README.md` - 回测系统文档

- **示例**:
  - `vae.ipynb` - VAE因子挖掘完整示例
  - `quantclassic/config/examples/` - 更多配置示例

---

**🎉 祝你使用愉快!**
