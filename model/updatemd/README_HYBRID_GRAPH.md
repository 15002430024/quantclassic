# HybridGraph 模型文档

## 概述

HybridGraph 是一个结合 **RNN+Self-Attention+GAT** 的混合神经网络架构，专门用于股票收益预测。它通过将时序特征提取（RNN）和截面信息交互（GAT）相结合，能够同时捕捉：

1. **时序规律**: 单只股票的历史走势和趋势
2. **截面关联**: 股票之间的行业联动或相关性效应

## 架构设计

### 核心分工

```
输入数据
   ↓
┌─────────────────────────────────────────┐
│  1. 时序提取器 (TemporalBlock)          │
│     - RNN (LSTM/GRU)                    │
│     - Self-Attention                    │
│     → 输出: 单只股票的时序特征向量        │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│  2. 特征融合 (可选)                      │
│     - 拼接基本面数据 (Fundamentals)      │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│  3. 截面交互器 (GraphBlock - GAT)       │
│     - 基于邻接矩阵聚合邻居信息            │
│     - 多头注意力机制                     │
│     → 输出: 融合截面信息的图特征          │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│  4. 融合预测器 (FusionBlock - MLP)      │
│     - 拼接: 时序特征 + 图特征 + 基本面   │
│     - MLP 输出最终预测                   │
└─────────────────────────────────────────┘
   ↓
预测结果
```

### 子模块详解

#### 1. TemporalBlock (时序提取器)

**作用**: 处理单只股票的时间序列数据，提取时序特征。

**组件**:
- **RNN层**: LSTM 或 GRU，捕捉长期依赖
- **Self-Attention层**: 强化关键时间点的权重（如趋势转折日）

**输入**: `[batch, seq_len, d_feat]`  
**输出**: `[batch, rnn_hidden]`

#### 2. GraphBlock (截面交互器)

**作用**: 通过图注意力网络（GAT）捕捉股票间的截面关联。

**机制**:
- 根据邻接矩阵定义股票间的连接关系
- 使用多头注意力计算邻居节点的重要性
- 加权聚合邻居特征，生成融合了市场环境的图特征

**两种邻接矩阵**:

| 类型 | 定义方式 | 适用场景 |
|------|---------|---------|
| **Standard GAT** | 基于行业分类<br>（同行业股票连通） | 学习行业轮动、板块联动 |
| **Correlation GAT** | 基于收益率相关性<br>（选取相关性最高的k个邻居） | 捕捉统计意义上的走势相似性 |

**输入**: `[num_stocks, in_dim]` + 邻接矩阵 `[num_stocks, num_stocks]`  
**输出**: `[num_stocks, gat_hidden]`

#### 3. FusionBlock (融合预测器)

**作用**: 将时序特征、图特征和基本面数据融合后进行预测。

**输入**: `[batch, time_feat + graph_feat + funda_feat]`  
**输出**: `[batch]` (预测的收益率)

## 使用方法

### 快速开始

```python
from model import HybridGraphModel

# 创建模型
model = HybridGraphModel(
    d_feat=20,           # 输入特征维度（量价数据）
    rnn_hidden=64,       # RNN隐藏层大小
    rnn_layers=2,        # RNN层数
    rnn_type='lstm',     # RNN类型: 'lstm' 或 'gru'
    use_attention=True,  # 使用Self-Attention
    use_graph=True,      # 使用图神经网络
    gat_hidden=32,       # GAT隐藏层维度
    gat_heads=4,         # GAT注意力头数
    n_epochs=100,
    lr=0.001
)

# 训练
model.fit(train_loader, valid_loader)

# 预测
predictions = model.predict(test_loader)
```

### 使用配置类

```python
from model.model_config import ModelConfigFactory

# 创建配置
config = ModelConfigFactory.get_template('hybrid_graph', 'large')

# 自定义配置
config.update(
    rnn_hidden=128,
    gat_heads=8,
    learning_rate=0.0005
)

# 使用配置创建模型
model = HybridGraphModel(**config.to_dict())
```

### 构建邻接矩阵

#### 方式 1: 基于行业分类

```python
from model.utils.adj_matrix_builder import AdjMatrixBuilder

builder = AdjMatrixBuilder()

# 假设有股票的行业代码
industry_codes = ['制造业', '金融', '制造业', '科技', ...]

# 构建邻接矩阵
adj_matrix = builder.build_industry_adj(industry_codes, self_loop=True)

# 保存
builder.save_adj_matrix(adj_matrix, 'adj_matrix_industry.pt')
```

#### 方式 2: 基于收益率相关性

```python
import pandas as pd

# 准备收益率数据 (DataFrame, shape=[time, stocks])
returns = pd.read_csv('returns.csv', index_col='date')

# 构建相关性邻接矩阵
adj_matrix = builder.build_correlation_adj(
    returns,
    top_k=10,        # 选取相关性最高的10个邻居
    method='pearson', # 相关性计算方法
    self_loop=True
)

# 保存
builder.save_adj_matrix(adj_matrix, 'adj_matrix_correlation.pt')
```

### 加入基本面数据

```python
# 1. 准备数据
X_train = ...  # [n_samples, seq_len, d_feat] 量价数据
F_train = ...  # [n_samples, funda_dim] 基本面数据
y_train = ...  # [n_samples] 标签

# 2. 创建 DataLoader (包含3个元素)
train_dataset = TensorDataset(X_train, F_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=256)

# 3. 创建模型 (指定基本面维度)
model = HybridGraphModel(
    d_feat=20,
    funda_dim=10,  # 基本面特征数量
    rnn_hidden=64,
    ...
)

# 4. 训练
model.fit(train_loader, valid_loader)
```

## 参数说明

### 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `d_feat` | int | 20 | 输入特征维度（量价数据） |
| `rnn_hidden` | int | 64 | RNN隐藏层大小 (64-256) |
| `rnn_layers` | int | 2 | RNN层数 (1-3) |
| `rnn_type` | str | 'lstm' | RNN类型: 'lstm' 或 'gru' |
| `use_attention` | bool | True | 是否使用Self-Attention |
| `use_graph` | bool | True | 是否使用图神经网络 |
| `gat_type` | str | 'standard' | GAT类型: 'standard' 或 'correlation' |
| `gat_hidden` | int | 32 | GAT隐藏层维度（必须能被gat_heads整除） |
| `gat_heads` | int | 4 | GAT注意力头数 (4-8) |
| `mlp_hidden_sizes` | list | [64] | MLP隐藏层尺寸列表 |
| `funda_dim` | int | None | 基本面数据维度（可选） |
| `dropout` | float | 0.3 | 全局Dropout概率 (0.1-0.5) |
| `adj_matrix_path` | str | None | 邻接矩阵文件路径 |

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_epochs` | int | 100 | 训练轮数 |
| `batch_size` | int | 256 | 批次大小 |
| `lr` | float | 0.001 | 学习率 |
| `early_stop` | int | 20 | 早停耐心值 |
| `optimizer` | str | 'adam' | 优化器: 'adam', 'adamw', 'sgd' |
| `loss_fn` | str | 'mse' | 损失函数: 'mse', 'mae', 'huber' |

## 数据格式要求

### 不使用图网络

```python
# DataLoader 返回: (X, y)
X: [batch_size, seq_len, d_feat]  # 时序量价数据
y: [batch_size]                    # 标签
```

### 使用图网络

```python
# 重要：batch_size 必须等于股票数量！
# 因为 GAT 需要处理整个截面的数据

# DataLoader 返回: (X, y)
X: [num_stocks, seq_len, d_feat]  # 当天所有股票的时序数据
y: [num_stocks]                    # 当天所有股票的标签

# 邻接矩阵
adj: [num_stocks, num_stocks]
```

### 包含基本面数据

```python
# DataLoader 返回: (X, F, y)
X: [batch_size, seq_len, d_feat]  # 时序量价数据
F: [batch_size, funda_dim]        # 基本面数据（不随时间变化）
y: [batch_size]                    # 标签
```

## 预定义模板

```python
from model.model_config import ModelConfigFactory

# 小型模型（快速实验）
config_small = ModelConfigFactory.get_template('hybrid_graph', 'small')

# 默认模型（均衡性能）
config_default = ModelConfigFactory.get_template('hybrid_graph', 'default')

# 大型模型（最佳性能）
config_large = ModelConfigFactory.get_template('hybrid_graph', 'large')
```

| 模板 | RNN隐藏层 | RNN层数 | GAT隐藏层 | GAT头数 | MLP隐藏层 | 训练轮数 |
|------|-----------|---------|-----------|---------|-----------|----------|
| `small` | 32 | 1 | 16 | 2 | [32] | 50 |
| `default` | 64 | 2 | 32 | 4 | [64] | 100 |
| `large` | 128 | 3 | 64 | 8 | [128, 64] | 200 |

## 完整示例

参见: `model/example_hybrid_graph.py`

运行示例:
```bash
cd /home/u2025210237/jupyterlab/quantclassic/model
python example_hybrid_graph.py
```

## 最佳实践

1. **邻接矩阵选择**:
   - 行业明确 → 使用 Standard GAT（基于行业）
   - 行业不明确或跨行业效应强 → 使用 Correlation GAT

2. **超参数调优**:
   - 从默认配置开始
   - 优先调整: `learning_rate`, `dropout`, `rnn_hidden`
   - GAT层的 `gat_heads` 和 `gat_hidden` 影响截面信息提取能力

3. **数据准备**:
   - 使用图网络时，确保 batch_size = num_stocks
   - 基本面数据应标准化（Z-Score或Min-Max）
   - 邻接矩阵可以提前计算并保存，避免重复构建

4. **训练技巧**:
   - 使用早停（early_stop）防止过拟合
   - 监控训练/验证损失曲线
   - GPU 加速: 设置 `device='cuda'`

## 引用

如果该模型架构参考了相关论文，请在此处添加引用。

## 许可

遵循项目主许可证。
