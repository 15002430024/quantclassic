# HybridGraph 模型实现总结

## 实现内容

已完成 **RNN+Attention+GAT** 混合图神经网络模型的完整实现，包括：

### 1. 核心文件

| 文件 | 说明 |
|------|------|
| `model_config.py` | 添加 `HybridGraphConfig` 配置类 |
| `hybrid_graph_models.py` | HybridGraph 模型完整实现 |
| `utils/adj_matrix_builder.py` | 邻接矩阵构建工具 |
| `example_hybrid_graph.py` | 使用示例（5个完整场景）|
| `README_HYBRID_GRAPH.md` | 详细文档 |
| `INSTALL.md` | 安装与快速开始指南 |

### 2. 模型架构

#### 子模块设计（面向对象）

```python
# 1. 时序提取器
class TemporalBlock(nn.Module):
    """RNN + Self-Attention"""
    - LSTM/GRU 捕捉长期依赖
    - Self-Attention 强化关键时间点
    
# 2. 截面交互器
class GraphBlock(nn.Module):
    """GAT - Graph Attention Network"""
    - 多头注意力机制
    - 基于邻接矩阵聚合邻居信息
    
# 3. 融合预测器
class FusionBlock(nn.Module):
    """MLP"""
    - 融合时序特征 + 图特征 + 基本面
    - 输出最终预测
    
# 4. 主模型
class HybridNet(nn.Module):
    """组合以上子模块"""
    - 灵活配置各模块参数
    - 支持可选的图网络和基本面数据
```

#### 数据流

```
输入: [batch, seq_len, d_feat]
  ↓
TemporalBlock (RNN + Attention)
  ↓
[batch, rnn_hidden] 时序特征
  ↓ (可选: 拼接基本面)
GraphBlock (GAT)
  ↓
[batch, gat_hidden] 图特征
  ↓
FusionBlock (MLP)
  ↓
[batch] 预测结果
```

### 3. 配置系统

#### HybridGraphConfig 参数

**时序模块:**
- `rnn_type`: 'lstm' / 'gru'
- `rnn_hidden`: RNN隐藏层大小
- `rnn_layers`: RNN层数
- `use_attention`: 是否使用Self-Attention

**截面模块:**
- `use_graph`: 是否使用图神经网络
- `gat_type`: 'standard' (行业) / 'correlation' (相关性)
- `gat_hidden`: GAT隐藏层维度
- `gat_heads`: 注意力头数
- `top_k_neighbors`: 相关性GAT的邻居数

**融合模块:**
- `mlp_hidden_sizes`: MLP隐藏层列表
- `funda_dim`: 基本面数据维度（可选）

**训练参数:**
- `n_epochs`, `batch_size`, `learning_rate`
- `dropout`, `early_stop`, `optimizer`, `loss_fn`

#### 预定义模板

| 模板 | RNN隐藏层 | GAT隐藏层 | 头数 | 训练轮数 |
|------|-----------|-----------|------|---------|
| small | 32 | 16 | 2 | 50 |
| default | 64 | 32 | 4 | 100 |
| large | 128 | 64 | 8 | 200 |

### 4. 邻接矩阵工具

`AdjMatrixBuilder` 类提供：

**基于行业分类:**
```python
adj = builder.build_industry_adj(industry_codes)
```

**基于收益率相关性:**
```python
adj = builder.build_correlation_adj(returns, top_k=10)
```

**加权邻接矩阵:**
```python
adj = builder.build_weighted_adj(returns, top_k=10)
```

**保存/加载:**
```python
builder.save_adj_matrix(adj, 'adj.pt')
adj = AdjMatrixBuilder.load_adj_matrix('adj.pt')
```

### 5. 使用示例

#### 示例 1: 不使用图网络（快速测试）

```python
model = HybridGraphModel(
    d_feat=20,
    rnn_hidden=64,
    use_graph=False,
    n_epochs=10
)
model.fit(train_loader)
```

#### 示例 2: 使用行业邻接矩阵

```python
# 构建邻接矩阵
builder = AdjMatrixBuilder()
adj = builder.build_industry_adj(industry_codes)
builder.save_adj_matrix(adj, 'industry_adj.pt')

# 创建模型
model = HybridGraphModel(
    d_feat=20,
    use_graph=True,
    gat_type='standard',
    adj_matrix_path='industry_adj.pt'
)
```

#### 示例 3: 使用相关性邻接矩阵

```python
# 构建邻接矩阵
adj = builder.build_correlation_adj(returns, top_k=10)
builder.save_adj_matrix(adj, 'corr_adj.pt')

# 创建模型
model = HybridGraphModel(
    d_feat=20,
    use_graph=True,
    gat_type='correlation',
    top_k_neighbors=10,
    adj_matrix_path='corr_adj.pt'
)
```

#### 示例 4: 加入基本面数据

```python
# 数据准备
X = ...  # [batch, seq_len, 20] 量价
F = ...  # [batch, 10] 基本面
y = ...  # [batch] 标签
dataset = TensorDataset(X, F, y)

# 创建模型
model = HybridGraphModel(
    d_feat=20,
    funda_dim=10,
    use_graph=True
)
```

## 技术亮点

### 1. 面向对象设计

- **模块化**: 三个独立的子模块（Temporal, Graph, Fusion）
- **可组合**: 通过配置灵活启用/禁用各模块
- **易测试**: 每个子模块可独立测试
- **易扩展**: 可轻松替换子模块实现

### 2. 配置系统

- **类型安全**: 使用 `@dataclass` 定义配置
- **参数验证**: 自动验证配置合法性
- **模板系统**: 提供预定义的小/中/大模板
- **序列化**: 支持 YAML 保存/加载

### 3. 灵活性

- **可选图网络**: `use_graph=False` 退化为纯时序模型
- **可选基本面**: 支持拼接基本面数据
- **两种GAT**: 支持行业和相关性两种邻接矩阵
- **RNN选择**: 支持 LSTM/GRU

### 4. 工程实践

- **早停机制**: 防止过拟合
- **梯度裁剪**: 稳定训练
- **设备管理**: 自动GPU/CPU选择
- **日志系统**: 详细的训练日志
- **保存/加载**: 完整的模型持久化

## 与研报对齐

实现与您提供的研报需求完全一致：

1. ✅ **RNN (LSTM/GRU)**: 处理时序数据
2. ✅ **Self-Attention**: 强化关键时间点
3. ✅ **GAT**: 截面信息交互
4. ✅ **行业邻接矩阵**: Standard GAT
5. ✅ **相关性邻接矩阵**: Correlation GAT
6. ✅ **基本面拼接**: 在GAT前拼接
7. ✅ **MLP融合**: 最终预测层
8. ✅ **特征拼接**: 时序 + 图 + 基本面

## 测试状态

### 已测试

✅ 配置系统 (`HybridGraphConfig`)
✅ 配置工厂 (`ModelConfigFactory`)
✅ 配置模板 (small/default/large)
✅ 配置验证
✅ 配置序列化 (YAML)

### 需要 PyTorch 测试

⏳ 子模块 (TemporalBlock, GraphBlock, FusionBlock)
⏳ 主模型 (HybridNet)
⏳ PyTorch封装 (HybridGraphModel)
⏳ 训练流程
⏳ 邻接矩阵工具

**测试命令:**
```bash
# 安装 PyTorch
pip install torch

# 运行测试
cd /home/u2025210237/jupyterlab/quantclassic/model
python hybrid_graph_models.py
python example_hybrid_graph.py
```

## 文件清单

```
model/
├── model_config.py               # ✅ 已更新（添加 HybridGraphConfig）
├── hybrid_graph_models.py        # ✅ 新建（完整实现）
├── example_hybrid_graph.py       # ✅ 新建（5个示例）
├── README_HYBRID_GRAPH.md        # ✅ 新建（详细文档）
├── INSTALL.md                    # ✅ 新建（安装指南）
├── IMPLEMENTATION_SUMMARY.md     # ✅ 本文件
├── __init__.py                   # ✅ 已更新（导出 HybridGraphModel）
└── utils/
    ├── __init__.py               # ✅ 新建
    └── adj_matrix_builder.py    # ✅ 新建（邻接矩阵工具）
```

## 下一步建议

1. **安装依赖**: 
   ```bash
   pip install torch numpy pandas tqdm
   ```

2. **运行测试**:
   ```bash
   python hybrid_graph_models.py
   ```

3. **准备数据**:
   - 量价数据: `[样本数, 时间步, 特征数]`
   - 邻接矩阵: 使用 `AdjMatrixBuilder`
   - (可选) 基本面数据

4. **训练模型**:
   ```python
   from model import HybridGraphModel
   
   model = HybridGraphModel(...)
   model.fit(train_loader, valid_loader)
   ```

5. **调优**:
   - 尝试不同的模板 (small/default/large)
   - 调整学习率、dropout
   - 实验不同的邻接矩阵类型

## 总结

✅ 已完成 RNN+Attention+GAT 混合模型的完整实现  
✅ 面向对象设计，模块清晰，易于维护  
✅ 配置系统完善，支持模板和验证  
✅ 提供邻接矩阵构建工具  
✅ 包含详细文档和多个示例  
✅ 与研报需求完全对齐  

现在您可以开始使用这个强大的混合模型进行股票预测了！
