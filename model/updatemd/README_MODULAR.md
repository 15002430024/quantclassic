# 模块化配置系统使用指南

## 📋 目录

- [概述](#概述)
- [快速开始](#快速开始)
- [核心概念](#核心概念)
- [使用方式](#使用方式)
- [模块详解](#模块详解)
- [实战案例](#实战案例)
- [迁移指南](#迁移指南)
- [常见问题](#常见问题)

---

## 📖 概述

### 为什么需要模块化配置?

**旧方式 (HybridGraphConfig)**:
```python
config = HybridGraphConfig(
    d_feat=20,
    rnn_hidden=64,
    rnn_layers=2,
    rnn_type='lstm',
    use_attention=True,
    gat_hidden=32,
    gat_heads=4,
    gat_type='standard',
    mlp_hidden_sizes=[64],
    dropout=0.3
)
```

❌ **问题**:
- 所有参数混在一起,难以理解和维护
- 扩展性差,添加新模块需要修改整个类
- 无法灵活组合不同的模块
- 参数命名容易混淆

**新方式 (CompositeModelConfig)**:
```python
config = ModelConfigBuilder() \
    .add_temporal(rnn_type='lstm', hidden_size=64, num_layers=2, use_attention=True) \
    .add_graph(gat_type='standard', hidden_dim=32, heads=4) \
    .add_fusion(hidden_sizes=[64]) \
    .build(d_feat=20)
```

✅ **优势**:
- **模块化**: 每个功能模块独立配置
- **可读性**: 清晰的模块划分和命名
- **灵活性**: 自由组合不同的模块
- **扩展性**: 轻松添加新的模块类型或变体
- **复用性**: 模块配置可以复用

---

## 🚀 快速开始

### 安装

```python
# 导入模块化配置
from model.modular_config import (
    ModelConfigBuilder,
    ConfigTemplates,
    CompositeModelConfig
)
```

### 三种使用方式

#### 1. 使用构建器 (推荐)

```python
config = ModelConfigBuilder() \
    .set_input(d_feat=20) \
    .add_temporal(rnn_type='lstm', hidden_size=64) \
    .add_graph(gat_type='correlation', hidden_dim=32) \
    .add_fusion(hidden_sizes=[64]) \
    .set_training(n_epochs=100, batch_size=256) \
    .build()
```

#### 2. 使用预定义模板

```python
# 纯时序模型
config = ConfigTemplates.pure_temporal(d_feat=20, model_size='default')

# 时序+图混合模型
config = ConfigTemplates.temporal_with_graph(
    d_feat=20,
    gat_type='correlation',
    model_size='large'
)
```

#### 3. 手动组合模块 (最灵活)

```python
from model.modular_config import (
    TemporalModuleConfig,
    GraphModuleConfig,
    FusionModuleConfig,
    CompositeModelConfig
)

temporal = TemporalModuleConfig(
    rnn_type='lstm',
    hidden_size=64,
    num_layers=2
)

graph = GraphModuleConfig(
    gat_type='correlation',
    hidden_dim=32,
    heads=4
)

fusion = FusionModuleConfig(
    hidden_sizes=[64]
)

config = CompositeModelConfig(
    temporal=temporal,
    graph=graph,
    fusion=fusion,
    d_feat=20
)
```

---

## 🧩 核心概念

### 三大核心模块

```
┌─────────────────────────────────────────────────┐
│          CompositeModelConfig (组合模型)          │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────────────┐  ┌──────────────────┐     │
│  │ TemporalModule  │  │  GraphModule     │     │
│  │  (时序特征)      │  │  (截面特征)       │     │
│  │                 │  │                  │     │
│  │ - RNN/LSTM/GRU │  │ - GAT           │     │
│  │ - Attention    │  │ - 邻接矩阵       │     │
│  └────────┬────────┘  └────────┬─────────┘     │
│           │                    │               │
│           └────────┬───────────┘               │
│                    ▼                           │
│         ┌──────────────────────┐               │
│         │   FusionModule       │               │
│         │   (特征融合)          │               │
│         │                      │               │
│         │ - MLP                │               │
│         │ - BatchNorm (可选)   │               │
│         │ - Residual (可选)    │               │
│         └──────────────────────┘               │
│                    │                           │
│                    ▼                           │
│            [ 预测输出 ]                         │
└─────────────────────────────────────────────────┘
```

### 模块类型

| 模块 | 作用 | 必需 |
|------|------|------|
| **TemporalModule** | 从时间序列提取时序特征 | 可选 |
| **GraphModule** | 通过图结构捕捉截面关联 | 可选 |
| **FusionModule** | 融合多模块特征并预测 | **必需** |

> ⚠️ 至少需要启用 Temporal 或 Graph 之一

---

## 📚 使用方式

### 方式 1: 构建器模式 (Builder Pattern)

**最简配置**:
```python
config = ModelConfigBuilder() \
    .add_temporal(rnn_type='lstm', hidden_size=64) \
    .add_fusion(hidden_sizes=[64]) \
    .build(d_feat=20)
```

**完整配置**:
```python
config = ModelConfigBuilder() \
    .set_input(
        d_feat=20,          # 输入特征维度
        funda_dim=10        # 基本面数据维度 (可选)
    ) \
    .add_temporal(
        rnn_type='lstm',
        hidden_size=64,
        num_layers=2,
        bidirectional=False,
        use_attention=True,
        attention_type='multi_head',
        attention_heads=8,
        dropout=0.3
    ) \
    .add_graph(
        gat_type='correlation',
        hidden_dim=32,
        heads=4,
        concat=True,
        top_k_neighbors=10,
        adj_matrix_path='./adj_matrix.pt',
        dropout=0.3
    ) \
    .add_fusion(
        hidden_sizes=[64, 32],
        activation='relu',
        use_batch_norm=False,
        use_residual=False,
        dropout=0.3
    ) \
    .set_training(
        device='cuda',
        n_epochs=100,
        batch_size=256,
        learning_rate=0.001,
        optimizer='adam',
        early_stop=20
    ) \
    .build()
```

**API 说明**:
- `.set_input()`: 设置输入特征
- `.add_temporal()`: 添加时序模块
- `.add_graph()`: 添加图模块
- `.add_fusion()`: 添加融合模块
- `.set_training()`: 设置训练参数
- `.build()`: 构建最终配置

### 方式 2: 预定义模板

```python
from model.modular_config import ConfigTemplates

# 1. 纯时序模型 (不使用图)
config = ConfigTemplates.pure_temporal(
    d_feat=20,
    model_size='small'  # 'small', 'default', 'large'
)

# 2. 时序+图混合模型
config = ConfigTemplates.temporal_with_graph(
    d_feat=20,
    gat_type='standard',  # 'standard' 或 'correlation'
    adj_matrix_path='./adj_matrix.pt',
    model_size='default'
)

# 3. 多头注意力+相关性图+深层融合
config = ConfigTemplates.attention_graph_fusion(
    d_feat=20,
    attention_type='multi_head',
    gat_type='correlation'
)
```

**模型尺寸对比**:

| 尺寸 | RNN Hidden | RNN Layers | GAT Hidden | GAT Heads | MLP Layers |
|------|-----------|-----------|-----------|----------|-----------|
| small | 32 | 1 | 16 | 2 | [32] |
| default | 64 | 2 | 32 | 4 | [64] |
| large | 128 | 3 | 64 | 8 | [128, 64] |

### 方式 3: 手动组合模块

**适用场景**: 需要精细控制每个模块的配置

```python
from model.modular_config import (
    TemporalModuleConfig,
    GraphModuleConfig,
    FusionModuleConfig,
    CompositeModelConfig
)

# 1. 配置时序模块
temporal = TemporalModuleConfig(
    rnn_type='lstm',
    hidden_size=64,
    num_layers=2,
    bidirectional=False,
    use_attention=True,
    attention_type='self',
    dropout=0.3
)

# 2. 配置图模块
graph = GraphModuleConfig(
    gat_type='correlation',
    hidden_dim=32,
    heads=4,
    concat=True,
    top_k_neighbors=10,
    dropout=0.3
)

# 3. 配置融合模块
fusion = FusionModuleConfig(
    hidden_sizes=[64],
    activation='relu',
    dropout=0.3
)

# 4. 组合成完整配置
config = CompositeModelConfig(
    temporal=temporal,
    graph=graph,
    fusion=fusion,
    d_feat=20,
    n_epochs=100,
    batch_size=256
)

# 5. 验证配置
config.validate()

# 6. 查看摘要
print(config.summary())
```

---

## 🔧 模块详解

### TemporalModule (时序模块)

**作用**: 从时间序列数据中提取时序特征

**核心参数**:

```python
TemporalModuleConfig(
    # RNN 配置
    rnn_type='lstm',        # 'lstm', 'gru', 'rnn'
    hidden_size=64,         # 隐藏层大小
    num_layers=2,           # RNN 层数
    bidirectional=False,    # 是否双向
    
    # 注意力配置
    use_attention=True,     # 是否使用注意力
    attention_type='self',  # 'self', 'multi_head', 'additive', 'dot_product'
    attention_heads=4,      # 多头注意力头数
    
    # 正则化
    dropout=0.3
)
```

**RNN 类型对比**:

| 类型 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| LSTM | 记忆能力强,适合长序列 | 参数多,训练慢 | 复杂时序模式 |
| GRU | 参数少,训练快 | 记忆能力略弱 | 快速实验 |
| RNN | 最简单 | 易梯度消失 | 简单序列 |

**注意力类型对比**:

| 类型 | 特点 | 计算复杂度 |
|------|------|-----------|
| self | 简单加权,可解释性强 | 低 |
| multi_head | 多视角特征,效果好 | 中 |
| additive | Bahdanau风格 | 中 |
| dot_product | Luong风格 | 低 |

**输出维度**:
```python
output_dim = hidden_size * (2 if bidirectional else 1)
```

### GraphModule (图模块)

**作用**: 通过图结构捕捉股票间的截面关联

**核心参数**:

```python
GraphModuleConfig(
    # GAT 配置
    gat_type='standard',    # 'standard', 'correlation', 'dynamic'
    hidden_dim=32,          # 隐藏层维度
    heads=4,                # 注意力头数
    concat=True,            # 是否拼接多头输出
    
    # 图结构配置
    top_k_neighbors=10,     # K近邻数量 (correlation模式)
    edge_threshold=0.0,     # 边权重阈值
    use_edge_features=False, # 是否使用边特征
    
    # 正则化
    dropout=0.3
)
```

**GAT 类型对比**:

| 类型 | 图结构 | 优点 | 缺点 |
|------|--------|------|------|
| standard | 静态(行业关系) | 简单,可解释性强 | 无法捕获动态关联 |
| correlation | 动态(相关性) | 自适应市场变化 | 需要计算相关性矩阵 |
| dynamic | 完全学习 | 最灵活 | 可能过拟合 |

**输出维度**:
```python
output_dim = hidden_dim  # (无论 concat 为何值)
```

### FusionModule (融合模块)

**作用**: 融合多模块特征并生成预测

**核心参数**:

```python
FusionModuleConfig(
    # MLP 配置
    hidden_sizes=[64],      # 隐藏层尺寸列表
    activation='relu',      # 'relu', 'gelu', 'tanh', 'leaky_relu'
    
    # 增强配置
    use_batch_norm=False,   # 是否使用 BatchNorm
    use_residual=False,     # 是否使用残差连接
    
    # 正则化
    dropout=0.3,
    
    # 输出
    output_dim=1            # 输出维度
)
```

**激活函数对比**:

| 函数 | 特点 | 适用场景 |
|------|------|---------|
| relu | 最常用,计算快 | 大多数情况 |
| gelu | 平滑,效果好 | Transformer风格 |
| tanh | 有界,收敛快 | 小规模网络 |
| leaky_relu | 解决死神经元 | ReLU失效时 |

---

## 💡 实战案例

### 案例 1: 纯时序 LSTM 模型

```python
config = ModelConfigBuilder() \
    .add_temporal(
        rnn_type='lstm',
        hidden_size=64,
        num_layers=2,
        use_attention=True
    ) \
    .add_fusion(hidden_sizes=[64]) \
    .build(d_feat=20)
```

**模型结构**:
```
输入 [batch, seq, 20]
  ↓
LSTM [64 hidden, 2 layers]
  ↓
Self-Attention
  ↓
MLP [64 -> 1]
  ↓
输出 [batch]
```

### 案例 2: 时序 + 行业图

```python
config = ModelConfigBuilder() \
    .add_temporal(
        rnn_type='gru',
        hidden_size=64,
        use_attention=True
    ) \
    .add_graph(
        gat_type='standard',
        hidden_dim=32,
        heads=4,
        adj_matrix_path='./industry_adj.pt'
    ) \
    .add_fusion(hidden_sizes=[64]) \
    .build(d_feat=20)
```

**模型结构**:
```
时序分支:                    图分支:
输入 [batch, seq, 20]      时序特征 [batch, 64]
  ↓                           ↓
GRU [64 hidden]             GAT [32 hidden, 4 heads]
  ↓                           ↓
Self-Attention              图特征 [batch, 32]
  ↓                           ↓
时序特征 [batch, 64]         
  └───────────┬──────────────┘
              ↓
         拼接 [batch, 96]
              ↓
         MLP [64 -> 1]
              ↓
         输出 [batch]
```

### 案例 3: 多头注意力 + 相关性图

```python
config = ModelConfigBuilder() \
    .add_temporal(
        rnn_type='gru',
        hidden_size=64,
        use_attention=True,
        attention_type='multi_head',
        attention_heads=8
    ) \
    .add_graph(
        gat_type='correlation',
        hidden_dim=64,
        heads=8,
        top_k_neighbors=15
    ) \
    .add_fusion(
        hidden_sizes=[128, 64],
        use_batch_norm=True
    ) \
    .build(d_feat=20)
```

**特点**:
- 多头注意力捕捉不同时间模式
- 相关性图自适应市场变化
- 深层MLP + BatchNorm 增强表达能力

### 案例 4: 双向 LSTM + 深层融合

```python
config = ModelConfigBuilder() \
    .add_temporal(
        rnn_type='lstm',
        hidden_size=64,
        num_layers=3,
        bidirectional=True,  # 双向
        use_attention=True
    ) \
    .add_fusion(
        hidden_sizes=[256, 128, 64],  # 深层MLP
        activation='gelu',
        use_batch_norm=True,
        use_residual=True  # 残差连接
    ) \
    .build(d_feat=20)
```

**特点**:
- 双向LSTM捕获前后文信息
- 深层MLP (3层) 增强非线性
- 残差连接防止梯度消失

---

## 🔄 迁移指南

### 从 HybridGraphConfig 迁移

**旧代码**:
```python
from model.model_config import HybridGraphConfig

config = HybridGraphConfig(
    d_feat=20,
    rnn_hidden=64,
    rnn_layers=2,
    rnn_type='lstm',
    use_attention=True,
    gat_hidden=32,
    gat_heads=4,
    gat_type='standard',
    mlp_hidden_sizes=[64],
    dropout=0.3,
    n_epochs=100,
    batch_size=256
)
```

**新代码**:
```python
from model.modular_config import ModelConfigBuilder

config = ModelConfigBuilder() \
    .add_temporal(
        rnn_type='lstm',
        hidden_size=64,      # rnn_hidden -> hidden_size
        num_layers=2,        # rnn_layers -> num_layers
        use_attention=True,
        dropout=0.3
    ) \
    .add_graph(
        gat_type='standard',
        hidden_dim=32,       # gat_hidden -> hidden_dim
        heads=4,             # gat_heads -> heads
        dropout=0.3
    ) \
    .add_fusion(
        hidden_sizes=[64],   # mlp_hidden_sizes -> hidden_sizes
        dropout=0.3
    ) \
    .set_training(
        n_epochs=100,
        batch_size=256
    ) \
    .build(d_feat=20)
```

**参数映射表**:

| 旧参数 | 新参数 | 所属模块 |
|--------|--------|---------|
| d_feat | d_feat | CompositeModelConfig |
| rnn_hidden | hidden_size | TemporalModuleConfig |
| rnn_layers | num_layers | TemporalModuleConfig |
| rnn_type | rnn_type | TemporalModuleConfig |
| use_attention | use_attention | TemporalModuleConfig |
| gat_hidden | hidden_dim | GraphModuleConfig |
| gat_heads | heads | GraphModuleConfig |
| gat_type | gat_type | GraphModuleConfig |
| mlp_hidden_sizes | hidden_sizes | FusionModuleConfig |
| dropout | dropout | 各模块 |

---

## ❓ 常见问题

### Q1: 可以只使用时序模块,不使用图模块吗?

**A**: 可以!只需在构建器中不调用 `.add_graph()` 即可:

```python
config = ModelConfigBuilder() \
    .add_temporal(rnn_type='lstm', hidden_size=64) \
    .add_fusion(hidden_sizes=[64]) \
    .build(d_feat=20)

# config.graph 为 None
```

### Q2: 如何添加基本面数据?

**A**: 使用 `.set_input(funda_dim=...)`

```python
config = ModelConfigBuilder() \
    .set_input(d_feat=20, funda_dim=10) \  # 10维基本面
    .add_temporal(...) \
    .add_fusion(...) \
    .build()
```

基本面数据会在适当位置自动拼接。

### Q3: 如何自定义新的模块类型?

**A**: 继承现有模块配置类:

```python
from model.modular_config import TemporalModuleConfig

class TransformerTemporalConfig(TemporalModuleConfig):
    """自定义: Transformer风格的时序模块"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_attention = True
        self.attention_type = 'multi_head'
        self.attention_heads = 8

# 使用
config = CompositeModelConfig(
    temporal=TransformerTemporalConfig(hidden_size=128),
    fusion=FusionModuleConfig(hidden_sizes=[64]),
    d_feat=20
)
```

### Q4: 如何保存和加载配置?

**A**: 使用内置的序列化方法:

```python
# 保存为 YAML
config.to_yaml('my_config.yaml')

# 保存为 JSON
config.to_json('my_config.json')

# 从文件加载
config = CompositeModelConfig.from_yaml('my_config.yaml')
```

### Q5: 如何查看配置摘要?

**A**: 调用 `.summary()` 方法:

```python
print(config.summary())
```

输出示例:
```
============================================================
组合模型配置摘要
============================================================

输入特征维度: 20

【时序模块】
  - RNN类型: lstm
  - 隐藏层: 64
  - 层数: 2
  ...

【图模块】
  - GAT类型: correlation
  - 隐藏维度: 32
  ...

【融合模块】
  - 融合策略: concat
  - 隐藏层: [64]
  ...
```

### Q6: 融合输入维度是如何计算的?

**A**: 使用 `.get_fusion_input_dim()` 方法:

```python
dim = config.get_fusion_input_dim()

# 计算规则:
# dim = temporal.output_dim + graph.output_dim + funda_dim
# 例如: 64 (temporal) + 32 (graph) + 0 (no funda) = 96
```

### Q7: 可以在训练过程中动态调整配置吗?

**A**: 可以使用 `.update()` 方法:

```python
# 调整学习率
config.update(learning_rate=0.0005)

# 调整早停轮数
config.update(early_stop=30)
```

但建议在训练前确定好配置。

---

## 📖 参考资料

- `modular_config.py`: 模块化配置源码
- `example_modular_usage.py`: 完整使用示例
- `hybrid_graph_models.py`: 模型实现
- `README_HYBRID_GRAPH.md`: 混合模型指南

---

## 🎯 最佳实践

1. **优先使用构建器模式**: 可读性强,易于维护
2. **从预定义模板开始**: 快速建立基线模型
3. **逐步调整参数**: 先用默认参数,再根据效果调优
4. **保存配置文件**: 便于复现实验
5. **查看配置摘要**: 训练前确认所有参数
6. **模块化思考**: 独立调优每个模块
7. **注释清晰**: 记录每个参数的选择原因

---

## 📝 更新日志

- **2025-01-27**: 初始版本发布
  - 实现三大核心模块
  - 支持构建器模式
  - 提供预定义模板

---

*Happy Modeling! 🚀*
