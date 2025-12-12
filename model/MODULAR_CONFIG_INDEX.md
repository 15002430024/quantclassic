# 模块化配置系统 - 快速索引

## 🎯 核心改进

**从整体配置到模块化配置,支持灵活组合和扩展**

### 旧方式 (兼容保留)
```python
# model_config.py - HybridGraphConfig
config = HybridGraphConfig(
    d_feat=20, rnn_hidden=64, gat_hidden=32, ...
)
```
❌ 所有参数混在一起,扩展性差

### 新方式 (推荐)
```python
# modular_config.py - CompositeModelConfig
config = ModelConfigBuilder() \
    .add_temporal(rnn_type='lstm', hidden_size=64) \
    .add_graph(gat_type='correlation', hidden_dim=32) \
    .add_fusion(hidden_sizes=[64]) \
    .build(d_feat=20)
```
✅ 模块独立,灵活组合,易于扩展

---

## 📂 文件索引

| 文件 | 说明 |
|------|------|
| **modular_config.py** | 模块化配置核心实现 |
| **example_modular_usage.py** | 10个完整使用示例 |
| **README_MODULAR.md** | 详细使用指南 (本文档) |
| **model_config.py** | 原配置类 (兼容模式) |

---

## 🚀 快速开始

### 1. 使用构建器 (最简单)
```python
from model.modular_config import ModelConfigBuilder

config = ModelConfigBuilder() \
    .add_temporal(rnn_type='lstm', hidden_size=64) \
    .add_graph(gat_type='correlation', hidden_dim=32) \
    .add_fusion(hidden_sizes=[64]) \
    .build(d_feat=20)
```

### 2. 使用模板 (最快)
```python
from model.modular_config import ConfigTemplates

config = ConfigTemplates.temporal_with_graph(
    d_feat=20, 
    gat_type='correlation',
    model_size='large'
)
```

### 3. 手动组合 (最灵活)
```python
from model.modular_config import (
    TemporalModuleConfig,
    GraphModuleConfig,
    FusionModuleConfig,
    CompositeModelConfig
)

config = CompositeModelConfig(
    temporal=TemporalModuleConfig(...),
    graph=GraphModuleConfig(...),
    fusion=FusionModuleConfig(...),
    d_feat=20
)
```

---

## 🧩 三大核心模块

### TemporalModule (时序特征)
```python
.add_temporal(
    rnn_type='lstm',      # LSTM/GRU/RNN
    hidden_size=64,
    use_attention=True    # Self-Attention
)
```

### GraphModule (截面特征)
```python
.add_graph(
    gat_type='correlation',  # 相关性图
    hidden_dim=32,
    heads=4
)
```

### FusionModule (特征融合)
```python
.add_fusion(
    hidden_sizes=[64],
    activation='relu'
)
```

---

## 📖 详细文档

👉 **[完整使用指南](README_MODULAR.md)**

包含:
- 核心概念详解
- 10+ 实战案例
- 参数详细说明
- 迁移指南
- 常见问题

---

## 💡 典型案例

### 纯时序模型
```python
config = ModelConfigBuilder() \
    .add_temporal(rnn_type='lstm', hidden_size=64) \
    .add_fusion(hidden_sizes=[64]) \
    .build(d_feat=20)
```

### 时序 + 行业图
```python
config = ModelConfigBuilder() \
    .add_temporal(rnn_type='gru', hidden_size=64) \
    .add_graph(gat_type='standard', hidden_dim=32) \
    .add_fusion(hidden_sizes=[64]) \
    .build(d_feat=20)
```

### 多头注意力 + 相关性图
```python
config = ModelConfigBuilder() \
    .add_temporal(
        rnn_type='gru',
        hidden_size=64,
        attention_type='multi_head',
        attention_heads=8
    ) \
    .add_graph(
        gat_type='correlation',
        hidden_dim=64,
        top_k_neighbors=15
    ) \
    .add_fusion(
        hidden_sizes=[128, 64],
        use_batch_norm=True
    ) \
    .build(d_feat=20)
```

---

## 🔄 从旧配置迁移

**参数映射**:
- `rnn_hidden` → `hidden_size` (TemporalModule)
- `rnn_layers` → `num_layers` (TemporalModule)
- `gat_hidden` → `hidden_dim` (GraphModule)
- `gat_heads` → `heads` (GraphModule)
- `mlp_hidden_sizes` → `hidden_sizes` (FusionModule)

详见 [迁移指南](README_MODULAR.md#迁移指南)

---

## ✅ 优势总结

| 特性 | 旧方式 | 新方式 |
|------|--------|--------|
| **可读性** | ❌ 参数混杂 | ✅ 模块清晰 |
| **扩展性** | ❌ 修改困难 | ✅ 插件式扩展 |
| **复用性** | ❌ 无法复用 | ✅ 模块可复用 |
| **组合性** | ❌ 固定结构 | ✅ 灵活组合 |
| **维护性** | ❌ 难以维护 | ✅ 职责分离 |

---

## 📝 运行示例

```bash
# 查看所有示例
python example_modular_usage.py

# 测试模块化配置
python modular_config.py
```

---

*更新时间: 2025-01-27*
