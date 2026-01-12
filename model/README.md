
# QuantClassic Model Module - 模型模块

标准化的量化模型接口和实现，参照 Qlib 设计。

> **🆕 版本: v2.0.0 (2026-01-11 重构)**
> - 统一训练引擎，`fit()` 代理到 `train/SimpleTrainer`
> - 统一 `predict()` 方法到基类，支持所有 batch 格式
> - 模块化配置系统，支持灵活组合时序/图/融合模块
> - 图构建统一到 `data_processor/graph_builder.py`

## 📦 模块结构

```
model/
├── base_model.py           # 模型基类 (BaseModel → Model → PyTorchModel)
├── pytorch_models.py       # PyTorch 模型实现 (LSTM/GRU/Transformer/VAE)
├── hybrid_graph_models.py  # 混合图模型 (HybridGraphModel + TemporalBlock/GraphBlock/FusionBlock)
├── model_factory.py        # 模型工厂和注册机制
├── model_config.py         # ⚠️ 旧配置（已废弃，请用 modular_config.py）
├── modular_config.py       # 🆕 模块化配置系统 (CompositeModelConfig)
├── loss.py                 # 损失函数 (UnifiedLoss, ICLoss, CorrelationRegularizer)
├── train/                  # 🆕 统一训练引擎
│   ├── base_trainer.py     #   训练基类 + TrainerConfig
│   ├── simple_trainer.py   #   简单训练器（单窗口）
│   ├── rolling_window_trainer.py  #  滚动窗口训练器
│   └── rolling_daily_trainer.py   #  日级滚动训练器
├── example/                # 使用示例
└── updatemd/               # 详细文档
```

## ✨ 核心特性

### 🎯 统一接口
- **标准化**: 所有模型继承自 `PyTorchModel` 基类
- **一致性**: 统一的 `fit()` 和 `predict()` 接口
- **🆕 通用 predict**: 基类实现统一预测逻辑，支持 `(x,y)` / `(x,y,adj,...)` / `dict` 等多种 batch 格式
- **🆕 Trainer 对齐**: `SimpleTrainer.predict` 优先委托模型的 `predict()`，确保与模型的 batch 解析保持一致；纯 `nn.Module` 自动回退内置实现

### 🏭 工厂模式
- **动态创建**: 通过配置字典创建模型
- **注册机制**: 使用 `@register_model` 装饰器注册模型
- **🆕 模块化配置**: `CompositeModelConfig` 支持时序/图/融合模块自由组合

### 🚀 自动化功能
- **GPU 管理**: 自动检测和使用 GPU
- **早停机制**: 内置早停避免过拟合
- **模型保存**: 自动保存最佳模型
- **日志记录**: 完整的训练日志
- **🆕 学习率调度**: 支持 ReduceLROnPlateau / Cosine / Step

### 🔧 训练引擎 (2026-01 重构)
- **训练代理**: `Model.fit()` 内部使用 `SimpleTrainer`，保持接口兼容
- **滚动训练**: `RollingWindowTrainer` 支持权重继承、优化器状态保存
- **相关性正则化**: 支持 `lambda_corr` 抑制特征冗余

## 🚀 快速开始

### 1. 基础使用

```python
from quantclassic.model import LSTMModel
from quantclassic.data_set import DataManager, DataConfig

# 准备数据
config = DataConfig(base_dir='rq_data_parquet')
manager = DataManager(config)
loaders = manager.run_full_pipeline()

# 创建模型
model = LSTMModel(
    d_feat=20,
    hidden_size=64,
    num_layers=2,
    n_epochs=100,
    lr=0.001
)

# 训练（内部使用 SimpleTrainer）
model.fit(loaders.train, loaders.val, save_path='output/model.pth')

# 预测（支持标准/图/日级 loader）
predictions = model.predict(loaders.test)
```

### 2. 模块化配置（推荐）

```python
from quantclassic.model.modular_config import ModelConfigBuilder, ConfigTemplates
from quantclassic.model import create_model_from_composite_config

# 方式1: 使用预定义模板
config = ConfigTemplates.pure_temporal(d_feat=20, model_size='default')

# 方式2: 使用 Builder 灵活组合
config = ModelConfigBuilder() \
    .set_input(d_feat=20) \
    .add_temporal(rnn_type='lstm', hidden_size=128, use_attention=True) \
    .add_graph(gat_type='correlation', hidden_dim=64, heads=4) \
    .add_fusion(hidden_sizes=[128, 64]) \
    .build()

# 创建模型
model = create_model_from_composite_config(config)
model.fit(train_loader, valid_loader)
```

### 3. 使用训练引擎

```python
from quantclassic.model.train import SimpleTrainer, TrainerConfig

# 创建训练配置
config = TrainerConfig(
    n_epochs=100,
    lr=0.001,
    early_stop=20,
    loss_fn='mse',
    lambda_corr=0.01,  # 相关性正则化
    use_scheduler=True,
    scheduler_type='plateau'
)

# 创建训练器（传入 nn.Module）
trainer = SimpleTrainer(model.model, config, device='cuda')
result = trainer.train(train_loader, valid_loader)
```

## 📚 类继承关系

```
BaseModel (抽象基类)
    ├── predict() - 抽象方法
    └── __call__() - 调用 predict()
    
Model (继承 BaseModel)
    ├── fit() - 抽象方法
    └── predict() - 抽象方法
    
PyTorchModel (继承 Model)
    ├── 🆕 通用 predict() - 支持所有 batch 格式
    │   ├── _parse_batch_data() - 统一解析 (x,y) / (x,y,adj,...) / dict
    │   ├── _forward_for_predict() - 前向传播钩子（可覆写）
    │   └── _post_process() - 后处理钩子（可覆写）
    ├── fit() - 代理到 SimpleTrainer
    ├── 自动 GPU 管理
    ├── 学习率调度器
    └── 相关性正则化支持
    
LSTMModel / GRUModel / TransformerModel
    └── 继承 PyTorchModel，使用基类 predict()
    
VAEModel
    ├── 继承 PyTorchModel
    ├── 覆写 predict() 支持 return_latent；在 return_latent=False 时复用基类通用 predict
    ├── 覆写 _forward_for_predict() 返回 y_pred
    └── extract_latent() - 提取潜在特征

HybridGraphModel
    ├── 继承 PyTorchModel
    ├── 覆写 _parse_batch_data() 解析 funda/stock_idx
    └── 支持图推理模式 (batch/cross_sectional/neighbor_sampling)
```

## 🔨 创建自定义模型

### 方法 1: 继承 PyTorchModel（推荐）

```python
import torch.nn as nn
from quantclassic.model import PyTorchModel, register_model

class MyNet(nn.Module):
    """自定义神经网络"""
    def __init__(self, d_feat, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(d_feat, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x, return_hidden=False):
        x = x[:, -1, :]  # 取最后时间步
        hidden = self.relu(self.fc1(x))
        pred = self.fc2(hidden).squeeze(-1)
        if return_hidden:
            return pred, hidden  # 支持相关性正则化
        return pred


@register_model('my_model')
class MyModel(PyTorchModel):
    """自定义模型"""
    
    def __init__(self, d_feat=20, hidden_size=64, **kwargs):
        super().__init__(**kwargs)
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        
        # 创建网络
        self.model = MyNet(d_feat, hidden_size).to(self.device)
        self.optimizer = self._get_optimizer()
        self.criterion = self._get_loss_fn()
    
    def fit(self, train_loader, valid_loader=None, save_path=None):
        """训练模型 - 使用 SimpleTrainer"""
        from quantclassic.model.train import SimpleTrainer, TrainerConfig
        
        config = TrainerConfig(
            n_epochs=self.n_epochs, lr=self.lr, early_stop=self.early_stop,
            loss_fn=self.loss_fn_name, lambda_corr=self.lambda_corr
        )
        trainer = SimpleTrainer(self.model, config, str(self.device))
        result = trainer.train(train_loader, valid_loader, save_path=save_path)
        
        self.fitted = True
        return result
    
    # predict() 继承自 PyTorchModel，无需实现
    # 如需自定义，可覆写 _forward_for_predict() 钩子
```

### 方法 2: 特殊输出模型（如 VAE）

```python
@register_model('my_vae')
class MyVAEModel(PyTorchModel):
    """VAE 类模型 - 需要特殊的前向逻辑"""
    
    def _forward_for_predict(self, x, adj=None, idx=None):
        """覆写前向钩子，只返回预测值"""
        _, y_pred, _, _, _ = self.model(x)  # VAE 返回多个输出
        return y_pred
    
    def predict(self, test_loader, return_numpy=True, return_latent=False):
        """扩展 predict 支持返回潜在特征"""
        if not return_latent:
            return super().predict(test_loader, return_numpy)
        
        # 自定义逻辑处理 return_latent
        ...
```

## 🎨 已实现的模型

| 模型 | 类名 | 注册名 | 特点 |
|------|------|--------|------|
| LSTM | `LSTMModel` | `'lstm'`, `'LSTM'` | 长短期记忆网络，适合时序 |
| GRU | `GRUModel` | `'gru'`, `'GRU'` | 参数更少，训练更快 |
| Transformer | `TransformerModel` | `'transformer'`, `'Transformer'` | 自注意力机制，捕捉长期依赖 |
| VAE | `VAEModel` | `'vae'`, `'VAE'` | 变分自编码器，因子提取、异常检测 |
| HybridGraph | `HybridGraphModel` | `'hybrid_graph'` | 🆕 时序+图混合模型 (RNN+Attention+GAT) |

## 🧩 混合图模型 (HybridGraphModel)

### 架构概述

```
输入: [batch, window, features]
           │
    ┌──────┴──────┐
    ▼             ▼
TemporalBlock  GraphBlock
 (RNN+Attn)     (GAT)
    │             │
    └──────┬──────┘
           ▼
      FusionBlock
        (MLP)
           │
           ▼
       预测输出
```

### 子模块说明

- **TemporalBlock**: RNN (LSTM/GRU) + Self-Attention + 残差连接
- **GraphBlock**: GAT 图注意力网络，支持行业图/相关性图/混合图
- **FusionBlock**: 多层 MLP + BatchNorm + 残差连接

### 使用示例

```python
from quantclassic.model import HybridGraphModel
from quantclassic.model.modular_config import ConfigTemplates

# 使用预定义模板
config = ConfigTemplates.temporal_with_graph(
    d_feat=20, gat_type='correlation', model_size='default'
)

model = HybridGraphModel(config)
model.fit(train_loader, val_loader)  # loader 需返回 (x, y, adj, ...)
predictions = model.predict(test_loader)
```

## 📋 训练配置参数

### TrainerConfig (train/base_trainer.py)

```python
from quantclassic.model.train import TrainerConfig

config = TrainerConfig(
    # 基础训练参数
    n_epochs=100,            # 训练轮数
    lr=0.001,                # 学习率
    early_stop=20,           # 早停耐心值
    
    # 优化器和损失
    optimizer='adam',        # 'adam', 'sgd', 'adamw'
    loss_fn='mse',           # 'mse', 'mae', 'huber', 'ic', 'mse_corr', 'unified' 等
    loss_kwargs={},          # 损失函数额外参数
    
    # 学习率调度器
    use_scheduler=True,
    scheduler_type='plateau',  # 'plateau', 'cosine', 'step'
    scheduler_patience=5,
    scheduler_factor=0.5,
    scheduler_min_lr=1e-6,
    
    # 🆕 相关性正则化（抑制特征冗余）
    lambda_corr=0.0,         # >0 启用，推荐 0.001~0.1
    
    # 检查点
    checkpoint_dir=None,
    save_best_only=True,
)
```

### 支持的损失函数

| 损失函数 | 说明 |
|----------|------|
| `mse` | 均方误差 |
| `mae` | 平均绝对误差 |
| `huber` | Huber 损失 |
| `ic` | 排序 IC Loss |
| `mse_corr` / `mae_corr` / `huber_corr` / `ic_corr` | 带相关性正则化 |
| `combined` | 组合损失 |
| `unified` | 统一损失 (UnifiedLoss) |

## 💾 模型保存和加载

```python
# 训练时自动保存
model.fit(train_loader, valid_loader, save_path='output/best_model.pth')

# 手动保存
model.save_model('output/my_model.pth')

# 加载模型
new_model = LSTMModel(d_feat=20, hidden_size=64)
new_model.load_model('output/best_model.pth')

# 继续训练
new_model.fit(train_loader, valid_loader)
```

## 🔗 与其他模块集成

### 图构建架构 (2026-01 重构)

```
graph_builder.py (HOW)        daily_graph_loader.py (WHEN)      base_model.py (WHO)
┌────────────────────┐       ┌──────────────────────┐        ┌─────────────────┐
│ GraphBuilderFactory│◄──────│ DailyGraphDataLoader │        │ _parse_batch_data│
│ ├─ industry        │       │   collate_daily()    │◄───────│   (x,y,adj,...)  │
│ ├─ correlation     │       │   每日触发图构建     │        │                  │
│ └─ hybrid          │       └──────────────────────┘        └─────────────────┘
└────────────────────┘
唯一实现入口                  数据加载时调用                  模型自动解析
```

- **图构建统一入口**: `data_processor/graph_builder.py` 的 `GraphBuilderFactory`
- **⚠️ 已废弃**: `model/utils/adj_matrix_builder.py`，请使用 `AdjMatrixUtils`

### 与 DataManager 集成

```python
from quantclassic.data_set import DataManager, DataConfig
from quantclassic.model import LSTMModel

# 1. 数据准备
config = DataConfig(base_dir='rq_data_parquet')
manager = DataManager(config)
loaders = manager.run_full_pipeline()

# 2. 模型训练
model = LSTMModel(d_feat=len(manager.feature_cols))
model.fit(loaders.train, loaders.val)

# 3. 预测
predictions = model.predict(loaders.test)
```

### 滚动训练

```python
from quantclassic.model.train import RollingWindowTrainer, RollingTrainerConfig

# 配置滚动训练
config = RollingTrainerConfig(
    n_epochs=50,
    weight_inheritance=True,    # 继承上窗口权重
    reset_optimizer=False,      # 🆕 保留优化器状态（动量）
    reset_scheduler=False,
    save_each_window=True,
)

# 创建滚动训练器
trainer = RollingWindowTrainer(model_factory, config)
results = trainer.train(rolling_loaders)
```

## 📊 完整工作流示例

```python
"""完整的量化研究流程"""

# 1. 数据准备
from quantclassic.data_set import DataManager, DataConfig
config = DataConfig(
    base_dir='rq_data_parquet',
    window_size=20,
    split_strategy='time_series'
)
manager = DataManager(config)
loaders = manager.run_full_pipeline()

# 2. 模型训练 (使用模块化配置)
from quantclassic.model.modular_config import ModelConfigBuilder
from quantclassic.model import create_model_from_composite_config

model_config = ModelConfigBuilder() \
    .set_input(d_feat=len(manager.feature_cols)) \
    .add_temporal(rnn_type='lstm', hidden_size=128, num_layers=3, use_attention=True) \
    .add_fusion(hidden_sizes=[128, 64]) \
    .set_training(n_epochs=200, lr=0.0005, early_stop=20, lambda_corr=0.01) \
    .build()

model = create_model_from_composite_config(model_config)
model.fit(
    loaders.train,
    loaders.val,
    save_path='output/best_model.pth'
)

# 3. 生成预测（自动支持各种 batch 格式）
predictions = model.predict(loaders.test)

# 4. 回测分析
from quantclassic.Factorsystem import FactorBacktestSystem, BacktestConfig
backtest_config = BacktestConfig(
    output_dir='output/backtest',
    save_plots=True
)
system = FactorBacktestSystem(backtest_config)

# 准备回测数据
test_df = manager.split_data[2]  # 测试集
test_df['factor'] = predictions

# 运行回测
results = system.run_backtest(test_df)

# 5. 查看结果
print(f"IC均值: {results['ic_stats']['ic_mean']:.4f}")
print(f"夏普比率: {results['performance_metrics']['long_short']['sharpe_ratio']:.4f}")
print(f"年化收益: {results['performance_metrics']['long_short']['annual_return']:.2%}")
```

## 🌟 VAE 模型详解

### VAE (Variational Autoencoder) 特性

VAE 是一种生成模型，在量化金融中特别适合：
- **因子提取**: 从高维特征中提取低维潜在因子
- **异常检测**: 通过重构误差检测异常交易模式
- **特征学习**: 学习数据的隐含结构

### VAE 模型架构

```
输入序列 [batch, window, features]
    ↓
编码器 (GRU) → 潜在空间 (μ, σ)
    ↓
重参数化 z = μ + ε·σ
    ↓
    ├→ 解码器 → 重构序列
    └→ 预测头 → 收益预测
```

### VAE 使用示例

```python
from quantclassic.model import VAEModel

# 创建 VAE 模型
vae_model = VAEModel(
    d_feat=20,              # 输入特征维度
    hidden_dim=128,         # GRU隐藏层维度
    latent_dim=16,          # 潜在空间维度
    window_size=40,         # 时间窗口
    dropout=0.3,
    
    # VAE 损失权重
    alpha_recon=0.1,        # 重构损失权重
    beta_kl=0.001,          # KL散度权重
    gamma_pred=1.0,         # 预测损失权重
    
    n_epochs=50,
    lr=0.001
)

# 训练
vae_model.fit(train_loader, valid_loader, save_path='output/vae_model.pth')

# 预测 + 提取潜在特征
predictions, latent_features = vae_model.predict(
    test_loader, 
    return_latent=True
)

# 或单独提取潜在特征（用于因子生成）
# 🆕 支持图/日级 loader，使用 _parse_batch_data 解析
mu, z = vae_model.extract_latent(test_loader)
```

### VAE 潜在特征可视化

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 提取潜在特征
mu, z = vae_model.extract_latent(test_loader)

# t-SNE 降维到 2D
tsne = TSNE(n_components=2)
z_2d = tsne.fit_transform(z)

# 可视化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Return')
plt.title('VAE Latent Space (t-SNE)')
plt.show()
```

## ⚙️ 配置系统迁移指南

### 旧配置 → 新配置

```python
# ❌ 旧方式（已废弃，会触发 DeprecationWarning）
from quantclassic.model.model_config import LSTMConfig, ModelConfigFactory
config = LSTMConfig(hidden_size=64, num_layers=2)

# ✅ 新方式 1: 使用模板
from quantclassic.model.modular_config import ConfigTemplates
config = ConfigTemplates.pure_temporal(d_feat=20, model_size='default')

# ✅ 新方式 2: 使用 Builder
from quantclassic.model.modular_config import ModelConfigBuilder
config = ModelConfigBuilder() \
    .set_input(d_feat=20) \
    .add_temporal(rnn_type='lstm', hidden_size=64, num_layers=2) \
    .add_fusion(hidden_sizes=[64]) \
    .build()

# ✅ 新方式 3: 直接构造
from quantclassic.model.modular_config import CompositeModelConfig, TemporalModuleConfig
config = CompositeModelConfig(
    temporal=TemporalModuleConfig(rnn_type='lstm', hidden_size=64),
    graph=None,
    d_feat=20
)
```

### 配置自动转换

```python
# 如果有旧配置对象，可自动转换
from quantclassic.model.model_config import to_composite_config
old_config = LSTMConfig(...)
new_config = to_composite_config(old_config)
```

## 🎯 已完成重构 (2026-01-11)

| 功能 | 状态 | 说明 |
|------|------|------|
| 统一 `predict()` 到基类 | ✅ | 支持 `(x,y)` / `(x,y,adj,...)` / `dict` 格式 |
| `fit()` 代理到 SimpleTrainer | ✅ | 保持接口兼容，内部使用统一训练引擎 |
| VAE.extract_latent 批次解包 | ✅ | 使用 `_parse_batch_data` 修复 unpack 错误 |
| 配置系统兼容层 | ✅ | 旧配置触发废弃警告，提供转换函数 |
| 图构建统一入口 | ✅ | `data_processor/graph_builder.py` |
| 滚动训练优化器状态保存 | ✅ | `reset_optimizer=False` 生效 |
| 损失函数白名单扩展 | ✅ | 支持 `mae_corr`, `unified` 等 |
| DailyRollingConfig 导出 | ✅ | `from model.train import DailyRollingConfig` |


## 📖 参考

- **Qlib**: https://github.com/microsoft/qlib
- **设计理念**: 参照 Qlib 的模型接口设计
- **VAE**: Kingma & Welling (2013) "Auto-Encoding Variational Bayes"

## 📝 更新日志

- **v2.0.0** (2026-01-11)
  - 🆕 统一训练引擎 `model/train/`，`fit()` 代理到 `SimpleTrainer`
  - 🆕 统一 `predict()` 方法到 `PyTorchModel` 基类
  - 🆕 模块化配置系统 `CompositeModelConfig`，旧配置标记废弃
  - 🆕 图构建合并到 `data_processor/graph_builder.py`
  - ✅ 修复 VAE.extract_latent 批次解包问题
  - ✅ 修复滚动训练优化器状态丢失问题
  - ✅ 扩展 TrainerConfig 损失函数支持列表

- **v1.1.0** (2025-11-19)
  - ✨ 添加 VAE (Variational Autoencoder) 模型
  - ✨ 支持潜在特征提取用于因子生成
  - ✅ 完善模型文档和使用示例

- **v1.0.0** (2025-11-19)
  - ✅ 创建模型基类系统
  - ✅ 实现模型工厂和注册机制
  - ✅ 添加 LSTM/GRU/Transformer 模型
  - ✅ 完整的使用示例

---

**Author**: QuantClassic Team  
**License**: Internal Use
