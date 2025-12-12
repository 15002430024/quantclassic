
# QuantClassic Model Module - 模型模块

标准化的量化模型接口和实现，参照 Qlib 设计。

## 📦 核心组件

```
model/
├── base_model.py           # 模型基类
├── model_factory.py        # 模型工厂和注册机制
├── pytorch_models.py       # PyTorch 模型实现
├── example_usage.py        # 完整使用示例
└── README.md              # 本文件
```

## ✨ 核心特性

### 🎯 统一接口
- **标准化**: 所有模型继承自 `Model` 基类
- **一致性**: 统一的 `fit()` 和 `predict()` 接口
- **兼容性**: 与 Qlib 设计理念一致

### 🏭 工厂模式
- **动态创建**: 通过配置字典创建模型
- **注册机制**: 使用装饰器注册模型
- **灵活配置**: 支持 YAML 配置文件

### 🚀 自动化功能
- **GPU 管理**: 自动检测和使用 GPU
- **早停机制**: 内置早停避免过拟合
- **模型保存**: 自动保存最佳模型
- **日志记录**: 完整的训练日志

### 🔧 PyTorch 优化
- **梯度裁剪**: 防止梯度爆炸
- **学习率调度**: 支持多种优化器
- **批量训练**: 高效的数据加载

## 🚀 快速开始

### 1. 基础使用

```python
from model import LSTMModel
from data_manager import DataManager, DataConfig

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

# 训练
model.fit(loaders.train, loaders.val, save_path='output/model.pth')

# 预测
predictions = model.predict(loaders.test)
```

### 2. 配置驱动

```python
from model import ModelFactory

# 模型配置
config = {
    'class': 'LSTM',
    'kwargs': {
        'd_feat': 20,
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'n_epochs': 200,
        'lr': 0.0005
    }
}

# 创建模型
model = ModelFactory.create_model(config)
model.fit(train_loader, valid_loader)
```

### 3. 模型对比

```python
from model import LSTMModel, GRUModel, TransformerModel, VAEModel

models = {
    'LSTM': LSTMModel(d_feat=20, hidden_size=64),
    'GRU': GRUModel(d_feat=20, hidden_size=64),
    'Transformer': TransformerModel(d_feat=20, d_model=64),
    'VAE': VAEModel(d_feat=20, hidden_dim=128, latent_dim=16)
}

results = {}
for name, model in models.items():
    model.fit(train_loader, valid_loader)
    predictions = model.predict(test_loader)
    results[name] = evaluate(predictions, labels)
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
    ├── 自动 GPU 管理
    ├── 内置早停机制
    ├── 模型保存/加载
    └── 训练循环封装
    
LSTMModel / GRUModel / TransformerModel
    └── 继承 PyTorchModel，实现具体模型
```

## 🔨 创建自定义模型

### 方法 1: 继承 PyTorchModel

```python
import torch.nn as nn
from model import PyTorchModel, register_model

class MyNet(nn.Module):
    """自定义神经网络"""
    def __init__(self, d_feat, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(d_feat, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x[:, -1, :]  # 取最后时间步
        x = self.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


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
        """训练模型"""
        for epoch in range(self.n_epochs):
            train_loss = self._train_epoch(train_loader)
            
            if valid_loader:
                valid_loss = self._valid_epoch(valid_loader)
                self.logger.info(
                    f"Epoch {epoch+1}: "
                    f"Train={train_loss:.6f}, Valid={valid_loss:.6f}"
                )
        
        self.fitted = True
    
    def predict(self, test_loader, return_numpy=True):
        """预测"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(self.device)
                pred = self.model(batch_x)
                predictions.append(pred.cpu())
        
        predictions = torch.cat(predictions)
        return predictions.numpy() if return_numpy else predictions
```

### 方法 2: 继承 Model (不使用 PyTorch)

```python
from model import Model, register_model
import lightgbm as lgb

@register_model('lgb')
class LightGBMModel(Model):
    """LightGBM 模型"""
    
    def __init__(self, num_leaves=31, learning_rate=0.05, n_estimators=100):
        super().__init__()
        self.params = {
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators
        }
        self.model = None
    
    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        """训练"""
        train_data = lgb.Dataset(X_train, label=y_train)
        
        if X_valid is not None:
            valid_data = lgb.Dataset(X_valid, label=y_valid)
            self.model = lgb.train(
                self.params,
                train_data,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(20)]
            )
        else:
            self.model = lgb.train(self.params, train_data)
        
        self.fitted = True
    
    def predict(self, X_test):
        """预测"""
        if not self.fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X_test)
```

## 🎨 已实现的模型

| 模型 | 类名 | 注册名 | 特点 |
|------|------|--------|------|
| LSTM | `LSTMModel` | `'lstm'`, `'LSTM'` | 长短期记忆网络，适合时序 |
| GRU | `GRUModel` | `'gru'`, `'GRU'` | 参数更少，训练更快 |
| Transformer | `TransformerModel` | `'transformer'`, `'Transformer'` | 自注意力机制，捕捉长期依赖 |
| VAE | `VAEModel` | `'vae'`, `'VAE'` | 变分自编码器，因子提取、异常检测 ✨ |

## 📋 模型参数说明

### LSTMModel / GRUModel

```python
model = LSTMModel(
    # 模型结构
    d_feat=20,           # 特征维度
    hidden_size=64,      # 隐藏层大小
    num_layers=2,        # RNN 层数
    dropout=0.1,         # Dropout 概率
    
    # 训练参数
    n_epochs=100,        # 训练轮数
    batch_size=256,      # 批量大小
    lr=0.001,            # 学习率
    early_stop=20,       # 早停耐心值
    
    # 优化器和损失
    optimizer='adam',    # 'adam', 'sgd', 'adamw'
    loss_fn='mse',      # 'mse', 'mae', 'huber'
    
    # 设备
    device=None         # None(自动), 'cuda', 'cpu'
)
```

### TransformerModel

```python
model = TransformerModel(
    d_feat=20,          # 特征维度
    d_model=64,         # Transformer 隐藏维度
    nhead=4,            # 注意力头数
    num_layers=2,       # Transformer 层数
    dropout=0.1,        # Dropout 概率
    # ... 其他参数同上
)
```

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

### 与 DataManager 集成

```python
from data_manager import DataManager, DataConfig
from model import LSTMModel

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

### 与 Factorsystem 集成

```python
from model import LSTMModel
from Factorsystem import FactorBacktestSystem, BacktestConfig

# 1. 训练模型
model = LSTMModel(d_feat=20)
model.fit(train_loader, valid_loader)

# 2. 生成因子
predictions = model.predict(test_loader)

# 3. 添加到数据框
df['factor'] = predictions

# 4. 回测
backtest_config = BacktestConfig()
system = FactorBacktestSystem(backtest_config)
results = system.run_backtest(df)
```

## 📊 完整工作流示例

```python
"""完整的量化研究流程"""

# 1. 数据准备
from data_manager import DataManager, DataConfig
config = DataConfig(
    base_dir='rq_data_parquet',
    window_size=20,
    split_strategy='time_series'
)
manager = DataManager(config)
loaders = manager.run_full_pipeline()

# 2. 模型训练
from model import ModelFactory
model_config = {
    'class': 'LSTM',
    'kwargs': {
        'd_feat': len(manager.feature_cols),
        'hidden_size': 128,
        'num_layers': 3,
        'n_epochs': 200,
        'lr': 0.0005,
        'early_stop': 20
    }
}
model = ModelFactory.create_model(model_config)
model.fit(
    loaders.train,
    loaders.val,
    save_path='output/best_model.pth'
)

# 3. 生成预测
predictions = model.predict(loaders.test)

# 4. 回测分析
from Factorsystem import FactorBacktestSystem, BacktestConfig
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
from model import VAEModel

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
mu, z = vae_model.extract_latent(test_loader)
```

### VAE 损失函数

VAE 使用三个损失的加权组合：

1. **重构损失** (Reconstruction Loss): 确保解码器能重构输入
   ```python
   L_recon = MSE(x_recon, x_true)
   ```

2. **KL 散度** (KL Divergence): 正则化潜在空间，使其接近标准正态分布
   ```python
   L_kl = -0.5 * mean(1 + log(σ²) - μ² - σ²)
   ```

3. **预测损失** (Prediction Loss): 监督学习收益预测
   ```python
   L_pred = MSE(y_pred, y_true)
   ```

总损失:
```python
L_total = α·L_recon + β·L_kl + γ·L_pred
```

### VAE 参数调优建议

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| `hidden_dim` | 64-256 | 编码器隐藏层大小 |
| `latent_dim` | 8-32 | 潜在空间维度（因子数量） |
| `alpha_recon` | 0.05-0.2 | 重构损失权重（较小） |
| `beta_kl` | 0.0001-0.01 | KL散度权重（很小） |
| `gamma_pred` | 0.5-2.0 | 预测损失权重（较大） |
| `dropout` | 0.2-0.4 | Dropout率 |

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
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()
```

### VAE 用于因子生成

```python
# 1. 训练 VAE
vae_model.fit(train_loader, valid_loader)

# 2. 提取潜在特征作为因子
mu_features, z_features = vae_model.extract_latent(test_loader)

# 3. 构建因子DataFrame
import pandas as pd
factor_df = pd.DataFrame({
    'latent_mean': mu_features.mean(axis=1),
    'latent_std': mu_features.std(axis=1),
    **{f'latent_{i}': mu_features[:, i] for i in range(mu_features.shape[1])}
})

# 4. 因子标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
factor_df_scaled = pd.DataFrame(
    scaler.fit_transform(factor_df),
    columns=factor_df.columns
)

# 5. 回测
from Factorsystem import FactorBacktestSystem
backtest_system = FactorBacktestSystem(backtest_config)
results = backtest_system.run_backtest(factor_df_scaled)
```

## 🎯 下一步计划

- [ ] 添加更多模型 (TabNet, TCN, ALSTM 等)
- [ ] 实现模型集成 (Ensemble)
- [ ] 添加超参数优化
- [ ] 实现增量学习
- [ ] 添加模型解释性工具
- [x] ✅ 添加 VAE 模型（因子提取、异常检测）
- [ ] 创建实验管理系统
- [ ] 支持分布式训练

## 📖 参考

- **Qlib**: https://github.com/microsoft/qlib
- **设计理念**: 参照 Qlib 的模型接口设计
- **VAE**: Kingma & Welling (2013) "Auto-Encoding Variational Bayes"

## 📝 更新日志

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
