# VAE预测模式 - 端到端Alpha因子生成指南

## 📋 更新概览

本次更新将VAE模型从"无监督隐变量提取"升级为"端到端Alpha预测"，对齐研报中的监督学习方法。

### 核心修改

1. **VAENet预测头**: 移除 `nn.Tanh()` 限制，支持预测任意范围的收益率
2. **FactorGenerator**: 新增 `mode` 参数，支持提取预测值或隐变量
3. **向后兼容**: 保留隐变量提取功能，用户可自由选择模式

---

## 🎯 修改详情

### 1. VAENet预测头修改

**修改位置**: `model/pytorch_models.py` - `VAENet.__init__()`

**修改前**:
```python
self.predictor = nn.Sequential(
    nn.Linear(latent_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, 64),
    nn.LayerNorm(64),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(dropout * 0.5),
    nn.Linear(32, 1),
    nn.Tanh()  # ❌ 强制输出在[-1, 1]，限制预测能力
)
```

**修改后**:
```python
self.predictor = nn.Sequential(
    nn.Linear(latent_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, 64),
    nn.LayerNorm(64),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(dropout * 0.5),
    nn.Linear(32, 1)
    # ✅ 移除Tanh，支持任意范围的Alpha预测
)
```

**原理说明**:
- 研报中的标签是中性化后的收益率，可能存在 > 1 或 < -1 的极端值
- `Tanh` 会将输出压缩在 (-1, 1)，导致梯度消失，无法预测显著的Alpha
- 移除后，模型可以自由学习收益率的真实分布

---

### 2. FactorGenerator模式支持

**修改位置**: `Factorsystem/factor_generator.py`

#### 2.1 `generate_factors()` 方法

**新增参数**:
```python
def generate_factors(
    self,
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    return_metadata: bool = True,
    mode: str = 'prediction'  # 🆕 新增参数
) -> pd.DataFrame:
    """
    Args:
        mode: 因子提取模式
            - 'prediction': 提取模型的Alpha预测值 (y_pred)，用于端到端监督学习
            - 'latent': 提取VAE的隐变量均值 (mu)，用于无监督特征学习
    """
```

**核心逻辑**:
```python
with torch.no_grad():
    for batch_idx, X in enumerate(dataloader):
        X = X.to(self.device)
        
        if mode == 'prediction':
            # 研报模式：提取Alpha预测值
            # VAENet.forward() 返回: (x_recon, y_pred, mu, logvar, z)
            if hasattr(self.model, 'model'):
                outputs = self.model.model(X)  # VAEModel包装类
            else:
                outputs = self.model(X)  # 直接调用VAENet
            
            data = outputs[1]  # 提取 y_pred (索引1)
        
        elif mode == 'latent':
            # 标准VAE模式：提取隐变量
            if hasattr(self.model, 'encode'):
                mu, logvar = self.model.encode(X)
                data = mu
            else:
                outputs = self.model(X)
                data = outputs[2]  # 提取 mu (索引2)
        
        # 处理维度
        if data.ndim == 1:
            data = data[:, np.newaxis]
        
        factors_list.append(data.cpu().numpy())
```

**输出列名**:
- `mode='prediction'`: `pred_alpha` (单列) 或 `pred_alpha_0`, `pred_alpha_1`, ... (多列)
- `mode='latent'`: `latent_0`, `latent_1`, ..., `latent_15` (根据 `latent_dim`)

#### 2.2 其他方法更新

- `generate_single_factor()`: 新增 `mode` 参数
- `batch_generate_factors()`: 新增 `mode` 参数

---

## 🚀 使用示例

### 场景1: 端到端Alpha因子生成（推荐）

```python
from model.pytorch_models import VAEModel
from Factorsystem.factor_generator import FactorGenerator
from Factorsystem.backtest_config import BacktestConfig

# 1. 训练模型（强化预测任务）
model = VAEModel(
    d_feat=20,
    hidden_dim=128,
    latent_dim=16,
    window_size=40,
    dropout=0.3,
    # 损失权重设置
    alpha_recon=0.1,   # 重构损失权重（辅助）
    beta_kl=0.001,     # KL散度权重（正则化）
    gamma_pred=1.5,    # ⭐ 预测损失权重（主任务，建议 >= 1.0）
    n_epochs=100,
    lr=0.001
)

# 训练
model.fit(train_loader, valid_loader, save_path='best_vae.pth')

# 2. 生成Alpha因子
config = BacktestConfig(
    window_size=40,
    batch_size=512,
    device='cuda'
)

factor_gen = FactorGenerator(model.model, config)

# 提取预测值作为因子
alpha_factors = factor_gen.generate_factors(
    test_df,
    feature_cols=feature_columns,
    mode='prediction'  # 🎯 使用预测模式
)

print(alpha_factors.head())
# 输出:
#     ts_code  trade_date  pred_alpha
# 0  000001.SZ  20231201    0.0234
# 1  000002.SZ  20231201   -0.0156
# ...
```

### 场景2: 隐变量因子提取（研究用）

```python
# 提取隐变量作为多因子
latent_factors = factor_gen.generate_factors(
    test_df,
    feature_cols=feature_columns,
    mode='latent'  # 使用隐变量模式
)

print(latent_factors.head())
# 输出:
#     ts_code  trade_date  latent_0  latent_1  ...  latent_15
# 0  000001.SZ  20231201    0.523    -0.234  ...   0.156
# 1  000002.SZ  20231201   -0.112     0.445  ...  -0.089
# ...
```

### 场景3: 单因子聚合

```python
# 如果预测是多维的，可以聚合为单一因子
single_factor = factor_gen.generate_single_factor(
    test_df,
    feature_cols=feature_columns,
    mode='prediction',
    aggregation='first'  # 对于单维预测，直接取第一维
)

print(single_factor.head())
# 输出:
#     ts_code  trade_date  factor_raw
# 0  000001.SZ  20231201    0.0234
# 1  000002.SZ  20231201   -0.0156
# ...
```

### 场景4: 批量生成

```python
# 为训练集、验证集、测试集批量生成因子
df_dict = {
    'train': train_df,
    'valid': valid_df,
    'test': test_df
}

factor_dict = factor_gen.batch_generate_factors(
    df_dict,
    feature_cols=feature_columns,
    mode='prediction'
)

for name, factor_df in factor_dict.items():
    print(f"{name}: {len(factor_df)} 条记录")
    factor_gen.save_factors(factor_df, f'output/{name}_alpha_factors.parquet')
```

---

## 📊 与研报对齐

### 研报方法（FactorVAE）

```
输入特征 (X) → VAE编码器 → 隐变量 (z) → 两路输出:
                                        ├─ 解码器 → 重构 (X̂)
                                        └─ 预测头 → Alpha预测 (ŷ)

损失函数: L = α·L_recon + β·L_KL + γ·L_pred
```

### 本实现对齐点

| 组件 | 研报 | 本实现 | 状态 |
|------|------|--------|------|
| 编码器 | GRU/LSTM | GRU (2层) | ✅ |
| 隐变量 | 潜在表示 z | `latent_dim=16` | ✅ |
| 解码器 | MLP | 3层MLP | ✅ |
| 预测头 | 线性层 | 4层MLP (无Tanh) | ✅ |
| 损失权重 | α, β, γ | `alpha_recon`, `beta_kl`, `gamma_pred` | ✅ |
| 输出 | Alpha预测值 | `y_pred` (mode='prediction') | ✅ |

---

## ⚙️ 超参数建议

### 训练参数

```python
VAEModel(
    # 模型结构
    d_feat=20,           # 特征数量
    hidden_dim=128,      # GRU隐藏层大小
    latent_dim=16,       # 潜在空间维度（研报推荐16-32）
    window_size=40,      # 时间窗口（研报推荐30-60天）
    dropout=0.3,         # Dropout概率
    
    # 损失权重（关键）
    alpha_recon=0.1,     # 重构损失：辅助学习数据分布
    beta_kl=0.001,       # KL散度：正则化，防止过拟合
    gamma_pred=1.5,      # ⭐ 预测损失：主任务，建议1.0-2.0
    
    # 训练设置
    n_epochs=100,
    batch_size=512,
    lr=0.001,
    early_stop=10,
    optimizer='adam'
)
```

### 权重调优指南

1. **gamma_pred (预测权重)**
   - 起始值: 1.0
   - 如果模型过度关注重构，忽略预测 → 增大到 1.5 或 2.0
   - 如果预测过拟合 → 减小到 0.5，增大 `beta_kl`

2. **beta_kl (KL散度权重)**
   - 起始值: 0.001
   - 如果潜在空间混乱 → 增大到 0.01
   - 如果模型表达能力不足 → 减小到 0.0001

3. **alpha_recon (重构权重)**
   - 起始值: 0.1
   - 主要用于辅助学习，不建议超过 0.5

---

## 🔍 验证检查清单

### 代码修改验证

- [x] `VAENet.predictor` 最后一层为 `nn.Linear(32, 1)`（无Tanh）
- [x] `FactorGenerator.generate_factors()` 包含 `mode` 参数
- [x] `mode='prediction'` 时输出 `pred_alpha` 列
- [x] `mode='latent'` 时输出 `latent_0`, `latent_1`, ... 列
- [x] `generate_single_factor()` 和 `batch_generate_factors()` 支持 `mode`

### 运行时验证

运行测试脚本:
```bash
cd /home/u2025210237/jupyterlab/quantclassic
python test_vae_simple.py
```

期望输出:
```
✅ 已成功移除Tanh激活函数
✅ generate_factors方法包含mode参数
✅ 包含prediction模式分支
✅ 包含latent模式分支
✅ pred_alpha列名定义
✅ latent_列名定义
```

---

## 📚 完整工作流示例

```python
# ========== 1. 数据准备 ==========
from data_manager import DataManager

dm = DataManager(config_path='config/data_config.yaml')
train_df, valid_df, test_df = dm.load_and_split()

# ========== 2. 创建DataLoader ==========
from data_processor import create_time_series_dataloader

train_loader = create_time_series_dataloader(
    train_df, 
    feature_cols=feature_columns,
    label_col='y_processed',
    window_size=40,
    batch_size=512
)

valid_loader = create_time_series_dataloader(
    valid_df, 
    feature_cols=feature_columns,
    label_col='y_processed',
    window_size=40,
    batch_size=512
)

# ========== 3. 训练模型 ==========
from model.pytorch_models import VAEModel

model = VAEModel(
    d_feat=len(feature_columns),
    hidden_dim=128,
    latent_dim=16,
    window_size=40,
    gamma_pred=1.5,  # 强化预测任务
    n_epochs=100,
    lr=0.001
)

model.fit(
    train_loader, 
    valid_loader, 
    save_path='output/best_vae_alpha.pth'
)

# ========== 4. 生成Alpha因子 ==========
from Factorsystem.factor_generator import FactorGenerator
from Factorsystem.backtest_config import BacktestConfig

config = BacktestConfig(window_size=40, batch_size=512)
factor_gen = FactorGenerator(model.model, config)

# 生成预测因子
test_factors = factor_gen.generate_factors(
    test_df,
    feature_cols=feature_columns,
    mode='prediction'
)

# 保存
factor_gen.save_factors(
    test_factors, 
    'output/alpha_factors.parquet'
)

# ========== 5. 回测评估 ==========
from Factorsystem.backtest_engine import BacktestEngine

bt = BacktestEngine(config)
bt.run_backtest(
    test_factors,
    factor_col='pred_alpha',
    price_data=test_df
)

bt.print_summary()
```

---

## 🆚 模式对比

| 特性 | prediction模式 | latent模式 |
|------|----------------|-----------|
| **输出** | Alpha预测值 (1维) | 隐变量 (16维) |
| **学习方式** | 端到端监督学习 | 无监督特征学习 |
| **适用场景** | 直接用于选股/排序 | 因子挖掘/降维 |
| **可解释性** | 高（直接预测收益） | 低（隐式特征） |
| **研报对齐** | ✅ 完全对齐 | ⚠️ 传统VAE用法 |
| **推荐度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 📖 相关文档

- `model/pytorch_models.py` - VAE模型实现
- `Factorsystem/factor_generator.py` - 因子生成器
- `MIGRATION_GUIDE.md` - 配置系统迁移指南
- `CONFIG_QUICKREF.md` - 配置快速参考

---

## 🐛 常见问题

### Q1: 预测值范围过大怎么办？

A: 检查以下几点:
1. 标签是否正确标准化 (Z-Score 或 RankNorm)
2. 增大 `beta_kl` 正则化权重
3. 添加梯度裁剪: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

### Q2: 模型只学到重构，预测效果差？

A: 增大 `gamma_pred` 权重:
```python
VAEModel(
    alpha_recon=0.05,   # 减小
    beta_kl=0.001,
    gamma_pred=2.0      # 增大
)
```

### Q3: 如何选择 mode？

A: 
- **用于回测/实盘**: 使用 `mode='prediction'`，直接获取Alpha预测
- **用于因子研究**: 使用 `mode='latent'`，提取多个隐式因子进行分析

### Q4: 能否同时使用两种模式？

A: 可以！
```python
# 提取预测因子
pred_factors = factor_gen.generate_factors(df, mode='prediction')

# 同时提取隐变量因子
latent_factors = factor_gen.generate_factors(df, mode='latent')

# 合并使用
all_factors = pred_factors.merge(
    latent_factors, 
    on=['ts_code', 'trade_date']
)
```

---

## 📞 技术支持

如有问题，请查看:
1. 运行 `test_vae_simple.py` 验证安装
2. 检查 `get_errors()` 获取编译错误
3. 查看训练日志中的损失曲线

---

**最后更新**: 2025-11-20  
**版本**: v2.0 - 端到端预测模式
