# VAE预测模式 - 快速参考卡片

## 🎯 一分钟上手

### 核心改动

```python
# ❌ 旧方式（无监督）
factor_df = factor_gen.generate_factors(df)  
# 输出: factor_0, factor_1, ..., factor_15

# ✅ 新方式（端到端预测）
factor_df = factor_gen.generate_factors(df, mode='prediction')
# 输出: pred_alpha  <-- 直接用于选股
```

---

## 📝 完整代码模板

```python
# ========== 训练 ==========
from model.pytorch_models import VAEModel

model = VAEModel(
    d_feat=20,
    hidden_dim=128,
    latent_dim=16,
    window_size=40,
    gamma_pred=1.5,  # ⭐ 关键：强化预测
    n_epochs=100
)
model.fit(train_loader, valid_loader, save_path='best_vae.pth')

# ========== 推理 ==========
from Factorsystem.factor_generator import FactorGenerator
from Factorsystem.backtest_config import BacktestConfig

config = BacktestConfig(window_size=40, batch_size=512)
factor_gen = FactorGenerator(model.model, config)

# 生成Alpha因子
alpha_df = factor_gen.generate_factors(
    test_df,
    feature_cols=feature_columns,
    mode='prediction'  # 🎯 预测模式
)

# ========== 回测 ==========
from Factorsystem.backtest_engine import BacktestEngine

bt = BacktestEngine(config)
bt.run_backtest(alpha_df, factor_col='pred_alpha', price_data=test_df)
bt.print_summary()
```

---

## 🔧 参数速查

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|----------|------|
| `gamma_pred` | 1.0 | 1.0 - 2.0 | 预测损失权重（越大越重视Alpha预测） |
| `alpha_recon` | 0.1 | 0.05 - 0.3 | 重构损失权重（辅助学习） |
| `beta_kl` | 0.001 | 0.0001 - 0.01 | KL散度权重（正则化） |
| `latent_dim` | 16 | 8 - 32 | 潜在空间维度 |
| `window_size` | 40 | 30 - 60 | 时间窗口（天） |
| `hidden_dim` | 128 | 64 - 256 | GRU隐藏层大小 |

---

## 🎨 模式对比

### Prediction模式（推荐）
```python
factor_df = factor_gen.generate_factors(df, mode='prediction')
# 输出列: ts_code, trade_date, pred_alpha
# 用途: 直接用于选股排序
```

### Latent模式（研究）
```python
factor_df = factor_gen.generate_factors(df, mode='latent')
# 输出列: ts_code, trade_date, latent_0, latent_1, ..., latent_15
# 用途: 因子挖掘、降维分析
```

---

## ⚡ 性能优化

```python
# GPU加速
config = BacktestConfig(device='cuda', batch_size=1024)

# 多进程数据加载
train_loader = DataLoader(
    dataset, 
    batch_size=512, 
    num_workers=4,  # ⚡ 加速数据加载
    pin_memory=True
)
```

---

## 🐛 问题排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 预测值都接近0 | 预测权重太小 | 增大 `gamma_pred` 到 1.5-2.0 |
| 训练不收敛 | 学习率过高 | 降低 `lr` 到 0.0001 |
| 过拟合 | 正则化不足 | 增大 `beta_kl` 和 `dropout` |
| GPU内存不足 | batch太大 | 减小 `batch_size` 到 256 |

---

## 📊 输出格式示例

```python
>>> alpha_df.head()

   ts_code  trade_date  pred_alpha
0  000001.SZ  20231201    0.0234
1  000002.SZ  20231201   -0.0156
2  000003.SZ  20231201    0.0445
3  000004.SZ  20231201   -0.0023
4  000005.SZ  20231201    0.0189

>>> alpha_df['pred_alpha'].describe()
count    5000.000
mean        0.001
std         0.045
min        -0.234
25%        -0.023
50%         0.002
75%         0.025
max         0.198
```

---

## 🔗 相关命令

```bash
# 验证修改
python test_vae_simple.py

# 查看模型结构
python -c "from model.pytorch_models import VAENet; print(VAENet(20,128,16,40,0.3))"

# 运行完整流程
python workflow/train_vae_alpha.py
```

---

## 📖 进阶阅读

- 完整指南: `VAE_PREDICTION_MODE.md`
- 配置系统: `MIGRATION_GUIDE.md`
- 模型实现: `model/pytorch_models.py`
- 因子生成: `Factorsystem/factor_generator.py`

---

**快速参考 v1.0** | 最后更新: 2025-11-20
