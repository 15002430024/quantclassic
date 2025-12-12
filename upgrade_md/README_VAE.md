# QuantClassic - VAE Alpha因子系统

## 🎉 最新更新 (2025-11-20)

### ⭐ VAE端到端预测模式

现在支持**直接生成Alpha预测值**作为因子，完全对齐研报方法！

```python
from model.pytorch_models import VAEModel
from Factorsystem.factor_generator import FactorGenerator

# 训练模型
model = VAEModel(gamma_pred=1.5)  # 强化预测任务
model.fit(train_loader, valid_loader)

# 生成Alpha因子
factor_gen = FactorGenerator(model.model, config)
alpha_df = factor_gen.generate_factors(
    test_df, 
    mode='prediction'  # 🆕 新增：提取预测值
)

# 输出: ts_code, trade_date, pred_alpha
# 直接用于选股回测！
```

**核心改进**:
- ✅ 移除Tanh限制，支持预测任意范围的收益率
- ✅ 新增prediction/latent双模式，灵活切换
- ✅ 完全对齐研报的端到端学习方法
- ✅ 保持向后兼容，旧代码无需修改

**快速开始**: 查看 [VAE_QUICKREF.md](./VAE_QUICKREF.md)  
**完整指南**: 查看 [VAE_PREDICTION_MODE.md](./VAE_PREDICTION_MODE.md)  
**修改详情**: 查看 [VAE_MODIFICATION_SUMMARY.md](./VAE_MODIFICATION_SUMMARY.md)

---

## 📚 文档导航

### 核心文档
- **[CONFIG_README.md](./CONFIG_README.md)** - 配置系统总览
- **[MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md)** - 配置迁移指南
- **[CONFIG_QUICKREF.md](./CONFIG_QUICKREF.md)** - 配置快速参考

### VAE模块
- **[VAE_QUICKREF.md](./VAE_QUICKREF.md)** - VAE快速参考 ⭐
- **[VAE_PREDICTION_MODE.md](./VAE_PREDICTION_MODE.md)** - VAE完整指南
- **[VAE_MODIFICATION_SUMMARY.md](./VAE_MODIFICATION_SUMMARY.md)** - 修改总结

### 其他
- **[REFACTORING_SUMMARY.md](./REFACTORING_SUMMARY.md)** - OOP重构总结
- **[REFACTORING_COMPLETE.md](./REFACTORING_COMPLETE.md)** - 重构完成报告

---

## 🚀 快速开始

### 1. 配置系统（OOP方式）

```python
from model.model_config import ModelConfigFactory, VAEConfig

# 使用模板快速创建
vae_config = ModelConfigFactory.get_template('vae_alpha')

# 或自定义配置
vae_config = VAEConfig(
    d_feat=20,
    hidden_dim=128,
    latent_dim=16,
    gamma_pred=1.5  # 强化Alpha预测
)

# 保存配置
vae_config.to_yaml('my_vae_config.yaml')
```

### 2. 训练模型

```python
from model.pytorch_models import VAEModel
from model.model_config import ModelConfigFactory

# 加载配置
config = ModelConfigFactory.from_yaml('my_vae_config.yaml')

# 创建模型
model = VAEModel(**config.to_dict())

# 训练
model.fit(train_loader, valid_loader, save_path='best_model.pth')
```

### 3. 生成因子

```python
from Factorsystem.factor_generator import FactorGenerator
from Factorsystem.backtest_config import BacktestConfig

# 创建因子生成器
config = BacktestConfig(window_size=40, batch_size=512)
factor_gen = FactorGenerator(model.model, config)

# 生成Alpha因子（预测模式）
alpha_df = factor_gen.generate_factors(
    test_df,
    feature_cols=feature_columns,
    mode='prediction'  # 端到端预测
)

# 或生成隐变量因子（研究模式）
latent_df = factor_gen.generate_factors(
    test_df,
    feature_cols=feature_columns,
    mode='latent'  # 无监督特征
)
```

### 4. 回测评估

```python
from Factorsystem.backtest_engine import BacktestEngine

bt = BacktestEngine(config)
bt.run_backtest(
    alpha_df,
    factor_col='pred_alpha',  # 使用预测值
    price_data=test_df
)

bt.print_summary()
bt.plot_cumulative_returns()
```

---

## 🏗️ 项目结构

```
quantclassic/
├── config/                      # 配置系统
│   ├── base_config.py          # 基础配置类
│   ├── loader.py               # 配置加载器
│   └── templates/              # 配置模板
│       └── vae_oop.yaml
│
├── model/                       # 模型模块
│   ├── base_model.py           # 模型基类
│   ├── pytorch_models.py       # PyTorch模型（LSTM/GRU/Transformer/VAE）
│   ├── model_config.py         # 模型配置类
│   └── model_factory.py        # 模型工厂
│
├── data_manager/                # 数据管理
│   ├── config.py               # 数据配置
│   ├── loader.py               # 数据加载
│   └── splitter.py             # 数据分割
│
├── data_processor/              # 数据处理
│   ├── preprocess_config.py    # 预处理配置
│   └── pipeline.py             # 处理流程
│
├── Factorsystem/                # 因子系统
│   ├── factor_generator.py     # 因子生成器 ⭐
│   ├── backtest_config.py      # 回测配置
│   └── backtest_engine.py      # 回测引擎
│
├── workflow/                    # 工作流
│   ├── workflow_config.py      # 工作流配置
│   └── experiment.py           # 实验管理
│
└── 文档/
    ├── VAE_QUICKREF.md         # VAE快速参考 ⭐
    ├── VAE_PREDICTION_MODE.md  # VAE完整指南
    ├── CONFIG_README.md        # 配置系统README
    └── MIGRATION_GUIDE.md      # 迁移指南
```

---

## 🎯 核心特性

### 1. 端到端Alpha预测 🆕

- 直接输出可交易的Alpha因子
- 移除Tanh限制，预测任意范围收益率
- 完全对齐量化研报方法

### 2. 双模式因子生成

- **Prediction模式**: 提取Alpha预测值（监督学习）
- **Latent模式**: 提取隐变量特征（无监督学习）

### 3. 面向对象配置系统

- 类型安全的配置类
- 自动验证和错误检测
- YAML/JSON序列化支持

### 4. 模型工厂与模板

- 快速创建常用模型配置
- 预定义模板（vae_alpha, lstm_basic, etc.）
- 灵活的自定义扩展

### 5. 完整的回测框架

- IC分析
- 分层回测
- 累计收益曲线
- 风险指标计算

---

## 📊 模型对比

| 模型 | 特点 | 适用场景 | 性能 |
|------|------|----------|------|
| **VAE** | 端到端预测 + 特征学习 | Alpha因子生成 | ⭐⭐⭐⭐⭐ |
| LSTM | 长短期记忆 | 趋势预测 | ⭐⭐⭐⭐ |
| GRU | 简化版LSTM | 快速训练 | ⭐⭐⭐⭐ |
| Transformer | 自注意力机制 | 长期依赖 | ⭐⭐⭐ |

---

## 🔧 配置示例

### VAE Alpha配置

```yaml
# config/templates/vae_alpha.yaml
model_type: vae
model_params:
  d_feat: 20
  hidden_dim: 128
  latent_dim: 16
  window_size: 40
  dropout: 0.3
  
  # 损失权重（关键）
  alpha_recon: 0.1    # 重构
  beta_kl: 0.001      # KL散度
  gamma_pred: 1.5     # Alpha预测 ⭐

training:
  n_epochs: 100
  batch_size: 512
  lr: 0.001
  early_stop: 10
  optimizer: adam

device: cuda
```

### 数据配置

```yaml
# data_config.yaml
data_path: rq_data_parquet/daily_data
split_method: date
split_params:
  train_ratio: 0.7
  valid_ratio: 0.15
  test_ratio: 0.15

feature_engineering:
  enabled: true
  methods:
    - zscore
    - winsorize
    - neutralize
```

---

## 🧪 测试与验证

### 验证VAE修改

```bash
# 运行验证脚本
python test_vae_simple.py

# 期望输出
✅ 已成功移除Tanh激活函数
✅ generate_factors方法包含mode参数
✅ 包含prediction模式分支
✅ 包含latent模式分支
```

### 完整流程测试

```bash
# 配置系统测试
python test_config_standalone.py

# 检查编译错误
python -c "from get_errors import check_all; check_all()"
```

---

## 📈 性能优化建议

### GPU加速

```python
# 配置
config = BacktestConfig(
    device='cuda',
    batch_size=1024  # GPU可用更大batch
)

# DataLoader优化
train_loader = DataLoader(
    dataset,
    batch_size=512,
    num_workers=4,      # 多进程加载
    pin_memory=True,    # 加速GPU传输
    persistent_workers=True
)
```

### 超参数调优

参考 [VAE_PREDICTION_MODE.md](./VAE_PREDICTION_MODE.md) 第⚙️节

---

## 🔗 常用命令

```bash
# 训练VAE模型
python workflow/train_vae_alpha.py

# 生成因子
python Factorsystem/generate_factors.py --mode prediction

# 运行回测
python Factorsystem/run_backtest.py --factor pred_alpha

# 配置验证
python config/validate_config.py --config my_config.yaml
```

---

## 📖 学习路径

### 新手入门

1. 阅读 [VAE_QUICKREF.md](./VAE_QUICKREF.md) - 快速上手
2. 运行示例代码 - 理解工作流
3. 阅读 [CONFIG_README.md](./CONFIG_README.md) - 了解配置

### 进阶使用

1. 阅读 [VAE_PREDICTION_MODE.md](./VAE_PREDICTION_MODE.md) - 深入理解
2. 阅读 [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md) - 配置最佳实践
3. 自定义模型和因子 - 扩展功能

### 高级研究

1. 修改模型结构 - `model/pytorch_models.py`
2. 自定义因子逻辑 - `Factorsystem/factor_generator.py`
3. 优化训练流程 - `workflow/experiment.py`

---

## 🤝 贡献指南

### 报告问题

1. 检查是否已存在相同issue
2. 提供最小可复现代码
3. 附上错误日志和环境信息

### 提交代码

1. Fork项目
2. 创建feature分支
3. 遵循代码风格
4. 添加测试用例
5. 提交Pull Request

---

## 📄 许可证

MIT License

---

## 🙏 致谢

- 感谢Qlib项目提供的基础框架
- 感谢FactorVAE研报提供的理论指导
- 感谢所有贡献者的支持

---

## 📞 联系方式

- 📧 Email: [Your Email]
- 💬 Issues: [GitHub Issues]
- 📚 Docs: [Documentation Site]

---

**最后更新**: 2025-11-20  
**版本**: v2.0  
**状态**: ✅ 生产就绪
