# DataManager 使用指南

## 🎯 简介

DataManager 是一个完整的工程化数据管理模块，已成功封装并通过所有测试。

## ✅ 模块状态

```
✅ 所有测试通过 (6/6)
✅ 完整文档
✅ 10个使用示例
✅ 快速开始脚本
✅ 滚动窗口训练支持（新增）
```

## 🆕 新功能：滚动窗口训练

DataManager 现在支持**滚动窗口（Walk-Forward）模型训练**，这是量化金融中最严谨的时间序列验证方法。

**快速使用**:
```python
# 1. 配置rolling策略
config = DataConfig(split_strategy='rolling', rolling_window_size=252, rolling_step=63)

# 2. 创建训练器
dm = DataManager(config)
dm.run_full_pipeline()
trainer = dm.create_rolling_window_trainer()

# 3. 训练所有窗口
results = trainer.train_all_windows(model_class=GRUModel, model_config=gru_config)

# 4. 预测并合并
predictions = trainer.predict_all_windows(results)
```

**详细文档**: 参见 [滚动窗口训练指南](./ROLLING_WINDOW_GUIDE.md)

## 🚀 三种使用方式

### 方式1: 快速开始脚本（推荐新手）

```bash
cd /home/u2025210237/jupyterlab/quantclassic/data_manager
python quickstart.py
```

### 方式2: 一键运行完整流水线

```python
from data_manager import DataManager, DataConfig

# 创建配置
config = DataConfig(
    base_dir='rq_data_parquet',
    data_file='train_data_final.parquet',
    window_size=40,
    batch_size=256
)

# 一键运行
manager = DataManager(config)
loaders = manager.run_full_pipeline()

# 开始训练
for batch_x, batch_y in loaders.train:
    # 您的训练代码
    pass
```

### 方式3: 分步骤执行（推荐高级用户）

```python
from data_manager import DataManager, DataConfig

config = DataConfig()
manager = DataManager(config)

# 步骤1: 加载数据
raw_data = manager.load_raw_data()

# 步骤2: 验证数据（可选）
report = manager.validate_data_quality()

# 步骤3: 特征工程
feature_cols = manager.preprocess_features(auto_filter=True)

# 步骤4-5: 创建数据集
datasets = manager.create_datasets()

# 步骤6: 获取数据加载器
loaders = manager.get_dataloaders(batch_size=256)
```

## 📊 实际使用示例

### 示例1: 训练VAE-NeuralODE模型

```python
from data_manager import DataManager, DataConfig
import torch
import torch.nn as nn

# 1. 准备数据
config = DataConfig(
    base_dir='rq_data_parquet',
    data_file='train_data_final.parquet',
    window_size=40,
    batch_size=256,
    split_strategy='time_series'
)

manager = DataManager(config)
loaders = manager.run_full_pipeline()

# 2. 创建模型（使用data.py中的模型）
from data_manager.data import VAE_NeuralODE, Config

model_config = Config()
model_config.LATENT_DIM = 32
model_config.HIDDEN_DIM = 64
model_config.WINDOW_SIZE = config.window_size

input_dim = len(manager.feature_cols)
model = VAE_NeuralODE(model_config, input_dim).to(config.DEVICE)

# 3. 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for batch_x, batch_y in loaders.train:
        batch_x = batch_x.to(config.DEVICE)
        batch_y = batch_y.to(config.DEVICE)
        
        # 前向传播
        x_recon, y_pred, mu, logvar = model(batch_x)
        
        # 计算损失（使用data.py中的损失函数）
        # loss = compute_loss(...)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 示例2: 回测场景（滚动窗口）

```python
from data_manager import DataManager, DataConfig

# 配置滚动窗口
config = DataConfig(
    split_strategy='rolling',
    rolling_window_size=252,  # 1年训练窗口
    rolling_step=63           # 每季度滚动一次
)

manager = DataManager(config)
# 注意：滚动窗口返回多个划分，需要特殊处理
```

### 示例3: 不同数据源

```python
# CSV格式
config = DataConfig(
    base_dir='data',
    data_file='stock_data.csv',
    data_format='csv'
)

# HDF5格式
config = DataConfig(
    base_dir='data',
    data_file='stock_data.h5',
    data_format='hdf5'
)
```

## 🔧 常用配置

### 快速测试配置

```python
from data_manager import ConfigTemplates

config = ConfigTemplates.quick_test()
# 特点: 小批量，快速验证代码
```

### 生产环境配置

```python
config = ConfigTemplates.production()
# 特点: 大批量，多进程，高性能
```

### 自定义配置

```python
config = DataConfig(
    # 数据
    base_dir='rq_data_parquet',
    data_file='train_data_final.parquet',
    
    # 特征
    window_size=60,
    label_col='y_processed',
    
    # 划分
    split_strategy='time_series',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    
    # 加载
    batch_size=512,
    num_workers=4,
    
    # 优化
    use_dtype_optimization=True,
    enable_cache=True,
)
```

## 📝 与data.py的集成

DataManager 可以完全替代 data.py 中的数据管理部分：

### 之前（data.py）

```python
# data.py 中的代码
df = pd.read_parquet('rq_data_parquet/train_data_final.parquet')
train_loader, val_loader, test_loader = create_dataloaders(df, config)
```

### 之后（使用DataManager）

```python
# 使用DataManager
from data_manager import DataManager, DataConfig

config = DataConfig()
manager = DataManager(config)
loaders = manager.run_full_pipeline()

# loaders.train, loaders.val, loaders.test
# 完全兼容原有的训练代码
```

### 保留data.py的模型部分

```python
# 继续使用data.py中的模型和训练器
from data_manager.data import VAE_NeuralODE, Trainer, FactorGenerator

# 只替换数据管理部分
from data_manager import DataManager, DataConfig

# 数据准备
manager = DataManager(DataConfig())
loaders = manager.run_full_pipeline()

# 模型训练（使用data.py的代码）
trainer = Trainer(config)
model, train_losses, val_losses = trainer.train(loaders.train, loaders.val)
```

## 🎨 高级功能

### 1. 特征过滤

```python
manager = DataManager(config)
raw_data = manager.load_raw_data()

# 自定义过滤条件
filtered_features = manager.feature_engineer.filter_features(
    raw_data,
    min_variance=1e-5,
    max_missing_ratio=0.3,
    max_correlation=0.95
)
```

### 2. 数据验证

```python
# 获取详细验证报告
report = manager.validate_data_quality()

if not report.is_valid:
    print("数据质量问题:")
    for error in report.errors:
        print(f"  ❌ {error}")
    
    for warning in report.warnings:
        print(f"  ⚠️ {warning}")
```

### 3. 状态保存

```python
# 第一次运行
manager = DataManager(config)
loaders = manager.run_full_pipeline()
manager.save_state('cache/my_state.pkl')

# 后续运行（快速加载）
manager = DataManager()
manager.load_state('cache/my_state.pkl')
loaders = manager.get_dataloaders()
```

### 4. 推理数据集

```python
# 创建推理数据集（无标签）
inference_dataset = manager.factory.create_inference_dataset(
    df=test_data,
    feature_cols=manager.feature_cols
)

# 用于因子生成
for sample in inference_dataset:
    factor = model.predict(sample)
```

## 📚 更多资源

- **完整文档**: `README.md`
- **使用示例**: `examples.py`（10个示例）
- **快速开始**: `quickstart.py`
- **项目总结**: `SUMMARY.md`
- **测试**: `test_module.py`

## 🔗 模块导入

```python
# 主要类
from data_manager import DataManager, DataConfig, ConfigTemplates

# 组件类
from data_manager import (
    DataLoaderEngine,      # 数据加载器
    FeatureEngineer,       # 特征工程师
    DataValidator,         # 数据验证器
    DatasetFactory,        # 数据集工厂
)

# 划分器
from data_manager import (
    TimeSeriesSplitter,         # 时间序列划分
    StratifiedStockSplitter,    # 分层划分
    RollingWindowSplitter,      # 滚动窗口
    create_splitter,            # 工厂函数
)

# 数据结构
from data_manager import (
    DatasetCollection,     # 数据集集合
    LoaderCollection,      # 加载器集合
    ValidationReport,      # 验证报告
)
```

## ⚠️ 注意事项

1. **路径配置**: 确保 `base_dir` 和 `data_file` 指向正确的数据文件
2. **内存管理**: 处理大数据集时设置 `chunk_size` 和 `use_dtype_optimization=True`
3. **时序数据**: 使用 `time_series` 或 `stratified` 划分，避免 `random`
4. **GPU训练**: 设置 `pin_memory=True` 和 `num_workers>0`

## 🐛 故障排除

### 问题: 找不到数据文件

```python
# 检查路径
config = DataConfig()
print(f"数据路径: {config.data_path}")

# 使用绝对路径
config = DataConfig(
    base_dir='/home/u2025210237/jupyterlab/rq_data_parquet',
    data_file='train_data_final.parquet'
)
```

### 问题: 内存不足

```python
config = DataConfig(
    batch_size=128,           # 减小批量
    chunk_size=50000,         # 分块加载
    use_dtype_optimization=True,
)
```

### 问题: 特征列为空

```python
# 手动指定特征列
config = DataConfig(
    feature_cols=['feature1', 'feature2', 'feature3']
)
```

## 📞 获取帮助

```bash
# 查看模块信息
python -c "from data_manager import get_version; print(get_version())"

# 运行测试
python test_module.py

# 查看示例
python examples.py
```

---

**最后更新**: 2025-11-19  
**版本**: 1.0.0  
**测试状态**: ✅ 所有测试通过
