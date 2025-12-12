# DataManager - 工程化数据管理模块

## 📖 概述

DataManager 是一个完整的工程化数据管理解决方案，专为量化交易和机器学习项目设计。它提供了从数据加载、验证、特征工程到数据集创建的完整流程，具有高度的可扩展性和可维护性。

## ✨ 核心特性

### 🎯 完整的数据处理流水线
- **自动化流程**: 一键完成数据加载、验证、特征工程和数据集创建
- **模块化设计**: 每个组件可独立使用，也可组合使用
- **灵活配置**: 支持YAML配置文件和代码配置

### 🔧 强大的功能组件
- **DataConfig**: 统一的配置管理，支持多种预设模板
- **DataLoader**: 多格式支持（Parquet/CSV/HDF5），内存优化
- **FeatureEngineer**: 自动特征选择、过滤和统计分析
- **DataSplitter**: 多种划分策略（时间序列、分层、滚动窗口）
- **DataValidator**: 全面的数据质量检查和报告
- **DatasetFactory**: PyTorch Dataset和DataLoader创建

### 🚀 性能优化
- **内存优化**: 自动数据类型优化，减少内存占用
- **智能缓存**: 多级缓存机制，避免重复计算
- **批量处理**: 支持大数据集的分块加载

### 📊 数据质量保障
- **自动验证**: 缺失值、异常值、时序连续性检查
- **质量报告**: 生成详细的数据质量报告
- **问题诊断**: 自动识别常见数据问题

## 📁 模块结构

```
data_manager/
├── __init__.py           # 模块初始化
├── config.py             # 配置管理（DataConfig）
├── loader.py             # 数据加载器（DataLoaderEngine）
├── feature_engineer.py   # 特征工程师（FeatureEngineer）
├── splitter.py           # 数据划分器（DataSplitter系列）
├── validator.py          # 数据验证器（DataValidator）
├── factory.py            # 数据集工厂（DatasetFactory）
├── manager.py            # 主控类（DataManager）
├── examples.py           # 使用示例
└── README.md             # 文档（本文件）
```

✅ 已完成的配置类：

✅ LabelConfig - label_generator.py（标签生成配置）

✅ ProcessingStep - preprocess_config.py（处理步骤）

✅ NeutralizeConfig - preprocess_config.py（中性化配置）

✅ PreprocessConfig - preprocess_config.py（预处理总配置）

✅ DataConfig - config.py（数据管理配置）

✅ BaseModelConfig - model_config.py（基础模型配置）

✅ LSTMConfig - model_config.py（LSTM 模型配置）

✅ GRUConfig - model_config.py（GRU 模型配置）

✅ TimeConfig - config_manager.py（时间配置）

✅ DataSourceConfig - config_manager.py（数据源配置）

✅ UniverseConfig - config_manager.py（股票池配置）

✅ BacktestConfig - backtest_config.py（回测配置）

✅ RecorderConfig - workflow_config.py（记录器配置）

✅ CheckpointConfig - workflow_config.py（检查点配置）

✅ ArtifactConfig - workflow_config.py（工件配置）

✅ WorkflowConfig - workflow_config.py（工作流配置）



## 🚀 快速开始

### 1. 最简单的使用方式

```python
from data_manager import DataManager, DataConfig

# 创建配置
config = DataConfig(
    base_dir='rq_data_parquet',
    data_file='train_data_final.parquet'
)

# 创建管理器并运行完整流水线
manager = DataManager(config)
loaders = manager.run_full_pipeline()

# 使用数据加载器训练模型
for batch_x, batch_y in loaders.train:
    # 训练代码
    pass
```

### 2. 逐步执行流水线

```python
manager = DataManager(config)

# 步骤1: 加载数据
raw_data = manager.load_raw_data()

# 步骤2: 验证数据质量
report = manager.validate_data_quality()

# 步骤3: 特征工程
feature_cols = manager.preprocess_features(auto_filter=True)

# 步骤4-5: 创建数据集
datasets = manager.create_datasets()

# 步骤6: 获取数据加载器
loaders = manager.get_dataloaders(batch_size=256)
```

## 📝 详细使用指南

### 配置管理

#### 使用默认配置
```python
from data_manager import DataConfig

config = DataConfig()  # 使用默认参数
```

#### 自定义配置
```python
config = DataConfig(
    # 数据路径
    base_dir='rq_data_parquet',
    data_file='train_data_final.parquet',
    
    # 特征参数
    window_size=60,         # 时间窗口
    label_col='y_processed',
    
    # 数据划分
    split_strategy='time_series',  # 或 'stratified', 'rolling', 'random'
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    
    # 数据加载
    batch_size=512,
    num_workers=4,
    
    # 优化选项
    use_dtype_optimization=True,
    enable_cache=True,
    enable_validation=True,
)
```

#### 使用配置模板
```python
from data_manager import ConfigTemplates

# 快速测试配置
config = ConfigTemplates.quick_test()

# 生产环境配置
config = ConfigTemplates.production()

# 回测配置
config = ConfigTemplates.backtest()
```

#### YAML配置文件
```python
# 保存配置
config.to_yaml('my_config.yaml')

# 加载配置
config = DataConfig.from_yaml('my_config.yaml')
```

### 数据划分策略

#### 1. 时间序列划分（推荐）
```python
config = DataConfig(
    split_strategy='time_series',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

#### 2. 分层股票划分
```python
config = DataConfig(
    split_strategy='stratified',  # 确保每只股票在各数据集中都有样本
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

#### 3. 滚动窗口划分（用于回测）
```python
config = DataConfig(
    split_strategy='rolling',
    rolling_window_size=252,  # 训练窗口：252个交易日（约1年）
    rolling_step=63           # 滚动步长：63个交易日（约1季度）
)
```

### 特征工程

#### 自动特征选择
```python
manager = DataManager(config)
raw_data = manager.load_raw_data()

# 自动选择数值型特征（排除指定列）
features = manager.feature_engineer.select_features(raw_data)
```

#### 特征过滤
```python
# 过滤低质量特征
filtered_features = manager.feature_engineer.filter_features(
    raw_data,
    min_variance=1e-5,        # 最小方差
    max_missing_ratio=0.3,    # 最大缺失率
    max_correlation=0.95      # 最大相关性
)
```

#### 特征统计
```python
# 计算特征统计信息
stats = manager.feature_engineer.compute_feature_stats(raw_data)

# 保存特征信息
manager.feature_engineer.save_feature_info('output/')
```

### 数据验证

```python
# 执行数据质量验证
report = manager.validate_data_quality()

# 查看验证结果
if report.is_valid:
    print("数据验证通过")
else:
    print(f"发现 {len(report.errors)} 个错误")
    for error in report.errors:
        print(f"  - {error}")

# 查看警告
for warning in report.warnings:
    print(f"  ⚠️ {warning}")

# 打印完整报告
report.print_report()
```

### 访问处理后的数据

```python
# 运行完整流水线
loaders = manager.run_full_pipeline()

# 访问原始数据
raw_data = manager.raw_data

# 访问特征列
feature_cols = manager.feature_cols

# 访问划分后的数据
train_df, val_df, test_df = manager.split_data

# 访问数据集
datasets = manager.datasets
print(f"训练样本数: {len(datasets.train)}")
print(f"验证样本数: {len(datasets.val)}")
print(f"测试样本数: {len(datasets.test)}")

# 访问元数据
metadata = datasets.metadata
print(f"特征数量: {metadata['num_features']}")
print(f"窗口大小: {metadata['window_size']}")
```

### 与模型训练集成

```python
import torch
import torch.nn as nn

# 准备数据
manager = DataManager(config)
loaders = manager.run_full_pipeline()

# 创建模型（示例）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = YourModel(...).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练循环
for epoch in range(num_epochs):
    # 训练
    model.train()
    for batch_x, batch_y in loaders.train:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # 前向传播
        # pred = model(batch_x)
        # loss = criterion(pred, batch_y)
        
        # 反向传播
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
    
    # 验证
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loaders.val:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # 验证代码
            # val_pred = model(batch_x)
            # val_loss = criterion(val_pred, batch_y)
```

### 状态保存和加载

```python
# 保存管理器状态
manager.save_state('cache/manager_state.pkl')

# 在另一个会话中加载
new_manager = DataManager()
new_manager.load_state('cache/manager_state.pkl')

# 继续使用
loaders = new_manager.get_dataloaders()
```

## 📊 预期输出示例

### 运行完整流水线的输出

```
================================================================================
🚀 开始完整数据处理流水线
================================================================================

================================================================================
步骤 1/5: 加载原始数据
================================================================================
📁 加载数据: rq_data_parquet/train_data_final.parquet
🔧 优化数据类型...
   内存优化: 1250.45MB → 625.23MB (减少 50.0%)
✅ 数据加载完成: 1,234,567 行, 156 列

================================================================================
📊 数据摘要
================================================================================
形状: 1,234,567 行 × 156 列
内存占用: 625.23 MB
股票数量: 4,500
时间范围: 2015-01-01 ~ 2023-12-31

数据类型分布:
  float32: 150 列
  int32: 3 列
  category: 3 列

缺失值 (前10列):
  feature_42: 12345 (1.00%)
  feature_87: 8901 (0.72%)
================================================================================

================================================================================
步骤 2/5: 验证数据质量
================================================================================
🔍 开始数据验证...
✅ 验证完成: 0 错误, 2 警告

================================================================================
📋 数据验证报告
================================================================================

状态: ✅ 通过

警告 (2):
  1. ⚠️  列 'feature_42' 缺失值过高: 1.00% (12345/1234567)
  2. ⚠️  10 只股票样本数少于 60

统计信息:
  total_rows: 1234567
  total_columns: 156
  num_stocks: 4500
  date_range: ('2015-01-01', '2023-12-31')
  num_features: 150
  numeric_features: 150
================================================================================

================================================================================
步骤 3/5: 特征工程
================================================================================
🔍 自动检测特征列...
✅ 自动选择特征列: 150 列
📊 计算特征统计信息...
🔧 过滤低质量特征...
   移除 15 个特征:
   - 低方差: 5
   - 高缺失: 3
   - 高相关: 7
✅ 保留 135 个特征
💾 特征信息已保存到: output/

================================================================================
步骤 4/5: 数据划分
================================================================================
📅 时间序列划分...
   训练集: 864,197 行 (2015-01-01 ~ 2021-12-31)
   验证集: 185,185 行 (2022-01-01 ~ 2022-12-31)
   测试集: 185,185 行 (2023-01-01 ~ 2023-12-31)

================================================================================
步骤 5/5: 创建数据集
================================================================================
🏭 创建数据集...
   训练集: 820,157 样本
   验证集: 175,145 样本
   测试集: 175,145 样本

================================================================================
✅ 完整数据处理流水线完成
================================================================================

================================================================================
📊 数据处理摘要
================================================================================
原始数据: 1,234,567 行
特征数量: 135

数据集:
  训练集: 820,157 样本
  验证集: 175,145 样本
  测试集: 175,145 样本

配置:
  窗口大小: 40
  批量大小: 256
  划分策略: time_series
================================================================================
```

### 数据加载器使用输出

```python
# 遍历训练集
for i, (batch_x, batch_y) in enumerate(loaders.train):
    print(f"批次 {i+1}: X={batch_x.shape}, Y={batch_y.shape}")
    if i >= 2:
        break

# 输出:
# 批次 1: X=torch.Size([256, 40, 135]), Y=torch.Size([256])
# 批次 2: X=torch.Size([256, 40, 135]), Y=torch.Size([256])
# 批次 3: X=torch.Size([256, 40, 135]), Y=torch.Size([256])
```

## 🔧 高级功能

### 自定义数据集

```python
from data_manager.factory import TimeSeriesStockDataset

# 创建自定义数据集
custom_dataset = TimeSeriesStockDataset(
    df=my_dataframe,
    feature_cols=my_features,
    label_col='my_label',
    window_size=60,
    stock_col='ts_code',
    time_col='trade_date'
)
```

### 推理数据集（无标签）

```python
from data_manager.factory import InferenceDataset

# 创建推理数据集
inference_dataset = manager.factory.create_inference_dataset(
    df=test_data,
    feature_cols=feature_cols
)

# 获取预测样本
for sample in inference_dataset:
    # sample 形状: [window_size, n_features]
    prediction = model(sample)
```

### 窗口级数据变换（研报标准）🆕

在 `Dataset.__getitem__` 中对每个窗口实时进行价格对数变换和成交量标准化。这确保了每个窗口使用自己的基准点（窗口末端的收盘价），而不是全局处理。

```python
# 启用窗口级变换
config = DataConfig(
    # 启用窗口变换
    enable_window_transform=True,
    
    # 价格对数变换: log(price / close_t)
    window_price_log=True,
    price_cols=['open', 'high', 'low', 'close', 'vwap'],
    close_col='close',
    
    # 成交量标准化: volume / mean(volume_in_window)
    window_volume_norm=True,
    volume_cols=['vol', 'amount']
)
```

**变换公式说明**：

| 变换类型 | 公式 | 效果 |
|---------|------|------|
| 价格对数变换 | `log(price_{t-i} / close_t)` | close_t = 0, 其他为相对偏差 |
| 成交量标准化 | `vol_{t-i} / mean(vol_window)` | 均值附近 ≈ 1.0 |

**为什么在 Dataset 阶段做？**
- 每个窗口有不同的基准点 `close_t`
- 全局预处理无法实现"窗口相对"变换
- 避免数据量爆炸（同一天数据在不同窗口中数值不同）

### 自定义验证规则

```python
# 修改验证参数
config = DataConfig(
    max_na_ratio=0.5,              # 允许50%缺失
    min_samples_per_stock=100,     # 每只股票至少100样本
    detect_outliers=True,
    outlier_std_threshold=5.0      # 5倍标准差为异常值
)
```

## 🤝 与其他模块的协作

### 与 data_loader 模块协作
- DataManager 使用自己的数据加载逻辑，**不依赖** data_loader
- data_loader 专注于原始数据获取和存储
- DataManager 专注于模型训练的数据管理

### 与 data_processor 模块协作
- data_processor 处理原始数据的清洗和特征计算
- DataManager 使用已处理的数据进行训练准备
- 推荐流程: data_loader → data_processor → DataManager

### 与模型训练模块协作
- DataManager 提供标准的 PyTorch DataLoader
- 支持分布式训练的数据分片
- 提供一致的数据接口

### 与回测框架协作
- 使用 `InferenceDataset` 进行因子生成
- 支持滚动窗口回测
- 提供时间点数据切片

## 📈 性能优化建议

### 内存优化
```python
config = DataConfig(
    use_dtype_optimization=True,  # 使用float32而非float64
    chunk_size=100000,            # 分块加载大文件
)
```

### 计算优化
```python
config = DataConfig(
    num_workers=4,       # 多进程数据加载
    pin_memory=True,     # 使用锁页内存（GPU训练）
    enable_cache=True,   # 启用缓存
)
```

### 大数据集处理
```python
# 对于超大数据集
config = DataConfig(
    chunk_size=50000,              # 分块加载
    use_dtype_optimization=True,   # 类型优化
    cache_feature_engineering=False,  # 不缓存中间结果
)
```

## ⚠️ 注意事项

1. **时序数据**: 建议使用 `time_series` 或 `stratified` 划分策略，避免使用 `random`
2. **数据泄漏**: 划分数据前不要进行跨时间的标准化
3. **内存管理**: 处理大数据集时注意监控内存使用
4. **缓存清理**: 长时间运行时定期清理缓存

## 🐛 故障排除

### 问题: 内存不足
```python
# 解决方案: 减小批量大小或使用分块加载
config = DataConfig(
    batch_size=128,      # 减小批量
    chunk_size=50000,    # 分块加载
)
```

### 问题: 数据验证失败
```python
# 解决方案: 调整验证参数或跳过验证
config = DataConfig(
    enable_validation=False,  # 跳过验证
    # 或调整参数
    max_na_ratio=0.5,
    min_samples_per_stock=30,
)
```

### 问题: 特征过滤太激进
```python
# 解决方案: 放宽过滤条件
filtered = manager.feature_engineer.filter_features(
    df,
    min_variance=1e-8,       # 降低方差阈值
    max_missing_ratio=0.6,   # 提高缺失率容忍
    max_correlation=0.99     # 提高相关性阈值
)
```

## 📚 更多示例

详细示例请查看 `examples.py` 文件，包含：
- 快速开始
- 自定义配置
- 逐步执行流水线
- 不同划分策略
- 配置模板使用
- 状态保存和加载
- 与训练集成
- 特征过滤
- 数据访问
- YAML配置

运行示例：
```bash
python examples.py
```

## 📄 许可证

本模块为内部项目的一部分，遵循项目整体许可协议。

## 👥 贡献者

quantclassic 团队

---

**版本**: 1.0.0  
**最后更新**: 2025-11-19
