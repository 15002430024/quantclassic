# 滚动窗口训练功能实现总结

## 📋 实现内容

### 1. 核心功能实现

✅ **RollingWindowTrainer 类** (`rolling_trainer.py`)
- 完整的滚动窗口训练和预测逻辑
- 支持独立训练和增量训练两种模式
- 自动管理模型保存和加载
- 提供详细的训练日志和统计信息

✅ **DataManager 集成** (`manager.py`)
- 新增 `create_rolling_window_trainer()` 方法
- 自动检测 rolling 策略并创建训练器
- 无缝集成现有数据流程

✅ **模块导出** (`__init__.py`)
- 导出 RollingWindowTrainer 类
- 更新 __all__ 列表

### 2. 文档和示例

✅ **完整使用指南** (`ROLLING_WINDOW_GUIDE.md`)
- 详细的功能说明和原理解释
- 完整的 API 参考文档
- 参数调优建议和最佳实践
- 多个使用示例和代码片段

✅ **快速示例脚本** (`example_rolling_training.py`)
- 可直接运行的完整示例
- 包含确认提示和错误处理
- 展示完整的训练和分析流程

✅ **Notebook 集成** (`vae.ipynb`)
- 新增滚动窗口训练 cell
- 提供对比说明和使用指导
- 包含交互式确认机制

✅ **更新现有文档** (`USAGE_GUIDE.md`)
- 添加新功能说明
- 提供快速使用示例

## 🎯 核心特性

### RollingWindowTrainer 功能列表

1. **训练管理**
   - `train_all_windows()` - 训练所有窗口
   - `train_window()` - 训练单个窗口
   - 支持独立训练（每个窗口独立初始化）
   - 支持增量训练（使用前一窗口权重初始化）

2. **预测功能**
   - `predict_all_windows()` - 预测所有窗口并合并结果
   - `predict_window()` - 预测单个窗口
   - 自动处理元数据（股票代码、日期）

3. **数据管理**
   - `create_datasets_for_window()` - 为指定窗口创建数据集
   - 自动处理训练/验证集划分
   - 支持自定义验证集比例

4. **统计分析**
   - `get_summary()` - 获取训练和预测汇总统计
   - 计算平均损失、最佳轮数等指标
   - 提供标准差评估模型稳定性

## 📊 使用流程

```
1. 配置 DataConfig
   ↓ (split_strategy='rolling')
   
2. 创建 DataManager
   ↓ (run_full_pipeline)
   
3. 创建 RollingWindowTrainer
   ↓ (create_rolling_window_trainer)
   
4. 训练所有窗口
   ↓ (train_all_windows)
   
5. 预测并合并结果
   ↓ (predict_all_windows)
   
6. 分析 IC 和模型稳定性
```

## 💡 典型使用场景

### 场景 1: 策略验证

```python
# 使用滚动窗口评估策略在不同市场环境下的表现
trainer = dm.create_rolling_window_trainer()
results = trainer.train_all_windows(
    model_class=GRUModel,
    model_config=gru_config,
    incremental=False  # 独立训练，更严格
)
predictions = trainer.predict_all_windows(results)

# 分析每个窗口的IC
for window_idx in predictions['window_idx'].unique():
    window_data = predictions[predictions['window_idx'] == window_idx]
    ic = calculate_ic(window_data)
    print(f"窗口 {window_idx}: IC={ic:.4f}")
```

### 场景 2: 生产部署模拟

```python
# 使用增量训练模拟实际交易中的模型更新
trainer = dm.create_rolling_window_trainer()
results = trainer.train_all_windows(
    model_class=GRUModel,
    model_config=gru_config,
    incremental=True,  # 增量训练，保持连续性
    save_dir='output/rolling_models'
)
```

### 场景 3: 模型对比

```python
# 对比不同模型在滚动窗口上的稳定性
gru_results = trainer.train_all_windows(model_class=GRUModel, ...)
lstm_results = trainer.train_all_windows(model_class=LSTMModel, ...)

gru_summary = trainer.get_summary()
lstm_summary = trainer.get_summary()

# 比较IC稳定性
print(f"GRU IC标准差: {gru_summary['std_val_loss']:.4f}")
print(f"LSTM IC标准差: {lstm_summary['std_val_loss']:.4f}")
```

## 🔧 技术实现细节

### 窗口管理

- 使用 `DataManager._rolling_windows` 存储所有窗口数据
- 每个窗口包含 `(train_df, test_df)` 元组
- 自动处理窗口索引和元数据

### 模型保存策略

- 每个窗口模型保存为独立文件: `window_N_model.pth`
- 包含完整训练历史和配置信息
- 支持加载任意窗口模型进行分析

### 预测合并

- 自动添加 `window_idx` 列标识来源窗口
- 保留原始元数据（股票代码、日期）
- 使用 `pd.concat` 高效合并大量预测结果

### 内存管理

- 逐窗口训练，避免内存占用过高
- 训练完成后释放中间数据
- 可选择只保存必要的模型权重

## ⚙️ 配置参数

### DataConfig 关键参数

```python
DataConfig(
    split_strategy='rolling',      # 必须设置为 'rolling'
    rolling_window_size=252,       # 训练窗口大小（天）
    rolling_step=63,               # 滚动步长（天）
    window_size=40,                # 时序窗口大小
    batch_size=512                 # 批次大小
)
```

### RollingWindowTrainer 参数

```python
trainer.train_all_windows(
    model_class=GRUModel,          # 模型类
    model_config=gru_config,       # 模型配置
    save_dir='output/models',      # 保存目录
    val_ratio=0.2,                 # 验证集比例
    incremental=False              # 是否增量训练
)
```

## 📈 性能考虑

### 训练时间估算

假设单窗口训练时间为 T：
- **独立训练**: 总时间 = N × T（N 为窗口数）
- **增量训练**: 总时间 ≈ N × 0.5T（利用前一模型初始化）
- **合并窗口**: 总时间 = T（当前默认方式）

### 内存占用

- 单窗口内存占用: 与数据集大小和模型大小相关
- 同时保存 N 个模型: 约 N × 模型大小
- 建议: 使用 SSD 存储模型，及时清理旧模型

### GPU 利用率

- 逐窗口训练可充分利用 GPU
- 不支持多窗口并行训练（避免显存溢出）
- 可通过减少 batch_size 适应小显存

## 🎓 进阶用法

### 自定义窗口划分

```python
# 手动创建窗口
from quantclassic.data_manager import RollingWindowTrainer

custom_windows = [
    (train_df_1, test_df_1),
    (train_df_2, test_df_2),
    # ...
]

trainer = RollingWindowTrainer(
    windows=custom_windows,
    config=data_config,
    feature_cols=feature_cols
)
```

### 单窗口调试

```python
# 只训练第5个窗口
result = trainer.train_window(
    window_idx=4,  # 索引从0开始
    model_class=GRUModel,
    model_config=gru_config,
    save_path='output/debug_model.pth'
)

print(f"训练损失: {result['train_loss']:.6f}")
print(f"验证损失: {result['val_loss']:.6f}")
```

### 加载已训练模型进行预测

```python
import pickle

# 保存训练结果
with open('output/rolling_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# 后续加载并预测
with open('output/rolling_results.pkl', 'rb') as f:
    loaded_results = pickle.load(f)

predictions = trainer.predict_all_windows(loaded_results)
```

## 🐛 故障排查

### 问题 1: 无法创建训练器

**错误**: `create_rolling_window_trainer()` 返回 `None`

**原因**: 配置中未使用 `split_strategy='rolling'`

**解决**:
```python
data_config.split_strategy = 'rolling'
dm = DataManager(config=data_config)
dm.run_full_pipeline()
trainer = dm.create_rolling_window_trainer()
```

### 问题 2: 内存不足

**错误**: `RuntimeError: CUDA out of memory`

**解决**:
1. 减小 batch_size
2. 减少模型参数（hidden_size, num_layers）
3. 使用更小的窗口大小
4. 清理不需要的模型文件

### 问题 3: 训练时间过长

**解决**:
1. 减少 n_epochs
2. 使用更激进的 early_stop
3. 考虑使用增量训练（incremental=True）
4. 减少窗口数量（增大 rolling_step）

## 📚 相关资源

- **使用指南**: `USAGE_GUIDE.md`
- **滚动窗口指南**: `ROLLING_WINDOW_GUIDE.md`
- **配置文档**: `README.md`
- **快速示例**: `example_rolling_training.py`
- **Notebook 示例**: `vae.ipynb` Cell 8

## 🔄 版本历史

- **v1.0.0** (2025-01-21)
  - ✅ 初始实现
  - ✅ 支持独立训练和增量训练
  - ✅ 完整文档和示例
  - ✅ Notebook 集成

## 👥 贡献

欢迎提出改进建议和功能请求！

---

**维护者**: quantclassic team  
**更新时间**: 2025-01-21
