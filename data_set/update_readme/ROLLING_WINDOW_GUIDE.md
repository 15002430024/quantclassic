# 滚动窗口模型训练指南

## 📋 概述

quantclassic 现已支持**滚动窗口（Walk-Forward）模型训练**，这是量化金融中最严谨的时间序列验证方法。

## 🎯 什么是滚动窗口训练？

滚动窗口训练将历史数据划分为多个时间窗口，每个窗口独立训练模型并在下一窗口测试，模拟真实交易环境。

### 传统训练 vs 滚动窗口训练

```
传统训练 (单次训练):
|--------训练集--------|--验证--|--测试--|
         ↓
      训练1次模型
         ↓
    在测试集预测

滚动窗口训练 (Walk-Forward):
Window 1: |----训练----|--测试--|
                ↓           ↓
             训练模型1    预测1

Window 2:     |----训练----|--测试--|
                   ↓           ↓
                训练模型2    预测2

Window 3:         |----训练----|--测试--|
                       ↓           ↓
                    训练模型3    预测3

最终预测 = 合并(预测1, 预测2, 预测3, ...)
```

### 优势

✅ **无未来信息泄露** - 每个窗口只使用历史数据训练  
✅ **更真实的回测** - 模拟实际交易中的模型更新流程  
✅ **评估模型稳定性** - 观察模型在不同市场环境下的表现  
✅ **检测过拟合** - 多窗口平均表现更可靠  

## 🚀 快速开始

### 1. 配置滚动窗口策略

```python
from quantclassic.data_manager import DataManager, DataConfig

# 创建配置 - 使用 rolling 策略
data_config = DataConfig(
    base_dir='output',
    data_file='train_data_final_01.parquet',
    stock_col='order_book_id',
    time_col='trade_date',
    label_col='alpha_label',
    split_strategy='rolling',        # 关键：使用滚动窗口策略
    rolling_window_size=252,         # 训练窗口大小（252个交易日≈1年）
    rolling_step=63,                 # 滚动步长（63个交易日≈1季度）
    window_size=40,                  # 时序窗口大小
    batch_size=512
)
```

### 2. 创建滚动窗口训练器

```python
# 创建 DataManager 并运行数据准备
dm = DataManager(config=data_config)
loaders = dm.run_full_pipeline()

# 创建滚动窗口训练器
trainer = dm.create_rolling_window_trainer()

print(f"生成了 {trainer.n_windows} 个滚动窗口")
```

### 3. 训练所有窗口

```python
from quantclassic.model.pytorch_models import GRUModel
from quantclassic.model.model_config import GRUConfig

# 模型配置
gru_config = GRUConfig(
    d_feat=len(dm.feature_cols),
    hidden_size=64,
    num_layers=2,
    dropout=0.3,
    n_epochs=30,
    batch_size=512,
    learning_rate=0.001,
    early_stop=10,
    device='cuda'
)

# 训练所有窗口（独立训练）
results = trainer.train_all_windows(
    model_class=GRUModel,
    model_config=gru_config,
    save_dir='output/rolling_models',  # 保存每个窗口的模型
    incremental=False                   # False=独立训练，True=增量训练
)

print(f"训练完成 {len(results)} 个窗口")
```

### 4. 预测并合并结果

```python
# 对所有窗口进行预测
predictions = trainer.predict_all_windows(results)

print(f"预测样本数: {len(predictions):,}")
print(predictions.head())

# 保存预测结果
predictions.to_parquet('output/rolling_predictions.parquet')
```

### 5. 分析结果

```python
import numpy as np
from scipy.stats import pearsonr

# 计算 IC 指标
pred_values = predictions['pred_alpha'].values
label_values = predictions['alpha_label'].values

ic, _ = pearsonr(pred_values, label_values)
print(f"总体 IC: {ic:.4f}")

# 按窗口分析
for window_idx in predictions['window_idx'].unique():
    window_data = predictions[predictions['window_idx'] == window_idx]
    window_ic, _ = pearsonr(
        window_data['pred_alpha'].values,
        window_data['alpha_label'].values
    )
    print(f"窗口 {window_idx} IC: {window_ic:.4f}")

# 获取训练汇总统计
summary = trainer.get_summary()
print(f"\n训练汇总:")
print(f"  平均训练损失: {summary['avg_train_loss']:.6f}")
print(f"  平均验证损失: {summary['avg_val_loss']:.6f}")
print(f"  平均最佳Epoch: {summary['avg_best_epoch']:.1f}")
```

## 📊 完整示例

```python
import sys
sys.path.insert(0, '/home/u2025210237/jupyterlab')

from pathlib import Path
from quantclassic.data_manager import DataManager, DataConfig, RollingWindowTrainer
from quantclassic.model.pytorch_models import GRUModel
from quantclassic.model.model_config import GRUConfig

# ==================== 1. 配置 ====================
print("=" * 80)
print("🔄 滚动窗口模型训练")
print("=" * 80)

# 数据配置
data_config = DataConfig(
    base_dir='output',
    data_file='train_data_final_01.parquet',
    stock_col='order_book_id',
    time_col='trade_date',
    label_col='alpha_label',
    split_strategy='rolling',
    rolling_window_size=252,  # 1年训练窗口
    rolling_step=63,          # 1季度滚动步长
    window_size=40,
    batch_size=512,
)

# 模型配置
gru_config = GRUConfig(
    hidden_size=64,
    num_layers=2,
    dropout=0.3,
    n_epochs=30,
    batch_size=512,
    learning_rate=0.001,
    early_stop=10,
    device='cuda'
)

# ==================== 2. 数据准备 ====================
print("\n步骤 1: 数据准备")
dm = DataManager(config=data_config)
loaders = dm.run_full_pipeline()

gru_config.d_feat = len(dm.feature_cols)
print(f"✅ 特征维度: {gru_config.d_feat}")

# ==================== 3. 创建训练器 ====================
print("\n步骤 2: 创建滚动窗口训练器")
trainer = dm.create_rolling_window_trainer()

if trainer is None:
    raise ValueError("无法创建滚动窗口训练器，请检查配置")

# ==================== 4. 训练所有窗口 ====================
print("\n步骤 3: 训练所有窗口")
results = trainer.train_all_windows(
    model_class=GRUModel,
    model_config=gru_config,
    save_dir='output/rolling_models',
    val_ratio=0.2,
    incremental=False  # 独立训练
)

# ==================== 5. 预测 ====================
print("\n步骤 4: 预测所有窗口")
predictions = trainer.predict_all_windows(results)

# ==================== 6. 保存结果 ====================
print("\n步骤 5: 保存结果")
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

predictions.to_parquet('output/rolling_predictions.parquet')
print(f"✅ 预测结果已保存: output/rolling_predictions.parquet")

# ==================== 7. 分析 ====================
print("\n步骤 6: 结果分析")
summary = trainer.get_summary()

print("\n【训练汇总】")
print(f"  窗口数量: {summary['n_windows']}")
print(f"  平均训练损失: {summary['avg_train_loss']:.6f}")
print(f"  平均验证损失: {summary['avg_val_loss']:.6f}")
print(f"  平均最佳Epoch: {summary['avg_best_epoch']:.1f}")

print("\n【预测汇总】")
print(f"  总预测样本: {len(predictions):,}")
print(f"  时间范围: {predictions[data_config.time_col].min()} ~ {predictions[data_config.time_col].max()}")
print(f"  股票数量: {predictions[data_config.stock_col].nunique()}")

# 计算 IC
from scipy.stats import pearsonr
pred_values = predictions['pred_alpha'].values
label_values = predictions[data_config.label_col].values
ic, _ = pearsonr(pred_values, label_values)

print(f"\n【IC指标】")
print(f"  总体 Pearson IC: {ic:.4f}")

print("\n" + "=" * 80)
print("✅ 滚动窗口训练完成！")
print("=" * 80)
```

## 🔧 高级功能

### 增量训练（Incremental Training）

增量训练使用前一窗口的模型权重初始化下一窗口，可以加速训练并保持模型连续性。

```python
# 增量训练模式
results = trainer.train_all_windows(
    model_class=GRUModel,
    model_config=gru_config,
    save_dir='output/rolling_models',
    incremental=True  # 启用增量训练
)
```

**对比：**
- **独立训练** (`incremental=False`): 每个窗口从随机初始化开始训练，更鲁棒但训练时间长
- **增量训练** (`incremental=True`): 使用前一窗口模型初始化，训练更快但可能累积偏差

### 单个窗口训练

如果需要调试或单独训练某个窗口：

```python
# 训练第3个窗口
result = trainer.train_window(
    window_idx=2,  # 索引从0开始
    model_class=GRUModel,
    model_config=gru_config,
    save_path='output/window_3_model.pth',
    val_ratio=0.2
)

print(f"最佳Epoch: {result['best_epoch']}")
print(f"验证损失: {result['val_loss']:.6f}")
```

### 自定义数据集创建

如果需要更灵活的数据集创建逻辑：

```python
# 为特定窗口创建数据集
train_ds, val_ds, test_ds = trainer.create_datasets_for_window(
    window_idx=0,
    val_ratio=0.2
)

print(f"训练集: {len(train_ds)} 样本")
print(f"验证集: {len(val_ds)} 样本")
print(f"测试集: {len(test_ds)} 样本")
```

## 📈 参数调优建议

### 窗口大小（rolling_window_size）

```python
# 短窗口（126天 ≈ 6个月）
rolling_window_size=126  # 适合快速变化的市场

# 中等窗口（252天 ≈ 1年）
rolling_window_size=252  # 推荐，平衡数据量和时效性

# 长窗口（504天 ≈ 2年）
rolling_window_size=504  # 适合稳定策略，需要更多历史数据
```

### 滚动步长（rolling_step）

```python
# 重叠窗口（步长 < 窗口大小）
rolling_step=63   # 1季度步长，窗口重叠75%

# 连续窗口（步长 = 窗口大小）
rolling_step=252  # 窗口不重叠，训练效率高但窗口数少

# 跳跃窗口（步长 > 窗口大小）
rolling_step=378  # 有间隙，适合长期策略
```

### 训练策略选择

| 场景 | 推荐策略 | 参数设置 |
|------|---------|---------|
| 快速验证 | 合并窗口 | 当前默认（80%训练/20%测试） |
| 严格回测 | 独立训练 | `incremental=False` |
| 生产部署 | 增量训练 | `incremental=True` |
| 模型对比 | 独立训练 | `incremental=False` |

## 🎯 最佳实践

### 1. 选择合适的窗口参数

```python
# 对于日频数据的推荐配置
data_config = DataConfig(
    split_strategy='rolling',
    rolling_window_size=252,  # 1年训练窗口
    rolling_step=63,          # 1季度滚动
    window_size=40,           # 40日时序窗口
    batch_size=512
)
```

### 2. 监控每个窗口的表现

```python
# 记录每个窗口的IC
window_ics = []
for window_idx in predictions['window_idx'].unique():
    window_data = predictions[predictions['window_idx'] == window_idx]
    ic, _ = pearsonr(
        window_data['pred_alpha'].values,
        window_data['alpha_label'].values
    )
    window_ics.append(ic)
    print(f"窗口 {window_idx}: IC={ic:.4f}")

# 分析IC稳定性
print(f"\nIC稳定性:")
print(f"  平均IC: {np.mean(window_ics):.4f}")
print(f"  IC标准差: {np.std(window_ics):.4f}")
print(f"  IC胜率: {np.mean(np.array(window_ics) > 0):.2%}")
```

### 3. 保存和加载训练结果

```python
import pickle

# 保存训练结果
with open('output/rolling_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# 加载训练结果
with open('output/rolling_results.pkl', 'rb') as f:
    loaded_results = pickle.load(f)

# 使用加载的结果进行预测
predictions = trainer.predict_all_windows(loaded_results)
```

### 4. 错误处理

```python
try:
    results = trainer.train_all_windows(
        model_class=GRUModel,
        model_config=gru_config,
        save_dir='output/rolling_models'
    )
except Exception as e:
    print(f"训练失败: {e}")
    # 可以单独训练失败的窗口
    for i in range(trainer.n_windows):
        try:
            result = trainer.train_window(
                window_idx=i,
                model_class=GRUModel,
                model_config=gru_config,
                save_path=f'output/window_{i+1}_model.pth'
            )
        except Exception as window_error:
            print(f"窗口 {i+1} 训练失败: {window_error}")
            continue
```

## 📚 API 参考

### RollingWindowTrainer

```python
class RollingWindowTrainer:
    """滚动窗口训练器"""
    
    def __init__(
        self,
        windows: List[Tuple[pd.DataFrame, pd.DataFrame]],
        config: DataConfig,
        feature_cols: List[str],
        logger: Optional[logging.Logger] = None
    )
    
    def train_all_windows(
        self,
        model_class: type,
        model_config: Any,
        save_dir: Optional[str] = None,
        val_ratio: float = 0.2,
        incremental: bool = False
    ) -> List[Dict[str, Any]]
    
    def predict_all_windows(
        self,
        window_results: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame
    
    def train_window(
        self,
        window_idx: int,
        model_class: type,
        model_config: Any,
        save_path: Optional[str] = None,
        val_ratio: float = 0.2,
        init_model_path: Optional[str] = None
    ) -> Dict[str, Any]
    
    def get_summary(self) -> Dict[str, Any]
```

### DataManager 新增方法

```python
class DataManager:
    """数据管理器"""
    
    def create_rolling_window_trainer(self) -> Optional[RollingWindowTrainer]:
        """创建滚动窗口训练器（仅rolling策略可用）"""
```

## ⚠️ 注意事项

1. **内存占用**: 滚动窗口训练会保存所有窗口的模型，确保有足够磁盘空间
2. **训练时间**: 完全独立训练 N 个窗口需要约 N 倍的时间
3. **数据质量**: 确保每个窗口都有足够的样本（建议 >1000）
4. **设备管理**: 使用 GPU 训练时注意显存占用

## 🔗 相关文档

- [DataManager 使用指南](./USAGE_GUIDE.md)
- [配置系统文档](./README.md)
- [模型配置指南](../model/README.md)

## 💡 下一步

完成滚动窗口训练后，你可以：

1. **因子分析**: 使用 Factorsystem 分析因子表现
2. **回测验证**: 使用 backtest_system 进行完整回测
3. **策略优化**: 调整模型参数和窗口参数优化策略
4. **生产部署**: 使用增量训练模式部署到生产环境

---

**版本**: v1.0.0  
**更新时间**: 2025-01-21  
**作者**: quantclassic team
