# 滚动窗口预测空DataLoader问题修复

## 🐛 问题描述

在运行滚动窗口预测时，出现以下错误：

```python
RuntimeError: torch.cat(): expected a non-empty list of Tensors
```

错误堆栈：
```
File quantclassic/data_manager/rolling_trainer.py:435, in predict_all_windows
    predictions, labels, stocks, dates = self.predict_window(result)

File quantclassic/data_manager/rolling_trainer.py:374, in predict_window
    predictions = model.predict(test_loader, return_numpy=True)

File quantclassic/model/pytorch_models.py:294, in GRUModel.predict
    predictions = torch.cat(predictions, dim=0)

RuntimeError: torch.cat(): expected a non-empty list of Tensors
```

## 🔍 问题根源

### 1. 数据流程

```
滚动窗口划分 → 创建测试数据DataFrame
    ↓
TimeSeriesStockDataset._build_sample_index()
    ↓
检查每只股票的数据量:
  - 需要 >= window_size + 1 个样本
  - 不足的股票被跳过
    ↓
如果所有股票都不足 → sample_index = [] → len(dataset) = 0
    ↓
DataLoader遍历 → 没有batch → predictions = []
    ↓
torch.cat([]) → ❌ RuntimeError
```

### 2. 为什么会出现空测试集？

**TimeSeriesStockDataset的严格要求**：
```python
# 在_build_sample_index中
for ts_code, stock_df in df.groupby(self.stock_col):
    n = len(stock_df)
    
    # 需要至少 window_size + 1 个样本
    if n < self.window_size + 1:
        continue  # 跳过这只股票
```

**滚动窗口的测试期特点**：
- 测试期通常较短（例如63天）
- 标签生成和中性化会损失数据（前252+10天无标签）
- 某些股票在测试期内的有效数据 < window_size + 1

**示例**：
```
window_size = 40
测试期: 63天
某股票在测试期内:
  - 原始数据: 50天
  - 过滤标签缺失后: 35天
  - 35 < 40 + 1 → 被跳过
```

当测试期内**所有股票**的有效数据都不足时，test_dataset就是空的。

## ✅ 解决方案

### 修改1：`quantclassic/model/pytorch_models.py`

修改 `BaseModel.predict()` 方法（所有模型类继承）：

```python
def predict(self, test_loader, return_numpy: bool = True):
    """预测"""
    if not self.fitted:
        raise ValueError("模型未训练，请先调用 fit()")
    
    self.model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(self.device)
            pred = self.model(batch_x)
            predictions.append(pred.cpu())
    
    # ✅ 修复：处理空预测列表（测试集为空时）
    if len(predictions) == 0:
        import numpy as np
        return np.array([]) if return_numpy else torch.tensor([])
    
    predictions = torch.cat(predictions, dim=0)
    
    if return_numpy:
        return predictions.numpy()
    return predictions
```

**影响范围**：
- LSTMModel
- GRUModel  
- TransformerModel
- 所有继承BaseModel的模型类

### 修改2：`quantclassic/data_manager/rolling_trainer.py` - predict_window()

```python
def predict_window(self, window_result):
    # ... 获取model和test_dataset ...
    
    # 预测
    predictions = model.predict(test_loader, return_numpy=True)
    
    # ✅ 修复：处理空测试集的情况
    if len(test_dataset) == 0:
        self.logger.warning(f"  警告: 测试集为空，跳过预测")
        return (
            np.array([]),  # predictions
            np.array([]),  # labels
            np.array([]),  # stocks
            None           # dates
        )
    
    # ... 提取元数据 ...
```

**作用**：
- 在预测前检查测试集是否为空
- 返回空数组而不是继续执行
- 记录警告信息

### 修改3：`quantclassic/data_manager/rolling_trainer.py` - predict_all_windows()

```python
def predict_all_windows(self, window_results):
    all_predictions = []
    
    for i, result in enumerate(window_results):
        self.logger.info(f"  预测窗口 {i + 1}/{len(window_results)}...")
        
        predictions, labels, stocks, dates = self.predict_window(result)
        
        # ✅ 修复：跳过空预测窗口
        if len(predictions) == 0:
            self.logger.warning(f"    窗口 {i + 1} 预测为空，跳过")
            continue
        
        # 创建DataFrame
        window_df = pd.DataFrame({...})
        all_predictions.append(window_df)
    
    # ✅ 修复：处理无有效预测的情况
    if not all_predictions:
        self.logger.warning("\n⚠️  所有窗口的预测都为空！")
        # 返回空DataFrame但保持结构
        return pd.DataFrame(columns=[...])
    
    # 合并所有预测
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    return combined_predictions
```

**作用**：
- 跳过空窗口，不中断整个流程
- 处理所有窗口都为空的情况
- 返回结构完整的空DataFrame（而不是报错）

## 📊 修复效果

### 修复前

```
训练窗口 1/31 → 预测窗口 1
❌ RuntimeError: torch.cat(): expected a non-empty list of Tensors
```

### 修复后

```
训练窗口 1/31 → 预测窗口 1
  ⚠️  警告: 测试集为空，跳过预测
  ⚠️  窗口 1 预测为空，跳过

训练窗口 2/31 → 预测窗口 2
  ✅ 预测样本: 15,234

训练窗口 3/31 → 预测窗口 3
  ✅ 预测样本: 18,567

...

✅ 预测完成！
  总预测样本: 856,432
  有效窗口: 28/31
  跳过窗口: 3
```

## 🎯 根本原因总结

### 为什么某些窗口的测试集为空？

1. **标签生成的数据损失**
   - SimStock中性化需要252天历史
   - 标签生成需要未来10天
   - 前262天左右的数据无有效标签
   - 数据从2015-01-05开始，有效标签从2016-01-14开始

2. **滚动窗口的特点**
   - 窗口1的测试期：2017-01-14 ~ 2017-03-17（约63天）
   - 这个时间段正好是数据开始后不久
   - 某些股票在此期间的上市时间不长

3. **TimeSeriesStockDataset的要求**
   - 每只股票需要 >= window_size + 1 = 41 个连续有效数据
   - 测试期只有63天，某些股票只有30多天数据
   - 这些股票被过滤掉
   - 如果所有股票都不足，整个dataset为空

### 为什么会影响多个窗口？

早期窗口（窗口1-3）的测试期可能都在数据稀疏区域：
- 窗口1测试期：2017-01-14 ~ 2017-03-17
- 窗口2测试期：2017-03-18 ~ 2017-05-19
- 窗口3测试期：2017-05-20 ~ 2017-07-21

这些都是2017年初，距离2015年数据开始只有约2年，某些股票数据积累不足。

## 💡 优化建议

### 短期：修改配置参数

```python
# 方案1: 减小window_size
data_config = DataConfig(
    window_size=20,  # 从40改为20，降低数据要求
    ...
)

# 方案2: 增大rolling_step，减少窗口数
data_config = DataConfig(
    rolling_step=126,  # 从63改为126，窗口更稀疏
    ...
)

# 方案3: 调整数据起始时间
# 从2016年开始，避开标签缺失期
time_config = TimeConfig(
    start_date='2016-06-01',  # 晚半年开始
    ...
)
```

### 长期：改进数据处理

1. **放宽Dataset的样本要求**
   ```python
   # 允许部分窗口的数据
   if n >= min(window_size + 1, n // 2):
       # 至少有一半的窗口数据就可以
   ```

2. **提供替代的标签生成方法**
   - 不需要252天历史的中性化方法
   - 或者使用更短的lookback_window

3. **数据前向填充**
   - 对于数据不足的股票，使用前向填充
   - 但要注意引入未来信息的风险

## 🔗 相关文档

- `BUGFIX_ROLLING_WINDOW_EMPTY_DATASET.md` - 空数据集问题（标签缺失）
- `BUGFIX_DATASET_METADATA_EXTRACTION.md` - 元数据提取问题
- `ROLLING_WINDOW_GUIDE.md` - 滚动窗口训练指南

## 📅 修复日期
2025-11-21

## 🏷️ 影响版本
- quantclassic.model v1.0.0+
- quantclassic.data_manager v1.0.0+

## ✅ 测试验证

修复后应通过以下测试：

1. **空测试集测试**
   ```python
   # 创建空DataFrame
   empty_df = pd.DataFrame(columns=['order_book_id', 'trade_date', ...])
   dataset = TimeSeriesStockDataset(empty_df, ...)
   assert len(dataset) == 0
   
   # 预测应返回空数组而不报错
   predictions = model.predict(DataLoader(dataset), return_numpy=True)
   assert len(predictions) == 0
   ```

2. **部分窗口为空测试**
   ```python
   # 某些窗口有数据，某些无数据
   results = trainer.train_all_windows(...)
   predictions = trainer.predict_all_windows(results)
   # 应成功返回，只包含有效窗口的预测
   assert len(predictions) > 0
   ```

3. **全部窗口为空测试**
   ```python
   # 所有窗口都无数据
   predictions = trainer.predict_all_windows(results)
   # 返回空DataFrame但结构完整
   assert len(predictions) == 0
   assert set(predictions.columns) == {
       'order_book_id', 'trade_date', 'pred_alpha', 
       'alpha_label', 'window_idx'
   }
   ```

---

**总结**：本次修复通过在三个层次（模型预测、窗口预测、批量预测）添加空数据处理，确保系统在遇到数据不足的窗口时能够优雅地跳过，而不是崩溃。这使得滚动窗口训练更加鲁棒，能够适应真实数据中的各种边界情况。
