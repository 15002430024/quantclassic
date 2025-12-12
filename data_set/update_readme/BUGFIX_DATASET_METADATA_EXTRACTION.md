# TimeSeriesStockDataset 元数据提取问题修复

## 🐛 问题描述

在运行滚动窗口预测时，出现以下错误：

```python
AttributeError: 'TimeSeriesStockDataset' object has no attribute 'labels'
```

错误发生在 `rolling_trainer.py` 的 `predict_window` 方法中：

```python
labels = test_dataset.labels  # ❌ TimeSeriesStockDataset没有这个属性
stocks = test_dataset.stocks  # ❌ 也没有这个属性
dates = test_dataset.dates    # ❌ 也没有这个属性
```

## 🔍 问题根源

### 1. TimeSeriesStockDataset 的数据结构

`TimeSeriesStockDataset` 为了优化性能，将数据存储在内部字典中，而不是直接暴露为属性：

```python
class TimeSeriesStockDataset(Dataset):
    def _build_sample_index(self, df):
        self.stock_data = {}  # 股票数据字典
        self.sample_index = []  # 样本索引列表
        
        for stock_idx, stock_df in enumerate(...):
            self.stock_data[stock_idx] = {
                'ts_code': ts_code,
                'features': features,
                'labels': labels,
                # ❌ 缺少日期信息！
                'n': n
            }
```

**问题1**：数据存储在嵌套字典中，无法直接通过 `dataset.labels` 访问  
**问题2**：原实现缺少日期信息存储

### 2. rolling_trainer 的错误假设

`predict_window` 方法假设数据集有 `labels`, `stocks`, `dates` 属性：

```python
def predict_window(self, window_result):
    test_dataset = window_result['test_dataset']
    
    # ❌ 错误假设：dataset有这些属性
    labels = test_dataset.labels
    stocks = test_dataset.stocks
    dates = test_dataset.dates
```

这种假设对某些数据集类型可能成立（如简单的Dataset），但对于优化过的 `TimeSeriesStockDataset` 不成立。

## ✅ 解决方案

### 修改1：`factory.py` - 添加日期信息存储

在 `TimeSeriesStockDataset._build_sample_index()` 中添加日期字段：

```python
# 提取特征和标签
features = stock_df[self.feature_cols].values.astype(np.float32)
labels = stock_df[self.label_col].values.astype(np.float32)
dates = stock_df[self.time_col].values  # ✅ 添加日期信息

# 存储股票数据
self.stock_data[stock_idx] = {
    'ts_code': ts_code,
    'features': features,
    'labels': labels,
    'dates': dates,  # ✅ 存储日期
    'n': n
}
```

### 修改2：`rolling_trainer.py` - 正确提取元数据

修改 `predict_window()` 方法，从内部数据结构中提取信息：

```python
def predict_window(self, window_result):
    # ... 预测代码 ...
    
    # ✅ 从TimeSeriesStockDataset中提取标签和元数据
    labels = []
    stocks = []
    dates = []
    
    for idx in range(len(test_dataset)):
        stock_idx, time_idx = test_dataset.sample_index[idx]
        stock_info = test_dataset.stock_data[stock_idx]
        
        # 标签是t+1时刻的值（因为预测的是未来收益）
        labels.append(stock_info['labels'][time_idx + 1])
        stocks.append(stock_info['ts_code'])
        
        # 提取日期信息
        if 'dates' in stock_info:
            dates.append(stock_info['dates'][time_idx + 1])
        else:
            dates.append(None)
    
    labels = np.array(labels)
    stocks = np.array(stocks)
    dates = np.array(dates) if dates[0] is not None else None
    
    return predictions, labels, stocks, dates
```

## 📊 修复效果

修复后的工作流程：

```
训练窗口 → 保存test_dataset
    ↓
预测窗口:
  1. 使用模型预测 → predictions
  2. 从sample_index遍历所有样本
  3. 从stock_data提取 labels, stocks, dates
  4. 返回完整元数据
    ↓
合并所有窗口的预测结果 → DataFrame
  包含: stock_col, time_col, pred_alpha, label_col, window_idx
```

## 🎯 关键要点

### 为什么要这样设计？

**TimeSeriesStockDataset的优化设计**：
- 每只股票的数据存储为连续数组（高效）
- 通过索引快速定位样本（O(1)复杂度）
- 避免重复存储相同股票的元数据

**不直接暴露属性的原因**：
- 数据是分散存储的（按股票）
- 样本顺序不等于原始数据顺序
- 需要通过sample_index动态提取

### 时间索引理解

```python
# 样本构建时的索引关系
for t in range(self.window_size - 1, n - 1):
    self.sample_index.append((stock_idx, t))
```

- `t` 是特征窗口的**结束位置**
- 特征窗口：`[t - window_size + 1, t]`
- 标签位置：`t + 1`（预测未来收益）
- 日期对应：`dates[t + 1]`（预测目标的日期）

### 示例

假设某股票有100天数据，window_size=40：

```
t=39: features[0:40], label[40], date[40]
t=40: features[1:41], label[41], date[41]
...
t=98: features[59:99], label[99], date[99]
```

## 🔧 适用场景

此修复适用于：
- ✅ 滚动窗口训练预测
- ✅ Walk-Forward验证
- ✅ 需要保留预测时间戳的场景
- ✅ 需要分窗口分析IC的场景

不影响：
- ✅ 单次训练预测（直接使用test DataFrame）
- ✅ 数据集的训练功能
- ✅ 其他类型的数据集

## 📌 后续改进建议

### 1. 添加辅助方法

可以为 `TimeSeriesStockDataset` 添加便捷方法：

```python
class TimeSeriesStockDataset(Dataset):
    def get_labels(self) -> np.ndarray:
        """提取所有样本的标签"""
        labels = []
        for idx in range(len(self)):
            stock_idx, time_idx = self.sample_index[idx]
            labels.append(self.stock_data[stock_idx]['labels'][time_idx + 1])
        return np.array(labels)
    
    def get_metadata(self) -> Dict[str, np.ndarray]:
        """提取所有样本的元数据"""
        labels, stocks, dates = [], [], []
        for idx in range(len(self)):
            stock_idx, time_idx = self.sample_index[idx]
            stock_info = self.stock_data[stock_idx]
            labels.append(stock_info['labels'][time_idx + 1])
            stocks.append(stock_info['ts_code'])
            dates.append(stock_info.get('dates', [None])[time_idx + 1])
        return {
            'labels': np.array(labels),
            'stocks': np.array(stocks),
            'dates': np.array(dates)
        }
```

### 2. 统一接口

考虑定义一个数据集接口协议：

```python
from typing import Protocol

class PredictableDataset(Protocol):
    """支持预测的数据集协议"""
    def get_labels(self) -> np.ndarray: ...
    def get_stocks(self) -> np.ndarray: ...
    def get_dates(self) -> np.ndarray: ...
```

## 🔗 相关代码

- `quantclassic/data_manager/factory.py` - TimeSeriesStockDataset定义
- `quantclassic/data_manager/rolling_trainer.py` - 滚动窗口训练器
- `quantclassic/data_manager/BUGFIX_ROLLING_WINDOW_EMPTY_DATASET.md` - 空数据集问题修复

## 📅 修复日期
2025-11-21

---

**教训总结**：
1. 优化的数据结构需要配套的访问接口
2. 不要假设所有Dataset都有相同的属性结构
3. 预测时需要保留完整的元数据（股票、日期）
4. 时间序列数据集的索引关系需要仔细处理（t vs t+1）
