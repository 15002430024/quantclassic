# 滚动窗口训练空数据集问题修复

## 🐛 问题描述

在运行滚动窗口训练（Walk-Forward Validation）时，出现以下错误：

```
ValueError: num_samples should be a positive integer value, but got num_samples=0
WARNING: 窗口 0 的训练数据集为空
```

## 🔍 问题根源

### 1. 标签生成时的数据缺失

在数据预处理流程中，生成未来收益率标签（`y_ret_1d`, `y_ret_5d`, `y_ret_10d`）后，进行SimStock中性化生成`alpha_label`时：

- **原始数据**：1,647,355 条（2015-01-05 ~ 2024-12-10）
- **标签缺失**：137,812 条（8.37%）
- **有效标签范围**：2016-01-14 ~ 2024-12-10（前一年数据无标签）

原因分析：
1. 标签生成需要未来数据（如10日收益率需要未来10个交易日）
2. SimStock中性化需要历史相似性计算（lookback_window=252天）
3. 因此，**数据开始的前252+10天内**的标签为缺失值

### 2. 滚动窗口划分的问题

滚动窗口配置：
```python
rolling_window_size = 252  # 训练窗口252天
rolling_step = 63          # 滚动步长63天
```

第一个窗口的训练数据范围：
- 时间范围：2015-01-05 ~ 2016-01-14（约252个交易日）
- 问题：这个时间段的`alpha_label`**全部为缺失值**！

结果：创建第一个窗口的训练数据集时，过滤掉缺失值后样本数为0，导致DataLoader失败。

## ✅ 解决方案

### 修改位置1：`quantclassic/data_manager/manager.py`

在 `split_datasets()` 方法中，**划分数据前先过滤标签缺失的行**：

```python
# 【修复】过滤掉标签缺失的数据（防止滚动窗口训练时出现空数据集）
original_len = len(df)
df = df[df[self.config.label_col].notna()].copy()
filtered_len = len(df)
if filtered_len < original_len:
    self.logger.info(f"   过滤标签缺失数据: {original_len:,} -> {filtered_len:,} (-{original_len-filtered_len:,})")
```

**作用**：
- 在数据划分前统一过滤，确保所有窗口都使用有效数据
- 避免滚动窗口训练时出现空数据集

### 修改位置2：`quantclassic/data_manager/splitter.py`

在 `RollingWindowSplitter.split()` 方法中，**排序后立即过滤标签缺失的数据**：

```python
# 【修复】确保数据中有有效标签（避免空窗口）
if self.config.label_col in df.columns:
    original_len = len(df)
    df = df[df[self.config.label_col].notna()].copy()
    if len(df) < original_len:
        self.logger.info(f"   过滤标签缺失数据: {original_len:,} -> {len(df):,}")
```

**作用**：
- 双重保险，确保splitter内部也进行过滤
- 防止其他代码路径直接调用splitter时出现问题

## 📊 修复效果

修复后的数据流程：

```
原始数据: 1,647,355 条（2015-01-05 ~ 2024-12-10）
    ↓
过滤标签缺失: -137,812 条
    ↓
有效数据: 1,509,543 条（2016-01-14 ~ 2024-12-10）
    ↓
滚动窗口划分: 
  - 窗口1: 2016-01-14 ~ 2016-12-30（有数据）✅
  - 窗口2: 2016-04-18 ~ 2017-04-03（有数据）✅
  - ...
  - 窗口35: 有数据 ✅
```

## 🎯 总结

### 问题本质
标签生成和中性化处理需要时间窗口，导致数据开始阶段的标签缺失，而滚动窗口恰好从数据开始处划分，造成第一个窗口无有效数据。

### 解决关键
**在数据划分前统一过滤标签缺失的数据**，而不是在创建Dataset时过滤，确保所有后续流程都使用有效数据。

### 适用场景
- 所有使用滚动窗口训练的场景
- 所有需要前瞻标签（future labels）的训练场景
- 所有使用中性化标签的训练场景

### 不影响的场景
- 静态划分（time_series, stratified, random）：这些划分方式通常会自动跳过数据开始的无效部分
- 在线预测：预测时会单独处理缺失值

## 📌 使用建议

1. **检查标签生成参数**：确保标签生成逻辑合理（如`return_periods`, `lookback_window`）
2. **预览数据时间范围**：训练前检查有效标签的时间范围
3. **调整窗口起始位置**：如果需要使用早期数据，可以调整标签生成策略
4. **监控日志输出**：留意"过滤标签缺失数据"的日志，了解实际使用的数据量

## 🔗 相关代码

- `quantclassic/data_manager/manager.py` - 数据管理器
- `quantclassic/data_manager/splitter.py` - 数据划分器
- `quantclassic/data_manager/rolling_trainer.py` - 滚动窗口训练器
- `quantclassic/data_processor/preprocess_config.py` - 预处理配置（标签生成）

## 📅 修复日期
2025-11-21
