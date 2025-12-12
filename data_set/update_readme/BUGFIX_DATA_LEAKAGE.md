# 🐛 数据泄露问题修复 - 标签列被错误地用作特征

## 📋 问题描述

在使用 `DataManager` 自动检测特征列时，发现**标签列 `alpha_label` 被错误地选择为特征列**，导致模型训练时使用了标签数据，造成严重的**数据泄露（Data Leakage）**。

### 问题表现

```
特征列数量: 42
特征列表: vol, amount, limit_up, ..., alpha_label  ❌
```

**实际训练时**：
- 模型看到了 `alpha_label` 作为输入特征
- `alpha_label` 同时也是预测目标
- 相当于"看着答案做题"，导致虚高的训练准确率

### 影响范围

✅ **已影响的训练**：
- Cell 8 的数据准备
- Cell 9 的滚动窗口训练
- 所有使用 `dm.feature_cols` 的模型训练

❌ **后果**：
- 训练性能虚高（模型直接学习标签）
- 泛化能力极差（真实预测时没有标签可用）
- IC值不真实（看着答案预测）
- 回测结果失效

## 🔍 根本原因

### 1. 配置问题

**用户配置（Cell 7）**：
```python
data_config = DataConfig(
    stock_col='order_book_id',    # 股票代码列
    time_col='trade_date',        # 时间列
    label_col='alpha_label',      # 标签列
    exclude_cols=['y_ret_1d', 'y_ret_5d', 'y_ret_10d'],  # ❌ 没有包含 label_col
    feature_cols=None,            # 自动检测
)
```

**问题**：
- `exclude_cols` 只包含了其他标签列，**没有包含 `alpha_label`**
- 用户期望系统自动排除 `label_col`

### 2. 代码问题

**原代码（`feature_engineer.py:68`）**：
```python
def select_features(self, df: pd.DataFrame, auto_select: bool = True) -> List[str]:
    self.logger.info("🔍 自动检测特征列...")
    
    # 排除列
    exclude = set(self.config.exclude_cols)  # ❌ 只用用户配置的 exclude_cols
    
    # 选择数值型列
    feature_cols = [
        col for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    return feature_cols
```

**问题分析**：
1. `stock_col='order_book_id'` → 字符串类型 → 自动被 `is_numeric_dtype` 过滤 ✅
2. `time_col='trade_date'` → 日期类型 → 自动被 `is_numeric_dtype` 过滤 ✅
3. **`label_col='alpha_label'` → 数值类型 → 没有被过滤** ❌

### 3. 设计缺陷

系统假设用户会手动将 `label_col` 加入 `exclude_cols`，但这不符合直觉：
- 用户已经在 `label_col` 参数中明确指定了标签列
- 理应由系统自动排除，而不是要求用户重复配置

## ✅ 解决方案

### 修复代码

修改 `quantclassic/data_manager/feature_engineer.py` 的 `select_features()` 方法：

```python
def select_features(self, df: pd.DataFrame, auto_select: bool = True) -> List[str]:
    """
    选择特征列
    
    Args:
        df: 数据DataFrame
        auto_select: 是否自动选择特征
        
    Returns:
        特征列列表
    """
    # 如果配置中已指定特征列
    if self.config.feature_cols is not None:
        self.feature_cols = self.config.feature_cols
        self.logger.info(f"✅ 使用配置的特征列: {len(self.feature_cols)} 列")
        return self.feature_cols
    
    if not auto_select:
        raise ValueError("未指定特征列且auto_select=False")
    
    self.logger.info("🔍 自动检测特征列...")
    
    # 排除列（包括配置的排除列 + 系统列）
    exclude = set(self.config.exclude_cols)
    
    # ✅ 【修复】强制排除系统列（stock_col, time_col, label_col）
    system_cols = {
        self.config.stock_col,
        self.config.time_col,
        self.config.label_col
    }
    exclude.update(system_cols)
    
    # 选择数值型列
    feature_cols = [
        col for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    self.feature_cols = feature_cols
    self.logger.info(f"✅ 自动选择特征列: {len(feature_cols)} 列")
    
    return feature_cols
```

### 关键改动

**新增代码（第 71-76 行）**：
```python
# 【修复】强制排除系统列（stock_col, time_col, label_col）
system_cols = {
    self.config.stock_col,
    self.config.time_col,
    self.config.label_col
}
exclude.update(system_cols)
```

**作用**：
- 自动将 `stock_col`、`time_col`、`label_col` 加入排除列表
- 无论用户是否在 `exclude_cols` 中指定，都会被强制排除
- 防止数据泄露

## 📊 修复效果

### 修复前

```
特征数量: 42
特征列表:
  1. vol
  2. amount
  ...
  41. is_limit_down
  42. alpha_label  ❌ [标签列被用作特征]

数据泄露检查:
  ❌ 错误：标签列 'alpha_label' 被用作特征（数据泄露）！
```

### 修复后

```
特征数量: 41
特征列表:
  1. vol
  2. amount
  ...
  41. is_limit_down
  [alpha_label 已正确排除]

数据泄露检查:
  ✅ 正确：标签列 'alpha_label' 已正确排除
```

## 🔄 重新训练步骤

由于之前的训练数据存在数据泄露，需要**重新训练所有模型**：

### 1. 清除缓存

```bash
# 删除缓存的特征列和数据集
rm -rf jupyterlab/cache/data_manager/*
rm -rf jupyterlab/output/feature_columns.txt
```

### 2. 重新运行 Cell 8

```python
# Cell 8 已更新为重新加载 feature_engineer 模块
importlib.reload(sys.modules['quantclassic.data_manager.feature_engineer'])

dm = DataManager(config=data_config)
loaders = dm.run_full_pipeline()
```

**预期输出**：
```
特征维度: 41  ✅ (原来是 42)
数据泄露检查:
  ✅ 正确：标签列 'alpha_label' 已正确排除
```

### 3. 重新运行 Cell 9

```python
# 滚动窗口训练
trainer = RollingWindowTrainer(...)
results = trainer.train_all_windows(...)
```

### 4. 验证修复

检查新生成的 `feature_columns.txt`：
```bash
cat jupyterlab/output/feature_columns.txt | grep alpha_label
# 应该没有任何输出（表示 alpha_label 不在特征列中）
```

## 💡 最佳实践

### 1. 配置时的注意事项

```python
# ❌ 错误做法 - 需要用户手动排除标签列
data_config = DataConfig(
    label_col='alpha_label',
    exclude_cols=['y_ret_1d', 'alpha_label'],  # 重复配置
)

# ✅ 正确做法 - 系统自动排除标签列
data_config = DataConfig(
    label_col='alpha_label',
    exclude_cols=['y_ret_1d', 'y_ret_5d'],  # 只配置其他需要排除的列
)
```

### 2. 数据泄露检查清单

在训练前，确保：
- ✅ 特征列中**不包含**标签列
- ✅ 特征列中**不包含**未来信息（如未来收益率）
- ✅ 特征列中**不包含**股票代码、时间等ID列
- ✅ 标签生成逻辑不使用未来数据

### 3. 自动验证

```python
# 在 DataManager 中添加验证
def validate_no_leakage(self):
    """验证是否存在数据泄露"""
    leakage_cols = set(self.feature_cols) & {
        self.config.label_col,
        self.config.stock_col,
        self.config.time_col
    }
    
    if leakage_cols:
        raise ValueError(f"数据泄露！以下列同时是特征和系统列: {leakage_cols}")
```

## 🔗 相关文档

- `BUGFIX_ROLLING_WINDOW_EMPTY_DATASET.md` - 空数据集问题
- `BUGFIX_DATASET_METADATA_EXTRACTION.md` - 元数据提取问题
- `BUGFIX_EMPTY_DATALOADER.md` - 空DataLoader问题

## 📅 修复日期
2025-11-21

## 🏷️ 影响版本
- quantclassic.data_manager v1.0.0+

## ✅ 测试验证

修复后应通过以下测试：

### 1. 特征列不包含标签

```python
dm = DataManager(config=data_config)
dm.run_full_pipeline()

assert data_config.label_col not in dm.feature_cols, "标签列不应在特征列中"
```

### 2. 系统列全部排除

```python
system_cols = {
    data_config.stock_col,
    data_config.time_col,
    data_config.label_col
}
feature_set = set(dm.feature_cols)

assert len(system_cols & feature_set) == 0, "系统列不应在特征列中"
```

### 3. 特征数量正确

```python
# 原始数据: 67 列
# 系统列: 3 列 (stock_col, time_col, label_col)
# 非数值列: 2 列 (industry_name)
# exclude_cols: 3 列 (y_ret_1d, y_ret_5d, y_ret_10d)
# 其他非数值: 17 列 (open, high, low, close, ...)
# 预期特征数: 67 - 3 - 2 - 3 - 17 = 42

# 修复后应该是 41 列（因为排除了 alpha_label）
assert len(dm.feature_cols) == 41, f"特征数量应为41，实际为{len(dm.feature_cols)}"
```

### 4. 端到端测试

```python
# 训练模型
model = GRUModel(gru_config)
model.fit(loaders.train, loaders.val)

# 预测
predictions = model.predict(loaders.test)

# 验证预测结果是否合理
# 如果存在数据泄露，IC值会异常高（>0.5）
ic = np.corrcoef(predictions, true_labels)[0, 1]
assert ic < 0.3, f"IC值过高 ({ic:.3f})，可能存在数据泄露"
```

---

**总结**：本次修复通过在 `feature_engineer.py` 中强制排除系统列（`stock_col`, `time_col`, `label_col`），彻底解决了标签列被错误用作特征的数据泄露问题。用户无需手动配置 `exclude_cols` 来排除标签列，系统会自动处理。
