# 数据集与模型重构衔接问题 & 解决方案

> **更新时间**: 2026-01-12  
> **状态**: ✅ 已修复

## 问题概述

| # | 问题 | 影响 | 状态 |
|---|------|------|------|
| 1 | 滚动训练示例不可用 | 示例脚本调用已移除的 `create_rolling_window_trainer()`，运行抛 `NotImplementedError` | ✅ 已修复 |
| 2 | 图构建配置透传不完整 | `create_daily_loaders` 不支持 dataclass 配置，与 rolling 版本行为不一致 | ✅ 已修复 |
| 3 | `use_daily_batch` 标志失效 | 配置类定义了该开关，但流水线从未读取，无法自动走日批次/动态图链路 | ✅ 已修复 |
| 4 | 截面采样不可达 | `DataManager.get_dataloaders` 未暴露 `use_cross_sectional` 参数 | ✅ 已修复 |
| 5 | 日批次加载逻辑重复 | `create_daily_loaders` 与 `create_rolling_daily_loaders` 各自维护一套构建流程 | ✅ 已修复 |
| 6 | 滚动示例预测阶段接口不匹配 | `RollingDailyTrainer.train` 返回 summary dict，示例仍按旧接口遍历窗口，预测阶段将抛异常/空结果 | ✅ 已修复 |

---

## 修复详情

### 1. 示例对齐：更新 `example_rolling_training.py`

**修改内容**：
- 移除对已废弃 `DataManager.create_rolling_window_trainer()` 的调用
- 移除对已废弃 `GRUConfig` 的依赖
- 改用新训练器架构：`model.train.RollingDailyTrainer` + `RollingTrainerConfig`
- 使用 `dm.create_rolling_daily_loaders()` 获取滚动窗口数据

**修改前**：
```python
from quantclassic.model.model_config import GRUConfig
trainer = dm.create_rolling_window_trainer()
results = trainer.train_all_windows(model_class=GRUModel, model_config=gru_config)
```

**修改后**：
```python
from quantclassic.model.train import RollingDailyTrainer, RollingTrainerConfig

rolling_loaders = dm.create_rolling_daily_loaders(val_ratio=0.15)
trainer = RollingDailyTrainer(model_factory=model_factory, config=trainer_config)

# 训练所有窗口
summary = trainer.train(rolling_loaders)

# 获取全部预测结果
predictions = trainer.get_all_predictions()
print(f"训练窗口数: {summary['n_windows']}")
print(f"平均训练损失: {summary['avg_train_loss']:.6f}")
```

---

### 2. 图配置一致化：新增 `_normalize_graph_builder_config()` 辅助函数

**修改文件**: `manager.py`

**修改内容**：
- 新增模块级辅助函数 `_normalize_graph_builder_config()`
- 统一处理 `graph_builder_config`：dict 或 dataclass 均可
- 自动注入行业图所需的 `stock_industry_mapping`
- `create_daily_loaders` 和 `create_rolling_daily_loaders` 共用此函数

**新增代码**：
```python
def _normalize_graph_builder_config(
    gb_config: Optional[Union[Dict, Any]],
    raw_data: Optional[pd.DataFrame] = None,
    stock_col: str = 'ts_code',
    logger: Optional[logging.Logger] = None
) -> Optional[Dict]:
    """统一处理 graph_builder_config，确保返回 dict 类型"""
    if gb_config is None:
        return None
    
    # 统一转换为 dict
    if isinstance(gb_config, dict):
        gb_dict = gb_config.copy()
    elif hasattr(gb_config, 'to_dict'):
        gb_dict = gb_config.to_dict()
    else:
        gb_dict = dict(gb_config)
    
    # 行业图自动注入映射
    if gb_dict.get('type') == 'industry':
        ...
    
    return gb_dict
```

---

### 3. 激活 `use_daily_batch` 配置

**修改文件**: `manager.py` (`run_full_pipeline` 方法)

**修改内容**：
- 读取 `config.use_daily_batch` 配置
- 当为 `True` 时，自动调用 `create_daily_loaders()` 返回日批次加载器
- 否则维持原逻辑返回逐样本 `LoaderCollection`

**修改代码**：
```python
# run_full_pipeline 中
use_daily = getattr(self.config, 'use_daily_batch', False)
if use_daily:
    self.logger.info("🆕 use_daily_batch=True，创建日批次加载器")
    loaders = self.create_daily_loaders(
        graph_builder_config=getattr(self.config, 'graph_builder_config', None),
        shuffle_dates=getattr(self.config, 'shuffle_dates', True)
    )
else:
    loaders = self.get_dataloaders()
```

---

### 4. 暴露截面采样参数

**修改文件**: `manager.py` (`get_dataloaders` 方法)

**修改内容**：
- 新增 `use_cross_sectional: bool = False` 参数
- 透传给 `DatasetCollection.get_loaders()`
- 用户可通过公开接口开启截面批采样（IC/相关性损失场景）

**修改签名**：
```python
def get_dataloaders(
    self, 
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    shuffle_train: Optional[bool] = None,
    use_cross_sectional: bool = False  # 🆕 新增参数
) -> LoaderCollection:
```

---

### 5. 去重日批次构建逻辑

**修改文件**: `manager.py`

**修改内容**：
- 提取公共 `_normalize_graph_builder_config()` 函数
- `create_daily_loaders` 和 `create_rolling_daily_loaders` 共用此函数
- 减少配置处理代码重复，避免后续漂移

---

### 6. 滚动示例预测阶段与新训练器接口对齐

**问题**：
- `RollingDailyTrainer.train` 返回 summary dict，并将预测保存在内部（通过 `get_all_predictions()` 获取）；示例原先把 `results` 当作窗口列表并访问 `window_result['model']`，导致类型不符。

**修改内容**：
- 训练返回变量改为 `summary`
- 预测阶段改用 `trainer.get_all_predictions()` 直接获取汇总结果
- 结果分析阶段使用 `summary['n_windows']`、`summary['avg_train_loss']`、`summary['avg_val_loss']`

**修改后**：
```python
# 训练
summary = trainer.train(rolling_loaders)

# 获取全部预测（含窗口标记）
predictions = trainer.get_all_predictions()
if predictions.empty:
    print("⚠️ 无预测结果（测试集可能为空）")
    return

print(f"汇总预测样本: {len(predictions):,}")
print(f"平均训练损失: {summary['avg_train_loss']:.6f}")
print(f"平均验证损失: {summary['avg_val_loss']:.6f}")
```

---

## 预期收益

- ✅ **示例可运行**：与新训练器架构一致，降低踩坑成本
- ✅ **配置透传一致**：dataclass/dict 均可传入，避免运行时错误
- ✅ **功能开关生效**：`use_daily_batch`、截面采样等配置按预期工作
- ✅ **代码去重**：日批次构建逻辑单一来源，后续更新不再需要双处维护

---

## 使用示例

### 1. 使用日批次模式（GNN 动态图训练）

```python
from quantclassic.data_set import DataManager, DataConfig

config = DataConfig(
    base_dir='rq_data_parquet',
    use_daily_batch=True,  # 🆕 启用日批次模式
    graph_builder_config={'type': 'hybrid', 'alpha': 0.7, 'top_k': 10}
)

dm = DataManager(config)
daily_loaders = dm.run_full_pipeline()  # 自动返回 DailyLoaderCollection

for X, y, adj, stock_ids, date in daily_loaders.train:
    pred = model(X, adj)
```

### 2. 使用截面批采样（IC Loss 训练）

```python
dm = DataManager(config)
dm.run_full_pipeline()

# 🆕 开启截面采样，确保每个 batch 来自同一交易日
loaders = dm.get_dataloaders(use_cross_sectional=True)
```

### 3. 滚动窗口训练（新架构）

```python
from quantclassic.model.train import RollingDailyTrainer, RollingTrainerConfig

dm = DataManager(DataConfig(split_strategy='rolling', ...))
dm.run_full_pipeline()

rolling_loaders = dm.create_rolling_daily_loaders(val_ratio=0.15)

trainer = RollingDailyTrainer(
    model_factory=lambda: GRUModel(d_feat=len(dm.feature_cols)),
    config=RollingTrainerConfig(weight_inheritance=True)
)
results = trainer.train(rolling_loaders)
```

data_set/update_readme/DATASET_MODEL_ALIGNMENT.md#L33-L44 的“修改后”示例仍用 results = trainer.train(rolling_loaders)，未展示获取预测的方式；与下方第 6 节已对齐的新接口说明存在轻微不一致，易让读者误以为返回的是窗口列表。
建议处理
在 DATASET_MODEL_ALIGNMENT 的“修改后”代码段中同步展示 summary = trainer.train(...) 与 trainer.get_all_predictions() 的用法，保持上下文一致。
