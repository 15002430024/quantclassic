# quantclassic/model 模块重构方案

## 重构状态总览

| Step | 描述 | 状态 |
|------|------|------|
| Step 1 | 提取纯 nn.Module 实现层 | ⏳ 待定（HybridGraphModel 仍有 fit/train_epoch） |
| Step 2 | 统一训练循环到 PyTorchModel 基类 | ✅ **已完成** |
| Step 3 | 删除 DynamicGraphTrainer | ✅ **已删除** |
| Step 4 | 统一损失函数到 loss.py | ✅ **已完成** |
| Step 5 | 合并配置系统 | ✅ **已完成** (统一predict + 配置兼容层) |
| Step 6 | 图构建下沉至 DataLoader 层 | ✅ **已完成**（原本就在） |

---

## 图构建架构 (最终方案)

```
┌─────────────────────────────────────────────────────────────────────┐
│  data_processor/graph_builder.py                                    │
│  ─────────────────────────────────────────────────────────────────  │
│  职责: 图构建 **算法** (HOW to build)                               │
│  • GraphBuilder (ABC) - 抽象基类                                    │
│  • CorrGraphBuilder - 相关性图算法                                  │
│  • IndustryGraphBuilder - 行业图算法                                │
│  • HybridGraphBuilder - 混合图算法                                  │
│  输入: DataFrame → 输出: adj_matrix                                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓ 依赖注入
┌─────────────────────────────────────────────────────────────────────┐
│  data_set/graph/daily_graph_loader.py                               │
│  ─────────────────────────────────────────────────────────────────  │
│  职责: 数据 **组织** (WHEN to call + 批次切分)                       │
│  • DailyBatchDataset - 按日组织，每天一个样本                        │
│  • collate_daily - 批次组装时调用 GraphBuilder                       │
│  • DailyGraphDataLoader - 封装 DataLoader                           │
│  数据流: Dataset → collate_fn(调用GraphBuilder) → (X, y, adj)       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓ 返回 (X, y, adj, stock_ids, date)
┌─────────────────────────────────────────────────────────────────────┐
│  model/base_model.py::_parse_batch_data()                           │
│  ─────────────────────────────────────────────────────────────────  │
│  职责: 自动解析 batch 格式，被动接收 adj                             │
│  支持格式: (x,y) | (x,y,adj) | (x,y,adj,idx) | dict                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                    model.forward(X, adj=adj)
```

**职责边界**:
- `data_processor`: 数据预处理 pipeline，包含图构建**算法**
- `data_set`: 数据切分、滚动窗口、训练集/验证集划分，**调用** GraphBuilder
- `model`: 只关心 forward/backward，被动接收 adj

---

## 已完成的修改

### ✅ Step 2: 扩展 PyTorchModel 基类

**文件**: `base_model.py`

- 新增 `_parse_batch_data()` 方法，支持多种 batch 格式
- 更新 `_train_epoch()` 和 `_valid_epoch()` 以支持动态图

### ✅ Step 3: 删除 DynamicGraphTrainer

**文件**: `dynamic_graph_trainer.py` → 🗑️ **已删除**

**迁移路径**:
```python
# ❌ 旧用法（已删除）
from quantclassic.model import DynamicGraphTrainer
trainer = DynamicGraphTrainer(model, graph_builder)
trainer.fit(loader)

# ✅ 新用法
from quantclassic.data_set.graph import DailyGraphDataLoader
from quantclassic.data_processor.graph_builder import HybridGraphBuilder

# 图构建在 DataLoader 层完成
graph_builder = HybridGraphBuilder(alpha=0.7, top_k=10)
loader = DailyGraphDataLoader(dataset, graph_builder=graph_builder)

# 模型直接 fit，adj 从 batch 中自动解析
model.fit(loader)
```

### ✅ Step 4: 统一损失函数

**文件**: `loss.py`

- 新增 `UnifiedLoss` 类，支持：
  - 多种基础损失 (`mse`, `mae`, `huber`, `ic`)
  - 相关性正则化 (`lambda_corr`)
  - 多因子正交化 (`lambda_ortho`)
- 更新 `get_loss_fn()` 工厂函数

### ✅ Step 6: 图构建架构确认

图构建已正确分层：
- `data_processor/graph_builder.py` - 算法实现
- `data_set/graph/daily_graph_loader.py` - 调用时机

---

## 重构后的文件结构

```
model/
├── __init__.py              # ✅ 已更新导出
├── base_model.py            # ✅ 新增 _parse_batch_data()
├── hybrid_graph_models.py   # 纯 nn.Module 组件
├── pytorch_models.py        # LSTM/GRU/Transformer
├── loss.py                  # ✅ 新增 UnifiedLoss
├── modular_config.py        # CompositeModelConfig
├── model_factory.py         # ✅ 新增 create_model_from_composite_config
├── rolling_daily_trainer.py
└── utils/
    └── adj_matrix_builder.py

data_processor/
└── graph_builder.py         # ✅ GraphBuilder 算法（已存在）

data_set/
└── graph/
    └── daily_graph_loader.py  # ✅ DailyGraphDataLoader（已存在）
```

---

## 待决策事项

1. **HybridGraphModel 是否精简？**
   - 当前状态：仍有完整的 `fit()`, `_train_epoch()` 方法
   - 选项 A：保留现状（向后兼容）
   - 选项 B：精简为仅 forward + predict，训练逻辑委托基类

2. **model_config.py 是否删除？**
   - 需要检查外部依赖后决定

---

## 使用示例

```python
from quantclassic.data_set.graph import DailyGraphDataLoader, DailyBatchDataset
from quantclassic.data_processor.graph_builder import HybridGraphBuilder
from quantclassic.model import HybridGraphModel, UnifiedLoss

# 1. 创建图构建器（算法层）
graph_builder = HybridGraphBuilder(
    alpha=0.7, 
    corr_method='cosine', 
    top_k=10
)

# 2. 创建数据加载器（数据层调用算法）
dataset = DailyBatchDataset(df, feature_cols, label_col, window_size=20)
train_loader = DailyGraphDataLoader(dataset, graph_builder=graph_builder)

# 3. 创建模型
model = HybridGraphModel(d_feat=7, use_graph=True)

# 4. 训练 - adj 从 batch 自动解析
model.fit(train_loader)
```
# 训练架构统一与滚动逻辑重构计划


## ⚖️ 工程化设计标准 (Checklist)

对于每个重构的核心模块，必须满足：
1. [ ] **唯一实现**：删除或废弃所有旧的冗余副本。
2. [ ] **离线能力**：在 `scripts/` 提供对应的 CLI 脚本（如离线数据清洗、离线推理）。
3. [ ] **文档对齐**：在 `docs/` 更新架构文档和调用示例。
4. [ ] **测试覆盖**：至少具备一套端到端的集成测试。


## 核心目标
- **解耦**：将训练策略（如何练）从数据集管理（数据如何切、如何喂）中分离。
- **收敛**：消除 `data_set/rolling_trainer.py` 与 `model/rolling_daily_trainer.py` 的重复实现。
- **通用化**：所有训练器均通过配置驱动（Loss、Optimizer、Scheduler），不再硬编码。

## 模块分工建议
| 模块层级 | 所在路径 (建议) | 主要职责 |
| :--- | :--- | :--- |
| **数据层 (Dataset)** | `data_set/` | 负责 DataFrame 清洗、Rolling 窗口切分逻辑、生成 `DataLoader`。**不包含任何训练循环**。 |
| **模型实现 (Model)** | `model/pytorch_models.py` | 定义网络结构（LSTM, GRU, GAT 等）及正向传播。 |
| **训练引擎 (Trainer)** | `model/train/` | **核心建议增加：** 定义 `BaseTrainer` 及其子类。负责训练循环、权重继承、损失计算。 |
| **配置层 (Config)** | `config/` | 提供 `RollingTrainerConfig` 等参数容器，透传给训练引擎。 |

## 重构步骤
1. **创建基础架构**：建立 `model/train/base_trainer.py` 定义 `BaseTrainer` 基类。
    - 抽象 `train_epoch` 与 `train_batch` 接口。
    - 统一实现 EarlyStopping、Checkpoint 保存、Logging 逻辑。
    - **移交循环逻辑**：原 `PyTorchModel.fit` 中的 Epoch 循环将被移动至此，消除 `LSTM`/`GRU` 中的代码重复。
2. **实现特化训练器**：建立针对不同模式的训练子类：
    - `SimpleTrainer`：接管常规训练，替换原 `PyTorchModel.fit` 的实现（改为代理调用）。
    - `RollingWindowTrainer`：通用滚动，增加权重继承逻辑。
    - `RollingDailyTrainer`：日滚动模式，处理高频模型切换，接管 Walk-Forward 逻辑。
3. **修复与修复逻辑**：重构 `model/rolling_daily_trainer.py`，移除对已删除 `DynamicGraphTrainer` 的引用，改写为调用现有的 `PyTorchModel` 并支持配置驱动。
4. **修复数据透传 Bug**：修复 `data_set/manager.py` 中的 `stock_industry_mapping` 获取 bug，确保正确引用 `_raw_data` 而非未定义的 `df`。
5. **统一任务入口**：在 `config/runner.py` 中统一入口，将原本散落在各处的训练逻辑收口至新的 `model/train` 模块。
6. **清理冗余**：完成后清理不再使用的 `data_set/rolling_trainer.py` 冗余文件。

## 下一步确认与细节
```prompt
# 训练架构统一与滚动逻辑重构计划（落地版）

## 落地范围
- 新增：`model/train/base_trainer.py`, `model/train/simple_trainer.py`, `model/train/rolling_window_trainer.py`, `model/train/rolling_daily_trainer.py`, `model/train/__init__.py`（集中训练循环、早停、检查点、日志）。
- 修改：`model/pytorch_models.py`（`fit` 改为代理到 `SimpleTrainer`）、`model/rolling_daily_trainer.py`（变薄为新 Trainer 的适配层或迁移代码后删除）、`config/base_config.py` 与 `config/loader.py`（补齐训练器配置类与解析）、`config/runner.py`（统一入口）、`data_set/manager.py` 与 `data_set/loader.py`/`splitter.py`（数据透传/返回类型对齐）、相关 `tests` 与示例脚本。
- 删除：完成迁移后移除 `data_set/rolling_trainer.py` 及其引用；清理旧的重复逻辑与未用的备份文件（例如 `model/rolling_daily_trainer.py` 旧实现若不再被引用则删除）。

## 落地步骤（建议顺序）
1) 训练引擎骨架
   - 在 `model/train/base_trainer.py` 写 `BaseTrainer`：`train()` 主循环，抽象 `train_epoch/train_batch/validate_epoch`，内置早停、最佳模型保存、日志记录，接受损失/优化器/调度器构造器。
   - `BaseTrainer` 的输入统一为 `TrainerArtifacts`（包含 `model`, `optimizer`, `scheduler`, `criterion`, `train_loader`, `val_loader`, `device`, `metrics`, `callbacks`），便于后续 Trainer 复用。
2) 常规训练器
   - 在 `model/train/simple_trainer.py` 实现 `SimpleTrainer(BaseTrainer)`：仅覆盖批次/验证逻辑，支持单窗训练。
   - 修改 `model/pytorch_models.py`：`PyTorchModel.fit(...)` 只负责准备 `TrainerArtifacts` 与配置，实例化 `SimpleTrainer` 并调用 `train()`，保留原接口签名以兼容旧调用。
3) 滚动窗口训练
   - 在 `model/train/rolling_window_trainer.py` 实现窗口循环，参数 `weight_inheritance` 控制是否复用上一窗权重；权重继承时沿用同一 `model` 实例，否则重建。
   - 期望数据输入统一为 `RollingLoaderCollection`（或现有集合类型），包含按窗的 `train/val/test` DataLoader 列表；如类型缺失，在 `data_set/loader.py`/`splitter.py` 增加数据类定义与构造，确保 Trainer 的遍历接口稳定。
4) 日级滚动训练
   - 在 `model/train/rolling_daily_trainer.py` 基于滚动窗口 Trainer 复用逻辑，补充日频窗口切换时的显存管理（切窗前 `model.to('cpu')`，必要时 `del model` + `torch.cuda.empty_cache()`）。
   - 重写 `model/rolling_daily_trainer.py` 为薄适配：仅保留向后兼容的入口，内部导入并调用新 `RollingDailyTrainer`；确认无引用后可删除旧文件。
5) 配置与入口
   - 在 `config/base_config.py` 定义 `TrainerConfig`/`RollingTrainerConfig`（含 `epochs`, `optimizer`, `scheduler`, `early_stopping`, `weight_inheritance`, `checkpoint_dir` 等）。
   - 更新 `config/loader.py` 解析新配置，确保 CLI/runner 读取时能构造 Trainer 配置对象。
   - 在 `config/runner.py` 统一入口：根据配置选择 `SimpleTrainer`/`RollingWindowTrainer`/`RollingDailyTrainer`，并透传数据与模型。替换现有分散的训练调用。
6) 数据透传修复
   - 修复 `data_set/manager.py` 中 `stock_industry_mapping` 取值：使用 `self._raw_data` 而非未定义的 `df`，并校验空值/索引对齐。
   - 若数据层返回对象不统一，整理 `data_set/loader.py`/`splitter.py` 输出，确保包含 `train/val/test` loader 与窗口标识，供 Trainer 消费。
7) 清理与对齐
   - 全局搜索旧引用：`data_set/rolling_trainer.py`、`model/rolling_daily_trainer.py` 旧类名、`DynamicGraphTrainer`。迁移后删除旧文件与 import。
   - 更新示例与文档：`config/QUICKSTART.md`、`config/RUN_GUIDE.md`、`backtest/example_*` 中的训练调用路径。
   - 更新测试：`config/tests/`、`model/tests/`（或现有 `tests/`）补充对新 Trainer 的单元/集成测试，移除对旧文件的引用。

## 关键接口对齐（避免踩坑）
- Trainer 输入：统一使用数据类或字典，至少包含 `model`, `train_loader`, `val_loader`（可选 `test_loader`）, `optimizer`, `scheduler`, `criterion`, `device`, `metrics`。避免 Trainer 自行访问数据层内部状态。
- 数据层输出：滚动模式输出 `List[WindowData]`，其中每个 `WindowData` 含 `window_id` 与 `train/val/test` loader；日滚动沿用此结构但窗口粒度为日。
- 配置透传：`TrainerConfig`/`RollingTrainerConfig` 由 `config/runner.py` 组装，禁止在 Trainer 内部硬编码超参。
- 日滚动显存：切窗时确保模型移回 CPU 或释放缓存；必要时保存/载入 `state_dict` 而非持久驻留 GPU。

## 完成判定
- `PyTorchModel.fit` 已改为薄代理；主循环仅存在于 `model/train/`。
- 数据层不再包含任何训练循环，`data_set/rolling_trainer.py` 被删除且无引用。
- 新配置生效：`config/runner.py` 能根据配置调起三类 Trainer 并完成一次端到端训练/滚动训练。
- 旧示例/测试已更新且通过。
```
---

## 🆕 修改汇报 (2026-01-08)

### 已完成的修改

#### 1. 新增文件 (`model/train/` 目录)

| 文件 | 说明 |
|------|------|
| `model/train/__init__.py` | 训练模块入口，导出所有训练器和配置类 |
| `model/train/base_trainer.py` | 训练基类 `BaseTrainer`，定义通用训练循环、早停、检查点逻辑；包含 `TrainerConfig`、`TrainerArtifacts`、`TrainerCallback`、`EarlyStoppingCallback`、`CheckpointCallback` |
| `model/train/simple_trainer.py` | `SimpleTrainer` 简单训练器，接管常规单窗口训练，支持相关性正则化 |
| `model/train/rolling_window_trainer.py` | `RollingWindowTrainer` 滚动窗口训练器，支持权重继承、断点续训；包含 `RollingTrainerConfig`、`WindowData`、`WindowResult` 数据类 |
| `model/train/rolling_daily_trainer.py` | `RollingDailyTrainer` 日级滚动训练器，继承 `RollingWindowTrainer`，增加显存管理（`gc_interval`、`offload_to_cpu`、`clear_cache_on_window_end`）；包含 `DailyRollingConfig`、`create_rolling_daily_trainer` 工厂函数 |

#### 2. 修改文件

| 文件 | 修改内容 |
|------|----------|
| `model/pytorch_models.py` | `LSTMModel.fit()` 和 `GRUModel.fit()` 改为代理到 `SimpleTrainer`，保持接口兼容；添加 `Path` 导入和重构说明 |
| `model/__init__.py` | 导出新训练模块（`BaseTrainer`、`SimpleTrainer`、`RollingWindowTrainer`、`RollingDailyTrainer` 等）；保留旧 `rolling_daily_trainer` 兼容导入 |
| `config/base_config.py` | 更新 `TaskConfig.trainer_class` 支持新训练器列表；新增 `TrainerConfigDC`、`RollingTrainerConfigDC` DataClass 配置类 |
| `config/runner.py` | 新增 `_train_simple()` 和 `_train_rolling_window()` 方法；更新 `_train_rolling()` 优先使用新训练架构；训练器选择逻辑支持 `SimpleTrainer`、`RollingWindowTrainer`、`RollingDailyTrainer` |
| `data_set/manager.py` | 修复 `stock_industry_mapping` bug：将 `self.df` 改为 `self._raw_data` |

#### 3. 核心架构变更

```
训练流程（重构后）:
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│  config/runner  │ --> │  model/train/*      │ --> │  model/*.py     │
│  (统一入口)      │     │  (训练引擎)          │     │  (模型定义)      │
└─────────────────┘     └─────────────────────┘     └─────────────────┘
        │                         │
        v                         v
┌─────────────────┐     ┌─────────────────────┐
│  TaskConfig     │     │  TrainerConfig      │
│  trainer_class  │     │  RollingTrainerConfig│
│  trainer_kwargs │     │  DailyRollingConfig │
└─────────────────┘     └─────────────────────┘

数据流:
data_set/manager.py --> DataLoader --> Trainer.train() --> model.forward()
                   ↑                         │
                   └── 不再包含训练循环 ──────┘
```

#### 4. 接口对齐

- **Trainer 输入**: `TrainerArtifacts` 数据类封装 `model`、`optimizer`、`criterion`、`train_loader`、`val_loader`、`device` 等
- **训练配置**: `TrainerConfig` 包含 `n_epochs`、`lr`、`early_stop`、`optimizer`、`loss_fn`、`use_scheduler`、`lambda_corr` 等
- **滚动配置**: `RollingTrainerConfig` 继承 `TrainerConfig`，增加 `weight_inheritance`、`save_each_window`、`reset_optimizer` 等
- **日级配置**: `DailyRollingConfig` 继承 `RollingTrainerConfig`，增加 `gc_interval`、`offload_to_cpu`、`clear_cache_on_window_end`

#### 5. 待完成项目

- [ ] 删除 `data_set/rolling_trainer.py`（需确认无外部引用）
- [ ] 更新示例文档 `config/QUICKSTART.md`、`config/RUN_GUIDE.md`
- [ ] 补充单元测试 `model/train/tests/`
- [ ] 将旧 `model/rolling_daily_trainer.py` 标记为废弃或删除

### 复核发现（2026-01-08）
- `model/__init__.py` 仍强依赖旧版 `model/rolling_daily_trainer.py`（该文件引用缺失的 `dynamic_graph_trainer`），当前导入 `quantclassic.model` 会因模块缺失报错，需删除兼容导入或恢复动态图训练模块。
- 数据层仍保留训练循环：`data_set/manager.py:837` 继续暴露 `create_rolling_window_trainer`，并依赖旧 `data_set/rolling_trainer.py`，与新 `model/train/*` 架构重复，未实现“数据不含训练循环”目标。
- 参数透传缺失：`config/runner.py:442` 将 `weight_inheritance`/`save_each_window` 剥离出 `trainer_kwargs` 但未传入 `RollingTrainerConfig`，用户无法关闭权重继承或逐窗保存。
- 损失函数缺口：`model/train/base_trainer.py:422` 未覆盖 `loss_fn='ic'/'ic_corr'` 等无相关性正则场景，设置 IC 损失会直接抛错。
- 动态图训练入口失效：`config/runner.py:611` 仍调用已删除的 `model.dynamic_graph_trainer`，`trainer_class='DynamicGraphTrainer'` 路径无法使用。
### ✅ 复核修复（2026-01-08）

针对上述复核发现的问题，已完成以下修复：

#### 1. `model/__init__.py` 导入修复

**问题**：强依赖旧 `rolling_daily_trainer.py`，该文件引用缺失的 `dynamic_graph_trainer`

**修复**：
- 移除对旧文件的直接 `from .rolling_daily_trainer import ...` 导入
- `create_rolling_trainer()` 改为延迟导入函数，运行时发出废弃警告
- 兼容类名 `LegacyRollingDailyTrainer`/`LegacyRollingTrainerConfig` 直接指向新训练器

```python
# 修复后
def create_rolling_trainer(*args, **kwargs):
    import warnings
    warnings.warn("已废弃，请使用 model.train.RollingDailyTrainer", DeprecationWarning)
    from .train import create_rolling_daily_trainer
    return create_rolling_daily_trainer(*args, **kwargs)

LegacyRollingDailyTrainer = RollingDailyTrainer  # 指向新实现
```

#### 2. 数据层训练循环废弃标记

**问题**：`data_set/manager.py:create_rolling_window_trainer()` 仍暴露训练循环

**修复**：
- 添加 `DeprecationWarning` 警告
- 文档标记 `.. deprecated:: 2026.01`
- 保留向后兼容但不再推荐使用

```python
def create_rolling_window_trainer(self, stock_universe=None):
    """⚠️ 已废弃 - 请使用 model.train.RollingWindowTrainer"""
    import warnings
    warnings.warn(
        "DataManager.create_rolling_window_trainer() 已废弃，"
        "请使用 model.train.RollingWindowTrainer",
        DeprecationWarning, stacklevel=2
    )
    # ... 原逻辑
```

#### 3. 参数透传修复

**问题**：`config/runner.py` 将 `weight_inheritance`/`save_each_window` 剥离后未传入配置

**修复**：
- 参数同时写入 `init_kwargs` 和 `config_kwargs`
- 确保用户设置的 `weight_inheritance=False` 能正确生效

```python
for key, value in trainer_kwargs.items():
    if key in init_params:
        init_kwargs[key] = value
        if key in {'weight_inheritance', 'save_each_window'}:
            config_kwargs[key] = value  # 🆕 同时传入 config
```

#### 4. 损失函数支持扩展

**问题**：`base_trainer.py:_create_criterion()` 不支持 `ic`/`ic_corr` 损失

**修复**：
- 优先调用 `loss.get_loss_fn()` 工厂函数
- 添加 `ic`/`ic_corr` 回退处理
- 捕获 `ValueError` 异常避免抛错

```python
def _create_criterion(self):
    try:
        from ..loss import get_loss_fn
        return get_loss_fn(loss_type=loss_name, lambda_corr=self.config.lambda_corr)
    except (ImportError, ValueError):
        # 回退到标准损失
        if loss_name in ['ic', 'ic_corr']:
            self.logger.warning("IC 损失需要 loss 模块，回退到 MSE")
            return nn.MSELoss()
```

#### 5. 动态图训练入口修复

**问题**：`config/runner.py:_train_dynamic_graph()` 调用已删除的 `DynamicGraphTrainer`

**修复**：
- 改用 `SimpleTrainer` 替代
- 文档标注 `DynamicGraphTrainer` 已废弃
- 保持接口兼容

```python
def _train_dynamic_graph(self, model, daily_loaders, trainer_kwargs):
    """🆕 使用 SimpleTrainer (DynamicGraphTrainer 已废弃)"""
    from ..model.train import SimpleTrainer, TrainerConfig
    trainer = SimpleTrainer(model=nn_model, config=config)
    results = trainer.train(train_loader, val_loader, n_epochs)
```

### ✅ 复核状态（2026-01-08 已修复）

| 问题 | 状态 | 修复说明 |
|------|------|----------|
| 旧 `model/rolling_daily_trainer.py` 报错 | ✅ 已修 | 文件已改为 shim，发出废弃警告并导入新训练器 |
| 数据层包含训练循环 | ✅ 已修 | `data_set/__init__.py` 和 `manager.py` 移除 `RollingWindowTrainer` 导入；`create_rolling_window_trainer()` 改为抛 `NotImplementedError` |
| 滚动参数透传缺失 | ✅ 已修 | `_train_rolling_window` 已将 `weight_inheritance`/`save_each_window` 写入 `RollingTrainerConfig` |
| IC 损失不支持 | ✅ 已修 | `_create_criterion` 通过 `loss.get_loss_fn` 支持 `ic`/`ic_corr` |
| DynamicGraphTrainer 缺失 | ✅ 已修 | Runner 入口添加废弃警告，内部使用 `SimpleTrainer` 执行 |

### 动态批次 / 动态图现状（2026-01-08 最终）
- ✅ 日批次动态邻接图：`DataManager.create_rolling_daily_loaders` → `DailyGraphDataLoader.collate_daily` 按日构图（corr/industry/hybrid），返回 `(X, y, adj, stock_ids, date)`
- ✅ 数据流：`TaskConfig.use_rolling_loaders=True` 且 `trainer_class='RollingDailyTrainer'` → `config/runner.py` 创建日滚动 loaders → `model/train/rolling_daily_trainer.py` 完成训练
- ✅ 动态图单模型模式：`trainer_class='DynamicGraphTrainer'` 保留兼容但发出废弃警告，内部使用 `SimpleTrainer`
- ✅ 数据层旧滚动训练器：引用已移除，`data_set/rolling_trainer.py` 文件可安全删除

### ✅ 已完成项目（2026-01-08）

- [x] 移除 `data_set/__init__.py` 和 `manager.py` 中的 `RollingWindowTrainer` 导入
- [x] `DataManager.create_rolling_window_trainer()` 改为抛 `NotImplementedError` 并提供迁移指南
- [x] 旧 `model/rolling_daily_trainer.py` 改为 shim，发出废弃警告
- [x] `config/runner.py` 的 `DynamicGraphTrainer` 路径添加废弃警告
- [x] 更新文档 `config/QUICKSTART.md`、`config/RUN_GUIDE.md`
- [x] 补充 `model/train/tests/`：覆盖 ic/ic_corr 创建、滚动参数透传测试

### 🗑️ 待清理文件

以下文件可安全删除（已无引用）:
- `data_set/rolling_trainer.py` - 旧滚动训练器实现，功能已迁移至 `model/train/`


### 🔎 新发现问题（模型模块，2026-01-11）
- `SimpleTrainer` 对已有 `config` 使用 `config.update(**kwargs)`，但 dataclass 无 `update` 方法，传入覆盖参数会直接抛 `AttributeError`，导致无法覆盖配置。
- 滚动训练的 `reset_optimizer/reset_scheduler=False` 失效：每窗都会重新 new 一个 `SimpleTrainer`，从未复用上一窗的优化器/调度器状态，warm-start 训练无法保留动量。
- `TrainerConfig.validate` 仅允许 `mse/mae/huber/ic` 等少数值，拒绝 `mae_corr/huber_corr/combined/unified` 等 loss；与 `loss.get_loss_fn` 能力不一致，合法配置会被误报 `ValueError`。
- 核心模型 `predict` 仍用 `for batch_x, _ in test_loader` 解包，遇到图/日级 loader `(x, y, adj, ...)` 会抛 “too many values to unpack”。需改为与 `_parse_batch_data` 一致的解析逻辑。
- `model/train/__init__.py` 未导出 `DailyRollingConfig`，`from quantclassic.model.train import DailyRollingConfig` 会失败，影响日滚配置在外部的直接引用。



---

## 🛠️ 问题修复方案（2026-01-11）

### 问题1：SimpleTrainer 配置覆盖错误

**现象**：
```python
trainer = SimpleTrainer(model, config=existing_config, n_epochs=50)  
# ❌ AttributeError: 'TrainerConfig' object has no attribute 'update'
```

**根因**：dataclass 没有内置 `update` 方法，直接调用会失败。

**解决方案**：
```python
# model/train/simple_trainer.py
from dataclasses import replace

def __init__(self, model, config=None, device=None, **kwargs):
    if config is None:
        config = TrainerConfig(**kwargs)
    elif kwargs:
        # ✅ 使用 dataclasses.replace 创建新实例
        config = replace(config, **kwargs)
    super().__init__(model, config, device)
```

**修改文件**：
- ✅ `model/train/simple_trainer.py` 第53-56行

---

### 问题2：滚动训练优化器/调度器状态丢失

**现象**：
- 配置 `reset_optimizer=False` 但每窗都重建优化器，动量信息丢失
- Warm-start 训练无法实现真正的增量学习

**根因**：
```python
# 当前逻辑：每窗都创建新的 SimpleTrainer 实例
window_trainer = SimpleTrainer(model, self.config, ...)
# 新实例会重新创建 optimizer/scheduler，丢失上一窗口状态
```

**解决方案**：
在 `RollingWindowTrainer` 中保存和恢复优化器/调度器状态：

```python
# model/train/rolling_window_trainer.py

class RollingWindowTrainer:
    def __init__(self, ...):
        # 🆕 添加状态保存字段
        self.current_optimizer_state: Optional[Dict] = None
        self.current_scheduler_state: Optional[Dict] = None
    
    def train(self, rolling_loaders, ...):
        for window_idx, loaders in enumerate(rolling_loaders):
            model = self._get_model_for_window(window_idx)
            window_trainer = SimpleTrainer(model, self.config, ...)
            
            # 🆕 恢复优化器状态（非首窗且配置不重置）
            if not self.config.reset_optimizer and window_idx > 0 and self.current_optimizer_state:
                window_trainer._create_optimizer()
                window_trainer.optimizer.load_state_dict(self.current_optimizer_state)
            
            #  恢复调度器状态
            if not self.config.reset_scheduler and window_idx > 0 and self.current_scheduler_state:
                window_trainer._create_scheduler()
                window_trainer.scheduler.load_state_dict(self.current_scheduler_state)
            
            # 训练...
            train_result = window_trainer.train(...)
            
            # 🆕 保存状态供下一窗口使用
            if not self.config.reset_optimizer and window_trainer.optimizer:
                self.current_optimizer_state = copy.deepcopy(window_trainer.optimizer.state_dict())
            if not self.config.reset_scheduler and window_trainer.scheduler:
                self.current_scheduler_state = copy.deepcopy(window_trainer.scheduler.state_dict())
```

**修改文件**：
- ✅ `model/train/rolling_window_trainer.py` 第168行（添加状态字段）
- ✅ 第316-330行（状态恢复逻辑）
- ✅ 第367-371行（状态保存逻辑）

---

### 问题3：TrainerConfig 损失函数白名单过严

**现象**：
```python
config = TrainerConfig(loss_fn='mae_corr')  
# ❌ ValueError: 不支持的损失函数: mae_corr
```

但 `loss.get_loss_fn('mae_corr')` 是合法的。

**根因**：`validate()` 白名单只包含 `['mse', 'mae', 'huber', 'ic', 'mse_corr', 'ic_corr']`，缺少 `mae_corr`, `huber_corr`, `combined`, `unified`。

**解决方案**：
```python
# model/train/base_trainer.py

def validate(self) -> bool:
    #  扩展损失函数白名单，与 loss.get_loss_fn 保持一致
    supported_losses = [
        'mse', 'mae', 'huber', 'ic',  # 标准损失
        'mse_corr', 'mae_corr', 'huber_corr', 'ic_corr',  # 带相关性正则
        'combined', 'unified'  # 组合/统一损失
    ]
    if self.loss_fn not in supported_losses:
        raise ValueError(
            f"不支持的损失函数: {self.loss_fn}. "
            f"支持的损失: {', '.join(supported_losses)}"
        )
    return True
```

**修改文件**：
- ✅ `model/train/base_trainer.py` 第109-122行

---

### 问题4：模型 predict 方法批次解包不兼容

**现象**：
```python
# 日级 loader 返回：(x, y, adj, stock_ids, date)
for batch_x, _ in test_loader:  # ❌ too many values to unpack (expected 2)
```

**根因**：老式解包 `batch_x, _` 假定只有2个元素，不支持图/日级 loader 的多元素格式。

**解决方案**：
使用与 `BaseTrainer._parse_batch_data` 一致的解析逻辑：

```python
# model/pytorch_models.py

def predict(self, test_loader, return_numpy=True):
    predictions = []
    with torch.no_grad():
        for batch_data in test_loader:
            # ✅ 支持多种批次格式
            if isinstance(batch_data, (list, tuple)):
                batch_x = batch_data[0]
            elif isinstance(batch_data, dict):
                batch_x = batch_data.get('x') or batch_data.get('features')
            else:
                batch_x = batch_data
            
            batch_x = batch_x.to(self.device)
            pred = self.model(batch_x)
            predictions.append(pred.cpu())
```

**修改文件**：
- ✅ `model/pytorch_models.py` 第519-540行（GRUModel.predict）
- ✅ 第877-890行（VAEWithPredictor.predict）
- ✅ 其他模型的 predict 方法同样修复

---

### 问题5：DailyRollingConfig 缺失导出

**现象**：
```python
from quantclassic.model.train import DailyRollingConfig
# ❌ ImportError: cannot import name 'DailyRollingConfig'
```

**根因**：`model/train/__init__.py` 只导入了 `RollingDailyTrainer`，未导入其配置类。

**解决方案**：
```python
# model/train/__init__.py

from .rolling_daily_trainer import RollingDailyTrainer, DailyRollingConfig  # ✅

__all__ = [
    ...,
    'RollingDailyTrainer',
    'RollingTrainerConfig',
    'DailyRollingConfig',  # ✅ 添加到导出列表
]
```

**修改文件**：
- ✅ `model/train/__init__.py` 第32行（import）、第51行（__all__）

---

## ✅ 修复汇总（2026-01-11）

| 问题 | 影响范围 | 修复方法 | 修改文件 |
|------|---------|---------|---------|
| 1. SimpleTrainer config.update() | 配置覆盖失败 | 使用 `dataclasses.replace` | `simple_trainer.py:53-56` |
| 2. 滚动训练状态丢失 | 优化器动量丢失 | 保存/恢复 optimizer/scheduler state_dict | `rolling_window_trainer.py:168,316-330,367-371` |
| 3. 损失函数白名单过严 | 合法 loss 被拒绝 | 扩展 `validate()` 支持列表 | `base_trainer.py:109-122` |
| 4. predict 批次解包错误 | 图/日级 loader 报错 | 统一使用 `_parse_batch_data` 逻辑 | `pytorch_models.py:519,877` |
| 5. DailyRollingConfig 缺失 | 外部无法引用 | 添加到 `__all__` | `train/__init__.py:32,51` |

### 验证方式

```python
# 1. 测试配置覆盖
config = TrainerConfig(n_epochs=100)
trainer = SimpleTrainer(model, config, n_epochs=50)  # ✅ 应成功

# 2. 测试优化器复用
config = RollingTrainerConfig(reset_optimizer=False)
trainer = RollingWindowTrainer(model_factory, config)
# 训练后检查：trainer.current_optimizer_state 应非 None

# 3. 测试损失函数
config = TrainerConfig(loss_fn='unified')  # ✅ 应通过验证

# 4. 测试图级 loader
daily_loader = create_rolling_daily_loaders(...)  # 返回 (x,y,adj,ids,date)
predictions = model.predict(daily_loader)  # ✅ 应正常运行

# 5. 测试导入
from quantclassic.model.train import DailyRollingConfig  # ✅ 应成功
```

### 🧭 动态图支持与图构建合并方案（2026-01-11）

- 支持现状：动态图路径已可用，`config/runner.py` 在 `trainer_class='DynamicGraphTrainer'` 时退化为 `SimpleTrainer` 并走日级 loaders 分支，训练/推理均正常。
- 调用关系：`config/runner.py` → `DataManager.create_daily_loaders/create_rolling_daily_loaders` → `GraphBuilderFactory` 构建图 → `DailyGraphDataLoader` → `collate_daily` 每个 batch 触发图构建 → `SimpleTrainer/RollingDailyTrainer` 前向时消费 `(X, y, adj, stock_ids, date)`。
- 图计算频率：`collate_daily` 会在每次迭代时调用 `graph_builder(df_day)`，等价于“每个 epoch × 每个交易日”都重新构图；行业图可复用预加载缓存，相关性/混合图默认每批重算。
- 图构建实现/文档重复：`model/utils/adj_matrix_builder.py` 与 `model/build_industry_adj.py` 与 `data_processor/graph_builder.py` 能力重叠。
- 合并建议：
    - 运行时统一依赖 `data_processor/graph_builder.py` + `data_set/graph/daily_graph_loader.py:collate_daily`；保留 `GraphBuilderFactory` 为唯一入口。
    - 将 `build_industry_adj.py` 作为离线 CLI/脚本并迁至 `scripts/`（或 `data_processor/cli/`），复用同一 GraphBuilder，去除重复实现。
    - `model/utils/adj_matrix_builder.py` 若无额外能力，标记废弃并移除；如需保留可改为调用 GraphBuilder，并作为纯工具类放在 data_processor 层。
    - 文档整合到 `docs/graph/adjacency.md`（新建），描述三种模式（corr/industry/hybrid）、缓存策略、离线/在线用法，链接唯一实现文件。
- TODO：
    - [x] 选定唯一实现（建议 `data_processor/graph_builder.py`）并替换其他引用
    - [x] 提取 `build_industry_adj.py` 的 CLI 能力至 `scripts/graph/build_adj.py`，调用 GraphBuilder
    - [x] 移除或标记废弃 `model/utils/adj_matrix_builder.py` 并更新文档指向唯一入口
    - [x] 补充 `docs/graph/adjacency.md`，写明调用链与缓存/性能注意事项


### ✅ 图构建合并完成（2026-01-11）

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/graph/build_adj.py` | 新增 | CLI 脚本，复用 `GraphBuilderFactory`，支持 `--type industry/corr/hybrid` |
| `docs/graph/adjacency.md` | 新增 | 完整文档：三种图模式、调用链、缓存策略、迁移指南 |
| `model/utils/adj_matrix_builder.py` | 废弃 | 添加 `DeprecationWarning`，指向 `AdjMatrixUtils` |

**CLI 用法：**
```bash
python scripts/graph/build_adj.py --data data.parquet --type industry --output output/industry_adj.pt
python scripts/graph/build_adj.py --data data.parquet --type hybrid --alpha 0.7 --top-k 10
```

**迁移指南：**
```python
# 旧（已废弃）
from model.utils.adj_matrix_builder import AdjMatrixBuilder
builder = AdjMatrixBuilder()  # ⚠️ 触发 DeprecationWarning

# 新（推荐）
from quantclassic.data_processor.graph_builder import AdjMatrixUtils, GraphBuilderFactory
adj = AdjMatrixUtils.build_industry_adj(codes)
```


### 🧭 统一模型 predict 方法方案（2026-01-11）
predict部分能否单独实现，让其他模型复用，还是说这个控制在model中实现是更好的？

这里有两条可选路径，按推荐顺序给出：

1) 统一到基类/工具，减少重复（推荐）
- 在 model/base_model.py 提供一个通用 `predict()`，内部复用已有 `_parse_batch_data`，只接收 `test_loader`、`return_numpy`，其余模型只需实现 `forward()`（或 `_forward_impl()`）并继承该 `predict` 即可。
- 若有少数模型需要额外后处理（如还原尺度、聚合多个输出），在子类里覆写一个小的后处理钩子（如 `_post_process(pred)`），保持主体预测流程一致。
- 这样可以避免在 model/pytorch_models.py 各个模型里重复写批次解包和设备迁移逻辑。

2) 保持在各模型中实现（不推荐，除非模型差异大）
- 仅当某些模型的预测链路与多数模型完全不同（例如生成式模型需要多步采样、多头输出需要特殊合并）时才单独实现。
- 缺点是批次格式扩展（图/日级 loader 等）容易遗漏，维护成本高。

建议采用方案 1，并在 `BaseModel` 中：
- 复用 `_parse_batch_data`，支持 `(x,y) / (x,y,adj,...) / dict`。
- 统一 `self.device` 迁移、`torch.no_grad()`、`return_numpy`。
- 提供一个可选 `_post_process` 钩子供子类覆写。

这样可以让所有模型（LSTM/GRU/HybridGraph等）共享预测逻辑，同时保留必要的扩展点。

# 模型配置重构计划（2026-01-11）


### 1. 为什么有两个 Config？
*   **model_config.py (旧/扁平化)**：
    *   **设计逻辑**：每个模型一个类（如 `LSTMConfig`, `GRUConfig`）。
    *   **优点**：简单直接，适合只用单一时序模型的场景。
    *   **缺点**：扩展性差。如果要给 LSTM 增加图注意力机制，就得新写一个 `LSTMWithGATConfig`，导致类爆炸。
*   **`model/modular_config.py` (新/模块化)**：
    *   **设计逻辑**：通过模块组合。模型被拆分为 `Temporal` (时序), `Graph` (图), `Fusion` (融合) 三大件。
    *   **优点**：极其灵活。你可以通过配置轻松实现“GRU + 行业图 + MLP融合”，而不需要修改代码。
    *   **地位**：它是 `HybridGraphModel` 的核心支撑，也是未来架构的方向。

### 2. 有必要留 model_config.py 吗？
从软件工程的**单一事务原则 (DRY)** 来看，**没有必要留**。

`modular_config.py` 中的 `CompositeModelConfig` 完全可以覆盖 `LSTMConfig` 的所有功能。例如，一个纯 LSTM 模型在模块化配置下就是：
```python
config = CompositeModelConfig(
    temporal=TemporalModuleConfig(rnn_type='lstm', hidden_size=64),
    graph=None, # 关闭图模块
    fusion=FusionModuleConfig(hidden_sizes=[64])
)
```

### 3. 重构建议路线
为了保证不破坏现有代码，建议采取以下步骤进行“消灭”：

1.  **统一基类**：确保所有配置都继承自 base_config.py 中的 `BaseConfig`。
2.  **快捷入口**：在 `modular_config.py` 中保留或增加类似 `ConfigTemplates.pure_lstm()` 的静态方法，让用户能以一行代码获取旧版 `LSTMConfig` 的效果。
3.  **重定向**：将 model_config.py 修改为“兼容层”，内部逻辑全部指向 `modular_config.py`，并标记 `DeprecationWarning`（废弃警告）。
4.  **最终删除**：待所有脚本和笔记本迁移完成后，直接删除 model_config.py。

**结论**：在你的重构计划中，建议将 **Step 5 (合并配置系统)** 落实为：**全面转向模块化配置，并将扁平配置作为其快捷模板实现。**

## ✅ 可行性评估与落地方案（2026-01-11）

### 统一 `predict` 路线
- 现状：存在两套解包逻辑，`PyTorchModel` 尚无通用 `predict`，子类各自实现（如 [model/pytorch_models.py](model/pytorch_models.py#L174-L219) 的 LSTM、[model/pytorch_models.py](model/pytorch_models.py#L344-L375) 的 GRU、[model/pytorch_models.py](model/pytorch_models.py#L527-L558) 的 Transformer、[model/pytorch_models.py](model/pytorch_models.py#L872-L908) 的 VAE）。其中 GRU 仍用 `for batch_x, _`，对日级/图 Loader 会报 unpack 错误；LSTM/Transformer 自行解包，存在重复代码。
- 支撑能力：`PyTorchModel._parse_batch_data` 已覆盖 `(x,y) / (x,y,adj,...) / dict` 等格式 [model/base_model.py](model/base_model.py#L349-L401)，可作为统一入口。HybridGraph 需额外解析 `funda/stock_idx`，已在自定义 `_parse_batch_data` 中完成 [model/hybrid_graph_models.py](model/hybrid_graph_models.py#L1285-L1355)。
- 可行性：高。多数模型仅需 `x`（或可选 `adj`），统一到基类后可在子类通过 `_post_process()` 钩子处理特殊输出（如 VAE 返回潜变量、多头输出）。

**重构步骤**
1) 在 `PyTorchModel` 增加通用 `predict(test_loader, return_numpy=True)`：复用 `_parse_batch_data`，将 `x/adj/idx` 迁移到 `self.device`，调用 `_forward_for_predict(x, adj=None, idx=None)`（默认调 `self.model(x)`），支持可选 `_post_process(pred)` 钩子。空集返回零长度张量/ndarray，保持旧行为。
2) 子类接入：
    - 将 LSTM/Transformer/GRU/VAE 的 `predict` 精简为调用 `super().predict` 或删除覆盖，必要时覆写 `_post_process`（例如 VAE 处理 `return_latent`）。
    - HybridGraph 保留自定义 `_parse_batch_data`，但可通过覆写 `_forward_for_predict` 复用通用流程。
3) Trainer 侧对齐：`SimpleTrainer.predict` 可直接委托模型的通用 `predict`，避免重复解包。
4) 测试与回归：补充/更新针对 `(x,y,adj,stock_ids,date)` loader 的单测，覆盖 `return_numpy=True/False`、空 loader、VAE latent 输出三种场景。

### 配置系统合并
- 现状：旧扁平配置在 [model/model_config.py](model/model_config.py#L112-L221) 定义 `LSTMConfig/GRUConfig/...` 并由 `ModelConfigFactory` 暴露模板 [model/model_config.py](model/model_config.py#L487-L603)；新模块化配置在 [model/modular_config.py](model/modular_config.py#L330-L416) (`CompositeModelConfig`) + 预置模板 [model/modular_config.py](model/modular_config.py#L746-L821)。两套并存，存在重复维护和分支逻辑。
- 可行性：高。模块化配置功能超集（同等时序/图/融合参数已覆盖），且已有 `ConfigTemplates.pure_temporal/temporal_with_graph` 等快捷入口，可承载旧模板。

**重构步骤**
1) 建立兼容层：在 `model_config.py` 中增加废弃提示并将 `ModelConfigFactory.create/from_dict/get_template` 代理到 `CompositeModelConfig` + `ConfigTemplates`，返回对象改为模块化配置；保留旧类定义用于类型兼容但标记 `DeprecationWarning`。
2) 统一入口：`model_factory.create_model_from_config`（如 [model/model_factory.py](model/model_factory.py#L179-L210)）默认接受 `CompositeModelConfig`，若收到旧配置则调用转换函数 `compat.to_composite(cfg)`。
3) 文档与示例：在 `README.md`/示例脚本中只展示模块化配置；旧配置使用示例改为“废弃+迁移示例”。
4) 清理时机：待示例和外部调用迁移完毕后，删除旧模板分支，保留一个轻量 shim（或彻底删除文件）。

### 落地优先级
1) ✅ 先落地通用 `predict` 以消除现有 unpack bug 并减少重复；
2) ✅ 随后完成配置兼容层，避免新老配置混用；
3) 两者完成后与文档同步。

---

### 🎉 重构完成记录 (2026-01-11)

**统一 predict 方法**
- 在 `PyTorchModel` 基类添加通用 `predict()` 方法 ([base_model.py](base_model.py#L673-L768))
- 提供 `_forward_for_predict()` 和 `_post_process()` 钩子供子类覆写
- LSTM/GRU/Transformer 已移除重复 predict 代码，使用基类实现
- VAE 保留 `return_latent` 支持，通过覆写 `_forward_for_predict()` 实现

**配置系统兼容层**
- `model_config.py` 添加废弃警告 (`DeprecationWarning`)
- 新增 `to_composite_config()` 转换函数，支持旧配置自动迁移
- `ModelConfigFactory.create/from_dict/get_template` 调用时触发废弃提示

### 🚩 新发现问题（2026-01-11）

#### 问题：VAEModel.extract_latent 批次解包不兼容

**现象**：
```python
# 日级 loader 返回：(x, y, adj, stock_ids, date)
latent = vae_model.extract_latent(daily_loader)
# ❌ ValueError: too many values to unpack (expected 2)
```

**根因**：[pytorch_models.py#L857](pytorch_models.py#L857) 使用 `for batch_x, _ in test_loader` 解包，假定只有 2 个元素。

**可行性**：✅ 非常高
- 代码改动小：与已完成的 `predict` 修复逻辑完全一致
- 无风险：`_parse_batch_data` 已在基类验证，支持所有 batch 格式
- 遗漏原因：`extract_latent` 是 VAE 特有方法，统一 `predict` 时被遗漏

**修复方案**：
```python
# model/pytorch_models.py - VAEModel.extract_latent

def extract_latent(self, test_loader, return_numpy: bool = True):
    """提取潜在特征（用于因子生成）"""
    if not self.fitted:
        raise ValueError("模型未训练，请先调用 fit()")
    
    self.model.eval()
    mu_list = []
    z_list = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            # 🆕 使用基类统一的 batch 解析（替代 for batch_x, _ in ...）
            batch_x, _, _, _ = self._parse_batch_data(batch_data)
            
            batch_x = batch_x.to(self.device)
            mu, logvar = self.model.encode(batch_x)
            z = self.model.reparameterize(mu, logvar)
            mu_list.append(mu.cpu())
            z_list.append(z.cpu())
    
    # 处理空输入
    if len(mu_list) == 0:
        import numpy as np
        empty = np.array([]) if return_numpy else torch.tensor([])
        return empty, empty
    
    mu_features = torch.cat(mu_list, dim=0)
    z_features = torch.cat(z_list, dim=0)
    
    if return_numpy:
        return mu_features.numpy(), z_features.numpy()
    return mu_features, z_features
```

**修改文件**：`model/pytorch_models.py` 第 840-870 行

**验证方式**：
```python
from quantclassic.model import VAEModel
from quantclassic.data_set.graph import DailyGraphDataLoader

# 创建日级 loader (返回 5 元素元组)
daily_loader = DailyGraphDataLoader(dataset, graph_builder=builder)

# 测试 extract_latent
vae = VAEModel(d_feat=20, latent_dim=16)
vae.fit(train_loader, val_loader)
mu, z = vae.extract_latent(daily_loader)  # ✅ 应正常运行
```

**状态**：✅ 已修复 (2026-01-11)

**实际修改**：
- `for batch_x, _ in test_loader` → `batch_x, _, _, _ = self._parse_batch_data(batch_data)`
- 新增空输入处理逻辑，与基类 `predict` 行为一致

---

## 🔍 代码复核发现（2026-01-11 README 更新后）

### 问题 1：`_parse_batch_data` 签名不一致

**现象**：
| 类 | 返回值 | 位置 |
|----|--------|------|
| `PyTorchModel` | `(x, y, adj, idx)` | [base_model.py#L349-L401](base_model.py#L349-L401) |
| `HybridGraphModel` | `(x, stock_idx, funda)` | [hybrid_graph_models.py#L1323-L1400](hybrid_graph_models.py#L1323-L1400) |

**影响**：
- 子类调用时签名不一致，容易混淆
- 无法在基类层面统一处理 `funda` 字段
- HybridGraphModel 无法复用基类的通用 `predict()`

**设计决策（保持现状）**：
- **理由**：两者职责不同
  - `PyTorchModel._parse_batch_data`：通用时序模型，只需 x/y/adj/idx
  - `HybridGraphModel._parse_batch_data`：图模型专用，需要 stock_idx 做图索引 + funda 基本面数据
- **风险**：低。HybridGraphModel 已完全覆写 `predict()`，不依赖基类实现
- **未来优化**：若需统一，可在基类增加第 5 个返回值 `funda`，默认 None

---

### 问题 2：predict 方法重复实现

**现状分析**：

| 模型/模块 | predict 来源 | 是否重复 | 原因 |
|-----------|-------------|----------|------|
| LSTM/GRU/Transformer | 基类 `PyTorchModel` | ❌ 否 | 已迁移，使用统一实现 |
| VAEModel | 完整覆写 | ⚠️ 部分 | 需支持 `return_latent` 参数 |
| HybridGraphModel | 完整覆写 | ⚠️ 部分 | 需支持图推理（缓存/截面/邻居采样） |
| SimpleTrainer | 独立实现 | ⚠️ 是 | Trainer 是独立模块，不继承模型 |

**VAEModel.predict 重复代码**：
```python
# 当前实现（pytorch_models.py#L780-L827）
def predict(self, test_loader, return_numpy=True, return_latent=False):
    # 完整实现了批次解包、设备迁移、空处理逻辑
    # 与基类 PyTorchModel.predict 高度相似
```

**解决方案 A：精简 VAEModel.predict（推荐）**

```python
# model/pytorch_models.py - VAEModel

def predict(self, test_loader, return_numpy=True, return_latent=False):
    """
    预测（扩展基类支持 return_latent）
    """
    if not return_latent:
        # 不需要潜变量时，直接使用基类实现
        return super().predict(test_loader, return_numpy)
    
    # 需要潜变量时，使用自定义逻辑
    if not self.fitted:
        raise ValueError("模型未训练，请先调用 fit()")
    
    self.model.eval()
    predictions = []
    latent_features = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            batch_x, _, _, _ = self._parse_batch_data(batch_data)
            batch_x = batch_x.to(self.device)
            _, y_pred, _, _, z = self.model(batch_x)
            predictions.append(y_pred.cpu())
            latent_features.append(z.cpu())
    
    # 空处理
    if len(predictions) == 0:
        empty = np.array([]) if return_numpy else torch.tensor([])
        return empty, empty
    
    predictions = torch.cat(predictions, dim=0)
    latent_features = torch.cat(latent_features, dim=0)
    
    if return_numpy:
        return predictions.numpy(), latent_features.numpy()
    return predictions, latent_features
```

**可行性**：✅ 高
- 代码改动小
- 复用基类的空处理和设备迁移逻辑
- 保持 `return_latent=True` 的特殊行为

---

### 问题 3：SimpleTrainer.predict 与模型 predict 重复

**现象**：
- `SimpleTrainer.predict()` 在 [train/simple_trainer.py#L195-L234](train/simple_trainer.py#L195-L234)
- `PyTorchModel.predict()` 在 [base_model.py#L673-L768](base_model.py#L673-L768)
- 两者逻辑高度相似（批次解析 → 设备迁移 → 前向 → 空处理）

**解决方案 B：SimpleTrainer 委托模型 predict**

```python
# model/train/simple_trainer.py

def predict(self, test_loader, return_numpy: bool = True):
    """
    预测 - 委托给模型的 predict 方法
    
    如果模型有自己的 predict()（如 PyTorchModel 子类），直接调用；
    否则回退到 Trainer 自己的实现。
    """
    # 🆕 检查模型是否有 predict 方法
    if hasattr(self.model, 'predict') and callable(getattr(self.model, 'predict')):
        # 委托给模型（模型的 predict 已包含完整逻辑）
        # 注意：nn.Module 没有 predict，但 PyTorchModel 子类有
        try:
            return self.model.predict(test_loader, return_numpy)
        except TypeError:
            pass  # 模型的 predict 签名不兼容，回退
    
    # 回退：Trainer 自己的实现（用于纯 nn.Module）
    self.model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            x, _, adj, _ = self._parse_batch_data(batch_data)
            x = x.to(self.device)
            if adj is not None:
                adj = adj.to(self.device)
            
            try:
                pred = self.model(x, adj=adj) if adj is not None else self.model(x)
            except TypeError:
                pred = self.model(x)
            
            if isinstance(pred, tuple):
                pred = pred[0]
            
            predictions.append(pred.cpu())
    
    if len(predictions) == 0:
        return np.array([]) if return_numpy else torch.tensor([])
    
    result = torch.cat(predictions, dim=0)
    return result.numpy() if return_numpy else result
```

**可行性**：✅ 中高
- 需要区分 `nn.Module`（无 predict）和 `PyTorchModel`（有 predict）
- 委托后可减少重复，但增加一层间接调用
- 建议：仅当传入的是 `PyTorchModel` 子类时委托

---

### 问题 4：HybridGraphModel.predict 无法复用基类

**现象**：HybridGraphModel 完全覆写了 `predict()`，包含 ~60 行代码

**原因分析**：
1. 需要自定义 `_parse_batch_data` 返回 `(x, stock_idx, funda)`
2. 需要调用 `_forward_step()` 而非简单的 `model(x)`
3. 支持多因子输出格式 `[N, F]`
4. 支持图推理模式切换（batch/cross_sectional/neighbor_sampling）

**设计决策（保持现状）**：
- **理由**：HybridGraphModel 的预测逻辑与时序模型差异太大，强行统一会增加复杂度
- **成本**：维护 ~60 行独立代码 vs 引入复杂的继承/组合结构
- **建议**：保持独立实现，但确保与基类行为兼容（空处理、return_numpy 等）

---

### 问题 5：model_config.py 冗余

**现状**：
- `model_config.py` 已标记废弃（文件头有 DeprecationWarning）
- `modular_config.py` 是推荐的新系统
- 两者仍可同时导入使用

**解决方案 C：彻底清理 model_config.py**

**阶段 1（当前）**：保持兼容层
```python
# model_config.py 保留但触发警告
_emit_deprecation_warning("LSTMConfig")
```

**阶段 2（下一版本）**：移除冗余类
- 删除 `LSTMConfig`, `GRUConfig`, `TransformerConfig` 等类定义
- 仅保留 `to_composite_config()` 转换函数
- 文件改名为 `model_config_compat.py`

**阶段 3（未来版本）**：完全删除
- 确认无外部依赖后删除文件
- `__init__.py` 移除相关导入

**时间表**：
| 阶段 | 时间 | 动作 |
|------|------|------|
| 1 | 2026-01（当前） | 触发 DeprecationWarning |
| 2 | 2026-Q2 | 移除冗余类，保留 compat 函数 |
| 3 | 2026-Q3+ | 确认无依赖后完全删除 |

---

## 📋 优化待办清单（2026-01-11）

### 高优先级（影响功能）

- [ ] **暂无** - 当前所有已知 bug 已修复

### 中优先级（代码质量）

| 任务 | 文件 | 说明 | 状态 |
|------|------|------|------|
| 精简 VAEModel.predict | `pytorch_models.py` | `return_latent=False` 时复用基类 | ✅ 已完成 (2026-01-11) |
| SimpleTrainer 委托 | `train/simple_trainer.py` | 对 PyTorchModel 子类委托 predict | ✅ 已完成 (2026-01-11) |
| 补充单测 | `train/tests/` | 覆盖图级 loader 的 predict 场景 | ⏳ 待办 |

### 低优先级（代码清理）

| 任务 | 文件 | 说明 | 状态 |
|------|------|------|------|
| 清理 model_config.py | `model_config.py` | 移除冗余类定义，仅保留 compat | ⏳ 计划 Q2 |
| 统一 _parse_batch_data | `base_model.py` | 考虑增加 `funda` 返回值 | ❌ 决定保持现状 |
| 删除历史注释 | 多个文件 | 清理 "🆕"、"替代 for batch_x, _" 等过渡注释 | ⏳ 低优先级 |

---

## ✅ 设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| `_parse_batch_data` 签名 | 保持不一致 | PyTorchModel 和 HybridGraphModel 职责不同 |
| HybridGraphModel.predict | 保持独立 | 图推理逻辑复杂，统一成本高 |
| VAEModel.predict | 精简复用 | `return_latent=False` 时可复用基类 |
| SimpleTrainer.predict | 条件委托 | 对 PyTorchModel 委托，纯 nn.Module 回退 |
| model_config.py | 分阶段清理 | 保持兼容，逐步废弃 |
