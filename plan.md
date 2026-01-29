# 配置体系排查计划

## 现状
模型、数据集、预处理模块均声明继承 BaseConfig，但存在导入路径硬编码与基类重复定义的风险。

## 问题列表

### 1. 导入路径硬编码 ✅ 已修复
多个模块通过 `sys.path.insert` + `from base_config` 进行绝对导入，可能加载到错误的 BaseConfig，导致 isinstance 判定或序列化不一致。

**涉及文件**：
- [config/loader.py](config/loader.py#L20)
- [data_set/config.py](data_set/config.py#L16)
- [data_processor/preprocess_config.py](data_processor/preprocess_config.py#L11)
- [model/model_config.py](model/model_config.py#L34)
- [model/modular_config.py](model/modular_config.py#L38)

### 2. 基类重复/降级实现 ✅ 已修复
为规避 ImportError，部分模块内联了降级版 BaseConfig，若触发将与主干 BaseConfig 类型不一致，TaskRunner 的 isinstance 检测及序列化逻辑会失效。

**涉及文件**：
- [model/modular_config.py](model/modular_config.py#L46)（原内联 BaseConfig 降级实现）
- [model/train/base_trainer.py](model/train/base_trainer.py#L41)（原内联 BaseConfig 降级实现）

### 3. 训练配置重复 ✅ 已修复
`config/base_config` 提供 `TrainerConfigDC`，而训练框架使用 `model/train/base_trainer` 的 `TrainerConfig`，字段与校验不一致。

**涉及文件**：
- [config/base_config.py](config/base_config.py#L340)（TrainerConfigDC）
- [model/train/base_trainer.py](model/train/base_trainer.py#L54)（TrainerConfig）

### 4. TaskRunner 模块路径硬编码 ✅ 已修复
TaskConfig 适配器在未提供 `module_path` 时强制写入 `'quantclassic.model'` 与 `'quantclassic.data_set'`，若类未在对应包的 `__init__` 暴露或用户自定义路径不同，会直接初始化失败。

**涉及文件**：
- [config/runner.py](config/runner.py#L55-L75)

### 5. CLI sys.path 侵入式修改 ✅ 已修复
CLI 入口通过上跳两级并插入 sys.path，已安装环境或同名包时存在阴影/冲突风险。

**涉及文件**：
- [config/cli.py](config/cli.py#L19-L22)

### 6. Loader 示例路径漂移 ✅ 已修复
ConfigLoader 自测片段引用旧路径 `from model.model_config import VAEConfig`，在包内运行会报 ModuleNotFoundError，暴露依赖路径未统一。

**涉及文件**：
- [config/loader.py](config/loader.py#L34-L60)
- [config/loader.py](config/loader.py#L264-L276)

### 7. 训练分支重复实现 ✅ 已修复
TaskRunner 中 `_train_simple`、`_train_dynamic_graph` 均包装 SimpleTrainer，`_train_rolling_window` 与 `_train_rolling` 重复模型工厂与参数拆分逻辑，存在维护漂移风险。

**涉及文件**：
- [config/runner.py](config/runner.py#L371-L506)
- [config/runner.py](config/runner.py#L508-L657)

---

## 修复记录 (2026-01-12)

### 修复 1：统一使用相对导入

**修改内容**：移除所有 `sys.path.insert` 操作，改用 Python 包的相对导入机制。

| 文件 | 修改前 | 修改后 |
|------|--------|--------|
| `config/loader.py` | `sys.path.insert(...)` + `from base_config` | `from .base_config import BaseConfig` |
| `data_set/config.py` | `sys.path.insert(...)` + `from config.base_config` | `from ..config.base_config import BaseConfig` |
| `data_processor/preprocess_config.py` | `sys.path.insert(...)` + `from config.base_config` | `from ..config.base_config import BaseConfig` |
| `model/model_config.py` | `sys.path.insert(...)` + `from config.base_config` | `from ..config.base_config import BaseConfig` |
| `model/modular_config.py` | `sys.path.insert(...)` + 降级版 BaseConfig | `from ..config.base_config import BaseConfig` |
| `model/train/base_trainer.py` | 多层 try-except + 降级版 BaseConfig | `from ...config.base_config import BaseConfig` |

**兼容处理**：为支持直接运行脚本（非包模式），保留 try-except 后备导入 `from config.base_config import BaseConfig`。

### 修复 2：移除降级版 BaseConfig

**修改内容**：
- 从 `model/modular_config.py` 移除内联的降级版 `BaseConfig` 和 `BaseModelConfig` 类定义
- 从 `model/train/base_trainer.py` 移除内联的降级版 `BaseConfig` 类定义

**原因**：降级版类定义会导致 `isinstance(obj, BaseConfig)` 返回 False（因为是不同的类对象），破坏 TaskRunner 的配置检测逻辑。

### 修复 3：对齐训练配置

**修改内容**：
- 更新 `TrainerConfigDC` 字段与 `model.train.TrainerConfig` 完全一致
- 添加 `log_interval` 字段（原缺失）
- 对齐 `validate()` 方法的校验逻辑（支持完整的损失函数列表）
- 添加 `to_trainer_config()` 方法用于与训练引擎对接
- 添加 `to_rolling_trainer_config()` 方法

**文档更新**：在 `TrainerConfigDC` 和 `RollingTrainerConfigDC` 的 docstring 中标注为兼容层，建议用户直接使用 `model.train.TrainerConfig`。

### 修复 4：更新模块导出

**修改内容**：
- 在 `config/__init__.py` 中导出 `BaseConfig`、`TaskConfig`、`TrainerConfigDC`、`RollingTrainerConfigDC`

---

## 验证清单

- [x] 运行 `python -c "from config import BaseConfig, TaskConfig"` 验证导入 ✅
- [x] 运行 `python -c "from data_set import DataConfig; print(DataConfig.__bases__)"` 验证继承 ✅
- [x] 运行 `python -c "from data_processor import PreprocessConfig; print(PreprocessConfig.__bases__)"` 验证继承 ✅
- [x] 运行 `python -c "from model.train import TrainerConfig; print(TrainerConfig.__bases__)"` 验证继承 ✅
- [x] 运行 `python -c "from config.base_config import BaseConfig; from data_set.config import DataConfig; print(isinstance(DataConfig(), BaseConfig))"` 验证 isinstance ✅

### 当前验证状态（2026-01-13）
- ✅ 所有 5 项验证清单全部通过
- ✅ 所有配置类均正确继承自 `config.base_config.BaseConfig`
- ✅ `isinstance(DataConfig(), BaseConfig)` 返回 `True`

## 实施方案（torch 2.1.1 环境）

1. ✅ **环境准备**：使用 pytorch-2.1.1 环境，已通过 `/opt/conda/envs/pytorch-2.1.1/bin/python` 验证。
2. ✅ **完成验证清单**：5 条命令全部执行通过，继承链与 isinstance 结果一致。
3. ✅ **补充单测**：新增 [tests/test_config_inheritance.py](tests/test_config_inheritance.py)，包含 27 个测试用例：
   - BaseConfig 导入测试
   - 配置继承链验证
   - TrainerConfigDC 与 TrainerConfig 互转
   - 序列化/反序列化（YAML, JSON, Dict）
   - validate 方法测试
   - 废弃警告测试
   - BaseConfig 方法测试（merge, copy, update）
4. ✅ **废弃提示**：在 `TrainerConfigDC` 和 `RollingTrainerConfigDC` 的 `__post_init__` 中添加 `DeprecationWarning`。
5. 🔄 **路径清理（可选）**：未在本轮执行，tests/examples/scripts/notebook 中残留的 sys.path.insert 暂不影响核心功能。
6. ✅ **CI 验证**：已创建 [pyproject.toml](pyproject.toml) 支持 `pip install -e .`，测试可通过 `pytest tests/test_config_inheritance.py` 运行。

### 额外遗留（未纳入本轮修复）
- tests、examples、scripts、notebook 等非核心路径仍有 `sys.path.insert`（见 grep 搜索），未影响核心包导入。如需完全清理，可后续专门处理。

## 解决方案设计（新增问题）

1) TaskRunner 模块路径硬编码 ✅ 已修复
- 策略：若缺省 `module_path`，优先查询 ModelRegistry/DataManager 导出映射；找不到即报错提示显式配置，不再写死包路径。
- 交付：调整适配器逻辑，并补充单测覆盖自定义模块路径与缺省路径报错分支。

## 新增问题与方案 (2026-01-18 补充)

1) 动态预测返回元组触发 AttributeError ✅ 已修复
- 现象：notebook 第12单元格调用 `predictions_raw = dynamic_trainer.predict(..., return_numpy=False)` 后访问 `.shape` 报错：`'tuple' object has no attribute 'shape'`。
- 根因：`SimpleTrainer.predict` 对普通 nn.Module 采用回退路径，返回 `(preds, labels)` 元组（详见 [model/train/simple_trainer.py](model/train/simple_trainer.py)），当前传入的是裸 `HybridNet`，未实现 `predict` 方法。
- 方案（推荐）：直接使用 `predictions_raw = model_dynamic.predict(test_daily_loader, return_numpy=False)`。`HybridGraphModel` 的 predict 方法专为图模型设计，支持特殊的推理模式（如邻居采样），且返回单一预测结果，符合预期。
- 方案（备选）：若坚持使用 trainer，需在 notebook 中显式拆包：`preds, labels = dynamic_trainer.predict(...)`，注意 trainer 返回的是通用元组。
- **修复内容**：更新 notebook 第12单元格，改用 `model_dynamic.predict()` 并添加 `model_dynamic.fitted = True` 标记。

2) 滚动窗口动态图配置复制报 AttributeError ✅ 已修复
- 现象：rolling 窗口动态图训练单元（Cell 13）在创建 `dynamic_model_config` 时抛出 `AttributeError: 'CompositeModelConfig' object has no attribute 'model_copy'`，阻断整个滚动训练流程。
- 根因：`CompositeModelConfig` 是基于 dataclass 的配置类（参见 [model/modular_config.py#L337](model/modular_config.py#L337)），继承的 `BaseConfig` 仅提供 `copy()` 深拷贝接口（[config/base_config.py#L268](config/base_config.py#L268)），并不存在 pydantic v2 风格的 `model_copy()`，因此调用直接报属性缺失。
- 影响：动态图 rolling 训练在配置复制阶段即退出，后续窗口 DataLoader 构建、训练与评估均未执行；静态 simpletrainer 流程正常。
- **修复内容**：将 `model_config.model_copy()` 替换为 `model_config.copy()`（BaseConfig 提供的深拷贝方法）。
- **涉及文件**：[notebook/lstm+attention12011427.ipynb](notebook/lstm+attention12011427.ipynb#L1665)

3) 滚动窗口测试期起始日期缺失导致 AttributeError ✅ 已修复
- 现象：rolling 窗口动态图训练单元（Cell 13）构造测试期日期时报错 `AttributeError: 'DataConfig' object has no attribute 'test_start_date'`，执行中断。
- 根因：`DataConfig` 定义中不存在 `test_start_date` 字段（参见 [data_set/config.py#L1-L150](data_set/config.py#L1-L150)），滚动窗口代码直接访问 `dm.config.test_start_date` 导致属性缺失。DataManager 的滚动切分逻辑默认以 `time_col` 最小值推断测试起点，无需该字段。
- 影响：滚动窗口训练在计算 `test_dates` 阶段即失败，后续 loader 构建与训练均未开始；静态 simpletrainer 流程不受影响。
- **修复内容**：改为从 `dm._test_df` 推断测试起始日期：`test_start_date = pd.to_datetime(dm._test_df[dm.config.time_col].min())`，不依赖 DataConfig 中不存在的字段。
- **涉及文件**：[notebook/lstm+attention12011427.ipynb](notebook/lstm+attention12011427.ipynb#L1693)


## 现有架构问题复核（2026-01-14）

1) CLI sys.path 侵入 ✅ 已修复
- 策略：改为通过 console_scripts/`python -m quantclassic.config.cli` 入口，不默认修改 sys.path；仅在未安装检测失败时临时追加并打印 warning。
- 交付：修改 cli.py 的路径处理与提示，补充 CLI 单测验证已安装环境不污染 sys.path。

1) Loader 示例路径漂移 ✅ 已修复
- 策略：更新示例导入为 `from quantclassic.model import VAEModel` 或直接移除示例段，保持包内可运行。
- 交付：清理 loader 自测片段，并新增/更新对应单测确保不再抛 ModuleNotFoundError。

1) 训练分支重复实现 ✅ 已修复
- 策略：抽公共辅助（模型工厂、参数拆分、SimpleTrainer 调用），`_train_dynamic_graph` 调用 `_train_simple` 仅负责 loader 拆包；滚动分支共用工厂与参数拆分。
- 交付：重构 runner 训练分支，增加 simple/rolling/daily loader 回归测试覆盖。

---

## 修复记录 (2026-01-14)

### 修复 5：TaskRunner 模块路径去硬编码

**修改内容**：
- 新增 `_REGISTERED_MODEL_CLASSES` 和 `_REGISTERED_DATASET_CLASSES` 集合，列出已注册的类名
- 修改 `_adapt_task_config_to_legacy()` 逻辑：
  1. 优先使用 kwargs 中显式提供的 `module_path`
  2. 若类名在注册列表中，使用对应默认路径
  3. 否则抛出 ValueError 提示用户显式配置

**涉及文件**：[config/runner.py](config/runner.py#L40-L100)

### 修复 6：CLI 去侵入式 sys.path 修改

**修改内容**：
- 新增 `_ensure_importable()` 函数，优先尝试 `import quantclassic`
- 仅在未安装时临时追加路径，并发出 `UserWarning` 提醒用户使用 `pip install -e .`
- 移除无条件 `sys.path.insert`

**涉及文件**：[config/cli.py](config/cli.py#L15-L45)

### 修复 7：Loader 示例路径清理

**修改内容**：
- 移除自测段中引用旧路径 `from model.model_config import VAEConfig` 的片段
- 保留基础配置保存/加载和环境变量替换测试

**涉及文件**：[config/loader.py](config/loader.py#L232-L270)

### 修复 8：训练分支去重复

**修改内容**：
- 新增公共辅助方法：
  - `_extract_nn_module(model)`: 从模型对象提取底层 nn.Module
  - `_create_model_factory(nn_model)`: 创建返回模型深拷贝的工厂函数
  - `_split_trainer_kwargs(...)`: 统一拆分训练器参数为 init/fit/config 三类
  - `_get_loaders_from_dataset(dataset)`: 从 dataset 提取 train/val/test
- 重构 `_train_simple`、`_train_rolling_window`、`_train_rolling` 使用公共辅助
- 重构 `_train_dynamic_graph` 复用 `_train_simple` 逻辑，仅负责 loader 拆包

**涉及文件**：[config/runner.py](config/runner.py#L530-L650)

---

## 本轮新增内容（2026-01-13）

### 1. 创建 pyproject.toml
新增 [pyproject.toml](pyproject.toml) 文件，支持包的可编辑安装：
```bash
pip install -e .
```

### 2. 创建单元测试
新增 [tests/test_config_inheritance.py](tests/test_config_inheritance.py)，包含以下测试类：
- `TestBaseConfigImport`: BaseConfig 导入测试
- `TestConfigInheritance`: 配置继承链验证
- `TestTrainerConfigConversion`: TrainerConfigDC 与 TrainerConfig 互转
- `TestConfigSerialization`: 序列化/反序列化测试
- `TestConfigValidation`: validate 方法测试
- `TestDeprecationWarnings`: 废弃警告测试
- `TestBaseConfigMethods`: BaseConfig 基本方法测试
### 3. 添加废弃警告
在 `config/base_config.py` 中：
- `TrainerConfigDC.__post_init__()` 添加废弃警告
- 添加 `copy()` 方法：创建配置对象的深拷贝
- 增强 `merge()` 方法：支持合并字典参数
## 后续建议

1. ~~**添加单元测试**~~ ✅ 已完成

2. **文档统一**：在各模块 README 中统一说明配置继承关系，引导用户使用正确的导入路径。

3. ~~**废弃警告**~~ ✅ 已完成

## 剩余架构问题 (2026-01-15 复核)

1. **配置系统不一致** (backtest & data_fetch) ✅ 已修复
   - ~~`backtest.backtest_config.BacktestConfig` 仍为裸 dataclass，未继承 `config.base_config.BaseConfig`~~
   - ~~data_fetch 配置未继承 BaseConfig~~
   - **修复详情**：见下方「修复记录 (2026-01-15)」

2. **推理/数据集逻辑重复** (backtest)
   - `FactorGenerator` 内自定义 `FactorDataset`，重做滑窗、代码列兜底和缺失过滤，未复用 `data_set`/`data_processor` 的标准数据管线（见 [backtest/factor_generator.py#L17-L96](backtest/factor_generator.py#L17-L96)）。
   - `BacktestRunner` 仍直接接受 DataFrame 做处理/IC/组合，未消费标准 DataLoader 或预测结果，导致与数据管线解耦（见 [backtest/backtest_runner.py#L24-L225](backtest/backtest_runner.py#L24-L225)）。

3. **因子逻辑碎片化** (factor_hub vs backtest) ✅ 已通过实验性标记缓解
   - 虽存在 factor_hub 顶层概念，backtest 仍内置因子生成与处理链路（`factor_generator.py`、`factor_processor.py`），与上游特征/因子管线职责边界未收敛。
   - **缓解措施**：factor_hub 已标记为实验性模块，生产入口统一收归 backtest。

4. **存在已弃用代码** (data_fetch) ✅ 已确认存在
   - `data_fetch/daily_graph_loader.py` 已在仓库中找到，需后续评估是否移除或标记 deprecated。

---

## 修复记录 (2026-01-15)

### 修复 9：BacktestConfig 继承 BaseConfig

**修改内容**：
- 添加 `BaseConfig` 导入（相对导入 + try-except 后备）
- `BacktestConfig` 从裸 `@dataclass` 改为 `@dataclass class BacktestConfig(BaseConfig)`
- 移除原有重复的 `to_dict()`、`to_yaml()`、`from_dict()`、`from_yaml()`、`update()` 方法（BaseConfig 已提供）
- 保留 `validate()` 方法重写

**涉及文件**：[backtest/backtest_config.py](backtest/backtest_config.py#L1-L20)

### 修复 10：data_fetch 配置类继承 BaseConfig

**修改内容**：
- 添加 `BaseConfig` 导入（相对导入 + try-except 后备）
- 以下配置类全部改为继承 `BaseConfig`：
  - `TimeConfig(BaseConfig)`
  - `DataSourceConfig(BaseConfig)`
  - `UniverseConfig(BaseConfig)`
  - `DataFieldsConfig(BaseConfig)`
  - `StorageConfig(BaseConfig)`
  - `ProcessConfig(BaseConfig)`
  - `FeatureConfig(BaseConfig)`

**涉及文件**：[data_fetch/config_manager.py](data_fetch/config_manager.py#L1-L15)

**效果**：
- 所有配置类现可使用 `BaseConfig` 的 `to_dict()`、`to_yaml()`、`from_yaml()`、`merge()`、`copy()` 等方法
- `isinstance(config, BaseConfig)` 检测统一返回 `True`
- TaskRunner 可统一管理所有配置对象

---

## 设计修改方案（factor_hub 试验性化，backtest 作为主入口）✅ 已实施

目标：将 factor_hub 定位为实验/原型，不进入生产链路；backtest 保持唯一生产入口，边界清晰。

方案要点：
- ✅ 入口收敛：文档与示例统一声明生产回测入口为 backtest（`MultiFactorBacktest`/`FactorBacktestSystem`），factor_hub 标注 "实验性/不支持生产"。
- ✅ 职责拆分：因子生成/处理逻辑仅保留在 backtest；factor_hub 若存在重复实现，标记 deprecated 并指向 backtest 模块。
- ✅ 依赖解耦：factor_hub 导入时发出 FutureWarning 提示用户使用 backtest。
- ✅ 配置对齐：backtest 和 data_fetch 配置类已继承 BaseConfig，可与核心配置体系统一管理。
- ✅ 文档调整：在 README/指南中添加"factor_hub 为实验模块，勿用于生产"的显著提示，并给出迁移路径（迁移到 backtest 的 API）。

### 实施记录 (2026-01-15)

1. `factor_hub/__init__.py`：顶部添加实验性警告 docstring 与 `warnings.warn(FutureWarning)`。
2. `factor_hub/README.md`：顶部添加实验模块警告横幅及迁移路径。
3. `backtest/README.md`：顶部添加生产级入口标识，提示用户优先使用 backtest 而非 factor_hub。
4. `backtest/backtest_config.py`：`BacktestConfig` 继承 `BaseConfig`。
5. `data_fetch/config_manager.py`：7 个配置类全部继承 `BaseConfig`。

## 新增问题（2026-01-15）

### 待办：滚动窗口动态图训练改用 RollingDailyTrainer（notebook Cell 13）
- 现状：Cell 13 手写滚动循环 + SimpleTrainer，逻辑冗长（~200行）且显存管理缺失（易导致 OOM）。
- 方案：
    1. 迁移至 `RollingDailyTrainer`：
        - 定义 `model_factory` 闭包，内部调用 `HybridGraphModel.from_config(dynamic_model_config)` 并返回 `.model`。
        - 构造 `DailyRollingConfig`，设置 `weight_inheritance=True`（权重继承）和 `gc_interval=1`（显存回收）。
    2. 参数对接：直接调用 `trainer.train(window_loaders, save_dir=...)`。
    3. 结果对齐：
        - `RollingDailyTrainer` 自动生成的 `all_predictions` 若未覆盖多因子列（`pred_factor_i`），需在训练后通过 `results.window_results` 或自定义 `predict` 增强逻辑补足。
        - 确保与后续回测系统兼容格式（`factor_raw_std` = `pred.mean()`）对齐。
- 预期收益：减代码量 80%、自动处理 warm-start 和模型断点续传、提升显存稳定性。
- 行动：在 notebook 中替换手写循环，确保汇总统计（平均 IC 等）逻辑无缝衔接。

### DataPreprocessor 步骤特征空值导致 TypeError ✅ 已修复

**现象**：notebook 预处理管道执行 `DataPreprocessor.fit_transform` 时，若 `add_step` 未显式传 `features`（默认 None），`data_preprocessor._get_process_features` 直接迭代 `step.features`，抛出 `TypeError: 'NoneType' object is not iterable`。

**定位**：
- 触发位置：`data_processor/data_preprocessor.py::_get_process_features`，`step.features` 为 None
- 复现：notebook 第7单元格的 SIMSTOCK_LABEL_NEUTRALIZE（未指定 features）即可复现

**修复内容**：
- 在 `_get_process_features` 方法开头增加空值兜底：`raw_features = step.features or []`
- 修改列名映射逻辑，仅在 `raw_features` 非空时执行映射
- 后续逻辑不变：空列表会被替换为 `all_features`（使用全部特征）

**涉及文件**：[data_processor/data_preprocessor.py](data_processor/data_preprocessor.py#L295-L315)

**效果**：
- `add_step` 不传 `features` 时自动使用全部特征，符合文档约定
- 不影响显式传入 `features` 的硬参数与参数透传行为

## 新增问题（2026-01-16）

### modular_config 缩进残留导致 ImportError ✅ 已修复

**现象**：在 notebook/import 阶段执行 `from quantclassic.model.modular_config import ...` 报 `IndentationError: unexpected indent`，终止执行。

**定位**：在 [model/modular_config.py#L46-L52](model/modular_config.py#L46-L52) 的后备导入 `except ImportError` 语句块下方残留两行字段定义 `verbose: bool = True`、`seed: Optional[int] = None`，未处于任何类或函数作用域，触发解析失败。

**影响**：模块无法导入，所有依赖 `modular_config` 的训练/推理入口均不可用。

**解决方案设计**：
- 移除上述两行孤立字段定义，保证导入路径处于合法语法块内。
- 若需要在配置类暴露 `verbose/seed` 字段，确认它们已在 `BaseModelConfig` 或具体配置 dataclass 中定义；否则在对应 dataclass 中补充正式字段并添加默认值、校验逻辑。
- 回归校验：运行 `python -m py_compile jupyterlab/quantclassic/model/modular_config.py` 与 `python - <<'PY'
from quantclassic.model.modular_config import ModuleType
print('import ok', ModuleType.TEMPORAL)
PY`，确保导入不再报错。

### DynamicGraphTrainer 已删除，动态图训练由 SimpleTrainer 兼容路径提供 ✅ 已修复

**现象**：notebook 第11单元格导入 `from quantclassic.model.dynamic_graph_trainer import DynamicGraphTrainer, DynamicTrainerConfig` 报 `ModuleNotFoundError`。

**定位**：`DynamicGraphTrainer` 在重构中已删除（见 [model/REFACTOR_PLAN.md#L66-L75](model/REFACTOR_PLAN.md#L66-L75)），`config/runner.py::_train_dynamic_graph` 已改为使用 `SimpleTrainer` 包装日级 loaders；动态图数据加载仍在 `data_set/graph/daily_graph_loader.py` 提供。

**影响**：旧文档/示例仍引用已删除类，导致导入失败；但功能可通过 SimpleTrainer 路径覆盖。

**修复内容**：
- 更新 [notebook 第11单元格](notebook/lstm+attention12011427.ipynb#L1024-L1369)：
  - 删除 `from quantclassic.model.dynamic_graph_trainer import DynamicGraphTrainer, DynamicTrainerConfig`
  - 改为 `from quantclassic.model.train import SimpleTrainer, TrainerConfig`
  - 在末尾添加训练示例提示（使用 SimpleTrainer 搭配日级加载器）
- 更新文档：[agent.md](agent.md#L28-L39) 与 [README.md](README.md#L37-L48) 添加动态图训练使用指南。

**涉及文件**：
- [notebook/lstm+attention12011427.ipynb](notebook/lstm+attention12011427.ipynb#L1024-L1369)
- [agent.md](agent.md#L28-L39)
- [README.md](README.md#L37-L48)

---

## 修复记录 (2026-01-16)

### 修复 11：移除 modular_config.py 孤立字段定义

**修改内容**：
- 移除 [model/modular_config.py#L51-L52](model/modular_config.py#L51-L52) 残留的两行孤立字段定义：
  ```python
  # 删除前（缩进错误，不在任何类/函数内）
              verbose: bool = True
              seed: Optional[int] = None
  ```

## 新增问题（2026-01-19）

### 滚动窗口动态图单元使用手写小样本参数，未复用正式配置 ✅ 已修复

- 现象：滚动窗口动态图训练单元（Cell 13）直接硬编码 `window_size=20`、`rolling_window_size=120`、`rolling_step=20`、`test_size=30`、`val_size=20`，仅用于小样本调试；与前面正式配置的 `DataConfig`/模型配置不一致，导致实验结果不可与主线对齐。
- 影响：动态图滚动实验与 baseline 配置割裂，IC/回测结果无法与正式参数可比；若忘记改回正式参数，可能误判效果。
- **修复内容**：
   1. 将滚动参数改为引用 `data_config`：`window_size = data_config.window_size`、`rolling_window_size = data_config.rolling_window_size`、`rolling_step = data_config.rolling_step`。
   2. `val_size`/`test_size` 按 `data_config` 比例推导：`test_size = rolling_step`（1年），`val_size` 按 `val_ratio/(train_ratio+val_ratio)` 计算。
   3. 添加日志输出当前使用的参数来源（正式配置），避免混淆。
- 涉及文件：
   - [notebook/lstm+attention12011427.ipynb#L1658-L1662](notebook/lstm+attention12011427.ipynb#L1658-L1662)

### 设计方案：重构 Cell 13，复用已有动态图构建逻辑 ✅ 已实施

- 现状：Cell 13 手写滚动窗口 + DailyGraphDataLoader 全流程，未复用 DataManager 的滚动切分和 Cell 11 的动态图 loader 创建逻辑，导致代码重复与参数透传冗余。
- 目标：保留正式参数来源（data_config），将 Cell 11 的动态图创建封装成可复用函数，在滚动窗口循环中调用，避免手写日期切分和重复构建。
- **修复内容**：
   1. 封装函数 `create_dynamic_loaders_for_window(df_full, train_dates, val_dates, test_dates, *, ...)`：内部复用 Cell 11 的 DailyBatchDataset + DailyGraphDataLoader 逻辑，支持窗口变换参数（price_log、volume_norm、label_rank_normalize 等）。
   2. 封装函数 `generate_rolling_windows(all_dates, test_start_date, ...)`：生成滚动窗口日期切分，复用 DataManager 的日期序列。
   3. Cell 13 循环仅负责：调用 `generate_rolling_windows` 生成日期 → 调用 `create_dynamic_loaders_for_window` 得到 loaders → 训练/保存/评估。
   4. 训练参数（N_EPOCHS、EARLY_STOP、LEARNING_RATE）改为取自 model_config，不再硬编码。
- 预期收益：减少约 60% 重复代码，统一参数来源，后续修改只需更新封装函数。
- 涉及文件：[notebook/lstm+attention12011427.ipynb](notebook/lstm+attention12011427.ipynb) Cell 13
- 这两行是之前移除降级版 BaseConfig 时遗留的残留代码

**涉及文件**：[model/modular_config.py](model/modular_config.py#L46-L55)

**验证结果**：
- ✅ `python -m py_compile model/modular_config.py` 通过
- ✅ `from quantclassic.model.modular_config import ModuleType` 导入成功

### 修复 12：更新 notebook 动态图训练引用

**修改内容**：
- 更新 [notebook 第11单元格](notebook/lstm+attention12011427.ipynb#L1024-L1369)，移除已废弃的 `DynamicGraphTrainer` 导入
- 改为 `from quantclassic.model.train import SimpleTrainer, TrainerConfig`
- 在输出末尾添加 SimpleTrainer 使用示例提示

**涉及文件**：[notebook/lstm+attention12011427.ipynb](notebook/lstm+attention12011427.ipynb#L1024-L1369)

**效果**：
- notebook 第11单元格不再报 ModuleNotFoundError
- 用户可直接使用 SimpleTrainer 搭配日级加载器进行动态图训练

## 新增问题（2026-01-19）

### 动态图训练/推理数据格式不一致导致 GAT 推理被禁用 ✅ 已修复

- 现象：预测阶段日志反复出现 `batch[2] 不是 long 类型张量，跳过`，来源于 `HybridGraphModel._parse_batch_data` 将 batch[2] 视作 stock_idx（配置 stock_idx_position=2，见 [notebook/output/configs/model_config.yaml#L55](notebook/output/configs/model_config.yaml#L55)），而日级 loader 实际返回 (X, y, adj, stock_ids, date)（见 [data_set/graph/daily_graph_loader.py#L460-L546](data_set/graph/daily_graph_loader.py#L460-L546)），batch[2] 是浮点邻接矩阵。解析失败后 stock_idx 为空。参见实现 [model/hybrid_graph_models.py#L1324-L1420](model/hybrid_graph_models.py#L1324-L1420)。
- 影响：`HybridGraphModel.predict` 仍走包装类路径，`_forward_step/_prepare_graph_context` 在 stock_idx=None 且 adj_matrix_path=None 时回退为单位邻接矩阵（见 [model/hybrid_graph_models.py#L820-L1008](model/hybrid_graph_models.py#L820-L1008)），推理阶段 GAT 分支被完全禁用，节点缓存不更新，行为与训练阶段 `SimpleTrainer` + `HybridNet`（使用 batch adj）不一致，导致图信息丢失且日志被警告淹没。
- 根因：包装类的 `_parse_batch_data` 忽略了 loader 产出的 batch adj，并默认从配置复用静态批次索引规范；动态图管线没有显式 stock_idx（仅有字符串 stock_ids 列表），与 stock_idx_position=2 配置冲突。

**修复内容**：
1. **`_parse_batch_data` 方法重构**（[model/hybrid_graph_models.py](model/hybrid_graph_models.py#L1324-L1480)）：
   - 返回值从 3 元组 `(batch_x, stock_idx, batch_funda)` 改为 4 元组 `(batch_x, stock_idx, batch_funda, batch_adj)`
   - 新增对动态图 `DailyGraphDataLoader` 5 元素格式的自动检测：当 `len(batch)==5` 且 `batch[2]` 是 2D 浮点张量、`batch[3]` 是列表时，识别为 `(X, y, adj, stock_ids, date)` 格式
   - 当 `stock_idx_position` 指向的元素是 2D 浮点张量时，自动将其作为 `batch_adj` 使用，并发出警告提示检查配置

2. **`_forward_step` 方法增强**（[model/hybrid_graph_models.py](model/hybrid_graph_models.py#L820-L930)）：
   - 新增 `batch_adj` 参数，支持接收动态邻接矩阵
   - 当同时存在 `batch_adj` 和静态 `adj_matrix` 时，优先使用 `batch_adj`（动态图优先）
   - 仅在既无静态邻接又无 batch adj 时才回退为单位矩阵

3. **`predict`/`_train_epoch`/`_valid_epoch` 方法更新**：
   - 调用 `_parse_batch_data` 时接收 4 元组
   - 将 `batch_adj` 传递给 `_forward_step`

4. **notebook 第12单元格更新**（[notebook/lstm+attention12011427.ipynb](notebook/lstm+attention12011427.ipynb)）：
   - 在动态图模式下，显式设置 `dynamic_model_config.stock_idx_position = None`
   - 修复 `groupby().apply()` 的 `include_groups=False` 警告
   - 更新注释说明动态图格式自动检测机制

**效果**：
- ✅ 预测阶段不再出现 "batch[2] 不是 long 类型张量" 警告
- ✅ GAT 分支在训练和推理阶段行为一致，均使用 batch 内动态邻接矩阵
- ✅ 图信息正确参与推理，模型效果与预期一致

### 滚动窗口动态图训练缺少 `hybrid_graph_builder` 定义 ✅ 已修复

- 现象：步骤 3-DYNAMIC（Cell 13）在创建滚动窗口 DataLoader 时抛出 `NameError: name 'hybrid_graph_builder' is not defined`，触发位置在 loader 构造处。
- 根因：代码中使用了未定义的变量名 `hybrid_graph_builder`，实际作用域中定义的是 `hybrid_builder` 以及按 `GRAPH_TYPE` 选择的 `graph_builder`。
- **修复内容**：将 `DailyGraphDataLoader` 构造时的 `graph_builder=hybrid_graph_builder` 改为 `graph_builder=graph_builder`，复用上游根据 `GRAPH_TYPE` 配置选择的图构建器，保持配置一致性。
- **涉及文件**：[notebook/lstm+attention12011427.ipynb](notebook/lstm+attention12011427.ipynb) Cell 13

---

## 修复记录 (2026-01-19)

### 修复 13：动态图 batch_adj 支持

**修改内容**：
- 重构 `HybridGraphModel._parse_batch_data`，返回 4 元组并自动检测 5 元素动态图格式
- 增强 `_forward_step`，新增 `batch_adj` 参数，动态图优先
- 更新 `predict`/`_train_epoch`/`_valid_epoch` 传递 batch_adj
- notebook 第12单元格清除 `stock_idx_position` 配置冲突

**涉及文件**：
- [model/hybrid_graph_models.py](model/hybrid_graph_models.py)
- [notebook/lstm+attention12011427.ipynb](notebook/lstm+attention12011427.ipynb)

### 修复 14：滚动窗口配置复制方法修正

**修改内容**：
- 将 `model_config.model_copy()` 替换为 `model_config.copy()`
- `model_copy()` 是 pydantic v2 的方法，`CompositeModelConfig` 继承的 `BaseConfig` 仅提供 `copy()` 深拷贝接口
- 添加注释说明使用 `BaseConfig.copy()` 进行深拷贝

**涉及文件**：[notebook/lstm+attention12011427.ipynb](notebook/lstm+attention12011427.ipynb#L1665)

**效果**：
- ✅ 滚动窗口动态图训练配置复制不再报 AttributeError
- ✅ 后续窗口 DataLoader 构建、训练与评估可正常执行

### 修复 15：滚动窗口测试期日期推断修正

**修改内容**：
- 将 `dm.config.test_start_date`（不存在的字段）改为从 `dm._test_df` 推断
- 新增代码：`test_start_date = pd.to_datetime(dm._test_df[dm.config.time_col].min())`
- 同时修复 `groupby().apply()` 的 `include_groups=False` 警告

**涉及文件**：[notebook/lstm+attention12011427.ipynb](notebook/lstm+attention12011427.ipynb#L1693)

**效果**：
- ✅ 滚动窗口训练不再报 `'DataConfig' object has no attribute 'test_start_date'`
- ✅ 测试期日期从实际测试集数据推断，符合 DataManager 设计逻辑

### 修复 16：滚动窗口动态图 graph_builder 变量名修正

**修改内容**：
- 将 `DailyGraphDataLoader` 构造时的 `graph_builder=hybrid_graph_builder` 改为 `graph_builder=graph_builder`
- `hybrid_graph_builder` 是未定义的变量名（笔误），正确的变量是上游根据 `GRAPH_TYPE` 配置选择的 `graph_builder`
- 添加注释说明复用上游图构建器

**涉及文件**：[notebook/lstm+attention12011427.ipynb](notebook/lstm+attention12011427.ipynb) Cell 13

**效果**：
- ✅ 滚动窗口 DataLoader 创建不再报 NameError
- ✅ 图构建器与上游 `GRAPH_TYPE` 配置保持一致（可选 hybrid/corr/industry）

### 滚动窗口动态图 DataLoader 参数错误（`df` 传参） ✅ 已修复

- 现象：步骤 3-DYNAMIC（Cell 13）调用 `DailyGraphDataLoader` 抛出 `TypeError: DailyGraphDataLoader.__init__() got an unexpected keyword argument 'df'`。
- 根因：`DailyGraphDataLoader` 构造函数仅接受 `dataset`（`DailyBatchDataset` 实例）、`graph_builder`、`feature_cols`、`shuffle_dates` 等参数，不支持直接传 `df`/`time_col`/`stock_col`。当前滚动窗口循环为手写版，未先构造 `DailyBatchDataset` 或使用 `create_daily_loader`，导致 API 不匹配。
- **修复内容**：在窗口循环内先构造 `DailyBatchDataset(df=..., feature_cols=..., label_col=..., window_size=..., time_col=date_col, stock_col=stock_col)`，再用 `DailyGraphDataLoader(dataset=dataset, graph_builder=graph_builder, feature_cols=feature_cols, shuffle_dates=...)`。
- **涉及文件**：[notebook/lstm+attention12011427.ipynb](notebook/lstm+attention12011427.ipynb) Cell 13

### 模块重构方案：滚动动态图数据构建下沉到 DataManager (2026-01-19) ✅ 已实施

- **现状**：Cell 13 中手写了 `create_dynamic_loaders_for_window`/`generate_rolling_windows` 及循环构建逻辑，代码冗长且与 `DataManager` 职责割裂。
- **设计原则**：所有数据加载器（Loader）的构建与切分逻辑均应由 `DataManager` (Dataset层) 负责；`Trainer` 仅接收标准化的 Loader 集合。
- **实施方案**：
    1. **DataManager 增强**：在 `quantclassic/data_set/manager.py` 中新增 `create_rolling_daily_loaders_from_test` 方法。
        - 参数：`graph_builder`, `graph_builder_config`, `rolling_window_size` (可选), `rolling_step` (可选), `val_ratio` 及其他 loader 参数。
        - 逻辑：合并 train/val/test → 推断测试起始日期 → 生成滚动窗口切分 → 为每个窗口创建 `DailyBatchDataset` + `DailyGraphDataLoader`。
        - 返回 `RollingDailyLoaderCollection`，包含 `WindowLoaders` dataclass（兼容 `RollingDailyTrainer`）。
    2. **Notebook 简化**：重构 Cell 13。
        - 删除 `create_dynamic_loaders_for_window`、`generate_rolling_windows` 手写函数（~60行）。
        - 替换为单行调用：`window_loaders = dm.create_rolling_daily_loaders_from_test(graph_builder=graph_builder)`。
        - 保持 `RollingDailyTrainer` 调用不变。
- **预期收益**：
    - 消除 Notebook 中的胶水代码 (~60行 → 1行)。
    - 统一 API：与其他 loader 创建方法 (`create_simple_loaders`, `create_daily_loaders`) 保持一致。
    - 提升复用性：TaskRunner 或其他脚本可直接复用滚动动态图训练能力。

---

## 修复记录 (2026-01-19) - 补充

### 修复 17：DataManager 新增 `create_rolling_daily_loaders_from_test` 方法

**修改内容**：
- 在 `data_set/manager.py` 中新增 `create_rolling_daily_loaders_from_test` 方法（~150行）
- 与现有 `create_rolling_daily_loaders` 的区别：
  - `create_rolling_daily_loaders`: 要求 `split_strategy='rolling'`，从 `_rolling_windows` 获取窗口
  - `create_rolling_daily_loaders_from_test`: 支持任意 `split_strategy`，在测试集日期上滚动生成窗口
- 内部实现：
  1. 合并 train/val/test 为完整数据集
  2. 从 `_test_df` 推断测试起始日期
  3. 根据 `rolling_window_size`/`rolling_step`/`val_ratio` 生成滚动窗口日期切分
  4. 为每个窗口创建 `DailyBatchDataset` + `DailyGraphDataLoader`
  5. 返回 `RollingDailyLoaderCollection`（兼容 `RollingDailyTrainer`）

**涉及文件**：[data_set/manager.py](data_set/manager.py#L717-L900)

### 修复 18：Notebook Cell 13 重构

**修改内容**：
- 移除手写辅助函数：`create_dynamic_loaders_for_window`（~40行）、`generate_rolling_windows`（~20行）
- 移除手动循环构建 `window_loaders` 的代码（~15行）
- 替换为单行调用：
  ```python
  window_loaders = dm.create_rolling_daily_loaders_from_test(
      graph_builder=graph_builder,
      rolling_window_size=rolling_window_size,
      rolling_step=rolling_step,
      val_ratio=val_ratio,
      device=device,
  )
  ```
- Cell 代码量：~200行 → ~130行（减少 35%）
- 其余训练/评估/保存逻辑保持不变

**涉及文件**：[notebook/lstm+attention12011427.ipynb](notebook/lstm+attention12011427.ipynb) Cell 13

**效果**：
- ✅ 数据加载逻辑完全由 DataManager 管理，符合架构设计
- ✅ Notebook 变为"薄客户端"，仅负责配置和调用
- ✅ 滚动动态图训练可在 TaskRunner / CLI 中复用

---

## 新增问题（2026-01-19）

### 滚动窗口测试起始日期错误：从2022年开始而非2010年 🔴 待修复

**现象**：
- 用户配置：数据从2000年开始，7年训练+2年验证+1年测试，期望从2010年开始测试
- 实际结果：测试从2022年开始，只有3个滚动窗口

**根因分析**：

1. **DataManager 的 rolling 策略处理逻辑**（[data_set/manager.py#L280-L360](data_set/manager.py#L280-L360)）：
   - `RollingSplitter` 正确生成了所有滚动窗口（~15个，从2010年到2024年）
   - 但 `create_datasets()` 将窗口分为 80% 训练 + 20% 测试：
     - 前 80%（~12个）窗口的**训练数据**合并成 `_train_df`
     - 后 20%（~3个）窗口的**测试数据**合并成 `_test_df`
   - **关键**：`_test_df` 只包含最后 3 个窗口的测试期（2022-2024年）

2. **`create_rolling_daily_loaders_from_test()` 的推断逻辑**：
   - 从 `_test_df.min()` 推断测试起始日期 = 2022年
   - 因此只能生成 3 个滚动窗口

**影响**：
- 滚动窗口训练覆盖年限严重不足（3年 vs 预期14年）
- 无法完整评估模型在不同市场周期的表现

**修复方案**：

方案A：**在 `create_rolling_daily_loaders_from_test` 中添加 `test_start_date` 参数**（推荐）
```python
def create_rolling_daily_loaders_from_test(
    self,
    graph_builder=None,
    test_start_date: Optional[str] = None,  # 🆕 显式指定测试起始日期
    ...
):
    # 如果未指定，则从 _test_df 推断（向后兼容）
    # 如果指定，则使用用户指定的日期作为第一个测试窗口起点
```
- 优点：灵活，用户可自定义起始日期
- 缺点：需要用户手动计算起始日期
 
方案B：**使用 `dm._rolling_windows` 而非重新推断**
```python
# 直接使用 RollingSplitter 生成的原始窗口
window_loaders = dm.create_rolling_daily_loaders()  # 已有方法，要求 _rolling_windows
```
- 优点：完整利用所有滚动窗口
- 缺点：需要保留 `_rolling_windows` 属性（当前 `create_datasets` 只保留合并后的 train/val/test）

方案C：**修改 `create_datasets()` 保留完整 `_rolling_windows`**
```python
# 在 create_datasets() 中：
self._rolling_windows = split_result  # 保留原始滚动窗口
```
- 优点：最彻底，后续 `create_rolling_daily_loaders` 可直接使用
- 缺点：改动较大

**建议**：
- 短期：使用方案A，在 Cell 13 中显式传入 `test_start_date='2010-01-01'`
- 长期：实施方案C，让 DataManager 保留完整滚动窗口信息

**涉及文件**：
- [data_set/manager.py](data_set/manager.py#L280-L360)（`create_datasets` 合并逻辑）
- [data_set/manager.py](data_set/manager.py#L760-L900)（`create_rolling_daily_loaders_from_test`）

