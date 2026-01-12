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

- [ ] 运行 `python -c "from quantclassic.config import BaseConfig, TaskConfig"` 验证导入
- [ ] 运行 `python -c "from quantclassic.data_set import DataConfig; print(DataConfig.__bases__)"` 验证继承（⚠️ 受 `torch` 依赖影响，需安装后再测）
- [ ] 运行 `python -c "from quantclassic.data_processor import PreprocessConfig; print(PreprocessConfig.__bases__)"` 验证继承
- [ ] 运行 `python -c "from quantclassic.model.train import TrainerConfig; print(TrainerConfig.__bases__)"` 验证继承
- [ ] 运行 `python -c "from quantclassic.config.base_config import BaseConfig; from quantclassic.data_set.config import DataConfig; print(isinstance(DataConfig(), BaseConfig))"` 验证 isinstance

### 当前验证状态（2026-01-12）
- 核心配置文件（config/base_config.py、config/loader.py、data_set/config.py、data_processor/preprocess_config.py、model/model_config.py）可直接导入并通过继承检查。
- data_set/__init__.py 链路依赖 torch，当前环境缺少 torch 导致完整验证被阻塞；安装 torch 后可完成 Checklist 中剩余两项。
- pandas 报告 bottleneck 版本警告（1.3.5），与本次配置继承无关，可按需升级。

## 实施方案（torch 2.1.1 环境）

1. **环境准备**：使用带 pytorch-2.1.1 环境（含 GPU 时匹配对应 CUDA 版本），确保能导入 data_set/__init__.py。
2. **完成验证清单**：按“验证清单”顺序执行 5 条命令，确认继承链与 isinstance 结果一致。
3. **补充单测**：新增 tests/test_config_inheritance.py 覆盖 BaseConfig 继承、TrainerConfigDC 与 TrainerConfig 互转、序列化/反序列化、validate；使用最小 dummy 模型/损失以触发 scheduler/loss 分支。
4. **废弃提示**：在 TrainerConfigDC 的 __init__/validate 中增加迁移警告，提醒改用 model.train.TrainerConfig；同步更新 README/文档说明。
5. **路径清理（可选）**：清理 tests/examples/scripts/notebook 中残留的 sys.path.insert，统一包导入方式。
6. **CI 验证（建议）**：新增工作流（Python 3.x + torch==2.1.1）执行 `pip install -e .`、验证导入、运行 `pytest tests/test_config_inheritance.py`。

### 额外遗留（未纳入本轮修复）
- tests、examples、scripts、notebook 等非核心路径仍有 `sys.path.insert`（见 grep 搜索），未影响核心包导入。如需完全清理，可后续专门处理。

---

## 后续建议

1. **添加单元测试**：为配置继承和序列化添加回归测试，确保所有子配置类的 `to_dict()`、`from_dict()`、`to_yaml()`、`from_yaml()` 方法正常工作。

2. **文档统一**：在各模块 README 中统一说明配置继承关系，引导用户使用正确的导入路径。

3. **废弃警告**：考虑在 `TrainerConfigDC` 中添加废弃警告，引导用户迁移到 `model.train.TrainerConfig`。
