# QuantClassic Config 模块

面向量化研究的 YAML 配置与一键运行入口，负责把“配置”映射为可执行的数据管线、模型和回测任务。

## 功能一览
- 配置驱动：支持 YAML 配置、继承 (`BASE_CONFIG_PATH`)、环境变量占位 (`${VAR:default}`) 与保存/导出。
- 自动装配：通过 `TaskRunner` 将配置解析为数据集、模型、训练器、回测器，默认记录实验信息。
- CLI 与 SDK：命令行 `qcrun` / `python -m quantclassic.config.cli`，或在 Python 中调用 `ConfigLoader` + `TaskRunner`。
- 模板复用：内置 LSTM/GRU/Transformer/VAE 等模板，便于快速试验与批量搜索。
- 验证与报错：加载时做字段校验、模块路径检查，明确提示缺失字段或导入失败原因。

## 目录结构
```
config/
├── base_config.py        # 配置基类与通用字段 (继承/更新/校验)
├── loader.py             # ConfigLoader，负责加载/保存/验证配置
├── runner.py             # TaskRunner，按配置实例化 dataset/model/backtest 并执行
├── cli.py                # CLI 入口，注册为命令 qcrun
├── utils.py              # 模块加载、环境变量替换、路径处理工具
├── templates/            # 官方模板 (lstm/gru/transformer/vae 等)
├── examples/             # 完整示例与自定义脚本 (rolling_hybrid_graph, vae_* 等)
├── tests/                # 单元测试 (加载、Runner 修复回归等)
├── QUICKSTART.md         # 快速上手
├── RUN_GUIDE.md          # 运行指南与常见参数
└── BEFORE_AFTER.md       # 重构前后对比与迁移说明
```

## 关键组件与依赖
- `ConfigLoader`：基于 `yaml` 解析，支持
  - 继承与合并：`BASE_CONFIG_PATH` 链式覆盖
  - 环境变量：`${VAR}` / `${VAR:default}` 替换
  - 验证：必需字段与模块可导性检查
- `TaskRunner`：依赖内部 `import_module` 工具按 `class` + `module_path` 动态实例化；串联训练/回测并可返回结果字典。
- `cli.py`：封装 `TaskRunner.run`，解析命令行参数并打印友好日志。
- 基础依赖：`pyyaml`、`importlib`、`logging`，其余模型/数据依赖由下游模块提供。

## 与其他模块的联系
- data_set：`task.dataset` 通常指向 `quantclassic.data_set.manager.DataManager`，Runner 将配置透传为数据加载、划分和 DataLoader 构建。
- data_processor：可在 DataManager 之前单独运行；若在配置中指定预处理，需保证列名与 `PreprocessConfig` 对齐。
- model：`task.model` 对应 `quantclassic.model` 下的注册模型或自定义模块路径；Runner 负责调用其 `fit/predict` 接口。
- workflow/记录：Runner 可选择写入日志或实验目录（取决于调用方配置），以便复现与对比。
- backtest：如果提供 `task.backtest` 字段，会实例化回测系统（如 `Factorsystem`），消费预测结果与数据切分。

## 快速开始
### 1) CLI 运行
```bash
qcrun config/templates/lstm_basic.yaml
# 或
python -m quantclassic.config.cli my_config.yaml
```

### 2) Python 调用
```python
from quantclassic.config import ConfigLoader, TaskRunner

cfg = ConfigLoader.load('config.yaml')
runner = TaskRunner()
results = runner.run(cfg, experiment_name='demo')

model = results['model']           # 已训练模型
dataset = results['dataset']       # DataManager 或 DatasetCollection
train_results = results['train_results']
```

### 3) 自定义模块路径
```yaml
task:
  model:
    class: MyModel
    module_path: my_pkg.models
    kwargs: {hidden_size: 128}
```

## 配置片段速览
```yaml
BASE_CONFIG_PATH: base_config.yaml        # 可选继承
experiment_name: lstm_experiment          # 可选实验名

task:
  model:
    class: LSTM
    module_path: quantclassic.model.pytorch_models
    kwargs:
      d_feat: 20
      hidden_size: 64
      n_epochs: 100
  dataset:
    class: DataManager
    module_path: quantclassic.data_set.manager
    kwargs:
      config:
        base_dir: rq_data_parquet
        window_size: 20
  backtest:
    n_groups: 10
    save_plots: true
    output_dir: output/backtest
```

## 常见问题
- 模块导入失败：确认 `module_path` 正确且在 `PYTHONPATH` 中。
- 配置缺字段：`ConfigLoader.validate` 会提示缺失的 `task/model/dataset` 字段。
- 环境变量未替换：确保写成 `${VAR}` 或 `${VAR:default}`，并在运行环境设置相应变量。

## 2026-01 行为更新
- TaskRunner 不再默认将 `task.model`/`task.dataset` 写死到 `quantclassic.*`，优先使用 `kwargs.module_path`，其次查询已注册类名，否则直接抛出缺少模块路径的异常，建议自定义类显式提供 `module_path` 或在包入口完成注册。
- CLI (`qcrun`/`python -m quantclassic.config.cli`) 仅在检测到 `quantclassic` 未安装时临时追加仓库路径并给出警告，推荐先执行 `pip install -e .`，避免全局 `sys.path` 污染。
- ConfigLoader 自检示例已改用 `quantclassic.model` 导出的类，移除旧的 `model.model_config` 引用，确保包内运行不再报 `ModuleNotFoundError`。
- Runner 训练分支抽公共助手（模型工厂/参数拆分/loader 提取），`_train_dynamic_graph` 复用 `_train_simple`，滚动训练与单窗口保持一致的参数处理，减少分支漂移风险。

## 参考
- 模板目录：`config/templates/`
- 示例脚本：`config/examples/`
- 数据/模型说明：参见 `data_set/README.md`、`model/README.md`
