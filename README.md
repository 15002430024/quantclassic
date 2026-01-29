# QuantClassic

配置驱动的端到端量化研究与回测流水线，生产可用模块：config（编排）、data_processor（预处理）、data_set（数据集与 DataLoader）、model（模型与训练引擎）、backtest（因子/策略回测）。

## 适用范围
- 单一数据源、YAML/CLI/SDK 驱动的因子/时序/图模型训练与回测。
- 需要可追溯的实验产物：特征列、预处理器、模型、预测、回测指标与图表。
- 生产就绪模块：config · data_processor · data_set · model · backtest。其余目录（data_fetch/factor_hub 等）为辅助或实验性，不作生产承诺。

## 快速开始
1) 安装（建议虚拟环境）：
```bash
pip install -e .
```

2) 使用模板配置直接跑通（默认记录到 workflow/output）：
```bash
qcrun config/templates/lstm_basic.yaml
# 或
python -m quantclassic.config.cli config/templates/lstm_basic.yaml
```

3) Python SDK 示例：
```python
from quantclassic.config import ConfigLoader, TaskRunner

cfg = ConfigLoader.load("config/templates/lstm_basic.yaml")
runner = TaskRunner()
results = runner.run(cfg, experiment_name="demo")

model = results["model"]
dataloaders = results["dataset"]
backtest_results = results.get("backtest_results")
```

## 运行链路速览
- 配置与编排：config/TaskRunner 解析 YAML/Dict，初始化 dataset、model，并可选调用 backtest；全过程写入 workflow 记录。
- 预处理：data_processor/DataPreprocessor 执行去极值、缺失填充、标准化/中性化，支持状态持久化。
- 数据集：data_set/DataManager 负责特征选择、划分（时间/滚动/随机/分层）、Dataset/DataLoader 构建，支持日级图数据。
- 模型与训练：model 模块提供 LSTM/GRU/Transformer/VAE/混合图模型，统一 fit/predict，包含滚动训练器；动态图训练使用 SimpleTrainer + 日级加载器（`data_set/graph/daily_graph_loader.py`），旧的 `DynamicGraphTrainer` 已移除（见 plan.md）。
- 回测：backtest 提供因子生成/适配、IC/分组/绩效评估与可视化，可消费模型预测或外部因子。

### 动态图训练如何调用
- 数据加载：`DailyBatchDataset` + `DailyGraphDataLoader`（行业/相关性/混合图，按日动态 batch）。
- 训练入口（推荐）：`TaskRunner` 配置 `trainer_class: "DynamicGraphTrainer"`，内部走 SimpleTrainer 兼容路径自动消费日级 loaders。
- 训练入口（直接调用）：`from quantclassic.model.train import SimpleTrainer, TrainerConfig`，用 `train_daily_loader/val_daily_loader` 调用 `SimpleTrainer(...).fit(...)`，预测用 `trainer.predict(test_loader)`。
- 图构建器：位于 `data_processor/graph_builder.py`（`IndustryGraphBuilder` / `CorrGraphBuilder` / `HybridGraphBuilder`）。

数据流：特征数据 → data_processor → data_set → model 训练/预测 → backtest → workflow/output。

## 目录导航（生产模块）
- config：配置体系与编排入口，见 [config/README.md](config/README.md)
- data_processor：预处理与中性化，见 [data_processor/README.md](data_processor/README.md)
- data_set：数据加载/划分/Loader 构建，见 [data_set/README.md](data_set/README.md)
- model：模型与训练引擎，见 [model/README.md](model/README.md)
- backtest：因子/策略回测，见 [backtest/README.md](backtest/README.md)

## 输出与缓存
- output/experiments：workflow 记录的参数、指标、对象（模型、结果）。
- output/backtest：回测指标、分组/IC 结果、图表与可选 Excel。
- cache/：数据管线/特征/拆分等缓存（由各模块管理）。

## 深入阅读
- 架构与数据流：见 [ARCHITECTURE.md](ARCHITECTURE.md)
- 运行指南与模板：config 目录下的 QUICKSTART/RUN_GUIDE/模板 YAML
- 回测细节与示例：backtest/update_readme、example_*.py

## 反馈
如需补充数据源、调整模块边界或新增训练/回测场景，请先在对应模块 README 查阅接口/约束，再反馈需求。