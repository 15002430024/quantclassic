# DataManager 数据集模块

面向量化研究的数据工程套件，覆盖数据加载、质量验证、特征筛选、数据划分、Dataset/DataLoader 构建，以及日级图数据管线。

## 模块结构与职责

```
概述
data_set/
├── __init__.py                 # 模块入口：统一导出核心配置、管理器及数据集类
├── config.py                   # 配置中心：定义数据架构（窗口、列名、切分比例）与预设模板
├── loader.py                   # 数据载入：实现多格式高效读取、分块处理与内存类型自动优化
├── feature_engineer.py         # 特征管线：执行自动特征提取、统计分析与低质量特征过滤
├── splitter.py                 # 划分策略：提供时间序列、股票分层及滚动窗口等多种切分方案
├── validator.py                # 质量守门员：执行异常值监控与时序单调性校验，生成验证报告
├── factory.py                  # 数据工厂：创建 PyTorch Dataset/DataLoader，支持窗口级实时变换
├── manager.py                  # 核心中枢：封装完整数据管线（载入->验证->特征->划分->工厂）
├── graph/                      # 图数据扩展：提供适配动态图训练的日级批次加载器与构建接口
├── examples.py / quickstart.py # 开发者指南：提供从基础流水线到高级自定义场景的示例参考
└── update_readme/              # 更新日志：记录模块演进、补丁修复与核心逻辑变更
```
```
类关系
data_set/
├── __init__.py                 # 统一导出
├── config.py                   # DataConfig, ConfigTemplates (继承 config.base_config.BaseConfig)
├── loader.py                   # DataLoaderEngine：Parquet/CSV/HDF5 加载 + dtype 优化
├── feature_engineer.py         # FeatureEngineer：自动/手动特征列、过滤、统计
├── splitter.py                 # DataSplitter 抽象基类 + TimeSeries/Stratified/Rolling/Random 实现
├── validator.py                # DataValidator + ValidationReport：缺失/时序/异常值/样本数检查
├── factory.py                  # DatasetFactory：TimeSeriesStockDataset(+日期)、CrossSectionalBatchSampler、InferenceDataset
├── manager.py                  # DataManager：编排全流程 + LoaderCollection / 日批次接口
├── graph/daily_graph_loader.py # DailyBatchDataset/DailyGraphDataLoader（动态图训练）
├── examples.py / quickstart.py # 示例脚本
└── update_readme/              # 变更记录与补丁说明
```

### 继承关系与接口契约
- 配置：`DataConfig` → `BaseConfig`，提供 `from_yaml/to_yaml`、`update` 等通用方法。
- 划分：`DataSplitter` 抽象类 → `TimeSeriesSplitter` / `StratifiedStockSplitter` / `RollingWindowSplitter` / `RandomSplitter`，由 `create_splitter` 工厂分发。
- 数据集：
  - `TimeSeriesStockDatasetWithDate`（别名 `TimeSeriesStockDataset`）继承 `torch.utils.data.Dataset`，支持窗口级变换、标签排名归一化、返回日期/股票 ID；`CrossSectionalBatchSampler` 继承 `Sampler` 保障同日批次（IC/相关性损失场景）。
  - `InferenceDataset` 用于无标签推理；`DatasetCollection` / `LoaderCollection` 为轻量 dataclass 聚合。
- 管理：`DataManager` 持有 `DataLoaderEngine`、`FeatureEngineer`、`DataValidator`、`DatasetFactory` 与 `DataSplitter`，负责状态缓存与管线编排。

### 核心依赖
- 基础：pandas、numpy、torch、yaml、logging。
- 内部：`config.base_config.BaseConfig`（配置基础类）、`quantclassic.data_processor.graph_builder`（动态图构建）、`quantclassic.model.train`（下游训练器使用的 DataLoader 接口）。

## 管线与协同逻辑

1) `load_raw_data` → `DataLoaderEngine` 按格式加载并做 dtype 优化、基础列校验。  
2) `validate_data_quality` → `DataValidator` 检查必需列、缺失率、时序连续性、异常值、每股样本数，生成 `ValidationReport`。  
3) `preprocess_features` → `FeatureEngineer` 自动/自定义特征列、统计、过滤（低方差/高缺失/高相关），可缓存到 `output/feature_columns.txt`。  
4) `create_datasets` → 由 `create_splitter` 生成划分器：
   - `time_series` / `stratified` / `random` 产出 train/val/test。
   - `rolling` 生成多个窗口并自动拼接 train、扩展 test 以保留窗口历史，记录 `test_valid_label_start_date` 防止泄漏。
   - 所有分支最终交给 `DatasetFactory` 生成 `TimeSeriesStockDataset`，支持窗口级价格对数变换、成交量窗口标准化、标签窗口排名归一化。
5) `get_dataloaders` → 生成 `LoaderCollection`；可切换 `use_cross_sectional=True` 启用截面采样器，确保 batch 内同一交易日。
6) 日级动态图（可选）：
   - `create_daily_loaders`：按日聚合 `DailyBatchDataset`，可注入 `GraphBuilderFactory` 构建行业/相关/混合图，返回日级 `DailyGraphDataLoader` 三元组。
   - `create_rolling_daily_loaders`：保留滚动窗口独立性，便于 Walk-Forward/滚动训练。

与模型模块的协同：
- 常规训练：`LoaderCollection` 直接馈入 `model.PyTorchModel`/`train.SimpleTrainer`。
- 图/日级训练：`DailyGraphDataLoader` 对接 `model.hybrid_graph_models` 或 `RollingDailyTrainer`。
- 推理/回测：`InferenceDataset` 用于生成因子，输出可与回测系统共享。

## 快速上手

```python
from data_set import DataManager, DataConfig

config = DataConfig(base_dir='rq_data_parquet', data_file='train_data_final.parquet')
manager = DataManager(config)
loaders = manager.run_full_pipeline()  # 常规批次

# 截面批次（IC/相关性损失场景）
loaders_cs = manager.get_dataloaders(use_cross_sectional=True)

# 日级动态图批次（需 graph_builder 配置）
# daily_loaders = manager.create_daily_loaders(graph_builder_config={'type': 'hybrid', 'top_k': 10})
```

## 关键设计要点
- 缓存与可重复性：特征列表、统计、状态可持久化到 `output/` 与 `cache/`，`manager.save_state/load_state` 支持复现同一拆分与特征集。
- 防泄漏策略：
  - 划分前过滤标签缺失行；滚动/测试集附加历史 lookback 并记录 `valid_label_start_date`。
  - 截面采样保证 batch 内同日，避免 IC Loss 计算跨日排序。
- 窗口级变换：在 `Dataset.__getitem__` 内完成价格对数与成交量标准化，确保每个窗口使用自身收盘价与成交量均值；可选标签窗口排名归一化。
- 日级动态图：数据层负责按日切片与特征窗口，图构建委托 `graph_builder`，训练逻辑由模型/训练器承担。

## 常用配置片段
- 快速测试：`ConfigTemplates.quick_test()`（小窗口、小 batch，无缓存）。
- 生产训练：`ConfigTemplates.production()`（大 batch、多进程、dtype 优化、缓存开启）。
- 回测/滚动：`ConfigTemplates.backtest()` + `split_strategy='rolling'`，可叠加 `create_rolling_daily_loaders` 走 Walk-Forward。

## 与其他子系统的交互
- data_processor：可在进入 DataManager 前完成清洗/特征计算；动态图依赖 `graph_builder` 统一入口。
- model：训练器接受 `LoaderCollection` 或日级 `DailyGraphDataLoader`；RollingDaily/Window 训练应使用本模块返回的窗口列表保持时间完整性。
- 回测/因子生成：使用 `InferenceDataset` 或测试集预测结果，与回测框架共享时间索引和股票代码。

## 版本
- 当前模块版本：1.1.0（见 __init__.py）。
