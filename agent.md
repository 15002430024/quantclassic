# Agent Guide for QuantClassic

本指南面向 AI 与人类协作者，快速定位应读的模块文档并正确调用 QuantClassic 的生产链路（config · data_processor · data_set · model · backtest）。

**最后更新**: 2026-02-02

## 📜 变更日志

### [2026-02-02] - REQ-001 多因子预测维度修复

**修复:**
- `model/train/rolling_daily_trainer.py`: `_predict_daily_window` 方法增加对多因子输出 `(N, F)` 的 squeeze / 取首列处理，解决 `ValueError: can only convert an array of size 1 to a Python scalar` 报错。

---

### [2026-01-30] - REQ-002 修复

**修复:**
- `data_set/manager.py`: `_normalize_graph_builder_config` 对 `type='hybrid'` 也注入 `stock_industry_mapping`，解决混合图训练时行业列缺失告警。

---

### [2026-01-30] - REQ-001 修复

**修复:**
- `model/train/simple_trainer.py`: 在 `train_batch` 方法中增加多因子预测聚合逻辑（`pred.mean(dim=1)`），与 `validate_epoch` 保持一致，解决 `output_dim > 1` 时张量维度不匹配的 RuntimeError。

**修改:**
- `data_set/graph/daily_graph_loader.py`: `groupby` 显式设置 `observed=False`，消除 pandas FutureWarning。

---

## 1. 思维框架
- 编排入口：始终以 config/TaskRunner 或 CLI (`qcrun` / `python -m quantclassic.config.cli`) 作为端到端入口。
- 数据流：特征/标签 → data_processor (预处理/中性化) → data_set (划分+Loader) → model (训练/预测) → backtest (IC/分组/绩效) → workflow/output。
- 生产保障：仅依赖上述五个模块；data_fetch/factor_hub 视为辅助或实验性，勿默认使用。

## 2. 询问/决策指引
- 运行/配置问题 → 先读 [config/README.md](config/README.md)（ConfigLoader、TaskRunner、模板、CLI）。
- 预处理/列名/中性化 → 读 [data_processor/README.md](data_processor/README.md)（PreprocessConfig/DataPreprocessor）。
- 划分策略/Loader/图数据 → 读 [data_set/README.md](data_set/README.md)（DataManager、splitter、DatasetFactory）。
- 模型/训练器/配置 → 读 [model/README.md](model/README.md)（PyTorchModel、Trainer、modular config）。
- 因子回测/可视化 → 读 [backtest/README.md](backtest/README.md)（FactorBacktestSystem/MultiFactorBacktest）。

## 3. 回答/执行准则（给 AI）
- 优先指路：若问题涉及具体功能，先指向对应模块 README，再给最短可行步骤或示例。
- 少猜测：不清楚数据路径、列名、目标持有期、GPU 资源时，先向用户澄清再给命令。
- 保持链路一致：
  - 端到端任务：使用 TaskRunner/CLI；不要跳过预处理或划分。
  - 自定义模块：确保配置含 class + module_path，遵循 config README 的字段约定。
- 检查输出：任务完成后提示用户查看 output/experiments 或 output/backtest；必要时给校验命令（如 `pytest tests/...`）。

## 4. 常见操作模板
- 运行模板配置：`qcrun config/templates/lstm_basic.yaml`。
- SDK 编排：`ConfigLoader.load(...)` → `TaskRunner().run(cfg, experiment_name=...)`。
- 只做预处理：`DataPreprocessor.fit_transform(df)` 并保存 `preprocessor.pkl`。
- 只做划分/Loader：`DataManager.run_full_pipeline()` 或 `create_daily_loaders`（图数据）。
- 回测：`MultiFactorBacktest.run(predictions_df, label_col=...)` 或 `FactorBacktestSystem.run_backtest(...)`。

### 🧭 动态图训练速览
- 数据加载：使用 `data_set/graph/daily_graph_loader.py` 提供的 `DailyBatchDataset` + `DailyGraphDataLoader`（支持行业/相关性/混合图，按日动态 batch）。
- 训练入口（推荐）：`TaskRunner` 配置 `trainer_class: "DynamicGraphTrainer"`，内部已兼容为 `SimpleTrainer` 路径并自动消费日级 loaders。
- 训练入口（直接调用）：`from quantclassic.model.train import SimpleTrainer, TrainerConfig`，使用上面的 `train_daily_loader/val_daily_loader` 调用 `SimpleTrainer(model, TrainerConfig(...)).fit(train_loader, val_loader, ...)`；预测用 `trainer.predict(test_loader)`。
- 图构建器：`IndustryGraphBuilder` / `CorrGraphBuilder` / `HybridGraphBuilder` 位于 `data_processor/graph_builder.py`，可在创建 `DailyGraphDataLoader` 时注入。
- 常见列名：`stock_col=order_book_id`，`time_col=trade_date`，行业列在行业/混合图中需提供（如 `industry_code`）。

## 5. 需要向用户确认的关键信息
- 数据位置与格式（parquet/csv）及列名映射（stock/time/label）。
- 目标标签/持有期、训练/验证/测试时间范围。
- 可用硬件与并行限制（GPU/CPU）。
- 是否需要滚动训练、动态图训练或多因子集成。
- 回测参数（分组数、换仓频率、基准指数、是否保存图表/Excel）。

## 6. 参考
- 架构与数据流总览：见 [ARCHITECTURE.md](ARCHITECTURE.md)。
- 更详细的运行指南：config 目录的 QUICKSTART/RUN_GUIDE，backtest/update_readme。

如遇未覆盖的问题，请先询问用户需求，再选择相应模块文档查阅。