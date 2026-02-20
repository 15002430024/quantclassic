# Agent Guide for QuantClassic

本指南面向 AI 与人类协作者，快速定位应读的模块文档并正确调用 QuantClassic 的生产链路（config · data_processor · data_set · model · backtest）。

**最后更新**: 2026-02-20

## 📜 变更日志

### [2026-02-20] - BUG-FIX: generate_weights long_only/short_only 权重归一化缺陷

**修复:**
- `backtest/portfolio_builder.py`: `generate_weights()` 在 `mode='long_only'` 或 `mode='short_only'` 时，权重之和从错误的 `long_ratio`/`short_ratio`（0.2）修正为 `1.0`（满仓）。
  - 根因：`long_ratio` 被同时用于"选股比例"（选前20%股票）和"资金部署比例"（权重和），但当拆分为独立的 long-only 回测时，应满仓部署。
  - 影响：此前每条腿仅部署 20% 资金，80% 闲置为现金，导致多空收益被稀释约 5 倍。
  - `long_short` 模式行为不变（仍归一化到 `long_ratio` / `short_ratio`）。
- `backtest/portfolio_builder.py`: `_validate_weights()` 新增 `mode` 参数，根据模式动态计算预期权重绝对值之和（`long_only`/`short_only`/`group` → 1.0，`long_short` → `long_ratio + short_ratio`），消除每调仓日的误报 warning。

---

### [2026-02-09] - Notebook Cell 17 回测单元格适配 GeneralBacktest v2.0

**修改:**
- `notebook/lstm+attention12011427.ipynb` Cell 17（步骤 4: 多因子回测）：从旧 API (`MultiFactorBacktest`) 迁移到 `GeneralBacktestAdapter`。
  - 移除 `from quantclassic.backtest import MultiFactorBacktest` 导入（该类已在 v2.0.0 中删除）。
  - 改用 `GeneralBacktestAdapter.run(factor_df, price_df, ...)` 作为回测入口。
  - 新增价格数据加载步骤（从 `data_config.base_dir / data_config.data_file` 读取 open/close）。
  - 新增内联 IC 加权多因子集成逻辑（替代旧 `MultiFactorBacktest` 的内置集成）。
  - 新增因子方向自动检测与修正（IC < 0 时反转因子）。
  - 新增收益列合并（从源数据读取 `y_ret_1d` 用于 IC 计算）。
  - 结果展示适配新 `metrics` 字典格式（15+ 指标：累计收益、年化收益、夏普、索提诺、卡玛、VaR 等）。
  - 新增 `bt_instance.plot_dashboard()` 综合仪表盘可视化。
  - 全局变量从 `backtest_results` 改名为 `static_backtest_results` 避免与 Cell 14 冲突。

---

### [2026-02-09] - GeneralBacktest 内嵌迁移

**重大变更:**
- `backtest/general_backtest/`: 将 GeneralBacktest (v1.0.2) 源码内嵌至 quantclassic，包含 `backtest.py`, `utils.py`, `__init__.py`，不再依赖外部安装。
- `backtest/__init__.py`: 重写，移除旧引擎导出（FactorBacktestSystem, BacktestRunner, MultiFactorBacktest, PerformanceEvaluator, ResultVisualizer, ResultVisualizerPlotly），新增 `GeneralBacktest` 直接导出。版本升至 v2.0.0。
- `backtest/general_backtest_adapter.py`: 简化导入逻辑，直接从内嵌模块导入 `GeneralBacktest`，移除 sys.path 兜底和外部包检测。
- `backtest/backtest_config.py`: `engine` 默认值改为 `'general_backtest'`，移除 `'quantclassic'` 选项。
- `config/runner.py`: `_run_backtest` 适配新架构，使用 `GeneralBacktestAdapter`。

**删除:**
- `backtest/backtest_system.py` — 旧回测主控制器
- `backtest/backtest_runner.py` — 旧单因子运行器
- `backtest/multi_factor_backtest.py` — 旧多因子回测入口
- `backtest/performance_evaluator.py` — 旧绩效评估器（被 GeneralBacktest 内置指标替代）
- `backtest/result_visualizer.py` — 旧可视化（被 GeneralBacktest plot 方法替代）
- `backtest/result_visualizer_plotly.py` — Plotly 可视化
- `backtest/benchmark_manager_backup.py`, 旧示例文件, plan.md, test 文件, update_readme/

**保留:**
- `backtest/backtest_config.py` — 配置
- `backtest/portfolio_builder.py` — 组合构建与权重生成（generate_weights）
- `backtest/factor_generator.py` — 因子生成
- `backtest/factor_processor.py` — 因子处理
- `backtest/ic_analyzer.py` — IC 分析
- `backtest/prediction_adapter.py` — 多因子预测适配
- `backtest/benchmark_manager.py` — 基准管理

---

### [2026-02-09] - REQ-007 引入 GeneralBacktest 作为回测后端

**新增:**
- `backtest/general_backtest_adapter.py`: 新增 `GeneralBacktestAdapter` 类，封装 GeneralBacktest 的导入、数据转换、回测调用与图表保存。支持 editable install 和 sys.path 兜底两种导入方式。
- `backtest/portfolio_builder.py`: `PortfolioBuilder` 新增 `generate_weights()` 方法，将因子排序选股逻辑转化为 `[date, code, weight]` 格式的权重表，支持 `long_only`/`short_only`/`long_short`/`group` 四种模式。新增 `_validate_weights()` 校验方法。
- `backtest/backtest_config.py`: `BacktestConfig` 新增 `engine`（`'quantclassic'`/`'general_backtest'`）、`buy_price`、`sell_price`、`general_backtest_options` 字段，支持回测引擎切换。
- `backtest/backtest_system.py`: `FactorBacktestSystem.run_backtest()` 新增引擎切换逻辑——当 `engine='general_backtest'` 时调用 `GeneralBacktestAdapter`，失败自动降级并记录 Error 日志。新增 `_run_general_backtest()` 方法。
- `backtest/__init__.py`: 导出 `GeneralBacktestAdapter`、`is_general_backtest_available`。

**文档更新:**
- `backtest/README.md`: 新增 GeneralBacktest 引擎接入使用说明与安装指引。
- `ARCHITECTURE.md`: 更新 backtest 模块架构描述，新增 GeneralBacktestAdapter 组件、Schema 与版本历史。

---

### [2026-02-06] - REQ-006 回测框架收益频率错配修复

**修复 (BUG-1 - 收益列频率错配):**
- `notebook/lstm+attention12011427.ipynb` Cell 13: `return_col_name` 从 `'y_ret_10d'` 改为 `'y_ret_1d'`，因子列 `pred` 重命名为 `factor_raw`（去掉 `_std` 后缀）。
- `notebook/lstm+attention12011427.ipynb` Cell 14: 新增 §1.6 即时修复，从源数据读取 `y_ret_1d` 替换 `future_return`，兼容已有磁盘缓存。

**修复 (BUG-2 - 因子截面分散度不足):**
- Cell 14 §1.6: 因子列 `factor_raw_std` → `factor_raw`，使 `FactorProcessor.process()` 的 winsorize + z-score 截面标准化流程对因子生效（原来 `_std` 后缀被跳过）。

**修复 (BUG-3 - rebalance_freq 未生效):**
- `backtest/portfolio_builder.py`: `create_long_portfolio` / `create_short_portfolio` 重写为支持调仓频率——仅在 `get_rebalance_dates()` 返回的日期重新选股，非调仓日维持持仓不变。
- `backtest/portfolio_builder.py`: `backtest_with_rebalance` 修复 `ts_code` 硬编码，改为自动检测 `order_book_id` / `ts_code`。

**验证结果（修复前 → 修复后）:**
- long 年化: 296%~387% → **18.59%** ✅
- long_short 夏普: 异常 → **2.77** ✅
- long_short 回撤: -90.17% → **-17.41%** ✅
- cum_return: 5.14 亿倍 → 正常范围 ✅
- 时间覆盖: 2010-2025 全覆盖 ✅
- 详见 [.requirements/REQ-006.md](.requirements/REQ-006.md)

---

### [2026-02-06] - REQ-005 回测收益起始时间与标签校验

**修复:**
- `backtest/backtest_system.py`: `_merge_returns` 方法修复 `ts_code` 硬编码，改为自动检测 `order_book_id` / `ts_code`，确保离线模式下收益列能正确合并。

**改进:**
- `notebook/lstm+attention12011427.ipynb` Cell 14: 增加数据完整性诊断模块（各年数据量、缺失统计）和回测结果时间区间诊断（组合净值起止日期），方便定位数据截断问题。
- Cell 14 增加 `importlib.reload(quantclassic.backtest.backtest_system)` 确保修复后的代码生效。

---

### [2026-02-06] - REQ-002 FactorBacktestSystem 离线回测支持

**修复 (REQ-002 - run_backtest TypeError):**
- `backtest/backtest_system.py`: `run_backtest` 新增 `stock_col`, `time_col`, `save_results` 参数，兼容 Notebook 中 `FactorBacktestSystem` 单因子回测调用。
- 新增离线模式：当 `model=None` 且 `data_df` 已包含因子列时，跳过因子生成步骤直接进入处理/IC/组合/绩效流程。
- 列名映射：自动将自定义 `stock_col`/`time_col` 重命名为内部标准名 `order_book_id`/`trade_date`。
- `save_results` 参数控制是否落地文件，`None` 时沿用配置默认值。
- 离线模式下如因子列已带 `_std`/`_neutral` 等后缀，自动跳过 `FactorProcessor`，避免自动检测因排除已处理列而抛 ValueError。

**改进 (Notebook 回测单元格独立运行):**
- `notebook/lstm+attention12011427.ipynb` Cell 13（步骤6-7）：训练结束后自动将 `all_predictions` 落盘为 `output/rolling_dynamic_cache/dynamic_predictions.parquet`，并将 `factor_cols`/`test_ics`/`window_ic_summary`/`config` 写入 `metadata.json`。
- `notebook/lstm+attention12011427.ipynb` Cell 14（步骤1）：新增三级数据加载逻辑：①内存变量优先 → ②磁盘缓存回退（`output/rolling_dynamic_cache/`）→ ③报错提示两种修复方式。回测单元格现在可以独立运行，无需从头执行整个 Notebook。
- Cell 13 的 `rolling_dynamic_results['window_results']` 改为序列化友好的 dict 列表（而非不可 pickle 的 trainer 内部对象），确保 Cell 14 从磁盘加载后仍可展示窗口 IC。

### [2026-02-06] - REQ-003 & REQ-004 Notebook 修复

**修复 (REQ-003 - Cell 13 滚动训练 OOM):**
- `notebook/lstm+attention12011427.ipynb` Cell 13 步骤5：将 `pd.concat([dm._train_df, dm._val_df, dm._test_df])` 替换为 `pd.read_parquet(data_path, columns=[...])` 仅读取3列收益数据，避免全量 DataFrame 拼接导致 Kernel OOM 崩溃。
- 增加训练结束后 `del window_loaders; gc.collect(); torch.cuda.empty_cache()` 释放显存。
- 在 `rolling_dynamic_results` 中增加 `factor_cols` 字段，供下游 Cell 14 回测使用。

**修复 (REQ-004 - Cell 15 列名错误):**
- `notebook/lstm+attention12011427.ipynb` Cell 15：`df_adj['industry_name'].nunique()` → `df_adj['industry_code'].nunique()`，修复因数据无 `industry_name` 列导致的 KeyError。

**修复 (FactorBacktestSystem ValueError):**
- `backtest/backtest_system.py`: `run_backtest` 在 `factor_cols=None` 但指定了 `factor_col` 时，自动将 `factor_col` 纳入处理列表，解决因单因子列名不带 standard prefix (`factor_`, `pred_`) 导致的自动检测失败。

---

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
- 因子回测/可视化 → 读 [backtest/README.md](backtest/README.md)（GeneralBacktestAdapter / GeneralBacktest）。

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
- 回测：`GeneralBacktestAdapter(config).run(factor_df, price_df, ...)` 或直接 `GeneralBacktest(start, end).run_backtest(...)`。

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