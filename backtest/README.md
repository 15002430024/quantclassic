# Factorsystem 回测子系统

> ✅ **生产级入口** — 本模块为因子生成/处理/回测的唯一生产入口。
>
> 如需进行回测，请优先使用本模块，而非 `factor_hub`（实验性）。

工程化的因子回测流水线，覆盖因子生成/适配、预处理、IC 分析、组合构建、绩效评估与可视化。支持单因子与多因子、端到端模型推理与外部预测两种入口。

## 快速上手

### 多因子（推荐，基于 RollingWindowTrainer 预测）
```python
from quantclassic.backtest import BacktestConfig, MultiFactorBacktest

config = BacktestConfig(output_dir="output/backtest", save_plots=True)
backtest = MultiFactorBacktest(config)

# predictions_df: 包含多因子预测及标签的 DataFrame
results = backtest.run(predictions_df, label_col="y_ret_10d")
```

### 端到端模型推理（生成因子再回测）
```python
from quantclassic.backtest import BacktestConfig, FactorBacktestSystem

config = BacktestConfig(output_dir="output/backtest")
system = FactorBacktestSystem(config)
model = system.load_model("output/best_model.pth")

# data_df 需包含特征列与 ts_code/trade_date
results = system.run_backtest(data_df, model)
```

## 架构与模块关系

- 因子生成/适配：`FactorGenerator`（端到端推理），`PredictionAdapter`（外部预测整合，多因子集成）
- 因子预处理：`FactorProcessor`（去极值、填充、标准化、中性化；自动检测 `factor_`/`pred_`/`latent_` 前缀）
- 分析与构建：`ICAnalyzer` → `PortfolioBuilder` → `PerformanceEvaluator`
- 可视化与报告：`ResultVisualizer` / `ResultVisualizerPlotly`
- 基准与缓存：`BenchmarkManager`（可选，用于基准收益拉取与缓存）

数据流：生成/适配 → 预处理 → IC → 分组/多空 → 绩效 → 报告/文件输出。

## 入口与适用场景

| 入口 | 适用 | 因子列默认 | 输入要求 |
|------|------|-----------|----------|
| **MultiFactorBacktest** | ✅ 推荐，多因子生产/研究 | 适配后自动选择 | predictions_df (含 label) |
| BacktestRunner | 单因子快速验证 | `factor_value` | 已有因子与收益列 |
| FactorBacktestSystem | 端到端模型推理 | 自动推断（优先 `*_std`，其次 `*_neutral/pred_`） | 原始特征 + 模型 |

提示：`FactorBacktestSystem.run_backtest` 未传 `factor_col` 时，会在处理后数据中自动选择首个 `*_std` 列；若有特定口径，请显式传 `factor_col`/`factor_cols`。

## 因子/收益列约定

- 自动检测前缀：`factor_`、`pred_`、`latent_`（排除 `_winsorized/_filled/_std/_neutral` 后缀）。
- 默认收益列：`y_true`（系统）/ `y_processed`（Runner）/ `label_col` 参数（MultiFactor）。
- 多因子集成：`PredictionAdapter` 支持 `mean` / `ic_weighted` / `best` / `custom` 权重。

## 配置要点（BacktestConfig）

- 因子处理：`winsorize_method`、`standardize_method`、`industry_neutral`、`market_value_neutral`。
- 组合构建：`n_groups`、`rebalance_freq`、`long_ratio`/`short_ratio`、`weight_method`。
- 绩效与可视化：`save_plots`、`generate_excel`、`plot_style`。
- 基准：`benchmark_index` 或 `custom_benchmark_col`（配合 `BenchmarkManager`）。

## 依赖与运行

```bash
pip install pandas numpy scipy scikit-learn torch matplotlib seaborn tqdm
```

## 目录速览

```
backtest/
    backtest_system.py       # 端到端控制器（含因子生成）
    multi_factor_backtest.py # 多因子主入口（推荐）
    backtest_runner.py       # 单因子快捷入口
    factor_generator.py      # 模型推理产因子
    prediction_adapter.py    # 预测结果适配与集成
    factor_processor.py      # 因子预处理
    ic_analyzer.py           # IC/ICIR 计算
    portfolio_builder.py     # 分组与多空构建
    performance_evaluator.py # 绩效指标
    result_visualizer.py     # 可视化
    benchmark_manager.py     # 基准拉取/缓存（可选）
    update_readme/           # 详细指南与示例
    plan.md                  # 问题与修改记录
```

## 文档与示例

- `update_readme/BACKTEST_GUIDE.md`：全流程详细说明与参数解释
- `example_backtest.py`：端到端示例
- `example_benchmark_usage.py`：基准管理示例

## 功能清单与实现要点

- 因子生成/适配：`FactorGenerator` 支持预测值与隐变量两种模式；`PredictionAdapter` 聚合多因子（mean/ic_weighted/best/custom）并计算因子 IC 权重。
- 因子自动识别与预处理：`FactorProcessor` 自动检测 `factor_`/`pred_`/`latent_` 前缀，去极值/填充/标准化/可选中性化，产出 `*_std`/`*_neutral` 列；默认分层按 `trade_date` 截面。
- 因子列自动推断：`FactorBacktestSystem.run_backtest` 在未传 `factor_col` 时自动选择首个 `*_std`（备选 `_neutral`/`pred_`/`latent_`/`factor_`）。
- IC 分析：`ICAnalyzer` 提供 Spearman/Pearson、IC/RankIC、ICIR、胜率、t 统计，并支持多期持有期配置。
- 组合构建：`PortfolioBuilder` 基于分位分组与 top/bottom 多空，支持等权/因子权重/市值权重，换仓频率与组数可配。
- 绩效评估：`PerformanceEvaluator` 计算年化收益、波动、夏普、最大回撤、卡玛等；支持基准列，配合 `BenchmarkManager` 自动拉取并合并基准收益。
- 可视化：`ResultVisualizer`/`ResultVisualizerPlotly` 输出累计收益、回撤、IC 序列/分布、分组表现等图表，可控制保存与输出格式。
- 快速测试：`FactorBacktestSystem.quick_test` 使用快速模板配置运行，并在完成后恢复原始配置与组件实例。
- 日志与输出：统一 logger（console/file），回测结果保存因子、IC、组合、绩效指标（含可选 Excel），图表输出到 `output_dir/plots`。

## 版本与变更

- v1.1.0：修复因子列检测、日志作用域、quick_test 状态恢复、因子列自动推断
- v1.0.0：初始版本
