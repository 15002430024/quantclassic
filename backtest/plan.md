# Backtest 问题与修改方案

- [x] 因子列自动检测冲突：`FactorBacktestSystem.run_backtest` 直接把 `FactorGenerator` 输出 (如 `pred_alpha`) 喂给 `FactorProcessor.process`，未显式传 `factor_cols`，而 `_auto_detect_factor_cols` 只接受 `factor_*` 前缀，触发 ValueError 并使默认 `factor_col='factor_raw_std'` 无法落地。方案：允许配置化列映射（例如 `factor_aliases={'pred_alpha':'factor_raw'}`）或在调用处传入 `factor_cols`，保证流水线可打通。参考 [backtest/backtest_system.py#L96-L192](backtest/backtest_system.py#L96-L192)、[backtest/factor_processor.py#L273-L285](backtest/factor_processor.py#L273-L285)。
- [x] 默认因子列硬编码导致默认流程仍不可用：`FactorBacktestSystem.run_backtest` 默认使用 `factor_col='factor_raw_std'` 做 IC/组合，但 `FactorGenerator.generate_factors` 的默认预测模式只产出 `pred_alpha`/`pred_alpha_*`，`FactorProcessor.process` 生成的是对应的 `_std` 列，未同步回 `factor_col`，导致未显式传 `factor_col` 时在 IC 处直接 KeyError（`quick_test` 同样受影响）。**已修复**：将 `factor_col` 默认值改为 `None`，新增 `_infer_factor_col` 方法自动从处理后 DataFrame 推断首个 `*_std` 列。参考 [backtest/backtest_system.py#L98-L150](backtest/backtest_system.py#L98-L150)、[backtest/factor_generator.py#L220-L234](backtest/factor_generator.py#L220-L234)。
- [x] 日志 formatter 作用域错误：`_setup_logging` 内 `formatter` 只在 console 分支定义，`console_log=False` 且设置 `log_file` 会抛 `NameError`。方案：在进入分支前创建 formatter，共用给 console/file handler。参考 [backtest/backtest_system.py#L48-L72](backtest/backtest_system.py#L48-L72)。
- [x] `get_processing_stats` 名称遮蔽：函数体内 `stats` 字典覆盖了导入的 `scipy.stats`，导致 `stats.skew/kurtosis` 调用报错。方案：重命名本地变量或将 `scipy.stats` 别名成 `scipy_stats`，避免遮蔽。参考 [backtest/factor_processor.py#L299-L338](backtest/factor_processor.py#L299-L338)。
- [x] `quick_test` 状态未恢复：该方法用快速配置重新 `__init__`，结束后仅还原 `self.config`，子组件仍指向临时配置。方案：保存/恢复组件实例，或重建组件时同步回原始配置，防止后续调用使用错误参数。参考 [backtest/backtest_system.py#L272-L299](backtest/backtest_system.py#L272-L299)。
- [x] 三套回测管线重复且默认不一致：`FactorBacktestSystem`、`BacktestRunner`、`MultiFactorBacktest` 步骤近似但默认列名/入口差异 (`factor_raw_std` vs `factor_value` vs 适配后列)，存在漂移风险。方案：提取统一 orchestrator/共享步骤（处理/IC/组合/评估/可视化），或明确官方入口并在文档标注其余为简版示例，减少维护面。参考 [backtest/backtest_system.py#L22-L192](backtest/backtest_system.py#L22-L192)、[backtest/backtest_runner.py#L20-L160](backtest/backtest_runner.py#L20-L160)、[backtest/multi_factor_backtest.py#L37-L137](backtest/multi_factor_backtest.py#L37-L137)。

## 拆解与实施建议

1) 因子列兼容：在 `FactorBacktestSystem.run_backtest` 增加 `factor_cols`/`factor_aliases` 参数透传到 `FactorProcessor.process`，并在 `BacktestConfig` 提供默认空映射；`FactorProcessor._auto_detect_factor_cols` 增加可选前缀列表（默认 `['factor_','pred_']`）。
2) 日志 formatter：在 `_setup_logging` 冒头定义 `formatter = logging.Formatter(...)`，后续 console/file handler 复用，避免分支变量缺失。
3) 统计函数遮蔽：把本地 `stats` 字典改名为 `stat_dict`，或把 `from scipy import stats` 改为 `import scipy.stats as scipy_stats` 并用 `scipy_stats.skew/kurtosis`。
4) 快速测试恢复：`quick_test` 记录原始组件实例或在恢复 `self.config` 后重新初始化 `factor_processor` 等组件为原配置；或改为上下文管理器式调用，确保 finally 段恢复。
5) 管线去重：提炼公共基类/助手（如 `run_pipeline(processed_df, factor_col, return_col, config)`），三入口只处理输入/默认差异；若不重构，至少在 README 注明推荐入口（例如 `MultiFactorBacktest`）并保持其参数为单一真源。
