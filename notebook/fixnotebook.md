# Notebook 待修复问题与方案

- 第4单元格（敏感信息）[jupyterlab/quantclassic/notebook/lstm+attention12011427.ipynb#L370-L376](jupyterlab/quantclassic/notebook/lstm+attention12011427.ipynb#L370-L376)：内嵌 rqdatac 账号密码，存在泄露与重复 init 风险。解决方案：改为从环境变量/本地未提交配置读取，清理单元格输出与明文字符串。
- 第7单元格（特征列缺失）[jupyterlab/quantclassic/notebook/lstm+attention12011427.ipynb#L1495-L1499](jupyterlab/quantclassic/notebook/lstm+attention12011427.ipynb#L1495-L1499)：`key_columns` 未包含 `industry_code`，后续动态图构图依赖该列，会导致缺列报错。解决方案：将 `industry_code` 加入 `key_columns`，确保全流程（保存 parquet 与 DataManager）保留该列且不被数值化。
- 第11单元格（废弃类导入）✅ 已修复：导入 `DynamicGraphTrainer` 报 ModuleNotFoundError，已改为 `from quantclassic.model.train import SimpleTrainer, TrainerConfig`，并添加使用示例。
- 第12单元格（动态图训练仍依赖已删除的 DynamicGraphTrainer）✅ 已修复（2026-01-16）：改为 `SimpleTrainer + TrainerConfig`，手写 `predict_with_metadata` 从 DailyGraphDataLoader 的 (X,y,adj,stock_ids,date) 重建 DataFrame（含 `pred_factor_*`、`pred`、`label`）；模型用 `HybridGraphModel.from_config(...).model`；训练用 `trainer.train(...)`；保存/加载沿用 BaseTrainer `best_score` 字段。
- 第13单元格（DataManager 滚动窗口分支仍调用 DynamicGraphTrainer）✅ 已修复（2026-01-16）：复用 SimpleTrainer + `predict_with_metadata` 管线，输入 loaders 为 `dm.create_rolling_daily_loaders(...)` 输出；检查点字段对齐 BaseTrainer（`best_score`）；`future_return` 优先使用 `y_ret_10d`（真实收益），缺失时回退 `dm.config.label_col`。
- 第13单元格（收益列取值不当）[jupyterlab/quantclassic/notebook/lstm+attention12011427.ipynb#L2892-L2899](jupyterlab/quantclassic/notebook/lstm+attention12011427.ipynb#L2892-L2899)：`future_return` 直接取自 `dm.config.label_col`（目前是中性化后的 `alpha_label`），回测将使用处理后的标签而非真实前瞻收益。解决方案：在预处理阶段额外保留原始前瞻收益列（如中性化前的 `y_ret_10d` 或基于价格的收益），在此处映射为 `future_return`，训练仍用 `alpha_label`。
- 第15单元格（邻接矩阵数据源与路径不一致）[jupyterlab/quantclassic/notebook/lstm+attention12011427.ipynb#L3256-L3305](jupyterlab/quantclassic/notebook/lstm+attention12011427.ipynb#L3256-L3305)：使用 `output/caitong_processed.parquet` 构图，而前面流程产出的是 `wind_processed_all.parquet`；当 `USE_STATIC_ADJ` 为 False 时，`adj_config` 仍指向 `output/industry_adj_matrix.pt`，与实际构造矩阵不符。解决方案：统一读取与训练一致的处理文件（wind parquet），如果切换到相关性邻接矩阵则同时将 `adj_save_path/static_adj_path` 指向新生成的相关性矩阵；若保持静态行业图则设 `USE_STATIC_ADJ=True` 并保持行业矩阵路径一致。
- 第12单元格（动态图训练维度不匹配）✅ 已修复（2025-01-16）：在 `loss.py` 中为 MSEWithCorrelationLoss、MAEWithCorrelationLoss、HuberWithCorrelationLoss、ICLoss 添加了多因子聚合逻辑（`preds.mean(dim=1)`），同时在 `simple_trainer.py` 的 `validate_epoch` 中增加相同处理。模型输出 64 维因子，聚合为标量后与标签计算损失。
- 第12/13单元格（行业列缺失导致图退化）✅ 已修复（2025-01-16）：
  - **根本原因**：`DailyBatchDataset` 返回的日批次数据只包含特征列，不包含 `industry_code`。`IndustryGraphBuilder` 在 `_build_from_scratch()` 中查找行业信息时，首先检查 `_stock_industry_mapping`，若为空则回退检查 `df_day` 中的 `industry_col` 列，两者都没有时返回自环矩阵。
  - **原代码问题**：`HybridGraphBuilder.__init__` 创建内部 `industry_builder` 时未传入 `stock_industry_mapping`；notebook 中用 `hasattr(hybrid_builder, '_industry_builder')` 检测（属性名错误，正确的是 `industry_builder`）。
  - **修复方案**：
    1. 修改 [graph_builder.py](quantclassic/data_processor/graph_builder.py)：`HybridGraphBuilder.__init__` 新增 `stock_industry_mapping` 参数并传递给内部 `industry_builder`
    2. 修改 notebook cell 11：构建 `stock_industry_mapping = dict(zip(df_full[stock_col], df_full['industry_code']))`，直接传入 `HybridGraphBuilder(stock_industry_mapping=stock_industry_mapping, ...)`
- 第12/13单元格（预测函数重复手写）✅ 已修复（2025-01-16）：在 cell 12 和 cell 13 中移除手写的 `predict_with_metadata` 函数，改用 `SimpleTrainer.predict(loader, return_numpy=False)` 获取预测张量，然后遍历 loader 获取 `(stock_ids, date)` 元数据重建 DataFrame。统一使用工具类预测，减少维护成本。
- 第12单元格（动态图训练空批次触发 0 节点 GAT 异常）✅ 已修复（2025-01-16）：
  - **根因**：源数据存在日期 `label_col` 全 NaN 的情况（前瞻收益缺失或截面标准化前即为空），`DailyBatchDataset.__getitem__` 的 `valid_mask` 全 False 导致该日批次 `n_stocks=0`，GAT 层收到 N=0 张量时 `view(N,N,-1)` 报 reshape 异常。
  - **修复**：
    1. `DailyGraphDataLoader.__iter__`：过滤 `X.size(0)==0` 的空批次，不 yield 给训练循环
    2. `SimpleTrainer.train_batch`：检测 `x.size(0)==0` 直接返回 `0.0`
    3. `SimpleTrainer.validate_epoch`：检测 `x.size(0)==0` 直接 `continue`
    4. `SimpleTrainer.predict`：检测 `x.size(0)==0` 直接 `continue`

