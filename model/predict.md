# 预测助手需求说明（中文版）

本文描述在 `quantclassic` 中可复用的预测助手应满足的要求，方便笔记本和脚本通用（例如 `predict_with_metadata`）。

## 目标
- 提供单一入口对任意 DataLoader 做推理，并重建含日期、标的、多因子列的 DataFrame。
- 适配多类模型：纯时序 MLP/LSTM、图模型（可选 adj）、混合多因子输出。
- 尽量不改动 `SimpleTrainer`，减少 notebook 手写样板。

## 输入契约
- `trainer`：暴露 `.model`（nn.Module），已在正确设备上。
- `loader`：batch 可能是 `(X, y)` / `(X, y, adj)` / `(X, y, adj, stock_ids, date)`，额外字段除非映射否则忽略。
- `device`：str 或 `torch.device`，用于将张量放到设备。
- 可选参数：
  - `parse_batch_fn(batch) -> dict`：把任意 batch 归一化为 `{x, y, adj, stock_ids, date, extra_forward_kwargs}`。
  - `reduce_factor`：`None` 保留全因子；`'mean'` 或可调用，将每样本因子向量压成标量 `pred`。
  - `return_label`：是否输出标签 DataFrame。
  - `return_tensor`：跳过 DataFrame，直接返回张量以提速。

## 前向规则
- 使用 `torch.no_grad()` 并设 `model.eval()`。
- 张量移到 `device`，元数据留在 CPU。
- 调用顺序：优先 `model(x, **extra_forward_kwargs)`；若有 adj 才传；模型返回元组取首元素为预测。
- 形状处理：若预测为 1-D，扩到 `(N, 1)`；`return_label=True` 时标签也对齐维度。

## 输出契约
- 默认返回 `pred_df`：列含 `trade_date`、`order_book_id`、`pred`，若因子维度>1 且未聚合，则附加 `pred_factor_{i}`。
- 可选 `label_df`：列含 `trade_date`、`order_book_id`、`label`。
- 若 batch 无 `stock_ids/date`，则返回张量（或索引行），跳过 DataFrame 合并。
- 过程应确定性且不修改模型/配置（例如不改 `d_feat`）。

## 错误处理与日志
- 仅在需要 DataFrame 输出时校验 `stock_ids`/`date`；否则继续张量模式。
- `adj` 缺失应被优雅跳过，不强制要求。
- batch 不受支持且无 `parse_batch_fn` 时给出清晰错误。

## 集成建议
- 作为工具函数暴露（如 `quantclassic.model.predict.predict_with_metadata`），供 notebook/脚本调用。
- 可在 `SimpleTrainer` 加一层薄包装转调该函数，减少用户端代码。
- 保持无副作用；保存 CSV/Parquet 等应由调用方显式指定。

## 最小 API 建议
```
def predict_with_metadata(
    trainer,
    loader,
    device,
    *,
    parse_batch_fn=None,
    reduce_factor='mean',
    return_label=True,
    return_tensor=False,
):
    ...
```
- DataFrame 模式返回 `(pred_df, label_df)`；`return_tensor=True` 时返回 `(pred_tensor, label_tensor)`。
- `parse_batch_fn` 作为自定义数据集的兜底适配器。

## 复用预期
- 兼容静态 DataLoader 与 `DailyGraphDataLoader`。
- 支持有/无邻接矩阵的模型。
- 支持单因子与多因子输出。
- 可在评估、滚动窗口中安全复用，不修改全局状态。

## 与 model 模块其他文件的关系与依赖
- `base_model.py` / `PyTorchModel`：定义 `_parse_batch_data`、`_forward_for_predict`、`_post_process` 钩子，预测助手应遵循这些约定，避免重复解析逻辑。
- `train/simple_trainer.py`：训练器暴露 `.model` 并可在 `predict` 中转调助手；助手不应假设训练细节，只依赖 forward 接口。
- `hybrid_graph_models.py`、`pytorch_models.py`：具体模型实现 forward，可能需要 `adj` 或返回多输出；助手需仅在存在 `adj` 时传参，并对元组输出取主预测。
- `model_factory.py` / `modular_config.py`：用于构建模型实例；助手无需依赖配置细节，只要模型 forward 符合约定即可。
- `rolling_daily_trainer.py` / `rolling_window_trainer.py`：滚动训练可共享同一预测助手，保证窗口内外推理一致。
