"""
预测助手模块 - predict.py

提供统一的带元数据预测接口，支持任意 DataLoader 和模型类型。
可被 notebook、脚本以及 SimpleTrainer 复用。

核心功能:
- predict_with_metadata: 从 DataLoader 预测并返回含日期/股票的 DataFrame
- 兼容静态 DataLoader 与 DailyGraphDataLoader
- 支持有/无邻接矩阵的模型
- 支持单因子与多因子输出

依赖关系:
- 与 base_model.py 的 _parse_batch_data 约定兼容
- 可被 train/simple_trainer.py 调用
- 与 hybrid_graph_models.py、pytorch_models.py 的 forward 接口兼容

Usage:
    from quantclassic.model.predict import predict_with_metadata
    
    pred_df, label_df = predict_with_metadata(
        trainer, test_loader, device='cuda'
    )
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union
)


def _default_parse_batch_fn(batch) -> Dict[str, Any]:
    """
    默认批次解析函数
    
    支持格式：
    - (x, y)
    - (x, y, adj)
    - (x, y, adj, stock_ids)
    - (x, y, adj, stock_ids, date)
    - dict 格式
    
    Returns:
        归一化后的字典 {x, y, adj, stock_ids, date, extra_forward_kwargs}
    """
    result = {
        'x': None,
        'y': None,
        'adj': None,
        'stock_ids': None,
        'date': None,
        'extra_forward_kwargs': {}
    }
    
    if isinstance(batch, dict):
        result['x'] = batch.get('x') or batch.get('features') or batch.get('input')
        result['y'] = batch.get('y') or batch.get('labels') or batch.get('target')
        result['adj'] = batch.get('adj') or batch.get('adj_matrix')
        result['stock_ids'] = batch.get('stock_ids') or batch.get('stock_idx') or batch.get('idx')
        result['date'] = batch.get('date') or batch.get('trade_date')
        return result
    
    if isinstance(batch, (list, tuple)):
        n = len(batch)
        if n >= 1:
            result['x'] = batch[0]
        if n >= 2:
            result['y'] = batch[1]
        if n >= 3:
            result['adj'] = batch[2]
        if n >= 4:
            result['stock_ids'] = batch[3]
        if n >= 5:
            result['date'] = batch[4]
        return result
    
    # 单个张量，作为 x
    result['x'] = batch
    return result


def predict_with_metadata(
    model_or_trainer: Union[nn.Module, Any],
    loader,
    device: Optional[Union[str, torch.device]] = None,
    *,
    parse_batch_fn: Optional[Callable] = None,
    reduce_factor: Optional[Union[str, Callable]] = 'mean',
    return_label: bool = True,
    return_tensor: bool = False,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[torch.Tensor, torch.Tensor]]:
    """
    带元数据的预测函数
    
    从 DataLoader 进行推理，并重建含日期、标的、多因子列的 DataFrame。
    兼容 SimpleTrainer（取其 .model）和直接的 nn.Module。
    
    Args:
        model_or_trainer: nn.Module 或暴露 .model 的 trainer（如 SimpleTrainer）
        loader: DataLoader，batch 可以是多种格式
        device: 计算设备，None 则自动检测
        parse_batch_fn: 自定义批次解析函数，返回 {x, y, adj, stock_ids, date, extra_forward_kwargs}
        reduce_factor: 因子聚合方式
            - None: 保留完整因子向量，生成 pred_factor_{i} 列
            - 'mean': 取均值作为 pred
            - callable: 自定义聚合函数 (factor_tensor -> scalar_tensor)
        return_label: 是否返回标签 DataFrame
        return_tensor: True 则跳过 DataFrame 构建，直接返回张量
        
    Returns:
        DataFrame 模式 (return_tensor=False):
            (pred_df, label_df) - 预测 DataFrame 和标签 DataFrame
            pred_df 列: trade_date, order_book_id, pred, [pred_factor_0, pred_factor_1, ...]
            label_df 列: trade_date, order_book_id, label
            
        Tensor 模式 (return_tensor=True):
            (pred_tensor, label_tensor)
            
    Raises:
        ValueError: batch 格式不受支持且未提供 parse_batch_fn
        
    Example:
        >>> from quantclassic.model.predict import predict_with_metadata
        >>> pred_df, label_df = predict_with_metadata(trainer, test_loader, 'cuda')
        >>> ic = pred_df.groupby('trade_date').apply(
        ...     lambda x: x['pred'].corr(label_df.set_index(['trade_date','order_book_id']).loc[x.name, 'label'])
        ... ).mean()
    """
    # 获取模型
    if hasattr(model_or_trainer, 'model') and isinstance(model_or_trainer.model, nn.Module):
        model = model_or_trainer.model
    elif isinstance(model_or_trainer, nn.Module):
        model = model_or_trainer
    else:
        raise ValueError(
            f"model_or_trainer 必须是 nn.Module 或暴露 .model 的对象，"
            f"实际类型: {type(model_or_trainer)}"
        )
    
    # 设备
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    
    # 解析函数
    if parse_batch_fn is None:
        parse_batch_fn = _default_parse_batch_fn
    
    # 推理模式
    model.eval()
    
    # 收集结果
    pred_list: List[torch.Tensor] = []
    label_list: List[torch.Tensor] = []
    metadata_list: List[Dict] = []  # [{stock_ids, date}, ...]
    
    has_metadata = None  # 延迟判断是否有元数据
    
    with torch.no_grad():
        for batch in loader:
            parsed = parse_batch_fn(batch)
            
            x = parsed['x']
            y = parsed['y']
            adj = parsed['adj']
            stock_ids = parsed['stock_ids']
            date = parsed['date']
            extra_kwargs = parsed.get('extra_forward_kwargs', {})
            
            if x is None:
                raise ValueError(
                    f"batch 解析后 x 为 None，batch 类型: {type(batch)}。"
                    "请提供 parse_batch_fn 或检查 DataLoader 格式。"
                )
            
            # 移动到设备
            x = x.to(device)
            if y is not None:
                y = y.to(device)
            if adj is not None:
                adj = adj.to(device)
            
            # 前向传播
            try:
                if adj is not None:
                    preds = model(x, adj=adj, **extra_kwargs)
                else:
                    preds = model(x, **extra_kwargs)
            except TypeError:
                # 模型不接受 adj 参数
                preds = model(x, **extra_kwargs)
            
            # 处理元组输出
            if isinstance(preds, tuple):
                preds = preds[0]
            
            # 确保维度
            preds = preds.detach().cpu()
            if preds.dim() == 1:
                preds = preds.unsqueeze(-1)
            
            pred_list.append(preds)
            
            # 标签
            if y is not None:
                y = y.detach().cpu()
                if y.dim() == 1:
                    y = y.unsqueeze(-1)
                label_list.append(y)
            
            # 元数据
            if has_metadata is None:
                has_metadata = (stock_ids is not None and date is not None)
            
            if has_metadata:
                n_samples = preds.shape[0]
                # stock_ids 可能是列表或张量
                if isinstance(stock_ids, torch.Tensor):
                    stock_ids = stock_ids.cpu().tolist()
                elif not isinstance(stock_ids, (list, tuple)):
                    stock_ids = [stock_ids] * n_samples
                
                # date 可能是单个值或列表
                if isinstance(date, (list, tuple)):
                    dates = date
                else:
                    dates = [date] * n_samples
                
                metadata_list.append({
                    'stock_ids': stock_ids,
                    'dates': dates
                })
    
    # 合并张量
    if len(pred_list) == 0:
        if return_tensor:
            return torch.tensor([]), torch.tensor([])
        else:
            return pd.DataFrame(), pd.DataFrame()
    
    all_preds = torch.cat(pred_list, dim=0)  # (N, F)
    all_labels = torch.cat(label_list, dim=0) if label_list else None  # (N, 1) or None
    
    # Tensor 模式
    if return_tensor:
        labels_out = all_labels if all_labels is not None else torch.tensor([])
        return all_preds, labels_out
    
    # DataFrame 模式
    n_samples = all_preds.shape[0]
    n_factors = all_preds.shape[1]
    
    # 构建基础数据
    pred_data = {}
    label_data = {}
    
    # 元数据列
    if has_metadata and metadata_list:
        all_stock_ids = []
        all_dates = []
        for meta in metadata_list:
            all_stock_ids.extend(meta['stock_ids'])
            all_dates.extend(meta['dates'])
        
        # 转换日期格式
        all_dates = [pd.to_datetime(d) if not isinstance(d, pd.Timestamp) else d for d in all_dates]
        
        pred_data['trade_date'] = all_dates
        pred_data['order_book_id'] = all_stock_ids
        label_data['trade_date'] = all_dates
        label_data['order_book_id'] = all_stock_ids
    else:
        # 无元数据，使用索引
        pred_data['sample_idx'] = list(range(n_samples))
        label_data['sample_idx'] = list(range(n_samples))
    
    # 预测值
    preds_np = all_preds.numpy()
    
    if reduce_factor is None:
        # 保留全因子
        for f_idx in range(n_factors):
            pred_data[f'pred_factor_{f_idx}'] = preds_np[:, f_idx]
        # pred 取第一个因子
        pred_data['pred'] = preds_np[:, 0]
    elif reduce_factor == 'mean':
        pred_data['pred'] = preds_np.mean(axis=1)
        # 如果多因子，也保留各因子列
        if n_factors > 1:
            for f_idx in range(n_factors):
                pred_data[f'pred_factor_{f_idx}'] = preds_np[:, f_idx]
    elif callable(reduce_factor):
        # 自定义聚合
        pred_data['pred'] = reduce_factor(all_preds).numpy()
        if n_factors > 1:
            for f_idx in range(n_factors):
                pred_data[f'pred_factor_{f_idx}'] = preds_np[:, f_idx]
    else:
        raise ValueError(f"reduce_factor 不支持: {reduce_factor}")
    
    pred_df = pd.DataFrame(pred_data)
    
    # 标签 DataFrame
    if return_label and all_labels is not None:
        labels_np = all_labels.numpy()
        label_data['label'] = labels_np.mean(axis=1) if labels_np.ndim > 1 else labels_np.flatten()
        label_df = pd.DataFrame(label_data)
    else:
        label_df = pd.DataFrame()
    
    return pred_df, label_df


def compute_ic(pred_df: pd.DataFrame, label_df: pd.DataFrame, 
               pred_col: str = 'pred', label_col: str = 'label',
               date_col: str = 'trade_date') -> pd.Series:
    """
    计算每日 IC（Information Coefficient）
    
    Args:
        pred_df: 预测 DataFrame，需含 trade_date, order_book_id, pred
        label_df: 标签 DataFrame，需含 trade_date, order_book_id, label
        pred_col: 预测值列名
        label_col: 标签列名
        date_col: 日期列名
        
    Returns:
        每日 IC 的 Series，index 为日期
    """
    # 合并
    if 'order_book_id' in pred_df.columns and 'order_book_id' in label_df.columns:
        merged = pred_df.merge(
            label_df[[date_col, 'order_book_id', label_col]],
            on=[date_col, 'order_book_id'],
            how='inner'
        )
    elif 'sample_idx' in pred_df.columns and 'sample_idx' in label_df.columns:
        merged = pred_df.merge(
            label_df[['sample_idx', label_col]],
            on='sample_idx',
            how='inner'
        )
        # 无日期，返回单一 IC
        return pd.Series({'all': merged[pred_col].corr(merged[label_col])})
    else:
        raise ValueError("pred_df 和 label_df 缺少匹配的 key 列")
    
    # 按日计算 IC
    daily_ic = merged.groupby(date_col).apply(
        lambda x: x[pred_col].corr(x[label_col])
    )
    
    return daily_ic


def compute_ic_stats(daily_ic: pd.Series) -> Dict[str, float]:
    """
    计算 IC 统计量
    
    Args:
        daily_ic: 每日 IC Series
        
    Returns:
        包含 ic_mean, ic_std, icir, ic_positive_ratio 的字典
    """
    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std()
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_positive_ratio = (daily_ic > 0).sum() / len(daily_ic) if len(daily_ic) > 0 else 0.0
    
    return {
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'ic_positive_ratio': ic_positive_ratio
    }
