"""
RollingDailyTrainer - 日级滚动训练器

基于 RollingWindowTrainer，专门处理日级粒度的滚动训练。
增加显存管理，支持高频模型切换。
"""

import gc
import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Union
)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from .base_trainer import TrainerConfig
from .rolling_window_trainer import (
    RollingWindowTrainer, 
    RollingTrainerConfig,
    WindowResult
)


@dataclass
class DailyRollingConfig(RollingTrainerConfig):
    """
    日级滚动训练配置
    
    继承 RollingTrainerConfig，增加显存管理相关配置。
    
    Args:
        gc_interval: 每隔多少窗口进行一次垃圾回收（0表示每窗口都回收）
        offload_to_cpu: 切窗时是否将模型移到CPU
        clear_cache_on_window_end: 窗口结束时是否清理CUDA缓存
    """
    gc_interval: int = 1
    offload_to_cpu: bool = True
    clear_cache_on_window_end: bool = True


class RollingDailyTrainer(RollingWindowTrainer):
    """
    日级滚动训练器
    
    基于 RollingWindowTrainer，专门处理日级粒度的滚动训练。
    主要增强:
    1. 显存管理：切窗时释放显存，避免OOM
    2. 日级窗口支持：适配日批次数据加载器
    3. 兼容性：保持与旧 rolling_daily_trainer.py 接口兼容
    
    Example:
        >>> from quantclassic.model.train import RollingDailyTrainer, DailyRollingConfig
        >>> 
        >>> config = DailyRollingConfig(
        ...     n_epochs=20,
        ...     weight_inheritance=True,
        ...     gc_interval=5
        ... )
        >>> 
        >>> trainer = RollingDailyTrainer(
        ...     model_factory=lambda: HybridGraphModel.from_config(cfg, d_feat=20).model,
        ...     config=config,
        ...     device='cuda'
        ... )
        >>> 
        >>> # rolling_loaders 来自 DataManager.create_rolling_daily_loaders()
        >>> results = trainer.fit(rolling_loaders, save_dir='output/models')
    """
    
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        config: Optional[DailyRollingConfig] = None,
        device: Optional[str] = None,
        warm_start: Optional[bool] = None,  # 兼容旧参数名
        save_each_window: Optional[bool] = None,  # 兼容旧参数名
        **kwargs
    ):
        """
        初始化日级滚动训练器
        
        Args:
            model_factory: 模型工厂函数
            config: 训练配置
            device: 计算设备
            warm_start: 是否继承权重（兼容旧API）
            save_each_window: 是否保存每窗口模型（兼容旧API）
            **kwargs: 其他配置参数
        """
        # 处理兼容性参数
        if config is None:
            config = DailyRollingConfig(**kwargs)
        
        if warm_start is not None:
            config.weight_inheritance = warm_start
        if save_each_window is not None:
            config.save_each_window = save_each_window
        
        super().__init__(model_factory, config, device)
        
        # 日级特有状态
        self._window_count = 0
    
    def _cleanup_memory(self, force: bool = False):
        """
        清理显存
        
        Args:
            force: 是否强制清理（忽略gc_interval）
        """
        self._window_count += 1
        gc_interval = getattr(self.config, 'gc_interval', 1)
        
        if force or gc_interval == 0 or self._window_count % gc_interval == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            self.logger.debug("  🧹 已清理显存")
    
    def _offload_model(self, model: nn.Module):
        """
        将模型移到CPU
        
        Args:
            model: 待卸载的模型
        """
        if getattr(self.config, 'offload_to_cpu', True):
            model.cpu()
            self.logger.debug("  📤 模型已移至CPU")
    
    def train(
        self,
        rolling_loaders,
        n_epochs: Optional[int] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        训练所有滚动窗口（覆盖父类以添加显存管理）
        
        Args:
            rolling_loaders: 滚动窗口数据加载器集合
            n_epochs: 每个窗口的训练轮数
            save_dir: 模型保存目录
            
        Returns:
            训练结果汇总字典
        """
        n_epochs = n_epochs or self.config.window_epochs or self.config.n_epochs
        n_windows = len(rolling_loaders)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🚀 开始日级滚动窗口训练 (Walk-Forward)")
        self.logger.info("=" * 80)
        self.logger.info(f"  总窗口数: {n_windows}")
        self.logger.info(f"  每窗口轮数: {n_epochs}")
        self.logger.info(f"  显存清理间隔: {getattr(self.config, 'gc_interval', 1)}")
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"  保存目录: {save_dir}")
        
        start_time = time.time()
        self.window_results = []
        self.all_predictions = []
        self._window_count = 0
        
        for window_idx, loaders in enumerate(rolling_loaders):
            self.logger.info("\n" + "-" * 60)
            self.logger.info(f"📅 窗口 {window_idx + 1}/{n_windows}")
            self.logger.info("-" * 60)
            
            # 解析 loaders
            train_loader = loaders.train if hasattr(loaders, 'train') else loaders[0]
            val_loader = loaders.val if hasattr(loaders, 'val') else (loaders[1] if len(loaders) > 1 else None)
            test_loader = loaders.test if hasattr(loaders, 'test') else (loaders[2] if len(loaders) > 2 else None)
            
            # 检查训练集是否为空
            if train_loader is None or len(train_loader) == 0:
                self.logger.warning(f"  ⚠️ 窗口 {window_idx + 1} 训练集为空，跳过")
                continue
            
            # 确定保存路径
            save_path = None
            if save_dir and self.config.save_each_window:
                save_path = str(save_dir / f"window_{window_idx + 1}.pth")
            
            # 检查是否存在已训练模型
            existing_checkpoint = self._check_existing_model(save_path)
            
            if existing_checkpoint:
                # 使用已有模型
                model = self.model_factory()
                model.to(self.device)
                model.load_state_dict(existing_checkpoint['model_state_dict'])
                
                window_result = WindowResult(
                    window_id=window_idx + 1,
                    best_epoch=existing_checkpoint.get('best_epoch', 0),
                    best_val_loss=existing_checkpoint.get('best_score', 0.0),
                    save_path=save_path,
                    skipped=True
                )
            else:
                # 训练新模型
                window_start = time.time()
                
                # 获取模型
                model = self._get_model_for_window(window_idx)
                
                # 创建窗口训练器
                from .simple_trainer import SimpleTrainer
                window_trainer = SimpleTrainer(model, self.config, str(self.device))
                
                # 重置优化器和调度器
                if self.config.reset_optimizer or window_idx == 0:
                    window_trainer.optimizer = None
                if self.config.reset_scheduler or window_idx == 0:
                    window_trainer.scheduler = None
                
                # 日志
                self.logger.info(f"  训练天数: {len(train_loader)}")
                if val_loader:
                    self.logger.info(f"  验证天数: {len(val_loader)}")
                
                # 训练
                train_result = window_trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    n_epochs=n_epochs,
                    save_path=save_path
                )
                
                window_elapsed = time.time() - window_start
                
                window_result = WindowResult(
                    window_id=window_idx + 1,
                    best_epoch=train_result['best_epoch'],
                    best_val_loss=train_result['best_score'],
                    train_losses=train_result['train_losses'],
                    val_losses=train_result['valid_losses'],
                    save_path=save_path,
                    elapsed_time=window_elapsed,
                    skipped=False
                )
                
                model = window_trainer.get_model()
            
            # 保存模型状态（用于下一窗口继承）
            if self.config.weight_inheritance:
                self.current_model_state = copy.deepcopy(model.state_dict())
            
            # 在测试集上预测
            if test_loader and len(test_loader) > 0:
                self.logger.info(f"  测试天数: {len(test_loader)}")
                predictions = self._predict_daily_window(model, test_loader, window_idx)
                if predictions is not None:
                    window_result.predictions = predictions
                    self.all_predictions.append(predictions)
                    self.logger.info(f"  预测样本: {len(predictions):,}")
            
            self.window_results.append(window_result)
            
            self.logger.info(
                f"✅ 窗口 {window_idx + 1} 完成 | "
                f"best_epoch={window_result.best_epoch + 1} | "
                f"best_val_loss={window_result.best_val_loss:.6f}"
            )
            
            # 显存管理：将模型移到CPU并清理缓存
            if getattr(self.config, 'clear_cache_on_window_end', True):
                self._offload_model(model)
                self._cleanup_memory()
            
            # 保持当前模型引用（在CPU上）
            self.current_model = model
        
        # 汇总统计
        elapsed = time.time() - start_time
        summary = self._build_summary(elapsed)
        
        self._print_summary(summary)
        
        return summary
    
    def _predict_daily_window(
        self,
        model: nn.Module,
        test_loader,
        window_idx: int
    ) -> Optional[pd.DataFrame]:
        """
        日级窗口预测（适配 DailyGraphDataLoader）
        
        Args:
            model: 训练好的模型
            test_loader: 测试数据加载器
            window_idx: 窗口索引
            
        Returns:
            预测结果 DataFrame
        """
        model.eval()
        model.to(self.device)
        
        all_records = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                # 解析 DailyGraphDataLoader 格式: (X, y, adj, stock_ids, date)
                if isinstance(batch_data, (list, tuple)):
                    if len(batch_data) >= 5:
                        x, y, adj, stock_ids, trade_date = batch_data[:5]
                    elif len(batch_data) >= 3:
                        x, y, adj = batch_data[:3]
                        stock_ids = batch_data[3] if len(batch_data) > 3 else None
                        trade_date = batch_data[4] if len(batch_data) > 4 else None
                    else:
                        x, y = batch_data[:2]
                        adj, stock_ids, trade_date = None, None, None
                else:
                    x = batch_data.get('x') or batch_data.get('features')
                    y = batch_data.get('y') or batch_data.get('labels')
                    adj = batch_data.get('adj')
                    stock_ids = batch_data.get('stock_ids')
                    trade_date = batch_data.get('trade_date')
                
                x = x.to(self.device)
                if adj is not None:
                    adj = adj.to(self.device)
                
                # 前向传播
                try:
                    pred = model(x, adj=adj) if adj is not None else model(x)
                except TypeError:
                    pred = model(x)
                
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                pred = pred.cpu().numpy()
                
                # 构建记录
                n_samples = len(pred)
                
                # 处理标签
                if y is not None:
                    y_np = y.cpu().numpy() if torch.is_tensor(y) else np.array(y)
                else:
                    y_np = None
                
                for i in range(n_samples):
                    record = {
                        'pred': float(pred[i]) if pred.ndim == 1 else float(pred[i].item()),
                        'window_idx': window_idx + 1
                    }
                    
                    if y_np is not None:
                        record['y_true'] = float(y_np[i]) if y_np.ndim == 1 else float(y_np[i].item())
                    
                    if stock_ids is not None:
                        if isinstance(stock_ids, (list, np.ndarray)):
                            record['order_book_id'] = stock_ids[i]
                        elif torch.is_tensor(stock_ids):
                            record['order_book_id'] = stock_ids[i].item()
                    
                    if trade_date is not None:
                        if isinstance(trade_date, str):
                            record['trade_date'] = trade_date
                        elif hasattr(trade_date, 'strftime'):
                            record['trade_date'] = trade_date.strftime('%Y-%m-%d')
                        else:
                            record['trade_date'] = str(trade_date)
                    
                    all_records.append(record)
        
        if all_records:
            return pd.DataFrame(all_records)
        return None
    
    # ==================== 兼容旧API ====================
    
    def fit(
        self,
        rolling_loaders,
        n_epochs: Optional[int] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        训练方法（兼容旧 rolling_daily_trainer.py 的 fit 接口）
        
        Args:
            rolling_loaders: 滚动窗口数据加载器
            n_epochs: 每窗口轮数
            save_dir: 保存目录
            
        Returns:
            训练结果汇总
        """
        return self.train(rolling_loaders, n_epochs, save_dir)


# ==================== 工厂函数 ====================

def create_rolling_daily_trainer(
    model_class,
    model_config: Dict[str, Any],
    d_feat: int,
    device: str = 'cuda',
    warm_start: bool = True,
    save_each_window: bool = True,
    **trainer_kwargs
) -> RollingDailyTrainer:
    """
    创建日级滚动训练器的便捷函数
    
    Args:
        model_class: 模型类（如 HybridGraphModel）
        model_config: 模型配置字典
        d_feat: 特征维度
        device: 计算设备
        warm_start: 是否继承权重
        save_each_window: 是否保存每窗口模型
        **trainer_kwargs: 其他训练配置
        
    Returns:
        RollingDailyTrainer 实例
    """
    def model_factory():
        if hasattr(model_class, 'from_config'):
            return model_class.from_config(model_config, d_feat=d_feat).model
        else:
            return model_class(d_feat=d_feat, **model_config)
    
    config = DailyRollingConfig(
        weight_inheritance=warm_start,
        save_each_window=save_each_window,
        **trainer_kwargs
    )
    
    return RollingDailyTrainer(
        model_factory=model_factory,
        config=config,
        device=device
    )
