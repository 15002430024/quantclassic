"""
RollingWindowTrainer - 滚动窗口训练器

实现 Walk-Forward 验证策略的滚动窗口训练。
支持权重继承模式和独立训练模式。
"""

import copy
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union
)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from .base_trainer import BaseTrainer, TrainerConfig, TrainerCallback
from .simple_trainer import SimpleTrainer


@dataclass
class RollingTrainerConfig(TrainerConfig):
    """
    滚动窗口训练器配置
    
    继承自 TrainerConfig，增加滚动窗口特有的配置项。
    
    Args:
        weight_inheritance: 是否继承上一窗口的模型权重
        save_each_window: 是否保存每个窗口的模型
        reset_optimizer: 每个窗口是否重置优化器
        reset_scheduler: 每个窗口是否重置学习率调度器
        window_epochs: 每个窗口的训练轮数（覆盖 n_epochs）
    """
    # 滚动窗口特有配置
    weight_inheritance: bool = True
    save_each_window: bool = True
    reset_optimizer: bool = True
    reset_scheduler: bool = True
    window_epochs: Optional[int] = None  # None 表示使用 n_epochs
    
    def validate(self) -> bool:
        """验证配置"""
        super().validate()
        return True


# ==================== 数据结构定义 ====================

@dataclass
class WindowData:
    """
    单个滚动窗口的数据
    
    Args:
        window_id: 窗口标识（索引或日期）
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（可选）
        test_loader: 测试数据加载器（可选）
        metadata: 窗口元数据（如日期范围等）
    """
    window_id: Union[int, str]
    train_loader: Any
    val_loader: Optional[Any] = None
    test_loader: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class WindowResult:
    """
    单个窗口的训练结果
    
    Args:
        window_id: 窗口标识
        best_epoch: 最佳 epoch
        best_val_loss: 最佳验证损失
        train_losses: 训练损失历史
        val_losses: 验证损失历史
        predictions: 预测结果（可选）
        save_path: 模型保存路径（可选）
        elapsed_time: 训练耗时
        skipped: 是否跳过训练（使用已有模型）
    """
    window_id: Union[int, str]
    best_epoch: int = 0
    best_val_loss: float = 0.0
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    predictions: Optional[pd.DataFrame] = None
    save_path: Optional[str] = None
    elapsed_time: float = 0.0
    skipped: bool = False


# ==================== 滚动窗口训练器 ====================

class RollingWindowTrainer:
    """
    滚动窗口训练器
    
    实现 Walk-Forward 验证策略：
    1. 遍历每个滚动窗口
    2. 在当前窗口的训练集上训练模型
    3. 在当前窗口的测试集上预测
    4. （可选）将模型权重传递给下一窗口
    5. 合并所有窗口的预测结果
    
    Features:
    - 支持完全独立的滚动窗口训练
    - 支持增量训练（使用前一窗口模型初始化）
    - 自动管理模型保存和加载
    - 支持断点续训
    
    Example:
        >>> from quantclassic.model.train import RollingWindowTrainer, RollingTrainerConfig
        >>> 
        >>> config = RollingTrainerConfig(
        ...     n_epochs=20,
        ...     weight_inheritance=True,
        ...     save_each_window=True
        ... )
        >>> 
        >>> trainer = RollingWindowTrainer(
        ...     model_factory=lambda: MyModel(d_feat=20),
        ...     config=config,
        ...     device='cuda'
        ... )
        >>> 
        >>> results = trainer.train(rolling_loaders, save_dir='output/models')
        >>> all_predictions = trainer.get_all_predictions()
    """
    
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        config: Optional[RollingTrainerConfig] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        初始化滚动窗口训练器
        
        Args:
            model_factory: 模型工厂函数，每次调用返回一个新的 nn.Module 实例
            config: 滚动训练配置
            device: 计算设备
            **kwargs: 额外配置参数
        """
        self.model_factory = model_factory
        self.config = config or RollingTrainerConfig(**kwargs)
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # 日志
        self.logger = self._setup_logger()
        
        # 训练状态
        self.window_results: List[WindowResult] = []
        self.all_predictions: List[pd.DataFrame] = []
        self.current_model_state: Optional[Dict] = None
        self.current_model: Optional[nn.Module] = None
        self.current_optimizer_state: Optional[Dict] = None  # 🆕 优化器状态
        self.current_scheduler_state: Optional[Dict] = None  # 🆕 调度器状态
        
        self.logger.info("=" * 80)
        self.logger.info("🔄 初始化滚动窗口训练器")
        self.logger.info("=" * 80)
        self.logger.info(f"  设备: {self.device}")
        self.logger.info(f"  训练策略: {'继承权重 (Warm Start)' if self.config.weight_inheritance else '独立训练'}")
        self.logger.info(f"  每窗口保存: {'是' if self.config.save_each_window else '否'}")
        self.logger.info(f"  优化器复用: {'否' if self.config.reset_optimizer else '是'}")
        self.logger.info(f"  调度器复用: {'否' if self.config.reset_scheduler else '是'}")
    
    def _setup_logger(self) -> logging.Logger:
        """配置日志"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _get_model_for_window(self, window_idx: int) -> nn.Module:
        """
        获取当前窗口的模型
        
        Args:
            window_idx: 窗口索引
            
        Returns:
            PyTorch 模型
        """
        if self.config.weight_inheritance and self.current_model_state is not None and window_idx > 0:
            # 继承上一窗口的权重
            model = self.model_factory()
            model.to(self.device)
            try:
                model.load_state_dict(self.current_model_state)
                self.logger.info(f"  🔗 继承窗口 {window_idx} 的模型权重")
            except Exception as e:
                self.logger.warning(f"  ⚠️ 无法加载前一窗口权重: {e}")
        else:
            # 使用全新模型
            model = self.model_factory()
            model.to(self.device)
            self.logger.info(f"  🆕 使用全新模型权重")
        
        return model
    
    def _check_existing_model(self, save_path: str) -> Optional[Dict]:
        """
        检查是否存在已训练的模型
        
        Args:
            save_path: 模型保存路径
            
        Returns:
            如果存在，返回检查点字典；否则返回 None
        """
        if save_path and Path(save_path).exists():
            try:
                checkpoint = torch.load(save_path, map_location=self.device)
                self.logger.info(f"  ✓ 发现已训练模型: {save_path}")
                return checkpoint
            except Exception as e:
                self.logger.warning(f"  ⚠️ 加载已存在模型失败: {e}")
        return None
    
    def train(
        self,
        rolling_loaders,
        n_epochs: Optional[int] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        训练所有滚动窗口
        
        Args:
            rolling_loaders: 滚动窗口数据加载器集合
                - 可迭代对象，每个元素包含 train/val/test loader
                - 如 RollingDailyLoaderCollection
            n_epochs: 每个窗口的训练轮数（覆盖配置）
            save_dir: 模型保存目录
            
        Returns:
            训练结果汇总字典
        """
        n_epochs = n_epochs or self.config.window_epochs or self.config.n_epochs
        n_windows = len(rolling_loaders)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🚀 开始滚动窗口训练 (Walk-Forward)")
        self.logger.info("=" * 80)
        self.logger.info(f"  总窗口数: {n_windows}")
        self.logger.info(f"  每窗口轮数: {n_epochs}")
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"  保存目录: {save_dir}")
        
        start_time = time.time()
        self.window_results = []
        self.all_predictions = []
        
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
            
            # 检查是否存在已训练模型（断点续训）
            existing_checkpoint = self._check_existing_model(save_path)
            
            if existing_checkpoint:
                # 使用已有模型，跳过训练
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
                window_trainer = SimpleTrainer(model, self.config, str(self.device))
                
                # 🆕 复用优化器/调度器状态（如果配置了不重置且非首窗口）
                if not self.config.reset_optimizer and window_idx > 0 and self.current_optimizer_state:
                    try:
                        window_trainer._create_optimizer()  # 先创建优化器
                        window_trainer.optimizer.load_state_dict(self.current_optimizer_state)
                        self.logger.info("  🔗 复用上一窗口的优化器状态")
                    except Exception as e:
                        self.logger.warning(f"  ⚠️ 无法加载优化器状态: {e}")
                
                if not self.config.reset_scheduler and window_idx > 0 and self.current_scheduler_state:
                    try:
                        if window_trainer.scheduler is None:
                            window_trainer._create_scheduler()  # 先创建调度器
                        window_trainer.scheduler.load_state_dict(self.current_scheduler_state)
                        self.logger.info("  🔗 复用上一窗口的调度器状态")
                    except Exception as e:
                        self.logger.warning(f"  ⚠️ 无法加载调度器状态: {e}")
                
                # 日志
                self.logger.info(f"  训练集: {len(train_loader)} batches")
                if val_loader:
                    self.logger.info(f"  验证集: {len(val_loader)} batches")
                
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
                
                # 更新当前模型
                model = window_trainer.get_model()
            
            # 保存模型状态（用于下一窗口继承）
            if self.config.weight_inheritance:
                self.current_model_state = copy.deepcopy(model.state_dict())
            self.current_model = model
            
            # 🆕 保存优化器/调度器状态（用于下一窗口复用）
            if not window_result.skipped:  # 只有训练过的窗口才保存
                if not self.config.reset_optimizer and window_trainer.optimizer:
                    self.current_optimizer_state = copy.deepcopy(window_trainer.optimizer.state_dict())
                if not self.config.reset_scheduler and window_trainer.scheduler:
                    self.current_scheduler_state = copy.deepcopy(window_trainer.scheduler.state_dict())
            
            # 在测试集上预测
            if test_loader and len(test_loader) > 0:
                self.logger.info(f"  测试集: {len(test_loader)} batches")
                predictions = self._predict_window(model, test_loader, window_idx)
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
        
        # 汇总统计
        elapsed = time.time() - start_time
        summary = self._build_summary(elapsed)
        
        self._print_summary(summary)
        
        return summary
    
    def _predict_window(
        self,
        model: nn.Module,
        test_loader,
        window_idx: int
    ) -> Optional[pd.DataFrame]:
        """
        在测试集上预测
        
        Args:
            model: 训练好的模型
            test_loader: 测试数据加载器
            window_idx: 窗口索引
            
        Returns:
            预测结果 DataFrame
        """
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                # 解析 batch
                if isinstance(batch_data, (list, tuple)):
                    if len(batch_data) >= 5:
                        # DailyGraphDataLoader 格式
                        x, y, adj, stock_ids, dates = batch_data[:5]
                    elif len(batch_data) >= 2:
                        x, y = batch_data[0], batch_data[1]
                        adj = batch_data[2] if len(batch_data) > 2 else None
                        stock_ids = batch_data[3] if len(batch_data) > 3 else None
                        dates = None
                    else:
                        continue
                else:
                    x = batch_data.get('x') or batch_data.get('features')
                    y = batch_data.get('y') or batch_data.get('labels')
                    adj = batch_data.get('adj')
                    stock_ids = batch_data.get('stock_ids')
                    dates = batch_data.get('dates')
                
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
                
                # 构建预测记录
                for i in range(len(pred)):
                    record = {
                        'pred': pred[i] if pred.ndim == 1 else pred[i].item(),
                        'window_idx': window_idx + 1
                    }
                    if y is not None:
                        y_np = y.cpu().numpy() if torch.is_tensor(y) else y
                        record['y_true'] = y_np[i] if y_np.ndim == 1 else y_np[i].item()
                    if stock_ids is not None:
                        record['order_book_id'] = stock_ids[i] if isinstance(stock_ids, list) else stock_ids[i].item()
                    if dates is not None:
                        record['trade_date'] = dates if isinstance(dates, str) else str(dates)
                    
                    predictions.append(record)
        
        if predictions:
            return pd.DataFrame(predictions)
        return None
    
    def _build_summary(self, elapsed: float) -> Dict[str, Any]:
        """构建训练汇总"""
        train_losses = [
            r.train_losses[-1] for r in self.window_results
            if r.train_losses and not r.skipped
        ]
        val_losses = [
            r.best_val_loss for r in self.window_results
            if r.best_val_loss > 0
        ]
        best_epochs = [r.best_epoch for r in self.window_results]
        
        total_preds = sum(
            len(df) for df in self.all_predictions
        ) if self.all_predictions else 0
        
        return {
            'n_windows': len(self.window_results),
            'elapsed_time': elapsed,
            'avg_train_loss': float(np.mean(train_losses)) if train_losses else 0.0,
            'avg_val_loss': float(np.mean(val_losses)) if val_losses else 0.0,
            'avg_best_epoch': float(np.mean(best_epochs)) if best_epochs else 0.0,
            'total_predictions': total_preds,
            'window_results': self.window_results
        }
    
    def _print_summary(self, summary: Dict[str, Any]):
        """打印训练汇总"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 滚动窗口训练汇总")
        self.logger.info("=" * 80)
        self.logger.info(f"  总窗口: {summary['n_windows']}")
        self.logger.info(f"  总耗时: {summary['elapsed_time']:.1f}s ({summary['elapsed_time']/60:.1f} min)")
        self.logger.info(f"  平均训练损失: {summary['avg_train_loss']:.6f}")
        self.logger.info(f"  平均验证损失: {summary['avg_val_loss']:.6f}")
        self.logger.info(f"  平均最佳轮数: {summary['avg_best_epoch']:.1f}")
        self.logger.info(f"  总预测样本: {summary['total_predictions']:,}")
    
    def get_all_predictions(self) -> pd.DataFrame:
        """
        获取所有窗口的合并预测结果
        
        Returns:
            合并的 DataFrame，包含列:
            - trade_date: 交易日期（如果有）
            - order_book_id: 股票代码（如果有）
            - pred: 预测值
            - y_true: 真实标签（如果有）
            - window_idx: 窗口索引
        """
        if not self.all_predictions:
            return pd.DataFrame()
        
        combined = pd.concat(self.all_predictions, ignore_index=True)
        
        # 去重：如果有重叠，保留最后一个窗口的预测
        if 'trade_date' in combined.columns and 'order_book_id' in combined.columns:
            combined = combined.sort_values(['trade_date', 'order_book_id', 'window_idx'])
            combined = combined.drop_duplicates(
                subset=['trade_date', 'order_book_id'],
                keep='last'
            )
        
        return combined.reset_index(drop=True)
    
    def get_window_predictions(self, window_idx: int) -> Optional[pd.DataFrame]:
        """获取指定窗口的预测结果"""
        for result in self.window_results:
            if result.window_id == window_idx + 1:
                return result.predictions
        return None
    
    def get_current_model(self) -> Optional[nn.Module]:
        """获取当前（最后一个窗口）的模型"""
        return self.current_model
    
    def get_summary(self) -> Dict[str, Any]:
        """获取训练汇总（兼容旧接口）"""
        if not self.window_results:
            return {}
        return self._build_summary(0)
