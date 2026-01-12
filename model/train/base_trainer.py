"""
BaseTrainer - 训练基类

定义通用的训练循环、早停、检查点保存和日志逻辑。
所有训练器都应继承此基类。

核心设计:
1. TrainerArtifacts: 训练所需的所有组件（模型、优化器、加载器等）
2. TrainerConfig: 训练配置参数容器
3. TrainerCallback: 训练过程中的回调机制
4. BaseTrainer: 训练循环的抽象基类
"""

import abc
import logging
import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 使用相对导入（相对于 quantclassic 包）
# 移除降级版 BaseConfig，强制依赖正确的基类
try:
    from ...config.base_config import BaseConfig
except ImportError:
    # 直接运行脚本时的后备导入
    from config.base_config import BaseConfig


# ==================== 配置数据类 ====================

@dataclass
class TrainerConfig(BaseConfig):
    """
    训练器配置
    
    包含所有训练相关的超参数，通过配置驱动训练行为。
    
    Args:
        n_epochs: 训练轮数
        lr: 学习率
        weight_decay: L2 正则化系数
        early_stop: 早停耐心值
        optimizer: 优化器名称 ('adam', 'sgd', 'adamw')
        loss_fn: 损失函数名称 ('mse', 'mae', 'huber', 'ic')
        loss_kwargs: 损失函数额外参数
        use_scheduler: 是否使用学习率调度器
        scheduler_type: 调度器类型 ('plateau', 'cosine', 'step')
        scheduler_patience: ReduceLROnPlateau 的耐心值
        scheduler_factor: 学习率衰减因子
        scheduler_min_lr: 最小学习率
        lambda_corr: 相关性正则化权重
        checkpoint_dir: 检查点保存目录
        save_best_only: 是否只保存最佳模型
        verbose: 是否打印详细日志
        log_interval: 日志打印间隔（batch数）
    """
    # 基本训练参数
    n_epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 0.0
    early_stop: int = 20
    
    # 优化器配置
    optimizer: str = 'adam'
    
    # 损失函数配置
    loss_fn: str = 'mse'
    loss_kwargs: Dict[str, Any] = field(default_factory=dict)
    lambda_corr: float = 0.0
    
    # 学习率调度器配置
    use_scheduler: bool = True
    scheduler_type: str = 'plateau'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # 检查点配置
    checkpoint_dir: Optional[str] = None
    save_best_only: bool = True
    
    # 日志配置
    verbose: bool = True
    log_interval: int = 50
    
    def validate(self) -> bool:
        """验证配置有效性"""
        if self.n_epochs <= 0:
            raise ValueError("n_epochs 必须大于 0")
        if self.lr <= 0:
            raise ValueError("lr 必须大于 0")
        if self.early_stop < 0:
            raise ValueError("early_stop 不能为负数")
        if self.optimizer not in ['adam', 'sgd', 'adamw']:
            raise ValueError(f"不支持的优化器: {self.optimizer}")
        
        # 🆕 扩展损失函数支持列表，与 loss.get_loss_fn 保持一致
        supported_losses = [
            'mse', 'mae', 'huber', 'ic',  # 标准损失
            'mse_corr', 'mae_corr', 'huber_corr', 'ic_corr',  # 带相关性正则
            'combined', 'unified'  # 组合/统一损失
        ]
        if self.loss_fn not in supported_losses:
            raise ValueError(
                f"不支持的损失函数: {self.loss_fn}. "
                f"支持的损失: {', '.join(supported_losses)}"
            )
        return True


@dataclass
class TrainerArtifacts:
    """
    训练组件容器
    
    封装训练所需的所有组件，统一传递给 Trainer。
    
    Args:
        model: PyTorch 模型 (nn.Module)
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        criterion: 损失函数
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（可选）
        test_loader: 测试数据加载器（可选）
        device: 计算设备
        metrics: 评估指标字典（可选）
        callbacks: 回调列表（可选）
    """
    model: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    train_loader: DataLoader
    device: torch.device
    
    scheduler: Optional[Any] = None
    val_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None
    metrics: Optional[Dict[str, Callable]] = None
    callbacks: Optional[List['TrainerCallback']] = None


# ==================== 回调机制 ====================

class TrainerCallback(abc.ABC):
    """
    训练回调基类
    
    定义训练过程中的钩子函数，子类可重写以实现自定义行为。
    """
    
    def on_train_begin(self, trainer: 'BaseTrainer', **kwargs):
        """训练开始时调用"""
        pass
    
    def on_train_end(self, trainer: 'BaseTrainer', **kwargs):
        """训练结束时调用"""
        pass
    
    def on_epoch_begin(self, trainer: 'BaseTrainer', epoch: int, **kwargs):
        """每个 epoch 开始时调用"""
        pass
    
    def on_epoch_end(self, trainer: 'BaseTrainer', epoch: int, 
                     train_loss: float, val_loss: Optional[float] = None, **kwargs):
        """每个 epoch 结束时调用"""
        pass
    
    def on_batch_begin(self, trainer: 'BaseTrainer', batch_idx: int, **kwargs):
        """每个 batch 开始时调用"""
        pass
    
    def on_batch_end(self, trainer: 'BaseTrainer', batch_idx: int, 
                     loss: float, **kwargs):
        """每个 batch 结束时调用"""
        pass


class EarlyStoppingCallback(TrainerCallback):
    """
    早停回调
    
    当验证损失不再改善时停止训练。
    
    Args:
        patience: 等待改善的 epoch 数
        min_delta: 被视为改善的最小变化量
        mode: 'min' 或 'max'，监控指标的优化方向
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False
    
    def _is_improvement(self, score: float) -> bool:
        """检查是否有改善"""
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta
    
    def on_epoch_end(self, trainer: 'BaseTrainer', epoch: int,
                     train_loss: float, val_loss: Optional[float] = None, **kwargs):
        """检查是否应该早停"""
        score = val_loss if val_loss is not None else train_loss
        
        if self._is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                trainer.logger.info(
                    f"⚡ 早停触发 at epoch {epoch + 1} "
                    f"(best epoch: {self.best_epoch + 1}, best score: {self.best_score:.6f})"
                )


class CheckpointCallback(TrainerCallback):
    """
    检查点保存回调
    
    定期保存模型检查点。
    
    Args:
        checkpoint_dir: 保存目录
        save_best_only: 是否只保存最佳模型
        monitor: 监控的指标名称
        mode: 'min' 或 'max'
    """
    
    def __init__(self, checkpoint_dir: str, save_best_only: bool = True,
                 monitor: str = 'val_loss', mode: str = 'min'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        
        self.best_score = float('inf') if mode == 'min' else float('-inf')
    
    def _is_better(self, score: float) -> bool:
        """检查分数是否更好"""
        if self.mode == 'min':
            return score < self.best_score
        else:
            return score > self.best_score
    
    def on_epoch_end(self, trainer: 'BaseTrainer', epoch: int,
                     train_loss: float, val_loss: Optional[float] = None, **kwargs):
        """保存检查点"""
        score = val_loss if val_loss is not None else train_loss
        
        should_save = False
        if self.save_best_only:
            if self._is_better(score):
                self.best_score = score
                should_save = True
        else:
            should_save = True
        
        if should_save:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            trainer.save_checkpoint(checkpoint_path)
            
            if self.save_best_only:
                # 同时保存为 best.pth
                best_path = self.checkpoint_dir / "best.pth"
                trainer.save_checkpoint(best_path)


# ==================== 基础训练器 ====================

class BaseTrainer(abc.ABC):
    """
    训练基类
    
    定义通用的训练循环框架，子类只需实现 train_batch 和 validate_epoch。
    
    核心方法:
    - train(): 主训练循环
    - train_epoch(): 单个 epoch 训练
    - train_batch(): 单个 batch 训练（抽象，子类实现）
    - validate_epoch(): 验证一个 epoch（抽象，子类实现）
    - save_checkpoint(): 保存检查点
    - load_checkpoint(): 加载检查点
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        初始化训练器
        
        Args:
            model: PyTorch 模型
            config: 训练配置
            device: 计算设备
        """
        self.model = model
        self.config = config
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # 移动模型到设备
        self.model = self.model.to(self.device)
        
        # 初始化优化器和损失函数（延迟到 train 时创建）
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.criterion: Optional[nn.Module] = None
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.train_losses: List[float] = []
        self.valid_losses: List[float] = []
        self.lr_history: List[float] = []
        self.best_score = float('inf')
        self.best_epoch = 0
        
        # 回调
        self.callbacks: List[TrainerCallback] = []
        
        # 日志
        self.logger = self._setup_logger()
        
        self.logger.info(f"初始化 {self.__class__.__name__}:")
        self.logger.info(f"  设备: {self.device}")
        self.logger.info(f"  训练轮数: {config.n_epochs}")
        self.logger.info(f"  学习率: {config.lr}")
    
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
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        opt_name = self.config.optimizer.lower()
        params = self.model.parameters()
        lr = self.config.lr
        wd = self.config.weight_decay
        
        if opt_name == 'adam':
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        elif opt_name == 'sgd':
            return torch.optim.SGD(params, lr=lr, weight_decay=wd)
        elif opt_name == 'adamw':
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"不支持的优化器: {opt_name}")
    
    def _create_scheduler(self) -> Optional[Any]:
        """创建学习率调度器"""
        if not self.config.use_scheduler or self.optimizer is None:
            return None
        
        sched_type = self.config.scheduler_type.lower()
        
        if sched_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                min_lr=self.config.scheduler_min_lr,
                verbose=True
            )
        elif sched_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.n_epochs,
                eta_min=self.config.scheduler_min_lr
            )
        elif sched_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_patience,
                gamma=self.config.scheduler_factor
            )
        else:
            self.logger.warning(f"未知的调度器类型: {sched_type}")
            return None
    
    def _create_criterion(self) -> nn.Module:
        """创建损失函数"""
        loss_name = self.config.loss_fn.lower()
        loss_kwargs = self.config.loss_kwargs
        
        # 🆕 优先使用 loss 模块的 get_loss_fn 工厂函数
        try:
            from ..loss import get_loss_fn
            
            # 如果启用相关性正则化且损失名称不带 _corr
            if self.config.lambda_corr > 0 and not loss_name.endswith('_corr'):
                if loss_name in ['mse', 'mae', 'huber', 'ic']:
                    loss_name = f"{loss_name}_corr"
            
            return get_loss_fn(
                loss_type=loss_name,
                lambda_corr=self.config.lambda_corr,
                **loss_kwargs
            )
        except ImportError:
            self.logger.warning("无法导入 loss 模块，使用标准损失")
        except ValueError as e:
            self.logger.warning(f"get_loss_fn 不支持 {loss_name}: {e}")
        
        # 备选: 标准损失函数
        if loss_name in ['mse', 'mse_corr']:
            return nn.MSELoss()
        elif loss_name in ['mae', 'mae_corr']:
            return nn.L1Loss()
        elif loss_name in ['huber', 'huber_corr']:
            delta = loss_kwargs.get('delta', 1.0)
            return nn.HuberLoss(delta=delta)
        elif loss_name in ['ic', 'ic_corr']:
            # IC 损失回退到 MSE
            self.logger.warning(f"IC 损失需要 loss 模块支持，回退到 MSE")
            return nn.MSELoss()
        else:
            raise ValueError(f"不支持的损失函数: {loss_name}")
    
    def _step_scheduler(self, val_loss: Optional[float] = None):
        """更新学习率调度器"""
        if self.scheduler is None:
            return
        
        current_lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
        
        sched_type = self.config.scheduler_type.lower()
        if sched_type == 'plateau' and val_loss is not None:
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
        
        new_lr = self.optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            self.logger.info(f"  📉 学习率调整: {current_lr:.2e} → {new_lr:.2e}")
    
    def _parse_batch_data(self, batch_data) -> Tuple[Any, Any, Any, Any]:
        """
        解析 Batch 数据
        
        支持多种格式：
        - (x, y): 基础格式
        - (x, y, adj): 带邻接矩阵
        - (x, y, adj, idx): 带邻接矩阵和股票索引
        - dict: 字典格式
        
        Returns:
            (x, y, adj, idx) - 特征、标签、邻接矩阵、股票索引
        """
        if isinstance(batch_data, dict):
            x = batch_data.get('x') or batch_data.get('features') or batch_data.get('input')
            y = batch_data.get('y') or batch_data.get('labels') or batch_data.get('target')
            adj = batch_data.get('adj') or batch_data.get('adj_matrix')
            idx = batch_data.get('stock_idx') or batch_data.get('idx')
            return x, y, adj, idx
        
        if isinstance(batch_data, (list, tuple)):
            if len(batch_data) == 2:
                return batch_data[0], batch_data[1], None, None
            elif len(batch_data) == 3:
                return batch_data[0], batch_data[1], batch_data[2], None
            elif len(batch_data) >= 4:
                return batch_data[0], batch_data[1], batch_data[2], batch_data[3]
        
        return batch_data, None, None, None
    
    def add_callback(self, callback: TrainerCallback):
        """添加回调"""
        self.callbacks.append(callback)
    
    def _run_callbacks(self, method: str, **kwargs):
        """运行所有回调的指定方法"""
        for callback in self.callbacks:
            getattr(callback, method)(self, **kwargs)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        n_epochs: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        主训练循环
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            n_epochs: 训练轮数（覆盖配置）
            save_path: 模型保存路径
            
        Returns:
            训练结果字典
        """
        n_epochs = n_epochs or self.config.n_epochs
        
        # 初始化优化器、调度器、损失函数
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()
        if self.scheduler is None:
            self.scheduler = self._create_scheduler()
        if self.criterion is None:
            self.criterion = self._create_criterion()
        
        # 设置早停回调
        early_stopping = EarlyStoppingCallback(patience=self.config.early_stop)
        self.add_callback(early_stopping)
        
        # 设置检查点回调
        if self.config.checkpoint_dir:
            checkpoint_callback = CheckpointCallback(
                checkpoint_dir=self.config.checkpoint_dir,
                save_best_only=self.config.save_best_only
            )
            self.add_callback(checkpoint_callback)
        
        self.logger.info("=" * 60)
        self.logger.info(f"🚀 开始训练 ({n_epochs} epochs)")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        self._run_callbacks('on_train_begin')
        
        for epoch in range(n_epochs):
            self.current_epoch = epoch
            self._run_callbacks('on_epoch_begin', epoch=epoch)
            
            # 训练一个 epoch
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate_epoch(val_loader)
                self.valid_losses.append(val_loss)
            
            # 日志
            if val_loss is not None:
                self.logger.info(
                    f"Epoch {epoch + 1}/{n_epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )
            else:
                self.logger.info(
                    f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {train_loss:.6f}"
                )
            
            # 更新调度器
            self._step_scheduler(val_loss if val_loss is not None else train_loss)
            
            # 更新最佳分数
            current_score = val_loss if val_loss is not None else train_loss
            if current_score < self.best_score:
                self.best_score = current_score
                self.best_epoch = epoch
                if save_path:
                    self.save_checkpoint(save_path)
            
            # 运行回调
            self._run_callbacks(
                'on_epoch_end', 
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss
            )
            
            # 检查早停
            if early_stopping.should_stop:
                break
        
        elapsed = time.time() - start_time
        self._run_callbacks('on_train_end')
        
        self.logger.info("=" * 60)
        self.logger.info(f"✅ 训练完成!")
        self.logger.info(f"  总耗时: {elapsed:.1f}s")
        self.logger.info(f"  最佳 epoch: {self.best_epoch + 1}")
        self.logger.info(f"  最佳分数: {self.best_score:.6f}")
        self.logger.info("=" * 60)
        
        return {
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'lr_history': self.lr_history,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'elapsed_time': elapsed
        }
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        训练一个 epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            self._run_callbacks('on_batch_begin', batch_idx=batch_idx)
            
            loss = self.train_batch(batch_data)
            total_loss += loss
            n_batches += 1
            self.global_step += 1
            
            self._run_callbacks('on_batch_end', batch_idx=batch_idx, loss=loss)
            
            # 日志
            if self.config.verbose and batch_idx % self.config.log_interval == 0:
                self.logger.debug(f"  Batch {batch_idx}: loss={loss:.6f}")
        
        return total_loss / max(n_batches, 1)
    
    @abc.abstractmethod
    def train_batch(self, batch_data) -> float:
        """
        训练单个 batch（抽象方法，子类实现）
        
        Args:
            batch_data: DataLoader 返回的 batch 数据
            
        Returns:
            batch 损失值
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """
        验证一个 epoch（抽象方法，子类实现）
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            平均验证损失
        """
        raise NotImplementedError
    
    def save_checkpoint(self, path: Union[str, Path]):
        """保存检查点"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config.to_dict(),
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"💾 检查点已保存: {path}")
    
    def load_checkpoint(self, path: Union[str, Path]):
        """加载检查点"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"检查点不存在: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.valid_losses = checkpoint.get('valid_losses', [])
        self.best_score = checkpoint.get('best_score', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        
        self.logger.info(f"📂 检查点已加载: {path}")
    
    def get_model(self) -> nn.Module:
        """获取模型"""
        return self.model
    
    def get_state_dict(self) -> Dict:
        """获取模型状态字典"""
        return copy.deepcopy(self.model.state_dict())
    
    def load_state_dict(self, state_dict: Dict):
        """加载模型状态字典"""
        self.model.load_state_dict(state_dict)
