"""
Base Model Classes - 模型基类

参照 Qlib 的设计，提供标准化的模型接口
"""

import abc
import torch
import pickle
from typing import Any, Dict, Optional, Union
from pathlib import Path
import logging


class BaseModel(abc.ABC):
    """
    基础模型类 - 所有模型的抽象基类
    
    参照 Qlib 的 BaseModel 设计，定义最基本的模型接口
    """
    
    @abc.abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """
        模型预测方法
        
        Returns:
            预测结果
        """
        raise NotImplementedError("predict method must be implemented")
    
    def __call__(self, *args, **kwargs) -> Any:
        """
        使模型可以像函数一样调用
        
        Example:
            prediction = model(x_test)  # 等价于 model.predict(x_test)
        """
        return self.predict(*args, **kwargs)
    
    def save(self, save_path: str):
        """
        保存模型到磁盘
        
        Args:
            save_path: 保存路径
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 获取所有不以 '_' 开头的属性（公共属性）
        state = {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, load_path: str):
        """
        从磁盘加载模型
        
        Args:
            load_path: 加载路径
        """
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
        for key, value in state.items():
            setattr(self, key, value)


class Model(BaseModel):
    """
    可训练模型类 - 继承自 BaseModel
    
    参照 Qlib 的 Model 设计，增加训练接口
    """
    
    def __init__(self):
        """初始化模型"""
        self.fitted = False
        self.logger = self._setup_logger()
    
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
    
    @abc.abstractmethod
    def fit(self, train_data, valid_data=None, **kwargs):
        """
        训练模型
        
        Args:
            train_data: 训练数据
            valid_data: 验证数据（可选）
            **kwargs: 其他训练参数
            
        Note:
            训练后应设置 self.fitted = True
        """
        raise NotImplementedError("fit method must be implemented")
    
    @abc.abstractmethod
    def predict(self, test_data, **kwargs) -> Any:
        """
        预测
        
        Args:
            test_data: 测试数据
            **kwargs: 其他预测参数
            
        Returns:
            预测结果
            
        Raises:
            ValueError: 如果模型未训练
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        raise NotImplementedError("predict method must be implemented")


class PyTorchModel(Model):
    """
    PyTorch 模型基类 - 为深度学习模型提供通用功能
    
    封装 PyTorch 模型的常用操作：
    - 自动 GPU 管理
    - 模型保存/加载
    - 早停机制
    - 训练循环
    - 🆕 学习率自动调整 (ReduceLROnPlateau)
    - 🆕 相关性正则化 (Correlation Regularization)
    
    ⚠️ 相关性正则化使用说明:
    --------------------------------
    要启用相关性正则化 (lambda_corr > 0)，模型必须满足以下要求：
    
    1. 模型的 forward() 方法必须支持 `return_hidden` 参数：
       ```python
       def forward(self, x, return_hidden=False):
           ...
           if return_hidden:
               return pred, hidden_features  # 返回 (预测值, 隐藏特征)
           return pred
       ```
    
    2. hidden_features 应为进入输出层前的融合特征，形状为 [batch_size, hidden_dim]
    
    3. 如果模型不支持 return_hidden，系统会自动降级并发出警告，
       但相关性正则化将不会生效。
    
    示例:
        # 启用相关性正则化
        model = MyModel(lambda_corr=0.01, ...)
        
        # 不启用（默认）
        model = MyModel(lambda_corr=0.0, ...)
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        n_epochs: int = 100,
        batch_size: int = 256,
        lr: float = 0.001,
        early_stop: int = 20,
        optimizer: str = 'adam',
        loss_fn: str = 'mse',
        # 🆕 学习率调度器参数
        use_scheduler: bool = True,
        scheduler_type: str = 'plateau',  # 'plateau' | 'cosine' | 'step' | None
        scheduler_patience: int = 5,      # ReduceLROnPlateau 的耐心值
        scheduler_factor: float = 0.5,    # 学习率衰减因子
        scheduler_min_lr: float = 1e-6,   # 最小学习率
        # 🆕 相关性正则化参数
        lambda_corr: float = 0.0,         # 相关性正则化权重，0 表示不使用
                                          # ⚠️ 设置 > 0 时需要模型支持 return_hidden 参数
        **kwargs
    ):
        """
        Args:
            device: 设备 ('cuda', 'cpu' 或 None 自动检测)
            n_epochs: 训练轮数
            batch_size: 批量大小
            lr: 学习率
            early_stop: 早停耐心值
            optimizer: 优化器名称
            loss_fn: 损失函数名称
        """
        super().__init__()
        
        # 设备管理
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 训练参数
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.early_stop = early_stop
        self.optimizer_name = optimizer.lower()
        self.loss_fn_name = loss_fn.lower()
        
        # 模型和优化器（子类中初始化）
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None  # 🆕 学习率调度器
        
        # 🆕 学习率调度器配置
        self.use_scheduler = use_scheduler
        self.scheduler_type = scheduler_type.lower() if scheduler_type else None
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_min_lr = scheduler_min_lr
        
        # 🆕 相关性正则化配置
        self.lambda_corr = lambda_corr
        self._use_corr_loss = lambda_corr > 0  # 标志位：是否使用相关性正则化损失
        
        # 训练历史
        self.train_losses = []
        self.valid_losses = []
        self.lr_history = []  # 🆕 记录学习率变化
        self.best_score = float('-inf')
        self.best_epoch = 0
        
        self.logger.info(f"初始化 PyTorchModel:")
        self.logger.info(f"  设备: {self.device}")
        self.logger.info(f"  训练轮数: {n_epochs}")
        self.logger.info(f"  批量大小: {batch_size}")
        self.logger.info(f"  学习率: {lr}")
        if use_scheduler:
            self.logger.info(f"  学习率调度器: {scheduler_type} (patience={scheduler_patience}, factor={scheduler_factor})")
        if lambda_corr > 0:
            self.logger.info(f"  🆕 相关性正则化: lambda={lambda_corr}")
    
    def _get_optimizer(self):
        """创建优化器"""
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer_name == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    def _get_scheduler(self):
        """
        🆕 创建学习率调度器
        
        支持的调度器类型:
        - 'plateau': ReduceLROnPlateau - 当验证损失停止下降时降低学习率
        - 'cosine': CosineAnnealingLR - 余弦退火
        - 'step': StepLR - 固定步长衰减
        
        Returns:
            学习率调度器或 None
        """
        if not self.use_scheduler or self.scheduler_type is None:
            return None
        
        if self.optimizer is None:
            self.logger.warning("优化器未初始化，无法创建调度器")
            return None
        
        if self.scheduler_type == 'plateau':
            # 当验证损失停止下降时自动降低学习率
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',                      # 监控损失（越小越好）
                factor=self.scheduler_factor,    # 学习率乘以这个因子
                patience=self.scheduler_patience, # 等待多少个 epoch
                min_lr=self.scheduler_min_lr,    # 最小学习率
                verbose=True                     # 打印学习率变化
            )
            self.logger.info(f"  ✅ 创建 ReduceLROnPlateau 调度器")
            
        elif self.scheduler_type == 'cosine':
            # 余弦退火
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.n_epochs,
                eta_min=self.scheduler_min_lr
            )
            self.logger.info(f"  ✅ 创建 CosineAnnealingLR 调度器")
            
        elif self.scheduler_type == 'step':
            # 固定步长衰减
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.scheduler_patience,
                gamma=self.scheduler_factor
            )
            self.logger.info(f"  ✅ 创建 StepLR 调度器")
            
        else:
            self.logger.warning(f"未知的调度器类型: {self.scheduler_type}，不使用调度器")
            return None
        
        return scheduler
    
    def _step_scheduler(self, val_loss: float = None):
        """
        🆕 更新学习率调度器
        
        Args:
            val_loss: 验证损失（ReduceLROnPlateau 需要）
        """
        if self.scheduler is None:
            return
        
        # 记录当前学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
        
        # 根据调度器类型调用 step
        if self.scheduler_type == 'plateau':
            if val_loss is not None:
                self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
        
        # 检查学习率是否变化
        new_lr = self.optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            self.logger.info(f"  📉 学习率调整: {current_lr:.2e} → {new_lr:.2e}")
    
    def _get_loss_fn(self):
        """
        创建损失函数（🆕 支持相关性正则化）
        
        如果 lambda_corr > 0，使用带相关性正则化的损失函数，
        否则使用标准损失函数。
        
        Returns:
            损失函数模块
        """
        # 如果启用相关性正则化，使用 loss.py 中的损失函数
        if self._use_corr_loss:
            try:
                from .loss import get_loss_fn
                loss_type = self.loss_fn_name
                if loss_type in ['mse', 'mae', 'huber', 'ic']:
                    loss_type = f"{loss_type}_corr"
                return get_loss_fn(loss_type=loss_type, lambda_corr=self.lambda_corr)
            except ImportError:
                self.logger.warning("无法导入 loss 模块，回退到标准损失函数")
                self._use_corr_loss = False
        
        # 标准损失函数
        if self.loss_fn_name == 'mse':
            return torch.nn.MSELoss()
        elif self.loss_fn_name == 'mae':
            return torch.nn.L1Loss()
        elif self.loss_fn_name == 'huber':
            return torch.nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss function: {self.loss_fn_name}")
    
    def save_model(self, save_path: str, save_optimizer: bool = False):
        """
        保存 PyTorch 模型
        
        Args:
            save_path: 保存路径
            save_optimizer: 是否保存优化器状态
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'config': {
                'n_epochs': self.n_epochs,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'early_stop': self.early_stop,
                'optimizer': self.optimizer_name,
                'loss_fn': self.loss_fn_name,
            },
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
        }
        
        if save_optimizer and self.optimizer is not None:
            state['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(state, save_path)
        self.logger.info(f"模型已保存: {save_path}")
    
    def load_model(self, load_path: str, load_optimizer: bool = False):
        """
        加载 PyTorch 模型
        
        Args:
            load_path: 加载路径
            load_optimizer: 是否加载优化器状态
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            if self.optimizer is None:
                self.optimizer = self._get_optimizer()
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复训练历史
        self.train_losses = checkpoint.get('train_losses', [])
        self.valid_losses = checkpoint.get('valid_losses', [])
        self.best_score = checkpoint.get('best_score', float('-inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        
        self.fitted = True
        self.logger.info(f"模型已加载: {load_path}")
    
    def _train_epoch(self, train_loader):
        """
        训练一个 epoch（🆕 支持相关性正则化）
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            平均训练损失
            
        Note:
            如果启用了相关性正则化 (_use_corr_loss=True)，
            模型需要返回 (predictions, hidden_features) 元组。
            可通过 model(x, return_hidden=True) 实现。
            如果模型不支持，会自动降级并发出警告。
        """
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # 🆕 根据是否使用相关性正则化决定前向传播方式
            if self._use_corr_loss:
                # 需要隐藏特征用于相关性正则化
                try:
                    output = self.model(batch_x, return_hidden=True)
                    if isinstance(output, tuple) and len(output) >= 2:
                        predictions = output[0]
                        hidden_features = output[-1]  # 最后一个是融合特征
                    else:
                        # 模型返回格式不符合预期，降级处理
                        self.logger.warning(
                            "⚠️ 模型返回格式不符合预期（应返回 (pred, hidden) 元组），"
                            "相关性正则化已禁用。"
                        )
                        self._use_corr_loss = False
                        self._corr_loss_disabled_logged = True
                        predictions = output if not isinstance(output, tuple) else output[0]
                        hidden_features = None
                except TypeError as e:
                    # 🆕 模型不支持 return_hidden 参数，自动降级
                    if 'return_hidden' in str(e):
                        self.logger.warning(
                            f"⚠️ 模型不支持 return_hidden 参数，相关性正则化已自动禁用。"
                            f"\n   要启用相关性正则化，请确保模型的 forward() 方法支持 return_hidden=True 参数。"
                        )
                        self._use_corr_loss = False
                        self._corr_loss_disabled_logged = True
                        # 回退到普通前向传播
                        predictions = self.model(batch_x)
                        hidden_features = None
                    else:
                        # 其他 TypeError，重新抛出
                        raise
                
                # 计算损失
                if self._use_corr_loss and hidden_features is not None:
                    loss = self.criterion(predictions, batch_y, hidden_features)
                else:
                    loss = self.criterion(predictions, batch_y)
            else:
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0
    
    def _valid_epoch(self, valid_loader):
        """
        验证一个 epoch（🆕 支持相关性正则化）
        
        Args:
            valid_loader: 验证数据加载器
            
        Returns:
            平均验证损失
        """
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # 🆕 根据是否使用相关性正则化决定前向传播方式
                if self._use_corr_loss:
                    try:
                        output = self.model(batch_x, return_hidden=True)
                        if isinstance(output, tuple) and len(output) >= 2:
                            predictions = output[0]
                            hidden_features = output[-1]
                        else:
                            predictions = output if not isinstance(output, tuple) else output[0]
                            hidden_features = None
                    except TypeError as e:
                        if 'return_hidden' in str(e):
                            # 在 _train_epoch 中已经记录过警告，这里不重复
                            if not getattr(self, '_corr_loss_disabled_logged', False):
                                self.logger.warning(
                                    f"⚠️ 模型不支持 return_hidden 参数，相关性正则化已自动禁用。"
                                )
                            self._use_corr_loss = False
                            predictions = self.model(batch_x)
                            hidden_features = None
                        else:
                            raise
                    
                    if self._use_corr_loss and hidden_features is not None:
                        loss = self.criterion(predictions, batch_y, hidden_features)
                    else:
                        loss = self.criterion(predictions, batch_y)
                else:
                    predictions = self.model(batch_x)
                    loss = self.criterion(predictions, batch_y)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0


class FineTunableModel(Model):
    """
    可微调模型类
    
    参照 Qlib 的 ModelFT 设计
    """
    
    @abc.abstractmethod
    def finetune(self, train_data, valid_data=None, **kwargs):
        """
        微调模型
        
        Args:
            train_data: 训练数据
            valid_data: 验证数据（可选）
            **kwargs: 其他微调参数
        """
        raise NotImplementedError("finetune method must be implemented")


if __name__ == '__main__':
    print("=" * 80)
    print("Base Model Classes 测试")
    print("=" * 80)
    
    # 测试基类定义
    print("\n✅ BaseModel 定义完成")
    print("✅ Model 定义完成")
    print("✅ PyTorchModel 定义完成")
    print("✅ FineTunableModel 定义完成")
    
    print("\n模型基类系统已准备就绪！")
