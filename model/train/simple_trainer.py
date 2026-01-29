"""
SimpleTrainer - 简单训练器

接管常规训练，替换原 PyTorchModel.fit 的实现。
适用于单窗口、单数据集的标准训练场景。
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple

from .base_trainer import BaseTrainer, TrainerConfig


class SimpleTrainer(BaseTrainer):
    """
    简单训练器
    
    实现标准的训练循环，支持：
    - 自动 GPU 管理
    - 早停机制
    - 学习率调度
    - 检查点保存
    - 相关性正则化
    
    用于替代 PyTorchModel.fit 中的训练逻辑。
    
    Example:
        >>> model = LSTMNet(d_feat=20, hidden_size=64, num_layers=2, dropout=0.1)
        >>> config = TrainerConfig(n_epochs=100, lr=0.001)
        >>> trainer = SimpleTrainer(model, config)
        >>> result = trainer.train(train_loader, val_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainerConfig] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        初始化简单训练器
        
        Args:
            model: PyTorch 模型
            config: 训练配置（None 则使用默认配置）
            device: 计算设备
            **kwargs: 额外配置参数（会覆盖 config）
        """
        if config is None:
            config = TrainerConfig(**kwargs)
        elif kwargs:
            # 用 kwargs 覆盖 config 字段（dataclass 无 update 方法）
            from dataclasses import replace
            config = replace(config, **kwargs)
        
        super().__init__(model, config, device)
        
        # 相关性正则化标志
        self._use_corr_loss = config.lambda_corr > 0
        self._supports_return_hidden = self._check_return_hidden_support()
        
        if self._use_corr_loss:
            if self._supports_return_hidden:
                self.logger.info("  ✅ 相关性正则化已启用 (模型支持 return_hidden)")
            else:
                self.logger.warning("  ⚠️ 模型不支持 return_hidden，相关性正则化降级")
                self._use_corr_loss = False
    
    def _check_return_hidden_support(self) -> bool:
        """检查模型是否支持 return_hidden 参数"""
        import inspect
        
        if hasattr(self.model, 'forward'):
            sig = inspect.signature(self.model.forward)
            return 'return_hidden' in sig.parameters
        return False
    
    def _forward_with_hidden(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None):
        """
        带 hidden 输出的前向传播
        
        Args:
            x: 输入特征
            adj: 邻接矩阵（可选）
            
        Returns:
            (predictions, hidden_features) 或 (predictions, None)
        """
        if self._use_corr_loss and self._supports_return_hidden:
            try:
                # 尝试传递 adj
                try:
                    output = self.model(x, adj=adj, return_hidden=True)
                except TypeError:
                    output = self.model(x, return_hidden=True)
                
                if isinstance(output, tuple) and len(output) >= 2:
                    return output[0], output[-1]  # (pred, hidden)
                else:
                    return output, None
            except Exception as e:
                self.logger.warning(f"return_hidden 调用失败: {e}")
                return self.model(x), None
        else:
            # 标准前向传播
            try:
                return self.model(x, adj=adj) if adj is not None else self.model(x), None
            except TypeError:
                return self.model(x), None
    
    def train_batch(self, batch_data) -> float:
        """
        训练单个 batch
        
        Args:
            batch_data: DataLoader 返回的数据
            
        Returns:
            batch 损失值（空批次返回 0.0）
        """
        x, y, adj, idx = self._parse_batch_data(batch_data)
        
        # 🆕 跳过空批次（避免 GAT 层 N=0 reshape 异常）
        if x.size(0) == 0:
            return 0.0
        
        # 移动数据到设备
        x = x.to(self.device)
        y = y.to(self.device)
        if adj is not None:
            adj = adj.to(self.device)
        
        # 前向传播
        self.optimizer.zero_grad()
        
        pred, hidden = self._forward_with_hidden(x, adj)
        
        # 计算损失
        if self._use_corr_loss and hidden is not None:
            # 带相关性正则化的损失
            loss = self.criterion(pred, y, hidden)
        else:
            # 标准损失
            loss = self.criterion(pred, y)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate_epoch(self, val_loader) -> float:
        """
        验证一个 epoch
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                x, y, adj, idx = self._parse_batch_data(batch_data)
                
                # 🆕 跳过空批次（避免 GAT 层 N=0 reshape 异常）
                if x.size(0) == 0:
                    continue
                
                x = x.to(self.device)
                y = y.to(self.device)
                if adj is not None:
                    adj = adj.to(self.device)
                
                # 前向传播（验证时不需要 hidden）
                try:
                    pred = self.model(x, adj=adj) if adj is not None else self.model(x)
                except TypeError:
                    pred = self.model(x)
                
                # 如果模型返回元组，取第一个元素
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                # 🆕 多因子预测聚合：如果 pred 是 [batch, F]，取均值得到 [batch]
                if pred.dim() == 2 and pred.size(1) > 1:
                    pred = pred.mean(dim=1)
                
                # 计算损失（验证时不使用相关性正则化）
                if hasattr(self.criterion, 'base_loss'):
                    # 对于带相关性的损失函数，只使用基础损失
                    loss = self.criterion.base_loss(pred, y)
                else:
                    loss = self.criterion(pred, y)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def predict(self, test_loader, return_numpy: bool = True):
        """
        预测 - 优先委托模型的 predict 方法
        
        如果模型有自己的 predict()（如 PyTorchModel 子类），直接委托调用；
        否则回退到 Trainer 自己的实现（用于纯 nn.Module）。
        
        Args:
            test_loader: 测试数据加载器
            return_numpy: 是否返回 numpy 数组
            
        Returns:
            预测结果
        """
        # 🆕 检查模型是否有 predict 方法（PyTorchModel 子类有，纯 nn.Module 没有）
        if hasattr(self.model, 'predict') and callable(getattr(self.model, 'predict')):
            try:
                # 委托给模型（模型的 predict 已包含完整的批次解析、设备迁移、空处理逻辑）
                return self.model.predict(test_loader, return_numpy)
            except (TypeError, AttributeError):
                # 模型的 predict 签名不兼容，回退到 Trainer 实现
                pass
        
        # 回退：Trainer 自己的实现（用于纯 nn.Module）
        self.model.eval()
        predictions = []
        labels = []  # 🆕 同时收集标签用于返回
        
        with torch.no_grad():
            for batch_data in test_loader:
                x, y, adj, _ = self._parse_batch_data(batch_data)
                
                # 🆕 跳过空批次（避免 GAT 层 N=0 reshape 异常）
                if x.size(0) == 0:
                    continue
                
                x = x.to(self.device)
                if adj is not None:
                    adj = adj.to(self.device)
                
                try:
                    pred = self.model(x, adj=adj) if adj is not None else self.model(x)
                except TypeError:
                    pred = self.model(x)
                
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                predictions.append(pred.cpu())
                labels.append(y.cpu())
        
        if len(predictions) == 0:
            import numpy as np
            empty_result = np.array([]) if return_numpy else torch.tensor([])
            return empty_result, empty_result
        
        result = torch.cat(predictions, dim=0)
        labels_result = torch.cat(labels, dim=0)
        
        if return_numpy:
            return result.numpy(), labels_result.numpy()
        return result, labels_result


def create_simple_trainer(
    model: nn.Module,
    n_epochs: int = 100,
    lr: float = 0.001,
    early_stop: int = 20,
    device: Optional[str] = None,
    **kwargs
) -> SimpleTrainer:
    """
    创建 SimpleTrainer 的便捷函数
    
    Args:
        model: PyTorch 模型
        n_epochs: 训练轮数
        lr: 学习率
        early_stop: 早停耐心值
        device: 计算设备
        **kwargs: 其他 TrainerConfig 参数
        
    Returns:
        SimpleTrainer 实例
    """
    config = TrainerConfig(
        n_epochs=n_epochs,
        lr=lr,
        early_stop=early_stop,
        **kwargs
    )
    return SimpleTrainer(model, config, device)
