"""
Loss Functions - 损失函数模块

提供多种可选的损失函数，支持相关性正则化以抑制特征冗余。

Loss = BaseLoss(pred, target) + lambda * CorrelationPenalty(hidden_features)

Available Losses:
- MSEWithCorrelationLoss: MSE + 特征去相关正则
- MAEWithCorrelationLoss: MAE + 特征去相关正则
- HuberWithCorrelationLoss: Huber + 特征去相关正则
- ICLoss: 排序 IC Loss (Information Coefficient)
- ICWithCorrelationLoss: IC Loss + 特征去相关正则

⚠️ 使用要求:
------------
要使用带相关性正则化的损失函数 (*WithCorrelationLoss)，模型必须：

1. 在 forward() 中支持 return_hidden=True 参数
2. 返回 (predictions, hidden_features) 元组
3. hidden_features 形状为 [batch_size, hidden_dim]

示例:
    class MyNet(nn.Module):
        def forward(self, x, return_hidden=False):
            features = self.encoder(x)
            pred = self.head(features)
            if return_hidden:
                return pred, features
            return pred
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ==================== 相关性正则化模块 ====================

class CorrelationRegularizer(nn.Module):
    """
    相关性正则化模块
    
    计算隐藏层特征之间的相关性惩罚，促使模型学习更加独立的特征表达。
    
    原理：
    1. 计算特征矩阵的相关性矩阵 (Correlation Matrix)
    2. 理想情况下，相关性矩阵应该接近单位矩阵（对角线为1，其余为0）
    3. 使用 Frobenius 范数计算与单位矩阵的差异作为惩罚项
    
    Args:
        lambda_corr: 正则化权重，控制去相关的强度
        eps: 数值稳定性的小常数
    """
    
    def __init__(self, lambda_corr: float = 0.01, eps: float = 1e-8):
        super().__init__()
        self.lambda_corr = lambda_corr
        self.eps = eps
    
    def forward(self, hidden_features: torch.Tensor) -> torch.Tensor:
        """
        计算相关性惩罚
        
        Args:
            hidden_features: [batch_size, hidden_dim] 隐藏层特征
            
        Returns:
            correlation_penalty: 标量，相关性惩罚值
        """
        if hidden_features is None or self.lambda_corr <= 0:
            return torch.tensor(0.0, device=hidden_features.device if hidden_features is not None else 'cpu')
        
        batch_size, hidden_dim = hidden_features.shape
        
        # 样本数太少时，相关性矩阵估计不可靠
        if batch_size < 2:
            return torch.tensor(0.0, device=hidden_features.device)
        
        # 1. 标准化特征 (Z-Score normalization along batch dimension)
        mean = hidden_features.mean(dim=0, keepdim=True)
        std = hidden_features.std(dim=0, keepdim=True) + self.eps
        normalized = (hidden_features - mean) / std
        
        # 2. 计算相关性矩阵: (X^T @ X) / (N - 1)
        # 结果是 [hidden_dim, hidden_dim] 的对称矩阵
        corr_matrix = (normalized.t() @ normalized) / (batch_size - 1)
        
        # 3. 目标：相关性矩阵应接近单位矩阵
        # 对角线为 1（自相关），非对角线为 0（无冗余）
        identity = torch.eye(hidden_dim, device=corr_matrix.device)
        
        # 4. 使用 Frobenius 范数计算差异
        # 可选：只惩罚非对角线元素（更精确）
        off_diagonal_mask = 1 - identity
        off_diagonal_corr = corr_matrix * off_diagonal_mask
        
        # Frobenius 范数的平方
        corr_penalty = torch.sum(off_diagonal_corr ** 2)
        
        # 归一化（除以非对角线元素数量）
        n_off_diagonal = hidden_dim * (hidden_dim - 1)
        if n_off_diagonal > 0:
            corr_penalty = corr_penalty / n_off_diagonal
        
        return self.lambda_corr * corr_penalty


# ==================== 基础损失函数 + 相关性正则化 ====================

class MSEWithCorrelationLoss(nn.Module):
    """
    MSE Loss + 特征相关性正则化
    
    Loss = MSE(pred, target) + lambda * CorrelationPenalty(hidden_features)
    
    Args:
        lambda_corr: 相关性正则化权重，建议范围 [0.001, 0.1]
    
    Example:
        >>> criterion = MSEWithCorrelationLoss(lambda_corr=0.01)
        >>> pred, hidden = model(x)  # 模型需返回 (预测, 隐藏特征)
        >>> loss = criterion(pred, target, hidden)
    """
    
    def __init__(self, lambda_corr: float = 0.01):
        super().__init__()
        self.mse = nn.MSELoss()
        self.corr_reg = CorrelationRegularizer(lambda_corr=lambda_corr)
        self.lambda_corr = lambda_corr
    
    def forward(
        self, 
        preds: torch.Tensor, 
        targets: torch.Tensor, 
        hidden_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算带正则项的损失
        
        Args:
            preds: [batch_size] 或 [batch_size, 1] 预测值
            targets: [batch_size] 或 [batch_size, 1] 目标值
            hidden_features: [batch_size, hidden_dim] 隐藏层特征（可选）
            
        Returns:
            loss: 标量，总损失
        """
        # 基础 MSE 损失
        base_loss = self.mse(preds.flatten(), targets.flatten())
        
        # 相关性正则化
        corr_penalty = self.corr_reg(hidden_features)
        
        return base_loss + corr_penalty
    
    def extra_repr(self) -> str:
        return f'lambda_corr={self.lambda_corr}'


class MAEWithCorrelationLoss(nn.Module):
    """
    MAE Loss + 特征相关性正则化
    
    Loss = MAE(pred, target) + lambda * CorrelationPenalty(hidden_features)
    """
    
    def __init__(self, lambda_corr: float = 0.01):
        super().__init__()
        self.mae = nn.L1Loss()
        self.corr_reg = CorrelationRegularizer(lambda_corr=lambda_corr)
        self.lambda_corr = lambda_corr
    
    def forward(
        self, 
        preds: torch.Tensor, 
        targets: torch.Tensor, 
        hidden_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        base_loss = self.mae(preds.flatten(), targets.flatten())
        corr_penalty = self.corr_reg(hidden_features)
        return base_loss + corr_penalty


class HuberWithCorrelationLoss(nn.Module):
    """
    Huber Loss + 特征相关性正则化
    
    Loss = Huber(pred, target) + lambda * CorrelationPenalty(hidden_features)
    
    Huber Loss 结合了 MSE 和 MAE 的优点：
    - 对于小误差，使用 MSE（平滑且可微）
    - 对于大误差，使用 MAE（对异常值鲁棒）
    """
    
    def __init__(self, lambda_corr: float = 0.01, delta: float = 1.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.corr_reg = CorrelationRegularizer(lambda_corr=lambda_corr)
        self.lambda_corr = lambda_corr
    
    def forward(
        self, 
        preds: torch.Tensor, 
        targets: torch.Tensor, 
        hidden_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        base_loss = self.huber(preds.flatten(), targets.flatten())
        corr_penalty = self.corr_reg(hidden_features)
        return base_loss + corr_penalty


# ==================== IC Loss (排序损失) ====================

class ICLoss(nn.Module):
    """
    IC Loss - Information Coefficient Loss
    
    基于排序相关性的损失函数，优化预测值与真实值之间的 Rank IC。
    
    IC = Pearson(rank(pred), rank(target))
    Loss = 1 - IC  (最大化 IC 等价于最小化 1-IC)
    
    优点：
    - 关注相对排序，而非绝对值预测
    - 更符合量化策略的实际需求（选股排名）
    
    Note:
        需要同一截面（同一天）的所有股票在一个 batch 中
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self, 
        preds: torch.Tensor, 
        targets: torch.Tensor,
        hidden_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算 IC Loss
        
        Args:
            preds: [batch_size] 预测值
            targets: [batch_size] 目标值
            hidden_features: 未使用，保持接口一致
            
        Returns:
            loss: 1 - IC
        """
        preds = preds.flatten()
        targets = targets.flatten()
        
        # 处理 NaN
        valid_mask = ~(torch.isnan(preds) | torch.isnan(targets))
        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
        
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        
        # 中心化
        pred_mean = preds.mean()
        target_mean = targets.mean()
        pred_centered = preds - pred_mean
        target_centered = targets - target_mean
        
        # Pearson 相关系数
        cov = (pred_centered * target_centered).sum()
        pred_std = torch.sqrt((pred_centered ** 2).sum() + 1e-8)
        target_std = torch.sqrt((target_centered ** 2).sum() + 1e-8)
        
        ic = cov / (pred_std * target_std)
        
        # IC 范围是 [-1, 1]，我们希望最大化 IC
        # 因此 Loss = 1 - IC (范围 [0, 2])
        # 或者使用 -IC (但这样 loss 可能为负)
        return 1.0 - ic


class ICWithCorrelationLoss(nn.Module):
    """
    IC Loss + 特征相关性正则化
    
    Loss = (1 - IC) + lambda * CorrelationPenalty(hidden_features)
    
    结合排序优化和特征去相关，适用于量化选股模型。
    """
    
    def __init__(self, lambda_corr: float = 0.01):
        super().__init__()
        self.ic_loss = ICLoss()
        self.corr_reg = CorrelationRegularizer(lambda_corr=lambda_corr)
        self.lambda_corr = lambda_corr
    
    def forward(
        self, 
        preds: torch.Tensor, 
        targets: torch.Tensor, 
        hidden_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        base_loss = self.ic_loss(preds, targets)
        corr_penalty = self.corr_reg(hidden_features)
        return base_loss + corr_penalty


# ==================== 组合损失函数 ====================

class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    支持同时使用多种损失的加权组合：
    Loss = w1 * MSE + w2 * IC + lambda * CorrelationPenalty
    
    Args:
        mse_weight: MSE 损失权重
        ic_weight: IC 损失权重
        lambda_corr: 相关性正则化权重
    """
    
    def __init__(
        self, 
        mse_weight: float = 1.0,
        ic_weight: float = 0.0,
        lambda_corr: float = 0.01
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.ic_weight = ic_weight
        
        self.mse = nn.MSELoss()
        self.ic_loss = ICLoss()
        self.corr_reg = CorrelationRegularizer(lambda_corr=lambda_corr)
    
    def forward(
        self, 
        preds: torch.Tensor, 
        targets: torch.Tensor, 
        hidden_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=preds.device)
        
        if self.mse_weight > 0:
            total_loss = total_loss + self.mse_weight * self.mse(preds.flatten(), targets.flatten())
        
        if self.ic_weight > 0:
            total_loss = total_loss + self.ic_weight * self.ic_loss(preds, targets)
        
        # 相关性正则化
        corr_penalty = self.corr_reg(hidden_features)
        total_loss = total_loss + corr_penalty
        
        return total_loss


# ==================== 统一损失函数 (重构新增) ====================

class UnifiedLoss(nn.Module):
    """
    🆕 统一损失函数 - 重构后的标准损失接口
    
    支持多种基础损失 + 可选的相关性正则化 + 可选的多因子正交化。
    
    Loss = BaseLoss(pred, target) 
         + lambda_corr * CorrelationPenalty(hidden_features)
         + lambda_ortho * OrthoPenalty(factors)
    
    Args:
        base_loss: 基础损失类型 ('mse' | 'mae' | 'huber' | 'ic' | 'rankic')
        lambda_corr: 相关性正则化权重，0 表示不使用
        lambda_ortho: 多因子正交化权重，0 表示不使用
        reduction: 损失 reduction 模式
        **kwargs: 传递给基础损失的额外参数（如 Huber 的 delta）
        
    Example:
        >>> criterion = UnifiedLoss(base_loss='mse', lambda_corr=0.01)
        >>> loss = criterion(pred, target, hidden_features=hidden)
        
        >>> # 多因子模式
        >>> criterion = UnifiedLoss(base_loss='mse', lambda_ortho=0.01)
        >>> loss = criterion(factor_preds, target, factors=factor_preds)
    """
    
    def __init__(
        self,
        base_loss: str = 'mse',
        lambda_corr: float = 0.0,
        lambda_ortho: float = 0.0,
        reduction: str = 'mean',
        **kwargs
    ):
        super().__init__()
        
        self.base_loss_name = base_loss.lower()
        self.lambda_corr = lambda_corr
        self.lambda_ortho = lambda_ortho
        self.reduction = reduction
        
        # 创建基础损失函数
        if self.base_loss_name == 'mse':
            self.base_loss = nn.MSELoss(reduction=reduction)
        elif self.base_loss_name == 'mae':
            self.base_loss = nn.L1Loss(reduction=reduction)
        elif self.base_loss_name == 'huber':
            delta = kwargs.get('delta', 1.0)
            self.base_loss = nn.HuberLoss(delta=delta, reduction=reduction)
        elif self.base_loss_name in ['ic', 'rankic']:
            self.base_loss = ICLoss()
        else:
            raise ValueError(f"不支持的基础损失类型: {base_loss}")
        
        # 相关性正则化器
        if lambda_corr > 0:
            self.corr_reg = CorrelationRegularizer(lambda_corr=lambda_corr)
        else:
            self.corr_reg = None
        
        # 多因子正交化正则化器
        if lambda_ortho > 0:
            self.ortho_reg = CorrelationRegularizer(lambda_corr=lambda_ortho)
        else:
            self.ortho_reg = None
    
    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        hidden_features: Optional[torch.Tensor] = None,
        factors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算统一损失
        
        Args:
            preds: [batch_size] 或 [batch_size, num_factors] 预测值
            targets: [batch_size] 或 [batch_size, num_factors] 目标值
            hidden_features: [batch_size, hidden_dim] 隐藏特征（用于相关性正则化）
            factors: [batch_size, num_factors] 多因子输出（用于正交化）
            
        Returns:
            loss: 标量，总损失
        """
        # 基础损失
        loss = self.base_loss(preds.flatten(), targets.flatten())
        
        # 相关性正则化
        if self.corr_reg is not None and hidden_features is not None:
            loss = loss + self.corr_reg(hidden_features)
        
        # 多因子正交化
        if self.ortho_reg is not None and factors is not None:
            # 对因子矩阵的转置做相关性惩罚，促使因子间正交
            # factors: [batch, num_factors] -> 转置后计算因子间相关性
            if factors.dim() == 2 and factors.size(1) > 1:
                loss = loss + self.ortho_reg(factors)
        
        return loss
    
    def extra_repr(self) -> str:
        return (
            f"base_loss={self.base_loss_name}, "
            f"lambda_corr={self.lambda_corr}, "
            f"lambda_ortho={self.lambda_ortho}"
        )


# ==================== 损失函数工厂 ====================

def get_loss_fn(
    loss_type: str = 'mse',
    lambda_corr: float = 0.0,
    lambda_ortho: float = 0.0,
    **kwargs
) -> nn.Module:
    """
    损失函数工厂
    
    Args:
        loss_type: 损失函数类型
            - 'mse': MSE Loss
            - 'mae': MAE Loss
            - 'huber': Huber Loss
            - 'ic': IC Loss
            - 'mse_corr': MSE + 相关性正则
            - 'mae_corr': MAE + 相关性正则
            - 'huber_corr': Huber + 相关性正则
            - 'ic_corr': IC + 相关性正则
            - 'combined': 组合损失
            - 'unified': 🆕 统一损失（推荐）
        lambda_corr: 相关性正则化权重（仅对 *_corr 和 unified 类型有效）
        lambda_ortho: 🆕 多因子正交化权重（仅对 unified 类型有效）
        **kwargs: 额外参数
        
    Returns:
        损失函数模块
        
    Example:
        >>> criterion = get_loss_fn('mse_corr', lambda_corr=0.01)
        >>> loss = criterion(pred, target, hidden_features)
        
        >>> # 使用统一损失（推荐）
        >>> criterion = get_loss_fn('unified', base_loss='mse', lambda_corr=0.01)
    """
    loss_type = loss_type.lower()
    
    # 🆕 统一损失（推荐）
    if loss_type == 'unified':
        base_loss = kwargs.pop('base_loss', 'mse')
        return UnifiedLoss(
            base_loss=base_loss,
            lambda_corr=lambda_corr,
            lambda_ortho=lambda_ortho,
            **kwargs
        )
    
    # 标准损失（无正则化）
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'mae':
        return nn.L1Loss()
    elif loss_type == 'huber':
        return nn.HuberLoss(**kwargs)
    elif loss_type == 'ic':
        return ICLoss()
    
    # 带相关性正则化的损失
    elif loss_type == 'mse_corr':
        return MSEWithCorrelationLoss(lambda_corr=lambda_corr)
    elif loss_type == 'mae_corr':
        return MAEWithCorrelationLoss(lambda_corr=lambda_corr)
    elif loss_type == 'huber_corr':
        return HuberWithCorrelationLoss(lambda_corr=lambda_corr, **kwargs)
    elif loss_type == 'ic_corr':
        return ICWithCorrelationLoss(lambda_corr=lambda_corr)
    
    # 组合损失
    elif loss_type == 'combined':
        return CombinedLoss(lambda_corr=lambda_corr, **kwargs)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                        f"Supported: mse, mae, huber, ic, mse_corr, mae_corr, huber_corr, ic_corr, combined, unified")


if __name__ == '__main__':
    print("=" * 80)
    print("Loss Functions 测试")
    print("=" * 80)
    
    # 模拟数据
    batch_size = 32
    hidden_dim = 64
    
    preds = torch.randn(batch_size)
    targets = torch.randn(batch_size)
    hidden = torch.randn(batch_size, hidden_dim)
    
    # 测试各种损失函数
    print("\n1. 测试 MSEWithCorrelationLoss:")
    criterion = MSEWithCorrelationLoss(lambda_corr=0.01)
    loss = criterion(preds, targets, hidden)
    print(f"   Loss: {loss.item():.6f}")
    
    print("\n2. 测试 ICLoss:")
    criterion = ICLoss()
    loss = criterion(preds, targets)
    print(f"   Loss: {loss.item():.6f}")
    
    print("\n3. 测试 ICWithCorrelationLoss:")
    criterion = ICWithCorrelationLoss(lambda_corr=0.01)
    loss = criterion(preds, targets, hidden)
    print(f"   Loss: {loss.item():.6f}")
    
    print("\n4. 测试 CombinedLoss:")
    criterion = CombinedLoss(mse_weight=0.5, ic_weight=0.5, lambda_corr=0.01)
    loss = criterion(preds, targets, hidden)
    print(f"   Loss: {loss.item():.6f}")
    
    print("\n5. 测试损失函数工厂:")
    for loss_type in ['mse', 'mse_corr', 'ic', 'ic_corr']:
        criterion = get_loss_fn(loss_type, lambda_corr=0.01)
        print(f"   {loss_type}: {criterion.__class__.__name__}")
    
    print("\n" + "=" * 80)
    print("✅ Loss Functions 测试完成")
    print("=" * 80)
