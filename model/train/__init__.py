"""
QuantClassic Model Training Module - 训练引擎模块

统一的训练架构，将训练策略从数据管理中解耦。

核心组件:
- BaseTrainer: 训练基类，定义通用训练循环、早停、检查点逻辑
- SimpleTrainer: 常规单窗口训练器
- RollingWindowTrainer: 滚动窗口训练器，支持权重继承
- RollingDailyTrainer: 日级滚动训练器，处理高频模型切换

配置驱动:
- TrainerConfig: 基础训练配置
- RollingTrainerConfig: 滚动训练配置

Usage:
    from quantclassic.model.train import SimpleTrainer, TrainerConfig
    
    config = TrainerConfig(n_epochs=100, lr=0.001)
    trainer = SimpleTrainer(model, config, device='cuda')
    trainer.train(train_loader, val_loader)
"""

from .base_trainer import (
    BaseTrainer,
    TrainerArtifacts,
    TrainerConfig,
    TrainerCallback,
    EarlyStoppingCallback,
    CheckpointCallback,
)
from .simple_trainer import SimpleTrainer
from .rolling_window_trainer import RollingWindowTrainer, RollingTrainerConfig
from .rolling_daily_trainer import RollingDailyTrainer, DailyRollingConfig


__all__ = [
    # 基类与配置
    'BaseTrainer',
    'TrainerArtifacts',
    'TrainerConfig',
    'TrainerCallback',
    'EarlyStoppingCallback',
    'CheckpointCallback',
    
    # 训练器
    'SimpleTrainer',
    'RollingWindowTrainer',
    'RollingDailyTrainer',
    
    # 配置
    'RollingTrainerConfig',
    'DailyRollingConfig',  # 🆕 导出日级滚动配置
]
