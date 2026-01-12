"""
rolling_daily_trainer.py - 兼容 shim (已废弃)

⚠️ 此文件已废弃！请使用 model.train.RollingDailyTrainer

此文件仅保留向后兼容性。所有类和函数均从新模块导入。

迁移指南:
    # 旧方式 (已废弃):
    from quantclassic.model.rolling_daily_trainer import RollingDailyTrainer, RollingTrainerConfig
    
    # 新方式 (推荐):
    from quantclassic.model.train import RollingDailyTrainer, RollingTrainerConfig

变更历史:
    - 2026-01-08: 原实现已迁移至 model/train/rolling_daily_trainer.py
    - 2026-01-08: 此文件改为 shim，发出废弃警告
"""

import warnings

# 发出废弃警告
warnings.warn(
    "model.rolling_daily_trainer 已废弃！\n"
    "请改用: from quantclassic.model.train import RollingDailyTrainer, RollingTrainerConfig",
    DeprecationWarning,
    stacklevel=2
)

# 从新位置导入所有类，保持向后兼容
from .train.rolling_daily_trainer import (
    RollingDailyTrainer,
    DailyRollingConfig as RollingTrainerConfig,  # 兼容旧名称
    create_rolling_daily_trainer as create_rolling_trainer,
)

# 兼容旧导出
__all__ = [
    'RollingDailyTrainer',
    'RollingTrainerConfig',
    'create_rolling_trainer',
]
