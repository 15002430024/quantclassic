"""
QuantClassic Workflow - 实验管理系统

参照 Qlib 的设计，提供实验追踪、参数记录、结果管理等功能

核心组件:
- Experiment: 实验管理
- Recorder: 记录器，记录参数、指标、对象
- ExpManager: 实验管理器
- R: 全局接口，简化使用

Example:
    >>> from workflow import R
    >>> 
    >>> with R.start(experiment_name='lstm_test'):
    ...     model.fit(train_loader, valid_loader)
    ...     R.log_params(lr=0.001, batch_size=256)
    ...     R.log_metrics(train_loss=0.05, ic=0.08)
    ...     R.save_objects(model=model)
"""

from .experiment import Experiment
from .recorder import Recorder
from .exp_manager import ExpManager
from .qc_recorder import QCRecorder, R as _R

# 创建全局实例
R = _R

__all__ = [
    'Experiment',
    'Recorder',
    'ExpManager',
    'QCRecorder',
    'R',
]

__version__ = '1.0.0'
