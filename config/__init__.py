"""
QuantClassic Config - YAML配置系统

提供与Qlib兼容的配置驱动工作流

Example:
    >>> from config import run_task
    >>> run_task('configs/lstm_experiment.yaml')
"""

from .loader import ConfigLoader
from .runner import TaskRunner
from .utils import init_instance_by_config

__all__ = [
    'ConfigLoader',
    'TaskRunner',
    'init_instance_by_config',
]

__version__ = '1.0.0'
