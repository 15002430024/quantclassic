"""
quantclassic.data_monitor - 数据泄漏检测模块

提供静态和动态的数据泄漏检测工具，便于在模型训练/推理前后进行自动化检查。

主要类:
- LeakageDetector: 主检测器，集成静态和动态检测
- LeakageDetectionConfig: 配置类
- StaticLeakageDetector: 静态检测器
- DynamicLeakageDetector: 动态检测器
- DataAccessMonitor: 数据访问监控器

快速使用:
    from quantclassic.data_monitor import LeakageDetector
    
    # 快速检查
    detector = LeakageDetector.quick_check()
    results = detector.detect(model, data)
    
    # 完整验证
    detector = LeakageDetector.full_validation()
    results = detector.detect(model, data)
"""

from .leakage_detection_config import (
    LeakageDetectionConfig,
    LeakageTestMode,
    LeakageDetectionTemplates,
)
from .static_leakage_detector import StaticLeakageDetector
from .dynamic_leakage_detector import DataAccessMonitor, DynamicLeakageDetector
from .leakage_detector import LeakageDetector

__version__ = "0.1.0"

__all__ = [
    # 主接口
    'LeakageDetector',
    
    # 配置
    'LeakageDetectionConfig',
    'LeakageTestMode',
    'LeakageDetectionTemplates',
    
    # 检测器
    'StaticLeakageDetector',
    'DynamicLeakageDetector',
    
    # 工具
    'DataAccessMonitor',
]
