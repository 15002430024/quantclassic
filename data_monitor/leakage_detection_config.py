"""
数据泄漏检测配置模块
"""
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.base_config import BaseConfig


class LeakageTestMode(Enum):
    """
    数据泄漏测试模式枚举
    
    定义不同的数据泄漏检测策略和测试级别。
    
    🎯 测试模式分类：
    
    1️⃣ STATIC_ONLY: 纯静态代码分析
        - 不执行数据加载和模型运行
        - 通过代码审查检测明显的时间泄漏
        - 速度快，适合快速检查
        
    2️⃣ DYNAMIC_ONLY: 纯动态运行时监控
        - 执行模型训练流程
        - 监控数据访问模式
        - 检测运行时的时间边界违规
        
    3️⃣ FULL: 完整测试（静态 + 动态）
        - 结合静态和动态检测
        - 提供最全面的数据泄漏检测
        - 推荐用于模型验证
    
    📊 使用建议：
        - 开发阶段: STATIC_ONLY (快速反馈)
        - 测试阶段: DYNAMIC_ONLY (运行时验证)
        - 生产验证: FULL (全面检测)
    """
    STATIC_ONLY = "static_only"
    DYNAMIC_ONLY = "dynamic_only"
    FULL = "full"


@dataclass
class LeakageDetectionConfig(BaseConfig):
    """
    数据泄漏检测配置类
    
    用于配置数据泄漏检测的各项参数和行为。
    
    Args:
        test_mode: 测试模式，默认 FULL (完整测试)
        verbose: 是否输出详细信息，默认 True
        time_column: 时间列名称，默认 'year_month'
        stock_column: 股票代码列名称，默认 'ts_code'
        return_column: 收益率列名称，默认 'rm_rf'
        label_column: 标签列名称，默认 'target'
        
        # 静态测试配置
        check_feature_window: 是否检查特征窗口时间泄漏，默认 True
        check_factor_input: 是否检查因子输入同期泄漏，默认 True
        check_calFactor: 是否检查calFactor历史性，默认 True
        check_source_code: 是否进行源代码分析，默认 True
        
        # 动态测试配置
        monitor_data_access: 是否监控数据访问，默认 True
        monitor_cache_growth: 是否监控缓存增长，默认 True
        enforce_time_boundary: 是否强制时间边界，默认 True
        max_cache_growth: 最大允许的缓存增长量，默认 1000
        
        # 测试数据配置
        test_months: 测试月份列表，默认使用训练期前5个月
        test_stocks_limit: 每个月测试的最大股票数量，默认 10
        epsilon: 浮点数比较精度，默认 1e-6
        
        # 报告配置
        generate_report: 是否生成测试报告，默认 True
        report_path: 报告保存路径，默认 './leakage_detection_report.txt'
        show_summary: 是否显示摘要，默认 True
    
    📊 配置示例：
        
        # 基础配置（快速检查）
        config = LeakageDetectionConfig(
            test_mode=LeakageTestMode.STATIC_ONLY,
            verbose=True
        )
        
        # 完整配置（详细测试）
        config = LeakageDetectionConfig(
            test_mode=LeakageTestMode.FULL,
            verbose=True,
            time_column='year_month',
            stock_column='ts_code',
            check_feature_window=True,
            check_factor_input=True,
            monitor_data_access=True,
            enforce_time_boundary=True,
            generate_report=True,
            report_path='./my_leakage_report.txt'
        )
        
        # 自定义测试月份
        config = LeakageDetectionConfig(
            test_mode=LeakageTestMode.FULL,
            test_months=[200801, 200802, 200803],
            test_stocks_limit=20
        )
    
    💡 最佳实践：
        - verbose=True: 便于理解测试过程
        - test_mode=FULL: 提供最全面的检测
        - generate_report=True: 保存测试结果供后续分析
        - test_stocks_limit适当设置: 平衡测试速度和覆盖率
    """
    
    # ========== 基础配置 ==========
    test_mode: LeakageTestMode = LeakageTestMode.FULL
    verbose: bool = True
    
    # ========== 数据列配置 ==========
    time_column: str = 'year_month'
    stock_column: str = 'ts_code'
    return_column: str = 'rm_rf'
    label_column: str = 'target'
    
    # ========== 静态测试开关 ==========
    check_feature_window: bool = True
    check_factor_input: bool = True
    check_calFactor: bool = True
    check_source_code: bool = True
    
    # ========== 动态测试开关 ==========
    monitor_data_access: bool = True
    monitor_cache_growth: bool = True
    enforce_time_boundary: bool = True
    max_cache_growth: int = 1000
    
    # ========== 测试数据配置 ==========
    test_months: Optional[List[int]] = None  # 如果为None，自动选择
    test_stocks_limit: int = 10
    epsilon: float = 1e-6
    
    # ========== 报告配置 ==========
    generate_report: bool = True
    report_path: str = './leakage_detection_report.txt'
    show_summary: bool = True
    
    def __post_init__(self):
        """初始化后处理"""
        # 转换 test_mode 为枚举
        if isinstance(self.test_mode, str):
            self.test_mode = LeakageTestMode(self.test_mode)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LeakageDetectionConfig':
        """从字典创建配置"""
        # 转换枚举字符串
        if 'test_mode' in config_dict and isinstance(config_dict['test_mode'], str):
            config_dict['test_mode'] = LeakageTestMode(config_dict['test_mode'])
        return cls(**config_dict)
    
    def enable_all_checks(self):
        """启用所有检查"""
        self.check_feature_window = True
        self.check_factor_input = True
        self.check_calFactor = True
        self.check_source_code = True
        self.monitor_data_access = True
        self.monitor_cache_growth = True
        self.enforce_time_boundary = True
    
    def disable_all_checks(self):
        """禁用所有检查"""
        self.check_feature_window = False
        self.check_factor_input = False
        self.check_calFactor = False
        self.check_source_code = False
        self.monitor_data_access = False
        self.monitor_cache_growth = False
        self.enforce_time_boundary = False
    
    def enable_static_checks(self):
        """仅启用静态检查"""
        self.check_feature_window = True
        self.check_factor_input = True
        self.check_calFactor = True
        self.check_source_code = True
        self.monitor_data_access = False
        self.monitor_cache_growth = False
        self.enforce_time_boundary = False
    
    def enable_dynamic_checks(self):
        """仅启用动态检查"""
        self.check_feature_window = False
        self.check_factor_input = False
        self.check_calFactor = False
        self.check_source_code = False
        self.monitor_data_access = True
        self.monitor_cache_growth = True
        self.enforce_time_boundary = True


class LeakageDetectionTemplates:
    """
    数据泄漏检测配置模板
    
    提供预定义的常用配置模板，方便快速使用。
    """
    
    @staticmethod
    def quick_check() -> LeakageDetectionConfig:
        """
        快速检查模板
        
        适用场景: 开发阶段快速验证
        特点: 静态检查，速度快
        """
        config = LeakageDetectionConfig(
            test_mode=LeakageTestMode.STATIC_ONLY,
            verbose=True,
            generate_report=False,
            show_summary=True
        )
        config.enable_static_checks()
        return config
    
    @staticmethod
    def full_validation() -> LeakageDetectionConfig:
        """
        完整验证模板
        
        适用场景: 模型上线前的完整验证
        特点: 静态+动态，全面检测
        """
        config = LeakageDetectionConfig(
            test_mode=LeakageTestMode.FULL,
            verbose=True,
            generate_report=True,
            show_summary=True,
            test_stocks_limit=20
        )
        config.enable_all_checks()
        return config
    
    @staticmethod
    def runtime_monitor() -> LeakageDetectionConfig:
        """
        运行时监控模板
        
        适用场景: 训练过程中的实时监控
        特点: 动态监控，轻量级
        """
        config = LeakageDetectionConfig(
            test_mode=LeakageTestMode.DYNAMIC_ONLY,
            verbose=False,  # 减少输出
            generate_report=True,
            show_summary=False,
            monitor_data_access=True,
            enforce_time_boundary=True
        )
        config.enable_dynamic_checks()
        return config
    
    @staticmethod
    def custom(
        test_mode: Union[str, LeakageTestMode] = LeakageTestMode.FULL,
        verbose: bool = True,
        **kwargs
    ) -> LeakageDetectionConfig:
        """
        自定义配置模板
        
        Args:
            test_mode: 测试模式
            verbose: 是否详细输出
            **kwargs: 其他配置参数
        
        Returns:
            自定义配置对象
        """
        return LeakageDetectionConfig(
            test_mode=test_mode,
            verbose=verbose,
            **kwargs
        )
