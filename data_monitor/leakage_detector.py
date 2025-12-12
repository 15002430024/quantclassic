"""
数据泄漏检测器 - 主类
"""
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
import pandas as pd
import json

from .leakage_detection_config import LeakageDetectionConfig, LeakageTestMode
from .static_leakage_detector import StaticLeakageDetector
from .dynamic_leakage_detector import DynamicLeakageDetector

logger = logging.getLogger(__name__)


class LeakageDetector:
    """
    数据泄漏检测器 - 统一接口
    
    集成静态和动态检测功能，提供一站式数据泄漏检测服务。
    
    主要功能:
    1. 自动选择检测模式（静态/动态/全面）
    2. 统一的测试接口和结果管理
    3. 详细的测试报告生成
    4. 灵活的配置管理
    
    适用场景:
    - 模型训练前的数据验证
    - CI/CD流程中的自动化检测
    - 模型审计和合规性检查
    
    Args:
        config: 数据泄漏检测配置对象，可以是：
            - LeakageDetectionConfig 对象
            - 配置文件路径（str）
            - 配置字典（dict）
            - None（使用默认配置）
    
    Examples:
        # 快速检查（静态）
        detector = LeakageDetector.quick_check()
        results = detector.detect(model, data)
        
        # 完整验证（静态+动态）
        config = LeakageDetectionConfig(test_mode=LeakageTestMode.FULL)
        detector = LeakageDetector(config)
        results = detector.detect(model, data)
        
        # 自定义配置
        config = LeakageDetectionConfig(
            test_mode=LeakageTestMode.FULL,
            verbose=True,
            check_feature_window=True,
            monitor_data_access=True,
            generate_report=True
        )
        detector = LeakageDetector(config)
        results = detector.detect(model, data)
    """
    
    def __init__(self, config: Union[LeakageDetectionConfig, str, dict, None] = None):
        """
        初始化数据泄漏检测器
        
        Args:
            config: 配置对象、配置文件路径、配置字典或None
        """
        # 加载配置
        self.config = self._load_config(config)
        
        # 初始化检测器
        self.static_detector = None
        self.dynamic_detector = None
        
        if self.config.test_mode in [LeakageTestMode.STATIC_ONLY, LeakageTestMode.FULL]:
            self.static_detector = StaticLeakageDetector(self.config)
        
        if self.config.test_mode in [LeakageTestMode.DYNAMIC_ONLY, LeakageTestMode.FULL]:
            self.dynamic_detector = DynamicLeakageDetector(self.config)
        
        # 测试结果
        self.test_results = {}
        self.all_test_details = {}
        
        logger.info(f"初始化数据泄漏检测器，模式: {self.config.test_mode.value}")
    
    def _load_config(self, config: Union[LeakageDetectionConfig, str, dict, None]) -> LeakageDetectionConfig:
        """
        加载配置
        
        Args:
            config: 配置对象、文件路径、字典或None
        
        Returns:
            配置对象
        """
        if config is None:
            return LeakageDetectionConfig()
        elif isinstance(config, LeakageDetectionConfig):
            return config
        elif isinstance(config, str):
            return LeakageDetectionConfig.from_yaml(config)
        elif isinstance(config, dict):
            return LeakageDetectionConfig.from_dict(config)
        else:
            raise TypeError(f"不支持的配置类型: {type(config)}")
    
    def detect(
        self,
        model: Any,
        data: pd.DataFrame,
        train_months: Optional[List[int]] = None,
        test_start_month: Optional[int] = None,
        inference_months: Optional[List[int]] = None
    ) -> Dict[str, bool]:
        """
        执行数据泄漏检测
        
        这是主要的检测接口，根据配置自动执行静态和/或动态检测。
        
        Args:
            model: 待检测的模型对象
                要求：
                - 必须有 _get_item(month) 方法
                - 可选：calFactor(month) 方法
                - 可选：_data_cache 属性
            
            data: 数据框
                要求：
                - 必须包含 config.time_column (默认 'year_month')
                - 必须包含 config.stock_column (默认 'ts_code')
                - 推荐包含 config.return_column (默认 'rm_rf')
                - 推荐包含 config.label_column (默认 'target')
            
            train_months: 训练月份列表（用于动态测试）
                如果为None，自动从数据中选择前几个月
            
            test_start_month: 测试期开始月份（用于设置时间边界）
                如果为None，自动选择数据中间月份
            
            inference_months: 推理月份列表（用于推理测试）
                如果为None，跳过推理测试
        
        Returns:
            测试结果字典，格式：
            {
                'feature_window': True/False,
                'factor_input': True/False,
                'calFactor': True/False,
                'training_loop': True/False,
                ...
            }
        
        Raises:
            ValueError: 如果输入数据不符合要求
        """
        # 验证输入
        self._validate_inputs(model, data)
        
        if self.config.verbose:
            self._print_detection_header()
        
        # 执行静态检测
        if self.static_detector:
            static_results = self.static_detector.run_all_tests(model, data)
            self.test_results.update(static_results)
            self.all_test_details.update(self.static_detector.get_test_results())
        
        # 执行动态检测
        if self.dynamic_detector:
            dynamic_results = self.dynamic_detector.run_all_tests(
                model, data, train_months, test_start_month, inference_months
            )
            self.test_results.update(dynamic_results)
            self.all_test_details.update(self.dynamic_detector.get_test_results())
        
        # 生成报告
        if self.config.generate_report:
            self.generate_report()
        
        # 显示摘要
        if self.config.show_summary:
            self.print_summary()
        
        return self.test_results
    
    def _validate_inputs(self, model: Any, data: pd.DataFrame):
        """
        验证输入数据
        
        Args:
            model: 模型对象
            data: 数据框
        
        Raises:
            ValueError: 如果输入不符合要求
        """
        # 检查模型方法
        if not hasattr(model, '_get_item'):
            raise ValueError("模型必须实现 _get_item(month) 方法")
        
        # 检查数据列
        required_columns = [self.config.time_column, self.config.stock_column]
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"数据缺少必需列: {missing_columns}")
        
        # 检查数据是否为空
        if len(data) == 0:
            raise ValueError("数据框为空")
    
    def _print_detection_header(self):
        """打印检测开始的标题"""
        print("\n" + "="*70)
        print("🔬 数据泄漏检测")
        print("="*70)
        print(f"检测模式: {self.config.test_mode.value}")
        print(f"时间列: {self.config.time_column}")
        print(f"股票列: {self.config.stock_column}")
        
        if self.config.test_mode == LeakageTestMode.STATIC_ONLY:
            checks = []
            if self.config.check_feature_window:
                checks.append("特征窗口")
            if self.config.check_factor_input:
                checks.append("因子输入")
            if self.config.check_calFactor:
                checks.append("calFactor")
            print(f"静态检查: {', '.join(checks)}")
        
        elif self.config.test_mode == LeakageTestMode.DYNAMIC_ONLY:
            checks = []
            if self.config.monitor_data_access:
                checks.append("数据访问")
            if self.config.monitor_cache_growth:
                checks.append("缓存增长")
            if self.config.enforce_time_boundary:
                checks.append("时间边界")
            print(f"动态监控: {', '.join(checks)}")
        
        else:  # FULL
            print("完整检测: 静态 + 动态")
        
        print("="*70)
    
    def print_summary(self):
        """打印测试摘要"""
        if not self.test_results:
            print("\n⚠️ 没有测试结果")
            return
        
        print("\n" + "="*70)
        print("📊 测试摘要")
        print("="*70)
        
        passed_count = sum(1 for passed in self.test_results.values() if passed)
        total_count = len(self.test_results)
        
        # 按类型分组
        static_tests = {}
        dynamic_tests = {}
        
        for test_name, passed in self.test_results.items():
            if test_name in ['feature_window', 'factor_input', 'calFactor']:
                static_tests[test_name] = passed
            else:
                dynamic_tests[test_name] = passed
        
        # 打印静态测试结果
        if static_tests:
            print("\n静态检测:")
            for test_name, passed in static_tests.items():
                status = "✅" if passed else "❌"
                test_display = self._get_test_display_name(test_name)
                print(f"  {status} {test_display}")
                
                # 显示失败详情
                if not passed and test_name in self.all_test_details:
                    details = self.all_test_details[test_name]
                    print(f"     └─ {details.get('message', '')}")
        
        # 打印动态测试结果
        if dynamic_tests:
            print("\n动态检测:")
            for test_name, passed in dynamic_tests.items():
                status = "✅" if passed else "❌"
                test_display = self._get_test_display_name(test_name)
                print(f"  {status} {test_display}")
                
                # 显示失败详情
                if not passed and test_name in self.all_test_details:
                    details = self.all_test_details[test_name]
                    print(f"     └─ {details.get('message', '')}")
        
        # 总结
        print(f"\n" + "="*70)
        print(f"总计: {passed_count}/{total_count} 测试通过")
        print("="*70)
        
        if passed_count == total_count:
            print("🎉 所有测试通过！未检测到数据泄漏。")
        else:
            print(f"⚠️ {total_count - passed_count} 个测试失败，存在数据泄漏风险！")
            print("\n建议:")
            self._print_recommendations()
    
    def _get_test_display_name(self, test_name: str) -> str:
        """获取测试的显示名称"""
        display_names = {
            'feature_window': '特征窗口时间泄漏',
            'factor_input': '因子输入同期泄漏',
            'calFactor': 'calFactor历史性',
            'training_loop': '训练循环数据访问',
            'inference_loop': '推理循环数据访问'
        }
        return display_names.get(test_name, test_name)
    
    def _print_recommendations(self):
        """打印修复建议"""
        recommendations = []
        
        for test_name, passed in self.test_results.items():
            if not passed:
                if test_name == 'feature_window':
                    recommendations.append(
                        "1. 特征窗口应使用 [t-T, t-1] 而不是 [t-T+1, t] 或 [t-T, t]"
                    )
                elif test_name == 'factor_input':
                    recommendations.append(
                        "2. 因子输入应使用 get_prev_month(month) 的数据"
                    )
                elif test_name == 'calFactor':
                    recommendations.append(
                        "3. calFactor 方法应使用历史月份数据，避免使用当前月"
                    )
                elif test_name == 'training_loop':
                    recommendations.append(
                        "4. 训练循环中不应访问测试期的数据"
                    )
        
        for rec in recommendations:
            print(f"   {rec}")
    
    def generate_report(self, report_path: Optional[str] = None) -> str:
        """
        生成详细的测试报告
        
        Args:
            report_path: 报告保存路径，如果为None则使用配置中的路径
        
        Returns:
            报告文件路径
        """
        if report_path is None:
            report_path = self.config.report_path
        
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # 标题
            f.write("="*70 + "\n")
            f.write("数据泄漏检测报告\n")
            f.write("="*70 + "\n\n")
            
            # 基本信息
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"检测模式: {self.config.test_mode.value}\n")
            f.write(f"详细输出: {self.config.verbose}\n")
            f.write("\n")
            
            # 配置信息
            f.write("配置参数:\n")
            f.write("-"*70 + "\n")
            f.write(f"  时间列: {self.config.time_column}\n")
            f.write(f"  股票列: {self.config.stock_column}\n")
            f.write(f"  收益率列: {self.config.return_column}\n")
            f.write(f"  标签列: {self.config.label_column}\n")
            f.write(f"  测试股票限制: {self.config.test_stocks_limit}\n")
            f.write(f"  精度: {self.config.epsilon}\n")
            f.write("\n")
            
            # 静态检测配置
            if self.config.test_mode in [LeakageTestMode.STATIC_ONLY, LeakageTestMode.FULL]:
                f.write("静态检测配置:\n")
                f.write("-"*70 + "\n")
                f.write(f"  特征窗口检查: {self.config.check_feature_window}\n")
                f.write(f"  因子输入检查: {self.config.check_factor_input}\n")
                f.write(f"  calFactor检查: {self.config.check_calFactor}\n")
                f.write(f"  源代码分析: {self.config.check_source_code}\n")
                f.write("\n")
            
            # 动态检测配置
            if self.config.test_mode in [LeakageTestMode.DYNAMIC_ONLY, LeakageTestMode.FULL]:
                f.write("动态检测配置:\n")
                f.write("-"*70 + "\n")
                f.write(f"  数据访问监控: {self.config.monitor_data_access}\n")
                f.write(f"  缓存增长监控: {self.config.monitor_cache_growth}\n")
                f.write(f"  时间边界强制: {self.config.enforce_time_boundary}\n")
                f.write(f"  最大缓存增长: {self.config.max_cache_growth}\n")
                f.write("\n")
            
            # 测试结果
            f.write("="*70 + "\n")
            f.write("测试结果\n")
            f.write("="*70 + "\n\n")
            
            passed_count = sum(1 for p in self.test_results.values() if p)
            total_count = len(self.test_results)
            
            f.write(f"总计: {passed_count}/{total_count} 测试通过\n\n")
            
            # 详细结果
            for test_name, passed in self.test_results.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                test_display = self._get_test_display_name(test_name)
                f.write(f"{status} {test_display}\n")
                
                if test_name in self.all_test_details:
                    details = self.all_test_details[test_name]
                    f.write(f"   消息: {details.get('message', '')}\n")
                    
                    if 'details' in details and details['details']:
                        f.write("   详情:\n")
                        for key, value in details['details'].items():
                            f.write(f"     - {key}: {value}\n")
                
                f.write("\n")
            
            # 建议
            if passed_count < total_count:
                f.write("="*70 + "\n")
                f.write("修复建议\n")
                f.write("="*70 + "\n\n")
                
                self._write_recommendations_to_file(f)
            
            f.write("\n" + "="*70 + "\n")
            f.write("报告结束\n")
            f.write("="*70 + "\n")
        
        if self.config.verbose:
            print(f"\n📝 测试报告已保存: {report_path}")
        
        return str(report_path)
    
    def _write_recommendations_to_file(self, f):
        """将修复建议写入文件"""
        for test_name, passed in self.test_results.items():
            if not passed:
                if test_name == 'feature_window':
                    f.write("1. 特��窗口时间泄漏:\n")
                    f.write("   - 问题: 特征窗口包含了当前月的数据\n")
                    f.write("   - 修复: 使用 [t-T, t-1] 窗口而不是 [t-T+1, t]\n")
                    f.write("   - 示例: window = stock_df.iloc[pos-T:pos] (不包含当前位置)\n\n")
                
                elif test_name == 'factor_input':
                    f.write("2. 因子输入同期泄漏:\n")
                    f.write("   - 问题: 因子输入使用了当前月的数据\n")
                    f.write("   - 修复: 使用上个月的数据作为因子输入\n")
                    f.write("   - 示例: factor = stock_df.loc[prev_month, 'rm_rf']\n\n")
                
                elif test_name == 'calFactor':
                    f.write("3. calFactor历史性问题:\n")
                    f.write("   - 问题: calFactor使用了当前月数据\n")
                    f.write("   - 修复: calFactor应使用历史数据\n")
                    f.write("   - 示例: prev_month = get_prev_month(month)\n")
                    f.write("           factor = self._get_item(prev_month)\n\n")
                
                elif test_name == 'training_loop':
                    f.write("4. 训练循环数据访问问题:\n")
                    f.write("   - 问题: 训练时访问了测试期的数据\n")
                    f.write("   - 修复: 确保训练循环只访问训练期数据\n")
                    f.write("   - 建议: 设置严格的时间边界检查\n\n")
    
    def get_test_results(self) -> Dict[str, bool]:
        """
        获取测试结果
        
        Returns:
            测试结果字典
        """
        return self.test_results
    
    def get_detailed_results(self) -> Dict[str, Dict]:
        """
        获取详细的测试结果（包含消息和详情）
        
        Returns:
            详细结果字典
        """
        return self.all_test_details
    
    def is_passed(self) -> bool:
        """
        判断是否所有测试都通过
        
        Returns:
            是否全部通过
        """
        if not self.test_results:
            return False
        return all(self.test_results.values())
    
    def get_failed_tests(self) -> List[str]:
        """
        获取失败的测试名称列表
        
        Returns:
            失败测试列表
        """
        return [name for name, passed in self.test_results.items() if not passed]
    
    # ========== 快捷工厂方法 ==========
    
    @classmethod
    def quick_check(cls, verbose: bool = True) -> 'LeakageDetector':
        """
        快速检查模式（仅静态检测）
        
        Args:
            verbose: 是否详细输出
        
        Returns:
            检测器实例
        """
        from .leakage_detection_config import LeakageDetectionTemplates
        config = LeakageDetectionTemplates.quick_check()
        config.verbose = verbose
        return cls(config)
    
    @classmethod
    def full_validation(cls, verbose: bool = True, generate_report: bool = True) -> 'LeakageDetector':
        """
        完整验证模式（静态+动态）
        
        Args:
            verbose: 是否详细输出
            generate_report: 是否生成报告
        
        Returns:
            检测器实例
        """
        from .leakage_detection_config import LeakageDetectionTemplates
        config = LeakageDetectionTemplates.full_validation()
        config.verbose = verbose
        config.generate_report = generate_report
        return cls(config)
    
    @classmethod
    def runtime_monitor(cls, report_path: str = './runtime_leakage_report.txt') -> 'LeakageDetector':
        """
        运行时监控模式（仅动态）
        
        Args:
            report_path: 报告路径
        
        Returns:
            检测器实例
        """
        from .leakage_detection_config import LeakageDetectionTemplates
        config = LeakageDetectionTemplates.runtime_monitor()
        config.report_path = report_path
        return cls(config)
