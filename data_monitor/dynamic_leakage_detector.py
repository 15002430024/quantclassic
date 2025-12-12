"""
动态数据访问监控器和动态泄漏检测器
"""
import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict

from .leakage_detection_config import LeakageDetectionConfig

logger = logging.getLogger(__name__)


class DataAccessMonitor:
    """
    数据访问监控器
    
    在模型训练/推理过程中监控数据访问行为，检测时间边界违规和异常缓存增长。
    
    功能:
    1. 记录所有数据访问操作
    2. 检测时间边界违规
    3. 监控缓存增长
    4. 生成访问日志
    
    适用场景:
    - 训练循环中的实时监控
    - 推理过程的数据访问验证
    - 性能分析和优化
    """
    
    def __init__(self, model: Any, config: LeakageDetectionConfig):
        """
        初始化数据访问监控器
        
        Args:
            model: 待监控的模型对象
            config: 配置对象
        """
        self.model = model
        self.config = config
        self.verbose = config.verbose
        
        # 访问日志
        self.access_log = []
        
        # 时间边界
        self.current_time_boundary = None
        
        # 缓存快照
        self.cache_snapshots = []
        
        # 违规记录
        self.violations = []
        
        # 原始方法备份
        self.original_methods = {}
    
    def set_time_boundary(self, max_month: int):
        """
        设置时间边界
        
        Args:
            max_month: 允许访问的最大月份（不包含）
        """
        self.current_time_boundary = max_month
        if self.verbose:
            print(f"   🕐 时间边界设置: 只能访问 < {max_month} 的数据")
    
    def log_access(self, method_name: str, accessed_month: int):
        """
        记录数据访问
        
        Args:
            method_name: 方法名称
            accessed_month: 访问的月份
        """
        access_record = {
            'method': method_name,
            'month': accessed_month,
            'boundary': self.current_time_boundary
        }
        self.access_log.append(access_record)
        
        # 检查时间边界违规
        if self.config.enforce_time_boundary and self.current_time_boundary is not None:
            if accessed_month >= self.current_time_boundary:
                violation = {
                    'method': method_name,
                    'accessed_month': accessed_month,
                    'boundary': self.current_time_boundary,
                    'violation_type': 'time_boundary'
                }
                self.violations.append(violation)
                
                if self.verbose:
                    print(
                        f"   ⚠️ 时间泄漏！{method_name} 访问了 {accessed_month}，"
                        f"超过边界 {self.current_time_boundary}"
                    )
    
    def snapshot_cache(self, label: str = "") -> Dict:
        """
        记录缓存状态快照
        
        Args:
            label: 快照标签
        
        Returns:
            缓存快照信息
        """
        cache_size = 0
        cache_keys = []
        
        # 检查模型是否有缓存
        if hasattr(self.model, '_data_cache'):
            cache = self.model._data_cache
            if isinstance(cache, dict):
                cache_size = len(cache)
                cache_keys = list(cache.keys())
        
        snapshot = {
            'label': label,
            'size': cache_size,
            'keys': cache_keys[:10] if len(cache_keys) > 10 else cache_keys  # 只保存前10个键
        }
        
        self.cache_snapshots.append(snapshot)
        return snapshot
    
    def wrap_method(self, method_name: str) -> Callable:
        """
        包装方法以进行监控
        
        Args:
            method_name: 要包装的方法名称
        
        Returns:
            原始方法（用于恢复）
        """
        if not hasattr(self.model, method_name):
            logger.warning(f"模型没有方法 {method_name}，跳过包装")
            return None
        
        original_method = getattr(self.model, method_name)
        self.original_methods[method_name] = original_method
        
        @functools.wraps(original_method)
        def wrapped(*args, **kwargs):
            # 提取月份参数
            month = None
            if args:
                month = args[0]
            elif 'month' in kwargs:
                month = kwargs['month']
            elif 'mon' in kwargs:
                month = kwargs['mon']
            
            # 缓存快照（调用前）
            if self.config.monitor_cache_growth:
                before_cache = self.snapshot_cache(f"before_{method_name}")
            
            # 调用原始方法
            result = original_method(*args, **kwargs)
            
            # 缓存快照（调用后）
            if self.config.monitor_cache_growth:
                after_cache = self.snapshot_cache(f"after_{method_name}")
                cache_growth = after_cache['size'] - before_cache['size']
                
                # 检查异常增长
                if cache_growth > self.config.max_cache_growth:
                    violation = {
                        'method': method_name,
                        'cache_growth': cache_growth,
                        'violation_type': 'cache_growth'
                    }
                    self.violations.append(violation)
                    
                    if self.verbose:
                        print(
                            f"   ⚠️ 缓存异常增长！{method_name} 增加了 {cache_growth} 条"
                        )
            
            # 记录数据访问
            if month is not None and self.config.monitor_data_access:
                self.log_access(method_name, month)
            
            return result
        
        # 替换模型方法
        setattr(self.model, method_name, wrapped)
        return original_method
    
    def restore_method(self, method_name: str):
        """
        恢复原始方法
        
        Args:
            method_name: 方法名称
        """
        if method_name in self.original_methods:
            setattr(self.model, method_name, self.original_methods[method_name])
            del self.original_methods[method_name]
    
    def restore_all_methods(self):
        """恢复所有被包装的方法"""
        for method_name in list(self.original_methods.keys()):
            self.restore_method(method_name)
    
    def get_violations(self) -> List[Dict]:
        """获取所有违规记录"""
        return self.violations
    
    def get_access_log(self) -> List[Dict]:
        """获取访问日志"""
        return self.access_log
    
    def get_cache_snapshots(self) -> List[Dict]:
        """获取缓存快照"""
        return self.cache_snapshots


class DynamicLeakageDetector:
    """
    动态数据泄漏检测器
    
    通过模拟训练流程并监控数据访问，检测运行时的数据泄漏问题。
    
    功能:
    1. 训练循环模拟
    2. 数据访问监控
    3. 时间边界检查
    4. 缓存行为分析
    
    适用场景:
    - 训练前的完整验证
    - 推理流程的数据检查
    - 模型行为分析
    """
    
    def __init__(self, config: LeakageDetectionConfig):
        """
        初始化动态检测器
        
        Args:
            config: 数据泄漏检测配置对象
        """
        self.config = config
        self.test_results = {}
        self.verbose = config.verbose
    
    def _print_test_header(self, test_name: str):
        """打印测试标题"""
        if self.verbose:
            print("\n" + "="*70)
            print(f"🧪 {test_name}")
            print("="*70)
    
    def _print_result(self, test_name: str, passed: bool, message: str, details: Dict = None):
        """
        记录测试结果
        
        Args:
            test_name: 测试名称
            passed: 是否通过
            message: 结果消息
            details: 额外的详细信息
        """
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results[test_name] = {
            'passed': passed,
            'message': message,
            'details': details or {}
        }
        
        if self.verbose:
            print(f"\n{status}: {message}")
            if details and not passed:
                for key, value in details.items():
                    print(f"   └─ {key}: {value}")
    
    def test_training_loop_simulation(
        self,
        model: Any,
        data: pd.DataFrame,
        train_months: Optional[List[int]] = None,
        test_start_month: Optional[int] = None
    ) -> bool:
        """
        测试训练循环模拟
        
        模拟模型训练流程，监控每次迭代的数据访问，检测是否访问了未来数据。
        
        Args:
            model: 待测试的模型对象
            data: 数据框
            train_months: 训练月份列表，如果为None则自动选择
            test_start_month: 测试期开始月份，用于设置时间边界
        
        Returns:
            是否通过测试
        """
        if not self.config.monitor_data_access:
            return True
        
        self._print_test_header("动态测试1: 训练循环数据访问监控")
        
        try:
            # 自动选择训练月份
            if train_months is None:
                all_months = sorted(data[self.config.time_column].unique())
                if len(all_months) < 10:
                    self._print_result(
                        "训练循环数据访问",
                        False,
                        "数据月份不足"
                    )
                    return False
                # 选择前5个月作为训练
                train_months = all_months[:5]
            
            # 设置测试期开始
            if test_start_month is None:
                all_months = sorted(data[self.config.time_column].unique())
                test_start_month = all_months[len(all_months)//2]  # 使用中间月份
            
            if self.verbose:
                print(f"\n   📅 训练月份: {train_months}")
                print(f"   📅 测试期开始: {test_start_month}")
            
            # 创建监控器
            monitor = DataAccessMonitor(model, self.config)
            
            # 包装需要监控的方法
            methods_to_monitor = ['_get_item']
            if hasattr(model, 'calFactor'):
                methods_to_monitor.append('calFactor')
            
            for method_name in methods_to_monitor:
                monitor.wrap_method(method_name)
            
            # 设置时间边界
            monitor.set_time_boundary(test_start_month)
            
            # 初始化缓存
            if hasattr(model, '_data_cache'):
                model._data_cache = {}
            
            # 模拟训练循环
            for i, mon in enumerate(train_months):
                if self.verbose:
                    print(f"\n   🔄 迭代 {i+1}: 处理月份 {mon}")
                
                # 调用模型方法
                try:
                    if hasattr(model, '_get_item'):
                        model._get_item(mon)
                    
                    if hasattr(model, 'calFactor'):
                        model.calFactor(mon)
                except Exception as e:
                    if self.verbose:
                        print(f"   ⚠️ 方法调用出错: {str(e)}")
            
            # 恢复原始方法
            monitor.restore_all_methods()
            
            # 分析违规
            violations = monitor.get_violations()
            time_violations = [v for v in violations if v.get('violation_type') == 'time_boundary']
            cache_violations = [v for v in violations if v.get('violation_type') == 'cache_growth']
            
            # 生成结果
            if len(time_violations) > 0:
                details = {
                    "时间边界违规次数": len(time_violations),
                    "违规示例": str(time_violations[:3]),
                    "缓存异常次数": len(cache_violations)
                }
                self._print_result(
                    "训练循环数据访问",
                    False,
                    f"检测到 {len(time_violations)} 次未来数据访问！",
                    details
                )
                return False
            elif len(cache_violations) > 0:
                details = {
                    "缓存异常次数": len(cache_violations),
                    "违规示例": str(cache_violations[:3])
                }
                self._print_result(
                    "训练循环数据访问",
                    False,
                    f"检测到 {len(cache_violations)} 次缓存异常增长！",
                    details
                )
                return False
            else:
                details = {
                    "训练月份数": len(train_months),
                    "数据访问次数": len(monitor.get_access_log())
                }
                self._print_result(
                    "训练循环数据访问",
                    True,
                    "训练循环正常，未访问未来数据",
                    details
                )
                return True
        
        except Exception as e:
            logger.error(f"训练循环测试出错: {str(e)}", exc_info=True)
            self._print_result(
                "训练循环数据访问",
                False,
                f"测试过程出错: {str(e)}"
            )
            return False
    
    def test_inference_loop_simulation(
        self,
        model: Any,
        data: pd.DataFrame,
        inference_months: Optional[List[int]] = None
    ) -> bool:
        """
        测试推理循环模拟
        
        模拟模型推理流程，检测推理时是否访问了未来数据。
        
        Args:
            model: 待测试的模型对象
            data: 数据框
            inference_months: 推理月份列表
        
        Returns:
            是否通过测试
        """
        if not self.config.monitor_data_access:
            return True
        
        self._print_test_header("动态测试2: 推理循环数据访问监控")
        
        try:
            # 自动选择推理月份
            if inference_months is None:
                all_months = sorted(data[self.config.time_column].unique())
                if len(all_months) < 10:
                    self._print_result(
                        "推理循环数据访问",
                        False,
                        "数据月份不足"
                    )
                    return False
                # 选择后5个月作为推理
                inference_months = all_months[-5:]
            
            if self.verbose:
                print(f"\n   📅 推理月份: {inference_months}")
            
            # 创建监控器
            monitor = DataAccessMonitor(model, self.config)
            
            # 包装需要监控的方法
            methods_to_monitor = ['_get_item']
            if hasattr(model, 'calFactor'):
                methods_to_monitor.append('calFactor')
            
            for method_name in methods_to_monitor:
                monitor.wrap_method(method_name)
            
            # 推理循环 - 每次推理都设置独立的时间边界
            all_violations = []
            
            for i, mon in enumerate(inference_months):
                if self.verbose:
                    print(f"\n   🔄 推理 {i+1}: 月份 {mon}")
                
                # 设置时间边界（推理时不能访问当月及未来数据）
                monitor.set_time_boundary(mon)
                
                # 调用模型方法
                try:
                    if hasattr(model, '_get_item'):
                        model._get_item(mon)
                    
                    if hasattr(model, 'calFactor'):
                        model.calFactor(mon)
                except Exception as e:
                    if self.verbose:
                        print(f"   ⚠️ 方法调用出错: {str(e)}")
                
                # 收集当前月的违规
                current_violations = [
                    v for v in monitor.get_violations()
                    if v not in all_violations
                ]
                all_violations.extend(current_violations)
            
            # 恢复原始方法
            monitor.restore_all_methods()
            
            # 分析违规
            time_violations = [v for v in all_violations if v.get('violation_type') == 'time_boundary']
            
            if len(time_violations) > 0:
                details = {
                    "时间边界违规次数": len(time_violations),
                    "违规示例": str(time_violations[:3])
                }
                self._print_result(
                    "推理循环数据访问",
                    False,
                    f"推理时检测到 {len(time_violations)} 次未来数据访问！",
                    details
                )
                return False
            else:
                details = {
                    "推理月份数": len(inference_months),
                    "数据访问次数": len(monitor.get_access_log())
                }
                self._print_result(
                    "推理循环数据访问",
                    True,
                    "推理循环正常，未访问未来数据",
                    details
                )
                return True
        
        except Exception as e:
            logger.error(f"推理循环测试出错: {str(e)}", exc_info=True)
            self._print_result(
                "推理循环数据访问",
                False,
                f"测试过程出错: {str(e)}"
            )
            return False
    
    def run_all_tests(
        self,
        model: Any,
        data: pd.DataFrame,
        train_months: Optional[List[int]] = None,
        test_start_month: Optional[int] = None,
        inference_months: Optional[List[int]] = None
    ) -> Dict[str, bool]:
        """
        运行所有动态测试
        
        Args:
            model: 待测试的模型对象
            data: 数据框
            train_months: 训练月份列表
            test_start_month: 测试期开始月份
            inference_months: 推理月份列表
        
        Returns:
            测试结果字典
        """
        if self.verbose:
            print("\n" + "🚀"*35)
            print("开始动态监控测试")
            print("🚀"*35)
        
        results = {}
        
        # 运行训练循环测试
        if self.config.monitor_data_access:
            results['training_loop'] = self.test_training_loop_simulation(
                model, data, train_months, test_start_month
            )
        
        # 运行推理循环测试（可选）
        # results['inference_loop'] = self.test_inference_loop_simulation(
        #     model, data, inference_months
        # )
        
        return results
    
    def get_test_results(self) -> Dict[str, Dict]:
        """获取详细的测试结果"""
        return self.test_results
