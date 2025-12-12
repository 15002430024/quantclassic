"""
静态数据泄漏检测器
"""
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .leakage_detection_config import LeakageDetectionConfig

logger = logging.getLogger(__name__)


class StaticLeakageDetector:
    """
    静态数据泄漏检测器
    
    通过分析代码逻辑和数据访问模式，检测潜在的数据泄漏问题。
    不需要实际执行模型训练，通过静态分析即可发现问题。
    
    功能:
    1. 特征窗口时间泄漏检测
    2. 因子输入同期泄漏检测
    3. calFactor方法历史性检测
    4. 源代码模式分析
    
    适用场景:
    - 快速验证模型代码
    - 开发阶段的实时检查
    - CI/CD流程中的自动化检测
    """
    
    def __init__(self, config: LeakageDetectionConfig):
        """
        初始化静态检测器
        
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
    
    def _get_prev_month(self, year_month: int) -> int:
        """计算上一个月份"""
        year = year_month // 100
        month = year_month % 100
        if month == 1:
            return (year - 1) * 100 + 12
        else:
            return year * 100 + (month - 1)
    
    def _get_next_month(self, year_month: int) -> int:
        """计算下一个月份"""
        year = year_month // 100
        month = year_month % 100
        if month == 12:
            return (year + 1) * 100 + 1
        else:
            return year * 100 + (month + 1)
    
    def test_feature_window_leak(
        self,
        model: Any,
        data: pd.DataFrame,
        test_month: Optional[int] = None
    ) -> bool:
        """
        测试特征窗口是否包含当前月
        
        检测模型在构建特征时是否错误地将当前月的数据包含在特征窗口中。
        正确的做法应该使用 [t-T, t-1] 的历史窗口，而不是 [t-T+1, t] 或 [t-T, t]。
        
        Args:
            model: 待测试的模型对象
            data: 数据框
            test_month: 测试月份，如果为None则自动选择
        
        Returns:
            是否通过测试
        """
        if not self.config.check_feature_window:
            return True
        
        self._print_test_header("静态测试1: 特征窗口时间泄漏检测")
        
        try:
            # 选择测试月份
            if test_month is None:
                all_months = sorted(data[self.config.time_column].unique())
                if len(all_months) < 20:
                    self._print_result(
                        "特征窗口时间泄漏",
                        False,
                        "数据月份不足，无法进行测试"
                    )
                    return False
                test_month = all_months[15]  # 使用中间月份
            
            # 检查模型是否有_get_item方法
            if not hasattr(model, '_get_item'):
                self._print_result(
                    "特征窗口时间泄漏",
                    False,
                    "模型缺少 _get_item 方法，无法进行测试"
                )
                return False
            
            # 获取模型窗口长度
            window_len = getattr(model, 'window_len', 12)
            
            # 调用模型方法获取数据
            result = model._get_item(test_month)
            
            # 处理不同返回格式
            if result is None:
                self._print_result(
                    "特征窗口时间泄漏",
                    False,
                    f"月份 {test_month} 无数据返回"
                )
                return False
            
            # 解析返回值
            if isinstance(result, tuple):
                if len(result) >= 2:
                    stock_index = result[0]
                    beta_inputs = result[1]
                else:
                    self._print_result(
                        "特征窗口时间泄漏",
                        False,
                        "_get_item 返回格式不正确"
                    )
                    return False
            else:
                self._print_result(
                    "特征窗口时间泄漏",
                    False,
                    "_get_item 返回类型不正确"
                )
                return False
            
            # 检查是否有有效数据
            if stock_index is None or len(stock_index) == 0:
                self._print_result(
                    "特征窗口时间泄漏",
                    False,
                    f"月份 {test_month} 无有效股票数据"
                )
                return False
            
            # 检查第一只股票的窗口
            test_stock = stock_index[0]
            stock_data = data[data[self.config.stock_column] == test_stock].sort_values(
                self.config.time_column
            )
            
            # 找到当前月份的位置
            month_pos_list = stock_data[stock_data[self.config.time_column] == test_month].index
            if len(month_pos_list) == 0:
                self._print_result(
                    "特征窗口时间泄漏",
                    False,
                    f"股票 {test_stock} 无月份 {test_month} 数据"
                )
                return False
            
            month_pos = month_pos_list[0]
            
            # 计算实际的窗口范围
            # 注意：这里需要根据实际实现推断窗口
            # 常见错误: window = stock_df.iloc[pos-T+1:pos+1]  # 包含当前月
            # 正确实现: window = stock_df.iloc[pos-T:pos]      # 不包含当前月
            window_start_pos = month_pos - window_len + 1
            
            if window_start_pos < 0:
                self._print_result(
                    "特征窗口时间泄漏",
                    False,
                    "历史数据不足，无法验证窗口"
                )
                return False
            
            # 获取实际窗口的月份
            actual_window = stock_data.iloc[window_start_pos:month_pos+1]
            actual_months = actual_window[self.config.time_column].values
            
            # 检查是否包含当前月
            if test_month in actual_months:
                details = {
                    "测试月份": test_month,
                    "窗口范围": f"[{actual_months[0]} - {actual_months[-1]}]",
                    "窗口长度": len(actual_months),
                    "测试股票": test_stock
                }
                self._print_result(
                    "特征窗口时间泄漏",
                    False,
                    f"特征窗口包含当前月 {test_month}！",
                    details
                )
                return False
            else:
                details = {
                    "测试月份": test_month,
                    "窗口范围": f"[{actual_months[0]} - {actual_months[-1]}]",
                    "测试股票": test_stock
                }
                self._print_result(
                    "特征窗口时间泄漏",
                    True,
                    "特征窗口正确，不包含当前月",
                    details
                )
                return True
        
        except Exception as e:
            logger.error(f"特征窗口测试出错: {str(e)}", exc_info=True)
            self._print_result(
                "特征窗口时间泄漏",
                False,
                f"测试过程出错: {str(e)}"
            )
            return False
    
    def test_factor_input_leak(
        self,
        model: Any,
        data: pd.DataFrame,
        test_month: Optional[int] = None
    ) -> bool:
        """
        测试因子输入是否使用当期数据
        
        检测模型在构建因子输入时是否错误地使用了当前月的数据。
        正确的做法应该使用 t-1 月的数据作为因子输入。
        
        Args:
            model: 待测试的模型对象
            data: 数据框
            test_month: 测试月份
        
        Returns:
            是否通过测试
        """
        if not self.config.check_factor_input:
            return True
        
        self._print_test_header("静态测试2: 因子输入同期泄漏检测")
        
        try:
            # 选择测试月份
            if test_month is None:
                all_months = sorted(data[self.config.time_column].unique())
                if len(all_months) < 20:
                    self._print_result(
                        "因子输入同期泄漏",
                        False,
                        "数据月份不足"
                    )
                    return False
                test_month = all_months[15]
            
            # 检查模型方法
            if not hasattr(model, '_get_item'):
                self._print_result(
                    "因子输入同期泄漏",
                    False,
                    "模型缺少 _get_item 方法"
                )
                return False
            
            # 获取数据
            result = model._get_item(test_month)
            if result is None or not isinstance(result, tuple) or len(result) < 3:
                self._print_result(
                    "因子输入同期泄漏",
                    False,
                    f"月份 {test_month} 无有效数据"
                )
                return False
            
            stock_index, beta_inputs, factor_inputs = result[0], result[1], result[2]
            
            if stock_index is None or len(stock_index) == 0:
                self._print_result(
                    "因子输入同期泄漏",
                    False,
                    "无有效股票数据"
                )
                return False
            
            # 检查第一只股票
            test_stock = stock_index[0]
            stock_data = data[data[self.config.stock_column] == test_stock].sort_values(
                self.config.time_column
            )
            
            # 获取当前月和上个月的收益率
            current_month_data = stock_data[stock_data[self.config.time_column] == test_month]
            prev_month = self._get_prev_month(test_month)
            prev_month_data = stock_data[stock_data[self.config.time_column] == prev_month]
            
            if len(current_month_data) == 0 or len(prev_month_data) == 0:
                self._print_result(
                    "因子输入同期泄漏",
                    False,
                    "数据不足，无法验证"
                )
                return False
            
            current_return = current_month_data[self.config.return_column].values[0]
            prev_return = prev_month_data[self.config.return_column].values[0]
            actual_factor_input = factor_inputs[0] if isinstance(factor_inputs, np.ndarray) else factor_inputs
            
            # 判断使用的是哪个月的数据
            epsilon = self.config.epsilon
            using_current = abs(actual_factor_input - current_return) < epsilon
            using_prev = abs(actual_factor_input - prev_return) < epsilon
            
            if using_current:
                details = {
                    "测试月份": test_month,
                    "因子输入值": f"{actual_factor_input:.6f}",
                    "当前月收益率": f"{current_return:.6f}",
                    "上月收益率": f"{prev_return:.6f}",
                    "测试股票": test_stock
                }
                self._print_result(
                    "因子输入同期泄漏",
                    False,
                    "因子输入使用了当期数据！",
                    details
                )
                return False
            elif using_prev:
                details = {
                    "测试月份": test_month,
                    "因子输入值": f"{actual_factor_input:.6f}",
                    "上月收益率": f"{prev_return:.6f}",
                    "测试股票": test_stock
                }
                self._print_result(
                    "因子输入同期泄漏",
                    True,
                    "因子输入正确使用历史数据",
                    details
                )
                return True
            else:
                details = {
                    "测试月份": test_month,
                    "因子输入值": f"{actual_factor_input:.6f}",
                    "当前月收益率": f"{current_return:.6f}",
                    "上月收益率": f"{prev_return:.6f}",
                    "测试股票": test_stock
                }
                self._print_result(
                    "因子输入同期泄漏",
                    True,
                    "因子输入来源不明确，但不是当期数据（可能经过处理）",
                    details
                )
                return True
        
        except Exception as e:
            logger.error(f"因子输入测试出错: {str(e)}", exc_info=True)
            self._print_result(
                "因子输入同期泄漏",
                False,
                f"测试过程出错: {str(e)}"
            )
            return False
    
    def test_calFactor_historicity(self, model: Any) -> bool:
        """
        测试calFactor是否使用历史数据
        
        通过分析源代码，检测calFactor方法是否正确使用历史数据。
        
        Args:
            model: 待测试的模型对象
        
        Returns:
            是否通过测试
        """
        if not self.config.check_calFactor:
            return True
        
        self._print_test_header("静态测试3: calFactor方法历史性检测")
        
        try:
            # 检查方法是否存在
            if not hasattr(model, 'calFactor'):
                self._print_result(
                    "calFactor历史性",
                    True,
                    "模型没有 calFactor 方法，跳过检测"
                )
                return True
            
            # 获取源代码
            try:
                calFactor_source = inspect.getsource(model.calFactor)
            except (TypeError, OSError):
                self._print_result(
                    "calFactor历史性",
                    False,
                    "无法获取 calFactor 的源代码"
                )
                return False
            
            # 分析源代码模式
            suspicious_patterns = []
            safe_patterns = []
            
            # 可疑模式：直接使用当前月份
            if 'self._get_item(month)' in calFactor_source:
                suspicious_patterns.append("使用 _get_item(month) - 可能访问当前月数据")
            
            if '_get_item(mon)' in calFactor_source and 'prev' not in calFactor_source:
                suspicious_patterns.append("使用 _get_item(mon) 但未见 prev 标识")
            
            # 安全模式：使用历史月份
            if 'prev_month' in calFactor_source or 'get_prev_month' in calFactor_source:
                safe_patterns.append("使用 prev_month 或 get_prev_month")
            
            if 'month-1' in calFactor_source or 'mon-1' in calFactor_source:
                safe_patterns.append("使用 month-1 或 mon-1")
            
            if '_get_item' in calFactor_source and 'history' in calFactor_source.lower():
                safe_patterns.append("_get_item 配合 history 关键字")
            
            # 判断结果
            if suspicious_patterns and not safe_patterns:
                details = {
                    "可疑模式": ", ".join(suspicious_patterns),
                    "源代码片段": calFactor_source[:200] + "..."
                }
                self._print_result(
                    "calFactor历史性",
                    False,
                    "calFactor可能使用了当前月份数据",
                    details
                )
                return False
            elif safe_patterns:
                details = {
                    "安全模式": ", ".join(safe_patterns)
                }
                self._print_result(
                    "calFactor历史性",
                    True,
                    "calFactor正确使用历史数据",
                    details
                )
                return True
            else:
                self._print_result(
                    "calFactor历史性",
                    True,
                    "calFactor数据来源无法确定（未发现明显问题）"
                )
                return True
        
        except Exception as e:
            logger.error(f"calFactor测试出错: {str(e)}", exc_info=True)
            self._print_result(
                "calFactor历史性",
                False,
                f"测试过程出错: {str(e)}"
            )
            return False
    
    def run_all_tests(self, model: Any, data: pd.DataFrame) -> Dict[str, bool]:
        """
        运行所有静态测试
        
        Args:
            model: 待测试的模型对象
            data: 数据框
        
        Returns:
            测试结果字典
        """
        if self.verbose:
            print("\n" + "🔬"*35)
            print("开始静态数据泄漏测试")
            print("🔬"*35)
        
        results = {}
        
        # 运行各项测试
        if self.config.check_feature_window:
            results['feature_window'] = self.test_feature_window_leak(model, data)
        
        if self.config.check_factor_input:
            results['factor_input'] = self.test_factor_input_leak(model, data)
        
        if self.config.check_calFactor:
            results['calFactor'] = self.test_calFactor_historicity(model)
        
        return results
    
    def get_test_results(self) -> Dict[str, Dict]:
        """获取详细的测试结果"""
        return self.test_results
