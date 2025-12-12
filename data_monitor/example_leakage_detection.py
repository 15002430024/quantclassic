"""
数据泄漏检测示例脚本

演示如何使用 quantclassic.data_monitor 进行数据泄漏检测
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_monitor import (
    LeakageDetector,
    LeakageDetectionConfig,
    LeakageTestMode,
    LeakageDetectionTemplates
)


# ==================== 示例1: 快速检查 ====================
def example_quick_check():
    """示例1: 快速检查（仅静态检测）"""
    print("\n" + "="*70)
    print("示例1: 快速检查")
    print("="*70)
    
    # 创建检测器
    detector = LeakageDetector.quick_check(verbose=True)
    
    # 假设已有 model 和 data
    # results = detector.detect(model, data)
    
    print("\n✅ 快速检查示例完成")


# ==================== 示例2: 完整验证 ====================
def example_full_validation():
    """示例2: 完整验证（静态+动态）"""
    print("\n" + "="*70)
    print("示例2: 完整验证")
    print("="*70)
    
    # 创建检测器
    detector = LeakageDetector.full_validation(
        verbose=True,
        generate_report=True
    )
    
    # 执行检测
    # results = detector.detect(
    #     model=your_model,
    #     data=your_data,
    #     train_months=[200701, 200702, 200703],
    #     test_start_month=201901
    # )
    
    # 查看结果
    # if detector.is_passed():
    #     print("✅ 所有测试通过")
    # else:
    #     print("❌ 发现数据泄漏")
    #     print("失败测试:", detector.get_failed_tests())
    
    print("\n✅ 完整验证示例完成")


# ==================== 示例3: 自定义配置 ====================
def example_custom_config():
    """示例3: 自定义配置"""
    print("\n" + "="*70)
    print("示例3: 自定义配置")
    print("="*70)
    
    # 创建自定义配置
    config = LeakageDetectionConfig(
        test_mode=LeakageTestMode.FULL,
        verbose=True,
        
        # 列名配置
        time_column='year_month',
        stock_column='ts_code',
        return_column='rm_rf',
        label_column='target',
        
        # 静态检测
        check_feature_window=True,
        check_factor_input=True,
        check_calFactor=True,
        check_source_code=True,
        
        # 动态监控
        monitor_data_access=True,
        monitor_cache_growth=True,
        enforce_time_boundary=True,
        max_cache_growth=1000,
        
        # 测试配置
        test_stocks_limit=20,
        epsilon=1e-6,
        
        # 报告配置
        generate_report=True,
        report_path='./custom_leakage_report.txt',
        show_summary=True
    )
    
    # 创建检测器
    detector = LeakageDetector(config)
    
    print(f"配置模式: {config.test_mode.value}")
    print(f"时间列: {config.time_column}")
    print(f"报告路径: {config.report_path}")
    
    print("\n✅ 自定义配置示例完成")


# ==================== 示例4: 从YAML加载配置 ====================
def example_yaml_config():
    """示例4: 从YAML加载配置"""
    print("\n" + "="*70)
    print("示例4: 从YAML配置文件")
    print("="*70)
    
    # 创建示例YAML文件
    yaml_content = """
test_mode: full
verbose: true

time_column: year_month
stock_column: ts_code
return_column: rm_rf
label_column: target

check_feature_window: true
check_factor_input: true
check_calFactor: true

monitor_data_access: true
enforce_time_boundary: true

generate_report: true
report_path: ./yaml_leakage_report.txt
"""
    
    config_path = Path('./example_leakage_config.yaml')
    config_path.write_text(yaml_content)
    
    # 从YAML加载
    detector = LeakageDetector(str(config_path))
    
    print(f"从 {config_path} 加载配置")
    print(f"配置模式: {detector.config.test_mode.value}")
    
    # 清理
    config_path.unlink()
    
    print("\n✅ YAML配置示例完成")


# ==================== 示例5: 使用配置模板 ====================
def example_config_templates():
    """示例5: 使用配置模板"""
    print("\n" + "="*70)
    print("示例5: 使用配置模板")
    print("="*70)
    
    # 快速检查模板
    quick_config = LeakageDetectionTemplates.quick_check()
    print(f"快速检查模板: {quick_config.test_mode.value}")
    
    # 完整验证模板
    full_config = LeakageDetectionTemplates.full_validation()
    print(f"完整验证模板: {full_config.test_mode.value}")
    
    # 运行时监控模板
    runtime_config = LeakageDetectionTemplates.runtime_monitor()
    print(f"运行时监控模板: {runtime_config.test_mode.value}")
    
    # 自定义模板
    custom_config = LeakageDetectionTemplates.custom(
        test_mode=LeakageTestMode.FULL,
        verbose=True,
        check_feature_window=True,
        monitor_data_access=True
    )
    print(f"自定义模板: {custom_config.test_mode.value}")
    
    print("\n✅ 配置模板示例完成")


# ==================== 示例6: 结果处理 ====================
def example_result_processing():
    """示例6: 结果处理"""
    print("\n" + "="*70)
    print("示例6: 结果处理")
    print("="*70)
    
    # 创建检测器
    detector = LeakageDetector.quick_check(verbose=False)
    
    # 模拟测试结果
    detector.test_results = {
        'feature_window': True,
        'factor_input': False,
        'calFactor': True
    }
    
    detector.all_test_details = {
        'feature_window': {
            'passed': True,
            'message': '特征窗口正确',
            'details': {}
        },
        'factor_input': {
            'passed': False,
            'message': '因子输入使用了当期数据',
            'details': {'test_month': 200801}
        },
        'calFactor': {
            'passed': True,
            'message': 'calFactor使用历史数据',
            'details': {}
        }
    }
    
    # 获取结果
    print("\n1. 基本结果:")
    results = detector.get_test_results()
    for test, passed in results.items():
        print(f"  {test}: {'✅' if passed else '❌'}")
    
    # 判断是否通过
    print(f"\n2. 全部通过: {detector.is_passed()}")
    
    # 获取失败测试
    print(f"\n3. 失败测试: {detector.get_failed_tests()}")
    
    # 获取详细结果
    print("\n4. 详细结果:")
    detailed = detector.get_detailed_results()
    for test, info in detailed.items():
        if not info['passed']:
            print(f"  {test}:")
            print(f"    消息: {info['message']}")
            print(f"    详情: {info.get('details', {})}")
    
    print("\n✅ 结果处理示例完成")


# ==================== 示例7: CI/CD集成 ====================
def example_ci_cd_integration():
    """示例7: CI/CD集成"""
    print("\n" + "="*70)
    print("示例7: CI/CD集成")
    print("="*70)
    
    # 创建非详细模式的检测器
    config = LeakageDetectionConfig(
        test_mode=LeakageTestMode.FULL,
        verbose=False,  # CI环境不需要详细输出
        generate_report=True,
        show_summary=False,
        report_path='./ci_leakage_report.txt'
    )
    
    detector = LeakageDetector(config)
    
    # 执行检测
    # results = detector.detect(model, data)
    
    # CI检查
    # if not detector.is_passed():
    #     print("❌ 数据泄漏检测失败！")
    #     print(f"失败测试: {detector.get_failed_tests()}")
    #     sys.exit(1)  # 失败时退出
    # else:
    #     print("✅ 数据泄漏检测通过")
    #     sys.exit(0)
    
    print("CI/CD 配置:")
    print(f"  - 详细输出: {config.verbose}")
    print(f"  - 生成报告: {config.generate_report}")
    print(f"  - 显示摘要: {config.show_summary}")
    
    print("\n✅ CI/CD集成示例完成")


# ==================== 示例8: 批量检测 ====================
def example_batch_detection():
    """示例8: 批量检测多个模型"""
    print("\n" + "="*70)
    print("示例8: 批量检测")
    print("="*70)
    
    # 模拟多个模型
    model_names = ['Model_A', 'Model_B', 'Model_C']
    
    # 创建检测器
    detector = LeakageDetector.full_validation(verbose=False)
    
    # 批量检测
    results_summary = {}
    
    for name in model_names:
        print(f"\n检测 {name}...")
        
        # 模拟检测结果
        # results = detector.detect(model, data)
        
        # 记录结果
        # results_summary[name] = detector.is_passed()
        
        # if not detector.is_passed():
        #     print(f"  ⚠️ {name} 存在数据泄漏")
        #     print(f"  失败测试: {detector.get_failed_tests()}")
        # else:
        #     print(f"  ✅ {name} 通过检测")
    
    # 总结
    # passed_models = sum(results_summary.values())
    # total_models = len(results_summary)
    # print(f"\n总结: {passed_models}/{total_models} 个模型通过检测")
    
    print("\n✅ 批量检测示例完成")


# ==================== 主函数 ====================
def main():
    """运行所有示例"""
    print("="*70)
    print("数据泄漏检测示例")
    print("="*70)
    
    examples = [
        ("快速检查", example_quick_check),
        ("完整验证", example_full_validation),
        ("自定义配置", example_custom_config),
        ("YAML配置", example_yaml_config),
        ("配置模板", example_config_templates),
        ("结果处理", example_result_processing),
        ("CI/CD集成", example_ci_cd_integration),
        ("批量检测", example_batch_detection),
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n❌ {name} 示例出错: {e}")
    
    print("\n" + "="*70)
    print("所有示例完成")
    print("="*70)


if __name__ == '__main__':
    main()
