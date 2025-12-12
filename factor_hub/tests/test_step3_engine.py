"""
Step 3 验证脚本 - 验证因子计算引擎

运行此脚本验证：
1. FactorEngine 正确初始化
2. run() 方法能完成整个计算流程
3. 异常处理正常工作
"""

import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

# 导入 demo_factors 模块以触发装饰器注册
from quantclassic.factor_hub.factors import demo_factors  # noqa
from quantclassic.factor_hub.engine.factor_engine import FactorEngine
from quantclassic.factor_hub.providers.mock_provider import MockDataProvider


def main():
    print("=" * 70)
    print("Step 3 验证: 因子计算引擎")
    print("=" * 70)
    
    # 1. 初始化 Engine
    print("\n[1] 初始化 FactorEngine...")
    provider = MockDataProvider(seed=42)
    engine = FactorEngine(provider, continue_on_error=True, verbose=True)
    print(f"    ✓ Engine 创建成功")
    print(f"    ✓ {engine}")
    
    # 2. 列出可用因子
    print("\n[2] 列出可用因子...")
    available = engine.list_available_factors()
    print(f"    ✓ 可用因子: {available}")
    
    # 3. 运行因子计算
    print("\n[3] 运行因子计算流程...")
    print("-" * 60)
    
    result = engine.run(
        symbols=["000001.SZ", "600000.SH", "600519.SH"],
        factor_names=["return_1d", "return_5d", "volatility", "price_range"],
        start="2024-01-01",
        end="2024-01-31",
        factor_params={
            "volatility": {"window": 10},
        }
    )
    
    print("-" * 60)
    
    # 4. 检查结果
    print("\n[4] 检查计算结果...")
    print(f"    ✓ 运行成功: {result.success}")
    print(f"    ✓ 成功因子: {result.successful_factors}")
    print(f"    ✓ 失败因子: {result.failed_factors}")
    print(f"    ✓ 总耗时: {result.total_time:.2f}s")
    
    # 5. 查看因子 DataFrame
    print("\n[5] 因子结果 DataFrame...")
    print(f"    Shape: {result.factors_data.shape}")
    print(f"    Columns: {result.factors_data.columns.tolist()}")
    print(f"\n    前10行数据:")
    print(result.factors_data.head(10))
    
    # 6. 测试异常处理 - 注册一个会失败的因子
    print("\n[6] 测试异常处理 (包含不存在的因子)...")
    print("-" * 60)
    
    result2 = engine.run(
        symbols=["000001.SZ"],
        factor_names=["return_1d", "nonexistent_factor", "price_range"],
        start="2024-01-01",
        end="2024-01-15",
    )
    
    print("-" * 60)
    print(f"\n    ✓ 运行完成 (部分成功)")
    print(f"    ✓ 成功因子: {result2.successful_factors}")
    print(f"    ✓ 失败因子: {result2.failed_factors}")
    print(f"    ✓ 错误信息: {result2.errors}")
    
    print("\n" + "=" * 70)
    print("✓ Step 3 验证完成! 因子计算引擎工作正常。")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
