"""
Step 2 验证脚本 - 验证因子基类与注册机制

运行此脚本验证：
1. FactorRegistry 装饰器正常工作
2. DemoFactor 已自动注册
3. 能手动调用 compute 方法
"""

import sys
sys.path.insert(0, "/home/u2025210237/jupyterlab/quantclassic")

# 导入 demo_factors 模块以触发装饰器注册
from factor_hub.factors import demo_factors  # noqa
from factor_hub.factors import factor_registry, BaseFactor
from factor_hub.providers.mock_provider import MockDataProvider
from factor_hub.protocols import StandardDataProtocol


def main():
    print("=" * 70)
    print("Step 2 验证: 因子基类与注册机制")
    print("=" * 70)
    
    # 1. 检查已注册的因子
    print("\n[1] 检查 FactorRegistry 中已注册的因子...")
    registered_factors = factor_registry.list_factors()
    print(f"    ✓ 已注册因子数量: {len(registered_factors)}")
    print(f"    ✓ 已注册因子列表: {registered_factors}")
    
    # 2. 检查 DemoFactor 是否在 Registry 中
    print("\n[2] 检查 DemoFactor 是否在 Registry 中...")
    expected_factors = ["return_1d", "return_5d", "volatility", "turnover_ratio", "price_range"]
    for factor_name in expected_factors:
        if factor_name in factor_registry:
            print(f"    ✓ {factor_name} 已注册")
        else:
            print(f"    ✗ {factor_name} 未注册")
            return False
    
    # 3. 按类别列出因子
    print("\n[3] 按类别列出因子...")
    factors_by_category = factor_registry.list_factors_by_category()
    for category, factors in factors_by_category.items():
        print(f"    ✓ {category}: {factors}")
    
    # 4. 获取因子类并实例化
    print("\n[4] 获取因子类并实例化...")
    Return1DFactor = factor_registry.get("return_1d")
    factor = Return1DFactor()
    print(f"    ✓ 因子实例: {factor}")
    print(f"    ✓ 因子名称: {factor.name}")
    print(f"    ✓ 因子描述: {factor.description}")
    print(f"    ✓ 因子类别: {factor.category}")
    print(f"    ✓ 因子参数: {factor.params}")
    
    # 5. 使用 factory 方法创建因子
    print("\n[5] 使用 registry.create() 创建因子...")
    vol_factor = factor_registry.create("volatility", window=10)
    print(f"    ✓ 因子实例: {vol_factor}")
    print(f"    ✓ 自定义参数 window: {vol_factor.params.get('window')}")
    
    # 6. 准备测试数据
    print("\n[6] 准备测试数据 (来自 Step 1 的 MockProvider)...")
    provider = MockDataProvider(seed=42)
    raw_data = provider.get_history(
        symbols=["000001.SZ", "600000.SH"],
        start="2024-01-01",
        end="2024-01-31"
    )
    std_data = StandardDataProtocol(raw_data)
    print(f"    ✓ 数据准备完成")
    
    # 7. 手动调用 compute 方法
    print("\n[7] 手动调用因子 compute 方法...")
    
    # 测试 return_1d
    return_factor = factor_registry.create("return_1d")
    returns = return_factor.compute(std_data)
    print(f"\n    [return_1d] 计算结果:")
    print(f"    - Shape: {returns.shape}")
    print(f"    - 非空值数量: {returns.notna().sum()}")
    print(f"    - 前5行:\n{returns.head()}")
    
    # 测试 volatility
    vol_factor = factor_registry.create("volatility", window=5)
    volatility = vol_factor.compute(std_data)
    print(f"\n    [volatility] 计算结果:")
    print(f"    - Shape: {volatility.shape}")
    print(f"    - 非空值数量: {volatility.notna().sum()}")
    print(f"    - 前5行:\n{volatility.head()}")
    
    # 测试 price_range
    range_factor = factor_registry.create("price_range")
    price_range = range_factor.compute(std_data)
    print(f"\n    [price_range] 计算结果:")
    print(f"    - Shape: {price_range.shape}")
    print(f"    - 非空值数量: {price_range.notna().sum()}")
    print(f"    - 前5行:\n{price_range.head()}")
    
    print("\n" + "=" * 70)
    print("✓ Step 2 验证完成! 因子基类与注册机制工作正常。")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
