"""
测试 benchmark_manager 的缓存功能
"""

import sys
import logging
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_manager import BenchmarkManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_cache_basic():
    """测试基本缓存功能"""
    print("\n" + "=" * 80)
    print("测试1: 基本缓存功能")
    print("=" * 80)
    
    manager = BenchmarkManager()
    
    # 第一次获取：应该从API下载
    print("\n第一次获取数据（应该从API下载）...")
    try:
        returns1 = manager.get_benchmark_returns(
            'hs300',
            start_date='2023-01-01',
            end_date='2023-12-31',
            data_source='rqdatac'
        )
        print(f"✓ 成功获取 {len(returns1)} 条数据")
        print(f"  日期范围: {returns1.index[0].date()} ~ {returns1.index[-1].date()}")
        print(f"  累计收益率: {(1 + returns1).prod() - 1:.2%}")
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False
    
    # 第二次获取：应该从缓存加载
    print("\n第二次获取相同数据（应该从缓存加载）...")
    try:
        returns2 = manager.get_benchmark_returns(
            'hs300',
            start_date='2023-01-01',
            end_date='2023-12-31',
            data_source='rqdatac'
        )
        print(f"✓ 成功获取 {len(returns2)} 条数据")
        
        # 验证数据一致性
        if returns1.equals(returns2):
            print("✓ 数据一致性验证通过")
        else:
            print("✗ 数据不一致！")
            return False
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False
    
    return True


def test_cache_expansion():
    """测试缓存扩展功能"""
    print("\n" + "=" * 80)
    print("测试2: 缓存扩展功能（增量更新）")
    print("=" * 80)
    
    manager = BenchmarkManager()
    
    # 获取更大的日期范围：应该触发增量下载
    print("\n扩展日期范围（应该触发增量下载）...")
    try:
        returns_extended = manager.get_benchmark_returns(
            'hs300',
            start_date='2022-01-01',
            end_date='2024-06-30',
            data_source='rqdatac'
        )
        print(f"✓ 成功获取 {len(returns_extended)} 条数据")
        print(f"  日期范围: {returns_extended.index[0].date()} ~ {returns_extended.index[-1].date()}")
        print(f"  累计收益率: {(1 + returns_extended).prod() - 1:.2%}")
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False
    
    # 再次获取子范围：应该从缓存快速加载
    print("\n再次获取子范围（应该从缓存加载）...")
    try:
        returns_sub = manager.get_benchmark_returns(
            'hs300',
            start_date='2023-06-01',
            end_date='2023-12-31',
            data_source='rqdatac'
        )
        print(f"✓ 成功获取 {len(returns_sub)} 条数据")
        print(f"  日期范围: {returns_sub.index[0].date()} ~ {returns_sub.index[-1].date()}")
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False
    
    return True


def test_multiple_indices():
    """测试多个指数的缓存"""
    print("\n" + "=" * 80)
    print("测试3: 多个指数缓存")
    print("=" * 80)
    
    manager = BenchmarkManager()
    
    indices = ['hs300', 'zz500', 'zz800']
    
    for idx in indices:
        print(f"\n获取 {idx} 指数数据...")
        try:
            returns = manager.get_benchmark_returns(
                idx,
                start_date='2023-01-01',
                end_date='2023-12-31',
                data_source='rqdatac'
            )
            print(f"✓ {idx}: {len(returns)} 条数据, "
                  f"累计收益率 {(1 + returns).prod() - 1:.2%}")
        except Exception as e:
            print(f"✗ {idx} 失败: {e}")
    
    return True


def test_cache_info():
    """测试缓存信息查看"""
    print("\n" + "=" * 80)
    print("测试4: 查看缓存信息")
    print("=" * 80)
    
    manager = BenchmarkManager()
    
    try:
        cache_info = manager.get_cache_info()
        
        if cache_info.empty:
            print("当前没有缓存数据")
        else:
            print("\n当前缓存信息:")
            print(cache_info.to_string(index=False))
            print(f"\n共缓存 {len(cache_info)} 个指数")
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False
    
    return True


def test_no_cache_mode():
    """测试不使用缓存模式"""
    print("\n" + "=" * 80)
    print("测试5: 不使用缓存模式")
    print("=" * 80)
    
    manager = BenchmarkManager()
    
    print("\n获取数据（不使用缓存）...")
    try:
        returns = manager.get_benchmark_returns(
            'hs300',
            start_date='2023-01-01',
            end_date='2023-01-31',
            data_source='rqdatac',
            use_cache=False  # 关闭缓存
        )
        print(f"✓ 成功获取 {len(returns)} 条数据（直接从API）")
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("开始测试 BenchmarkManager 缓存功能")
    print("=" * 80)
    
    tests = [
        ("基本缓存功能", test_cache_basic),
        ("缓存扩展功能", test_cache_expansion),
        ("多指数缓存", test_multiple_indices),
        ("缓存信息查看", test_cache_info),
        ("不使用缓存模式", test_no_cache_mode),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"测试 {test_name} 出现异常: {e}", exc_info=True)
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\n总计: {passed}/{total} 测试通过")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
