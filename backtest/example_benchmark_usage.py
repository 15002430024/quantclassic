"""
BenchmarkManager 快速使用示例
演示如何在实际项目中使用基准管理器
"""

import logging
from benchmark_manager import BenchmarkManager

# 配置日志
logging.basicConfig(level=logging.INFO)


def example_1_basic_usage():
    """示例1: 基本使用"""
    print("\n" + "=" * 60)
    print("示例1: 基本使用 - 获取沪深300指数收益率")
    print("=" * 60)
    
    manager = BenchmarkManager()
    
    # 获取2023年沪深300收益率
    returns = manager.get_benchmark_returns(
        'hs300',
        start_date='2023-01-01',
        end_date='2023-12-31',
        data_source='rqdatac'
    )
    
    print(f"\n数据概览:")
    print(f"  日期范围: {returns.index[0].date()} ~ {returns.index[-1].date()}")
    print(f"  数据点数: {len(returns)}")
    print(f"  平均日收益: {returns.mean():.4%}")
    print(f"  收益率标准差: {returns.std():.4%}")
    print(f"  累计收益率: {(1 + returns).prod() - 1:.2%}")
    print(f"  最大单日涨幅: {returns.max():.2%}")
    print(f"  最大单日跌幅: {returns.min():.2%}")


def example_2_compare_indices():
    """示例2: 对比多个指数"""
    print("\n" + "=" * 60)
    print("示例2: 对比多个指数的表现")
    print("=" * 60)
    
    manager = BenchmarkManager()
    
    indices = {
        'hs300': '沪深300',
        'zz500': '中证500',
        'zz800': '中证800'
    }
    
    print(f"\n{'指数':<10} {'累计收益':<12} {'年化波动':<12} {'夏普比率':<12}")
    print("-" * 50)
    
    for code, name in indices.items():
        try:
            returns = manager.get_benchmark_returns(
                code,
                start_date='2023-01-01',
                end_date='2023-12-31',
                data_source='rqdatac'
            )
            
            # 计算指标
            total_return = (1 + returns).prod() - 1
            annual_vol = returns.std() * (252 ** 0.5)
            sharpe = (returns.mean() * 252) / annual_vol if annual_vol > 0 else 0
            
            print(f"{name:<10} {total_return:>10.2%}  {annual_vol:>10.2%}  {sharpe:>10.2f}")
        
        except Exception as e:
            print(f"{name:<10} 获取失败: {e}")


def example_3_cache_info():
    """示例3: 查看缓存信息"""
    print("\n" + "=" * 60)
    print("示例3: 查看缓存信息")
    print("=" * 60)
    
    manager = BenchmarkManager()
    
    cache_info = manager.get_cache_info()
    
    if cache_info.empty:
        print("\n当前没有缓存数据")
    else:
        print(f"\n当前共有 {len(cache_info)} 个指数缓存:\n")
        print(cache_info.to_string(index=False))


def example_4_excess_returns():
    """示例4: 计算超额收益"""
    print("\n" + "=" * 60)
    print("示例4: 计算策略相对基准的超额收益")
    print("=" * 60)
    
    import pandas as pd
    import numpy as np
    
    manager = BenchmarkManager()
    
    # 获取基准收益率
    benchmark = manager.get_benchmark_returns(
        'hs300',
        start_date='2023-01-01',
        end_date='2023-12-31',
        data_source='rqdatac'
    )
    
    # 模拟策略收益率（这里用随机数模拟）
    np.random.seed(42)
    strategy_returns = pd.Series(
        benchmark.values + np.random.normal(0.001, 0.01, len(benchmark)),
        index=benchmark.index
    )
    
    # 计算超额收益
    excess = manager.calculate_excess_returns(strategy_returns, benchmark)
    
    print(f"\n策略表现分析:")
    print(f"  策略累计收益: {(1 + strategy_returns).prod() - 1:.2%}")
    print(f"  基准累计收益: {(1 + benchmark).prod() - 1:.2%}")
    print(f"  累计超额收益: {(1 + excess).prod() - 1:.2%}")
    print(f"  平均超额收益: {excess.mean():.4%}")
    print(f"  超额收益波动: {excess.std():.4%}")
    print(f"  信息比率: {(excess.mean() * 252) / (excess.std() * (252 ** 0.5)):.2f}")


def example_5_time_range_expansion():
    """示例5: 演示缓存扩展"""
    print("\n" + "=" * 60)
    print("示例5: 演示智能缓存扩展")
    print("=" * 60)
    
    manager = BenchmarkManager()
    
    print("\n第一步: 获取2023年数据（如果缓存不存在，会从API下载）")
    returns_2023 = manager.get_benchmark_returns(
        'zz800',
        start_date='2023-01-01',
        end_date='2023-12-31',
        data_source='rqdatac'
    )
    print(f"  获取 {len(returns_2023)} 条数据")
    
    print("\n第二步: 再次获取2023年数据（应该从缓存加载）")
    returns_2023_cached = manager.get_benchmark_returns(
        'zz800',
        start_date='2023-01-01',
        end_date='2023-12-31',
        data_source='rqdatac'
    )
    print(f"  获取 {len(returns_2023_cached)} 条数据")
    print(f"  数据一致: {returns_2023.equals(returns_2023_cached)}")
    
    print("\n第三步: 扩展到2022-2024年（会增量下载缺失部分）")
    returns_extended = manager.get_benchmark_returns(
        'zz800',
        start_date='2022-01-01',
        end_date='2024-06-30',
        data_source='rqdatac'
    )
    print(f"  获取 {len(returns_extended)} 条数据")
    print(f"  新的日期范围: {returns_extended.index[0].date()} ~ {returns_extended.index[-1].date()}")


def example_6_convenient_function():
    """示例6: 使用便捷函数"""
    print("\n" + "=" * 60)
    print("示例6: 使用便捷函数快速获取数据")
    print("=" * 60)
    
    from benchmark_manager import get_benchmark_returns
    
    # 直接调用便捷函数
    returns = get_benchmark_returns('hs300', '2023-01-01', '2023-12-31')
    
    print(f"\n一行代码获取数据:")
    print(f"  数据点数: {len(returns)}")
    print(f"  累计收益: {(1 + returns).prod() - 1:.2%}")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("BenchmarkManager 使用示例集")
    print("=" * 60)
    
    examples = [
        example_1_basic_usage,
        example_2_compare_indices,
        example_3_cache_info,
        example_4_excess_returns,
        example_5_time_range_expansion,
        example_6_convenient_function,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n示例执行失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
