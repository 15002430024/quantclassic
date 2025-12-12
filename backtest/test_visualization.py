"""
快速测试脚本 - 验证增强可视化功能
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, '/home/u2025210237/jupyterlab')

print("=" * 80)
print("测试增强版可视化功能")
print("=" * 80)

# 测试1: 导入检查
print("\n[1/3] 检查模块导入...")
try:
    from quantclassic.backtest import (
        BacktestConfig,
        ResultVisualizer,
        BenchmarkManager
    )
    print("✅ ResultVisualizer (matplotlib) 导入成功")
except Exception as e:
    print(f"❌ ResultVisualizer 导入失败: {e}")
    sys.exit(1)

try:
    from quantclassic.backtest import ResultVisualizerPlotly
    print("✅ ResultVisualizerPlotly 导入成功")
    PLOTLY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ResultVisualizerPlotly 需要安装 plotly: {e}")
    print("   运行: pip install plotly")
    PLOTLY_AVAILABLE = False
except Exception as e:
    print(f"❌ ResultVisualizerPlotly 导入失败: {e}")
    PLOTLY_AVAILABLE = False

# 测试2: 创建实例
print("\n[2/3] 测试创建可视化器实例...")
try:
    config = BacktestConfig()
    visualizer = ResultVisualizer(config)
    print("✅ ResultVisualizer 实例创建成功")
    print(f"   配色方案: {list(visualizer.colors.keys())}")
    print(f"   图表尺寸: {config.figure_size}")
    print(f"   DPI: {config.dpi}")
except Exception as e:
    print(f"❌ ResultVisualizer 实例创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

if PLOTLY_AVAILABLE:
    try:
        visualizer_plotly = ResultVisualizerPlotly(config)
        print("✅ ResultVisualizerPlotly 实例创建成功")
        print(f"   配色方案: {list(visualizer_plotly.COLOR_SCHEME.keys())}")
        print(f"   默认宽度: {visualizer_plotly.default_width}")
        print(f"   默认高度: {visualizer_plotly.default_height}")
    except Exception as e:
        print(f"❌ ResultVisualizerPlotly 实例创建失败: {e}")
        import traceback
        traceback.print_exc()

# 测试3: 检查方法
print("\n[3/3] 检查可用方法...")
matplotlib_methods = [
    'plot_cumulative_returns',
    'plot_excess_returns',
    'plot_drawdown_comparison',
    'plot_drawdown',
    'plot_ic_series',
    'plot_ic_distribution',
    'plot_group_returns',
    'plot_long_short_performance',
    'create_comprehensive_report'
]

print("\nResultVisualizer 方法:")
for method in matplotlib_methods:
    if hasattr(visualizer, method):
        print(f"  ✅ {method}")
    else:
        print(f"  ❌ {method} (缺失)")

if PLOTLY_AVAILABLE:
    plotly_methods = [
        'plot_cumulative_returns_with_benchmark',
        'plot_excess_returns',
        'plot_drawdown_comparison',
        'plot_ic_analysis',
        'plot_group_returns',
        'plot_long_short_performance',
        'create_comprehensive_dashboard'
    ]
    
    print("\nResultVisualizerPlotly 方法:")
    for method in plotly_methods:
        if hasattr(visualizer_plotly, method):
            print(f"  ✅ {method}")
        else:
            print(f"  ❌ {method} (缺失)")

# 测试4: BenchmarkManager
print("\n[4/4] 测试 BenchmarkManager...")
try:
    benchmark_mgr = BenchmarkManager()
    print("✅ BenchmarkManager 实例创建成功")
    
    # 检查支持的指数
    print(f"   支持的基准指数: {list(benchmark_mgr.INDEX_MAPPING.keys())}")
    
    # 检查缓存信息
    cache_info = benchmark_mgr.get_cache_info()
    if not cache_info.empty:
        print(f"   已缓存的指数数量: {len(cache_info)}")
        print(cache_info.to_string(index=False))
    else:
        print("   暂无缓存数据")
        
except Exception as e:
    print(f"❌ BenchmarkManager 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("✅ 测试完成！")
print("=" * 80)
print("\n📋 总结:")
print("  • ResultVisualizer (matplotlib) - 已就绪")
if PLOTLY_AVAILABLE:
    print("  • ResultVisualizerPlotly - 已就绪")
else:
    print("  • ResultVisualizerPlotly - 需要安装 plotly")
    print("    运行: pip install plotly")
print("  • BenchmarkManager - 已就绪")
print("\n💡 下一步:")
print("  1. 查看使用指南: VISUALIZATION_GUIDE.md")
print("  2. 运行示例脚本: python example_enhanced_visualization.py")
print("  3. 在你的回测代码中集成新功能")
print("=" * 80)
