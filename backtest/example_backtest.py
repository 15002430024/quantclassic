"""
因子回测系统使用示例
演示如何使用 backtest 进行完整的因子回测
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from backtest import (
    BacktestConfig,
    ConfigTemplates,
    FactorBacktestSystem
)

# ============================================================================
# 示例1: 基础回测流程
# ============================================================================

def example_basic_backtest():
    """基础回测示例"""
    print("="*80)
    print("示例1: 基础因子回测")
    print("="*80)
    
    # 1. 创建配置
    config = BacktestConfig(
        data_dir='output',
        output_dir='output/backtest_results',
        batch_size=256,
        window_size=40,
        n_groups=10,
        rebalance_freq='monthly',
        ic_method='spearman',
        save_plots=True,
        generate_excel=True
    )
    
    # 2. 初始化回测系统
    backtest_system = FactorBacktestSystem(config)
    
    # 3. 加载模型
    model_path = 'output/best_model.pth'
    if os.path.exists(model_path):
        model = backtest_system.load_model(model_path)
    else:
        print(f"模型文件不存在: {model_path}")
        return
    
    # 4. 加载数据
    data_path = 'output/train_data_final_01.parquet'
    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        return
    
    df = pd.read_parquet(data_path)
    print(f"数据加载完成: {df.shape}")
    
    # 5. 运行回测
    results = backtest_system.run_backtest(
        data_df=df,
        factor_col='factor_raw_std',  # 使用标准化后的因子
        return_col='y_processed'      # 使用处理后的收益
    )
    
    # 6. 查看结果
    print("\n回测结果摘要:")
    print(f"  因子数据: {len(results['raw_factors'])} 条")
    print(f"  IC均值: {results['ic_stats']['ic_mean']:.4f}")
    print(f"  ICIR: {results['ic_stats']['icir']:.4f}")
    
    if 'long_short' in results['performance_metrics']:
        ls_metrics = results['performance_metrics']['long_short']
        print(f"\n多空组合绩效:")
        print(f"  年化收益: {ls_metrics['annual_return']:.2%}")
        print(f"  夏普比率: {ls_metrics['sharpe_ratio']:.4f}")
        print(f"  最大回撤: {ls_metrics['max_drawdown']:.2%}")
    
    return results


# ============================================================================
# 示例2: 使用预设配置模板
# ============================================================================

def example_config_templates():
    """使用配置模板示例"""
    print("\n" + "="*80)
    print("示例2: 使用预设配置模板")
    print("="*80)
    
    # 使用快速测试配置
    config = ConfigTemplates.fast_test()
    config.output_dir = 'output/fast_test_results'
    
    backtest_system = FactorBacktestSystem(config)
    
    # 其余步骤同示例1
    print("使用快速测试配置，适合初步验证")
    print(f"  批次大小: {config.batch_size}")
    print(f"  分组数: {config.n_groups}")
    print(f"  持有期: {config.holding_periods}")


# ============================================================================
# 示例3: 自定义因子处理流程
# ============================================================================

def example_custom_processing():
    """自定义因子处理示例"""
    print("\n" + "="*80)
    print("示例3: 自定义因子处理流程")
    print("="*80)
    
    # 创建自定义配置
    config = BacktestConfig(
        # 因子处理配置
        winsorize_method='mad',        # 使用MAD去极值
        mad_threshold=3.0,
        standardize_method='rank',     # 使用排序标准化
        industry_neutral=True,         # 行业中性化
        market_value_neutral=True,     # 市值中性化
        
        # 组合构建配置
        weight_method='factor_weight', # 因子加权
        long_ratio=0.3,                # 做多前30%
        short_ratio=0.3,               # 做空后30%
        
        # 成本配置
        consider_cost=True,
        commission_rate=0.0003,
        slippage_rate=0.001,
        
        # 输出配置
        output_dir='output/custom_backtest',
        save_plots=True
    )
    
    backtest_system = FactorBacktestSystem(config)
    
    print("自定义配置已创建:")
    print(f"  去极值方法: {config.winsorize_method}")
    print(f"  标准化方法: {config.standardize_method}")
    print(f"  行业中性化: {config.industry_neutral}")
    print(f"  权重方法: {config.weight_method}")


# ============================================================================
# 示例4: 分步骤运行回测
# ============================================================================

def example_step_by_step():
    """分步骤运行回测示例"""
    print("\n" + "="*80)
    print("示例4: 分步骤运行回测")
    print("="*80)
    
    config = BacktestConfig(output_dir='output/step_by_step')
    
    # 加载数据和模型
    data_path = 'output/train_data_final_01.parquet'
    model_path = 'output/best_model.pth'
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("数据或模型文件不存在")
        return
    
    df = pd.read_parquet(data_path)
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    
    # 步骤1: 生成因子
    from backtest import FactorGenerator
    factor_gen = FactorGenerator(model, config)
    factor_df = factor_gen.generate_factors(df)
    print(f"\n步骤1完成: 生成 {len(factor_df)} 条因子数据")
    
    # 步骤2: 处理因子
    from backtest import FactorProcessor
    processor = FactorProcessor(config)
    processed_df = processor.process(factor_df)
    print(f"步骤2完成: 因子处理完成")
    
    # 步骤3: IC分析
    from backtest import ICAnalyzer
    ic_analyzer = ICAnalyzer(config)
    
    # 需要添加收益列
    if 'y_processed' in df.columns:
        processed_df = pd.merge(
            processed_df,
            df[['ts_code', 'trade_date', 'y_processed']],
            on=['ts_code', 'trade_date'],
            how='left'
        )
        
        ic_df = ic_analyzer.calculate_ic(
            processed_df, 
            'factor_raw_std', 
            'y_processed'
        )
        ic_stats = ic_analyzer.analyze_ic_statistics(ic_df)
        print(f"步骤3完成: IC均值 = {ic_stats['ic_mean']:.4f}")
    
    # 步骤4: 构建组合
    from backtest import PortfolioBuilder
    portfolio_builder = PortfolioBuilder(config)
    portfolios = portfolio_builder.build_portfolios(
        processed_df, 
        'factor_raw_std', 
        'y_processed'
    )
    print(f"步骤4完成: 构建了 {len(portfolios)} 个组合")
    
    # 步骤5: 绩效评估
    from backtest import PerformanceEvaluator
    evaluator = PerformanceEvaluator(config)
    
    if 'long_short' in portfolios:
        metrics = evaluator.evaluate_portfolio(portfolios['long_short'])
        print(f"步骤5完成: 夏普比率 = {metrics['sharpe_ratio']:.4f}")
    
    # 步骤6: 可视化
    from backtest import ResultVisualizer
    visualizer = ResultVisualizer(config)
    visualizer.create_comprehensive_report(
        portfolios, ic_df, {}, 'output/step_by_step/plots'
    )
    print("步骤6完成: 图表已生成")


# ============================================================================
# 示例5: 多因子回测
# ============================================================================

def example_multi_factor():
    """多因子回测示例"""
    print("\n" + "="*80)
    print("示例5: 多因子回测")
    print("="*80)
    
    # 可以对多个因子列分别进行回测
    factor_cols = ['factor_0', 'factor_1', 'factor_2']
    
    config = BacktestConfig(output_dir='output/multi_factor')
    
    print("多因子回测配置:")
    print(f"  待测试因子: {factor_cols}")
    print("  可以循环对每个因子执行回测流程")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    print("因子回测系统使用示例\n")
    
    # 运行示例1: 基础回测
    # example_basic_backtest()
    
    # 运行示例2: 配置模板
    example_config_templates()
    
    # 运行示例3: 自定义处理
    example_custom_processing()
    
    # 运行示例4: 分步骤回测
    # example_step_by_step()
    
    # 运行示例5: 多因子回测
    example_multi_factor()
    
    print("\n" + "="*80)
    print("示例运行完成!")
    print("="*80)
