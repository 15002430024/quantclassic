"""
回测运行器 - 一键运行完整回测流程
整合因子处理、IC分析、组合构建、绩效评估、可视化
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

from .backtest_config import BacktestConfig
from .factor_processor import FactorProcessor
from .ic_analyzer import ICAnalyzer
from .portfolio_builder import PortfolioBuilder
from .performance_evaluator import PerformanceEvaluator
from .result_visualizer import ResultVisualizer


class BacktestRunner:
    """
    回测运行器 - 一键执行完整回测流程
    
    封装了完整的回测流程，从因子处理到可视化报告生成
    
    Example:
        >>> config = BacktestConfig(...)
        >>> runner = BacktestRunner(config)
        >>> results = runner.run_backtest(
        ...     factor_df=factor_df,
        ...     output_dir='output/backtest'
        ... )
        >>> 
        >>> # 访问结果
        >>> ic_stats = results['ic_stats']
        >>> metrics = results['metrics']
        >>> portfolios = results['portfolios']
    """
    
    def __init__(self, config: BacktestConfig):
        """
        初始化回测运行器
        
        Args:
            config: 回测配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化所有组件
        self.processor = FactorProcessor(config)
        self.ic_analyzer = ICAnalyzer(config)
        self.builder = PortfolioBuilder(config)
        self.evaluator = PerformanceEvaluator(config)
        self.visualizer = ResultVisualizer(config)
        
        self.logger.info("BacktestRunner 初始化完成")
    
    def run_backtest(self,
                     factor_df: pd.DataFrame,
                     factor_col: str = 'factor_value',
                     return_col: str = 'y_processed',
                     output_dir: Optional[str] = None,
                     save_plots: bool = True,
                     verbose: bool = True) -> Dict:
        """
        运行完整回测流程
        
        Args:
            factor_df: 因子数据，必须包含 order_book_id, trade_date, factor_col, return_col
            factor_col: 因子列名
            return_col: 收益列名
            output_dir: 输出目录，如果为 None 则使用 config 中的配置
            save_plots: 是否保存图表
            verbose: 是否打印进度信息
            
        Returns:
            Dict: 包含所有结果的字典
                - processed_df: 处理后的因子数据
                - ic_df: IC 分析结果
                - ic_stats: IC 统计指标
                - portfolios: 组合数据字典
                - metrics: 绩效指标字典
                - plots_dir: 图表保存路径（如果 save_plots=True）
        """
        if verbose:
            print("\n" + "=" * 80)
            print("🚀 开始回测流程")
            print("=" * 80)
        
        # 确定输出目录
        if output_dir is None:
            output_dir = self.config.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ========== 步骤 1: 因子预处理 ==========
        if verbose:
            print("\n【1/5】因子预处理...")
        
        processed_df = self.processor.process(
            factor_df,
            factor_cols=[factor_col]
        )
        
        # 使用标准化后的因子
        processed_factor_col = f'{factor_col}_std'
        
        if verbose:
            print(f"   ✓ 原始数据: {len(factor_df):,} 行")
            print(f"   ✓ 处理后数据: {len(processed_df):,} 行")
            print(f"   ✓ 因子列: {processed_factor_col}")
        
        # ========== 步骤 2: IC 分析 ==========
        if verbose:
            print("\n【2/5】IC 分析...")
        
        ic_df = self.ic_analyzer.calculate_ic(
            processed_df,
            factor_col=processed_factor_col,
            return_col=return_col
        )
        
        ic_stats = self.ic_analyzer.analyze_ic_statistics(ic_df)
        
        if verbose:
            print(f"   ✓ IC 均值: {ic_stats['ic_mean']:.4f}")
            print(f"   ✓ ICIR: {ic_stats['icir']:.4f}")
            print(f"   ✓ IC 胜率: {ic_stats['ic_win_rate']:.2%}")
            print(f"   ✓ t 统计量: {ic_stats['t_stat']:.4f} ({'显著' if abs(ic_stats['t_stat']) > 2 else '不显著'})")
        
        # ========== 步骤 3: 构建组合 ==========
        if verbose:
            print("\n【3/5】构建组合...")
        
        portfolios = self.builder.build_portfolios(
            processed_df,
            factor_col=processed_factor_col,
            return_col=return_col
        )
        
        if verbose:
            print(f"   ✓ 多头组合: {len(portfolios['long']):,} 期")
            print(f"   ✓ 空头组合: {len(portfolios['short']):,} 期")
            print(f"   ✓ 多空组合: {len(portfolios['long_short']):,} 期")
        
        # ========== 步骤 4: 绩效评估 ==========
        if verbose:
            print("\n【4/5】绩效评估...")
        
        metrics = {}
        for portfolio_name in ['long', 'short', 'long_short']:
            if portfolio_name in portfolios and 'portfolio_return' in portfolios[portfolio_name].columns:
                metrics[portfolio_name] = self.evaluator.evaluate_portfolio(
                    portfolios[portfolio_name],
                    return_col='portfolio_return',
                    benchmark_col=None
                )
        
        if verbose and 'long_short' in metrics:
            ls_metrics = metrics['long_short']
            print(f"   ✓ 年化收益: {ls_metrics['annual_return']:.2%}")
            print(f"   ✓ 夏普比率: {ls_metrics['sharpe_ratio']:.4f}")
            print(f"   ✓ 最大回撤: {ls_metrics['max_drawdown']:.2%}")
            print(f"   ✓ 卡玛比率: {ls_metrics['calmar_ratio']:.4f}")
        
        # ========== 步骤 5: 生成图表 ==========
        plots_dir = None
        if save_plots:
            if verbose:
                print("\n【5/5】生成可视化图表...")
            
            plots_dir = output_dir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            self.visualizer.create_comprehensive_report(
                portfolios=portfolios,
                ic_df=ic_df,
                metrics=metrics,
                output_dir=str(plots_dir)
            )
            
            if verbose:
                print(f"   ✓ 累计收益曲线")
                print(f"   ✓ 回撤曲线")
                print(f"   ✓ IC 时间序列")
                print(f"   ✓ IC 分布")
                print(f"   ✓ 分组收益")
                print(f"   ✓ 多空表现")
                print(f"\n   💾 图表已保存到: {plots_dir}")
        
        # ========== 保存数据 ==========
        if verbose:
            print("\n【数据保存】...")
        
        # 保存 IC 结果
        ic_df.to_csv(output_dir / 'ic_analysis.csv', index=False)
        
        # 保存组合数据
        for name, portfolio_df in portfolios.items():
            portfolio_df.to_csv(output_dir / f'portfolio_{name}.csv', index=False)
        
        # 保存绩效指标
        import json
        with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            # 转换 numpy 类型为 Python 原生类型
            metrics_serializable = {}
            for k, v in metrics.items():
                metrics_serializable[k] = {
                    key: float(val) if isinstance(val, (np.integer, np.floating)) else val
                    for key, val in v.items()
                }
            json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)
        
        with open(output_dir / 'ic_stats.json', 'w', encoding='utf-8') as f:
            ic_stats_serializable = {
                key: float(val) if isinstance(val, (np.integer, np.floating)) else val
                for key, val in ic_stats.items()
            }
            json.dump(ic_stats_serializable, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"   ✓ IC 分析: ic_analysis.csv")
            print(f"   ✓ 组合数据: portfolio_*.csv")
            print(f"   ✓ 绩效指标: metrics.json, ic_stats.json")
        
        # ========== 完成 ==========
        if verbose:
            print("\n" + "=" * 80)
            print("✅ 回测完成！")
            print("=" * 80)
            self._print_summary(ic_stats, metrics)
        
        # 返回所有结果
        return {
            'processed_df': processed_df,
            'ic_df': ic_df,
            'ic_stats': ic_stats,
            'portfolios': portfolios,
            'metrics': metrics,
            'plots_dir': str(plots_dir) if plots_dir else None,
            'output_dir': str(output_dir)
        }
    
    def _print_summary(self, ic_stats: Dict, metrics: Dict):
        """打印回测总结"""
        print("\n📋 回测总结")
        print("-" * 80)
        
        print("\n【因子效果】")
        print(f"  IC 均值: {ic_stats['ic_mean']:.4f}")
        print(f"  ICIR: {ic_stats['icir']:.4f}")
        print(f"  IC 胜率: {ic_stats['ic_win_rate']:.2%}")
        print(f"  显著性: {'✓ 显著 (|t|>2)' if abs(ic_stats['t_stat']) > 2 else '✗ 不显著'}")
        
        if 'long_short' in metrics:
            print("\n【多空组合】")
            ls = metrics['long_short']
            print(f"  年化收益: {ls['annual_return']:.2%}")
            print(f"  年化波动: {ls['annual_volatility']:.2%}")
            print(f"  夏普比率: {ls['sharpe_ratio']:.4f}")
            print(f"  最大回撤: {ls['max_drawdown']:.2%}")
            print(f"  卡玛比率: {ls['calmar_ratio']:.4f}")
            print(f"  胜率: {ls['win_rate']:.2%}")
        
        print("\n" + "-" * 80)
    
    def generate_report_text(self, 
                            ic_stats: Dict, 
                            metrics: Dict,
                            output_path: Optional[str] = None) -> str:
        """
        生成文本格式的回测报告
        
        Args:
            ic_stats: IC 统计指标
            metrics: 绩效指标
            output_path: 报告保存路径（可选）
            
        Returns:
            str: 报告文本
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("回测报告")
        report_lines.append("=" * 80)
        
        # 配置信息
        report_lines.append("\n【回测配置】")
        report_lines.append(f"  调仓频率: {self.config.rebalance_freq}")
        report_lines.append(f"  分组数量: {self.config.n_groups}")
        report_lines.append(f"  多空比例: 多 {self.config.long_ratio:.0%} / 空 {self.config.short_ratio:.0%}")
        report_lines.append(f"  交易成本: 佣金 {self.config.commission_rate:.4f} + 印花税 {self.config.stamp_tax_rate:.4f}")
        
        # IC 分析
        report_lines.append("\n【因子效果】")
        report_lines.append(f"  IC 均值: {ic_stats['ic_mean']:.4f}")
        report_lines.append(f"  IC 标准差: {ic_stats['ic_std']:.4f}")
        report_lines.append(f"  ICIR: {ic_stats['icir']:.4f}")
        report_lines.append(f"  IC 胜率: {ic_stats['ic_win_rate']:.2%}")
        report_lines.append(f"  t 统计量: {ic_stats['t_stat']:.4f}")
        report_lines.append(f"  p 值: {ic_stats.get('p_value', 0):.6f}")
        report_lines.append(f"  显著性: {'显著 (|t|>2)' if abs(ic_stats['t_stat']) > 2 else '不显著'}")
        
        # 组合表现
        for name in ['long', 'short', 'long_short']:
            if name in metrics:
                m = metrics[name]
                report_lines.append(f"\n【{name.upper()} 组合】")
                report_lines.append(f"  累计收益: {m['total_return']:.2%}")
                report_lines.append(f"  年化收益: {m['annual_return']:.2%}")
                report_lines.append(f"  年化波动: {m['annual_volatility']:.2%}")
                report_lines.append(f"  夏普比率: {m['sharpe_ratio']:.4f}")
                report_lines.append(f"  最大回撤: {m['max_drawdown']:.2%}")
                report_lines.append(f"  卡玛比率: {m['calmar_ratio']:.4f}")
                report_lines.append(f"  索提诺比率: {m['sortino_ratio']:.4f}")
                report_lines.append(f"  胜率: {m['win_rate']:.2%}")
                report_lines.append(f"  盈亏比: {m['profit_loss_ratio']:.4f}")
        
        report_lines.append("\n" + "=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # 保存报告
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            self.logger.info(f"报告已保存到: {output_path}")
        
        return report_text


if __name__ == '__main__':
    # 测试示例
    print("=" * 80)
    print("BacktestRunner 测试")
    print("=" * 80)
    
    # 创建配置
    config = BacktestConfig(
        output_dir='output/test_backtest',
        n_groups=10,
        rebalance_freq='weekly'
    )
    
    # 创建运行器
    runner = BacktestRunner(config)
    print(f"\n✅ BacktestRunner 创建成功")
    print(f"   配置: {config.n_groups} 组, {config.rebalance_freq} 调仓")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成")
    print("=" * 80)
