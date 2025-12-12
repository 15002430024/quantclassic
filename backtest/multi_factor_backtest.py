"""
multi_factor_backtest.py - 多因子回测系统

专为 RollingWindowTrainer 多因子输出设计的回测流水线。
整合 PredictionAdapter、FactorProcessor、ICAnalyzer、PortfolioBuilder、PerformanceEvaluator。

Usage:
    from quantclassic.backtest.multi_factor_backtest import MultiFactorBacktest
    
    backtest = MultiFactorBacktest(config)
    results = backtest.run(
        rolling_predictions,
        stock_col='order_book_id',
        time_col='trade_date',
        label_col='y_ret_10d'
    )
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import json
from datetime import datetime

from .backtest_config import BacktestConfig
from .prediction_adapter import PredictionAdapter
from .factor_processor import FactorProcessor
from .ic_analyzer import ICAnalyzer
from .portfolio_builder import PortfolioBuilder
from .performance_evaluator import PerformanceEvaluator
from .result_visualizer import ResultVisualizer
from .benchmark_manager import BenchmarkManager


class MultiFactorBacktest:
    """
    多因子回测系统
    
    一站式因子回测流水线，专为 RollingWindowTrainer 的多因子输出设计。
    
    Pipeline:
        1. PredictionAdapter: 适配预测结果，多因子集成
        2. FactorProcessor: 因子标准化（去极值、Z-score、中性化）
        3. ICAnalyzer: IC/ICIR 分析
        4. PortfolioBuilder: 多空组合构建
        5. PerformanceEvaluator: 绩效评估
        6. ResultVisualizer: 可视化报告生成
    
    Features:
        - 支持多因子集成（mean, ic_weighted, best）
        - 自动因子预处理
        - 完整的 IC 分析（日度、月度、衰减）
        - 多空组合绩效评估
        - 自动化报告生成
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        初始化多因子回测系统
        
        Args:
            config: 回测配置
        """
        self.config = config or BacktestConfig()
        self.logger = self._setup_logger()
        
        # 初始化各组件
        self.adapter = PredictionAdapter(self.config)
        self.processor = FactorProcessor(self.config)
        self.ic_analyzer = ICAnalyzer(self.config)
        self.portfolio_builder = PortfolioBuilder(self.config)
        self.evaluator = PerformanceEvaluator(self.config)
        self.visualizer = ResultVisualizer(self.config)
        self.benchmark_manager = BenchmarkManager()
        
        # 结果缓存
        self._results: Dict[str, Any] = {}
        
        self.logger.info("=" * 60)
        self.logger.info("📊 多因子回测系统初始化")
        self.logger.info("=" * 60)
        self.logger.info(f"  输出目录: {self.config.output_dir}")
        self.logger.info(f"  分组数量: {self.config.n_groups}")
        self.logger.info(f"  调仓频率: {self.config.rebalance_freq}")
        self.logger.info(f"  基准指数: {self.config.benchmark_index or '未设置'}")
    
    def _setup_logger(self) -> logging.Logger:
        """配置日志"""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers and self.config.console_log:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run(
        self,
        predictions_df: pd.DataFrame,
        stock_col: str = 'order_book_id',
        time_col: str = 'trade_date',
        label_col: str = 'y_ret_10d',
        ensemble_method: str = 'mean',
        custom_weights: Optional[Dict[str, float]] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        运行完整回测流程
        
        Args:
            predictions_df: RollingWindowTrainer 的预测结果
            stock_col: 股票列名
            time_col: 时间列名
            label_col: 标签列名
            ensemble_method: 多因子集成方法 ('mean', 'ic_weighted', 'best', 'custom')
            custom_weights: 自定义权重（仅当 ensemble_method='custom' 时使用）
            save_results: 是否保存结果
            
        Returns:
            回测结果字典，包含：
            - adapted_df: 适配后的数据
            - processed_df: 处理后的因子数据
            - ic_df: IC 时间序列
            - ic_stats: IC 统计指标
            - portfolios: 组合收益数据
            - metrics: 绩效指标
            - factor_ics: 各因子 IC
            - ensemble_weights: 集成权重
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🚀 开始多因子回测")
        self.logger.info("=" * 60)
        
        results = {}
        
        # ========== Step 1: 适配预测结果 ==========
        self.logger.info("\n【1/6】适配预测结果")
        adapted_df = self.adapter.adapt(
            predictions_df,
            stock_col=stock_col,
            time_col=time_col,
            label_col=label_col,
            ensemble_method=ensemble_method,
            custom_weights=custom_weights,
            output_factor_col='factor_raw'
        )
        results['adapted_df'] = adapted_df
        results['factor_ics'] = dict(self.adapter.get_factor_ics())  # 复制以避免引用问题
        results['ensemble_weights'] = dict(self.adapter.get_ensemble_weights())  # 复制以避免引用问题
        
        # ========== Step 2: 因子处理 ==========
        self.logger.info("\n【2/6】因子预处理")
        processed_df = self.processor.process(adapted_df, factor_cols=['factor_raw'])
        results['processed_df'] = processed_df
        
        # 确定使用的因子列（处理后）
        factor_col = 'factor_raw_std'
        return_col = 'y_true' if 'y_true' in processed_df.columns else 'y_processed'
        
        # ========== Step 3: IC 分析 ==========
        self.logger.info("\n【3/6】IC 分析")
        ic_df = self.ic_analyzer.calculate_ic(processed_df, factor_col, return_col)
        ic_stats = self.ic_analyzer.analyze_ic_statistics(ic_df)
        
        results['ic_df'] = ic_df
        results['ic_stats'] = ic_stats
        
        self._print_ic_stats(ic_stats)
        
        # ========== Step 4: 构建组合 ==========
        self.logger.info("\n【4/6】构建多空组合")
        portfolios = self.portfolio_builder.build_portfolios(
            processed_df, factor_col, return_col
        )
        
        # ========== Step 4.5: 获取基准收益（新增） ==========
        benchmark_returns = None
        if self.config.benchmark_index:
            self.logger.info(f"\n【4.5】获取基准收益: {self.config.benchmark_index}")
            try:
                # 获取日期范围
                if 'long_short' in portfolios and 'trade_date' in portfolios['long_short'].columns:
                    ls_df = portfolios['long_short']
                    start_date = pd.to_datetime(ls_df['trade_date'].min()).strftime('%Y-%m-%d')
                    end_date = pd.to_datetime(ls_df['trade_date'].max()).strftime('%Y-%m-%d')
                    
                    # 从 BenchmarkManager 获取基准收益
                    benchmark_returns = self.benchmark_manager.get_benchmark_returns(
                        self.config.benchmark_index,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    self.logger.info(f"     ✓ 成功获取基准数据: {len(benchmark_returns)} 条")
                    self.logger.info(f"     日期范围: {start_date} ~ {end_date}")
                    
                    # 将基准收益合并到各组合 DataFrame
                    benchmark_df = pd.DataFrame({
                        'trade_date': benchmark_returns.index,
                        'benchmark_return': benchmark_returns.values
                    })
                    benchmark_df['trade_date'] = pd.to_datetime(benchmark_df['trade_date'])
                    
                    for name in portfolios:
                        if 'trade_date' in portfolios[name].columns:
                            portfolios[name]['trade_date'] = pd.to_datetime(portfolios[name]['trade_date'])
                            portfolios[name] = pd.merge(
                                portfolios[name], 
                                benchmark_df, 
                                on='trade_date', 
                                how='left'
                            )
                            portfolios[name]['benchmark_return'] = portfolios[name]['benchmark_return'].fillna(0)
                            
                            # 计算基准累计收益
                            if 'benchmark_return' in portfolios[name].columns:
                                portfolios[name]['benchmark_cumret'] = (1 + portfolios[name]['benchmark_return']).cumprod() - 1
                    
                    self.logger.info(f"     ✓ 已将基准收益合并到组合数据")
                else:
                    self.logger.warning("     ⚠️ 无法获取组合日期范围，跳过基准获取")
                    
            except Exception as e:
                self.logger.warning(f"     ⚠️ 获取基准收益失败: {e}")
                benchmark_returns = None
        
        results['portfolios'] = portfolios
        results['benchmark_returns'] = benchmark_returns
        
        # ========== Step 5: 绩效评估 ==========
        self.logger.info("\n【5/6】绩效评估")
        metrics = {}
        
        # 确定是否有基准列
        benchmark_col = 'benchmark_return' if benchmark_returns is not None else None
        
        for name, portfolio_df in portfolios.items():
            if 'portfolio_return' in portfolio_df.columns:
                m = self.evaluator.evaluate_portfolio(
                    portfolio_df, 
                    benchmark_col=benchmark_col
                )
                metrics[name] = m
        
        results['metrics'] = metrics
        self._print_metrics_summary(metrics, has_benchmark=(benchmark_col is not None))
        
        # ========== Step 6: 可视化与保存 ==========
        self.logger.info("\n【6/6】生成报告")
        
        if save_results:
            self._save_results(results)
        
        if self.config.save_plots:
            plot_dir = Path(self.config.output_dir) / 'plots'
            plot_dir.mkdir(parents=True, exist_ok=True)
            self.visualizer.create_comprehensive_report(
                portfolios, ic_df, metrics, str(plot_dir),
                benchmark_name=self.config.benchmark_index
            )
        
        # 汇总
        self.logger.info("\n" + "=" * 60)
        self.logger.info("✅ 多因子回测完成!")
        self.logger.info("=" * 60)
        
        self._results = results
        return results
    
    def run_multi_ensemble(
        self,
        predictions_df: pd.DataFrame,
        stock_col: str = 'order_book_id',
        time_col: str = 'trade_date',
        label_col: str = 'y_ret_10d',
        methods: List[str] = ['mean', 'ic_weighted', 'best']
    ) -> Dict[str, Dict[str, Any]]:
        """
        对比多种集成方法
        
        Args:
            predictions_df: 预测结果
            stock_col: 股票列名
            time_col: 时间列名
            label_col: 标签列名
            methods: 要对比的集成方法列表
            
        Returns:
            各方法的回测结果字典
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 多集成方法对比")
        self.logger.info("=" * 60)
        
        all_results = {}
        
        for method in methods:
            self.logger.info(f"\n>>> 测试集成方法: {method}")
            
            # 修改输出目录
            original_output = self.config.output_dir
            self.config.output_dir = f"{original_output}_{method}"
            
            results = self.run(
                predictions_df,
                stock_col=stock_col,
                time_col=time_col,
                label_col=label_col,
                ensemble_method=method,
                save_results=True
            )
            
            all_results[method] = results
            
            # 恢复输出目录
            self.config.output_dir = original_output
        
        # 打印对比结果
        self._print_comparison(all_results)
        
        return all_results
    
    def _print_ic_stats(self, stats: Dict[str, Any]):
        """打印 IC 统计"""
        self.logger.info(f"\n  📊 IC 统计:")
        self.logger.info(f"     IC 均值:   {stats['ic_mean']:+.4f}")
        self.logger.info(f"     IC 标准差: {stats['ic_std']:.4f}")
        self.logger.info(f"     ICIR:      {stats['icir']:.4f}")
        self.logger.info(f"     IC 胜率:   {stats['ic_win_rate']:.2%}")
        self.logger.info(f"     t 统计量:  {stats['t_stat']:.4f}")
    
    def _print_metrics_summary(self, metrics: Dict[str, Dict[str, float]], has_benchmark: bool = False):
        """打印绩效汇总"""
        self.logger.info(f"\n  📈 绩效汇总:")
        
        for name in ['long', 'short', 'long_short']:
            if name in metrics:
                m = metrics[name]
                base_info = (
                    f"     {name:12s}: "
                    f"年化={m['annual_return']:+.2%}, "
                    f"波动={m['annual_volatility']:.2%}, "
                    f"夏普={m['sharpe_ratio']:.2f}, "
                    f"回撤={m['max_drawdown']:.2%}"
                )
                self.logger.info(base_info)
                
                # 如果有基准，打印相对指标
                if has_benchmark and 'excess_return' in m:
                    relative_info = (
                        f"                   "
                        f"超额={m.get('excess_return', 0):+.2%}, "
                        f"IR={m.get('information_ratio', 0):.2f}, "
                        f"Alpha={m.get('alpha', 0):+.2%}, "
                        f"Beta={m.get('beta', 0):.2f}"
                    )
                    self.logger.info(relative_info)
    
    def _print_comparison(self, all_results: Dict[str, Dict[str, Any]]):
        """打印多方法对比"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 集成方法对比汇总")
        self.logger.info("=" * 60)
        
        comparison = []
        for method, results in all_results.items():
            ic_stats = results.get('ic_stats', {})
            metrics = results.get('metrics', {})
            ls_metrics = metrics.get('long_short', {})
            
            comparison.append({
                'Method': method,
                'IC_Mean': ic_stats.get('ic_mean', 0),
                'ICIR': ic_stats.get('icir', 0),
                'Annual_Return': ls_metrics.get('annual_return', 0),
                'Sharpe': ls_metrics.get('sharpe_ratio', 0),
                'Max_Drawdown': ls_metrics.get('max_drawdown', 0)
            })
        
        comparison_df = pd.DataFrame(comparison)
        print(comparison_df.to_string(index=False))
    
    def _save_results(self, results: Dict[str, Any]):
        """保存结果"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存适配后数据
        if 'adapted_df' in results:
            results['adapted_df'].to_parquet(output_dir / 'adapted_predictions.parquet')
        
        # 保存 IC
        if 'ic_df' in results:
            results['ic_df'].to_csv(output_dir / 'ic_analysis.csv', index=False)
        
        # 保存 IC 统计
        if 'ic_stats' in results:
            with open(output_dir / 'ic_stats.json', 'w') as f:
                json.dump(results['ic_stats'], f, indent=2, default=str)
        
        # 保存组合
        if 'portfolios' in results:
            for name, df in results['portfolios'].items():
                df.to_csv(output_dir / f'portfolio_{name}.csv', index=False)
        
        # 保存绩效指标
        if 'metrics' in results:
            metrics_df = pd.DataFrame(results['metrics']).T
            metrics_df.to_csv(output_dir / 'performance_metrics.csv')
            
            if self.config.generate_excel:
                metrics_df.to_excel(output_dir / 'performance_metrics.xlsx')
        
        # 保存因子 IC
        if 'factor_ics' in results and results['factor_ics']:
            with open(output_dir / 'factor_ics.json', 'w') as f:
                json.dump(results['factor_ics'], f, indent=2)
        
        self.logger.info(f"\n  💾 结果已保存至: {output_dir}")
    
    def get_results(self) -> Dict[str, Any]:
        """获取最近一次回测结果"""
        return self._results


# ==================== 便捷函数 ====================

def run_factor_backtest(
    predictions_df: pd.DataFrame,
    stock_col: str = 'order_book_id',
    time_col: str = 'trade_date',
    label_col: str = 'y_ret_10d',
    ensemble_method: str = 'mean',
    output_dir: str = 'output/backtest',
    n_groups: int = 10,
    rebalance_freq: str = 'biweekly'
) -> Dict[str, Any]:
    """
    便捷函数：一键运行因子回测
    
    Example:
        from quantclassic.backtest.multi_factor_backtest import run_factor_backtest
        
        results = run_factor_backtest(
            rolling_predictions,
            stock_col='order_book_id',
            time_col='trade_date',
            label_col='y_ret_10d',
            ensemble_method='ic_weighted'
        )
    """
    config = BacktestConfig(
        output_dir=output_dir,
        n_groups=n_groups,
        rebalance_freq=rebalance_freq,
        save_plots=True,
        generate_excel=True
    )
    
    backtest = MultiFactorBacktest(config)
    return backtest.run(
        predictions_df,
        stock_col=stock_col,
        time_col=time_col,
        label_col=label_col,
        ensemble_method=ensemble_method
    )
