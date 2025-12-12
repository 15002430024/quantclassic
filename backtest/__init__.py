"""
backtest - 因子回测系统

多因子回测框架，支持：
- 传统因子回测流程
- RollingWindowTrainer 多因子输出适配
- 多因子集成（mean, ic_weighted, best, custom）
- 完整的 IC 分析与绩效评估

Quick Start (多因子模型):
    from quantclassic.backtest import MultiFactorBacktest, run_factor_backtest
    
    # 方式1: 一键回测
    results = run_factor_backtest(
        rolling_predictions,
        stock_col='order_book_id',
        time_col='trade_date',
        label_col='y_ret_10d',
        ensemble_method='ic_weighted'
    )
    
    # 方式2: 完整控制
    backtest = MultiFactorBacktest(config)
    results = backtest.run(rolling_predictions, ...)
"""

from .backtest_config import BacktestConfig, FactorConfig, ConfigTemplates
from .factor_generator import FactorGenerator
from .factor_processor import FactorProcessor
from .portfolio_builder import PortfolioBuilder
from .ic_analyzer import ICAnalyzer
from .performance_evaluator import PerformanceEvaluator
from .result_visualizer import ResultVisualizer
from .result_visualizer_plotly import ResultVisualizerPlotly
from .backtest_system import FactorBacktestSystem
from .benchmark_manager import BenchmarkManager, get_benchmark_returns
from .backtest_runner import BacktestRunner

# 多因子适配器与回测系统
from .prediction_adapter import PredictionAdapter, adapt_predictions
from .multi_factor_backtest import MultiFactorBacktest, run_factor_backtest

__all__ = [
    # 配置
    'BacktestConfig',
    'FactorConfig',
    'ConfigTemplates',
    # 因子处理
    'FactorGenerator',
    'FactorProcessor',
    # 组合与分析
    'PortfolioBuilder',
    'ICAnalyzer',
    'PerformanceEvaluator',
    # 可视化
    'ResultVisualizer',
    'ResultVisualizerPlotly',
    # 回测系统
    'FactorBacktestSystem',
    'BacktestRunner',
    # 基准
    'BenchmarkManager',
    'get_benchmark_returns',
    # 多因子支持 (NEW)
    'PredictionAdapter',
    'adapt_predictions',
    'MultiFactorBacktest',
    'run_factor_backtest',
]

__version__ = '1.1.0'
