"""
backtest - 因子回测系统

基于 GeneralBacktest 引擎的回测框架，支持：
- 通用量化策略回测（GeneralBacktest 内嵌引擎）
- 因子处理（去极值、标准化、中性化）
- IC/ICIR 分析
- 因子权重生成与组合构建
- RollingWindowTrainer 多因子输出适配
- 多因子集成（mean, ic_weighted, best, custom）
- 基准管理

Quick Start:
    from quantclassic.backtest import GeneralBacktestAdapter, BacktestConfig

    config = BacktestConfig(
        rebalance_freq='monthly',
        long_ratio=0.2,
        short_ratio=0.2,
        buy_price='open',
        sell_price='close',
    )
    adapter = GeneralBacktestAdapter(config)
    results = adapter.run(
        factor_df=processed_df,
        price_df=price_df,
        factor_col='factor_raw_std',
        weight_mode='long_short',
    )

直接使用 GeneralBacktest:
    from quantclassic.backtest.general_backtest import GeneralBacktest

    bt = GeneralBacktest(start_date='2020-01-01', end_date='2025-12-31')
    results = bt.run_backtest(
        weights_data=weights_df,
        price_data=price_df,
        buy_price='open',
        sell_price='close',
        adj_factor_col='adj_factor',
        close_price_col='close',
    )
"""

from .backtest_config import BacktestConfig, FactorConfig, ConfigTemplates
from .factor_generator import FactorGenerator
from .factor_processor import FactorProcessor
from .portfolio_builder import PortfolioBuilder
from .ic_analyzer import ICAnalyzer
from .benchmark_manager import BenchmarkManager, get_benchmark_returns

# 多因子适配器
from .prediction_adapter import PredictionAdapter, adapt_predictions

# GeneralBacktest 回测引擎（内嵌）
from .general_backtest import GeneralBacktest
from .general_backtest_adapter import GeneralBacktestAdapter, is_general_backtest_available

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
    # 基准
    'BenchmarkManager',
    'get_benchmark_returns',
    # 多因子支持
    'PredictionAdapter',
    'adapt_predictions',
    # GeneralBacktest 回测引擎
    'GeneralBacktest',
    'GeneralBacktestAdapter',
    'is_general_backtest_available',
]

__version__ = '2.0.0'
