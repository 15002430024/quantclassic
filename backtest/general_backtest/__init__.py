"""
GeneralBacktest - 通用量化策略回测框架 (内嵌于 quantclassic)

原始项目: https://github.com/ElenYoung/GeneralBacktest
版本: 1.0.2
作者: Elen Young

支持特性：
- 灵活的调仓时间（不固定频率）
- 向量化计算（高性能）
- 丰富的性能指标（15+）
- 多样化的可视化（8+图表）

使用示例：
    from quantclassic.backtest.general_backtest import GeneralBacktest
    bt = GeneralBacktest(start_date="2023-01-01", end_date="2023-12-31")
    results = bt.run_backtest(weights_data, price_data, ...)
"""

from .backtest import GeneralBacktest

__version__ = '1.0.2'
__author__ = 'Elen Young'
__all__ = ['GeneralBacktest']
