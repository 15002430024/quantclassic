# 回测子系统 (backtest)

> 基于 GeneralBacktest 引擎的因子回测框架

本模块将 [GeneralBacktest](https://github.com/ElenYoung/GeneralBacktest) 引擎内嵌于 quantclassic，提供因子处理、IC 分析、权重生成、组合回测、可视化的完整链路。

## 快速上手

### 方式一：通过适配器（推荐，端到端因子→回测）

```python
from quantclassic.backtest import BacktestConfig, GeneralBacktestAdapter

config = BacktestConfig(
    rebalance_freq='monthly',
    long_ratio=0.2,
    short_ratio=0.2,
    buy_price='open',
    sell_price='close',
)

adapter = GeneralBacktestAdapter(config)
results = adapter.run(
    factor_df=processed_df,    # 含 trade_date, order_book_id, factor_col
    price_df=price_df,         # 含 trade_date, order_book_id, open, close, adj_factor
    factor_col='factor_raw_std',
    weight_mode='long_short',
)

# results 包含: nav_series, positions, trade_records, metrics, bt_instance
results['bt_instance'].plot_dashboard()
```

### 方式二：直接使用 GeneralBacktest

```python
from quantclassic.backtest.general_backtest import GeneralBacktest

bt = GeneralBacktest(start_date='2020-01-01', end_date='2025-12-31')
results = bt.run_backtest(
    weights_data=weights_df,     # [date, code, weight]
    price_data=price_df,         # [date, code, open, close, adj_factor]
    buy_price='open',
    sell_price='close',
    adj_factor_col='adj_factor',
    close_price_col='close',
)

bt.print_metrics()
bt.plot_dashboard()
```

## 架构

```
backtest/
├── general_backtest/               # GeneralBacktest 引擎（内嵌）
│   ├── __init__.py
│   ├── backtest.py                 # 核心回测引擎
│   └── utils.py                    # 指标计算工具
├── general_backtest_adapter.py     # quantclassic → GeneralBacktest 适配层
├── backtest_config.py              # 回测配置（BacktestConfig）
├── portfolio_builder.py            # 权重生成 + 组合构建
├── factor_generator.py             # 模型推理产因子
├── factor_processor.py             # 因子预处理（去极值/标准化/中性化）
├── ic_analyzer.py                  # IC/ICIR 分析
├── prediction_adapter.py           # 多因子预测适配与集成
├── benchmark_manager.py            # 基准指数管理
└── __init__.py
```

数据流：
```
因子生成/适配 → 因子预处理 → IC 分析 → 权重生成 → GeneralBacktest 回测 → 指标/可视化
```

## 配置要点（BacktestConfig）

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `buy_price` | 买入价格字段 | `'open'` |
| `sell_price` | 卖出价格字段 | `'close'` |
| `rebalance_freq` | 调仓频率 | `'monthly'` |
| `long_ratio` | 做多比例 | `0.2` |
| `short_ratio` | 做空比例 | `0.2` |
| `weight_method` | 权重方法 | `'equal'` |
| `consider_cost` | 是否计交易成本 | `False` |
| `commission_rate` | 佣金率 | `0.0003` |
| `slippage_rate` | 滑点率 | `0.001` |

## GeneralBacktest 功能

内嵌的 GeneralBacktest (v1.0.2) 支持：
- 灵活调仓时间（不固定频率）
- 向量化计算（高性能）
- 15+ 性能指标（夏普、索提诺、卡玛、信息比率等）
- 8+ 可视化图表（净值曲线、月度热力图、持仓热力图、换手率分析等）
- 基准对比分析
- 交易成本与滑点模拟

## 版本

- v2.0.0：GeneralBacktest 内嵌迁移，移除旧回测引擎
- v1.2.0：引入 GeneralBacktest 适配层
- v1.1.0：修复因子列检测
- v1.0.0：初始版本
