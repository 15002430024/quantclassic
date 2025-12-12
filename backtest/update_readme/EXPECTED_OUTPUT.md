# 因子回测系统 - 预期输出示例

## 1. 控制台输出示例

### 完整回测流程输出

```
================================================================================
开始因子回测
================================================================================

步骤1: 生成因子
生成因子...
100%|██████████████████████████████████████| 489/489 [00:15<00:00, 31.23it/s]
  生成样本数: 125,432
  生成因子数: 125,432 条记录, 16 个因子维度
  因子统计信息:
    factor_0: 均值=0.0023, 标准差=1.2456, 最小值=-4.5623, 最大值=5.2341, 缺失值=0
    factor_1: 均值=-0.0012, 标准差=1.1823, 最小值=-3.9876, 最大值=4.8765, 缺失值=0
    ...
  因子生成完成: 125432 条记录, 16 个因子维度

步骤2: 处理因子
  处理因子: factor_0
  处理因子: factor_1
  ...
  因子处理完成

步骤3: IC分析
  IC计算完成: 245 个交易日

IC统计指标:
  IC均值: 0.0523
  Rank IC均值: 0.0487
  IC标准差: 0.1234
  ICIR: 0.4237
  Rank ICIR: 0.3946
  IC胜率: 58.37%
  绝对IC均值: 0.0892
  t统计量: 6.6342

步骤4: 构建投资组合
  分组完成: 10 组
  投资组合构建完成

步骤5: 绩效评估

评估 long 组合:
  累计收益: 45.23%
  年化收益: 18.34%
  年化波动率: 15.67%
  夏普比率: 0.9782
  最大回撤: -12.34%
  卡玛比率: 1.4865
  胜率: 54.29%

评估 short 组合:
  累计收益: 22.56%
  年化收益: 9.12%
  年化波动率: 14.23%
  夏普比率: 0.4287
  最大回撤: -18.45%
  卡玛比率: 0.4945
  胜率: 52.04%

评估 long_short 组合:
  累计收益: 67.89%
  年化收益: 25.43%
  年化波动率: 18.23%
  夏普比率: 1.2304
  最大回撤: -15.67%
  卡玛比率: 1.6221
  胜率: 56.12%

步骤6: 生成可视化图表
  报告图表已保存到: output/backtest/plots

步骤7: 保存结果
  因子已保存: output/backtest/factors.csv
  IC分析已保存: output/backtest/ic_analysis.csv
  long组合已保存: output/backtest/portfolio_long.csv
  short组合已保存: output/backtest/portfolio_short.csv
  long_short组合已保存: output/backtest/portfolio_long_short.csv
  绩效指标已保存: output/backtest/performance_metrics.xlsx

================================================================================
回测完成!
================================================================================
```

---

## 2. 文件输出示例

### 目录结构
```
output/backtest/
├── factors.csv                      # 因子数据 (125,432行)
├── ic_analysis.csv                  # IC分析 (245行, 每个交易日)
├── portfolio_long.csv               # 多头组合 (245行)
├── portfolio_short.csv              # 空头组合 (245行)
├── portfolio_long_short.csv         # 多空组合 (245行)
├── performance_metrics.xlsx         # 绩效指标汇总
└── plots/                           # 图表目录
    ├── cumulative_returns.png
    ├── drawdown.png
    ├── ic_series.png
    ├── ic_distribution.png
    ├── group_returns.png
    └── long_short_performance.png
```

### factors.csv 示例 (前5行)
```csv
ts_code,trade_date,factor_0,factor_1,...,factor_15,factor_raw_winsorized,factor_raw_filled,factor_raw_std
000001.SZ,2023-01-03,0.234,-0.567,...,1.234,0.234,0.234,0.456
000001.SZ,2023-01-04,0.189,-0.423,...,1.156,0.189,0.189,0.378
000002.SZ,2023-01-03,-0.345,0.678,...,-0.987,-0.345,-0.345,-0.689
000002.SZ,2023-01-04,-0.298,0.534,...,-0.876,-0.298,-0.298,-0.596
...
```

### ic_analysis.csv 示例 (前5行)
```csv
trade_date,ic,rank_ic,abs_ic,p_value,n_samples,significant,cum_ic,cum_rank_ic
2023-01-03,0.0623,0.0587,0.0623,0.0234,3456,True,0.0623,0.0587
2023-01-04,0.0412,0.0398,0.0412,0.0567,3478,True,0.1035,0.0985
2023-01-05,-0.0134,-0.0145,0.0134,0.4523,3502,False,0.0901,0.0840
2023-01-06,0.0789,0.0723,0.0789,0.0012,3489,True,0.1690,0.1563
...
```

### portfolio_long_short.csv 示例 (前5行)
```csv
trade_date,portfolio_return_long,portfolio_return_short,portfolio_return,cum_return,cum_return_long,cum_return_short
2023-01-03,0.0123,-0.0087,0.0210,0.0210,0.0123,0.0087
2023-01-04,0.0089,-0.0045,0.0134,0.0346,0.0213,0.0133
2023-01-05,-0.0034,0.0056,0.0022,0.0369,0.0179,0.0190
2023-01-06,0.0156,-0.0098,0.0254,0.0629,0.0338,0.0293
...
```

### performance_metrics.xlsx 示例

| 指标 | long | short | long_short |
|------|------|-------|------------|
| total_return | 0.4523 | 0.2256 | 0.6789 |
| annual_return | 0.1834 | 0.0912 | 0.2543 |
| annual_volatility | 0.1567 | 0.1423 | 0.1823 |
| sharpe_ratio | 0.9782 | 0.4287 | 1.2304 |
| max_drawdown | -0.1234 | -0.1845 | -0.1567 |
| calmar_ratio | 1.4865 | 0.4945 | 1.6221 |
| sortino_ratio | 1.3456 | 0.5678 | 1.7890 |
| win_rate | 0.5429 | 0.5204 | 0.5612 |
| profit_loss_ratio | 1.2345 | 1.0987 | 1.3456 |
| skewness | 0.1234 | -0.2345 | 0.0567 |
| kurtosis | 2.3456 | 3.4567 | 2.6789 |
| var_95 | -0.0234 | -0.0267 | -0.0289 |
| cvar_95 | -0.0345 | -0.0389 | -0.0412 |

---

## 3. Python返回对象示例

### results 字典结构
```python
results = {
    'raw_factors': pd.DataFrame,  # shape: (125432, 20+)
    
    'processed_factors': pd.DataFrame,  # shape: (125432, 30+)
    # 包含列: ts_code, trade_date, factor_0...factor_15, 
    #        factor_raw_winsorized, factor_raw_filled, factor_raw_std, etc.
    
    'ic_df': pd.DataFrame,  # shape: (245, 8)
    # 列: trade_date, ic, rank_ic, abs_ic, p_value, n_samples, 
    #     significant, cum_ic, cum_rank_ic
    
    'ic_stats': {
        'ic_mean': 0.0523,
        'rank_ic_mean': 0.0487,
        'ic_std': 0.1234,
        'rank_ic_std': 0.1189,
        'icir': 0.4237,
        'rank_icir': 0.3946,
        'ic_win_rate': 0.5837,
        'rank_ic_win_rate': 0.5714,
        'abs_ic_mean': 0.0892,
        'abs_rank_ic_mean': 0.0834,
        'ic_max': 0.1456,
        'ic_min': -0.0987,
        'significant_ratio': 0.6327,
        't_stat': 6.6342,
        'n_periods': 245
    },
    
    'portfolios': {
        'long': pd.DataFrame,  # shape: (245, 4)
        # 列: trade_date, portfolio_return, n_stocks, avg_factor, cum_return
        
        'short': pd.DataFrame,  # shape: (245, 4)
        
        'long_short': pd.DataFrame,  # shape: (245, 6)
        # 列: trade_date, portfolio_return_long, portfolio_return_short,
        #     portfolio_return, cum_return, cum_return_long, cum_return_short
        
        'groups': pd.DataFrame  # shape: (2450, 6)
        # 列: group, return_mean, return_std, stock_count, 
        #     factor_mean, trade_date
    },
    
    'performance_metrics': {
        'long': {
            'total_return': 0.4523,
            'annual_return': 0.1834,
            'mean_return': 0.0007,
            'median_return': 0.0006,
            'volatility': 0.0099,
            'annual_volatility': 0.1567,
            'max_drawdown': -0.1234,
            'max_drawdown_duration': 23,
            'downside_risk': 0.0071,
            'annual_downside_risk': 0.1127,
            'sharpe_ratio': 0.9782,
            'calmar_ratio': 1.4865,
            'sortino_ratio': 1.3456,
            'win_rate': 0.5429,
            'profit_loss_ratio': 1.2345,
            'skewness': 0.1234,
            'kurtosis': 2.3456,
            'var_95': -0.0234,
            'cvar_95': -0.0345
        },
        'short': {...},  # 同样的指标
        'long_short': {...}  # 同样的指标
    }
}
```

---

## 4. 图表输出示例

### 累计收益曲线 (cumulative_returns.png)
- X轴: 时间 (2023-01-03 to 2023-12-29)
- Y轴: 累计收益率 (-0.2 to 0.8)
- 线条: 组合收益 (蓝色实线), 基准收益 (橙色虚线, 如有)
- 特点: 整体上升趋势, 中间有小幅回撤

### 回撤曲线 (drawdown.png)
- X轴: 时间
- Y轴: 回撤幅度 (-0.2 to 0)
- 填充: 红色区域表示回撤
- 标注: 最大回撤点 (-15.67%)

### IC时间序列 (ic_series.png)
- 两个子图:
  1. IC时序: IC和Rank IC随时间波动, 在0附近
  2. 累计IC: 持续上升, 最终达到~40

### IC分布 (ic_distribution.png)
- 直方图: 50个bins
- 垂直线: IC均值 (0.0523, 红色虚线)
- 形状: 近似正态分布, 中心略偏右

### 分组收益 (group_returns.png)
- 柱状图: 10组
- 颜色: 正收益绿色, 负收益红色
- 趋势: 从第1组到第10组递增 (单调性)
- 数值标签: 每个柱顶部显示收益值

### 多空表现 (long_short_performance.png)
- 三条线:
  - 多头组合 (绿色): 上升到~45%
  - 空头组合 (红色): 上升到~23%
  - 多空组合 (蓝色虚线): 上升到~68%

---

## 5. 使用代码示例

```python
from Factorsystem import BacktestConfig, FactorBacktestSystem
import pandas as pd

# 创建配置
config = BacktestConfig(
    output_dir='output/backtest',
    save_plots=True,
    generate_excel=True
)

# 初始化系统
system = FactorBacktestSystem(config)

# 加载模型和数据
model = system.load_model('output/best_model.pth')
df = pd.read_parquet('output/train_data_final_01.parquet')

# 运行回测
results = system.run_backtest(df)

# 访问结果
print("IC统计:")
print(results['ic_stats'])

print("\n多空组合绩效:")
print(results['performance_metrics']['long_short'])

# 查看因子数据
print("\n因子数据样例:")
print(results['processed_factors'].head())

# 查看IC时序
print("\nIC时间序列:")
print(results['ic_df'].head())

# 查看组合收益
print("\n多空组合收益:")
print(results['portfolios']['long_short'].head())
```

---

## 6. 常见场景的预期输出

### 场景1: 因子表现优秀
```
IC均值: 0.0852
ICIR: 0.6734
夏普比率: 1.5678
最大回撤: -8.45%
```

### 场景2: 因子表现一般
```
IC均值: 0.0234
ICIR: 0.1892
夏普比率: 0.4567
最大回撤: -18.34%
```

### 场景3: 因子无效
```
IC均值: 0.0012
ICIR: 0.0145
夏普比率: -0.1234
最大回撤: -35.67%
```

### 场景4: 因子反向
```
IC均值: -0.0456
ICIR: -0.3245
提示: 考虑反向使用该因子
```

---

## 7. 错误处理示例

### 模型文件不存在
```
ERROR: 模型文件不存在: output/best_model.pth
请检查路径是否正确
```

### 数据缺少必要列
```
ERROR: 未找到股票代码列，请确保数据中包含ts_code列
候选列: ['order_book_id', 'code', 'symbol', 'stock_code', 'ticker']
```

### 样本数量不足
```
WARNING: 日期 2023-01-03 样本数太少 (5 < 10)，跳过IC计算
```

---

以上是系统的完整预期输出示例，涵盖了控制台输出、文件输出、返回对象、图表以及不同场景下的表现。
