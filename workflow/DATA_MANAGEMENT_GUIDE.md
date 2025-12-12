# Workflow 实验数据完整保存指南

## 📦 完整数据保存流程

Workflow 现在支持保存整个实验流程的所有数据，包括：
- 配置文件
- 训练数据（模型输入）
- 预测结果
- 因子数据
- 回测数据
- 组合数据
- 绩效指标
- 可视化图表

## 🎯 保存内容清单

### 1. 配置文件 (`.pkl`)

| 文件名 | 内容 | 说明 |
|--------|------|------|
| `data_config.pkl` | DataConfig 对象 | 数据加载和预处理配置 |
| `lstm_config.pkl` | LSTMConfig 对象 | LSTM 模型超参数配置 |
| `backtest_config.pkl` | BacktestConfig 对象 | 回测策略配置 |

### 2. 训练相关数据 (`.pkl`)

| 文件名 | 内容 | 说明 |
|--------|------|------|
| `feature_cols.pkl` | List[str] | 特征列名列表 |

**注意**：原始训练数据（`train_data_processed`）由于体积较大，需要根据实际情况决定是否保存。

### 3. 预测数据 (`.pkl`)

| 文件名 | 内容 | 说明 |
|--------|------|------|
| `rolling_predictions.pkl` | DataFrame | 滚动窗口预测结果 |
| `alpha_factors.pkl` | DataFrame | 生成的 Alpha 因子值 |

**数据结构**：
```python
rolling_predictions:
    - order_book_id: 股票代码
    - trade_date: 交易日期
    - y_pred: 预测值
    - window_idx: 窗口编号
    
alpha_factors:
    - order_book_id: 股票代码
    - trade_date: 交易日期
    - factor_value: 因子值
```

### 4. 回测数据 (`.pkl`)

| 文件名 | 内容 | 说明 |
|--------|------|------|
| `backtest_df.pkl` | DataFrame | 回测输入数据（合并预测和收益） |
| `processed_df.pkl` | DataFrame | 因子处理后数据（标准化、去极值） |
| `ic_df.pkl` | DataFrame | IC 分析结果 |

**数据结构**：
```python
backtest_df:
    - order_book_id: 股票代码
    - trade_date: 交易日期
    - factor_value: 因子值
    - y_processed: 未来收益
    
processed_df:
    - order_book_id: 股票代码
    - trade_date: 交易日期
    - factor_value: 原始因子
    - factor_value_winsorized: 去极值后
    - factor_value_std: 标准化后
    - y_processed: 未来收益
    
ic_df:
    - trade_date: 交易日期
    - ic: IC 值
    - rank_ic: Rank IC 值
```

### 5. 组合数据 (`.pkl`)

| 文件名 | 内容 | 说明 |
|--------|------|------|
| `long_portfolio.pkl` | DataFrame | 多头组合持仓和收益 |
| `short_portfolio.pkl` | DataFrame | 空头组合持仓和收益 |
| `long_short_portfolio.pkl` | DataFrame | 多空组合持仓和收益 |

**数据结构**：
```python
long_short_portfolio:
    - trade_date: 交易日期
    - portfolio_return: 组合收益率
    - long_weight: 多头权重
    - short_weight: 空头权重
    - turnover: 换手率
```

### 6. 绩效指标 (`.pkl`)

| 文件名 | 内容 | 说明 |
|--------|------|------|
| `all_metrics.pkl` | Dict | 所有组合的绩效指标 |
| `ic_stats.pkl` | Dict | IC 统计指标 |

**数据结构**：
```python
all_metrics = {
    'long': {...},
    'short': {...},
    'long_short': {
        'total_return': 累计收益,
        'annual_return': 年化收益,
        'annual_volatility': 年化波动,
        'sharpe_ratio': 夏普比率,
        'max_drawdown': 最大回撤,
        'calmar_ratio': 卡玛比率,
        'sortino_ratio': 索提诺比率,
        'win_rate': 胜率,
        'profit_loss_ratio': 盈亏比,
        ...
    }
}

ic_stats = {
    'ic_mean': IC均值,
    'ic_std': IC标准差,
    'icir': ICIR,
    'ic_win_rate': IC胜率,
    't_stat': t统计量,
    'p_value': p值,
    'significant_ratio': 显著比例,
    ...
}
```

### 7. 可视化图表 (`.png`)

保存在 `artifacts/plots/` 目录下：

| 文件名 | 内容 | 说明 |
|--------|------|------|
| `cumulative_returns.png` | 累计收益曲线 | 含基准对比 |
| `drawdown.png` | 回撤曲线 | 最大回撤可视化 |
| `ic_series.png` | IC 时间序列 | 含移动平均 |
| `ic_distribution.png` | IC 分布直方图 | 含统计指标 |
| `group_returns.png` | 分组收益柱状图 | 因子单调性检验 |
| `long_short_performance.png` | 多空表现对比 | 多头 vs 空头 |
| `comprehensive_analysis.png` | 综合分析图 | 6 合 1 综合视图 |

## 💾 完整保存示例

```python
from quantclassic.workflow import R

with R.start(experiment_name="my_lstm_experiment"):
    # 1. 保存配置
    R.save_objects(
        data_config=data_config,
        lstm_config=lstm_config,
        backtest_config=backtest_config
    )
    
    # 2. 记录指标
    R.log_params(
        data_config=data_config.__dict__,
        lstm_config=lstm_config.__dict__,
        backtest_config=backtest_config.__dict__
    )
    R.log_metrics(**training_metrics)
    R.log_metrics(**ic_stats)
    R.log_metrics(**backtest_metrics)
    
    # 3. 保存中间数据
    R.save_objects(
        # 训练数据
        feature_cols=feature_cols,
        
        # 预测数据
        rolling_predictions=rolling_predictions,
        alpha_factors=alpha_factors,
        
        # 回测数据
        backtest_df=backtest_df,
        processed_df=processed_df,
        ic_df=ic_df,
        
        # 组合数据
        long_portfolio=portfolios['long'],
        short_portfolio=portfolios['short'],
        long_short_portfolio=portfolios['long_short'],
        
        # 绩效指标
        all_metrics=all_metrics,
        ic_stats=ic_stats
    )
    
    # 4. 保存图表
    if R.current_recorder:
        import shutil
        from pathlib import Path
        
        artifacts_dir = Path(R.current_recorder.recorder_dir) / "artifacts" / "plots"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制所有图表
        for plot_file in Path('output/plots').glob('*.png'):
            shutil.copy2(plot_file, artifacts_dir)
```

## 🔍 加载已保存的数据

### 方法 1: 通过 Workflow 加载

```python
from quantclassic.workflow import R

experiment_name = "my_lstm_experiment"
recorder_id = "rec_20250124_143052_123456"

# 加载配置
data_config = R.load_object(experiment_name, recorder_id, 'data_config')
lstm_config = R.load_object(experiment_name, recorder_id, 'lstm_config')

# 加载预测数据
predictions = R.load_object(experiment_name, recorder_id, 'rolling_predictions')
factors = R.load_object(experiment_name, recorder_id, 'alpha_factors')

# 加载回测数据
backtest_df = R.load_object(experiment_name, recorder_id, 'backtest_df')
processed_df = R.load_object(experiment_name, recorder_id, 'processed_df')
ic_df = R.load_object(experiment_name, recorder_id, 'ic_df')

# 加载组合数据
long_short = R.load_object(experiment_name, recorder_id, 'long_short_portfolio')

# 加载指标
all_metrics = R.load_object(experiment_name, recorder_id, 'all_metrics')
ic_stats = R.load_object(experiment_name, recorder_id, 'ic_stats')
```

### 方法 2: 直接从文件加载

```python
import pickle
from pathlib import Path

# 实验目录
exp_dir = Path("output/experiments/exp_my_lstm_experiment_20250124_143052")
rec_dir = exp_dir / "rec_20250124_143052_123456"
artifacts_dir = rec_dir / "artifacts"

# 加载数据
with open(artifacts_dir / "rolling_predictions.pkl", "rb") as f:
    predictions = pickle.load(f)

with open(artifacts_dir / "alpha_factors.pkl", "rb") as f:
    factors = pickle.load(f)

# 加载图表
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open(artifacts_dir / "plots" / "comprehensive_analysis.png")
plt.figure(figsize=(20, 12))
plt.imshow(img)
plt.axis('off')
plt.show()
```

### 方法 3: 批量加载所有数据

```python
from quantclassic.workflow import R

def load_experiment_data(experiment_name, recorder_id):
    """加载实验的所有数据"""
    
    # 获取 recorder
    recorder = R.exp_manager.get_recorder(
        experiment_name=experiment_name,
        recorder_id=recorder_id
    )
    
    # 获取所有保存的对象名称
    artifacts = recorder.list_artifacts()
    
    # 批量加载
    data = {}
    for artifact_name in artifacts:
        data[artifact_name] = recorder.load_object(artifact_name)
    
    return data

# 使用示例
all_data = load_experiment_data("my_lstm_experiment", "rec_xxx")

# 访问数据
predictions = all_data['rolling_predictions']
factors = all_data['alpha_factors']
metrics = all_data['all_metrics']
```

## 📊 数据分析示例

### 1. 分析 IC 序列

```python
import pandas as pd
import matplotlib.pyplot as plt

# 加载 IC 数据
ic_df = R.load_object(experiment_name, recorder_id, 'ic_df')
ic_stats = R.load_object(experiment_name, recorder_id, 'ic_stats')

# 计算 IC 统计
print(f"IC 均值: {ic_stats['ic_mean']:.4f}")
print(f"ICIR: {ic_stats['icir']:.4f}")
print(f"IC 胜率: {ic_stats['ic_win_rate']:.2%}")
print(f"t 统计量: {ic_stats['t_stat']:.4f}")

# 绘制 IC 时间序列
plt.figure(figsize=(15, 5))
plt.plot(ic_df['trade_date'], ic_df['ic'], alpha=0.7)
plt.axhline(y=0, color='gray', linestyle='--')
plt.axhline(y=ic_stats['ic_mean'], color='red', linestyle='--', 
            label=f"IC均值={ic_stats['ic_mean']:.4f}")
plt.title("IC Time Series")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 2. 分析组合表现

```python
# 加载组合数据
long_short = R.load_object(experiment_name, recorder_id, 'long_short_portfolio')
metrics = R.load_object(experiment_name, recorder_id, 'all_metrics')

# 计算累计净值
cumulative_return = (1 + long_short['portfolio_return']).cumprod()

# 绘制净值曲线
plt.figure(figsize=(15, 6))
plt.plot(long_short['trade_date'], cumulative_return, linewidth=2)
plt.title(f"多空组合净值曲线 (Sharpe={metrics['long_short']['sharpe_ratio']:.4f})")
plt.xlabel("日期")
plt.ylabel("累计净值")
plt.grid(True, alpha=0.3)
plt.show()

# 打印关键指标
print(f"年化收益: {metrics['long_short']['annual_return']:.2%}")
print(f"年化波动: {metrics['long_short']['annual_volatility']:.2%}")
print(f"夏普比率: {metrics['long_short']['sharpe_ratio']:.4f}")
print(f"最大回撤: {metrics['long_short']['max_drawdown']:.2%}")
print(f"卡玛比率: {metrics['long_short']['calmar_ratio']:.4f}")
```

### 3. 对比多个实验

```python
# 获取所有实验
experiments = R.list_experiments()

# 收集所有实验的指标
results = []
for exp_info in experiments:
    exp_name = exp_info['name']
    recorders = R.list_recorders(exp_name)
    
    for rec_id, rec_info in recorders.items():
        metrics = rec_info.get('metrics', {})
        results.append({
            'experiment': exp_name,
            'recorder': rec_id[:8],
            'sharpe': metrics.get('sharpe_ratio', 0),
            'ic': metrics.get('ic_mean', 0),
            'icir': metrics.get('icir', 0),
            'return': metrics.get('annual_return', 0)
        })

# 创建对比表格
comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values('sharpe', ascending=False)

print("\n实验对比 (按 Sharpe 排序):")
print(comparison_df.to_string(index=False))

# 可视化对比
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Sharpe 对比
axes[0].barh(comparison_df['experiment'], comparison_df['sharpe'])
axes[0].set_xlabel('Sharpe Ratio')
axes[0].set_title('Sharpe Ratio 对比')

# IC 对比
axes[1].barh(comparison_df['experiment'], comparison_df['ic'])
axes[1].set_xlabel('IC Mean')
axes[1].set_title('IC 对比')

# 年化收益对比
axes[2].barh(comparison_df['experiment'], comparison_df['return'])
axes[2].set_xlabel('Annual Return')
axes[2].set_title('年化收益对比')

plt.tight_layout()
plt.show()
```

## 🎯 最佳实践

### 1. 命名规范

建议使用有意义的实验名称：

```python
# ✅ 推荐
experiment_name = "lstm_alpha_dropout03_lr001_20250124"

# ❌ 避免
experiment_name = "test1"
```

### 2. 定期清理

定期清理旧实验以节省磁盘空间：

```python
from quantclassic.workflow import R
import datetime

# 删除 30 天前的实验
cutoff_date = datetime.datetime.now() - datetime.timedelta(days=30)

experiments = R.list_experiments()
for exp_info in experiments:
    exp_date = datetime.datetime.fromisoformat(exp_info['create_time'])
    if exp_date < cutoff_date:
        R.exp_manager.delete_experiment(exp_info['id'])
        print(f"已删除旧实验: {exp_info['name']}")
```

### 3. 备份重要实验

```python
import shutil
from pathlib import Path

# 备份重要实验
exp_dir = Path(f"output/experiments/{exp_id}")
backup_dir = Path(f"backups/{exp_id}")

shutil.copytree(exp_dir, backup_dir)
print(f"实验已备份到: {backup_dir}")
```

## 📚 相关文档

- [Workflow 使用指南](./USAGE_EXAMPLES.md)
- [实验报告生成](./REPORT_GENERATION.md)
- [配置管理](./workflow_config.py)
