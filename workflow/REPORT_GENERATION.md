# Workflow 实验报告生成功能

## 📊 功能概述

Workflow 内置了实验报告自动生成功能，可以将实验的配置、指标、结果自动整理成易读的报告。

## 🎯 核心特性

### 1. 自动报告生成

在实验结束后自动生成报告，无需手动编写代码：

```python
from quantclassic.workflow import R

# 运行实验
with R.start(experiment_name="lstm_alpha"):
    R.save_objects(data_config=config, lstm_config=model_config)
    R.log_metrics(sharpe_ratio=2.5, ic_mean=0.08)
    R.save_objects(predictions=pred_df)

# 自动生成报告
recorders = R.list_recorders("lstm_alpha")
recorder_id = list(recorders.keys())[0]

report = R.generate_report(
    experiment_name="lstm_alpha",
    recorder_id=recorder_id,
    report_type="summary"  # 或 "detailed"
)

print(report)
```

### 2. 报告类型

#### 摘要报告 (summary)

简洁版本，包含关键指标：

```
================================================================================
实验摘要报告
================================================================================

【实验信息】
  实验名称: lstm_alpha_20250124
  Recorder: main_run (rec_20250124_143052)
  创建时间: 2025-01-24 14:30:52
  状态: FINISHED

【模型配置】
  模型类型: LSTM
  特征维度: 157
  隐藏层: 64 x 2层
  训练轮数: 20 (学习率: 0.001)

【训练结果】
  窗口数量: 10
  平均训练损失: 0.004523
  平均验证损失: 0.005124

【因子效果】
  IC均值: 0.0823
  ICIR: 2.34
  IC胜率: 68.50%
  显著性: 显著 (t=4.23)

【回测表现】
  年化收益: 24.50%
  夏普比率: 2.4521
  最大回撤: -12.30%
  卡玛比率: 1.9919

【保存路径】
  output/experiments/lstm_alpha_20250124/rec_20250124_143052/

================================================================================
```

#### 详细报告 (detailed)

完整版本，包含所有配置和指标：

- 完整数据配置（文件路径、窗口设置、批次大小等）
- 完整模型配置（所有超参数）
- 完整训练结果（含标准差、最佳epoch等）
- 完整因子分析（IC统计、显著性检验）
- 完整回测配置（调仓频率、交易成本等）
- 完整回测指标（所有绩效指标）
- 保存的对象列表
- 文件结构说明

## 📝 使用示例

### 基础用法

```python
from quantclassic.workflow import R

# 步骤1: 运行实验并保存
with R.start(experiment_name="my_experiment"):
    # 保存配置
    R.save_objects(
        data_config=data_config,
        model_config=model_config,
        backtest_config=backtest_config
    )
    
    # 记录指标
    R.log_params(
        data_config=data_config.__dict__,
        model_config=model_config.__dict__,
        backtest_config=backtest_config.__dict__
    )
    R.log_metrics(**training_metrics)
    R.log_metrics(**ic_stats)
    R.log_metrics(**backtest_metrics)
    
    # 保存数据
    R.save_objects(
        predictions=predictions,
        portfolios=portfolios
    )

# 步骤2: 生成报告
recorders = R.list_recorders("my_experiment")
recorder_id = list(recorders.keys())[0]

# 生成摘要报告
summary = R.generate_report(
    experiment_name="my_experiment",
    recorder_id=recorder_id,
    report_type="summary"
)
print(summary)

# 生成详细报告
detailed = R.generate_report(
    experiment_name="my_experiment",
    recorder_id=recorder_id,
    report_type="detailed",
    save_path="my_detailed_report.txt"  # 自定义保存路径
)
```

### 批量生成报告

```python
# 为某个实验的所有runs生成报告
experiment_name = "lstm_alpha"
recorders = R.list_recorders(experiment_name)

for recorder_id in recorders.keys():
    report = R.generate_report(
        experiment_name=experiment_name,
        recorder_id=recorder_id,
        report_type="summary"
    )
    
    # 提取关键指标
    if "夏普比率: 2." in report:  # 简单筛选
        print(f"发现高夏普run: {recorder_id}")
        print(report)
```

### 对比实验

```python
# 生成多个实验的对比报告
experiments = ["lstm_v1", "lstm_v2", "gru_v1"]

for exp_name in experiments:
    recorders = R.list_recorders(exp_name)
    for rec_id in recorders.keys():
        summary = R.generate_report(exp_name, rec_id, "summary")
        
        # 解析关键指标（可以用正则表达式更精确地提取）
        print(f"\n{'='*60}")
        print(f"实验: {exp_name}")
        print(summary)
```

## 🔧 高级配置

### 自定义报告内容

报告内容会自动从 `recorder.get_params()` 和 `recorder.get_metrics()` 提取：

```python
# 确保保存了必要的参数
R.log_params(
    data_config=data_config.__dict__,   # 数据配置
    lstm_config=lstm_config.__dict__,   # 模型配置
    backtest_config=backtest_config.__dict__  # 回测配置
)

# 确保记录了必要的指标
R.log_metrics(
    # 训练指标
    n_windows=10,
    avg_train_loss=0.005,
    avg_val_loss=0.006,
    
    # IC指标
    ic_mean=0.08,
    icir=2.3,
    ic_win_rate=0.65,
    t_stat=4.2,
    p_value=0.001,
    
    # 回测指标
    annual_return=0.25,
    sharpe_ratio=2.5,
    max_drawdown=-0.12,
    calmar_ratio=2.0,
    win_rate=0.68
)
```

### 报告保存位置

默认情况下，报告保存在 recorder 目录下：

```
output/experiments/
└── lstm_alpha_20250124/
    └── rec_20250124_143052/
        ├── meta.json
        ├── recorder.log
        ├── EXPERIMENT_REPORT.txt       # 摘要报告（自动生成）
        ├── DETAILED_REPORT.txt         # 详细报告（可选）
        └── artifacts/
            ├── data_config.pkl
            ├── lstm_config.pkl
            └── predictions.pkl
```

可以通过 `save_path` 参数自定义：

```python
R.generate_report(
    experiment_name="my_exp",
    recorder_id="rec_xxx",
    report_type="detailed",
    save_path="/custom/path/my_report.txt"
)
```

## 📋 报告字段说明

### 参数字段（从 log_params 提取）

| 配置组 | 字段 | 说明 |
|--------|------|------|
| data_config | data_file | 数据文件路径 |
| | rolling_window_size | 滚动窗口大小 |
| | rolling_step | 滚动步长 |
| | window_size | 序列窗口大小 |
| | batch_size | 批次大小 |
| lstm_config | d_feat | 特征维度 |
| | hidden_size | 隐藏单元数 |
| | num_layers | 网络层数 |
| | dropout | Dropout比率 |
| | n_epochs | 训练轮数 |
| | learning_rate | 学习率 |
| backtest_config | rebalance_freq | 调仓频率 |
| | n_groups | 分组数量 |
| | long_ratio | 多头比例 |
| | short_ratio | 空头比例 |
| | commission_rate | 佣金率 |

### 指标字段（从 log_metrics 提取）

| 类别 | 字段 | 说明 |
|------|------|------|
| 训练结果 | n_windows | 窗口数量 |
| | avg_train_loss | 平均训练损失 |
| | avg_val_loss | 平均验证损失 |
| | std_train_loss | 训练损失标准差 |
| | std_val_loss | 验证损失标准差 |
| | avg_best_epoch | 平均最佳epoch |
| IC分析 | ic_mean | IC均值 |
| | ic_std | IC标准差 |
| | icir | ICIR |
| | ic_win_rate | IC胜率 |
| | t_stat | t统计量 |
| | p_value | p值 |
| 回测指标 | annual_return | 年化收益 |
| | annual_volatility | 年化波动 |
| | sharpe_ratio | 夏普比率 |
| | max_drawdown | 最大回撤 |
| | calmar_ratio | 卡玛比率 |
| | win_rate | 胜率 |

## 💡 最佳实践

### 1. 统一命名规范

使用一致的字段名称：

```python
# ✅ 推荐：使用标准字段名
R.log_metrics(
    ic_mean=0.08,
    icir=2.3,
    sharpe_ratio=2.5
)

# ❌ 避免：使用非标准字段名
R.log_metrics(
    IC=0.08,  # 应该用 ic_mean
    IR=2.3,   # 应该用 icir
    SR=2.5    # 应该用 sharpe_ratio
)
```

### 2. 完整记录配置

确保保存所有配置对象的 `__dict__`：

```python
with R.start(experiment_name="my_exp"):
    # ✅ 保存配置对象
    R.save_objects(
        data_config=data_config,
        model_config=model_config
    )
    
    # ✅ 记录配置参数（用于报告生成）
    R.log_params(
        data_config=data_config.__dict__,
        model_config=model_config.__dict__
    )
```

### 3. 及时生成报告

在实验完成后立即生成报告：

```python
with R.start(experiment_name="my_exp") as recorder:
    # ... 实验代码 ...
    pass

# 立即生成报告
recorders = R.list_recorders("my_exp")
current_id = list(recorders.keys())[0]
R.generate_report("my_exp", current_id, "summary")
```

### 4. 版本控制

将报告纳入版本控制：

```bash
# 将重要实验的报告提交到Git
git add output/experiments/lstm_baseline_v1/*/EXPERIMENT_REPORT.txt
git commit -m "Add baseline experiment report"
```

## 🚀 未来扩展

计划支持的功能：

1. **HTML报告** - 生成交互式HTML报告
2. **图表嵌入** - 将性能图表嵌入报告
3. **对比报告** - 自动生成多实验对比报告
4. **邮件通知** - 实验完成后自动发送报告邮件
5. **报告模板** - 支持自定义报告模板

## 📚 相关文档

- [Workflow 使用指南](./USAGE_EXAMPLES.md)
- [实验管理最佳实践](./README.md)
- [配置管理](./workflow_config.py)
