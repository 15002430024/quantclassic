# QuantClassic Workflow使用示例

## 目录
1. [快速开始](#快速开始)
2. [基本使用](#基本使用)
3. [高级功能](#高级功能)
4. [实战案例](#实战案例)

---

## 快速开始

### 最简单的使用方式

```python
from quantclassic.workflow import R

# 使用上下文管理器自动管理实验生命周期
with R.start(experiment_name="my_first_exp"):
    # 记录参数
    R.log_params(learning_rate=0.001, batch_size=32)
    
    # 记录指标
    R.log_metrics(epoch=1, loss=0.5, accuracy=0.85)
    
    # 保存对象
    R.save_objects(model=my_model, config=config_dict)
```

### 为什么使用Workflow？

- ✅ **自动追踪**: 无需手动管理日志文件
- ✅ **版本管理**: 每次运行自动生成唯一ID
- ✅ **可复现**: 参数、指标、模型全部保存
- ✅ **对比分析**: 轻松比较不同实验结果

---

## 基本使用

### 1. 记录训练过程

```python
from quantclassic.workflow import R
import torch
import torch.nn as nn

# 定义模型
model = nn.LSTM(input_size=10, hidden_size=64, num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 开始实验
with R.start(experiment_name="lstm_training", 
             model_type="LSTM",
             hidden_size=64,
             num_layers=2):
    
    # 训练循环
    for epoch in range(10):
        # ... 训练代码 ...
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)
        
        # 记录每个epoch的指标
        R.log_metrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss
        )
    
    # 保存最终模型
    R.save_objects(
        model=model.state_dict(),
        optimizer=optimizer.state_dict()
    )
```

### 2. 超参数调优

```python
from quantclassic.workflow import R

# 定义超参数搜索空间
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'hidden_size': [32, 64, 128],
    'dropout': [0.1, 0.3, 0.5]
}

# 遍历所有组合
for lr in param_grid['learning_rate']:
    for hidden in param_grid['hidden_size']:
        for dropout in param_grid['dropout']:
            
            # 每组参数启动一个新的recorder
            with R.start(experiment_name="hyperparameter_search",
                        learning_rate=lr,
                        hidden_size=hidden,
                        dropout=dropout):
                
                # 训练模型
                model = build_model(hidden, dropout)
                train_model(model, lr)
                
                # 记录最终性能
                val_ic = evaluate(model, val_data)
                R.log_metrics(val_ic=val_ic)
                
                # 如果是最佳模型，保存
                if val_ic > best_ic:
                    R.save_objects(best_model=model)
```

### 3. 查看实验结果

```python
from quantclassic.workflow import R

# 列出所有实验
experiments = R.list_experiments()
print("所有实验:")
for name, info in experiments.items():
    print(f"  {name}: {info['recorder_count']} runs")

# 查看某个实验的所有runs
recorders = R.list_recorders("hyperparameter_search")
for rec_id, rec_info in recorders.items():
    params = rec_info.get('params', {})
    print(f"Run {rec_id}:")
    print(f"  LR={params.get('learning_rate')}, "
          f"Hidden={params.get('hidden_size')}")

# 搜索最佳run
best_runs = R.search_recorders(
    experiment_name="hyperparameter_search",
    status="FINISHED"
)

# 找出IC最高的run
best_ic = -float('inf')
best_recorder = None

for run in best_runs:
    recorder = R.get_recorder(
        experiment_name="hyperparameter_search",
        recorder_id=run['recorder_id']
    )
    metrics = recorder.get_metrics()
    if metrics and 'val_ic' in metrics:
        ic = metrics['val_ic'][-1][1]  # 获取最后一个IC值
        if ic > best_ic:
            best_ic = ic
            best_recorder = recorder

print(f"最佳IC: {best_ic}")
print(f"最佳参数: {best_recorder.params}")
```

### 4. 加载已保存的对象

```python
from quantclassic.workflow import R

# 加载之前保存的模型
model_state = R.load_object(
    experiment_name="lstm_training",
    recorder_id="rec_20240115_120000_abc123",
    object_name="model"
)

# 恢复模型
model = build_model()
model.load_state_dict(model_state)

# 继续训练或推理
predictions = model(test_data)
```

---

## 高级功能

### 1. 恢复中断的训练

```python
from quantclassic.workflow import R

# 恢复之前的recorder
with R.start(experiment_name="long_training",
            recorder_name="my_training",
            resume=True):
    
    # 加载checkpoint
    checkpoint = R.current_recorder.load_object("checkpoint")
    start_epoch = checkpoint['epoch']
    
    # 从中断处继续
    for epoch in range(start_epoch, total_epochs):
        train_loss = train_one_epoch(model, train_loader)
        R.log_metrics(epoch=epoch, loss=train_loss)
        
        # 定期保存checkpoint
        if epoch % 10 == 0:
            R.save_objects(
                checkpoint={
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
            )
```

### 2. 嵌套实验管理

```python
from quantclassic.workflow import R

# 主实验: 测试不同模型架构
model_types = ['LSTM', 'GRU', 'Transformer']

for model_type in model_types:
    with R.start(experiment_name=f"model_comparison_{model_type}",
                model_type=model_type):
        
        # 子实验: 每个架构的超参数调优
        for lr in [0.001, 0.01]:
            for hidden in [64, 128]:
                
                with R.start(experiment_name=f"tuning_{model_type}",
                            learning_rate=lr,
                            hidden_size=hidden):
                    
                    model = build_model(model_type, hidden)
                    train_model(model, lr)
                    
                    val_ic = evaluate(model, val_data)
                    R.log_metrics(val_ic=val_ic)
```

### 3. 自定义Recorder

```python
from quantclassic.workflow import ExpManager, Recorder

# 不使用全局R，而是直接使用ExpManager
exp_manager = ExpManager(exp_dir="custom_output/experiments")

# 手动创建experiment
experiment = exp_manager.create_experiment("custom_exp")

# 手动创建recorder
recorder_id = exp_manager.start_recorder(
    experiment_name="custom_exp",
    recorder_name="custom_run"
)

# 获取recorder实例
recorder = exp_manager.get_recorder("custom_exp", recorder_id)

# 使用recorder
recorder.log_params(custom_param=123)
recorder.log_metrics(step=1, custom_metric=0.99)

# 手动结束
exp_manager.end_recorder("custom_exp", recorder_id, status="FINISHED")
```

---

## 实战案例

### 案例1: 因子挖掘实验

```python
from quantclassic.workflow import R
from quantclassic.Factorsystem import FactorGenerator
import pandas as pd

# 实验: 测试不同因子组合
factor_combinations = [
    ['momentum', 'reversal'],
    ['momentum', 'volatility'],
    ['value', 'quality']
]

for factors in factor_combinations:
    factor_name = "+".join(factors)
    
    with R.start(experiment_name="factor_mining",
                factor_combination=factors):
        
        # 生成因子
        fg = FactorGenerator()
        factor_data = fg.generate(factors)
        
        # 计算IC
        ic_values = calculate_ic(factor_data, returns)
        
        # 记录结果
        R.log_metrics(
            mean_ic=ic_values.mean(),
            ic_std=ic_values.std(),
            ic_ir=ic_values.mean() / ic_values.std()
        )
        
        # 保存因子数据
        R.save_objects(
            factor_data=factor_data,
            ic_series=ic_values
        )

# 分析最佳因子组合
all_runs = R.list_recorders("factor_mining")
best_ir = -float('inf')

for rec_id, rec_info in all_runs.items():
    recorder = R.get_recorder("factor_mining", rec_id)
    metrics = recorder.get_metrics()
    
    if 'ic_ir' in metrics:
        ir = metrics['ic_ir'][-1][1]
        if ir > best_ir:
            best_ir = ir
            best_factors = recorder.params['factor_combination']

print(f"最佳因子组合: {best_factors}, IR={best_ir}")
```

### 案例2: 模型集成实验

```python
from quantclassic.workflow import R
from quantclassic.model import ModelFactory
import numpy as np

# 训练多个模型
models_info = []

for i in range(5):
    with R.start(experiment_name="ensemble_training",
                model_id=i,
                random_seed=i*100):
        
        # 使用不同随机种子训练
        np.random.seed(i * 100)
        
        model = ModelFactory.create_model('LSTM', config)
        train_model(model, train_data)
        
        # 评估
        val_pred = model.predict(val_data)
        val_ic = calculate_ic(val_pred, val_returns)
        
        R.log_metrics(val_ic=val_ic)
        R.save_objects(model=model)
        
        # 记录模型信息用于后续集成
        models_info.append({
            'experiment': 'ensemble_training',
            'recorder_id': R.current_recorder.recorder_id,
            'ic': val_ic
        })

# 集成所有模型
with R.start(experiment_name="ensemble_prediction"):
    predictions = []
    
    for info in models_info:
        # 加载每个模型
        model = R.load_object(
            experiment_name=info['experiment'],
            recorder_id=info['recorder_id'],
            object_name='model'
        )
        
        # 预测
        pred = model.predict(test_data)
        predictions.append(pred)
    
    # 平均集成
    ensemble_pred = np.mean(predictions, axis=0)
    
    # 评估集成效果
    test_ic = calculate_ic(ensemble_pred, test_returns)
    R.log_metrics(
        test_ic=test_ic,
        num_models=len(predictions)
    )
    
    R.save_objects(
        ensemble_prediction=ensemble_pred,
        individual_predictions=predictions
    )
```

### 案例3: A/B测试

```python
from quantclassic.workflow import R
import datetime

# 策略A: 基础版本
with R.start(experiment_name="strategy_ab_test",
            strategy_version="A",
            test_date=str(datetime.date.today())):
    
    # 运行策略A
    positions_a = strategy_a.generate_positions(data)
    returns_a = backtest(positions_a, price_data)
    
    R.log_metrics(
        total_return=returns_a.sum(),
        sharpe_ratio=calculate_sharpe(returns_a),
        max_drawdown=calculate_mdd(returns_a)
    )
    
    R.save_objects(
        positions=positions_a,
        returns=returns_a
    )

# 策略B: 改进版本
with R.start(experiment_name="strategy_ab_test",
            strategy_version="B",
            test_date=str(datetime.date.today())):
    
    # 运行策略B
    positions_b = strategy_b.generate_positions(data)
    returns_b = backtest(positions_b, price_data)
    
    R.log_metrics(
        total_return=returns_b.sum(),
        sharpe_ratio=calculate_sharpe(returns_b),
        max_drawdown=calculate_mdd(returns_b)
    )
    
    R.save_objects(
        positions=positions_b,
        returns=returns_b
    )

# 对比分析
recorders = R.list_recorders("strategy_ab_test")

print("A/B测试结果对比:")
print("-" * 50)

for rec_id, rec_info in recorders.items():
    version = rec_info['params']['strategy_version']
    recorder = R.get_recorder("strategy_ab_test", rec_id)
    metrics = recorder.get_metrics()
    
    print(f"\n策略 {version}:")
    print(f"  总收益: {metrics['total_return'][-1][1]:.2%}")
    print(f"  夏普比率: {metrics['sharpe_ratio'][-1][1]:.3f}")
    print(f"  最大回撤: {metrics['max_drawdown'][-1][1]:.2%}")
```

---

## 最佳实践

### 1. 命名规范

```python
# ✅ 好的命名
with R.start(experiment_name="lstm_price_prediction_v2"):
    ...

# ❌ 避免的命名
with R.start(experiment_name="test"):
    ...
```

### 2. 参数记录

```python
# ✅ 记录所有重要参数
with R.start(experiment_name="training",
            learning_rate=0.001,
            batch_size=256,
            hidden_size=64,
            dropout=0.3,
            optimizer="Adam",
            data_range="2020-2023"):
    ...

# ❌ 参数记录不完整
with R.start(experiment_name="training"):
    R.log_params(lr=0.001)  # 其他参数缺失
```

### 3. 指标记录

```python
# ✅ 使用step参数保持顺序
for epoch in range(100):
    R.log_metrics(epoch=epoch, loss=loss, acc=acc)

# ✅ 记录多维度指标
R.log_metrics(
    train_loss=0.1,
    val_loss=0.15,
    test_loss=0.12,
    train_ic=0.05,
    val_ic=0.04
)
```

### 4. 对象保存

```python
# ✅ 保存完整的可复现信息
R.save_objects(
    model=model.state_dict(),
    optimizer=optimizer.state_dict(),
    config=config_dict,
    scaler=data_scaler,
    feature_names=feature_columns
)

# ⚠️ 只保存模型可能不够
R.save_objects(model=model)
```

---

## 故障排除

### 问题1: "请先使用 R.start() 启动recorder"

```python
# ❌ 错误: 在start之外使用
R.log_metrics(loss=0.5)  # 报错!

# ✅ 正确: 在start内部使用
with R.start(experiment_name="test"):
    R.log_metrics(loss=0.5)  # OK
```

### 问题2: Recorder目录找不到

```python
# 检查实验目录
experiments = R.list_experiments()
print(experiments)

# 检查recorder
recorders = R.list_recorders("my_experiment")
print(recorders)
```

### 问题3: 加载对象失败

```python
# 确保使用正确的experiment_name和recorder_id
try:
    obj = R.load_object(
        experiment_name="training",
        recorder_id="rec_20240115_120000_abc123",
        object_name="model"
    )
except FileNotFoundError:
    print("对象不存在，检查参数是否正确")
```

---

## 与Qlib的对比

| 功能 | Qlib | QuantClassic |
|------|------|--------------|
| 全局接口 | `R` | `R` ✅ |
| 上下文管理器 | `R.start()` | `R.start()` ✅ |
| 参数记录 | `R.log_params()` | `R.log_params()` ✅ |
| 指标记录 | `R.log_metrics()` | `R.log_metrics()` ✅ |
| 对象保存 | `R.save_objects()` | `R.save_objects()` ✅ |
| 实验搜索 | `R.search_recorders()` | `R.search_recorders()` ✅ |
| 后端存储 | MLflow/Custom | File-based ✅ |

---

## 总结

QuantClassic Workflow提供了与Qlib兼容的实验管理接口，让你可以:

1. 🎯 **轻松追踪**: 自动记录所有实验参数和结果
2. 📊 **对比分析**: 方便比较不同模型和策略
3. 🔄 **可复现**: 完整保存实验状态，随时可恢复
4. 🚀 **提高效率**: 专注于模型开发，而不是日志管理

开始使用吧！ 🎉
