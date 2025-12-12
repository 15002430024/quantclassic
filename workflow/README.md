# QuantClassic Workflow - 实验管理系统

参照Qlib设计的实验追踪和管理系统，为量化研究提供完整的实验生命周期管理。

## 📋 目录

- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [架构设计](#架构设计)
- [核心组件](#核心组件)
- [使用指南](#使用指南)
- [API文档](#api文档)

---

## 🎯 核心特性

### 1. 自动化实验追踪
- ✅ 自动记录实验参数、指标、对象
- ✅ 唯一ID生成，避免覆盖
- ✅ 时间戳追踪，完整历史记录

### 2. 统一的全局接口
```python
from quantclassic.workflow import R

with R.start(experiment_name="my_exp"):
    R.log_params(lr=0.001)
    R.log_metrics(loss=0.5)
    R.save_objects(model=model)
```

### 3. 灵活的存储结构
```
output/experiments/
├── experiment_1/
│   ├── metadata.json          # 实验级元数据
│   ├── recorder_1/
│   │   ├── metadata.json      # Recorder元数据
│   │   ├── params.json        # 参数
│   │   ├── metrics.json       # 指标
│   │   └── objects/           # 保存的对象
│   │       ├── model.pkl
│   │       └── config.pkl
│   └── recorder_2/
│       └── ...
└── index.json                 # 全局索引
```

### 4. 强大的搜索和对比
- 按实验名、状态、参数搜索
- 批量加载和对比结果
- 支持嵌套实验管理

---

## 🚀 快速开始

### 安装

Workflow是QuantClassic的一部分，无需单独安装：

```python
from quantclassic.workflow import R
```

### 第一个实验

```python
from quantclassic.workflow import R

# 训练模型并自动记录
with R.start(experiment_name="first_experiment", 
             learning_rate=0.001, 
             batch_size=32):
    
    # 训练循环
    for epoch in range(10):
        loss = train_one_epoch(model, data)
        R.log_metrics(epoch=epoch, loss=loss)
    
    # 保存模型
    R.save_objects(model=model)
```

### 查看结果

```python
# 列出所有实验
experiments = R.list_experiments()
print(experiments)

# 查看特定实验的所有runs
recorders = R.list_recorders("first_experiment")
for rec_id, info in recorders.items():
    print(f"{rec_id}: {info['params']}")
```

### 加载保存的模型

```python
# 加载之前保存的模型
model = R.load_object(
    experiment_name="first_experiment",
    recorder_id="rec_20240115_120000_abc123",
    object_name="model"
)
```

---

## 🏗️ 架构设计

### 三层架构

```
┌─────────────────────────────────────┐
│      QCRecorder (Global R)          │  ← 用户接口层
│  - 上下文管理                         │
│  - 简化API                           │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│        ExpManager                    │  ← 管理层
│  - 实验创建/删除                      │
│  - Recorder生命周期管理               │
│  - 全局索引维护                       │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   Experiment + Recorder              │  ← 核心层
│  - 数据持久化                         │
│  - 参数/指标记录                      │
│  - 对象序列化                         │
└─────────────────────────────────────┘
```

### 设计理念

1. **分层清晰**: 用户接口、管理逻辑、存储实现分离
2. **扩展性强**: 可替换存储后端（当前为文件，未来可支持数据库）
3. **Qlib兼容**: API设计与Qlib保持一致
4. **独立运行**: 不依赖外部服务（MLflow等）

---

## 📦 核心组件

### 1. Recorder - 记录器

单次实验运行的记录器，负责记录参数、指标和对象。

```python
from quantclassic.workflow import Recorder

recorder = Recorder(
    recorder_id="rec_001",
    experiment_name="test",
    save_dir="output/experiments/test/rec_001"
)

# 记录参数
recorder.log_params(lr=0.001, batch_size=32)

# 记录指标
recorder.log_metrics(step=1, loss=0.5, acc=0.85)

# 保存对象
recorder.save_objects(model=model, config=config)

# 加载对象
loaded_model = recorder.load_object("model")
```

**主要方法**:
- `log_params(**params)`: 记录参数
- `log_metrics(step=None, **metrics)`: 记录指标
- `save_objects(**objects)`: 保存Python对象
- `load_object(name)`: 加载对象
- `get_params()`: 获取所有参数
- `get_metrics()`: 获取所有指标
- `set_status(status)`: 设置状态

### 2. Experiment - 实验

管理同一实验下的多个Recorder。

```python
from quantclassic.workflow import Experiment

experiment = Experiment(
    experiment_name="hyperparameter_tuning",
    save_dir="output/experiments"
)

# 创建多个recorder
for lr in [0.001, 0.01, 0.1]:
    recorder = experiment.create_recorder()
    recorder.log_params(learning_rate=lr)
    # ... 训练和记录 ...

# 列出所有recorders
recorders = experiment.list_recorders()
```

**主要方法**:
- `create_recorder(recorder_name=None)`: 创建新recorder
- `get_recorder(recorder_id)`: 获取recorder
- `list_recorders()`: 列出所有recorders
- `delete_recorder(recorder_id)`: 删除recorder

### 3. ExpManager - 实验管理器

顶层管理器，管理所有实验。

```python
from quantclassic.workflow import ExpManager

manager = ExpManager(exp_dir="output/experiments")

# 创建实验
experiment = manager.create_experiment("new_experiment")

# 启动recorder
recorder_id = manager.start_recorder(
    experiment_name="new_experiment",
    recorder_name="run_1"
)

# 获取recorder
recorder = manager.get_recorder("new_experiment", recorder_id)

# 结束recorder
manager.end_recorder("new_experiment", recorder_id, status="FINISHED")

# 搜索
results = manager.search_recorders(
    experiment_name="new_experiment",
    status="FINISHED"
)
```

**主要方法**:
- `create_experiment(name)`: 创建实验
- `start_recorder(experiment_name, recorder_name, resume)`: 启动recorder
- `end_recorder(experiment_name, recorder_id, status)`: 结束recorder
- `get_recorder(experiment_name, recorder_id)`: 获取recorder
- `list_experiments()`: 列出所有实验
- `list_recorders(experiment_name)`: 列出实验的recorders
- `search_recorders(experiment_name, status, **params)`: 搜索recorders

### 4. QCRecorder (R) - 全局接口

提供最简化的使用接口。

```python
from quantclassic.workflow import R

# 基本使用
with R.start(experiment_name="test"):
    R.log_params(lr=0.001)
    R.log_metrics(loss=0.5)
    R.save_objects(model=model)

# 恢复训练
with R.start(experiment_name="test", 
            recorder_name="my_run",
            resume=True):
    # 继续之前的训练
    checkpoint = R.current_recorder.load_object("checkpoint")
```

**主要方法**:
- `start(experiment_name, recorder_name, resume, **params)`: 启动上下文
- `log_params(**params)`: 记录参数
- `log_metrics(step, **metrics)`: 记录指标
- `save_objects(**objects)`: 保存对象
- `load_object(experiment_name, recorder_id, object_name)`: 加载对象
- `list_experiments()`: 列出实验
- `list_recorders(experiment_name)`: 列出recorders
- `search_recorders(experiment_name, status, **params)`: 搜索

---

## 📖 使用指南

### 场景1: 模型训练

```python
from quantclassic.workflow import R
from quantclassic.model import ModelFactory

config = {
    'model_type': 'LSTM',
    'input_size': 10,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.3
}

with R.start(experiment_name="lstm_training", **config):
    # 创建模型
    model = ModelFactory.create_model('LSTM', config)
    
    # 训练
    for epoch in range(100):
        train_loss = train_epoch(model, train_loader)
        val_loss = validate(model, val_loader)
        
        R.log_metrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss
        )
        
        # 早停
        if should_early_stop(val_loss):
            break
    
    # 最终评估
    test_metrics = evaluate(model, test_loader)
    R.log_metrics(**test_metrics)
    
    # 保存
    R.save_objects(
        model=model.state_dict(),
        config=config
    )
```

### 场景2: 超参数搜索

```python
from quantclassic.workflow import R
import itertools

# 定义搜索空间
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'hidden_size': [32, 64, 128],
    'dropout': [0.1, 0.3, 0.5]
}

# Grid Search
best_ic = -float('inf')
best_params = None

for lr, hidden, dropout in itertools.product(*param_grid.values()):
    with R.start(experiment_name="grid_search",
                learning_rate=lr,
                hidden_size=hidden,
                dropout=dropout):
        
        # 训练
        model = build_model(hidden, dropout)
        train_model(model, lr)
        
        # 验证
        val_ic = evaluate(model, val_data)
        R.log_metrics(val_ic=val_ic)
        
        # 跟踪最佳
        if val_ic > best_ic:
            best_ic = val_ic
            best_params = {'lr': lr, 'hidden': hidden, 'dropout': dropout}
            R.save_objects(best_model=model)

print(f"最佳参数: {best_params}, IC={best_ic}")
```

### 场景3: 因子回测

```python
from quantclassic.workflow import R
from quantclassic.Factorsystem import ICAnalyzer, BacktestSystem

with R.start(experiment_name="factor_backtest",
            factor_name="momentum_reversal",
            lookback_period=20):
    
    # 生成因子
    factor_data = generate_factor(data, lookback=20)
    
    # IC分析
    ic_analyzer = ICAnalyzer()
    ic_results = ic_analyzer.analyze(factor_data, returns)
    
    R.log_metrics(
        mean_ic=ic_results['mean_ic'],
        ic_ir=ic_results['ic_ir'],
        win_rate=ic_results['win_rate']
    )
    
    # 回测
    backtest = BacktestSystem()
    bt_results = backtest.run(factor_data, price_data)
    
    R.log_metrics(
        total_return=bt_results['total_return'],
        sharpe_ratio=bt_results['sharpe'],
        max_drawdown=bt_results['mdd']
    )
    
    # 保存
    R.save_objects(
        factor_data=factor_data,
        ic_results=ic_results,
        backtest_results=bt_results
    )
```

### 场景4: 模型对比

```python
from quantclassic.workflow import R

model_types = ['LSTM', 'GRU', 'Transformer']

for model_type in model_types:
    with R.start(experiment_name="model_comparison",
                model_type=model_type):
        
        model = build_model(model_type)
        train_model(model, train_data)
        
        # 多维度评估
        test_results = evaluate(model, test_data)
        
        R.log_metrics(
            test_ic=test_results['ic'],
            test_rankic=test_results['rankic'],
            sharpe=test_results['sharpe']
        )
        
        R.save_objects(model=model)

# 对比分析
recorders = R.list_recorders("model_comparison")
comparison_df = []

for rec_id, info in recorders.items():
    recorder = R.get_recorder("model_comparison", rec_id)
    metrics = recorder.get_metrics()
    
    comparison_df.append({
        'model_type': info['params']['model_type'],
        'ic': metrics['test_ic'][-1][1],
        'rankic': metrics['test_rankic'][-1][1],
        'sharpe': metrics['test_sharpe'][-1][1]
    })

import pandas as pd
df = pd.DataFrame(comparison_df)
print(df.sort_values('sharpe', ascending=False))
```

---

## 🔧 API文档

### R.start()

启动一个recorder的上下文管理器。

**参数**:
- `experiment_name` (str): 实验名称，必需
- `recorder_name` (str, optional): Recorder名称，默认自动生成
- `resume` (bool): 是否恢复已有recorder，默认False
- `**params`: 初始参数

**返回**: Recorder实例

**示例**:
```python
with R.start(experiment_name="test", lr=0.001):
    R.log_metrics(loss=0.5)
```

### R.log_params()

记录参数。

**参数**:
- `**params`: 参数键值对

**示例**:
```python
R.log_params(
    learning_rate=0.001,
    batch_size=32,
    model_type="LSTM"
)
```

### R.log_metrics()

记录指标。

**参数**:
- `step` (int, optional): 步数/epoch，默认自动递增
- `**metrics`: 指标键值对

**示例**:
```python
R.log_metrics(epoch=1, loss=0.5, accuracy=0.85)
```

### R.save_objects()

保存Python对象。

**参数**:
- `**objects`: 对象键值对

**示例**:
```python
R.save_objects(
    model=model.state_dict(),
    optimizer=optimizer.state_dict(),
    config=config_dict
)
```

### R.load_object()

加载保存的对象。

**参数**:
- `experiment_name` (str): 实验名称
- `recorder_id` (str): Recorder ID
- `object_name` (str): 对象名称

**返回**: 加载的对象

**示例**:
```python
model = R.load_object(
    experiment_name="training",
    recorder_id="rec_20240115_120000_abc123",
    object_name="model"
)
```

### R.list_experiments()

列出所有实验。

**返回**: Dict[str, Dict] - 实验信息字典

**示例**:
```python
experiments = R.list_experiments()
for name, info in experiments.items():
    print(f"{name}: {info['recorder_count']} runs")
```

### R.list_recorders()

列出实验的所有recorders。

**参数**:
- `experiment_name` (str): 实验名称

**返回**: Dict[str, Dict] - Recorder信息字典

**示例**:
```python
recorders = R.list_recorders("my_experiment")
for rec_id, info in recorders.items():
    print(f"{rec_id}: {info['status']}")
```

### R.search_recorders()

搜索符合条件的recorders。

**参数**:
- `experiment_name` (str, optional): 实验名称过滤
- `status` (str, optional): 状态过滤（FINISHED/FAILED/RUNNING）
- `**params`: 参数过滤

**返回**: List[Dict] - 符合条件的recorder列表

**示例**:
```python
# 搜索所有已完成的runs
finished = R.search_recorders(status="FINISHED")

# 搜索特定参数的runs
lr_001_runs = R.search_recorders(
    experiment_name="grid_search",
    learning_rate=0.001
)
```

---

## 🎨 最佳实践

### 1. 使用描述性的实验名称

```python
# ✅ 好
with R.start(experiment_name="lstm_csi300_daily_20240101"):
    ...

# ❌ 差
with R.start(experiment_name="test"):
    ...
```

### 2. 记录完整的参数

```python
# ✅ 完整
with R.start(experiment_name="training",
            model_type="LSTM",
            learning_rate=0.001,
            batch_size=256,
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
            optimizer="Adam",
            data_start="2020-01-01",
            data_end="2023-12-31"):
    ...
```

### 3. 阶段性保存checkpoint

```python
with R.start(experiment_name="long_training"):
    for epoch in range(1000):
        train()
        
        # 每10个epoch保存一次
        if epoch % 10 == 0:
            R.save_objects(
                checkpoint={
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
            )
```

### 4. 使用有意义的metric名称

```python
# ✅ 清晰
R.log_metrics(
    train_loss=0.1,
    val_loss=0.15,
    test_ic=0.05,
    test_rankic=0.06
)

# ❌ 模糊
R.log_metrics(loss1=0.1, loss2=0.15, metric1=0.05)
```

---

## 📊 与Qlib对比

| 特性 | Qlib | QuantClassic |
|------|------|--------------|
| 全局R接口 | ✅ | ✅ |
| 上下文管理器 | ✅ | ✅ |
| 参数记录 | ✅ | ✅ |
| 指标记录 | ✅ | ✅ |
| 对象保存 | ✅ | ✅ |
| 实验搜索 | ✅ | ✅ |
| 后端存储 | MLflow/File | File |
| 依赖外部服务 | 可选 | 不需要 ✅ |
| 分布式支持 | ✅ | 待开发 |
| Web UI | ✅ | 待开发 |

---

## 🔮 未来规划

### v1.1
- [ ] Web UI界面
- [ ] 实验对比可视化
- [ ] 导出为PDF报告

### v1.2
- [ ] 数据库后端支持（MongoDB）
- [ ] 分布式实验管理
- [ ] API服务器

### v2.0
- [ ] 自动超参数优化集成
- [ ] 实验A/B测试框架
- [ ] 云端同步支持

---

## 📝 更多资源

- [使用示例](./USAGE_EXAMPLES.md) - 详细的使用案例
- [Qlib文档](https://qlib.readthedocs.io/) - 参考设计来源
- [问题反馈](../../issues) - 报告bug或建议

---

## 📄 许可证

与QuantClassic主项目相同。

---

**Happy Experimenting! 🎉**
