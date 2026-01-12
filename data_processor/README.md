# 数据预处理模块使用指南

## 📖 目录

- [简介](#简介)
- [模块综述与职责边界](#模块综述与职责边界)
- [核心特性](#核心特性)
- [继承关系与依赖](#继承关系与依赖)
- [与 data_set / model 的协同](#与-data_set--model-的协同)
- [重复实现与硬编码检查](#重复实现与硬编码检查)
- [快速开始](#快速开始)
- [模块架构](#模块架构)
- [详细使用说明](#详细使用说明)
- [处理方法详解](#处理方法详解)
- [配置与自定义](#配置与自定义)
- [完整示例](#完整示例)
- [最佳实践](#最佳实践)
- [API参考](#api参考)

---

## 简介

量化数据预处理模块是一个工程化的特征处理工具,专为量化投资研究设计。提供标准化、中性化、极值处理等多种数据预处理功能,支持配置驱动、状态保存、训练推理分离。

## 模块综述与职责边界

- 目标: 在进入数据集/模型阶段前完成列名对齐、异常值处理、中性化和尺度归一,保证后续特征工程、图构建与训练输入一致。
- 定位: 面向 `DataManager`/回测/训练的前置管线,不承担特征生成或模型训练逻辑,仅聚焦特征清洗与可重复性。
- 流程角色: `fit_transform()` 产出可复现的标准化特征; `transform()` 复用训练态参数用于推理/日更。
- 状态管理: 通过 `save()/load()` 持久化配置与统计量,确保不同进程、滚动窗口和线上/线下环境一致。
- 输出契约: 保留原始 ID/时间列,仅对特征列做可追溯的变换; 不改动标签列,不做特征自动生成。

### ✨ 核心特性

- **配置驱动**: 通过配置对象灵活控制处理流程
- **模块化设计**: 清晰的职责分离,易于扩展
- **训练推理分离**: fit_transform/transform分离模式
- **状态持久化**: 保存和加载预处理器状态
- **丰富的处理方法**: 10+ 种预处理算法
- **灵活的字段选择**: 支持对不同特征应用不同处理
- **🆕 列名自适应**: 自动兼容 `ts_code`(Tushare/DataManager) 和 `order_book_id`(RiceQuant)
- **🆕 防重复转换**: 窗口处理器内置转换标记，防止重复应用

## 继承关系与依赖

### 模块结构与职责

```
data_processor/
├─ data_preprocessor.py   # 编排器: 执行 fit/transform/save/load
├─ preprocess_config.py   # 配置与模板: ProcessMethod/ProcessingStep/PreprocessConfig
├─ feature_processor.py   # 算法引擎: 缺失/极值/标准化/中性化/秩归一
├─ window_processor.py    # 窗口级转换: 对数变换、成交量标准化、防重复标记
├─ graph_builder.py       # 图构建工厂: 行业/相关/混合, 唯一入口
├─ label_generator.py     # 标签生成辅助: 未来收益等衍生标签
├─ example_preprocess.py  # 快速示例
└─ preprocess_config.py   # 预设模板与列名自适配
```

内部类关系:

```
DataPreprocessor (入口)
    ├─ PreprocessConfig (步骤/列名/分组配置)
    │    └─ List[ProcessingStep] (方法枚举于 ProcessMethod)
    └─ FeatureProcessor (按配置执行算法)

WindowProcessor (可选窗口预处理, 提供转换标记)
GraphBuilderFactory (图构建唯一入口, 由 graph_builder 提供)
```

### 依赖与约定

- 运行时依赖: pandas/numpy/scipy/statsmodels(OLS 中性化) 等常见科学计算库。
- 列名约定: 默认 `stock_col='order_book_id'`, `time_col='trade_date'`, `auto_adapt_columns` 自动映射 `ts_code/stock_code/symbol`。
- 图构建: `GraphBuilderFactory` 作为唯一工厂, 支持行业/相关/混合图, 可从 `DataConfig` 透传列名与股票池。
- 数据集衔接: 预处理输出列名需与 `data_set.DataManager` 的 `stock_col/time_col` 保持一致, 防止分组、滚动窗口或图构建失败。

## 与 data_set / model 的协同

- DataManager 前置清洗: 在调用 `DataManager.run_full_pipeline()` 前, 使用 `DataPreprocessor.fit_transform()` 生成干净特征; 线上增量数据使用同一预处理器 `transform()` 保持统计量一致。
- 列名自适应链路: PreprocessConfig.detect → FeatureProcessor.stock_col → GraphBuilderConfig.adapt_stock_col → DataManager.stock_col, 避免 `ts_code/order_book_id` 不一致导致的分组、中性化或图构建错误。
- 窗口与图数据: `WindowProcessor` 的转换标记避免与 `TimeSeriesStockDataset`/`DailyGraphDataLoader` 二次执行对数/标准化; 图构建统一通过 `GraphBuilderFactory`，model 侧仅消费生成的邻接矩阵。
- 模型输入一致性: 经过预处理的数据直接喂给 `model.PyTorchModel` 及其训练器，保持训练/推理尺度对齐，避免因重复标准化或列名缺失造成的 batch 解析错误。

## 重复实现与硬编码检查

- 列名硬编码清理: `PreprocessConfig.get_stock_col()` 自动探测 `order_book_id/ts_code/stock_code/symbol`, `FeatureProcessor` 与中性化、时序标准化统一使用该列。
- 窗口转换去重: `WindowProcessor` 引入 `_WINDOW_TRANSFORM_MARKER` 及 `is_transformed()/mark_transformed()`，与数据集内的窗口级变换互斥，避免重复 log/scale。
- 图构建统一入口: 仅保留 `data_processor/graph_builder.py` 作为 canonical path，`GraphBuilderConfig.from_data_config()` 支持列名透传；旧的 model 侧静态邻接工具已标注为 deprecated。
- Daily collate 列名解耦: `daily_graph_loader.collate_daily` 根据图配置动态读取股票列，避免硬编码 `order_book_id`。
- 处理方法去硬编码: 时序标准化与中性化参数已移除固定列名，全部从配置或自动探测获取，防止回退全局统计导致的隐性失效。

### ⚠️ 与 DataManager 集成注意事项

本模块与 `data_set.DataManager` 集成时，需注意以下列名差异：

| 模块 | 股票代码列默认值 | 时间列默认值 |
|------|-----------------|-------------|
| data_processor | `order_book_id` | `trade_date` |
| data_set (DataManager) | `ts_code` | `trade_date` |

**解决方案**（已内置，默认启用）：
- `PreprocessConfig` 设置 `auto_adapt_columns=True`（默认）
- 处理器会在 `fit_transform()` 时自动检测并适配列名

---

## 快速开始

### 1. 最简单的使用

```python
from quantclassic.data_processor import DataPreprocessor, PreprocessTemplates

# 使用预定义模板
preprocessor = DataPreprocessor(PreprocessTemplates.basic_pipeline())

# 处理数据
df_processed = preprocessor.fit_transform(df)
```

### 2. 自定义处理流程

```python
from quantclassic.data_processor import (
    DataPreprocessor,
    PreprocessConfig,
    ProcessingStep,
    ProcessMethod
)

# 创建自定义配置
config = PreprocessConfig()
config.add_step('去极值', ProcessMethod.WINSORIZE, params={'limits': (0.025, 0.025)})
config.add_step('标准化', ProcessMethod.Z_SCORE)
config.add_step('秩归一化', ProcessMethod.RANK, params={'output_range': (-1, 1)})

# 创建预处理器
preprocessor = DataPreprocessor(config)

# 拟合并转换
df_processed = preprocessor.fit_transform(df, feature_columns=['feature1', 'feature2'])
```

### 3. 训练推理分离

```python
# 训练阶段
preprocessor = DataPreprocessor(config)
df_train_processed = preprocessor.fit_transform(df_train)

# 保存预处理器
preprocessor.save('preprocessor.pkl')

# 推理阶段
loaded_preprocessor = DataPreprocessor.load('preprocessor.pkl')
df_test_processed = loaded_preprocessor.transform(df_test)
```

---

## 模块架构

```
data_processor/
├── preprocess_config.py       # 配置管理
│   ├── ProcessMethod          # 处理方法枚举
│   ├── ProcessingStep         # 单步处理配置
│   ├── NeutralizeConfig       # 中性化配置
│   ├── PreprocessConfig       # 总配置
│   └── PreprocessTemplates    # 预定义模板
│
├── feature_processor.py       # 算法实现引擎
│   └── FeatureProcessor       # 特征处理器
│       ├── 标准化方法
│       ├── 中性化方法
│       ├── 极值处理
│       └── 缺失值处理
│
├── data_preprocessor.py       # 主编排器
│   └── DataPreprocessor       # 数据预处理器
│       ├── fit_transform()
│       ├── transform()
│       ├── save() / load()
│       └── 工具方法
│
└── example_preprocess.py      # 使用示例
```

### 核心类关系

```
DataPreprocessor (主编排器)
    │
    ├── PreprocessConfig (配置管理)
    │       └── List[ProcessingStep]
    │
    └── FeatureProcessor (算法引擎)
            └── 各种处理方法的具体实现
```

---

## 详细使用说明

### 基本工作流程

```python
# 步骤1: 准备数据
df = pd.DataFrame({
    'trade_date': [...],
    'order_book_id': [...],
    'industry_name': [...],
    'total_mv': [...],
    'feature_1': [...],
    'feature_2': [...]
})

# 步骤2: 创建配置
config = PreprocessConfig()
config.add_step('去极值', ProcessMethod.WINSORIZE)
config.add_step('标准化', ProcessMethod.Z_SCORE)

# 步骤3: 创建预处理器
preprocessor = DataPreprocessor(config)

# 步骤4: 处理数据
df_processed = preprocessor.fit_transform(
    df,
    feature_columns=['feature_1', 'feature_2']
)

# 步骤5: 保存状态(可选)
preprocessor.save('my_preprocessor.pkl')
```

### 使用预定义模板

系统提供3个预定义模板:

#### 1. 基础模板

```python
config = PreprocessTemplates.basic_pipeline()
# 步骤:
# 1. 处理无穷值
# 2. 填充缺失值(中位数)
# 3. 去极值(Winsorize 2.5%)
# 4. Z-score标准化
```

#### 2. 高级模板(含中性化)

```python
config = PreprocessTemplates.advanced_pipeline()
# 步骤:
# 1. 处理无穷值
# 2. 填充缺失值(中位数)
# 3. 去极值(Winsorize 2.5%)
# 4. 市值行业中性化(OLS)
# 5. 秩归一化到(-1, 1)
```

#### 3. Alpha研究模板

```python
config = PreprocessTemplates.alpha_pipeline()
# 步骤:
# 1. 处理无穷值
# 2. 填充缺失值(中位数)
# 3. 去极值(Winsorize 1%)
# 4. SimStock中性化
# 5. 秩归一化到(-1, 1)
```

---

## 处理方法详解

### 1. 标准化/归一化

#### Z-score 标准化

```python
config.add_step(
    'Z-score标准化',
    ProcessMethod.Z_SCORE,
    features=['feature1', 'feature2'],
    params={
        'ddof': 1,           # 标准差自由度
        'clip_sigma': 3.0    # 可选的sigma截断
    }
)
```

**效果**: 转换为均值0、标准差1的分布

#### 最小最大归一化

```python
config.add_step(
    'MinMax归一化',
    ProcessMethod.MINMAX,
    params={'output_range': (0, 1)}
)
```

**效果**: 归一化到指定区间[0, 1]

#### 秩归一化

```python
config.add_step(
    '秩归一化',
    ProcessMethod.RANK,
    params={
        'output_range': (-1, 1),
        'rank_method': 'average'  # 'average', 'min', 'max', 'dense'
    }
)
```

**效果**: 按秩归一化到指定区间[-1, 1],对异常值稳健

### 2. 中性化处理

#### OLS 市值行业中性化

```python
config.add_step(
    'OLS中性化',
    ProcessMethod.OLS_NEUTRALIZE,
    params={
        'industry_column': 'industry_name',
        'market_cap_column': 'total_mv',
        'min_samples': 10
    }
)
```

**原理**: 使用OLS回归剔除行业和市值的影响
$$\text{feature\_residual} = \text{feature} - (\beta_{\text{industry}} \times \text{industry\_dummy} + \beta_{\text{mv}} \times \log(\text{market\_cap}))$$

#### 减均值版中性化

```python
config.add_step(
    '均值中性化',
    ProcessMethod.MEAN_NEUTRALIZE,
    params={
        'industry_column': 'industry_name',
        'market_cap_column': 'total_mv',
        'n_quantiles': 5  # 市值分组数
    }
)
```

**原理**: 在行业-市值分组内减去组内均值

#### SimStock 去中性化

```python
config.add_step(
    'SimStock中性化',
    ProcessMethod.SIMSTOCK_NEUTRALIZE,
    params={
        'similarity_threshold': 0.7,      # 相关性阈值
        'lookback_window': 252,           # 回溯窗口
        'min_similar_stocks': 5,          # 最小相似股票数
        'correlation_method': 'pearson'   # 'pearson' 或 'spearman'
    }
)
```

**原理**: 使用同类型股票(基于收益率相关性)的平均特征作为基准
$$\text{SimStock}_i = \{j: \text{corr}(\text{ret}_i, \text{ret}_j) \geq \text{threshold}\}$$
$$\text{Alpha}_i = \text{feature}_i - \text{mean}(\text{feature}_{\text{SimStock}_i})$$

**注意**: 需要提供 `target_column` 参数(收益率列)

### 3. 极值处理

#### 缩尾处理 (Winsorize)

```python
config.add_step(
    '缩尾处理',
    ProcessMethod.WINSORIZE,
    params={'limits': (0.025, 0.025)}  # 上下各2.5%
)
```

**效果**: 将极端值替换为指定百分位的值

#### 截尾处理 (Clip)

```python
config.add_step(
    '截尾处理',
    ProcessMethod.CLIP,
    params={
        'lower_percentile': 1,
        'upper_percentile': 99
    }
)
```

**效果**: 将极端值截断到指定百分位

### 4. 缺失值处理

```python
# 中位数填充
config.add_step('填充缺失值', ProcessMethod.FILLNA_MEDIAN)

# 均值填充
config.add_step('填充缺失值', ProcessMethod.FILLNA_MEAN)

# 前向填充
config.add_step('填充缺失值', ProcessMethod.FILLNA_FORWARD)

# 填充0
config.add_step('填充缺失值', ProcessMethod.FILLNA_ZERO)
```

**层级填充策略**:
1. 行业内统计量填充(如果提供industry_column)
2. 市场统计量填充
3. 0填充

---

## 配置与自定义

### ProcessingStep 配置

每个处理步骤包含:

```python
ProcessingStep(
    name='步骤名称',              # 步骤描述
    method=ProcessMethod.Z_SCORE, # 处理方法
    features=['f1', 'f2'],        # 处理的特征(None=所有)
    enabled=True,                  # 是否启用
    params={'key': 'value'}        # 方法特定参数
)
```

### 灵活的特征选择

```python
# 对所有特征应用
config.add_step('标准化', ProcessMethod.Z_SCORE, features=None)

# 对特定特征应用
config.add_step('标准化', ProcessMethod.Z_SCORE, features=['feature1', 'feature2'])

# 对不同特征组应用不同处理
config.add_step('Z-score', ProcessMethod.Z_SCORE, features=['returns'])
config.add_step('秩归一化', ProcessMethod.RANK, features=['valuations'])
```

### 禁用某个步骤

```python
step = ProcessingStep(
    name='临时禁用',
    method=ProcessMethod.Z_SCORE,
    enabled=False  # 禁用此步骤
)
```

---

## 完整示例

### 示例1: 因子预处理完整流程

```python
import pandas as pd
from quantclassic.data_processor import DataPreprocessor, PreprocessConfig, ProcessMethod

# 准备数据
df = pd.read_parquet('features_raw.parquet')

# 创建配置
config = PreprocessConfig()

# 步骤1: 处理无穷值和缺失值
config.add_step('处理无穷值', ProcessMethod.CLIP, params={'lower': -1e10, 'upper': 1e10})
config.add_step('填充缺失值', ProcessMethod.FILLNA_MEDIAN)

# 步骤2: 极值处理
config.add_step('去极值', ProcessMethod.WINSORIZE, params={'limits': (0.01, 0.01)})

# 步骤3: 中性化
config.add_step(
    'OLS中性化',
    ProcessMethod.OLS_NEUTRALIZE,
    params={
        'industry_column': 'industry_name',
        'market_cap_column': 'total_mv'
    }
)

# 步骤4: 标准化
config.add_step(
    '秩归一化',
    ProcessMethod.RANK,
    params={'output_range': (-1, 1)}
)

# 创建预处理器
preprocessor = DataPreprocessor(config)

# 查看管道
print(preprocessor.get_pipeline_summary())

# 处理数据
feature_cols = ['pe_ratio', 'pb_ratio', 'momentum_20d', 'vol_20d']
df_processed = preprocessor.fit_transform(df, feature_columns=feature_cols)

# 保存预处理器
preprocessor.save('factor_preprocessor.pkl')

# 验证数据质量
report = preprocessor.validate_data(df_processed)
print(report)
```

### 示例2: 增量数据处理

```python
# === 初始训练 ===
# 历史数据
df_historical = pd.read_parquet('historical_data.parquet')

# 训练预处理器
config = PreprocessConfig()
config.add_step('标准化', ProcessMethod.Z_SCORE)

preprocessor = DataPreprocessor(config)
df_historical_processed = preprocessor.fit_transform(df_historical)

# 保存状态
preprocessor.save('production_preprocessor.pkl')

# === 日常更新 ===
# 新数据到达
df_new = pd.read_parquet('new_data.parquet')

# 加载预处理器
preprocessor = DataPreprocessor.load('production_preprocessor.pkl')

# 使用相同参数处理新数据
df_new_processed = preprocessor.transform(df_new)
```

### 示例3: Alpha因子研究

```python
# 生成因子
df_factors = calculate_alpha_factors(df_raw)

# Alpha预处理模板
config = PreprocessTemplates.alpha_pipeline()
preprocessor = DataPreprocessor(config)

# 处理(需要提供收益率列用于SimStock)
df_processed = preprocessor.fit_transform(
    df_factors,
    feature_columns=['alpha1', 'alpha2', 'alpha3'],
    target_column='ret_1d'  # 用于SimStock中性化
)

# 验证中性化效果
print("行业均值(应接近0):")
print(df_processed.groupby('industry_name')['alpha1'].mean())
```

---

## 最佳实践

### 1. 处理流程顺序建议

```python
# 推荐顺序
config = PreprocessConfig()
config.add_step('1.处理无穷值', ProcessMethod.CLIP)
config.add_step('2.填充缺失值', ProcessMethod.FILLNA_MEDIAN)
config.add_step('3.去极值', ProcessMethod.WINSORIZE)
config.add_step('4.中性化', ProcessMethod.OLS_NEUTRALIZE)
config.add_step('5.标准化', ProcessMethod.RANK)
```

**原因**:
1. 先处理异常值,避免影响统计量计算
2. 中性化在标准化前,保留特征的原始量纲信息
3. 标准化作为最后一步,统一特征尺度

### 2. 分组配置

```python
# 默认按日期分组
config = PreprocessConfig(groupby_columns=['trade_date'])

# 某些场景可能需要其他分组
# config = PreprocessConfig(groupby_columns=['trade_date', 'market'])
```

### 3. 特征分组处理

```python
# 为不同类型的特征使用不同处理
returns_features = ['ret_1d', 'ret_5d', 'ret_20d']
valuation_features = ['pe_ratio', 'pb_ratio', 'ps_ratio']

config = PreprocessConfig()

# 收益率特征: 去极值 + Z-score
config.add_step('收益去极值', ProcessMethod.WINSORIZE, 
                features=returns_features, params={'limits': (0.05, 0.05)})
config.add_step('收益标准化', ProcessMethod.Z_SCORE, features=returns_features)

# 估值特征: 去极值 + 秩归一化
config.add_step('估值去极值', ProcessMethod.WINSORIZE,
                features=valuation_features, params={'limits': (0.01, 0.01)})
config.add_step('估值归一化', ProcessMethod.RANK, features=valuation_features)
```

### 4. 验证预处理效果

```python
# 处理前
print("处理前:")
print(df[feature_cols].describe())
print(df.groupby('industry_name')[feature_cols].mean())

# 处理
df_processed = preprocessor.fit_transform(df, feature_columns=feature_cols)

# 处理后
print("\n处理后:")
print(df_processed[feature_cols].describe())
print(df_processed.groupby('industry_name')[feature_cols].mean())

# 质量报告
report = preprocessor.validate_data(df_processed)
print("\n质量报告:")
print(f"缺失率: {report['missing_ratio']}")
print(f"零方差特征: {report['zero_std_features']}")
```

### 5. 生产环境部署

```python
# 训练阶段
preprocessor = DataPreprocessor(config)
df_train_processed = preprocessor.fit_transform(df_train)
preprocessor.save('models/preprocessor_v1.pkl')

# 推理阶段
class FactorPredictor:
    def __init__(self):
        self.preprocessor = DataPreprocessor.load('models/preprocessor_v1.pkl')
        self.model = load_model('models/model_v1.pkl')
    
    def predict(self, df_new):
        # 预处理
        df_processed = self.preprocessor.transform(df_new)
        
        # 预测
        predictions = self.model.predict(df_processed)
        
        return predictions
```

---

## API 参考

### DataPreprocessor

#### 初始化

```python
DataPreprocessor(config: Union[PreprocessConfig, str, dict])
```

- `config`: 配置对象、配置文件路径或字典

#### 主要方法

##### fit_transform()

```python
fit_transform(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    target_column: Optional[str] = None
) -> pd.DataFrame
```

拟合并转换数据(训练模式)

**参数**:
- `df`: 输入数据框
- `feature_columns`: 特征列(None=自动推断)
- `target_column`: 目标列(用于SimStock等)

**返回**: 处理后的DataFrame

##### transform()

```python
transform(
    df: pd.DataFrame,
    target_column: Optional[str] = None
) -> pd.DataFrame
```

转换数据(推理模式,使用已拟合参数)

##### save() / load()

```python
save(save_path: str)
load(load_path: str) -> DataPreprocessor  # 类方法
```

保存和加载预处理器状态

##### 工具方法

```python
get_pipeline_summary() -> pd.DataFrame  # 获取管道步骤摘要
validate_data(df) -> Dict               # 验证数据质量
set_pipeline_steps(steps)               # 更新处理步骤
```

### PreprocessConfig

#### 创建配置

```python
config = PreprocessConfig(
    pipeline_steps=[],
    groupby_columns=['trade_date'],
    id_columns=['order_book_id', 'trade_date']
)
```

#### 添加步骤

```python
config.add_step(
    name: str,
    method: Union[str, ProcessMethod],
    features: Union[str, List[str], None] = None,
    enabled: bool = True,
    **params
)
```

### ProcessMethod 枚举

```python
ProcessMethod.Z_SCORE              # Z-score标准化
ProcessMethod.MINMAX               # MinMax归一化
ProcessMethod.RANK                 # 秩归一化
ProcessMethod.OLS_NEUTRALIZE       # OLS中性化
ProcessMethod.MEAN_NEUTRALIZE      # 均值中性化
ProcessMethod.SIMSTOCK_NEUTRALIZE  # SimStock中性化
ProcessMethod.WINSORIZE            # 缩尾处理
ProcessMethod.CLIP                 # 截尾处理
ProcessMethod.FILLNA_MEDIAN        # 中位数填充
ProcessMethod.FILLNA_MEAN          # 均值填充
ProcessMethod.FILLNA_FORWARD       # 前向填充
ProcessMethod.FILLNA_ZERO          # 零填充
```

---

## 常见问题

### Q1: 如何选择中性化方法?

**OLS中性化**: 适用于需要精确控制行业和市值影响的场景,效果更好但计算量大

**均值中性化**: 简单快速,适用于快速实验

**SimStock中性化**: 适用于股票间存在显著相关性的场景,能捕捉股票特质信息

### Q2: 秩归一化 vs Z-score?

**秩归一化**: 对异常值稳健,保证输出在固定区间,适合作为模型输入

**Z-score**: 保留原始分布信息,适合需要统计推断的场景

### Q3: 如何处理新股票?

新股票在历史窗口内可能数据不足,建议:
1. 设置合理的 `min_samples` 参数
2. 使用稳健的填充策略
3. 考虑排除上市不足N天的股票

### Q4: 内存占用过大怎么办?

1. 分批处理数据
2. 使用更高效的数据类型(如int8, float32)
3. 及时释放不需要的中间变量
4. 考虑使用Dask等分布式框架

---

## 更新日志

### v1.0.0 (2025-11-19)
- ✨ 初始版本发布
- ✨ 支持10+种预处理方法
- ✨ 配置驱动架构
- ✨ fit_transform/transform分离
- ✨ 状态持久化
- ✨ 完整文档和示例

---

## 技术支持

遇到问题请查看示例代码 `example_preprocess.py` 或联系开发团队。

## 许可证

内部使用工具,禁止外部分发。
