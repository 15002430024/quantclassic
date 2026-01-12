# 数据处理模块（Data Processor）复核计划

## 范围
- 数据处理器（data_processor）、数据集（data_set）与模型（model）之间的集成
- 列名/ID 对齐、窗口转换（window transforms）、图构建器（graph builders）

## 发现的问题与修复计划

### 1) data_processor 与 data_set 之间的列名不一致 ✅ 已修复
- **位置**: `PreprocessConfig`/`FeatureProcessor` 默认预期 `order_book_id`；而 `DataConfig` 默认使用 `ts_code`。
- **风险**: ID 列可能被误当作特征处理；前向填充（forward-fill）和中性化（neutralize）步骤可能会因为分组键错误而失效。
- **修复内容**:
  - ✅ 在 `PreprocessConfig` 中新增 `stock_col`、`time_col` 和 `auto_adapt_columns` 配置项
  - ✅ 新增 `adapt_columns()` 方法，自动检测并映射 `ts_code` -> `order_book_id`
  - ✅ 新增 `get_stock_col()` 和 `get_time_col()` 辅助方法
  - ✅ 扩展 `id_columns` 默认值，支持多种命名约定
  - ✅ 更新 `FeatureProcessor` 支持动态 `stock_col`
  - ✅ 更新 `DataPreprocessor.fit_transform()` 自动执行列名适配

### 2) 窗口转换（Window Transforms）的重复触发 ✅ 已修复
- **位置**: `data_processor/window_processor.py` 和数据集（`TimeSeriesStockDataset`/`DailyBatchDataset`）都会执行价格对数变换（log transform）和成交量标准化。
- **风险**: 重复执行会导致特征变形，例如对已经取过对数的数据再次取对数。
- **修复内容**:
  - ✅ 新增 `_WINDOW_TRANSFORM_MARKER` 全局标记
  - ✅ 新增 `is_transformed()` 静态方法检测数据是否已转换
  - ✅ 新增 `mark_transformed()` 静态方法标记已转换数据
  - ✅ `process_window()` 和 `process_dataset()` 增加 `skip_if_transformed` 参数
  - ✅ 在配置和方法文档中添加防重复转换警告

### 3) 图构建器（Graph Builder）列名默认值与数据集不一致 ✅ 已修复
- **位置**: `GraphBuilderConfig` 默认 `stock_col='order_book_id'`；`DataManager` 默认 `stock_col='ts_code'`；`DailyGraph` 加载器直接将 DataFrame 传递给 `GraphBuilderFactory` 而未做调整。
- **风险**: 图构建失败或使用了错误的列进行对齐。
- **修复内容**:
  - ✅ 新增 `GraphBuilderConfig.adapt_stock_col()` 方法自动检测列名
  - ✅ 新增 `GraphBuilderConfig.from_data_config()` 类方法从 DataConfig 透传 stock_col
  - ✅ 更新 `GraphBuilderFactory.create()` 支持 `data_config` 参数自动透传
  - ✅ 更新 `CorrGraphBuilder`、`IndustryGraphBuilder` 使用自适应列名
  - ✅ `DataManager.create_daily_loaders` 已补充 `gb_dict.setdefault('stock_col', self.config.stock_col)`

### 4) 存在两套并行的图构建工具 ✅ 已修复
- **位置**: `data_processor/graph_builder.py`（支持 corr/industry/hybrid 及 top-k/threshold） vs `model/build_industry_adj.py` & `build_correlation_adjacency_matrix`（静态基准工具）。
- **风险**: 训练和回测路径下使用的邻接矩阵语义不一致，导致结果不可靠。
- **修复内容**:
  - ✅ 在 `GraphBuilder` 和 `GraphBuilderFactory` 文档中标注为"统一入口（canonical path）"
  - ✅ 标注 model 侧工具为 deprecated/静态基准
  - ✅ 在 model/build_industry_adj.py 中添加 DeprecationWarning

### 5) FeatureProcessor 时序标准化仍含硬编码列名 ✅ 已修复
- **位置**: `FeatureProcessor.z_score_normalize` 与 `minmax_normalize` 的 `time_series` 分支只检查 `order_book_id`，缺少 `ts_code`/自适应。
- **风险**: 当仅有 `ts_code` 时回退到全局统计，导致时序标准化失效。
- **修复内容**: 使用 `self.stock_col` 并沿用探测列表（`order_book_id/ts_code/stock_code/symbol`），保持 per-stock 归一。

### 6) WindowProcessor 列名自适应函数未实现 ✅ 已修复
- **位置**: `_adapt_column_names` 原为 no-op，但在 `__init__` 调用。
- **风险**: 文档承诺的列名自适配未生效，可能导致误解；调用链多一次空操作。
- **修复内容**: 实现 `_adapt_column_names(df)` 方法，在 `process_dataset` 时传入数据执行检测并更新 config。

### 7) DailyGraphLoader collate_daily 硬编码股票列 ✅ 已修复
- **位置**: `data_set/graph/daily_graph_loader.py` 的 `collate_daily` 函数中硬编码 `order_book_id`。
- **风险**: 当 GraphBuilder 配置使用 `ts_code` 时，DataFrame 列名不匹配，导致图构建失败或节点错位。
- **修复内容**: 从 `graph_builder.config.stock_col` 动态获取列名（默认 `order_book_id`），在单日批次和多日批次分支都使用动态列名构造 DataFrame。

## 后续步骤
1. ~~确定目标列名规范（建议统一使用 `stock_col` 变量名），并在 `PreprocessConfig` 中实现映射垫片~~ ✅ 完成
2. ~~增加配置标志或文档注释以避免重复的窗口转换~~ ✅ 完成
3. ~~在 `Daily` 加载器中补充 `graph_builder_config.setdefault('stock_col', self.config.stock_col)`~~ ✅ 完成
4. ~~修正 `FeatureProcessor` 时序标准化分支的列名探测~~ ✅ 完成
5. ~~实现或移除 `WindowProcessor._adapt_column_names`~~ ✅ 完成
6. ~~将 `model` 侧的邻接矩阵构建工具标注为遗产/基准代码（DeprecationWarning）~~ ✅ 完成
7. ~~修复 `collate_daily` 函数硬编码列名~~ ✅ 完成

## 总结

所有发现的依赖错误、配置硬编码、功能重复实现问题已全部修复：

✅ **列名兼容性**：data_processor 和 data_set 之间实现了 `ts_code`/`order_book_id` 自动适配  
✅ **窗口转换去重**：通过标记机制防止重复执行  
✅ **图构建器统一**：GraphBuilderFactory 作为统一入口，旧工具已标注弃用  
✅ **动态列名传递**：从 DataConfig → GraphBuilder → collate_daily 全链路支持动态 stock_col  
✅ **时序标准化修复**：FeatureProcessor 支持多种列名探测，避免回退全局统计  

系统现已具备良好的跨模块兼容性和可维护性。

## 使用示例

### 与 DataManager 集成
```python
from data_set import DataManager, DataConfig
from data_processor import DataPreprocessor, PreprocessConfig

# DataManager 使用 ts_code
data_config = DataConfig(base_dir='rq_data_parquet', stock_col='ts_code')
manager = DataManager(data_config)
df = manager.load_raw_data()

# DataPreprocessor 自动适配列名
preprocess_config = PreprocessConfig(auto_adapt_columns=True)
preprocess_config.add_step('去极值', ProcessMethod.WINSORIZE, limits=[0.025, 0.025])
preprocess_config.add_step('标准化', ProcessMethod.Z_SCORE)

processor = DataPreprocessor(preprocess_config)
df_processed = processor.fit_transform(df)  # 自动检测 ts_code 并适配
```

### 图构建器与 DataConfig 集成
```python
from data_set import DataConfig
from data_processor.graph_builder import GraphBuilderConfig, GraphBuilderFactory

# 方式1: 从 DataConfig 透传
data_config = DataConfig(stock_col='ts_code')
graph_config = GraphBuilderConfig.from_data_config(data_config, type='hybrid', top_k=10)
builder = GraphBuilderFactory.create(graph_config)

# 方式2: 通过工厂直接传入 data_config
builder = GraphBuilderFactory.create({'type': 'hybrid', 'top_k': 10}, data_config=data_config)
```

### 避免窗口转换重复
```python
from data_processor.window_processor import WindowProcessor

# 检查数据是否已转换
if WindowProcessor.is_transformed(df):
    print("数据已转换，跳过")
else:
    processor = WindowProcessor(config)
    df = processor.process_dataset(df)
```

## 修复验证

所有修复已通过语法检查，核心改动：

1. **列名自适应链路**：PreprocessConfig.adapt_columns → FeatureProcessor.stock_col → GraphBuilder.stock_col → collate_daily 动态列名
2. **防重复转换**：WindowProcessor 标记机制 + enable_window_transform 配置指引
3. **图构建统一**：GraphBuilderFactory 为统一入口，model 侧工具已弃用警告
4. **动态列名传递**：从 DataManager → GraphBuilder → DailyLoader 全链路支持 ts_code/order_book_id

系统现已具备跨模块兼容性，消除了硬编码依赖和重复实现风险。