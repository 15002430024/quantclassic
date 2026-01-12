# 邻接矩阵构建指南

本文档描述 quantclassic 中邻接矩阵（图结构）的构建方式、调用链与最佳实践。

## 1. 唯一实现入口

**所有图构建逻辑统一使用 `data_processor/graph_builder.py`**

```python
from quantclassic.data_processor.graph_builder import (
    GraphBuilderFactory,   # 工厂函数（推荐）
    CorrGraphBuilder,      # 相关性图
    IndustryGraphBuilder,  # 行业图
    HybridGraphBuilder,    # 混合图
    AdjMatrixUtils,        # 兼容工具类
)
```

## 2. 三种图模式

### 2.1 行业图 (Industry Graph)
- **原理**：同行业股票互相连接，不同行业不连接
- **优点**：静态、可缓存、无数据泄露风险
- **适用**：研报 baseline、行业轮动策略

```python
config = {'type': 'industry', 'industry_col': 'industry_name'}
builder = GraphBuilderFactory.create(config)
adj, stocks, mapping = builder(df_day)
```

### 2.2 相关性图 (Correlation Graph)
- **原理**：基于特征相似度/收益率相关性构建 Top-K 邻接
- **优点**：捕捉股票间动态关系
- **注意**：每批次重算，计算成本较高

```python
config = {'type': 'corr', 'corr_method': 'cosine', 'top_k': 10}
builder = GraphBuilderFactory.create(config)
adj, stocks, mapping = builder(df_day)
```

### 2.3 混合图 (Hybrid Graph)
- **原理**：`A = α * A_corr + (1-α) * A_industry`
- **优点**：兼顾动态特征与静态行业结构
- **推荐**：α ∈ [0.6, 0.8]

```python
config = {'type': 'hybrid', 'alpha': 0.7, 'top_k': 10}
builder = GraphBuilderFactory.create(config)
adj, stocks, mapping = builder(df_day)
```

## 3. 调用链（运行时）

```
config/runner.py
    │
    ├── DataManager.create_daily_loaders()
    │       │
    │       └── GraphBuilderFactory.create(config)
    │               │
    │               └── 返回 CorrGraphBuilder / IndustryGraphBuilder / HybridGraphBuilder
    │
    ├── DailyGraphDataLoader
    │       │
    │       └── collate_daily()
    │               │
    │               └── graph_builder(df_day)  ← 每个 batch 调用一次
    │
    └── SimpleTrainer / RollingDailyTrainer
            │
            └── 消费 (X, y, adj, stock_ids, date)
```

**图计算频率**：
- 行业图：首次构建后可切片缓存复用
- 相关性/混合图：每个 epoch × 每个交易日 都重新计算

## 4. 离线预计算

当图结构固定（如纯行业图）时，可离线预生成 `.pt` 文件：

```bash
# 使用 CLI 脚本（复用同一 GraphBuilder）
python scripts/graph/build_adj.py \
    --data data.parquet \
    --type industry \
    --output output/industry_adj.pt
```

训练时加载预计算矩阵：
```python
config = {
    'type': 'industry',
    'industry_adj_path': 'output/industry_adj.pt'  # 直接切片使用
}
```

## 5. 缓存策略

### IndustryGraphBuilder 缓存
```python
builder = IndustryGraphBuilder(
    industry_adj_path='output/industry_adj.pt'  # 预加载全局矩阵
)
# 每日构图时仅按 stock_list 切片，O(N²) 内存但几乎 O(1) 时间
```

### 相关性图无缓存
相关性图基于每日特征动态计算，无法跨日缓存。如需优化：
1. 减小 `top_k` 降低计算量
2. 使用混合图，降低 `alpha` 增加行业图权重

## 6. 性能注意事项

| 图类型 | 构建复杂度 | 内存占用 | 建议场景 |
|--------|-----------|---------|---------|
| industry | O(N²) 首次，后续 O(1) | 低（可预加载） | 生产环境、大规模回测 |
| corr | O(N² × F) 每批次 | 中 | 小规模实验 |
| hybrid | O(N² × F) 每批次 | 中 | 研究探索 |

## 7. 废弃文件说明

以下文件已废弃，请勿使用：
- ~~`model/utils/adj_matrix_builder.py`~~ → 使用 `AdjMatrixUtils`
- ~~`model/build_industry_adj.py`~~ → 使用 `scripts/graph/build_adj.py`

如有旧代码引用，请迁移：
```python
# 旧（已废弃）
from model.utils.adj_matrix_builder import AdjMatrixBuilder
builder = AdjMatrixBuilder()
adj = builder.build_industry_adj(codes)

# 新（推荐）
from quantclassic.data_processor.graph_builder import AdjMatrixUtils
adj = AdjMatrixUtils.build_industry_adj(codes)
```

## 8. 完整示例

```python
from quantclassic.data_set.manager import DataManager
from quantclassic.data_set.config import DataConfig

# 配置
config = DataConfig(
    # ... 数据配置 ...
    graph_builder_config={
        'type': 'hybrid',
        'alpha': 0.7,
        'top_k': 10,
        'corr_method': 'cosine',
    }
)

# 创建数据管理器
dm = DataManager(config=config)
dm.run_full_pipeline()

# 获取日批次加载器（每批次自动构图）
daily_loaders = dm.create_daily_loaders()

# 训练
for X, y, adj, stock_ids, date in daily_loaders.train:
    pred = model(X, adj)
    loss = criterion(pred, y)
```
