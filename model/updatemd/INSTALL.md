# HybridGraph 模型安装与快速开始

## 安装依赖

### 1. 安装 PyTorch

根据您的系统和 CUDA 版本选择合适的命令：

**CPU 版本:**
```bash
pip install torch torchvision torchaudio
```

**GPU 版本 (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**GPU 版本 (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

参考官方文档: https://pytorch.org/get-started/locally/

### 2. 安装其他依赖

```bash
pip install numpy pandas tqdm pyyaml
```

### 3. 验证安装

```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
```

## 快速测试

### 测试 1: 配置系统（无需 PyTorch）

```bash
cd /home/u2025210237/jupyterlab/quantclassic/model
python -c "
from model_config import HybridGraphConfig, ModelConfigFactory

# 创建配置
config = HybridGraphConfig(d_feat=20, rnn_hidden=64)
config.validate()
print('✅ 配置系统正常')

# 测试模板
templates = ['small', 'default', 'large']
for t in templates:
    cfg = ModelConfigFactory.get_template('hybrid_graph', t)
    print(f'  {t}: rnn_hidden={cfg.rnn_hidden}, gat_hidden={cfg.gat_hidden}')
"
```

### 测试 2: 完整模型（需要 PyTorch）

```bash
cd /home/u2025210237/jupyterlab/quantclassic/model
python hybrid_graph_models.py
```

预期输出:
```
================================================================================
Hybrid Graph Models 测试
================================================================================

1. 测试 TemporalBlock:
  输入: torch.Size([32, 40, 20]) -> 输出: torch.Size([32, 64])

2. 测试 GraphBlock:
  输入: torch.Size([100, 64]) + adj torch.Size([100, 100]) -> 输出: torch.Size([100, 32])

3. 测试 FusionBlock:
  输入: torch.Size([32, 96]) -> 输出: torch.Size([32])

4. 测试 HybridNet:
  输入: torch.Size([32, 40, 20]) + adj torch.Size([32, 32]) -> 输出: torch.Size([32])

5. 测试 HybridGraphModel 创建:
✅ HybridGraphModel 创建成功

================================================================================
✅ Hybrid Graph Models 测试完成
================================================================================
```

### 测试 3: 运行示例

```bash
cd /home/u2025210237/jupyterlab/quantclassic/model
python example_hybrid_graph.py
```

## 快速开始示例

### 最简单的例子（不使用图网络）

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import HybridGraphModel

# 1. 准备数据
X_train = torch.randn(1000, 40, 20)  # [样本数, 时间步, 特征数]
y_train = torch.randn(1000)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# 2. 创建模型
model = HybridGraphModel(
    d_feat=20,
    rnn_hidden=64,
    use_graph=False,  # 先不使用图网络
    n_epochs=10
)

# 3. 训练
model.fit(train_loader)

# 4. 预测
X_test = torch.randn(100, 40, 20)
y_test = torch.randn(100)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=256)

predictions = model.predict(test_loader)
print(f"预测结果: {predictions[:5]}")
```

### 使用图网络的例子

```python
from model import HybridGraphModel
from model.utils.adj_matrix_builder import AdjMatrixBuilder

# 1. 构建邻接矩阵
builder = AdjMatrixBuilder()

# 假设有100只股票，分属10个行业
industry_codes = [f'Industry_{i % 10}' for i in range(100)]
adj_matrix = builder.build_industry_adj(industry_codes)

# 保存邻接矩阵
adj_path = 'adj_matrix.pt'
builder.save_adj_matrix(adj_matrix, adj_path)

# 2. 准备数据（注意：batch_size 必须等于股票数量）
n_stocks = 100
n_days = 200

# 每天所有股票的数据
train_data = [
    (torch.randn(n_stocks, 40, 20), torch.randn(n_stocks))
    for _ in range(n_days)
]

# 3. 创建模型
model = HybridGraphModel(
    d_feat=20,
    rnn_hidden=64,
    use_graph=True,
    gat_hidden=32,
    gat_heads=4,
    adj_matrix_path=adj_path,
    batch_size=n_stocks,  # 重要！
    n_epochs=10
)

# 4. 训练（需要自定义 DataLoader）
# model.fit(train_loader)
```

## 文件说明

```
model/
├── model_config.py              # 配置类（包含 HybridGraphConfig）
├── hybrid_graph_models.py       # HybridGraph 模型实现
├── example_hybrid_graph.py      # 使用示例
├── README_HYBRID_GRAPH.md       # 详细文档
├── INSTALL.md                   # 本文件
└── utils/
    ├── __init__.py
    └── adj_matrix_builder.py    # 邻接矩阵构建工具
```

## 常见问题

### Q1: ModuleNotFoundError: No module named 'torch'

**A:** 需要先安装 PyTorch:
```bash
pip install torch
```

### Q2: 邻接矩阵应该如何准备？

**A:** 有两种方式：

1. **基于行业分类:**
```python
from model.utils.adj_matrix_builder import AdjMatrixBuilder

builder = AdjMatrixBuilder()
industry_codes = ['金融', '金融', '科技', '制造', ...]
adj = builder.build_industry_adj(industry_codes)
builder.save_adj_matrix(adj, 'industry_adj.pt')
```

2. **基于收益率相关性:**
```python
import pandas as pd

# returns: DataFrame, shape=[时间, 股票]
returns = pd.read_csv('returns.csv')

adj = builder.build_correlation_adj(returns, top_k=10)
builder.save_adj_matrix(adj, 'correlation_adj.pt')
```

### Q3: 使用图网络时 batch_size 如何设置？

**A:** 使用图网络时，`batch_size` 必须等于股票数量。因为 GAT 需要处理整个截面的数据，而邻接矩阵定义了所有股票之间的关系。

### Q4: 如何添加基本面数据？

**A:** 指定 `funda_dim` 参数，并在 DataLoader 中返回 3 个元素：

```python
# 创建模型
model = HybridGraphModel(
    d_feat=20,
    funda_dim=10,  # 基本面维度
    ...
)

# 准备数据
X = ...  # [batch, seq_len, 20] 量价数据
F = ...  # [batch, 10] 基本面数据
y = ...  # [batch] 标签

dataset = TensorDataset(X, F, y)
loader = DataLoader(dataset, batch_size=256)
```

### Q5: 模型训练很慢怎么办？

**A:** 
1. 使用 GPU: `device='cuda'`
2. 减小模型尺寸: 使用 `'small'` 模板
3. 增大 batch_size（如果显存允许）
4. 减少 `rnn_layers` 和 `gat_heads`

### Q6: 如何保存和加载模型？

**A:**
```python
# 训练时保存
model.fit(train_loader, valid_loader, save_path='best_model.pth')

# 加载模型
model = HybridGraphModel(...)
model.load_model('best_model.pth')

# 预测
predictions = model.predict(test_loader)
```

## 下一步

- 阅读 [README_HYBRID_GRAPH.md](README_HYBRID_GRAPH.md) 了解详细文档
- 查看 [example_hybrid_graph.py](example_hybrid_graph.py) 学习完整示例
- 根据您的数据准备邻接矩阵
- 开始训练您的第一个模型！

## 支持

如有问题，请查阅：
1. 详细文档: `README_HYBRID_GRAPH.md`
2. 代码示例: `example_hybrid_graph.py`
3. 源代码注释: `hybrid_graph_models.py`
