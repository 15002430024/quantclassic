# BenchmarkManager 智能缓存使用指南

## 概述

BenchmarkManager 是一个增强版的基准指数管理器，支持从米筐（RQData）API 获取指数数据，并具备智能缓存功能。

## 主要特性

### 1. 智能缓存机制

- **自动缓存**: 从 API 获取的数据会自动保存到本地
- **范围检查**: 获取数据前自动检查缓存是否覆盖所需日期范围
- **增量更新**: 如果请求范围超出缓存，只下载缺失的部分并合并
- **元数据管理**: 每个缓存都有元数据文件，记录日期范围和更新时间

### 2. 支持的指数

指数代码已更新为米筐格式（.XSHG/.XSHE）:

| 指数名称 | 代码 | 米筐格式 |
|---------|------|---------|
| 沪深300 | hs300 | 000300.XSHG |
| 中证500 | zz500 | 000905.XSHG |
| 中证800 | zz800 | 000906.XSHG |
| 上证50  | sz50  | 000016.XSHG |
| 中证1000 | zz1000 | 000852.XSHG |
| 中证2000 | csi2000 | 932000.CSI |
| 深证成指 | szzs | 399001.XSHE |
| 创业板指 | cybz | 399006.XSHE |

## 使用方法

### 基本使用

```python
from Factorsystem.benchmark_manager import BenchmarkManager

# 初始化管理器
manager = BenchmarkManager()

# 获取沪深300指数收益率（自动使用缓存）
hs300_returns = manager.get_benchmark_returns(
    'hs300',
    start_date='2023-01-01',
    end_date='2023-12-31',
    data_source='rqdatac'  # 使用米筐数据源
)

print(f"数据长度: {len(hs300_returns)}")
print(f"累计收益率: {(1 + hs300_returns).prod() - 1:.2%}")
```

### 缓存行为

#### 场景1: 第一次获取（无缓存）

```python
# 第一次获取 2023年 数据
returns = manager.get_benchmark_returns(
    'hs300',
    start_date='2023-01-01',
    end_date='2023-12-31',
    data_source='rqdatac'
)
# 结果: 从米筐API下载数据，并保存到缓存
```

#### 场景2: 再次获取相同范围（命中缓存）

```python
# 再次获取 2023年 数据
returns = manager.get_benchmark_returns(
    'hs300',
    start_date='2023-01-01',
    end_date='2023-12-31',
    data_source='rqdatac'
)
# 结果: 直接从缓存加载，不调用API
```

#### 场景3: 获取更大范围（增量更新）

```python
# 获取 2022-2024年 数据（比缓存范围大）
returns = manager.get_benchmark_returns(
    'hs300',
    start_date='2022-01-01',
    end_date='2024-12-31',
    data_source='rqdatac'
)
# 结果: 
# 1. 只下载缺失的 2022年 和 2024年 数据
# 2. 与缓存中的 2023年 数据合并
# 3. 保存更大范围的缓存
```

#### 场景4: 获取子范围（从缓存提取）

```python
# 获取 2023年上半年 数据（是缓存的子集）
returns = manager.get_benchmark_returns(
    'hs300',
    start_date='2023-01-01',
    end_date='2023-06-30',
    data_source='rqdatac'
)
# 结果: 从缓存中筛选出指定日期范围
```

### 不使用缓存

```python
# 强制从API获取，不使用缓存
returns = manager.get_benchmark_returns(
    'hs300',
    start_date='2023-01-01',
    end_date='2023-12-31',
    data_source='rqdatac',
    use_cache=False  # 关闭缓存
)
```

### 查看缓存信息

```python
# 查看所有缓存的指数信息
cache_info = manager.get_cache_info()
print(cache_info)

# 输出示例:
# benchmark_name   index_code    start_date    end_date  record_count      last_updated
#         hs300  000300.XSHG    2022-01-04  2024-12-31           730  2024-11-24 10:30:15
#         zz500  000905.XSHG    2023-01-03  2023-12-29           244  2024-11-24 10:35:20
#         zz800  000906.XSHG    2023-01-03  2023-12-29           244  2024-11-24 10:40:10
```

### 清除缓存

```python
# 清除特定指数的缓存
manager.clear_cache('000300.XSHG')

# 清除所有缓存
manager.clear_cache()
```

### 计算超额收益

```python
# 获取组合收益率（假设已有）
portfolio_returns = ...

# 获取基准收益率
benchmark_returns = manager.get_benchmark_returns('hs300', '2023-01-01', '2023-12-31')

# 计算超额收益
excess_returns = manager.calculate_excess_returns(portfolio_returns, benchmark_returns)

print(f"平均超额收益: {excess_returns.mean():.4%}")
print(f"超额收益波动率: {excess_returns.std():.4%}")
```

## 缓存存储位置

默认缓存位置: `cache/benchmark/`

文件结构:
```
cache/benchmark/
├── 000300_XSHG.parquet          # 沪深300数据
├── 000300_XSHG_meta.json        # 沪深300元数据
├── 000905_XSHG.parquet          # 中证500数据
├── 000905_XSHG_meta.json        # 中证500元数据
└── ...
```

元数据文件示例 (`000300_XSHG_meta.json`):
```json
{
  "index_code": "000300.XSHG",
  "start_date": "2022-01-04",
  "end_date": "2024-12-31",
  "record_count": 730,
  "last_updated": "2024-11-24 10:30:15"
}
```

## 自定义缓存目录

```python
# 使用自定义缓存目录
manager = BenchmarkManager(cache_dir="data/my_benchmark_cache")
```

## 与回测系统集成

在回测配置中使用:

```python
from Factorsystem.backtest_system import BacktestSystem
from Factorsystem.benchmark_manager import BenchmarkManager

# 初始化基准管理器
benchmark_manager = BenchmarkManager()

# 获取基准收益率
benchmark_returns = benchmark_manager.get_benchmark_returns(
    'hs300',
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# 在回测中使用
backtest = BacktestSystem(...)
results = backtest.run(...)

# 计算相对基准的表现
excess_returns = benchmark_manager.calculate_excess_returns(
    results['portfolio_returns'],
    benchmark_returns
)
```

## 便捷函数

```python
from Factorsystem.benchmark_manager import get_benchmark_returns

# 快速获取基准收益率
hs300_returns = get_benchmark_returns('hs300', '2023-01-01', '2023-12-31')
```

## 注意事项

1. **首次使用需要米筐账号**: 确保已配置 rqdatac 并有有效的 license
2. **日期范围**: 如果未指定日期，默认获取最近3年数据
3. **缓存更新**: 缓存不会自动更新，如需最新数据，请扩展日期范围或清除缓存
4. **网络依赖**: 首次获取或扩展范围时需要网络连接
5. **存储空间**: 每个指数的缓存大小约 几KB 到 几MB，取决于日期范围

## 测试

运行测试脚本验证功能:

```bash
cd /home/u2025210237/jupyterlab/quantclassic/Factorsystem
python test_benchmark_cache.py
```

或使用内置测试:

```bash
python benchmark_manager.py
```

## 故障排除

### 问题1: 米筐初始化失败

```python
# 手动初始化米筐
import rqdatac as rq
rq.init('your_username', 'your_password')
# 或使用 license 文件
rq.init()
```

### 问题2: 缓存损坏

```python
# 清除损坏的缓存
manager = BenchmarkManager()
manager.clear_cache('000300.XSHG')  # 清除特定指数
# 或
manager.clear_cache()  # 清除所有
```

### 问题3: 数据不一致

确保使用正确的指数代码格式（米筐格式）。如果从其他系统迁移，注意代码转换:

- 旧格式: `000300.SH` → 新格式: `000300.XSHG`
- 旧格式: `399001.SZ` → 新格式: `399001.XSHE`

## 版本历史

- **v2.0** (2024-11-24): 添加智能缓存功能，更新为米筐格式
- **v1.0**: 初始版本

## 技术细节

### 缓存检查逻辑

```
请求范围: [req_start, req_end]
缓存范围: [cache_start, cache_end]

情况1: 缓存不存在
  → 下载完整范围 [req_start, req_end]

情况2: 请求范围完全在缓存内
  → 从缓存提取，不调用API

情况3: 请求范围超出缓存
  → 下载合并范围 [min(req_start, cache_start), max(req_end, cache_end)]
  → 与缓存合并，去重后保存
```

### 数据格式

缓存数据格式 (Parquet):
```
trade_date (datetime64): 交易日期
close (float64): 收盘价
```

返回数据格式 (pandas.Series):
```
index: trade_date (DatetimeIndex)
values: 收益率 (float64)
```

## 联系方式

如有问题或建议，请联系开发团队。
