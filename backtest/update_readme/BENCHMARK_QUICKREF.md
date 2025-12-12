# BenchmarkManager 快速参考

## 一分钟上手

```python
from Factorsystem.benchmark_manager import BenchmarkManager

manager = BenchmarkManager()

# 获取指数收益率（自动缓存）
returns = manager.get_benchmark_returns(
    'hs300',                    # 指数代码
    start_date='2023-01-01',    # 开始日期
    end_date='2023-12-31',      # 结束日期
    data_source='rqdatac'       # 数据源
)
```

## 支持的指数

| 代码 | 名称 | 米筐格式 |
|------|------|---------|
| `hs300` | 沪深300 | 000300.XSHG |
| `zz500` | 中证500 | 000905.XSHG |
| `zz800` | 中证800 | 000906.XSHG |
| `sz50` | 上证50 | 000016.XSHG |
| `zz1000` | 中证1000 | 000852.XSHG |
| `csi2000` | 中证2000 | 932000.CSI |
| `szzs` | 深证成指 | 399001.XSHE |
| `cybz` | 创业板指 | 399006.XSHE |

## 常用操作

### 查看缓存

```python
cache_info = manager.get_cache_info()
print(cache_info)
```

### 清除缓存

```python
manager.clear_cache()              # 清除所有
manager.clear_cache('000300.XSHG') # 清除指定
```

### 计算超额收益

```python
excess = manager.calculate_excess_returns(
    portfolio_returns,
    benchmark_returns
)
```

### 不使用缓存

```python
returns = manager.get_benchmark_returns(
    'hs300', '2023-01-01', '2023-12-31',
    use_cache=False
)
```

## 缓存机制

✅ **自动缓存**: API数据自动保存  
✅ **智能检查**: 自动判断是否需要下载  
✅ **增量更新**: 只下载缺失部分  
✅ **快速加载**: 重复请求秒级响应  

## 文件位置

- **代码**: `Factorsystem/benchmark_manager.py`
- **缓存**: `cache/benchmark/`
- **备份**: `Factorsystem/benchmark_manager_backup.py`

## 测试

```bash
# 测试缓存功能
python test_benchmark_cache.py

# 运行示例
python example_benchmark_usage.py

# 内置测试
python benchmark_manager.py
```

## 完整文档

📚 详细指南: `BENCHMARK_CACHE_GUIDE.md`  
📖 使用示例: `example_benchmark_usage.py`  
📄 升级说明: `BENCHMARK_UPGRADE_README.md`

## 常见问题

**Q: 如何初始化米筐？**
```python
import rqdatac as rq
rq.init()
```

**Q: 缓存在哪里？**
```
cache/benchmark/
```

**Q: 如何强制更新数据？**
```python
manager.clear_cache('000300.XSHG')
returns = manager.get_benchmark_returns(...)
```

**Q: 向后兼容吗？**

是的，完全兼容旧代码。

## 性能

- 首次获取: 正常速度（API）
- 缓存命中: ~100x 速度提升
- 增量更新: ~2-10x 速度提升
