# 快速开始指南

## 📦 安装依赖

```bash
pip install rqdatac pandas numpy pyyaml tqdm
```

## 🚀 3分钟上手

### 1. 最简单的使用

```python
from quantclassic.data_loader import QuantDataPipeline

# 创建流水线
pipeline = QuantDataPipeline()

# 获取数据
df = pipeline.run_full_pipeline()

print(f"数据形状: {df.shape}")
```

### 2. 自定义配置

```python
from quantclassic.data_loader import QuantDataPipeline, ConfigManager

# 创建配置
config = ConfigManager()
config.time.start_date = '2020-01-01'
config.time.end_date = '2024-12-31'
config.universe.universe_type = 'csi300'

# 创建流水线
pipeline = QuantDataPipeline(config=config)
df = pipeline.run_full_pipeline()
```

### 3. 使用配置文件

```python
from quantclassic.data_loader import QuantDataPipeline

# 使用YAML配置文件
pipeline = QuantDataPipeline(config_path='config.yaml')
df = pipeline.run_full_pipeline()
```

## 📂 项目结构

```
quantclassic/data_loader/
├── __init__.py              # 包初始化
├── config_manager.py        # 配置管理
├── data_fetcher.py          # 数据获取
├── data_processor.py        # 数据处理
├── data_validator.py        # 数据验证
├── pipeline.py              # 主流水线
├── config.yaml              # 配置文件模板
├── example.py               # 使用示例
├── rq_data_readme.md        # 详细文档
└── QUICKSTART.md            # 本文件
```

## 🎯 输出说明

执行完成后，会在 `rq_data_parquet/` 目录下生成:

```
rq_data_parquet/
├── basic_data/              # 基础数据
│   ├── stock_basic.parquet      # 股票列表
│   ├── trade_calendar.parquet   # 交易日历
│   └── industry_classify.parquet # 行业分类
│
├── daily_data/              # 日频数据
│   ├── daily_price.parquet      # 行情数据
│   ├── daily_valuation.parquet  # 估值数据
│   └── daily_share.parquet      # 股本数据
│
├── features_raw.parquet     # 最终特征矩阵 ⭐
├── feature_columns.txt      # 特征列名清单
└── data_quality_report.txt  # 数据质量报告
```

## 🔍 数据包含的特征

### 基础字段
- 价格: open, high, low, close, pre_close
- 成交: vol, amount, turnover_rate, volume_ratio
- 估值: pe, pe_ttm, pb, ps, total_mv, circ_mv

### 技术指标
- 收益率: ret_1d, ret_5d, ret_10d, ret_20d
- 波动率: vol_20d
- 均线: ma_close_5d, ma_close_20d, ma_vol_5d, ma_vol_20d

### 滞后特征(避免数据泄漏)
- 价格滞后: close_lag_1, close_lag_2, close_lag_3, close_lag_5, close_lag_10
- 收益率滞后: ret_lag_1, ret_lag_2, ret_lag_3, ret_lag_5, ret_lag_10
- 相对强度: close_to_ma5_lag_1, close_to_ma20_lag_1
- 动量: momentum_lag_1_5, momentum_lag_1_10

## 💡 常用场景

### 获取不同股票池

```python
# 中证800
config.universe.universe_type = 'csi800'

# 沪深300
config.universe.universe_type = 'csi300'

# 中证500
config.universe.universe_type = 'csi500'

# 全部A股
config.universe.universe_type = 'all_a'

# 自定义
pipeline.run_custom_universe(['000001.XSHE', '600000.XSHG'])
```

### 增量更新

```python
# 只更新最新一天的数据
pipeline.run_incremental_update('2024-12-20')
```

### 加载已有数据

```python
# 加载之前保存的数据
df = pipeline.load_existing_data()

# 查看数据摘要
summary = pipeline.get_data_summary()
```

## ⚙️ 配置修改

编辑 `config.yaml` 文件:

```yaml
# 修改时间范围
time_settings:
  start_date: "2020-01-01"
  end_date: "2024-12-31"

# 修改股票池
universe:
  universe_type: "csi300"

# 修改特征配置
features:
  lag_periods: [1, 5, 10, 20]
  ma_windows: [5, 20, 60]
```

## 📖 详细文档

查看完整文档: [rq_data_readme.md](rq_data_readme.md)

查看使用示例: [example.py](example.py)

## ⚠️ 注意事项

1. **米筐API认证**: 使用前需要先初始化米筐API
   ```python
   import rqdatac
   rqdatac.init('username', 'password')
   ```

2. **数据量控制**: 首次使用建议先测试小范围数据
   ```python
   config.time.start_date = '2024-01-01'  # 只获取1年数据
   ```

3. **内存管理**: 大数据集建议使用 parquet 格式
   ```python
   config.storage.file_format = 'parquet'
   ```

4. **数据验证**: 建议始终开启数据验证
   ```python
   df = pipeline.run_full_pipeline(validate=True)
   ```

## 🆘 问题排查

### 问题1: 米筐API连接失败
```
解决: 检查账号密码是否正确,网络是否正常
```

### 问题2: 内存不足
```
解决: 减少时间范围或股票数量,使用分步执行
```

### 问题3: 数据缺失
```
解决: 检查日期范围,某些股票可能在特定时间段无数据
```

## 📞 技术支持

遇到问题请查看详细文档或联系开发团队。
