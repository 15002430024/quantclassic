# ClickHouse 数据源集成指南

## 概述

quantclassic 数据加载模块现已支持 ClickHouse 数据源（基于 quantchdb），可以方便地从 ClickHouse 数据库获取 ETF 数据，并自动标准化字段格式。

## 功能特性

### 1. 多数据源支持
- **米筐 (RQ)**: 传统 A 股数据源
- **ClickHouse**: ETF 数据源（基于 quantchdb）
- 统一的数据接口，无缝切换

### 2. 自动字段映射
- ClickHouse 字段 → 标准字段
- 数据类型自动转换
- 衍生字段自动计算

### 3. 数据标准化
- 统一的日期格式
- 统一的数值类型
- 统一的字段命名

## 快速开始

### 方式一：使用配置文件

#### 1. 修改 config.yaml

```yaml
# 数据源配置
data_source:
  source: "clickhouse"     # 切换到 ClickHouse 数据源
  market: "cn"
  
  # ClickHouse 连接配置
  clickhouse_config:
    host: "10.13.66.5"
    port: 20108
    user: "etf_visitor"
    password: "etf_lh_2025"
    database: "etf"

# 时间配置
time_settings:
  start_date: "2024-08-25"
  end_date: "2024-11-23"
  frequency: "1d"

# 存储配置
storage:
  save_dir: "etf_data"
  file_format: "parquet"
```

#### 2. 使用 UnifiedDataSource

```python
from quantclassic.data_loader import ConfigManager, UnifiedDataSource

# 加载配置
config = ConfigManager('config.yaml')

# 创建统一数据源
data_source = UnifiedDataSource(config)

# 测试连接
if data_source.test_connection():
    print("✓ 数据源连接成功")

# 获取 ETF 列表
df_etf = data_source.get_stock_list()
print(f"共 {len(df_etf)} 只 ETF")

# 获取完整日频数据
df_daily = data_source.get_all_daily_data()
print(f"数据形状: {df_daily.shape}")
print(f"字段列表: {df_daily.columns.tolist()}")
```

### 方式二：直接使用 ClickHouseFetcher

```python
from quantclassic.data_loader import ClickHouseFetcher, ConfigManager

# 创建配置
config = ConfigManager()
config.data_source.source = 'clickhouse'
config.data_source.clickhouse_config = {
    'host': '10.13.66.5',
    'port': 20108,
    'user': 'etf_visitor',
    'password': 'etf_lh_2025',
    'database': 'etf'
}
config.time.start_date = '2024-08-25'
config.time.end_date = '2024-11-23'

# 创建 ClickHouse 数据获取器
fetcher = ClickHouseFetcher(config)

# 获取 ETF 列表
df_etf = fetcher.get_etf_list()

# 获取日频数据
df_daily = fetcher.get_daily_data()

# 获取基本信息
df_info = fetcher.get_basic_info()
```

### 方式三：手动字段映射

```python
from quantclassic.data_loader import FieldMapper
import pandas as pd

# 假设从 ClickHouse 获取了原始数据
df_raw = pd.DataFrame({
    'Symbol': ['510300', '510500'],
    'TradingDate': ['2024-11-20', '2024-11-20'],
    'OpenPrice': [4.5, 6.8],
    'ClosePrice': [4.52, 6.85]
})

# 字段映射
df_standard = FieldMapper.map_fields(df_raw, source='clickhouse')

# 数据类型标准化
df_standard = FieldMapper.standardize_data_types(df_standard)

# 添加衍生字段
df_standard = FieldMapper.add_derived_fields(df_standard)

print(df_standard.columns)
# ['order_book_id', 'trade_date', 'open', 'close', 'return', ...]
```

## 字段映射规则

### ClickHouse → 标准字段

| ClickHouse 字段 | 标准字段 | 说明 |
|----------------|----------|------|
| Symbol | order_book_id | 股票代码 |
| TradingDate | trade_date | 交易日期 |
| OpenPrice | open | 开盘价 |
| HighPrice | high | 最高价 |
| LowPrice | low | 最低价 |
| ClosePrice | close | 收盘价 |
| Volume | vol | 成交量 |
| Amount | amount | 成交额 |
| TurnoverRate | turnover_rate | 换手率 |
| PE | pe | 市盈率 |
| PB | pb | 市净率 |
| TotalMarketValue | total_mv | 总市值 |
| CirculationMarketValue | circ_mv | 流通市值 |
| NetAssetValue | nav | 净值 |
| PremiumRate | premium_rate | 溢价率 |

### 自动添加的衍生字段

- `return`: 日收益率（基于 close 计算）
- `log_return`: 对数收益率
- `market`: 市场标识（SH/SZ）
- `year`, `month`, `quarter`, `weekday`: 时间相关字段

## 完整示例

### 示例 1：获取最近 3 个月 ETF 数据

```python
from quantclassic.data_loader import ConfigManager, UnifiedDataSource
from datetime import datetime, timedelta

# 配置
config = ConfigManager()
config.data_source.source = 'clickhouse'
config.data_source.clickhouse_config = {
    'host': '10.13.66.5',
    'port': 20108,
    'user': 'etf_visitor',
    'password': 'etf_lh_2025',
    'database': 'etf'
}

# 设置时间范围（最近 3 个月）
end_date = datetime.now().date()
start_date = end_date - timedelta(days=90)
config.time.start_date = str(start_date)
config.time.end_date = str(end_date)

# 获取数据
data_source = UnifiedDataSource(config)
df = data_source.get_all_daily_data()

print(f"数据时间范围: {df['trade_date'].min()} 到 {df['trade_date'].max()}")
print(f"ETF 数量: {df['order_book_id'].nunique()}")
print(f"记录数: {len(df)}")

# 保存数据
data_source.save_data(df, 'daily_data', 'etf_3months')
```

### 示例 2：比较不同数据源

```python
from quantclassic.data_loader import ConfigManager, UnifiedDataSource

# 配置 1: ClickHouse
config_ch = ConfigManager()
config_ch.data_source.source = 'clickhouse'
config_ch.data_source.clickhouse_config = {...}
config_ch.time.start_date = '2024-10-01'
config_ch.time.end_date = '2024-10-31'

# 配置 2: 米筐
config_rq = ConfigManager()
config_rq.data_source.source = 'rq'
config_rq.time.start_date = '2024-10-01'
config_rq.time.end_date = '2024-10-31'

# 获取数据
ds_ch = UnifiedDataSource(config_ch)
df_ch = ds_ch.get_all_daily_data()

ds_rq = UnifiedDataSource(config_rq)
df_rq = ds_rq.get_all_daily_data()

# 两个数据源的字段已经标准化，可以直接比较
print("ClickHouse 字段:", df_ch.columns.tolist())
print("米筐字段:", df_rq.columns.tolist())
```

### 示例 3：与 CompBase 集成

```python
from CompBase import BaseResearcher
from quantclassic.data_loader import ConfigManager, UnifiedDataSource
import pandas as pd

class ETFResearcher(BaseResearcher,
                    researcher_name="ETF_Strategy",
                    api_config=api_config,
                    db_config=db_config):
    
    def __init__(self):
        # 初始化 quantclassic 数据加载器
        self.config = ConfigManager()
        self.config.data_source.source = 'clickhouse'
        self.config.data_source.clickhouse_config = db_config  # 复用 db_config
        
        self.data_source = UnifiedDataSource(self.config)
        self.data = {}
    
    def load_data_all(self, start_date, end_date):
        """加载所有数据"""
        # 设置时间范围
        self.config.time.start_date = start_date
        self.config.time.end_date = end_date
        
        # 获取标准化的数据
        df = self.data_source.get_all_daily_data()
        
        self.data['all_data'] = df
        return df
    
    def load_data_curr(self, curr_date):
        """加载当前日期数据"""
        # 设置时间范围为单日
        self.config.time.start_date = curr_date
        self.config.time.end_date = curr_date
        
        # 获取数据
        df = self.data_source.get_all_daily_data()
        
        self.data['curr_data'] = df
        return df
    
    def get_daily_holdings(self, start_date, end_date):
        """生成每日持仓"""
        # 使用标准化的数据进行策略计算
        df = self.load_data_all(start_date, end_date)
        
        # 策略逻辑（示例）
        holdings = {}
        for date in df['trade_date'].unique():
            df_date = df[df['trade_date'] == date]
            # 选择当日收益率前10的 ETF
            top_etfs = df_date.nlargest(10, 'return_daily')
            holdings[str(date)] = {
                row['order_book_id']: 0.1 
                for _, row in top_etfs.iterrows()
            }
        
        return holdings
    
    def get_current_holdings(self, curr_date):
        """获取当前持仓"""
        daily_holdings = self.get_daily_holdings(curr_date, curr_date)
        return {curr_date: daily_holdings.get(curr_date, {})}
```

## 字段验证

```python
from quantclassic.data_loader import FieldMapper

# 验证数据字段
validation_result = FieldMapper.validate_standard_fields(df)

if validation_result['is_valid']:
    print("✓ 数据字段验证通过")
else:
    print("✗ 缺少必需字段:", validation_result['missing_required'])
    print("  缺少可选字段:", validation_result['missing_optional'])
```

## 数据质量检查

```python
from quantclassic.data_loader import DataValidator, ConfigManager

config = ConfigManager()
validator = DataValidator(config)

# 检查数据质量
quality_report = validator.check_data_quality(df)
print(f"数据质量分数: {quality_report['overall_score']:.2f}")

# 检查缺失值
missing_report = validator.check_missing_data(df)
print("缺失值统计:")
for col, pct in missing_report['missing_percentage'].items():
    if pct > 0:
        print(f"  {col}: {pct:.2%}")
```

## 注意事项

1. **安装依赖**
   ```bash
   pip install quantchdb
   ```

2. **配置数据库连接**
   - 确保网络可以访问 ClickHouse 服务器
   - 使用环境变量或 .env 文件管理敏感信息

3. **字段差异**
   - ClickHouse 数据源主要提供 ETF 数据
   - 米筐数据源主要提供 A 股数据
   - 部分字段仅在特定数据源中存在

4. **性能优化**
   - 使用日期范围筛选减少数据量
   - 对于大批量数据，建议分批获取
   - 启用数据缓存避免重复查询

## 故障排查

### 连接失败
```python
# 测试连接
if not data_source.test_connection():
    print("连接失败，请检查：")
    print("1. 网络是否畅通")
    print("2. 配置参数是否正确")
    print("3. 用户权限是否足够")
```

### 字段缺失
```python
# 查看实际获取的字段
print("实际字段:", df.columns.tolist())

# 查看标准字段列表
standard_fields = FieldMapper.get_standard_field_list()
print("标准字段:", standard_fields)
```

### 数据类型错误
```python
# 强制标准化数据类型
df = FieldMapper.standardize_data_types(df)

# 检查数据类型
print(df.dtypes)
```

## 更多信息

- 源码位置：`quantclassic/data_loader/`
- 配置示例：`quantclassic/data_loader/config.yaml`
- quantchdb 文档：`jupyterlab/chdb/readme.md`
- CompBase 文档：`jupyterlab/CompBase/README.md`
