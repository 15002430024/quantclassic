"""
ClickHouse 数据源使用示例
演示如何使用 quantclassic 从 ClickHouse 获取 ETF 数据
"""
import os
import sys
from datetime import datetime, timedelta

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data_fetch import ConfigManager, UnifiedDataSource, FieldMapper


def example1_basic_usage():
    """示例 1: 基本用法"""
    print("=" * 60)
    print("示例 1: 基本用法 - 从 ClickHouse 获取 ETF 数据")
    print("=" * 60)
    
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
    
    # 设置时间范围（最近 1 个月）
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    config.time.start_date = str(start_date)
    config.time.end_date = str(end_date)
    
    # 创建数据源
    data_source = UnifiedDataSource(config)
    
    # 测试连接
    if data_source.test_connection():
        print("✓ 数据源连接成功\n")
    else:
        print("✗ 数据源连接失败\n")
        return
    
    # 获取 ETF 列表
    print("1. 获取 ETF 列表...")
    df_etf = data_source.get_stock_list()
    print(f"   共 {len(df_etf)} 只 ETF")
    print(f"   前 5 只: {df_etf['order_book_id'].head().tolist()}\n")
    
    # 获取交易日历
    print("2. 获取交易日历...")
    df_calendar = data_source.get_trading_calendar()
    print(f"   共 {len(df_calendar)} 个交易日")
    print(f"   最近 5 日: {df_calendar['trade_date'].tail().tolist()}\n")
    
    # 获取日频数据（前 10 只 ETF）
    print("3. 获取日频数据（前 10 只 ETF）...")
    sample_etfs = df_etf['order_book_id'].head(10).tolist()
    df_daily = data_source.get_all_daily_data(order_book_ids=sample_etfs)
    
    print(f"   数据形状: {df_daily.shape}")
    print(f"   字段数量: {len(df_daily.columns)}")
    print(f"   记录数: {len(df_daily)}")
    print(f"   日期范围: {df_daily['trade_date'].min()} 到 {df_daily['trade_date'].max()}\n")
    
    # 显示部分数据
    print("4. 数据预览:")
    print(df_daily[['order_book_id', 'trade_date', 'open', 'close', 'vol', 'return']].head(10))
    print()


def example2_field_mapping():
    """示例 2: 字段映射"""
    print("=" * 60)
    print("示例 2: 字段映射和标准化")
    print("=" * 60)
    
    import pandas as pd
    
    # 模拟从 ClickHouse 获取的原始数据
    df_raw = pd.DataFrame({
        'Symbol': ['510300', '510500', '159915'],
        'TradingDate': ['2024-11-20', '2024-11-20', '2024-11-20'],
        'OpenPrice': [4.50, 6.80, 3.20],
        'HighPrice': [4.55, 6.90, 3.25],
        'LowPrice': [4.48, 6.75, 3.18],
        'ClosePrice': [4.52, 6.85, 3.22],
        'Volume': [1000000, 500000, 800000],
        'Amount': [4520000, 3425000, 2576000],
        'PE': [12.5, 15.3, 18.2],
        'PB': [1.2, 1.5, 1.8],
    })
    
    print("1. 原始数据（ClickHouse 字段）:")
    print(df_raw.head())
    print(f"   字段: {df_raw.columns.tolist()}\n")
    
    # 字段映射
    print("2. 执行字段映射...")
    df_mapped = FieldMapper.map_fields(df_raw, source='clickhouse')
    print(f"   映射后字段: {df_mapped.columns.tolist()}\n")
    
    # 数据类型标准化
    print("3. 数据类型标准化...")
    df_standard = FieldMapper.standardize_data_types(df_mapped)
    print("   数据类型:")
    print(df_standard.dtypes)
    print()
    
    # 添加衍生字段
    print("4. 添加衍生字段...")
    df_enhanced = FieldMapper.add_derived_fields(df_standard)
    print(f"   增强后字段: {df_enhanced.columns.tolist()}\n")
    
    # 字段验证
    print("5. 字段验证...")
    validation_result = FieldMapper.validate_standard_fields(df_enhanced)
    if validation_result['is_valid']:
        print("   ✓ 字段验证通过")
    else:
        print(f"   ✗ 缺少必需字段: {validation_result['missing_required']}")
    print()


def example3_data_comparison():
    """示例 3: 数据源比较"""
    print("=" * 60)
    print("示例 3: ClickHouse vs 米筐数据源对比")
    print("=" * 60)
    
    # ClickHouse 配置
    config_ch = ConfigManager()
    config_ch.data_source.source = 'clickhouse'
    config_ch.data_source.clickhouse_config = {
        'host': '10.13.66.5',
        'port': 20108,
        'user': 'etf_visitor',
        'password': 'etf_lh_2025',
        'database': 'etf'
    }
    config_ch.time.start_date = '2024-11-01'
    config_ch.time.end_date = '2024-11-20'
    
    # 获取 ClickHouse 数据信息
    ds_ch = UnifiedDataSource(config_ch)
    info_ch = ds_ch.get_data_info()
    
    print("ClickHouse 数据源:")
    print(f"  类型: {info_ch['source_type']}")
    print(f"  数据库: {info_ch.get('database', 'N/A')}")
    print(f"  主机: {info_ch.get('host', 'N/A')}")
    print(f"  时间范围: {info_ch['start_date']} 到 {info_ch['end_date']}")
    
    # 获取少量数据进行比较
    df_ch = ds_ch.get_all_daily_data()
    if not df_ch.empty:
        print(f"  数据量: {len(df_ch)} 条")
        print(f"  字段数: {len(df_ch.columns)}")
        print(f"  股票数: {df_ch['order_book_id'].nunique()}")
        print(f"  主要字段: {df_ch.columns[:10].tolist()}")
    
    print()
    print("注：两个数据源获取的数据都已标准化到相同的字段格式")
    print("    可以无缝切换数据源而不需要修改后续分析代码")
    print()


def example4_save_data():
    """示例 4: 保存数据"""
    print("=" * 60)
    print("示例 4: 获取并保存 ETF 数据")
    print("=" * 60)
    
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
    
    # 最近 3 个月
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=90)
    config.time.start_date = str(start_date)
    config.time.end_date = str(end_date)
    
    # 配置存储
    config.storage.save_dir = 'output/etf_data'
    config.storage.file_format = 'parquet'
    config.storage.compression = 'snappy'
    
    # 获取数据
    data_source = UnifiedDataSource(config)
    
    print(f"获取数据: {start_date} 到 {end_date}")
    df = data_source.get_all_daily_data()
    
    if not df.empty:
        print(f"✓ 数据获取成功: {len(df)} 条记录")
        print(f"  ETF 数量: {df['order_book_id'].nunique()}")
        print(f"  交易日数: {df['trade_date'].nunique()}")
        
        # 保存数据
        print("\n保存数据...")
        data_source.save_data(df, 'daily_data', 'etf_3months')
        print(f"✓ 数据已保存到: {config.storage.get_full_path('daily_data', 'etf_3months.parquet')}")
    else:
        print("✗ 未获取到数据")
    
    print()


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("ClickHouse 数据源集成示例")
    print("=" * 60 + "\n")
    
    try:
        # 示例 1: 基本用法
        example1_basic_usage()
        
        # 示例 2: 字段映射
        example2_field_mapping()
        
        # 示例 3: 数据源比较
        example3_data_comparison()
        
        # 示例 4: 保存数据
        # example4_save_data()  # 取消注释以运行
        
        print("=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
