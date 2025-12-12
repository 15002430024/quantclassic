"""
量化数据获取工具使用示例

演示如何使用工程化的数据获取工具
"""

# ============================================================================
# 示例1: 最简单的使用方式
# ============================================================================
def example_1_basic_usage():
    """基础使用 - 使用默认配置获取数据"""
    print("=" * 80)
    print("示例1: 基础使用")
    print("=" * 80)
    
    from quantclassic.data_fetch.pipeline import QuantDataPipeline
    
    # 创建流水线实例(使用默认配置)
    pipeline = QuantDataPipeline()
    
    # 执行完整流水线
    df = pipeline.run_full_pipeline()
    
    # 查看结果
    print(f"\n数据形状: {df.shape}")
    print(f"股票数量: {df['order_book_id'].nunique()}")
    print(f"特征列数: {len(df.columns)}")
    print(f"\n前5行数据:")
    print(df.head())


# ============================================================================
# 示例2: 使用配置文件
# ============================================================================
def example_2_with_config_file():
    """使用YAML配置文件"""
    print("=" * 80)
    print("示例2: 使用配置文件")
    print("=" * 80)
    
    from quantclassic.data_fetch.pipeline import QuantDataPipeline
    
    # 使用配置文件初始化
    pipeline = QuantDataPipeline(config_path='config.yaml')
    
    # 执行流水线
    df = pipeline.run_full_pipeline()
    
    print(f"数据获取完成: {df.shape}")


# ============================================================================
# 示例3: 自定义配置
# ============================================================================
def example_3_custom_config():
    """自定义配置参数"""
    print("=" * 80)
    print("示例3: 自定义配置")
    print("=" * 80)
    
    from quantclassic.data_fetch.pipeline import QuantDataPipeline
    from quantclassic.data_fetch.config_manager import ConfigManager
    
    # 创建配置管理器
    config = ConfigManager()
    
    # 自定义时间范围
    config.time.start_date = '2020-01-01'
    config.time.end_date = '2024-12-31'
    
    # 自定义股票池
    config.universe.universe_type = 'csi300'  # 使用沪深300
    
    # 自定义特征
    config.feature.lag_periods = [1, 5, 10, 20]
    config.feature.ma_windows = [5, 20, 60]
    
    # 自定义存储
    config.storage.save_dir = 'my_data'
    config.storage.file_format = 'parquet'
    
    # 创建流水线
    pipeline = QuantDataPipeline(config=config)
    
    # 执行
    df = pipeline.run_full_pipeline()
    
    print(f"自定义配置完成: {df.shape}")


# ============================================================================
# 示例4: 自定义股票池
# ============================================================================
def example_4_custom_stocks():
    """使用自定义股票列表"""
    print("=" * 80)
    print("示例4: 自定义股票池")
    print("=" * 80)
    
    from quantclassic.data_fetch.pipeline import QuantDataPipeline
    
    # 指定特定股票
    my_stocks = [
        '000001.XSHE',  # 平安银行
        '600000.XSHG',  # 浦发银行
        '000858.XSHE',  # 五粮液
        '600519.XSHG',  # 贵州茅台
    ]
    
    pipeline = QuantDataPipeline()
    pipeline.run_custom_universe(my_stocks)
    
    print("自定义股票池处理完成")


# ============================================================================
# 示例5: 分步执行
# ============================================================================
def example_5_step_by_step():
    """分步执行流水线"""
    print("=" * 80)
    print("示例5: 分步执行")
    print("=" * 80)
    
    from quantclassic.data_fetch.pipeline import QuantDataPipeline
    
    pipeline = QuantDataPipeline()
    
    # 第一步: 只获取基础数据
    print("\n步骤1: 获取基础数据...")
    pipeline.run_full_pipeline(
        steps=['fetch_basic'],
        save_intermediate=True,
        validate=False
    )
    
    # 第二步: 获取日频数据
    print("\n步骤2: 获取日频数据...")
    pipeline.run_full_pipeline(
        steps=['fetch_daily'],
        save_intermediate=True,
        validate=False
    )
    
    # 第三步: 数据处理和特征工程
    print("\n步骤3: 数据处理...")
    df = pipeline.run_full_pipeline(
        steps=['merge', 'features', 'validate', 'save']
    )
    
    print(f"\n分步执行完成: {df.shape}")


# ============================================================================
# 示例6: 增量更新
# ============================================================================
def example_6_incremental_update():
    """增量更新数据"""
    print("=" * 80)
    print("示例6: 增量更新")
    print("=" * 80)
    
    from quantclassic.data_fetch.pipeline import QuantDataPipeline
    from datetime import datetime, timedelta
    
    pipeline = QuantDataPipeline()
    
    # 更新昨天的数据
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"增量更新日期: {yesterday}")
    pipeline.run_incremental_update(yesterday)
    
    print("增量更新完成")


# ============================================================================
# 示例7: 加载已有数据
# ============================================================================
def example_7_load_existing():
    """加载已保存的数据"""
    print("=" * 80)
    print("示例7: 加载已有数据")
    print("=" * 80)
    
    from quantclassic.data_fetch.pipeline import QuantDataPipeline
    
    pipeline = QuantDataPipeline()
    
    try:
        # 加载已保存的数据
        df = pipeline.load_existing_data()
        
        # 查看数据摘要
        summary = pipeline.get_data_summary()
        
        print("\n数据摘要:")
        print(f"  - 数据形状: {summary['shape']}")
        print(f"  - 股票数量: {summary['stocks']}")
        print(f"  - 日期范围: {summary['date_range']}")
        print(f"  - 特征数量: {summary['features']}")
        print(f"  - 缺失率: {summary['missing_ratio']:.2%}")
        print(f"  - 内存占用: {summary['memory_usage']}")
        
    except FileNotFoundError:
        print("未找到已保存的数据,请先运行完整流水线")


# ============================================================================
# 示例8: 数据验证
# ============================================================================
def example_8_validation():
    """数据质量验证"""
    print("=" * 80)
    print("示例8: 数据验证")
    print("=" * 80)
    
    from quantclassic.data_fetch.pipeline import QuantDataPipeline
    
    pipeline = QuantDataPipeline()
    
    # 执行流水线并验证
    df = pipeline.run_full_pipeline(validate=True)
    
    # 验证报告保存在: rq_data_parquet/data_quality_report.txt
    print("\n数据验证完成,详细报告已保存")
    
    # 也可以单独验证
    from quantclassic.data_fetch.data_validator import DataValidator
    from quantclassic.data_fetch.config_manager import ConfigManager
    
    config = ConfigManager()
    validator = DataValidator(config)
    
    # 运行各项验证
    validator.validate_data_integrity(df)
    validator.check_data_leakage(df)
    validator.check_data_consistency(df)
    
    # 生成报告
    report = validator.generate_quality_report()
    print(f"\n验证状态: {report['overall_status']}")


# ============================================================================
# 示例9: 自定义数据处理
# ============================================================================
def example_9_custom_processing():
    """自定义数据处理流程"""
    print("=" * 80)
    print("示例9: 自定义数据处理")
    print("=" * 80)
    
    from quantclassic.data_fetch.pipeline import QuantDataPipeline
    from quantclassic.data_fetch.data_processor import DataProcessor
    from quantclassic.data_fetch.config_manager import ConfigManager
    
    # 继承DataProcessor并添加自定义处理
    class MyDataProcessor(DataProcessor):
        def create_custom_features(self, df):
            """自定义特征"""
            # 添加自己的特征计算逻辑
            df['my_custom_feature'] = df.groupby('order_book_id')['close'].transform(
                lambda x: x.rolling(10).mean() / x.rolling(20).mean()
            )
            return df
    
    # 使用自定义处理器
    config = ConfigManager()
    pipeline = QuantDataPipeline(config=config)
    
    # 替换处理器
    pipeline.processor = MyDataProcessor(config)
    
    # 执行
    df = pipeline.run_full_pipeline()
    
    print(f"自定义处理完成: {df.shape}")
    print(f"包含自定义特征: {'my_custom_feature' in df.columns}")


# ============================================================================
# 示例10: 保存配置
# ============================================================================
def example_10_save_config():
    """保存自定义配置到文件"""
    print("=" * 80)
    print("示例10: 保存配置")
    print("=" * 80)
    
    from quantclassic.data_fetch.config_manager import ConfigManager
    
    # 创建配置
    config = ConfigManager()
    
    # 修改配置
    config.time.start_date = '2018-01-01'
    config.time.end_date = '2024-12-31'
    config.universe.universe_type = 'csi500'
    config.feature.lag_periods = [1, 3, 5, 10, 20, 60]
    
    # 保存到文件
    config.save_to_yaml('my_custom_config.yaml')
    
    print("配置已保存到: my_custom_config.yaml")
    
    # 之后可以加载使用
    from quantclassic.data_fetch.pipeline import QuantDataPipeline
    
    pipeline = QuantDataPipeline(config_path='my_custom_config.yaml')
    print("配置加载成功")


# ============================================================================
# 示例11: 完整的生产环境示例
# ============================================================================
def example_11_production():
    """生产环境完整示例"""
    print("=" * 80)
    print("示例11: 生产环境完整示例")
    print("=" * 80)
    
    import logging
    from datetime import datetime
    from quantclassic.data_fetch.pipeline import QuantDataPipeline
    from quantclassic.data_fetch.config_manager import ConfigManager
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'data_pipeline_{datetime.now():%Y%m%d}.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        # 加载配置
        config = ConfigManager(config_path='config.yaml')
        
        # 添加时间戳到保存目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config.storage.save_dir = f'data/version_{timestamp}'
        
        # 创建流水线
        pipeline = QuantDataPipeline(config=config)
        
        # 执行完整流水线
        logging.info("开始数据获取流程...")
        df = pipeline.run_full_pipeline(
            save_intermediate=True,
            validate=True
        )
        
        # 获取数据摘要
        summary = pipeline.get_data_summary()
        
        # 检查数据质量
        if summary['missing_ratio'] > 0.1:
            logging.warning(f"警告: 缺失值比例较高 ({summary['missing_ratio']:.2%})")
        
        logging.info(f"数据获取完成: {summary['shape']}")
        logging.info(f"数据保存在: {config.storage.save_dir}")
        
        return df
        
    except Exception as e:
        logging.error(f"数据获取失败: {e}", exc_info=True)
        # 可以在这里添加告警通知
        raise


# ============================================================================
# 主函数 - 运行示例
# ============================================================================
def main():
    """运行所有示例"""
    print("\n" + "=" * 80)
    print("量化数据获取工具 - 使用示例")
    print("=" * 80 + "\n")
    
    # 选择要运行的示例
    examples = {
        '1': ('基础使用', example_1_basic_usage),
        '2': ('使用配置文件', example_2_with_config_file),
        '3': ('自定义配置', example_3_custom_config),
        '4': ('自定义股票池', example_4_custom_stocks),
        '5': ('分步执行', example_5_step_by_step),
        '6': ('增量更新', example_6_incremental_update),
        '7': ('加载已有数据', example_7_load_existing),
        '8': ('数据验证', example_8_validation),
        '9': ('自定义处理', example_9_custom_processing),
        '10': ('保存配置', example_10_save_config),
        '11': ('生产环境示例', example_11_production),
    }
    
    print("可用示例:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    print("\n提示: 直接运行特定示例函数,例如:")
    print("  from example import example_1_basic_usage")
    print("  example_1_basic_usage()")
    
    # 或者运行示例1作为演示
    print("\n" + "=" * 80)
    print("运行示例1作为演示...")
    print("=" * 80 + "\n")
    
    # 注释掉实际执行,避免真的调用API
    # example_1_basic_usage()
    
    print("\n建议:")
    print("1. 先配置米筐API认证")
    print("2. 根据需要修改 config.yaml")
    print("3. 选择合适的示例运行")


if __name__ == '__main__':
    main()
