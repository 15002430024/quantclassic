"""
DataManager 使用示例

演示DataManager模块的各种使用场景
"""

import sys
sys.path.append('..')

from data_set import DataManager, DataConfig, ConfigTemplates
import torch


def example_1_quick_start():
    """示例1: 快速开始 - 使用默认配置"""
    print("\n" + "=" * 80)
    print("示例 1: 快速开始")
    print("=" * 80)
    
    # 创建默认配置
    config = DataConfig()
    
    # 创建管理器
    manager = DataManager(config)
    
    # 运行完整流水线
    loaders = manager.run_full_pipeline()
    
    # 使用数据加载器
    for batch_x, batch_y in loaders.train:
        print(f"训练批次: X={batch_x.shape}, Y={batch_y.shape}")
        break
    
    return manager


def example_2_custom_config():
    """示例2: 自定义配置"""
    print("\n" + "=" * 80)
    print("示例 2: 自定义配置")
    print("=" * 80)
    
    # 自定义配置
    config = DataConfig(
        base_dir='rq_data_parquet',
        data_file='train_data_final.parquet',
        window_size=60,  # 60天窗口
        batch_size=512,
        split_strategy='stratified',  # 分层划分
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        enable_validation=True,
        verbose=True
    )
    
    # 创建管理器并运行
    manager = DataManager(config)
    loaders = manager.run_full_pipeline()
    
    return manager


def example_3_step_by_step():
    """示例3: 逐步执行流水线"""
    print("\n" + "=" * 80)
    print("示例 3: 逐步执行流水线")
    print("=" * 80)
    
    config = DataConfig()
    manager = DataManager(config)
    
    # 步骤1: 加载数据
    print("\n[步骤 1] 加载数据...")
    raw_data = manager.load_raw_data()
    print(f"加载完成: {len(raw_data):,} 行")
    
    # 步骤2: 验证数据
    print("\n[步骤 2] 验证数据...")
    report = manager.validate_data_quality()
    print(f"验证结果: {'通过' if report.is_valid else '失败'}")
    
    # 步骤3: 特征工程
    print("\n[步骤 3] 特征工程...")
    feature_cols = manager.preprocess_features(auto_filter=True)
    print(f"特征数量: {len(feature_cols)}")
    
    # 步骤4-5: 创建数据集
    print("\n[步骤 4-5] 创建数据集...")
    datasets = manager.create_datasets()
    print(f"训练集: {len(datasets.train):,} 样本")
    print(f"验证集: {len(datasets.val):,} 样本")
    print(f"测试集: {len(datasets.test):,} 样本")
    
    # 步骤6: 创建数据加载器
    print("\n[步骤 6] 创建数据加载器...")
    loaders = manager.get_dataloaders(batch_size=256)
    
    return manager


def example_4_different_split_strategies():
    """示例4: 不同的数据划分策略"""
    print("\n" + "=" * 80)
    print("示例 4: 不同的数据划分策略")
    print("=" * 80)
    
    # 4.1 时间序列划分
    print("\n4.1 时间序列划分:")
    config_ts = DataConfig(split_strategy='time_series')
    manager_ts = DataManager(config_ts)
    manager_ts.load_raw_data()
    manager_ts.preprocess_features()
    datasets_ts = manager_ts.create_datasets()
    
    # 4.2 分层划分
    print("\n4.2 分层股票划分:")
    config_strat = DataConfig(split_strategy='stratified')
    manager_strat = DataManager(config_strat)
    manager_strat.load_raw_data()
    manager_strat.preprocess_features()
    datasets_strat = manager_strat.create_datasets()
    
    # 4.3 滚动窗口划分
    print("\n4.3 滚动窗口划分:")
    config_roll = DataConfig(
        split_strategy='rolling',
        rolling_window_size=252,  # 一年
        rolling_step=63  # 一个季度
    )
    # 注意: 滚动窗口返回多个划分，不直接支持DatasetCollection


def example_5_config_templates():
    """示例5: 使用配置模板"""
    print("\n" + "=" * 80)
    print("示例 5: 使用配置模板")
    print("=" * 80)
    
    # 5.1 快速测试模板
    print("\n5.1 快速测试模板:")
    config_test = ConfigTemplates.quick_test()
    manager_test = DataManager(config_test)
    
    # 5.2 生产环境模板
    print("\n5.2 生产环境模板:")
    config_prod = ConfigTemplates.production()
    manager_prod = DataManager(config_prod)
    
    # 5.3 回测模板
    print("\n5.3 回测模板:")
    config_backtest = ConfigTemplates.backtest()
    manager_backtest = DataManager(config_backtest)
    
    print("\n所有模板创建成功!")


def example_6_save_and_load_state():
    """示例6: 保存和加载状态"""
    print("\n" + "=" * 80)
    print("示例 6: 保存和加载状态")
    print("=" * 80)
    
    # 创建并处理数据
    config = DataConfig()
    manager = DataManager(config)
    manager.load_raw_data()
    manager.preprocess_features()
    manager.create_datasets()
    
    # 保存状态
    print("\n保存状态...")
    manager.save_state('cache/my_manager_state.pkl')
    
    # 创建新管理器并加载状态
    print("\n加载状态...")
    new_manager = DataManager()
    new_manager.load_state('cache/my_manager_state.pkl')
    
    print(f"特征数量: {len(new_manager.feature_cols)}")


def example_7_integrate_with_training():
    """示例7: 与模型训练集成"""
    print("\n" + "=" * 80)
    print("示例 7: 与模型训练集成")
    print("=" * 80)
    
    # 准备数据
    config = DataConfig(
        window_size=40,
        batch_size=256,
        num_workers=0,
        pin_memory=False
    )
    
    manager = DataManager(config)
    loaders = manager.run_full_pipeline()
    
    # 模拟训练循环
    print("\n模拟训练...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(2):  # 只训练2个epoch作为示例
        print(f"\nEpoch {epoch+1}/2")
        
        for i, (batch_x, batch_y) in enumerate(loaders.train):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # 这里是模型训练代码
            # loss = model(batch_x, batch_y)
            # optimizer.step()
            
            if i >= 2:  # 只展示前3个批次
                break
        
        # 验证
        print(f"验证...")
        for i, (batch_x, batch_y) in enumerate(loaders.val):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # 验证代码
            # val_loss = model.evaluate(batch_x, batch_y)
            
            if i >= 1:  # 只展示前2个批次
                break
    
    print("\n模拟训练完成!")


def example_8_feature_filtering():
    """示例8: 特征过滤"""
    print("\n" + "=" * 80)
    print("示例 8: 特征过滤")
    print("=" * 80)
    
    config = DataConfig()
    manager = DataManager(config)
    
    # 加载数据
    raw_data = manager.load_raw_data()
    
    # 不过滤特征
    print("\n不过滤特征:")
    features_all = manager.feature_engineer.select_features(raw_data)
    print(f"特征数量: {len(features_all)}")
    
    # 过滤特征
    print("\n过滤低质量特征:")
    features_filtered = manager.feature_engineer.filter_features(
        raw_data,
        min_variance=1e-5,
        max_missing_ratio=0.3,
        max_correlation=0.95
    )
    print(f"过滤后特征数量: {len(features_filtered)}")
    print(f"移除了 {len(features_all) - len(features_filtered)} 个特征")


def example_9_access_data():
    """示例9: 访问处理后的数据"""
    print("\n" + "=" * 80)
    print("示例 9: 访问处理后的数据")
    print("=" * 80)
    
    config = DataConfig()
    manager = DataManager(config)
    loaders = manager.run_full_pipeline()
    
    # 访问原始数据
    print("\n原始数据:")
    print(f"  形状: {manager.raw_data.shape}")
    print(f"  列: {list(manager.raw_data.columns[:5])}...")
    
    # 访问特征列
    print("\n特征列:")
    print(f"  数量: {len(manager.feature_cols)}")
    print(f"  前5个: {manager.feature_cols[:5]}")
    
    # 访问划分后的数据
    train_df, val_df, test_df = manager.split_data
    print("\n划分后的数据:")
    print(f"  训练集: {len(train_df):,} 行")
    print(f"  验证集: {len(val_df):,} 行")
    print(f"  测试集: {len(test_df):,} 行")
    
    # 访问数据集元数据
    print("\n数据集元数据:")
    for key, value in manager.datasets.metadata.items():
        print(f"  {key}: {value}")


def example_10_yaml_config():
    """示例10: 使用YAML配置文件"""
    print("\n" + "=" * 80)
    print("示例 10: 使用YAML配置文件")
    print("=" * 80)
    
    # 创建配置并保存为YAML
    config = DataConfig(
        window_size=60,
        batch_size=512,
        split_strategy='stratified'
    )
    
    yaml_path = 'cache/my_config.yaml'
    config.to_yaml(yaml_path)
    print(f"\n配置已保存到: {yaml_path}")
    
    # 从YAML加载配置
    loaded_config = DataConfig.from_yaml(yaml_path)
    print(f"\n从YAML加载的配置:")
    print(f"  window_size: {loaded_config.window_size}")
    print(f"  batch_size: {loaded_config.batch_size}")
    print(f"  split_strategy: {loaded_config.split_strategy}")


def main():
    """运行所有示例"""
    print("=" * 80)
    print("DataManager 使用示例集")
    print("=" * 80)
    
    examples = [
        ("快速开始", example_1_quick_start),
        ("自定义配置", example_2_custom_config),
        ("逐步执行", example_3_step_by_step),
        ("不同划分策略", example_4_different_split_strategies),
        ("配置模板", example_5_config_templates),
        ("保存和加载状态", example_6_save_and_load_state),
        ("与训练集成", example_7_integrate_with_training),
        ("特征过滤", example_8_feature_filtering),
        ("访问数据", example_9_access_data),
        ("YAML配置", example_10_yaml_config),
    ]
    
    print("\n可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n选择要运行的示例编号（1-10），或输入 'all' 运行所有示例：")
    choice = input("> ").strip()
    
    if choice.lower() == 'all':
        for name, example_func in examples:
            try:
                example_func()
            except FileNotFoundError:
                print(f"\n⚠️  {name}: 数据文件不存在，跳过")
            except Exception as e:
                print(f"\n❌ {name}: 发生错误: {e}")
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                name, example_func = examples[idx]
                example_func()
            else:
                print("无效的选择")
        except ValueError:
            print("请输入有效的数字")
        except FileNotFoundError:
            print("\n⚠️  数据文件不存在")
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")


if __name__ == '__main__':
    # 如果数据文件存在，运行示例1
    try:
        print("运行快速开始示例...")
        example_1_quick_start()
    except FileNotFoundError:
        print("\n⚠️  数据文件不存在")
        print("✅ 示例代码已就绪，请确保数据文件存在后运行")
