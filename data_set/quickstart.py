"""
DataManager 快速开始脚本

演示最基本的使用方法
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_set import DataManager, DataConfig


def main():
    """快速开始示例"""
    print("=" * 80)
    print("DataManager 快速开始")
    print("=" * 80)
    
    # 步骤1: 创建配置
    print("\n步骤 1: 创建配置")
    config = DataConfig(
        base_dir='../rq_data_parquet',  # 数据目录
        data_file='train_data_final.parquet',  # 数据文件
        window_size=40,  # 时间窗口
        batch_size=256,  # 批量大小
        split_strategy='time_series',  # 划分策略
        verbose=True  # 显示详细信息
    )
    print(f"✅ 配置创建完成")
    print(f"   - 数据路径: {config.data_path}")
    print(f"   - 窗口大小: {config.window_size}")
    print(f"   - 批量大小: {config.batch_size}")
    
    # 步骤2: 创建DataManager
    print("\n步骤 2: 创建DataManager")
    manager = DataManager(config)
    print(f"✅ DataManager创建完成")
    
    # 步骤3: 运行完整流水线
    print("\n步骤 3: 运行完整数据处理流水线")
    print("-" * 80)
    
    try:
        loaders = manager.run_full_pipeline()
        
        # 步骤4: 查看结果
        print("\n步骤 4: 查看处理结果")
        print("-" * 80)
        
        print(f"\n数据集信息:")
        print(f"  训练集: {len(manager.datasets.train):,} 样本")
        print(f"  验证集: {len(manager.datasets.val):,} 样本")
        print(f"  测试集: {len(manager.datasets.test):,} 样本")
        
        print(f"\n特征信息:")
        print(f"  特征数量: {len(manager.feature_cols)}")
        print(f"  窗口大小: {config.window_size}")
        
        print(f"\n数据加载器:")
        print(f"  训练批次数: {len(loaders.train)}")
        print(f"  验证批次数: {len(loaders.val)}")
        print(f"  测试批次数: {len(loaders.test)}")
        
        # 步骤5: 测试数据加载
        print("\n步骤 5: 测试数据加载")
        print("-" * 80)
        
        batch_x, batch_y = next(iter(loaders.train))
        print(f"\n训练批次示例:")
        print(f"  特征张量形状: {batch_x.shape}")
        print(f"  标签张量形状: {batch_y.shape}")
        print(f"  特征维度: [批量大小, 窗口大小, 特征数量]")
        print(f"           = [{batch_x.shape[0]}, {batch_x.shape[1]}, {batch_x.shape[2]}]")
        
        # 成功提示
        print("\n" + "=" * 80)
        print("✅ 快速开始完成！")
        print("=" * 80)
        print("\n现在您可以:")
        print("  1. 使用 loaders.train 训练模型")
        print("  2. 使用 loaders.val 验证模型")
        print("  3. 使用 loaders.test 测试模型")
        print("\n更多示例请查看 examples.py")
        print("详细文档请查看 README.md")
        
        return manager, loaders
        
    except FileNotFoundError as e:
        print("\n" + "=" * 80)
        print("❌ 错误: 数据文件不存在")
        print("=" * 80)
        print(f"\n{e}")
        print("\n请确保:")
        print(f"  1. 数据文件存在: {config.data_path}")
        print(f"  2. 路径配置正确")
        print("\n您可以修改 config.base_dir 和 config.data_file 指向正确的数据文件")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ 发生错误")
        print("=" * 80)
        print(f"\n错误信息: {e}")
        print("\n请检查:")
        print("  1. 数据文件格式是否正确")
        print("  2. 数据是否包含必需的列（ts_code, trade_date, y_processed）")
        print("  3. 内存是否充足")
        
        import traceback
        print("\n详细错误信息:")
        traceback.print_exc()


if __name__ == '__main__':
    main()
