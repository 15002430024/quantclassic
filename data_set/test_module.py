"""
DataManager 模块测试脚本

测试所有组件是否正常工作
"""

import sys
import os

# 添加父目录到路径以便导入
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def test_imports():
    """测试所有导入"""
    print("\n" + "=" * 80)
    print("测试 1: 导入测试")
    print("=" * 80)
    
    try:
        from data_set import (
            DataConfig, ConfigTemplates,
            DataLoaderEngine, FeatureEngineer,
            DataSplitter, TimeSeriesSplitter, StratifiedStockSplitter,
            DataValidator, DatasetFactory, DataManager
        )
        print("✅ 所有模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_config():
    """测试配置类"""
    print("\n" + "=" * 80)
    print("测试 2: 配置类")
    print("=" * 80)
    
    try:
        from data_set import DataConfig, ConfigTemplates
        
        # 测试默认配置
        config = DataConfig()
        print(f"✅ 默认配置创建成功")
        
        # 测试自定义配置
        config = DataConfig(window_size=60, batch_size=512)
        assert config.window_size == 60
        assert config.batch_size == 512
        print(f"✅ 自定义配置创建成功")
        
        # 测试配置模板
        config_test = ConfigTemplates.quick_test()
        config_prod = ConfigTemplates.production()
        config_backtest = ConfigTemplates.backtest()
        print(f"✅ 配置模板创建成功")
        
        # 测试配置更新
        config.update(window_size=100)
        assert config.window_size == 100
        print(f"✅ 配置更新成功")
        
        return True
    except Exception as e:
        print(f"❌ 配置类测试失败: {e}")
        return False


def test_components():
    """测试各个组件"""
    print("\n" + "=" * 80)
    print("测试 3: 组件创建")
    print("=" * 80)
    
    try:
        from data_set import (
            DataConfig, DataLoaderEngine, FeatureEngineer,
            DataValidator, DatasetFactory
        )
        
        config = DataConfig()
        
        # 测试各组件创建
        loader = DataLoaderEngine(config)
        print(f"✅ DataLoaderEngine 创建成功")
        
        engineer = FeatureEngineer(config)
        print(f"✅ FeatureEngineer 创建成功")
        
        validator = DataValidator(config)
        print(f"✅ DataValidator 创建成功")
        
        factory = DatasetFactory(config)
        print(f"✅ DatasetFactory 创建成功")
        
        return True
    except Exception as e:
        print(f"❌ 组件创建失败: {e}")
        return False


def test_splitters():
    """测试数据划分器"""
    print("\n" + "=" * 80)
    print("测试 4: 数据划分器")
    print("=" * 80)
    
    try:
        from data_set import (
            DataConfig, TimeSeriesSplitter, 
            StratifiedStockSplitter, RollingWindowSplitter,
            create_splitter
        )
        import pandas as pd
        import numpy as np
        
        # 创建测试数据
        dates = pd.date_range('2020-01-01', periods=100)
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 100,
            'trade_date': dates,
            'y_processed': np.random.randn(100),
            'feature1': np.random.randn(100),
        })
        
        config = DataConfig()
        
        # 测试时间序列划分
        splitter = TimeSeriesSplitter(config)
        train, val, test = splitter.split(df)
        assert len(train) > 0 and len(val) > 0 and len(test) > 0
        print(f"✅ TimeSeriesSplitter 测试成功")
        
        # 测试分层划分
        splitter = StratifiedStockSplitter(config)
        train, val, test = splitter.split(df)
        print(f"✅ StratifiedStockSplitter 测试成功")
        
        # 测试工厂函数
        splitter = create_splitter(config)
        print(f"✅ create_splitter 测试成功")
        
        return True
    except Exception as e:
        print(f"❌ 数据划分器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_manager():
    """测试DataManager"""
    print("\n" + "=" * 80)
    print("测试 5: DataManager")
    print("=" * 80)
    
    try:
        from data_set import DataManager, DataConfig
        
        config = DataConfig()
        manager = DataManager(config)
        print(f"✅ DataManager 创建成功")
        
        # 测试属性访问
        assert hasattr(manager, 'config')
        assert hasattr(manager, 'loader')
        assert hasattr(manager, 'feature_engineer')
        assert hasattr(manager, 'validator')
        assert hasattr(manager, 'factory')
        print(f"✅ DataManager 属性验证成功")
        
        return True
    except Exception as e:
        print(f"❌ DataManager测试失败: {e}")
        return False


def test_with_mock_data():
    """使用模拟数据测试完整流程"""
    print("\n" + "=" * 80)
    print("测试 6: 完整流程（模拟数据）")
    print("=" * 80)
    
    try:
        import pandas as pd
        import numpy as np
        from data_set import DataManager, DataConfig
        import tempfile
        import os
        
        # 创建模拟数据
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200)
        stocks = ['000001.SZ', '000002.SZ']
        
        data = []
        for stock in stocks:
            for date in dates:
                row = {
                    'ts_code': stock,
                    'trade_date': date,
                    'y_processed': np.random.randn(),
                }
                # 添加特征
                for i in range(10):
                    row[f'feature_{i}'] = np.random.randn()
                data.append(row)
        
        df = pd.DataFrame(data)
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = f.name
        
        df.to_parquet(temp_path)
        
        try:
            # 创建配置
            temp_dir = os.path.dirname(temp_path)
            temp_file = os.path.basename(temp_path)
            
            config = DataConfig(
                base_dir=temp_dir,
                data_file=temp_file,
                window_size=20,
                batch_size=32,
                enable_validation=False,
                verbose=False
            )
            
            # 创建管理器
            manager = DataManager(config)
            
            # 运行流水线
            loaders = manager.run_full_pipeline(validate=False)
            
            # 验证结果
            assert loaders is not None
            assert loaders.train is not None
            assert loaders.val is not None
            assert loaders.test is not None
            
            # 测试数据加载
            batch_x, batch_y = next(iter(loaders.train))
            assert batch_x.shape[0] <= 32  # batch_size
            assert batch_x.shape[1] == 20  # window_size
            assert batch_x.shape[2] == 10  # num_features
            
            print(f"✅ 完整流程测试成功")
            print(f"   - 训练集: {len(manager.datasets.train)} 样本")
            print(f"   - 验证集: {len(manager.datasets.val)} 样本")
            print(f"   - 测试集: {len(manager.datasets.test)} 样本")
            print(f"   - 批次形状: {batch_x.shape}")
            
            return True
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    except Exception as e:
        print(f"❌ 完整流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("=" * 80)
    print("DataManager 模块测试套件")
    print("=" * 80)
    
    tests = [
        ("导入测试", test_imports),
        ("配置类", test_config),
        ("组件创建", test_components),
        ("数据划分器", test_splitters),
        ("DataManager", test_manager),
        ("完整流程", test_with_mock_data),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ 测试 '{name}' 异常: {e}")
            results.append((name, False))
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
    
    print("\n" + "=" * 80)
    print(f"总计: {passed}/{total} 测试通过 ({passed/total*100:.1f}%)")
    print("=" * 80)
    
    if passed == total:
        print("\n🎉 所有测试通过！DataManager 模块工作正常。")
    else:
        print(f"\n⚠️  有 {total-passed} 个测试失败，请检查错误信息。")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
