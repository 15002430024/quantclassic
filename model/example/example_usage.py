"""
QuantClassic 模型使用完整示例

展示如何使用模型基类系统进行端到端的训练和预测
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from pathlib import Path

# 导入自定义模块
from model import LSTMModel, GRUModel, TransformerModel, ModelFactory
from data_set import DataManager, DataConfig


def example_1_basic_usage():
    """示例 1: 基础使用 - 直接创建模型"""
    print("=" * 80)
    print("示例 1: 基础使用 - 直接创建和训练模型")
    print("=" * 80)
    
    # 1. 创建模型
    model = LSTMModel(
        d_feat=20,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        n_epochs=50,
        lr=0.001,
        batch_size=256,
        early_stop=10
    )
    
    print("\n✅ LSTM 模型创建成功")
    print(f"   设备: {model.device}")
    print(f"   参数量: {sum(p.numel() for p in model.model.parameters()):,}")
    
    # 2. 准备数据（使用 DataManager）
    config = DataConfig(
        base_dir='../rq_data_parquet',
        data_file='train_data_final.parquet',
        window_size=20,
        batch_size=256,
        split_strategy='time_series',
        enable_validation=False,
        verbose=False
    )
    
    try:
        manager = DataManager(config)
        loaders = manager.run_full_pipeline()
        
        print("\n✅ 数据加载完成")
        print(f"   训练集批次数: {len(loaders.train)}")
        print(f"   验证集批次数: {len(loaders.val)}")
        
        # 3. 训练模型
        print("\n开始训练...")
        model.fit(
            train_loader=loaders.train,
            valid_loader=loaders.val,
            save_path='../output/models/lstm_model.pth'
        )
        
        # 4. 预测
        print("\n进行预测...")
        predictions = model.predict(loaders.test)
        print(f"✅ 预测完成: {predictions.shape}")
        
        # 5. 保存结果
        pd.Series(predictions).to_csv('../output/predictions.csv', index=False)
        print("✅ 预测结果已保存")
        
    except FileNotFoundError:
        print("\n⚠️  数据文件不存在，跳过训练步骤")
        print("✅ 模型创建和接口测试完成")


def example_2_config_driven():
    """示例 2: 配置驱动 - 使用配置文件创建模型"""
    print("\n" + "=" * 80)
    print("示例 2: 配置驱动 - 从配置字典创建模型")
    print("=" * 80)
    
    # 模型配置
    model_config = {
        'class': 'LSTM',
        'kwargs': {
            'd_feat': 20,
            'hidden_size': 128,
            'num_layers': 3,
            'dropout': 0.2,
            'n_epochs': 100,
            'lr': 0.0005,
            'batch_size': 512,
            'early_stop': 15,
            'optimizer': 'adam',
            'loss_fn': 'mse'
        }
    }
    
    # 使用工厂创建模型
    model = ModelFactory.create_model(model_config)
    
    print(f"\n✅ 从配置创建模型: {model.__class__.__name__}")
    print(f"   隐藏层大小: {model.hidden_size}")
    print(f"   层数: {model.num_layers}")
    print(f"   学习率: {model.lr}")


def example_3_compare_models():
    """示例 3: 模型对比 - 比较多个模型"""
    print("\n" + "=" * 80)
    print("示例 3: 模型对比 - 训练和比较多个模型")
    print("=" * 80)
    
    # 定义多个模型配置
    model_configs = {
        'LSTM': {
            'class': 'LSTM',
            'kwargs': {
                'd_feat': 20,
                'hidden_size': 64,
                'num_layers': 2,
                'n_epochs': 30,
                'lr': 0.001
            }
        },
        'GRU': {
            'class': 'GRU',
            'kwargs': {
                'd_feat': 20,
                'hidden_size': 64,
                'num_layers': 2,
                'n_epochs': 30,
                'lr': 0.001
            }
        },
        'Transformer': {
            'class': 'Transformer',
            'kwargs': {
                'd_feat': 20,
                'd_model': 64,
                'nhead': 4,
                'num_layers': 2,
                'n_epochs': 30,
                'lr': 0.001
            }
        }
    }
    
    # 创建所有模型
    models = {}
    for name, config in model_configs.items():
        models[name] = ModelFactory.create_model(config)
        print(f"✅ 创建模型: {name}")
    
    print(f"\n已创建 {len(models)} 个模型，准备对比测试")
    
    # 如果有数据，可以训练和对比
    try:
        config = DataConfig(
            base_dir='../rq_data_parquet',
            data_file='train_data_final.parquet',
            window_size=20,
            batch_size=256,
            verbose=False
        )
        
        manager = DataManager(config)
        loaders = manager.run_full_pipeline()
        
        results = {}
        for name, model in models.items():
            print(f"\n训练 {name}...")
            model.fit(loaders.train, loaders.val)
            
            # 记录结果
            results[name] = {
                'best_epoch': model.best_epoch,
                'best_score': model.best_score,
                'train_loss': model.train_losses[-1] if model.train_losses else None,
                'valid_loss': model.valid_losses[-1] if model.valid_losses else None
            }
        
        # 打印对比结果
        print("\n" + "=" * 80)
        print("模型对比结果")
        print("=" * 80)
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"  最佳 Epoch: {result['best_epoch']}")
            print(f"  最佳分数: {result['best_score']:.6f}")
            print(f"  最终训练损失: {result['train_loss']:.6f}")
            print(f"  最终验证损失: {result['valid_loss']:.6f}")
        
    except FileNotFoundError:
        print("\n⚠️  数据文件不存在，仅展示模型创建")


def example_4_save_load():
    """示例 4: 模型保存和加载"""
    print("\n" + "=" * 80)
    print("示例 4: 模型保存和加载")
    print("=" * 80)
    
    # 创建模型
    model1 = LSTMModel(
        d_feat=20,
        hidden_size=64,
        num_layers=2,
        n_epochs=5
    )
    
    print("✅ 创建原始模型")
    
    # 模拟训练（设置 fitted 标志）
    model1.fitted = True
    model1.best_score = 0.85
    model1.best_epoch = 3
    
    # 保存模型
    save_path = '../output/models/test_model.pth'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model1.save_model(save_path)
    print(f"✅ 模型已保存: {save_path}")
    
    # 加载模型
    model2 = LSTMModel(
        d_feat=20,
        hidden_size=64,
        num_layers=2
    )
    model2.load_model(save_path)
    print(f"✅ 模型已加载")
    print(f"   最佳分数: {model2.best_score}")
    print(f"   最佳 Epoch: {model2.best_epoch}")
    print(f"   已训练: {model2.fitted}")


def example_5_integration_with_factorsystem():
    """示例 5: 与 backtest 集成"""
    print("\n" + "=" * 80)
    print("示例 5: 模型 + DataManager + backtest 完整流程")
    print("=" * 80)
    
    print("""
完整量化研究流程:

1. 数据准备 (DataManager)
   └─> 加载数据、特征工程、数据划分

2. 模型训练 (Model)
   └─> 选择模型、训练、验证、保存

3. 因子生成 (Model.predict)
   └─> 使用训练好的模型生成因子

4. 回测评估 (backtest)
   └─> IC 分析、组合构建、绩效评估

示例代码:
    """)
    
    code = """
# 步骤 1: 准备数据
from data_set import DataManager, DataConfig
config = DataConfig(base_dir='rq_data_parquet')
manager = DataManager(config)
loaders = manager.run_full_pipeline()

# 步骤 2: 训练模型
from model import LSTMModel
model = LSTMModel(d_feat=20, hidden_size=64, n_epochs=100)
model.fit(loaders.train, loaders.val, save_path='output/best_model.pth')

# 步骤 3: 生成因子
predictions = model.predict(loaders.test)

# 步骤 4: 回测
from backtest import FactorBacktestSystem, BacktestConfig
backtest_config = BacktestConfig(output_dir='output/backtest')
system = FactorBacktestSystem(backtest_config)
results = system.run_backtest(df_with_predictions)
    """
    print(code)
    print("✅ 完整流程示例展示完成")


def show_model_architecture():
    """展示模型架构"""
    print("\n" + "=" * 80)
    print("QuantClassic 模型系统架构")
    print("=" * 80)
    
    print("""
quantclassic/model/
├── __init__.py              # 模块导出
├── base_model.py            # 基类定义
│   ├── BaseModel           # 最基础的抽象类
│   ├── Model               # 可训练模型基类
│   ├── PyTorchModel        # PyTorch 模型基类
│   └── FineTunableModel    # 可微调模型基类
│
├── model_factory.py         # 模型工厂
│   ├── ModelRegistry       # 模型注册表
│   ├── register_model()    # 注册装饰器
│   └── ModelFactory        # 工厂类
│
├── pytorch_models.py        # PyTorch 模型实现
│   ├── LSTMModel          # LSTM 模型
│   ├── GRUModel           # GRU 模型
│   └── TransformerModel   # Transformer 模型
│
└── example_usage.py         # 使用示例（本文件）

核心优势:
✅ 统一接口: fit() 和 predict() 标准化
✅ 配置驱动: 支持配置文件创建模型
✅ 自动化: GPU 管理、早停、模型保存
✅ 可扩展: 轻松添加新模型
✅ 兼容性: 与 DataManager、backtest 无缝集成
    """)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    
    print("\n" + "=" * 80)
    print("QuantClassic 模型系统 - 完整使用示例")
    print("=" * 80)
    
    # 展示架构
    show_model_architecture()
    
    # 运行示例
    example_1_basic_usage()
    example_2_config_driven()
    example_3_compare_models()
    example_4_save_load()
    example_5_integration_with_factorsystem()
    
    print("\n" + "=" * 80)
    print("✅ 所有示例运行完成")
    print("=" * 80)
    
    print("""
下一步:
1. 将此模型系统集成到您的研究流程
2. 添加更多自定义模型（继承 PyTorchModel）
3. 创建 YAML 配置文件支持
4. 实现实验管理和结果追踪系统
5. 与 backtest 完整集成进行回测
    """)
