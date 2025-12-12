#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 YAML 配置文件运行 VAE 因子挖掘

示例1: 使用预定义模板
    python run_vae_from_config.py --config templates/vae_basic.yaml

示例2: 使用自定义配置
    python run_vae_from_config.py --config my_vae_config.yaml --exp my_experiment

示例3: 批量运行（超参数搜索）
    for latent in 8 16 32; do
        python run_vae_from_config.py --config templates/vae_basic.yaml --exp vae_latent${latent} --latent-dim $latent
    done
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from quantclassic.config import ConfigLoader, TaskRunner


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用配置文件运行 VAE 因子挖掘')
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='配置文件路径 (如: templates/vae_basic.yaml)'
    )
    
    parser.add_argument(
        '--exp', '-e',
        type=str,
        default=None,
        help='实验名称 (覆盖配置文件中的 experiment_name)'
    )
    
    # 可选：覆盖配置文件中的参数
    parser.add_argument('--latent-dim', type=int, help='潜在空间维度')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--n-epochs', type=int, help='训练轮数')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='设备')
    
    return parser.parse_args()


def override_config(config: dict, args: argparse.Namespace) -> dict:
    """根据命令行参数覆盖配置"""
    
    # 覆盖实验名称
    if args.exp:
        config['experiment_name'] = args.exp
    
    # 获取模型配置
    if 'task' not in config or 'model' not in config['task']:
        print("警告: 配置文件中没有 task.model 部分")
        return config
    
    model_kwargs = config['task']['model']['kwargs']
    
    # 覆盖模型参数
    if args.latent_dim:
        model_kwargs['latent_dim'] = args.latent_dim
        print(f"✏️  覆盖 latent_dim = {args.latent_dim}")
    
    if args.batch_size:
        model_kwargs['batch_size'] = args.batch_size
        # 同时更新 dataset 的 batch_size
        if 'dataset' in config['task']:
            config['task']['dataset']['kwargs']['config']['batch_size'] = args.batch_size
        print(f"✏️  覆盖 batch_size = {args.batch_size}")
    
    if args.lr:
        model_kwargs['lr'] = args.lr
        print(f"✏️  覆盖 lr = {args.lr}")
    
    if args.n_epochs:
        model_kwargs['n_epochs'] = args.n_epochs
        print(f"✏️  覆盖 n_epochs = {args.n_epochs}")
    
    if args.device:
        model_kwargs['device'] = args.device
        print(f"✏️  覆盖 device = {args.device}")
    
    return config


def main():
    """主函数"""
    args = parse_args()
    
    # 检查配置文件是否存在
    config_path = Path(args.config)
    if not config_path.exists():
        # 尝试相对于 config 目录查找
        config_path = Path(__file__).parent.parent / args.config
    
    if not config_path.exists():
        print(f"❌ 错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    print("=" * 80)
    print("🚀 QuantClassic VAE 配置驱动运行")
    print("=" * 80)
    print(f"📄 配置文件: {config_path}")
    
    # 加载配置
    try:
        config = ConfigLoader.load(str(config_path))
        print(f"✅ 配置加载成功")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 覆盖配置
    config = override_config(config, args)
    
    # 显示配置摘要
    print("\n" + "-" * 80)
    print("📋 配置摘要:")
    print("-" * 80)
    print(f"  实验名称: {config.get('experiment_name', 'N/A')}")
    
    if 'task' in config and 'model' in config['task']:
        model_config = config['task']['model']['kwargs']
        print(f"  潜在维度: {model_config.get('latent_dim', 'N/A')}")
        print(f"  批次大小: {model_config.get('batch_size', 'N/A')}")
        print(f"  学习率: {model_config.get('lr', 'N/A')}")
        print(f"  训练轮数: {model_config.get('n_epochs', 'N/A')}")
        print(f"  设备: {model_config.get('device', 'N/A')}")
    
    if 'task' in config and 'dataset' in config['task']:
        dataset_config = config['task']['dataset']['kwargs']['config']
        print(f"  窗口大小: {dataset_config.get('window_size', 'N/A')}")
        print(f"  数据划分: {dataset_config.get('train_ratio', 'N/A')}/{dataset_config.get('val_ratio', 'N/A')}/{dataset_config.get('test_ratio', 'N/A')}")
    
    print("-" * 80)
    
    # 运行任务
    print("\n🏃 开始运行任务...")
    print("=" * 80)
    
    try:
        runner = TaskRunner()
        results = runner.run(
            config,
            experiment_name=config.get('experiment_name', 'vae_exp')
        )
        
        print("\n" + "=" * 80)
        print("🎉 任务完成!")
        print("=" * 80)
        
        # 显示结果
        if 'metrics' in results:
            print("\n📊 关键指标:")
            print("-" * 80)
            metrics = results['metrics']
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        if 'model_path' in results:
            print(f"\n💾 模型保存: {results['model_path']}")
        
        if 'factors' in results:
            print(f"\n📈 因子数据: {len(results['factors'])} 行")
        
        print("\n✅ 所有结果已保存到输出目录")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
