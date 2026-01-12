"""
HybridGraph 模型使用示例

演示如何使用 RNN+Attention+GAT 混合模型进行股票预测。
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model.hybrid_graph_models import HybridGraphModel
from model.utils.adj_matrix_builder import AdjMatrixBuilder


# ==================== 示例 1: 快速开始 ====================

def example_quickstart():
    """快速开始示例"""
    print("\n" + "=" * 80)
    print("示例 1: 快速开始")
    print("=" * 80)
    
    # 1. 准备模拟数据
    batch_size = 128
    seq_len = 40
    d_feat = 20
    n_samples = 1000
    
    # 生成模拟的训练数据
    X_train = torch.randn(n_samples, seq_len, d_feat)
    y_train = torch.randn(n_samples)
    
    X_valid = torch.randn(200, seq_len, d_feat)
    y_valid = torch.randn(200)
    
    # 创建 DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    valid_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False
    )
    
    # 2. 创建模型（不使用图网络）
    model = HybridGraphModel(
        d_feat=d_feat,
        rnn_hidden=64,
        rnn_layers=2,
        rnn_type='lstm',
        use_attention=True,
        use_graph=False,  # 先不使用图网络
        n_epochs=5,
        batch_size=batch_size,
        lr=0.001
    )
    
    # 3. 训练模型
    print("\n开始训练...")
    model.fit(train_loader, valid_loader)
    
    # 4. 预测
    X_test = torch.randn(100, seq_len, d_feat)
    y_test = torch.randn(100)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    predictions = model.predict(test_loader)
    print(f"\n预测结果: shape={predictions.shape}")
    print(f"预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")


# ==================== 示例 2: 使用行业邻接矩阵 ====================

def example_industry_graph():
    """使用行业邻接矩阵示例"""
    print("\n" + "=" * 80)
    print("示例 2: 使用行业邻接矩阵")
    print("=" * 80)
    
    # 1. 构建行业邻接矩阵
    print("\n构建行业邻接矩阵...")
    
    # 假设有 100 只股票，分属 10 个行业
    n_stocks = 100
    n_industries = 10
    industry_codes = [f'Industry_{i % n_industries}' for i in range(n_stocks)]
    
    builder = AdjMatrixBuilder()
    adj_matrix = builder.build_industry_adj(industry_codes, self_loop=True)
    
    # 保存邻接矩阵
    adj_path = '/tmp/industry_adj_matrix.pt'
    builder.save_adj_matrix(adj_matrix, adj_path)
    print(f"邻接矩阵已保存: {adj_path}")
    print(f"邻接矩阵 shape: {adj_matrix.shape}, 边数: {adj_matrix.sum().item()}")
    
    # 2. 准备数据（每个batch包含所有股票的数据）
    batch_size = n_stocks  # 重要：batch_size 必须等于股票数量
    seq_len = 40
    d_feat = 20
    n_days = 20  # 20个交易日
    
    # 生成数据：[n_days, n_stocks, seq_len, d_feat]
    X_train = torch.randn(n_days, n_stocks, seq_len, d_feat)
    y_train = torch.randn(n_days, n_stocks)
    
    # 重塑为 DataLoader 格式：[(n_stocks, seq_len, d_feat), (n_stocks,)]
    train_data = [(X_train[i], y_train[i]) for i in range(n_days)]
    train_loader = train_data  # 简化示例，实际应使用 DataLoader
    
    # 3. 创建模型（使用图网络）
    model = HybridGraphModel(
        d_feat=d_feat,
        rnn_hidden=64,
        rnn_layers=2,
        use_graph=True,
        gat_type='standard',
        gat_hidden=32,
        gat_heads=4,
        adj_matrix_path=adj_path,
        n_epochs=3,
        batch_size=batch_size,
        lr=0.001
    )
    
    print(f"\n模型已创建，使用邻接矩阵: {adj_path}")
    
    # 注意：实际训练需要特殊的 DataLoader，这里仅展示配置


# ==================== 示例 3: 使用相关性邻接矩阵 ====================

def example_correlation_graph():
    """使用相关性邻接矩阵示例"""
    print("\n" + "=" * 80)
    print("示例 3: 使用相关性邻接矩阵")
    print("=" * 80)
    
    # 1. 准备收益率数据
    print("\n准备收益率数据...")
    n_stocks = 50
    n_days = 252  # 一年的交易日
    
    # 生成模拟收益率数据
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.randn(n_days, n_stocks) * 0.02,  # 日收益率，标准差2%
        columns=[f'stock_{i:03d}' for i in range(n_stocks)]
    )
    
    # 2. 构建相关性邻接矩阵
    print("\n构建相关性邻接矩阵...")
    builder = AdjMatrixBuilder()
    
    # 选取相关性最高的 10 个邻居
    adj_matrix = builder.build_correlation_adj(
        returns,
        top_k=10,
        method='pearson',
        self_loop=True
    )
    
    # 保存
    adj_path = '/tmp/correlation_adj_matrix.pt'
    builder.save_adj_matrix(adj_matrix, adj_path)
    print(f"邻接矩阵已保存: {adj_path}")
    print(f"邻接矩阵 shape: {adj_matrix.shape}, 边数: {adj_matrix.sum().item()}")
    
    # 3. 创建模型
    model = HybridGraphModel(
        d_feat=20,
        rnn_hidden=64,
        rnn_layers=2,
        use_graph=True,
        gat_type='correlation',
        gat_hidden=32,
        gat_heads=4,
        top_k_neighbors=10,
        adj_matrix_path=adj_path,
        n_epochs=3,
        batch_size=n_stocks,
        lr=0.001
    )
    
    print(f"\n模型已创建，使用相关性邻接矩阵")


# ==================== 示例 4: 加入基本面数据 ====================

def example_with_fundamentals():
    """加入基本面数据示例"""
    print("\n" + "=" * 80)
    print("示例 4: 加入基本面数据")
    print("=" * 80)
    
    # 1. 准备数据
    batch_size = 128
    seq_len = 40
    d_feat = 20
    funda_dim = 10  # 基本面特征维度（如PE、PB、ROE等）
    n_samples = 1000
    
    # 量价数据
    X_train = torch.randn(n_samples, seq_len, d_feat)
    
    # 基本面数据（每只股票一个向量，不随时间变化）
    F_train = torch.randn(n_samples, funda_dim)
    
    # 标签
    y_train = torch.randn(n_samples)
    
    # 创建 DataLoader（包含3个元素）
    train_dataset = torch.utils.data.TensorDataset(X_train, F_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    # 2. 创建模型（指定基本面维度）
    model = HybridGraphModel(
        d_feat=d_feat,
        funda_dim=funda_dim,  # 关键：指定基本面维度
        rnn_hidden=64,
        rnn_layers=2,
        use_graph=False,  # 可选择是否使用图网络
        n_epochs=5,
        batch_size=batch_size,
        lr=0.001
    )
    
    print(f"\n模型已创建，支持基本面数据 (维度={funda_dim})")
    
    # 3. 训练
    print("\n开始训练...")
    # model.fit(train_loader)  # 实际训练时取消注释


# ==================== 示例 5: 完整工作流 ====================

def example_complete_workflow():
    """完整工作流示例"""
    print("\n" + "=" * 80)
    print("示例 5: 完整工作流")
    print("=" * 80)
    
    # 配置
    n_stocks = 100
    seq_len = 40
    d_feat = 20
    n_train_days = 200
    n_valid_days = 50
    
    print("\n1. 构建邻接矩阵...")
    industry_codes = [f'Ind_{i % 10}' for i in range(n_stocks)]
    builder = AdjMatrixBuilder()
    adj_matrix = builder.build_industry_adj(industry_codes)
    adj_path = '/tmp/complete_adj_matrix.pt'
    builder.save_adj_matrix(adj_matrix, adj_path)
    
    print("\n2. 准备数据...")
    # 训练数据：每天所有股票的数据
    # 注意：这里为了演示简化，实际应该按天组织数据
    train_data = [
        (torch.randn(n_stocks, seq_len, d_feat), torch.randn(n_stocks))
        for _ in range(n_train_days)
    ]
    
    valid_data = [
        (torch.randn(n_stocks, seq_len, d_feat), torch.randn(n_stocks))
        for _ in range(n_valid_days)
    ]
    
    print(f"训练数据: {n_train_days} 天 × {n_stocks} 只股票")
    print(f"验证数据: {n_valid_days} 天 × {n_stocks} 只股票")
    
    print("\n3. 创建模型...")
    model = HybridGraphModel(
        d_feat=d_feat,
        rnn_hidden=64,
        rnn_layers=2,
        rnn_type='lstm',
        use_attention=True,
        use_graph=True,
        gat_type='standard',
        gat_hidden=32,
        gat_heads=4,
        mlp_hidden_sizes=[64, 32],
        dropout=0.3,
        adj_matrix_path=adj_path,
        n_epochs=10,
        lr=0.001,
        early_stop=5
    )
    
    print("\n4. 模型配置:")
    print(f"  - RNN: {model.rnn_type.upper()}, hidden={model.rnn_hidden}, layers={model.rnn_layers}")
    print(f"  - Attention: {model.use_attention}")
    print(f"  - GAT: hidden={model.gat_hidden}, heads={model.gat_heads}")
    print(f"  - MLP: {model.mlp_hidden_sizes}")
    
    print("\n模型已就绪！")
    print("实际训练请调用: model.fit(train_loader, valid_loader)")


# ==================== 主函数 ====================

def main():
    """运行所有示例"""
    print("=" * 80)
    print("HybridGraph 模型使用示例")
    print("=" * 80)
    
    # 运行示例（按需选择）
    example_quickstart()
    # example_industry_graph()
    # example_correlation_graph()
    # example_with_fundamentals()
    # example_complete_workflow()
    
    print("\n" + "=" * 80)
    print("✅ 所有示例完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
