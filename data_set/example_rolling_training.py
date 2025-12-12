"""
滚动窗口训练快速示例
=====================

展示如何使用 quantclassic 进行滚动窗口模型训练
"""

import sys
sys.path.insert(0, '/home/u2025210237/jupyterlab')

from pathlib import Path
from quantclassic.data_set import DataManager, DataConfig
from quantclassic.model.pytorch_models import GRUModel
from quantclassic.model.model_config import GRUConfig

def main():
    print("=" * 80)
    print("🔄 滚动窗口模型训练示例")
    print("=" * 80)
    
    # ==================== 1. 配置 ====================
    print("\n📝 步骤 1: 创建配置")
    
    # 数据配置 - 使用 rolling 策略
    data_config = DataConfig(
        base_dir='output',
        data_file='train_data_final_01.parquet',
        stock_col='order_book_id',
        time_col='trade_date',
        label_col='alpha_label',
        split_strategy='rolling',        # 关键：使用滚动窗口
        rolling_window_size=252,         # 1年训练窗口
        rolling_step=63,                 # 1季度滚动步长
        window_size=40,
        batch_size=512,
        enable_cache=False
    )
    
    print("✅ 数据配置创建完成")
    print(f"  窗口大小: {data_config.rolling_window_size} 天")
    print(f"  滚动步长: {data_config.rolling_step} 天")
    
    # ==================== 2. 数据准备 ====================
    print("\n📊 步骤 2: 数据准备")
    
    dm = DataManager(config=data_config)
    loaders = dm.run_full_pipeline()
    
    print(f"✅ 数据准备完成")
    print(f"  特征维度: {len(dm.feature_cols)}")
    
    # ==================== 3. 创建训练器 ====================
    print("\n🔧 步骤 3: 创建滚动窗口训练器")
    
    trainer = dm.create_rolling_window_trainer()
    
    if trainer is None:
        raise ValueError("无法创建滚动窗口训练器，请检查配置")
    
    print(f"✅ 训练器创建成功")
    print(f"  窗口数量: {trainer.n_windows}")
    
    # ==================== 4. 模型配置 ====================
    print("\n⚙️  步骤 4: 配置模型")
    
    gru_config = GRUConfig(
        d_feat=len(dm.feature_cols),
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        n_epochs=30,
        batch_size=512,
        learning_rate=0.001,
        early_stop=10,
        optimizer='adam',
        device='cuda'
    )
    
    print("✅ 模型配置完成")
    print(f"  模型: GRU")
    print(f"  隐藏层: {gru_config.hidden_size}")
    print(f"  层数: {gru_config.num_layers}")
    
    # ==================== 5. 训练所有窗口 ====================
    print("\n🚀 步骤 5: 训练所有窗口")
    print(f"⚠️  注意: 将训练 {trainer.n_windows} 个独立模型，需要较长时间")
    
    # 确认是否继续
    response = input("\n是否继续? (y/n): ")
    
    if response.lower() != 'y':
        print("❌ 已取消训练")
        return
    
    results = trainer.train_all_windows(
        model_class=GRUModel,
        model_config=gru_config,
        save_dir='output/rolling_models',
        val_ratio=0.2,
        incremental=False  # False=独立训练，True=增量训练
    )
    
    print(f"\n✅ 训练完成")
    
    # ==================== 6. 预测 ====================
    print("\n🔮 步骤 6: 预测所有窗口")
    
    predictions = trainer.predict_all_windows(results)
    
    # ==================== 7. 保存结果 ====================
    print("\n💾 步骤 7: 保存结果")
    
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    predictions.to_parquet('output/rolling_predictions.parquet')
    print(f"✅ 预测结果已保存: output/rolling_predictions.parquet")
    
    # ==================== 8. 分析结果 ====================
    print("\n📈 步骤 8: 结果分析")
    print("=" * 80)
    
    # 训练汇总
    summary = trainer.get_summary()
    
    print("\n【训练汇总】")
    print(f"  窗口数量: {summary['n_windows']}")
    print(f"  平均训练损失: {summary['avg_train_loss']:.6f}")
    print(f"  平均验证损失: {summary['avg_val_loss']:.6f}")
    print(f"  平均最佳Epoch: {summary['avg_best_epoch']:.1f}")
    
    # 预测汇总
    print("\n【预测汇总】")
    print(f"  总预测样本: {len(predictions):,}")
    print(f"  时间范围: {predictions[data_config.time_col].min()} ~ {predictions[data_config.time_col].max()}")
    print(f"  股票数量: {predictions[data_config.stock_col].nunique()}")
    
    # IC分析
    print("\n【IC分析】")
    from scipy.stats import pearsonr, spearmanr
    import numpy as np
    
    pred_values = predictions['pred_alpha'].values
    label_values = predictions[data_config.label_col].values
    
    overall_ic, _ = pearsonr(pred_values, label_values)
    overall_rank_ic, _ = spearmanr(pred_values, label_values)
    
    print(f"  总体 Pearson IC: {overall_ic:.4f}")
    print(f"  总体 Spearman IC: {overall_rank_ic:.4f}")
    
    # 各窗口IC
    print("\n  各窗口IC:")
    window_ics = []
    for window_idx in sorted(predictions['window_idx'].unique()):
        window_data = predictions[predictions['window_idx'] == window_idx]
        window_ic, _ = pearsonr(
            window_data['pred_alpha'].values,
            window_data[data_config.label_col].values
        )
        window_ics.append(window_ic)
        print(f"    窗口 {window_idx:2d}: IC={window_ic:7.4f}")
    
    # IC稳定性
    print("\n  IC稳定性:")
    print(f"    平均IC: {np.mean(window_ics):.4f}")
    print(f"    IC标准差: {np.std(window_ics):.4f}")
    print(f"    IC胜率: {np.mean(np.array(window_ics) > 0):.2%}")
    
    print("\n" + "=" * 80)
    print("✅ 滚动窗口训练流程完成！")
    print("=" * 80)
    
    print("\n💡 下一步:")
    print("  1. 查看 output/rolling_models/ 目录中的各窗口模型")
    print("  2. 对比不同窗口的IC表现，分析市场环境影响")
    print("  3. 使用 backtest 进行更深入的因子分析")
    print("  4. 参考文档: quantclassic/data_set/ROLLING_WINDOW_GUIDE.md")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断训练")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
