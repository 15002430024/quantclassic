"""
滚动窗口训练快速示例 (v2.0 - 使用新训练器架构)
=====================

展示如何使用 quantclassic 进行滚动窗口模型训练。

⚠️ 更新说明 (2026-01-12):
    - 原 DataManager.create_rolling_window_trainer() 已移除
    - 改用 model.train.RollingDailyTrainer 完成滚动训练
    - 模型配置改用 modular_config 或直接传参
"""

import sys
sys.path.insert(0, '/home/u2025210237/jupyterlab')

from pathlib import Path
from quantclassic.data_set import DataManager, DataConfig
from quantclassic.model import GRUModel
from quantclassic.model.train import RollingDailyTrainer, RollingTrainerConfig


def main():
    print("=" * 80)
    print("🔄 滚动窗口模型训练示例 (v2.0)")
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
    dm.run_full_pipeline()  # 执行数据处理流水线
    
    print(f"✅ 数据准备完成")
    print(f"  特征维度: {len(dm.feature_cols)}")
    
    # ==================== 3. 创建滚动日批次加载器 ====================
    print("\n🔧 步骤 3: 创建滚动日批次加载器")
    
    # 🆕 使用 create_rolling_daily_loaders 获取滚动窗口数据
    rolling_loaders = dm.create_rolling_daily_loaders(val_ratio=0.15)
    
    print(f"✅ 滚动加载器创建成功")
    print(f"  窗口数量: {len(rolling_loaders)}")
    
    # ==================== 4. 定义模型工厂和训练配置 ====================
    print("\n⚙️  步骤 4: 配置模型和训练器")
    
    d_feat = len(dm.feature_cols)
    
    # 🆕 模型工厂函数：每个窗口可创建新模型或复用
    def model_factory():
        return GRUModel(
            d_feat=d_feat,
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
            n_epochs=30,
            lr=0.001,
            early_stop=10,
            device='cuda'
        )
    
    # 🆕 滚动训练配置
    trainer_config = RollingTrainerConfig(
        n_epochs=30,
        lr=0.001,
        early_stop=10,
        weight_inheritance=True,    # 继承上一窗口权重
        reset_optimizer=False,      # 保留优化器状态（动量）
        save_each_window=True,      # 保存每个窗口的模型
        checkpoint_dir='output/rolling_models'
    )
    
    print("✅ 训练配置完成")
    print(f"  模型: GRU")
    print(f"  权重继承: {trainer_config.weight_inheritance}")
    print(f"  保存每窗模型: {trainer_config.save_each_window}")
    
    # ==================== 5. 训练所有窗口 ====================
    print("\n🚀 步骤 5: 训练所有窗口")
    print(f"⚠️  注意: 将训练 {len(rolling_loaders)} 个窗口，需要较长时间")
    
    # 确认是否继续
    response = input("\n是否继续? (y/n): ")
    
    if response.lower() != 'y':
        print("❌ 已取消训练")
        return
    
    # 🆕 创建滚动训练器并训练
    trainer = RollingDailyTrainer(
        model_factory=model_factory,
        config=trainer_config
    )
    
    # 执行滚动训练
    summary = trainer.train(rolling_loaders)
    
    print(f"\n✅ 训练完成")
    print(
        f"  总窗口: {summary['n_windows']} | "
        f"avg_train_loss={summary['avg_train_loss']:.6f} | "
        f"avg_val_loss={summary['avg_val_loss']:.6f}"
    )
    
    # ==================== 6. 汇总预测 ====================
    print("\n🔮 步骤 6: 汇总预测结果")
    
    # 🆕 使用 trainer.get_all_predictions() 获取所有窗口的预测（训练时已自动在测试集上预测）
    predictions = trainer.get_all_predictions()
    if predictions.empty:
        print("⚠️ 无预测结果（测试集可能为空），结束流程")
        return
    
    print(f"✅ 预测结果已汇总: {len(predictions):,} 样本")
    
    # ==================== 7. 保存结果 ====================
    print("\n💾 步骤 7: 保存结果")
    
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    predictions.to_parquet('output/rolling_predictions.parquet')
    print(f"✅ 预测结果已保存: output/rolling_predictions.parquet")
    
    # ==================== 8. 分析结果 ====================
    print("\n📈 步骤 8: 结果分析")
    print("=" * 80)
    
    # 训练汇总（直接使用 summary 中的统计信息）
    print("\n【训练汇总】")
    print(f"  窗口数量: {summary['n_windows']}")
    print(f"  平均训练损失: {summary['avg_train_loss']:.6f}")
    print(f"  平均验证损失: {summary['avg_val_loss']:.6f}")
    
    # 预测汇总
    print("\n【预测汇总】")
    print(f"  总预测样本: {len(predictions):,}")
    
    # 各窗口统计
    print("\n  各窗口样本数:")
    for window_idx in sorted(predictions['window_idx'].unique()):
        window_count = len(predictions[predictions['window_idx'] == window_idx])
        print(f"    窗口 {window_idx:2d}: {window_count:,} 样本")
    
    print("\n" + "=" * 80)
    print("✅ 滚动窗口训练流程完成！")
    print("=" * 80)
    
    print("\n💡 下一步:")
    print("  1. 查看 output/rolling_models/ 目录中的各窗口模型")
    print("  2. 对比不同窗口的损失表现，分析市场环境影响")
    print("  3. 使用 backtest 进行更深入的因子分析")
    print("  4. 参考文档: quantclassic/data_set/update_readme/ROLLING_WINDOW_GUIDE.md")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断训练")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
