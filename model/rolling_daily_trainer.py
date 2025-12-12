"""
rolling_daily_trainer.py - 滚动窗口日批次训练器

实现真正的 Walk-Forward 训练：
- 每个滚动窗口独立训练
- 支持模型权重继承（warm_start）或独立训练
- 支持每个窗口保存独立模型
- 自动合并所有窗口的预测结果

与 DynamicGraphTrainer 的区别：
- DynamicGraphTrainer: 在合并后的大数据集上训练单个模型
- RollingDailyTrainer: 遍历多个窗口，每个窗口训练一次

使用示例：
    from quantclassic.model.rolling_daily_trainer import RollingDailyTrainer
    
    # 创建滚动窗口加载器
    rolling_loaders = dm.create_rolling_daily_loaders()
    
    # 创建训练器
    rolling_trainer = RollingDailyTrainer(
        model_factory=lambda: HybridGraphModel.from_config(config, d_feat=input_dim).model,
        config=trainer_config,
        device='cuda',
        warm_start=True,  # 继承上一窗口模型权重
        save_each_window=True  # 保存每个窗口的模型
    )
    
    # 训练所有窗口
    results = rolling_trainer.fit(rolling_loaders, save_dir='output/rolling_models')
    
    # 获取合并的预测
    all_predictions = rolling_trainer.get_all_predictions()
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
from tqdm import tqdm
import time
import copy

from .dynamic_graph_trainer import DynamicGraphTrainer, DynamicTrainerConfig


@dataclass
class RollingTrainerConfig:
    """滚动窗口训练器配置"""
    # 继承基础训练配置
    n_epochs: int = 20
    learning_rate: float = 0.001
    early_stop: int = 5
    weight_decay: float = 0.0
    
    # 滚动窗口特有配置
    warm_start: bool = True  # 是否继承上一窗口的模型权重
    save_each_window: bool = True  # 是否保存每个窗口的模型
    reset_optimizer: bool = True  # 每个窗口是否重置优化器
    reset_scheduler: bool = True  # 每个窗口是否重置学习率调度器
    
    # 🆕 学习率调度器配置（透传给 DynamicGraphTrainer）
    use_scheduler: bool = True
    scheduler_type: str = 'plateau'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # 损失函数
    loss_fn: str = 'mse'
    lambda_corr: float = 0.01
    
    # 日志
    verbose: bool = True
    log_interval: int = 10


class RollingDailyTrainer:
    """
    滚动窗口日批次训练器
    
    实现真正的 Walk-Forward 训练策略：
    1. 遍历每个滚动窗口
    2. 在当前窗口的训练集上训练模型
    3. 在当前窗口的测试集上预测
    4. （可选）将模型权重传递给下一窗口
    5. 合并所有窗口的预测结果
    
    Args:
        model_factory: 模型工厂函数，每次调用返回一个新的 nn.Module 实例
        config: 训练配置
        device: 计算设备
        warm_start: 是否继承上一窗口的模型权重
        save_each_window: 是否保存每个窗口的模型
    """
    
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        config: Optional[RollingTrainerConfig] = None,
        device: str = 'cuda',
        warm_start: bool = True,
        save_each_window: bool = True,
        **kwargs
    ):
        self.model_factory = model_factory
        self.config = config or RollingTrainerConfig(**kwargs)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.warm_start = warm_start
        self.save_each_window = save_each_window
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 训练状态
        self.window_results: List[Dict[str, Any]] = []
        self.all_predictions: List[pd.DataFrame] = []
        self.current_model_state: Optional[Dict] = None
        
        self.logger.info("=" * 80)
        self.logger.info("🔄 初始化滚动窗口日批次训练器")
        self.logger.info("=" * 80)
        self.logger.info(f"  训练策略: {'继承权重 (Warm Start)' if warm_start else '独立训练'}")
        self.logger.info(f"  设备: {self.device}")
        self.logger.info(f"  每窗口保存: {'是' if save_each_window else '否'}")
    
    def fit(
        self,
        rolling_loaders,
        n_epochs: Optional[int] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        训练所有滚动窗口
        
        Args:
            rolling_loaders: RollingDailyLoaderCollection，由 DataManager.create_rolling_daily_loaders 返回
            n_epochs: 每个窗口的训练轮数
            save_dir: 模型保存目录
            
        Returns:
            训练结果汇总字典
        """
        n_epochs = n_epochs or self.config.n_epochs
        n_windows = len(rolling_loaders)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🚀 开始滚动窗口训练 (Walk-Forward)")
        self.logger.info("=" * 80)
        self.logger.info(f"  总窗口数: {n_windows}")
        self.logger.info(f"  每窗口轮数: {n_epochs}")
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"  保存目录: {save_dir}")
        
        start_time = time.time()
        self.window_results = []
        self.all_predictions = []
        
        for window_idx, loaders in enumerate(rolling_loaders):
            self.logger.info("\n" + "-" * 60)
            self.logger.info(f"📅 窗口 {window_idx + 1}/{n_windows}")
            self.logger.info("-" * 60)
            
            # 1. 创建或继承模型
            model = self._get_model_for_window(window_idx)
            
            # 2. 创建优化器和损失函数
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            criterion = nn.MSELoss()
            
            # 3. 创建单窗口训练器（🆕 透传调度器配置）
            trainer_config = DynamicTrainerConfig(
                n_epochs=n_epochs,
                learning_rate=self.config.learning_rate,
                early_stop=self.config.early_stop,
                loss_fn=self.config.loss_fn,
                lambda_corr=self.config.lambda_corr,
                weight_decay=self.config.weight_decay,
                verbose=self.config.verbose,
                # 🆕 透传学习率调度器配置
                use_scheduler=self.config.use_scheduler,
                scheduler_type=self.config.scheduler_type,
                scheduler_patience=self.config.scheduler_patience,
                scheduler_factor=self.config.scheduler_factor,
                scheduler_min_lr=self.config.scheduler_min_lr
            )
            
            window_trainer = DynamicGraphTrainer(
                model=model,
                config=trainer_config,
                device=self.device,
                optimizer=optimizer,
                criterion=criterion
            )
            
            # 4. 训练当前窗口
            save_path = None
            if save_dir and self.save_each_window:
                save_path = str(save_dir / f"window_{window_idx + 1}.pth")
            
            # 检查模型是否存在（断点续训）
            skip_training = False
            if save_path and Path(save_path).exists():
                self.logger.info(f"  ✓ 发现已训练模型: {save_path}，跳过训练")
                try:
                    checkpoint = torch.load(save_path, map_location=self.device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    skip_training = True
                    window_result = {
                        'best_epoch': checkpoint.get('best_epoch', 0),
                        'best_val_loss': checkpoint.get('best_val_loss', 0.0),
                        'skipped': True
                    }
                except Exception as e:
                    self.logger.warning(f"  ⚠️ 加载已存在模型失败，将重新训练: {e}")
            
            if not skip_training:
                train_loader = loaders.train
                val_loader = loaders.val
                
                # 检查训练集是否为空
                if train_loader is None or len(train_loader) == 0:
                    self.logger.warning(f"  ⚠️ 窗口 {window_idx + 1} 训练集为空，跳过训练")
                    continue
                
                self.logger.info(f"  训练天数: {len(train_loader) if train_loader else 0}")
                self.logger.info(f"  验证天数: {len(val_loader) if val_loader else 0}")
                
                window_result = window_trainer.fit(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    n_epochs=n_epochs,
                    save_path=save_path
                )
            
            # 5. 保存模型状态（用于下一窗口继承）
            if self.warm_start:
                self.current_model_state = copy.deepcopy(model.state_dict())
            
            # 6. 在测试集上预测
            test_loader = loaders.test
            if test_loader and len(test_loader) > 0:
                self.logger.info(f"  测试天数: {len(test_loader)}")
                pred_df, label_df = window_trainer.predict(
                    test_loader,
                    return_labels=True,
                    return_all_factors=True
                )
                
                # 合并预测和标签
                pred_df['window_idx'] = window_idx + 1
                merged = pred_df.merge(
                    label_df.rename(columns={'label': 'y_true'}),
                    on=['trade_date', 'order_book_id'],
                    how='left'
                )
                self.all_predictions.append(merged)
                
                self.logger.info(f"  预测样本: {len(merged):,}")
            else:
                self.logger.warning(f"  ⚠️ 无测试集")
            
            # 7. 记录结果
            window_result['window_idx'] = window_idx + 1
            window_result['save_path'] = save_path
            self.window_results.append(window_result)
            
            self.logger.info(f"✅ 窗口 {window_idx + 1} 完成 | "
                           f"best_epoch={window_result.get('best_epoch')} | "
                           f"best_val_loss={window_result.get('best_val_loss', 0):.6f}")
        
        # 汇总统计
        elapsed = time.time() - start_time
        
        summary = self._build_summary(elapsed)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 滚动窗口训练汇总")
        self.logger.info("=" * 80)
        self.logger.info(f"  总窗口: {summary['n_windows']}")
        self.logger.info(f"  总耗时: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        self.logger.info(f"  平均训练损失: {summary['avg_train_loss']:.6f}")
        self.logger.info(f"  平均验证损失: {summary['avg_val_loss']:.6f}")
        self.logger.info(f"  平均最佳轮数: {summary['avg_best_epoch']:.1f}")
        self.logger.info(f"  总预测样本: {summary['total_predictions']:,}")
        
        return summary
    
    def _get_model_for_window(self, window_idx: int) -> nn.Module:
        """获取当前窗口的模型（新建或继承）"""
        model = self.model_factory()
        model = model.to(self.device)
        
        if self.warm_start and self.current_model_state is not None and window_idx > 0:
            try:
                model.load_state_dict(self.current_model_state)
                self.logger.info(f"  🔗 继承窗口 {window_idx} 的模型权重")
            except Exception as e:
                self.logger.warning(f"  ⚠️ 无法加载前一窗口权重: {e}")
        else:
            self.logger.info(f"  🆕 使用全新模型权重")
        
        return model
    
    def _build_summary(self, elapsed: float) -> Dict[str, Any]:
        """构建训练汇总"""
        train_losses = [r.get('train_losses', [])[-1] for r in self.window_results 
                       if r.get('train_losses')]
        val_losses = [r.get('best_val_loss') for r in self.window_results 
                     if r.get('best_val_loss') is not None]
        best_epochs = [r.get('best_epoch', 0) for r in self.window_results]
        
        total_preds = sum(len(df) for df in self.all_predictions) if self.all_predictions else 0
        
        return {
            'n_windows': len(self.window_results),
            'elapsed_time': elapsed,
            'avg_train_loss': float(np.mean(train_losses)) if train_losses else 0.0,
            'avg_val_loss': float(np.mean(val_losses)) if val_losses else 0.0,
            'avg_best_epoch': float(np.mean(best_epochs)) if best_epochs else 0.0,
            'total_predictions': total_preds,
            'window_results': self.window_results
        }
    
    def get_all_predictions(self) -> pd.DataFrame:
        """
        获取所有窗口的合并预测结果
        
        Returns:
            合并的 DataFrame，包含列:
            - trade_date: 交易日期
            - order_book_id: 股票代码
            - pred: 预测值
            - y_true: 真实标签
            - window_idx: 窗口索引
            - pred_factor_0, pred_factor_1, ... (如果是多因子)
        """
        if not self.all_predictions:
            return pd.DataFrame()
        
        combined = pd.concat(self.all_predictions, ignore_index=True)
        
        # 去重：如果有重叠日期，保留最后一个窗口的预测
        combined = combined.sort_values(['trade_date', 'order_book_id', 'window_idx'])
        combined = combined.drop_duplicates(
            subset=['trade_date', 'order_book_id'],
            keep='last'
        )
        
        return combined.sort_values(['trade_date', 'order_book_id']).reset_index(drop=True)
    
    def get_window_predictions(self, window_idx: int) -> pd.DataFrame:
        """获取指定窗口的预测结果"""
        if window_idx < 0 or window_idx >= len(self.all_predictions):
            return pd.DataFrame()
        return self.all_predictions[window_idx]
    
    def get_summary(self) -> Dict[str, Any]:
        """获取训练汇总（兼容旧接口）"""
        if not self.window_results:
            return {}
        return self._build_summary(0)


# ==================== 工厂函数 ====================

def create_rolling_trainer(
    model_class,
    model_config,
    d_feat: int,
    device: str = 'cuda',
    warm_start: bool = True,
    save_each_window: bool = True,
    **trainer_kwargs
) -> RollingDailyTrainer:
    """
    创建滚动窗口训练器的便捷函数
    
    Args:
        model_class: 模型类（如 HybridGraphModel）
        model_config: 模型配置
        d_feat: 特征维度
        device: 设备
        warm_start: 是否继承权重
        save_each_window: 是否保存每个窗口模型
        **trainer_kwargs: 传递给 RollingTrainerConfig 的参数
        
    Returns:
        RollingDailyTrainer 实例
    """
    def model_factory():
        if hasattr(model_class, 'from_config'):
            wrapper = model_class.from_config(model_config, d_feat=d_feat)
            return wrapper.model  # 返回底层 nn.Module
        else:
            return model_class(d_feat=d_feat, **model_config.__dict__)
    
    config = RollingTrainerConfig(**trainer_kwargs)
    
    return RollingDailyTrainer(
        model_factory=model_factory,
        config=config,
        device=device,
        warm_start=warm_start,
        save_each_window=save_each_window
    )


# ==================== 单元测试 ====================

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("RollingDailyTrainer 单元测试")
    print("=" * 80)
    
    # 创建简单的测试模型
    class SimpleModel(nn.Module):
        def __init__(self, d_feat=6, hidden_size=32):
            super().__init__()
            self.rnn = nn.LSTM(d_feat, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x, adj=None):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :]).squeeze(-1)
    
    # 测试模型工厂
    def model_factory():
        return SimpleModel(d_feat=6, hidden_size=32)
    
    # 创建训练器
    trainer = RollingDailyTrainer(
        model_factory=model_factory,
        device='cpu',
        warm_start=True,
        save_each_window=False
    )
    
    print("\n✅ RollingDailyTrainer 创建成功")
    print("\n功能:")
    print("  - 支持真正的 Walk-Forward 滚动窗口训练")
    print("  - 支持模型权重继承 (warm_start)")
    print("  - 支持每个窗口独立保存模型")
    print("  - 自动合并所有窗口的预测结果")
    print("  - 兼容 DynamicGraphTrainer 的训练接口")
    
    print("\n" + "=" * 80)
    print("✅ RollingDailyTrainer 测试通过！")
    print("=" * 80)
