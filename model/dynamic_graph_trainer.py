"""
dynamic_graph_trainer.py - 动态图训练器

支持日批次训练和动态图构建的训练器，专为 GNN 模型设计。

核心特性：
1. 每日构建邻接矩阵：在 collate_fn 中调用 GraphBuilder
2. 日批次训练：每个 batch 是一个交易日的所有股票
3. 兼容现有模型：可与 HybridGraphModel 无缝集成

使用示例：
    from quantclassic.model.dynamic_graph_trainer import DynamicGraphTrainer
    from quantclassic.data_processor.graph_builder import HybridGraphBuilder
    from quantclassic.model.hybrid_graph_models import HybridGraphModel
    
    # 创建图构建器
    graph_builder = HybridGraphBuilder(alpha=0.7, top_k=10)
    
    # 创建训练器
    trainer = DynamicGraphTrainer(
        model=model,
        graph_builder=graph_builder,
        device='cuda'
    )
    
    # 训练
    results = trainer.fit(
        train_loader=train_daily_loader,
        val_loader=val_daily_loader,
        n_epochs=20
    )
    
    # 预测
    predictions = trainer.predict(test_daily_loader)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
from tqdm import tqdm
import time


@dataclass
class DynamicTrainerConfig:
    """动态图训练器配置"""
    # 训练配置
    n_epochs: int = 20
    learning_rate: float = 0.001
    early_stop: int = 5
    optimizer: str = 'adam'
    weight_decay: float = 0.0
    
    # 学习率调度
    use_scheduler: bool = True
    scheduler_type: str = 'plateau'
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # 损失函数
    loss_fn: str = 'mse'
    lambda_corr: float = 0.01  # 相关性正则化权重
    
    # 设备
    device: str = 'cuda'
    
    # 日志
    verbose: bool = True
    log_interval: int = 10  # 每 N 个 batch 打印一次


class DynamicGraphTrainer:
    """
    动态图训练器
    
    专为日批次 + 动态图构建设计的训练器。
    
    与传统训练器的区别：
    - 输入是 DailyGraphDataLoader，每个 batch 是一天的所有股票
    - 邻接矩阵在 DataLoader 中动态构建
    - 支持截面 IC 作为评估指标
    
    Args:
        model: PyTorch 模型（需要支持 forward(X, adj) 接口）
        graph_builder: GraphBuilder 实例（可选，如果 loader 已包含则不需要）
        config: DynamicTrainerConfig 配置
        device: 计算设备
    """
    
    def __init__(
        self,
        model: nn.Module,
        graph_builder: Optional[Any] = None,
        config: Optional[DynamicTrainerConfig] = None,
        device: str = 'cuda',
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
        **kwargs
    ):
        """初始化动态图训练器。

        Args:
            model: 需要训练的 PyTorch 模型 (nn.Module)
            graph_builder: 图构建器 (可选)
            config: 训练配置，若为 None 则使用 kwargs 初始化 DynamicTrainerConfig
            device: 训练设备字符串
            optimizer: 外部传入的优化器 (可选)
            scheduler: 外部传入的学习率调度器 (可选)
            criterion: 外部传入的损失函数 (可选)
            **kwargs: 当 config 为 None 时用于初始化 DynamicTrainerConfig 的关键字参数
        """

        self.config = config or DynamicTrainerConfig(**kwargs)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.graph_builder = graph_builder
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 创建 / 接受优化器
        self.optimizer = optimizer or self._create_optimizer()
        
        # 创建 / 接受学习率调度器
        self.scheduler = scheduler or self._create_scheduler()
        
        # 创建 / 接受损失函数
        self.criterion = criterion or self._create_criterion()
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.val_ics = []
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        if self.config.optimizer.lower() == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        if not self.config.use_scheduler:
            return None
        
        if self.config.scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                min_lr=self.config.scheduler_min_lr
            )
        elif self.config.scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.n_epochs
            )
        else:
            return None
    
    def _create_criterion(self) -> nn.Module:
        """创建损失函数"""
        if self.config.loss_fn.lower() == 'mse':
            return nn.MSELoss()
        elif self.config.loss_fn.lower() == 'mae':
            return nn.L1Loss()
        elif self.config.loss_fn.lower() == 'huber':
            return nn.HuberLoss()
        else:
            return nn.MSELoss()
    
    def _compute_ic(self, pred: torch.Tensor, label: torch.Tensor) -> float:
        """
        计算截面 IC (Information Coefficient)
        
        IC = Pearson(pred_ranks, label_ranks)
        """
        if len(pred) < 2:
            return 0.0
        
        pred_np = pred.detach().cpu().numpy().flatten()
        label_np = label.detach().cpu().numpy().flatten()
        
        # 处理多因子输出：取平均
        if len(pred_np.shape) > 1:
            pred_np = pred_np.mean(axis=-1)
        
        # 移除 NaN
        mask = ~(np.isnan(pred_np) | np.isnan(label_np))
        if mask.sum() < 2:
            return 0.0
        
        pred_np = pred_np[mask]
        label_np = label_np[mask]
        
        # 计算相关系数
        corr = np.corrcoef(pred_np, label_np)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        n_epochs: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器（DailyGraphDataLoader）
            val_loader: 验证数据加载器（可选）
            n_epochs: 训练轮数（覆盖配置）
            save_path: 模型保存路径
            
        Returns:
            训练结果字典
        """
        n_epochs = n_epochs or self.config.n_epochs
        
        self.logger.info(f"开始训练 (日批次模式)")
        self.logger.info(f"  训练天数: {len(train_loader)}")
        if val_loader:
            self.logger.info(f"  验证天数: {len(val_loader)}")
        self.logger.info(f"  训练轮数: {n_epochs}")
        self.logger.info(f"  设备: {self.device}")
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            # 训练
            train_loss, train_ic, train_mse, train_reg = self._train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, val_ic = 0.0, 0.0
            if val_loader:
                val_loss, val_ic = self._validate_epoch(val_loader)
                self.val_losses.append(val_loss)
                self.val_ics.append(val_ic)
            
            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loader else train_loss)
                else:
                    self.scheduler.step()
            
            # 打印进度
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                msg = f"Epoch {epoch+1}/{n_epochs} | "
                msg += f"Train Loss: {train_loss:.6f} (MSE: {train_mse:.6f}, Reg: {train_reg:.6f}) | Train IC: {train_ic:.4f}"
                if val_loader:
                    msg += f" | Val Loss: {val_loss:.6f} | Val IC: {val_ic:.4f}"
                msg += f" | LR: {lr:.2e}"
                self.logger.info(msg)
            
            # 早停
            current_loss = val_loss if val_loader else train_loss
            if current_loss < self.best_val_loss:
                self.best_val_loss = current_loss
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                
                # 保存最佳模型
                if save_path:
                    self._save_model(save_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stop:
                    self.logger.info(f"早停触发于 Epoch {epoch+1}")
                    break
        
        elapsed = time.time() - start_time
        
        # 加载最佳模型
        if save_path and Path(save_path).exists():
            self._load_model(save_path)
        
        results = {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_ics': self.val_ics,
            'elapsed_time': elapsed
        }
        
        self.logger.info(f"训练完成! 最佳 Epoch: {self.best_epoch}, "
                        f"最佳验证损失: {self.best_val_loss:.6f}, "
                        f"耗时: {elapsed:.1f}s")
        
        return results
    
    def _train_epoch(
        self, 
        loader: DataLoader,
        epoch: int
    ) -> Tuple[float, float, float, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_reg = 0.0
        total_ic = 0.0
        n_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", disable=not self.config.verbose)
        
        # 首 batch 邻接矩阵统计（仅第一个 epoch 第一个 batch 打印）
        first_batch_logged = (epoch > 0)
        
        for batch_idx, batch in enumerate(pbar):
            # 解包 batch
            X, y, adj, stock_ids, date = batch
            
            # 跳过空 batch
            if len(y) == 0:
                continue
            
            # 移动到设备
            X = X.to(self.device)
            y = y.to(self.device)
            if adj is not None:
                adj = adj.to(self.device)
            
            # 🆕 首 batch 邻接矩阵日志（仅首个 epoch 首个 batch）
            if not first_batch_logged:
                first_batch_logged = True
                if adj is not None:
                    adj_cpu = adj.detach().cpu()
                    n = adj_cpu.shape[0]
                    diag_sum = int(adj_cpu.diag().sum().item())
                    nonzero = int((adj_cpu > 0).sum().item())
                    off_diag = nonzero - diag_sum
                    self.logger.info(f"✅ 动态邻接矩阵已传入模型 | 日期={date} | "
                                     f"N={n} | 边数={nonzero} | 跨股票边={off_diag}")
                else:
                    self.logger.warning("⚠️ 邻接矩阵 adj=None，模型将使用自环（单位阵）")
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # 模型推理
            pred = self.model(X, adj)
            
            # 处理多因子输出
            if len(pred.shape) > 1 and pred.shape[-1] > 1:
                # 多因子取平均
                pred_for_loss = pred.mean(dim=-1)
            else:
                pred_for_loss = pred.squeeze()
            
            # 计算损失
            mse_loss = self.criterion(pred_for_loss, y)
            
            # 相关性正则化
            reg_loss = torch.tensor(0.0, device=self.device)
            if self.config.lambda_corr > 0 and len(pred.shape) > 1:
                reg_loss = self._correlation_regularization(pred)
                loss = mse_loss + self.config.lambda_corr * reg_loss
            else:
                loss = mse_loss
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_reg += reg_loss.item()
            total_ic += self._compute_ic(pred_for_loss, y)
            n_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mse': f"{mse_loss.item():.4f}",
                'reg': f"{reg_loss.item():.4f}",
                'ic': f"{self._compute_ic(pred_for_loss, y):.4f}"
            })
        
        avg_loss = total_loss / max(n_batches, 1)
        avg_ic = total_ic / max(n_batches, 1)
        avg_mse = total_mse / max(n_batches, 1)
        avg_reg = total_reg / max(n_batches, 1)
        
        return avg_loss, avg_ic, avg_mse, avg_reg
    
    def _validate_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """验证一个 epoch"""
        self.model.eval()
        total_loss = 0.0
        total_ic = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                X, y, adj, stock_ids, date = batch
                
                if len(y) == 0:
                    continue
                
                X = X.to(self.device)
                y = y.to(self.device)
                if adj is not None:
                    adj = adj.to(self.device)
                
                pred = self.model(X, adj)
                
                if len(pred.shape) > 1 and pred.shape[-1] > 1:
                    pred_for_loss = pred.mean(dim=-1)
                else:
                    pred_for_loss = pred.squeeze()
                
                loss = self.criterion(pred_for_loss, y)
                
                total_loss += loss.item()
                total_ic += self._compute_ic(pred_for_loss, y)
                n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        avg_ic = total_ic / max(n_batches, 1)
        
        return avg_loss, avg_ic
    
    def _correlation_regularization(self, pred: torch.Tensor) -> torch.Tensor:
        """
        计算多因子之间的相关性正则化项
        
        鼓励不同因子之间正交（低相关性）
        """
        if len(pred.shape) < 2 or pred.shape[-1] <= 1:
            return torch.tensor(0.0, device=pred.device)
        
        # pred: [N, F] 多因子输出
        # 计算因子之间的相关系数矩阵
        pred_centered = pred - pred.mean(dim=0, keepdim=True)
        pred_std = pred.std(dim=0, keepdim=True) + 1e-8
        pred_normalized = pred_centered / pred_std
        
        corr_matrix = torch.mm(pred_normalized.T, pred_normalized) / pred.shape[0]
        
        # 正则化项：非对角元素的平方和
        mask = 1 - torch.eye(corr_matrix.shape[0], device=pred.device)
        reg = (corr_matrix * mask).pow(2).sum()
        
        return reg
    
    def predict(
        self, 
        loader: DataLoader,
        return_labels: bool = False,
        return_all_factors: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        预测
        
        Args:
            loader: 测试数据加载器
            return_labels: 是否返回标签
            return_all_factors: 是否返回所有因子列（多因子输出时）
            
        Returns:
            预测结果 DataFrame（包含日期、股票ID、预测值）
            如果 return_all_factors=True，则包含 pred_factor_0, pred_factor_1, ... 列
        """
        self.model.eval()
        
        all_preds = []           # 存储平均后的预测值
        all_factor_preds = []    # 存储所有因子的预测值 (多因子时)
        all_labels = []
        all_stocks = []
        all_dates = []
        n_factors = None
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="预测中", disable=not self.config.verbose):
                X, y, adj, stock_ids, date = batch
                
                if len(y) == 0:
                    continue
                
                X = X.to(self.device)
                if adj is not None:
                    adj = adj.to(self.device)
                
                pred = self.model(X, adj)
                
                # 处理多因子输出
                pred_np = pred.detach().cpu().numpy()
                if len(pred_np.shape) > 1 and pred_np.shape[-1] > 1:
                    # 多因子模式：[N, F]
                    n_factors = pred_np.shape[-1]
                    if return_all_factors:
                        all_factor_preds.append(pred_np)  # [N, F]
                    # 取平均作为主预测值
                    pred_mean = pred_np.mean(axis=-1)  # [N]
                    all_preds.append(pred_mean)
                else:
                    # 单因子模式
                    all_preds.append(pred_np.flatten())
                
                all_labels.append(y.cpu().numpy().flatten())
                all_stocks.extend(stock_ids)
                all_dates.extend([date] * len(stock_ids))
        
        # 构建结果 DataFrame
        pred_values = np.concatenate(all_preds) if all_preds else np.array([])
        
        pred_df = pd.DataFrame({
            'trade_date': all_dates,
            'order_book_id': all_stocks,
            'pred': pred_values
        })
        
        # 如果需要返回所有因子
        if return_all_factors and all_factor_preds and n_factors:
            all_factors_np = np.concatenate(all_factor_preds, axis=0)  # [total_N, F]
            for f_idx in range(n_factors):
                pred_df[f'pred_factor_{f_idx}'] = all_factors_np[:, f_idx]
        
        if return_labels:
            label_df = pd.DataFrame({
                'trade_date': all_dates,
                'order_book_id': all_stocks,
                'label': np.concatenate(all_labels) if all_labels else []
            })
            return pred_df, label_df
        
        return pred_df
    
    def _save_model(self, path: str):
        """保存模型"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
        }, path)
        self.logger.info(f"模型已保存: {path}")
    
    def _load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"模型已加载: {path}")


# ==================== 工厂函数 ====================

def create_dynamic_trainer(
    model: nn.Module,
    graph_builder_config: Optional[Dict] = None,
    trainer_config: Optional[Dict] = None,
    device: str = 'cuda'
) -> DynamicGraphTrainer:
    """
    创建动态图训练器的便捷函数
    
    Args:
        model: 模型
        graph_builder_config: 图构建器配置
        trainer_config: 训练器配置
        device: 设备
        
    Returns:
        DynamicGraphTrainer 实例
    """
    # 创建图构建器
    graph_builder = None
    if graph_builder_config:
        from quantclassic.data_processor.graph_builder import GraphBuilderFactory
        graph_builder = GraphBuilderFactory.create(graph_builder_config)
    
    # 创建训练器配置
    config = DynamicTrainerConfig(**(trainer_config or {}))
    
    return DynamicGraphTrainer(
        model=model,
        graph_builder=graph_builder,
        config=config,
        device=device
    )


# ==================== 单元测试 ====================

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("DynamicGraphTrainer 单元测试")
    print("=" * 80)
    
    # 创建简单的测试模型
    class SimpleGNNModel(nn.Module):
        def __init__(self, d_feat, hidden_size, output_dim=1):
            super().__init__()
            self.rnn = nn.LSTM(d_feat, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_dim)
        
        def forward(self, x, adj=None):
            # x: [N, T, F]
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :]).squeeze(-1)
    
    # 创建模型
    model = SimpleGNNModel(d_feat=6, hidden_size=32, output_dim=1)
    
    # 创建训练器
    config = DynamicTrainerConfig(
        n_epochs=2,
        learning_rate=0.001,
        verbose=True
    )
    trainer = DynamicGraphTrainer(model=model, config=config, device='cpu')
    
    print("\n✅ DynamicGraphTrainer 创建成功")
    
    # 创建模拟数据
    print("\n【测试训练流程】")
    
    # 模拟 DailyGraphDataLoader 的输出
    class MockDailyLoader:
        def __init__(self, n_days=5, n_stocks=10):
            self.n_days = n_days
            self.n_stocks = n_stocks
        
        def __len__(self):
            return self.n_days
        
        def __iter__(self):
            for i in range(self.n_days):
                X = torch.randn(self.n_stocks, 20, 6)  # [N, T, F]
                y = torch.randn(self.n_stocks)  # [N]
                adj = torch.eye(self.n_stocks)  # [N, N]
                stock_ids = [f'stock_{j}' for j in range(self.n_stocks)]
                date = f'2024-01-{i+1:02d}'
                yield X, y, adj, stock_ids, date
    
    mock_loader = MockDailyLoader(n_days=5, n_stocks=10)
    
    # 测试训练
    results = trainer.fit(
        train_loader=mock_loader,
        val_loader=MockDailyLoader(n_days=2, n_stocks=10),
        n_epochs=2
    )
    
    print(f"\n训练结果:")
    print(f"  最佳 Epoch: {results['best_epoch']}")
    print(f"  最佳验证损失: {results['best_val_loss']:.6f}")
    print(f"  耗时: {results['elapsed_time']:.2f}s")
    
    # 测试预测
    print("\n【测试预测流程】")
    pred_df = trainer.predict(MockDailyLoader(n_days=3, n_stocks=10))
    print(f"  预测结果形状: {pred_df.shape}")
    print(f"  列名: {list(pred_df.columns)}")
    
    print("\n" + "=" * 80)
    print("✅ 所有 DynamicGraphTrainer 测试通过！")
    print("=" * 80)
