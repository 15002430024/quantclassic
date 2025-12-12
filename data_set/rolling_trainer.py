"""
Rolling Window Trainer - 滚动窗口训练器

实现滚动窗口（Walk-Forward）模型训练和预测
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from .factory import TimeSeriesStockDataset, CrossSectionalBatchSampler


class RollingWindowTrainer:
    """
    滚动窗口训练器
    
    实现 Walk-Forward 验证策略：
    1. 将时间序列划分为多个滚动窗口
    2. 在每个窗口上独立训练模型
    3. 在下一个窗口上测试
    4. 合并所有窗口的预测结果
    
    Features:
    - 支持完全独立的滚动窗口训练
    - 支持增量训练（使用前一窗口模型初始化）
    - 自动管理模型保存和加载
    - 提供详细的训练和预测日志
    
    Example:
        >>> trainer = RollingWindowTrainer(
        ...     windows=rolling_windows,
        ...     config=data_config,
        ...     feature_cols=feature_cols
        ... )
        >>> results = trainer.train_all_windows(
        ...     model_class=GRUModel,
        ...     model_config=gru_config,
        ...     save_dir='output/rolling_models'
        ... )
        >>> predictions = trainer.predict_all_windows(results)
    """
    
    def __init__(
        self,
        windows: List[Tuple[pd.DataFrame, pd.DataFrame]],
        config: Any,
        feature_cols: List[str],
        logger: Optional[logging.Logger] = None,
        stock_universe: Optional[List[str]] = None  # 🆕 全局股票池
    ):
        """
        初始化滚动窗口训练器
        
        Args:
            windows: 滚动窗口列表 [(train_df_1, test_df_1), ...]
            config: DataConfig 配置对象
            feature_cols: 特征列名列表
            logger: 日志记录器（可选）
            stock_universe: 全局股票代码列表（用于统一ID映射）
        """
        self.windows = windows
        self.config = config
        self.feature_cols = feature_cols
        self.logger = logger or self._setup_logger()
        
        # 🆕 构建全局股票映射
        if stock_universe:
            self.stock_map = {stock: i for i, stock in enumerate(sorted(stock_universe))}
            self.logger.info(f"  全局股票池: {len(stock_universe)} 只")
        else:
            self.stock_map = None
            self.logger.info(f"  未提供全局股票池，将使用局部映射")
        
        self.n_windows = len(windows)
        self.window_results = []  # 存储每个窗口的训练结果
        
        self.logger.info("=" * 80)
        self.logger.info("🔄 初始化滚动窗口训练器")
        self.logger.info("=" * 80)
        self.logger.info(f"  总窗口数: {self.n_windows}")
        self.logger.info(f"  特征维度: {len(feature_cols)}")
        
        # 统计窗口信息
        train_sizes = [len(train_df) for train_df, _ in windows]
        test_sizes = [len(test_df) for _, test_df in windows]
        
        self.logger.info(f"  训练集大小: {min(train_sizes):,} ~ {max(train_sizes):,} 样本")
        self.logger.info(f"  测试集大小: {min(test_sizes):,} ~ {max(test_sizes):,} 样本")
    
    def _setup_logger(self) -> logging.Logger:
        """配置日志"""
        logger = logging.getLogger('RollingWindowTrainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_datasets_for_window(
        self,
        window_idx: int,
        val_ratio: float = 0.2
    ) -> Tuple[Any, Any, Any]:
        """
        为指定窗口创建数据集
        
        Args:
            window_idx: 窗口索引
            val_ratio: 从训练集中划分验证集的比例
            
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        from torch.utils.data import Subset
        import numpy as np
        
        train_df, test_df = self.windows[window_idx]
        
        # 🆕 传递窗口变换配置
        enable_wt = getattr(self.config, 'enable_window_transform', False)
        price_log = getattr(self.config, 'window_price_log', False)
        vol_norm = getattr(self.config, 'window_volume_norm', False)
        price_cols = getattr(self.config, 'price_cols', ['open', 'high', 'low', 'close', 'vwap'])
        close_col = getattr(self.config, 'close_col', 'close')
        volume_cols = getattr(self.config, 'volume_cols', ['vol', 'amount'])
        
        # 🆕 标签窗口级排名标准化配置
        label_rank_norm = getattr(self.config, 'label_rank_normalize', False)
        label_rank_range = getattr(self.config, 'label_rank_output_range', (-1, 1))

        # 🆕 按时间划分训练集和验证集 (Time-Series Split)
        # 避免使用 Subset 导致的按股票ID划分问题
        
        all_dates = sorted(train_df[self.config.time_col].unique())
        n_dates = len(all_dates)
        
        # 确保数据足够长以进行划分
        if n_dates > self.config.window_size + 20 and val_ratio > 0:
            # 计算分割点
            split_idx = int(n_dates * (1 - val_ratio))
            split_date = all_dates[split_idx]
            
            # 1. 构建训练集 DataFrame (split_date 之前)
            train_split_df = train_df[train_df[self.config.time_col] < split_date].copy()
            
            # 2. 构建验证集 DataFrame (包含足够的回看窗口)
            # 我们需要 split_date 开始的预测，所以需要往前推 window_size 天的数据作为特征
            lookback_idx = max(0, split_idx - self.config.window_size)
            lookback_date = all_dates[lookback_idx]
            val_split_df = train_df[train_df[self.config.time_col] >= lookback_date].copy()
            
            self.logger.info(f"  数据集划分 (Time-Series Split):")
            self.logger.info(f"    训练集截止: {split_date} (不含)")
            self.logger.info(f"    验证集开始: {split_date} (预测日期)")
            
            # 创建训练数据集
            train_dataset = TimeSeriesStockDataset(
                df=train_split_df,
                feature_cols=self.feature_cols,
                label_col=self.config.label_col,
                window_size=self.config.window_size,
                stock_col=self.config.stock_col,
                time_col=self.config.time_col,
                return_stock_id=True,
                enable_window_transform=enable_wt,
                window_price_log=price_log,
                window_volume_norm=vol_norm,
                price_cols=price_cols,
                close_col=close_col,
                volume_cols=volume_cols,
                label_rank_normalize=label_rank_norm,
                label_rank_output_range=label_rank_range,
                stock_map=self.stock_map
            )
            
            # 创建验证数据集
            val_dataset = TimeSeriesStockDataset(
                df=val_split_df,
                feature_cols=self.feature_cols,
                label_col=self.config.label_col,
                window_size=self.config.window_size,
                stock_col=self.config.stock_col,
                time_col=self.config.time_col,
                return_stock_id=True,
                enable_window_transform=enable_wt,
                window_price_log=price_log,
                window_volume_norm=vol_norm,
                price_cols=price_cols,
                close_col=close_col,
                volume_cols=volume_cols,
                label_rank_normalize=label_rank_norm,
                label_rank_output_range=label_rank_range,
                stock_map=self.stock_map
            )
            
        else:
            # 数据太少，不划分验证集
            self.logger.warning(f"  警告: 数据不足以进行时间序列划分，使用全量训练")
            train_dataset = TimeSeriesStockDataset(
                df=train_df,
                feature_cols=self.feature_cols,
                label_col=self.config.label_col,
                window_size=self.config.window_size,
                stock_col=self.config.stock_col,
                time_col=self.config.time_col,
                return_stock_id=True,
                enable_window_transform=enable_wt,
                window_price_log=price_log,
                window_volume_norm=vol_norm,
                price_cols=price_cols,
                close_col=close_col,
                volume_cols=volume_cols,
                label_rank_normalize=label_rank_norm,
                label_rank_output_range=label_rank_range,
                stock_map=self.stock_map
            )
            val_dataset = None
        
        # 创建测试数据集
        # 🔴 修复: 测试集需要包含 lookback window 的历史数据
        # 否则 TimeSeriesStockDataset 会因为数据长度不足而丢弃样本
        
        # 1. 获取测试集的开始日期
        if not test_df.empty:
            test_start_date = test_df[self.config.time_col].min()
            test_end_date = test_df[self.config.time_col].max()
            
            # 2. 从原始数据中获取包含 lookback 的数据
            # 注意：这里我们需要访问原始数据，但 self.windows 只包含切分后的数据
            # 幸好 train_df 通常包含 test_df 之前的数据（如果是 rolling 且无 gap）
            # 或者我们可以假设 train_df 的末尾就是 test_df 的开始前一天
            
            # 更稳健的方法：我们需要访问 DataManager 的原始数据，但这里没有引用
            # 替代方案：利用 train_df 的末尾数据作为 lookback
            
            train_end_date = train_df[self.config.time_col].max()
            
            # 检查 train_df 是否紧邻 test_df
            # 如果 train_end_date < test_start_date，说明可能有 gap，或者 train_df 就是历史数据
            
            # 提取 train_df 中最后 window_size * 2 天的数据
            lookback_start = test_start_date - pd.Timedelta(days=self.config.window_size * 2)
            lookback_df = train_df[train_df[self.config.time_col] >= lookback_start].copy()
            
            if not lookback_df.empty:
                # 合并 lookback 和 test_df
                extended_test_df = pd.concat([lookback_df, test_df], ignore_index=True)
                # 去重
                extended_test_df = extended_test_df.drop_duplicates(subset=[self.config.stock_col, self.config.time_col])
                # 排序
                extended_test_df = extended_test_df.sort_values([self.config.stock_col, self.config.time_col])
                
                # 使用扩展后的数据创建测试集
                test_dataset_df = extended_test_df
                valid_label_start = pd.Timestamp(test_start_date)
            else:
                test_dataset_df = test_df
                valid_label_start = None
        else:
            test_dataset_df = test_df
            valid_label_start = None

        test_dataset = TimeSeriesStockDataset(
            df=test_dataset_df,
            feature_cols=self.feature_cols,
            label_col=self.config.label_col,
            window_size=self.config.window_size,
            stock_col=self.config.stock_col,
            time_col=self.config.time_col,
            # 🆕 启用返回股票ID
            return_stock_id=True,
            # 🆕 传递窗口变换参数
            enable_window_transform=enable_wt,
            window_price_log=price_log,
            window_volume_norm=vol_norm,
            price_cols=price_cols,
            close_col=close_col,
            volume_cols=volume_cols,
            # 🆕 标签窗口级排名标准化
            label_rank_normalize=label_rank_norm,
            label_rank_output_range=label_rank_range,
            # 🆕 传递全局股票映射
            stock_map=self.stock_map,
            # 🆕 传递有效标签起始日期
            valid_label_start_date=valid_label_start
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def train_window(
        self,
        window_idx: int,
        model_class: type,
        model_config: Any,
        save_path: Optional[str] = None,
        val_ratio: float = 0.2,
        init_model_path: Optional[str] = None,
        use_cross_sectional: bool = False
    ) -> Dict[str, Any]:
        """
        训练单个窗口
        
        Args:
            window_idx: 窗口索引
            model_class: 模型类（如 GRUModel）
            model_config: 模型配置对象
            save_path: 模型保存路径（可选）
            val_ratio: 验证集比例
            init_model_path: 初始化模型路径（增量训练）
            use_cross_sectional: 是否使用截面批采样（按日期组织Batch）
            
        Returns:
            训练结果字典，包含：
            - window_idx: 窗口索引
            - model: 训练好的模型
            - train_loss: 训练损失
            - val_loss: 验证损失
            - best_epoch: 最佳轮数
            - save_path: 模型保存路径
        """
        self.logger.info("=" * 80)
        self.logger.info(f"🔄 训练窗口 {window_idx + 1}/{self.n_windows}")
        self.logger.info("=" * 80)
        
        # 创建数据集
        train_dataset, val_dataset, test_dataset = self.create_datasets_for_window(
            window_idx, val_ratio
        )
        
        self.logger.info(f"  训练样本: {len(train_dataset):,}")
        if val_dataset:
            self.logger.info(f"  验证样本: {len(val_dataset):,}")
        self.logger.info(f"  测试样本: {len(test_dataset):,}")
        
        # 创建数据加载器
        if use_cross_sectional:
            self.logger.info("  采样策略: 截面批采样 (Cross-Sectional Batch)")
            self.logger.info("    - 每个 Batch 包含同一交易日的股票")
            self.logger.info("    - 日期顺序随机打乱")
            
            # 使用截面批采样器
            train_sampler = CrossSectionalBatchSampler(
                train_dataset, 
                batch_size=self.config.batch_size, 
                shuffle_dates=True
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                num_workers=0
            )
        else:
            self.logger.info("  采样策略: 全局随机打乱 (Global Shuffle)")
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=False
            )
        
        val_loader = None
        if val_dataset and len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False
            )
        
        # 创建模型
        # 优先使用 from_config 类方法（如果存在）
        if hasattr(model_class, 'from_config'):
            model = model_class.from_config(model_config, d_feat=len(self.feature_cols))
        else:
            # 兼容旧的初始化方式
            model = model_class(
                d_feat=len(self.feature_cols),
                hidden_size=model_config.hidden_size,
                num_layers=model_config.num_layers,
                dropout=model_config.dropout,
                weight_decay=getattr(model_config, 'weight_decay', 0.0001),
                n_epochs=model_config.n_epochs,
                batch_size=model_config.batch_size,
                lr=model_config.learning_rate,
                early_stop=model_config.early_stop,
                optimizer=model_config.optimizer,
                device=model_config.device
            )
        
        # 如果提供了初始化模型，加载权重（增量训练）
        if init_model_path and Path(init_model_path).exists():
            self.logger.info(f"  加载初始化模型: {init_model_path}")
            model.load_model(init_model_path)
        
        # 训练模型
        self.logger.info(f"  开始训练...")
        model.fit(train_loader, val_loader, save_path=save_path)
        
        result = {
            'window_idx': window_idx,
            'model': model,
            'train_loss': model.train_losses[-1] if model.train_losses else None,
            'val_loss': model.valid_losses[-1] if model.valid_losses else None,
            'best_epoch': model.best_epoch,
            'best_score': model.best_score,
            'save_path': save_path,
            'test_dataset': test_dataset
        }
        
        self.logger.info(f"✅ 窗口 {window_idx + 1} 训练完成")
        self.logger.info(f"  最佳Epoch: {model.best_epoch + 1}")
        self.logger.info(f"  最佳得分: {model.best_score:.6f}")
        
        return result
    
    def train_all_windows(
        self,
        model_class: type,
        model_config: Any,
        save_dir: Optional[str] = None,
        val_ratio: float = 0.2,
        incremental: bool = False,
        # 🆕 动态邻接矩阵参数
        dynamic_adj: bool = False,
        adj_config: Optional[Dict] = None,
        # 🆕 采样策略参数
        use_cross_sectional: bool = False
    ) -> List[Dict[str, Any]]:
        """
        训练所有滚动窗口
        
        Args:
            model_class: 模型类
            model_config: 模型配置
            save_dir: 模型保存目录（可选）
            val_ratio: 验证集比例
            incremental: 是否使用增量训练（每个窗口用前一窗口模型初始化）
            dynamic_adj: 是否为每个窗口动态构建邻接矩阵
            adj_config: 邻接矩阵构建配置 (dict)
            use_cross_sectional: 是否使用截面批采样（按日期组织Batch）
            
        Returns:
            所有窗口的训练结果列表
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🚀 开始滚动窗口训练")
        self.logger.info("=" * 80)
        self.logger.info(f"  训练策略: {'增量训练' if incremental else '独立训练'}")
        self.logger.info(f"  采样策略: {'截面批采样 (按日期)' if use_cross_sectional else '全局随机打乱'}")
        self.logger.info(f"  总窗口数: {self.n_windows}")
        
        if dynamic_adj:
            self.logger.info(f"  动态图构建: 已启用 (每年重新计算邻接矩阵)")
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"  保存目录: {save_dir}")
        
        results = []
        prev_model_path = None
        
        for i in range(self.n_windows):
            # 生成保存路径
            save_path = None
            if save_dir:
                save_path = str(save_dir / f'window_{i+1}_model.pth')
            
            # 🆕 动态构建邻接矩阵
            if dynamic_adj and adj_config:
                self.logger.info(f"  🔄 正在为窗口 {i+1} 构建动态邻接矩阵...")
                train_df, _ = self.windows[i]
                
                # 构建矩阵
                adj_matrix = self._build_adj_matrix(train_df, adj_config)
                
                # 保存矩阵
                if save_dir:
                    adj_path = str(save_dir / f'adj_window_{i+1}.pt')
                    torch.save(adj_matrix, adj_path)
                    # 更新配置
                    model_config.adj_matrix_path = adj_path
                    self.logger.info(f"     已保存并应用: {adj_path}")
            
            # 训练窗口
            result = self.train_window(
                window_idx=i,
                model_class=model_class,
                model_config=model_config,
                save_path=save_path,
                val_ratio=val_ratio,
                init_model_path=prev_model_path if incremental else None,
                use_cross_sectional=use_cross_sectional
            )
            
            results.append(result)
            
            # 更新前一模型路径（用于增量训练）
            if incremental and save_path:
                prev_model_path = save_path
        
        self.window_results = results
        
        # 汇总统计
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 滚动窗口训练汇总")
        self.logger.info("=" * 80)
        
        train_losses = [r['train_loss'] for r in results if r['train_loss'] is not None]
        val_losses = [r['val_loss'] for r in results if r['val_loss'] is not None]
        best_epochs = [r['best_epoch'] for r in results]
        
        if train_losses:
            self.logger.info(f"  平均训练损失: {np.mean(train_losses):.6f}")
        if val_losses:
            self.logger.info(f"  平均验证损失: {np.mean(val_losses):.6f}")
        self.logger.info(f"  平均最佳Epoch: {np.mean(best_epochs):.1f}")
        
        self.logger.info("\n✅ 所有窗口训练完成！")
        
        return results

    def _build_adj_matrix(self, df: pd.DataFrame, config: Dict) -> torch.Tensor:
        """
        构建邻接矩阵 (内部方法)
        
        已更新：使用新的 AdjMatrixUtils（来自 graph_builder.py）代替旧的 AdjMatrixBuilder
        """
        from quantclassic.data_processor.graph_builder import AdjMatrixUtils
        
        # 1. 准备收益率数据
        # 使用配置中的列名，默认为 'y_ret_10d'
        ret_col = config.get('return_col', 'y_ret_10d')
        
        # Pivot table
        returns_pivot = df.pivot_table(
            index=self.config.time_col,
            columns=self.config.stock_col,
            values=ret_col,
            aggfunc='first'
        )
        
        # 填充缺失值
        returns_pivot = returns_pivot.ffill().bfill().fillna(0)
        
        # 2. 如果有全局映射，需要对齐列
        if self.stock_map:
            # 创建一个包含所有全局股票的DataFrame
            full_returns = pd.DataFrame(
                0.0, 
                index=returns_pivot.index, 
                columns=sorted(self.stock_map.keys()) # 确保按字母顺序排序，与stock_map一致
            )
            # 更新有数据的部分
            common_cols = returns_pivot.columns.intersection(full_returns.columns)
            full_returns[common_cols] = returns_pivot[common_cols]
            returns_pivot = full_returns
        
        # 3. 构建矩阵（使用新的 AdjMatrixUtils）
        adj = AdjMatrixUtils.build_correlation_adj(
            returns=returns_pivot,
            top_k=config.get('top_k', 10),
            method=config.get('method', 'pearson'),
            self_loop=True
        )
        
        return adj
    
    def predict_window(
        self,
        window_result: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        对单个窗口进行预测
        
        Args:
            window_result: 窗口训练结果
            
        Returns:
            (predictions, labels, stocks, dates)
        """
        model = window_result['model']
        test_dataset = window_result['test_dataset']
        
        # 创建测试数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )
        
        # 预测
        predictions = model.predict(test_loader, return_numpy=True)
        
        # 【修复】处理空测试集的情况
        if len(test_dataset) == 0:
            self.logger.warning(f"  警告: 测试集为空，跳过预测")
            return (
                np.array([]),  # predictions
                np.array([]),  # labels
                np.array([]),  # stocks
                None           # dates
            )
        
        # 【修复】从TimeSeriesStockDataset中提取标签和元数据
        # TimeSeriesStockDataset将数据存储在sample_index和stock_data中
        labels = []
        stocks = []
        dates = []
        
        for idx in range(len(test_dataset)):
            stock_idx, time_idx = test_dataset.sample_index[idx]
            stock_info = test_dataset.stock_data[stock_idx]
            
            # 标签是t+1时刻的值
            labels.append(stock_info['labels'][time_idx + 1])
            stocks.append(stock_info['ts_code'])
            
            # 如果有日期信息，也提取出来
            if 'dates' in stock_info:
                dates.append(stock_info['dates'][time_idx + 1])
            else:
                dates.append(None)
        
        labels = np.array(labels)
        stocks = np.array(stocks)
        dates = np.array(dates) if dates and dates[0] is not None else None
        
        return predictions, labels, stocks, dates
    
    def predict_all_windows(
        self,
        window_results: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """
        对所有窗口进行预测并合并结果
        
        🆕 支持多因子输出 (N×F 矩阵)
        
        Args:
            window_results: 窗口训练结果（可选，默认使用self.window_results）
            
        Returns:
            合并的预测结果DataFrame，包含列：
            - stock_col: 股票代码
            - time_col: 日期
            - pred_alpha: 预测值 (单因子) 或 pred_factor_0, pred_factor_1, ... (多因子)
            - label_col: 真实标签
            - window_idx: 窗口索引
        """
        if window_results is None:
            window_results = self.window_results
        
        if not window_results:
            raise ValueError("没有可用的窗口结果，请先运行 train_all_windows()")
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🔮 开始滚动窗口预测")
        self.logger.info("=" * 80)
        
        all_predictions = []
        
        for i, result in enumerate(window_results):
            self.logger.info(f"  预测窗口 {i + 1}/{len(window_results)}...")
            
            predictions, labels, stocks, dates = self.predict_window(result)
            
            # 【修复】跳过空预测窗口
            if len(predictions) == 0:
                self.logger.warning(f"    窗口 {i + 1} 预测为空，跳过")
                continue
            
            # 🆕 处理多因子输出 (N×F 矩阵)
            if predictions.ndim == 2 and predictions.shape[1] > 1:
                # 多因子输出
                n_factors = predictions.shape[1]
                self.logger.info(f"    检测到多因子输出: F = {n_factors}")
                
                # 创建DataFrame，每个因子一列
                window_df = pd.DataFrame({
                    self.config.stock_col: stocks,
                    self.config.time_col: dates,
                    self.config.label_col: labels,
                    'window_idx': i + 1
                })
                
                # 添加每个因子列
                for f_idx in range(n_factors):
                    window_df[f'pred_factor_{f_idx}'] = predictions[:, f_idx]
                
                # 同时添加简单平均作为默认预测列
                window_df['pred_alpha'] = predictions.mean(axis=1)
                
            else:
                # 单因子输出
                window_df = pd.DataFrame({
                    self.config.stock_col: stocks,
                    self.config.time_col: dates,
                    'pred_alpha': predictions.flatten(),
                    self.config.label_col: labels,
                    'window_idx': i + 1
                })
            
            all_predictions.append(window_df)
            
            self.logger.info(f"    预测样本: {len(window_df):,}")
        
        # 【修复】处理无有效预测的情况
        if not all_predictions:
            self.logger.warning("\n⚠️  所有窗口的预测都为空！")
            self.logger.warning("  这通常意味着测试数据不足以创建有效样本")
            # 返回空DataFrame但保持结构
            return pd.DataFrame(columns=[
                self.config.stock_col,
                self.config.time_col,
                'pred_alpha',
                self.config.label_col,
                'window_idx'
            ])
        
        # 合并所有预测
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        
        self.logger.info("\n✅ 预测完成！")
        self.logger.info(f"  总预测样本: {len(combined_predictions):,}")
        self.logger.info(f"  时间范围: {combined_predictions[self.config.time_col].min()} ~ {combined_predictions[self.config.time_col].max()}")
        self.logger.info(f"  股票数量: {combined_predictions[self.config.stock_col].nunique()}")
        
        # 🆕 显示因子列信息
        factor_cols = [c for c in combined_predictions.columns if c.startswith('pred_factor_')]
        if factor_cols:
            self.logger.info(f"  多因子输出: {len(factor_cols)} 个因子 ({factor_cols[0]}, ..., {factor_cols[-1]})")
        
        return combined_predictions
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取滚动窗口训练和预测的汇总统计
        
        Returns:
            汇总统计字典
        """
        if not self.window_results:
            return {}
        
        train_losses = [r['train_loss'] for r in self.window_results if r['train_loss'] is not None]
        val_losses = [r['val_loss'] for r in self.window_results if r['val_loss'] is not None]
        best_epochs = [r['best_epoch'] for r in self.window_results]
        best_scores = [r['best_score'] for r in self.window_results if r['best_score'] is not None]
        
        summary = {
            'n_windows': self.n_windows,
            'avg_train_loss': float(np.mean(train_losses)) if train_losses else None,
            'avg_val_loss': float(np.mean(val_losses)) if val_losses else None,
            'avg_best_epoch': float(np.mean(best_epochs)),
            'avg_best_score': float(np.mean(best_scores)) if best_scores else None,
            'std_train_loss': float(np.std(train_losses)) if train_losses else None,
            'std_val_loss': float(np.std(val_losses)) if val_losses else None,
        }
        
        return summary


if __name__ == '__main__':
    print("=" * 80)
    print("Rolling Window Trainer 测试")
    print("=" * 80)
    
    print("\n✅ RollingWindowTrainer 定义完成")
    print("\n功能:")
    print("  - 支持完全独立的滚动窗口训练")
    print("  - 支持增量训练（使用前一窗口模型初始化）")
    print("  - 自动管理模型保存和加载")
    print("  - 提供详细的训练和预测日志")
    print("  - 合并所有窗口的预测结果")
    
    print("\n滚动窗口训练器已准备就绪！")
