"""
DataManager - 数据管理主控类

整合所有数据管理组件，提供统一的接口
"""

import os
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, Union
import logging
from pathlib import Path
import pickle
from datetime import datetime

from .config import DataConfig, ConfigTemplates
from .loader import DataLoaderEngine
from .feature_engineer import FeatureEngineer
from .splitter import create_splitter, DataSplitter
from .validator import DataValidator, ValidationReport
from .factory import DatasetFactory, DatasetCollection, LoaderCollection
# ⚠️ RollingWindowTrainer 已移除 - 请使用 model.train.RollingWindowTrainer


# =============================================================================
# 🆕 日批次构建辅助函数（去重 create_daily_loaders 与 create_rolling_daily_loaders）
# =============================================================================
def _normalize_graph_builder_config(
    gb_config: Optional[Union[Dict, Any]],
    raw_data: Optional[pd.DataFrame] = None,
    stock_col: str = 'ts_code',
    logger: Optional[logging.Logger] = None
) -> Optional[Dict]:
    """
    统一处理 graph_builder_config，确保返回 dict 类型，并注入行业映射（如需要）
    
    Args:
        gb_config: 图构建配置（dict 或 dataclass）
        raw_data: 原始数据（用于提取行业映射）
        stock_col: 股票代码列名
        logger: 日志记录器
        
    Returns:
        标准化后的 dict 配置，或 None
    """
    if gb_config is None:
        return None
    
    # 统一转换为 dict
    if isinstance(gb_config, dict):
        gb_dict = gb_config.copy()
    elif hasattr(gb_config, 'to_dict'):
        gb_dict = gb_config.to_dict()
    else:
        gb_dict = dict(gb_config)
    
    # 如果是行业图，预先构建全局股票-行业映射
    if gb_dict.get('type') == 'industry':
        industry_col = gb_dict.get('industry_col', 'industry_name')
        if raw_data is not None and industry_col in raw_data.columns:
            stock_industry_mapping = dict(zip(
                raw_data[stock_col],
                raw_data[industry_col]
            ))
            gb_dict['stock_industry_mapping'] = stock_industry_mapping
            if logger:
                logger.info(f"  已构建全局股票-行业映射: {len(stock_industry_mapping)} 只股票")
    
    return gb_dict


class DataManager:
    """
    数据管理主控类
    
    整合数据加载、特征工程、数据划分、验证和数据集创建的完整流程
    """
    
    def __init__(self, config: Optional[DataConfig] = None, **kwargs):
        """
        Args:
            config: DataConfig配置对象或字典（None则使用默认配置）
            **kwargs: 额外的配置参数,会覆盖 config 中的值
        """
        # 支持三种初始化方式:
        # 1. DataManager(config=DataConfig(...))
        # 2. DataManager(config={'base_dir': '...'})
        # 3. DataManager(base_dir='...', data_file='...', ...)
        
        if isinstance(config, dict):
            # 字典形式: 合并 config 和 kwargs
            merged_config = {**config, **kwargs}
            self.config = DataConfig(**merged_config)
        elif config is not None:
            # DataConfig 对象: 用 kwargs 更新
            self.config = config
            if kwargs:
                self.config.update(**kwargs)
        else:
            # 仅 kwargs: 创建新 DataConfig
            self.config = DataConfig(**kwargs) if kwargs else DataConfig()
        
        self.logger = self._setup_logger()
        
        # 初始化各组件
        self.loader = DataLoaderEngine(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.validator = DataValidator(self.config)
        self.factory = DatasetFactory(self.config)
        
        # 数据缓存
        self._raw_data: Optional[pd.DataFrame] = None
        self._train_df: Optional[pd.DataFrame] = None
        self._val_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None
        self._feature_cols: Optional[List[str]] = None
        self._datasets: Optional[DatasetCollection] = None
        
        self.logger.info("✅ DataManager 初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """配置日志"""
        logger = logging.getLogger('DataManager')
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_raw_data(self, file_path: Optional[str] = None,
                     use_cache: bool = True) -> pd.DataFrame:
        """
        加载原始数据
        
        Args:
            file_path: 数据文件路径（None则使用配置路径）
            use_cache: 是否使用缓存
            
        Returns:
            原始数据DataFrame
        """
        self.logger.info("=" * 80)
        self.logger.info("步骤 1/5: 加载原始数据")
        self.logger.info("=" * 80)
        
        if use_cache and self._raw_data is not None:
            self.logger.info("使用缓存数据")
            return self._raw_data
        
        # 加载数据
        self._raw_data = self.loader.load_data(file_path, use_cache)
        
        # 打印数据摘要
        if self.config.verbose:
            self.loader.print_data_summary(self._raw_data)
        
        return self._raw_data
    
    def validate_data_quality(self, df: Optional[pd.DataFrame] = None,
                             feature_cols: Optional[List[str]] = None) -> ValidationReport:
        """
        验证数据质量
        
        Args:
            df: 待验证数据（None则使用已加载的原始数据）
            feature_cols: 特征列列表
            
        Returns:
            ValidationReport对象
        """
        self.logger.info("=" * 80)
        self.logger.info("步骤 2/5: 验证数据质量")
        self.logger.info("=" * 80)
        
        if df is None:
            if self._raw_data is None:
                raise ValueError("未加载数据，请先调用 load_raw_data()")
            df = self._raw_data
        
        # 执行验证
        report = self.validator.validate(df, feature_cols or self._feature_cols)
        
        # 打印报告
        if self.config.verbose:
            report.print_report()
        
        # 保存报告
        if self.config.save_data_report:
            self._save_validation_report(report)
        
        return report
    
    def preprocess_features(self, df: Optional[pd.DataFrame] = None,
                          auto_filter: bool = True) -> List[str]:
        """
        特征预处理和选择
        
        Args:
            df: 数据DataFrame（None则使用已加载的原始数据）
            auto_filter: 是否自动过滤低质量特征
            
        Returns:
            特征列列表
        """
        self.logger.info("=" * 80)
        self.logger.info("步骤 3/5: 特征工程")
        self.logger.info("=" * 80)
        
        if df is None:
            if self._raw_data is None:
                raise ValueError("未加载数据，请先调用 load_raw_data()")
            df = self._raw_data
        
        # 选择特征
        self._feature_cols = self.feature_engineer.select_features(df)
        
        # 计算特征统计
        self.feature_engineer.compute_feature_stats(df)
        
        # 过滤特征
        if auto_filter:
            self._feature_cols = self.feature_engineer.filter_features(df)
        
        # 保存特征信息
        if self.config.enable_cache:
            self.feature_engineer.save_feature_info()
        
        return self._feature_cols
    
    def create_datasets(self, df: Optional[pd.DataFrame] = None,
                       feature_cols: Optional[List[str]] = None,
                       split_strategy: Optional[str] = None) -> DatasetCollection:
        """
        创建训练、验证、测试数据集
        
        Args:
            df: 数据DataFrame（None则使用已加载的原始数据）
            feature_cols: 特征列列表（None则使用预处理的特征）
            split_strategy: 划分策略（None则使用配置的策略）
            
        Returns:
            DatasetCollection对象
        """
        self.logger.info("=" * 80)
        self.logger.info("步骤 4/5: 数据划分")
        self.logger.info("=" * 80)
        
        if df is None:
            if self._raw_data is None:
                raise ValueError("未加载数据，请先调用 load_raw_data()")
            df = self._raw_data
        
        if feature_cols is None:
            if self._feature_cols is None:
                raise ValueError("未选择特征，请先调用 preprocess_features()")
            feature_cols = self._feature_cols
        
        # 【修复】过滤掉标签缺失的数据（防止滚动窗口训练时出现空数据集）
        original_len = len(df)
        df = df[df[self.config.label_col].notna()].copy()
        filtered_len = len(df)
        if filtered_len < original_len:
            self.logger.info(f"   过滤标签缺失数据: {original_len:,} -> {filtered_len:,} (-{original_len-filtered_len:,})")
        
        # 创建划分器
        if split_strategy:
            original_strategy = self.config.split_strategy
            self.config.split_strategy = split_strategy
            splitter = create_splitter(self.config)
            self.config.split_strategy = original_strategy
        else:
            splitter = create_splitter(self.config)
        
        # 划分数据
        split_result = splitter.split(df)
        
        # 处理不同splitter的返回值
        if self.config.split_strategy == 'rolling':
            # RollingWindowSplitter 返回 List[Tuple[train, test]]
            # 策略: 使用所有窗口的数据进行扩展训练
            if not split_result:
                raise ValueError("滚动窗口划分失败：无有效窗口")
            
            self.logger.info(f"   生成 {len(split_result)} 个滚动窗口")
            
            # 保存所有窗口供后续walk-forward使用
            self._rolling_windows = split_result
            
            # 合并策略：使用前80%窗口的训练数据，后20%窗口的测试数据
            n_windows = len(split_result)
            train_window_count = max(1, int(n_windows * 0.8))
            
            self.logger.info(f"   使用前 {train_window_count} 个窗口的训练数据")
            self.logger.info(f"   使用后 {n_windows - train_window_count} 个窗口的测试数据")
            
            # 合并训练数据（前80%窗口）
            train_dfs = []
            for i in range(train_window_count):
                train_df, _ = split_result[i]
                train_dfs.append(train_df)
            combined_train = pd.concat(train_dfs, ignore_index=True)
            
            # 合并测试数据（后20%窗口）
            # 🔴 修复: 测试集需要包含 lookback window 的历史数据
            # 否则 TimeSeriesStockDataset 会因为数据长度不足而丢弃样本
            test_dfs = []
            original_test_dfs = []  # 保存原始测试集（用于确定有效标签日期范围）
            window_size = self.config.window_size
            
            for i in range(train_window_count, n_windows):
                _, test_df = split_result[i]
                original_test_dfs.append(test_df)  # 保存原始测试集
                
                # 获取测试集的开始日期
                if not test_df.empty:
                    test_start_date = test_df[self.config.time_col].min()
                    test_end_date = test_df[self.config.time_col].max()
                    
                    # 向前回溯 window_size * 2 天（留有余量，考虑非交易日）
                    lookback_date = test_start_date - pd.Timedelta(days=window_size * 2)
                    
                    extended_test_df = df[
                        (df[self.config.time_col] >= lookback_date) & 
                        (df[self.config.time_col] <= test_end_date)
                    ].copy()
                    
                    test_dfs.append(extended_test_df)
            
            if test_dfs:
                combined_test = pd.concat(test_dfs, ignore_index=True)
                # 去重，因为可能有重叠
                combined_test = combined_test.drop_duplicates(subset=[self.config.stock_col, self.config.time_col])
                
                # 🆕 记录原始测试集的有效标签起始日期
                if original_test_dfs:
                    original_test_combined = pd.concat(original_test_dfs, ignore_index=True)
                    self._test_valid_label_start_date = pd.Timestamp(original_test_combined[self.config.time_col].min())
                    self.logger.info(f"   测试集有效标签起始日期: {self._test_valid_label_start_date}")
                else:
                    self._test_valid_label_start_date = None
            else:
                combined_test = split_result[-1][1]
                self._test_valid_label_start_date = None
            
            # 将训练集进一步划分为 train/val
            n_train = len(combined_train)
            val_size = int(n_train * self.config.val_ratio / (self.config.train_ratio + self.config.val_ratio))
            
            self._train_df = combined_train.iloc[:-val_size].copy() if val_size > 0 else combined_train
            self._val_df = combined_train.iloc[-val_size:].copy() if val_size > 0 else combined_train.head(0)
            self._test_df = combined_test
            
            self.logger.info(f"   训练集: {len(self._train_df):,} 样本")
            self.logger.info(f"   验证集: {len(self._val_df):,} 样本")
            self.logger.info(f"   测试集(含历史): {len(self._test_df):,} 样本")
        else:
            # 其他splitter返回 (train, val, test)
            self._train_df, self._val_df, self._test_df = split_result
            self._test_valid_label_start_date = None
        
        # 创建数据集
        self.logger.info("=" * 80)
        self.logger.info("步骤 5/5: 创建数据集")
        self.logger.info("=" * 80)
        
        # 🆕 传递测试集的有效标签起始日期
        self._datasets = self.factory.create_datasets(
            self._train_df, self._val_df, self._test_df, feature_cols,
            test_valid_label_start_date=getattr(self, '_test_valid_label_start_date', None)
        )
        
        return self._datasets
    
    def get_dataloaders(self, batch_size: Optional[int] = None,
                       num_workers: Optional[int] = None,
                       shuffle_train: Optional[bool] = None,
                       use_cross_sectional: bool = False) -> LoaderCollection:
        """
        获取数据加载器
        
        Args:
            batch_size: 批量大小（None则使用配置值）
            num_workers: 工作进程数（None则使用配置值）
            shuffle_train: 是否打乱训练集（None则使用配置值）
            use_cross_sectional: 🆕 是否使用截面批采样（IC/相关性损失场景必须开启）
            
        Returns:
            LoaderCollection对象
        """
        if self._datasets is None:
            raise ValueError("未创建数据集，请先调用 create_datasets()")
        
        return self._datasets.get_loaders(
            batch_size=batch_size or self.config.batch_size,
            num_workers=num_workers or self.config.num_workers,
            shuffle_train=shuffle_train if shuffle_train is not None else self.config.shuffle_train,
            use_cross_sectional=use_cross_sectional  # 🆕 透传截面采样参数
        )
    
    def create_daily_loaders(
        self,
        graph_builder_config: Optional[Dict] = None,
        shuffle_dates: Optional[bool] = None,
        device: str = 'cuda'
    ):
        """
        创建日批次数据加载器（用于动态图 GNN 训练）
        
        每个 batch 是一个交易日的所有股票数据，支持动态图构建。
        
        Args:
            graph_builder_config: 图构建器配置，None 则使用 self.config.graph_builder_config
            shuffle_dates: 是否打乱日期顺序，None 则使用 self.config.shuffle_dates
            device: 计算设备
            
        Returns:
            NamedTuple(train, val, test) 包含三个 DailyGraphDataLoader
            
        Example:
            >>> dm = DataManager(config=data_config)
            >>> dm.run_full_pipeline()
            >>> daily_loaders = dm.create_daily_loaders(
            ...     graph_builder_config={'type': 'hybrid', 'alpha': 0.7, 'top_k': 10}
            ... )
            >>> for X, y, adj, stocks, date in daily_loaders.train:
            ...     pred = model(X, adj)
        """
        if self._train_df is None or self._feature_cols is None:
            raise ValueError("未准备数据，请先调用 run_full_pipeline()")
        
        from quantclassic.data_set.graph import (
            DailyBatchDataset, DailyGraphDataLoader
        )
        from quantclassic.data_processor.graph_builder import GraphBuilderFactory
        
        # 使用配置
        gb_config = graph_builder_config or getattr(self.config, 'graph_builder_config', None)
        shuffle = shuffle_dates if shuffle_dates is not None else getattr(self.config, 'shuffle_dates', True)
        
        # 创建图构建器
        graph_builder = None
        if gb_config:
            # 🆕 使用公共辅助函数统一处理配置
            gb_dict = _normalize_graph_builder_config(
                gb_config, self._raw_data, self.config.stock_col, self.logger
            )
            # 🆕 确保 stock_col 透传到图构建器，避免 ts_code 场景退回默认值
            gb_dict.setdefault('stock_col', self.config.stock_col)
            graph_builder = GraphBuilderFactory.create(gb_dict)
            self.logger.info(f"图构建器类型: {gb_dict.get('type', 'corr')}, stock_col: {gb_dict.get('stock_col')}")
        
        # 创建数据集
        def make_daily_dataset(df):
            return DailyBatchDataset(
                df=df,
                feature_cols=self._feature_cols,
                label_col=self.config.label_col,
                window_size=self.config.window_size,
                time_col=self.config.time_col,
                stock_col=self.config.stock_col,
                enable_window_transform=self.config.enable_window_transform,
                window_price_log=self.config.window_price_log,
                window_volume_norm=self.config.window_volume_norm,
                price_cols=self.config.price_cols,
                close_col=self.config.close_col,
                volume_cols=self.config.volume_cols,
                label_rank_normalize=self.config.label_rank_normalize,
                label_rank_output_range=self.config.label_rank_output_range,
            )
        
        # 创建三个数据集
        train_dataset = make_daily_dataset(self._train_df)
        val_dataset = make_daily_dataset(self._val_df) if len(self._val_df) > 0 else None
        test_dataset = make_daily_dataset(self._test_df) if len(self._test_df) > 0 else None
        
        # 创建加载器
        train_loader = DailyGraphDataLoader(
            dataset=train_dataset,
            graph_builder=graph_builder,
            feature_cols=self._feature_cols,
            shuffle_dates=shuffle,
            device=device
        )
        
        val_loader = None
        if val_dataset and len(val_dataset) > 0:
            val_loader = DailyGraphDataLoader(
                dataset=val_dataset,
                graph_builder=graph_builder,
                feature_cols=self._feature_cols,
                shuffle_dates=False,  # 验证集不打乱
                device=device
            )
        
        test_loader = None
        if test_dataset and len(test_dataset) > 0:
            test_loader = DailyGraphDataLoader(
                dataset=test_dataset,
                graph_builder=graph_builder,
                feature_cols=self._feature_cols,
                shuffle_dates=False,  # 测试集不打乱
                device=device
            )
        
        self.logger.info(f"日批次加载器创建完成:")
        self.logger.info(f"  训练集: {len(train_loader)} 天")
        if val_loader:
            self.logger.info(f"  验证集: {len(val_loader)} 天")
        if test_loader:
            self.logger.info(f"  测试集: {len(test_loader)} 天")
        
        # 返回命名元组
        from collections import namedtuple
        DailyLoaderCollection = namedtuple('DailyLoaderCollection', ['train', 'val', 'test'])
        return DailyLoaderCollection(train_loader, val_loader, test_loader)
    
    def create_rolling_daily_loaders(
        self,
        graph_builder_config: Optional[Dict] = None,
        val_ratio: float = 0.15,
        device: str = 'cuda'
    ):
        """
        创建滚动窗口的日批次数据加载器列表（真正的 Walk-Forward）
        
        与 create_daily_loaders 的区别：
        - create_daily_loaders: 合并所有窗口，返回单个 loader 三元组
        - create_rolling_daily_loaders: 保留窗口独立性，返回 loader 三元组列表
        
        Args:
            graph_builder_config: 图构建器配置
            val_ratio: 从每个窗口的训练集中划分验证集的比例
            device: 计算设备
            
        Returns:
            RollingDailyLoaderCollection 对象，包含:
            - windows: List[DailyLoaderCollection]，每个元素是 (train, val, test)
            - n_windows: 窗口数量
            - __iter__: 支持遍历
            
        Example:
            >>> rolling_loaders = dm.create_rolling_daily_loaders()
            >>> for i, loaders in enumerate(rolling_loaders):
            ...     print(f"Window {i+1}: train={len(loaders.train)} days")
            ...     trainer.fit(loaders.train, loaders.val)
            ...     preds = trainer.predict(loaders.test)
        """
        if self.config.split_strategy != 'rolling':
            raise ValueError(
                f"当前 split_strategy='{self.config.split_strategy}'，"
                "请使用 split_strategy='rolling' 以启用滚动窗口模式"
            )
        
        if not hasattr(self, '_rolling_windows') or not self._rolling_windows:
            raise ValueError(
                "滚动窗口数据不可用。请先调用 run_full_pipeline() 生成滚动窗口。"
            )
        
        if self._feature_cols is None:
            raise ValueError("特征列不可用，请先调用 run_full_pipeline()")
        
        from quantclassic.data_set.graph import (
            DailyBatchDataset, DailyGraphDataLoader
        )
        from quantclassic.data_processor.graph_builder import GraphBuilderFactory
        from collections import namedtuple
        
        # 🆕 使用公共辅助函数统一处理配置
        gb_config = graph_builder_config or getattr(self.config, 'graph_builder_config', None)
        gb_dict = _normalize_graph_builder_config(
            gb_config, self._raw_data, self.config.stock_col, self.logger
        )
        graph_builder = GraphBuilderFactory.create(gb_dict) if gb_dict else None
        
        DailyLoaderCollection = namedtuple('DailyLoaderCollection', ['train', 'val', 'test'])
        
        def make_daily_dataset(df, valid_label_start_date=None):
            """创建日批次数据集"""
            return DailyBatchDataset(
                df=df,
                feature_cols=self._feature_cols,
                label_col=self.config.label_col,
                window_size=self.config.window_size,
                time_col=self.config.time_col,
                stock_col=self.config.stock_col,
                enable_window_transform=self.config.enable_window_transform,
                window_price_log=self.config.window_price_log,
                window_volume_norm=self.config.window_volume_norm,
                price_cols=self.config.price_cols,
                close_col=self.config.close_col,
                volume_cols=self.config.volume_cols,
                label_rank_normalize=self.config.label_rank_normalize,
                label_rank_output_range=self.config.label_rank_output_range,
                valid_label_start_date=valid_label_start_date
            )
        
        def make_loader(dataset, shuffle):
            """创建日批次加载器"""
            if dataset is None or len(dataset) == 0:
                return None
            return DailyGraphDataLoader(
                dataset=dataset,
                graph_builder=graph_builder,
                feature_cols=self._feature_cols,
                shuffle_dates=shuffle,
                device=device
            )
        
        # 遍历每个窗口，创建独立的 loader 三元组
        window_loaders = []
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🔄 创建滚动窗口日批次加载器")
        self.logger.info("=" * 80)
        self.logger.info(f"  总窗口数: {len(self._rolling_windows)}")
        
        for i, (train_df, test_df) in enumerate(self._rolling_windows):
            # 1. 从 train_df 中按时间划分出 val_df
            all_dates = sorted(train_df[self.config.time_col].unique())
            n_dates = len(all_dates)
            
            # 需要足够的日期来划分验证集
            min_dates_for_val = self.config.window_size + 20
            if n_dates > min_dates_for_val and val_ratio > 0:
                split_idx = int(n_dates * (1 - val_ratio))
                split_date = all_dates[split_idx]
                
                # 训练集: split_date 之前
                window_train_df = train_df[train_df[self.config.time_col] < split_date].copy()
                
                # 验证集: 需要包含回看窗口
                lookback_idx = max(0, split_idx - self.config.window_size)
                lookback_date = all_dates[lookback_idx]
                window_val_df = train_df[train_df[self.config.time_col] >= lookback_date].copy()
                val_valid_start = pd.Timestamp(split_date)
            else:
                window_train_df = train_df
                window_val_df = None
                val_valid_start = None
            
            # 2. 处理测试集：需要包含回看窗口的历史数据
            if not test_df.empty:
                test_start_date = test_df[self.config.time_col].min()
                test_end_date = test_df[self.config.time_col].max()
                
                # 从 train_df 末尾获取回看窗口
                lookback_start = test_start_date - pd.Timedelta(days=self.config.window_size * 2)
                lookback_df = train_df[train_df[self.config.time_col] >= lookback_start].copy()
                
                if not lookback_df.empty:
                    extended_test_df = pd.concat([lookback_df, test_df], ignore_index=True)
                    extended_test_df = extended_test_df.drop_duplicates(
                        subset=[self.config.stock_col, self.config.time_col]
                    ).sort_values([self.config.stock_col, self.config.time_col])
                    test_valid_start = pd.Timestamp(test_start_date)
                else:
                    extended_test_df = test_df
                    test_valid_start = None
            else:
                extended_test_df = test_df
                test_valid_start = None
            
            # 3. 创建数据集
            train_dataset = make_daily_dataset(window_train_df)
            val_dataset = make_daily_dataset(window_val_df, val_valid_start) if window_val_df is not None else None
            test_dataset = make_daily_dataset(extended_test_df, test_valid_start)
            
            # 4. 创建加载器
            train_loader = make_loader(train_dataset, shuffle=True)
            
            # 如果训练集为空，跳过该窗口
            if train_loader is None or len(train_loader) == 0:
                self.logger.warning(f"  ⚠️ 窗口 {i+1} 训练集为空 (可能窗口太小)，跳过")
                continue
                
            val_loader = make_loader(val_dataset, shuffle=False)
            test_loader = make_loader(test_dataset, shuffle=False)
            
            window_loaders.append(DailyLoaderCollection(train_loader, val_loader, test_loader))
            
            if (i + 1) % 5 == 0 or i == 0:
                self.logger.info(
                    f"  窗口 {i+1}: train={len(train_loader) if train_loader else 0} days, "
                    f"val={len(val_loader) if val_loader else 0} days, "
                    f"test={len(test_loader) if test_loader else 0} days"
                )
        
        self.logger.info(f"\n✅ 已创建 {len(window_loaders)} 个窗口的日批次加载器")
        
        # 返回可迭代的集合类
        class RollingDailyLoaderCollection:
            """滚动窗口日批次加载器集合"""
            def __init__(self, windows):
                self.windows = windows
                self.n_windows = len(windows)
            
            def __len__(self):
                return self.n_windows
            
            def __iter__(self):
                return iter(self.windows)
            
            def __getitem__(self, idx):
                return self.windows[idx]
            
            def enumerate(self):
                """返回 (window_idx, loaders) 迭代器"""
                return enumerate(self.windows)
        
        return RollingDailyLoaderCollection(window_loaders)
    
    def create_rolling_daily_loaders_from_test(
        self,
        graph_builder=None,
        graph_builder_config: Optional[Dict] = None,
        rolling_window_size: Optional[int] = None,
        rolling_step: Optional[int] = None,
        val_ratio: float = 0.15,
        device: str = 'cuda',
    ):
        """
        从已有的 train/val/test 划分创建滚动窗口日批次加载器
        
        与 create_rolling_daily_loaders 的区别：
        - create_rolling_daily_loaders: 要求 split_strategy='rolling'，从 _rolling_windows 获取窗口
        - create_rolling_daily_loaders_from_test: 支持任意 split_strategy，在测试集上滚动生成窗口
        
        滚动逻辑：
        - 合并 train/val/test 为完整数据集
        - 从 test_start_date 开始，每隔 rolling_step 生成一个测试窗口
        - 每个窗口的训练集取测试期前 rolling_window_size 天，并按 val_ratio 划分验证集
        
        Args:
            graph_builder: 图构建器实例（直接传入），优先于 graph_builder_config
            graph_builder_config: 图构建器配置 dict
            rolling_window_size: 滚动窗口训练集大小（天），默认 config.rolling_window_size
            rolling_step: 滚动步长（天），默认 config.rolling_step
            val_ratio: 从训练集中划分验证集的比例
            device: 计算设备
            
        Returns:
            RollingDailyLoaderCollection 对象，可直接传给 RollingDailyTrainer
            
        Example:
            >>> dm.run_full_pipeline()  # split_strategy='time' 或 'ratio'
            >>> loaders = dm.create_rolling_daily_loaders_from_test(
            ...     graph_builder=my_graph_builder,
            ...     rolling_window_size=120,
            ...     rolling_step=20,
            ... )
            >>> results = rolling_trainer.train(loaders, save_dir='...')
        """
        if self._train_df is None or self._feature_cols is None:
            raise ValueError("未准备数据，请先调用 run_full_pipeline()")
        
        from quantclassic.data_set.graph import DailyBatchDataset, DailyGraphDataLoader
        from quantclassic.data_processor.graph_builder import GraphBuilderFactory
        from collections import namedtuple
        from dataclasses import dataclass
        
        # 参数默认值
        rolling_window_size = rolling_window_size or getattr(self.config, 'rolling_window_size', 120)
        rolling_step = rolling_step or getattr(self.config, 'rolling_step', 20)
        test_size = rolling_step  # 测试期长度 = 滚动步长
        
        # 创建图构建器
        if graph_builder is None and graph_builder_config is not None:
            gb_dict = _normalize_graph_builder_config(
                graph_builder_config, self._raw_data, self.config.stock_col, self.logger
            )
            gb_dict.setdefault('stock_col', self.config.stock_col)
            graph_builder = GraphBuilderFactory.create(gb_dict)
        
        # 合并数据
        df_full = pd.concat([self._train_df, self._val_df, self._test_df], ignore_index=True)
        df_full[self.config.time_col] = pd.to_datetime(df_full[self.config.time_col])
        all_dates = sorted(df_full[self.config.time_col].unique())
        
        # 推断测试起始日期
        test_start_date = pd.to_datetime(self._test_df[self.config.time_col].min())
        
        # 计算验证集大小
        val_size = int(rolling_window_size * val_ratio)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🔄 创建滚动窗口日批次加载器 (from_test 模式)")
        self.logger.info("=" * 80)
        self.logger.info(f"  rolling_window_size={rolling_window_size}, rolling_step={rolling_step}")
        self.logger.info(f"  val_size={val_size}, test_size={test_size}")
        self.logger.info(f"  测试起始日期: {test_start_date}")
        
        # 生成滚动窗口日期切分
        test_period_dates = [d for d in all_dates if d >= test_start_date]
        n_windows = (len(test_period_dates) - test_size) // rolling_step + 1
        
        rolling_windows = []
        for w_idx in range(n_windows):
            test_start_idx = w_idx * rolling_step
            test_end_idx = test_start_idx + test_size
            if test_end_idx > len(test_period_dates):
                break
            
            test_dates_w = test_period_dates[test_start_idx:test_end_idx]
            test_start = test_dates_w[0]
            test_start_pos = list(all_dates).index(test_start)
            
            val_start_pos = max(0, test_start_pos - val_size)
            train_end_pos = max(0, val_start_pos)
            train_start_pos = max(0, train_end_pos - rolling_window_size)
            
            train_dates = list(all_dates[train_start_pos:train_end_pos])
            val_dates = list(all_dates[val_start_pos:test_start_pos])
            
            if train_dates and val_dates and test_dates_w:
                rolling_windows.append((train_dates, val_dates, test_dates_w))
        
        self.logger.info(f"  生成 {len(rolling_windows)} 个滚动窗口")
        
        # 公共数据集参数
        common_kwargs = dict(
            feature_cols=self._feature_cols,
            label_col=self.config.label_col,
            window_size=self.config.window_size,
            time_col=self.config.time_col,
            stock_col=self.config.stock_col,
            enable_window_transform=self.config.enable_window_transform,
            window_price_log=self.config.window_price_log,
            window_volume_norm=self.config.window_volume_norm,
            price_cols=self.config.price_cols,
            close_col=self.config.close_col,
            volume_cols=self.config.volume_cols,
            label_rank_normalize=self.config.label_rank_normalize,
            label_rank_output_range=self.config.label_rank_output_range,
        )
        
        def make_daily_dataset(dates_list, valid_label_start_date=None):
            df_subset = df_full[df_full[self.config.time_col].isin(dates_list)].copy()
            return DailyBatchDataset(df=df_subset, valid_label_start_date=valid_label_start_date, **common_kwargs)
        
        def make_loader(dataset, shuffle):
            if dataset is None or len(dataset) == 0:
                return None
            return DailyGraphDataLoader(
                dataset=dataset,
                graph_builder=graph_builder,
                feature_cols=self._feature_cols,
                shuffle_dates=shuffle,
                device=device,
                num_workers=0,
                pin_memory=False,
            )
        
        # 用于兼容 RollingDailyTrainer 的 WindowLoaders 类
        @dataclass
        class WindowLoaders:
            train: DailyGraphDataLoader
            val: DailyGraphDataLoader
            test: DailyGraphDataLoader
            train_dates: list
            val_dates: list
            test_dates: list
        
        DailyLoaderCollection = namedtuple('DailyLoaderCollection', ['train', 'val', 'test'])
        
        # 计算有效标签起始日期（避免窗口首部无标签）
        valid_label_start_date = all_dates[self.config.window_size] if len(all_dates) > self.config.window_size else None
        
        window_loaders = []
        for w_idx, (train_dates, val_dates, test_dates_w) in enumerate(rolling_windows):
            train_dataset = make_daily_dataset(train_dates, valid_label_start_date if w_idx == 0 else None)
            val_dataset = make_daily_dataset(val_dates)
            test_dataset = make_daily_dataset(test_dates_w)
            
            train_loader = make_loader(train_dataset, shuffle=True)
            val_loader = make_loader(val_dataset, shuffle=False)
            test_loader = make_loader(test_dataset, shuffle=False)
            
            if train_loader is None or len(train_loader) == 0:
                self.logger.warning(f"  ⚠️ 窗口 {w_idx+1} 训练集为空，跳过")
                continue
            
            window_loaders.append(WindowLoaders(
                train=train_loader, val=val_loader, test=test_loader,
                train_dates=train_dates, val_dates=val_dates, test_dates=test_dates_w
            ))
            
            if w_idx == 0:
                self.logger.info(f"  窗口 1: train={len(train_dates)}天, val={len(val_dates)}天, test={len(test_dates_w)}天")
        
        self.logger.info(f"\n✅ 已创建 {len(window_loaders)} 个窗口的日批次加载器")
        
        # 返回可迭代集合
        class RollingDailyLoaderCollection:
            def __init__(self, windows):
                self.windows = windows
                self.n_windows = len(windows)
            def __len__(self):
                return self.n_windows
            def __iter__(self):
                return iter(self.windows)
            def __getitem__(self, idx):
                return self.windows[idx]
            def enumerate(self):
                return enumerate(self.windows)
        
        return RollingDailyLoaderCollection(window_loaders)
    
    def run_full_pipeline(self, file_path: Optional[str] = None,
                         validate: bool = True,
                         auto_filter_features: bool = True) -> LoaderCollection:
        """
        运行完整的数据处理流水线
        
        Args:
            file_path: 数据文件路径
            validate: 是否验证数据质量
            auto_filter_features: 是否自动过滤特征
            
        Returns:
            LoaderCollection对象
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🚀 开始完整数据处理流水线")
        self.logger.info("=" * 80 + "\n")
        
        # 1. 加载数据
        self.load_raw_data(file_path)
        
        # 2. 验证数据（可选）
        if validate and self.config.enable_validation:
            report = self.validate_data_quality()
            if not report.is_valid:
                self.logger.warning("⚠️  数据验证未通过，但继续处理")
        
        # 3. 特征工程
        self.preprocess_features(auto_filter=auto_filter_features)
        
        # 4-5. 创建数据集
        self.create_datasets()
        
        # 6. 创建数据加载器
        # 🆕 根据 use_daily_batch 配置决定返回类型
        use_daily = getattr(self.config, 'use_daily_batch', False)
        if use_daily:
            self.logger.info("🆕 use_daily_batch=True，创建日批次加载器")
            loaders = self.create_daily_loaders(
                graph_builder_config=getattr(self.config, 'graph_builder_config', None),
                shuffle_dates=getattr(self.config, 'shuffle_dates', True)
            )
        else:
            loaders = self.get_dataloaders()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("✅ 完整数据处理流水线完成")
        self.logger.info("=" * 80 + "\n")
        
        # 打印摘要
        self._print_pipeline_summary()
        
        return loaders
    
    def _print_pipeline_summary(self):
        """打印流水线摘要"""
        print("\n" + "=" * 80)
        print("📊 数据处理摘要")
        print("=" * 80)
        
        if self._raw_data is not None:
            print(f"原始数据: {len(self._raw_data):,} 行")
        
        if self._feature_cols is not None:
            print(f"特征数量: {len(self._feature_cols)}")
        
        if self._datasets is not None:
            print(f"\n数据集:")
            print(f"  训练集: {self._datasets.metadata['train_samples']:,} 样本")
            print(f"  验证集: {self._datasets.metadata['val_samples']:,} 样本")
            print(f"  测试集: {self._datasets.metadata['test_samples']:,} 样本")
        
        print(f"\n配置:")
        print(f"  窗口大小: {self.config.window_size}")
        print(f"  批量大小: {self.config.batch_size}")
        print(f"  划分策略: {self.config.split_strategy}")
        
        print("=" * 80 + "\n")
    
    def _save_validation_report(self, report: ValidationReport):
        """保存验证报告"""
        report_dir = os.path.join(self.config.output_dir, 'reports')
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(report_dir, f'validation_report_{timestamp}.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("数据验证报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"状态: {'通过' if report.is_valid else '失败'}\n\n")
            
            if report.errors:
                f.write(f"错误 ({len(report.errors)}):\n")
                for i, error in enumerate(report.errors, 1):
                    f.write(f"  {i}. {error}\n")
                f.write("\n")
            
            if report.warnings:
                f.write(f"警告 ({len(report.warnings)}):\n")
                for i, warning in enumerate(report.warnings, 1):
                    f.write(f"  {i}. {warning}\n")
                f.write("\n")
            
            if report.stats:
                f.write("统计信息:\n")
                for key, value in report.stats.items():
                    f.write(f"  {key}: {value}\n")
        
        self.logger.info(f"📄 验证报告已保存: {report_path}")
    
    def save_state(self, save_path: Optional[str] = None):
        """保存管理器状态"""
        if save_path is None:
            save_path = os.path.join(self.config.cache_dir, 'manager_state.pkl')
        
        state = {
            'config': self.config,
            'feature_cols': self._feature_cols,
            'train_df': self._train_df,
            'val_df': self._val_df,
            'test_df': self._test_df,
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        
        self.logger.info(f"💾 状态已保存: {save_path}")
    
    def load_state(self, load_path: Optional[str] = None):
        """加载管理器状态"""
        if load_path is None:
            load_path = os.path.join(self.config.cache_dir, 'manager_state.pkl')
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"状态文件不存在: {load_path}")
        
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state['config']
        self._feature_cols = state['feature_cols']
        self._train_df = state['train_df']
        self._val_df = state['val_df']
        self._test_df = state['test_df']
        
        self.logger.info(f"📁 状态已加载: {load_path}")
    
    @property
    def raw_data(self) -> Optional[pd.DataFrame]:
        """获取原始数据"""
        return self._raw_data
    
    @property
    def feature_cols(self) -> Optional[List[str]]:
        """获取特征列"""
        return self._feature_cols
    
    @property
    def datasets(self) -> Optional[DatasetCollection]:
        """获取数据集集合"""
        return self._datasets
    
    @property
    def split_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """获取划分后的数据"""
        return self._train_df, self._val_df, self._test_df
    
    def create_rolling_window_trainer(
        self, 
        stock_universe: Optional[List[str]] = None
    ):
        """
        ⚠️ 已废弃并移除 - 请使用 model.train.RollingWindowTrainer 或 RollingDailyTrainer
        
        .. deprecated:: 2026.01
            数据层不应包含训练循环。此方法已移除。
            
            请改用:
            >>> from quantclassic.model.train import RollingWindowTrainer, RollingDailyTrainer
            >>> rolling_loaders = dm.create_rolling_daily_loaders()
            >>> trainer = RollingDailyTrainer(model_factory=..., config=...)
            >>> trainer.fit(rolling_loaders)
        
        Raises:
            DeprecationWarning: 始终抛出，指导用户迁移到新 API
        """
        raise NotImplementedError(
            "\n" + "=" * 70 + "\n"
            "⚠️  DataManager.create_rolling_window_trainer() 已移除！\n\n"
            "数据层不应包含训练循环。请改用 model.train 模块:\n\n"
            "    from quantclassic.model.train import RollingDailyTrainer, RollingTrainerConfig\n\n"
            "    # 1. 创建滚动日批次加载器\n"
            "    rolling_loaders = dm.create_rolling_daily_loaders()\n\n"
            "    # 2. 定义模型工厂\n"
            "    def model_factory():\n"
            "        return MyModel(d_feat=len(feature_cols))\n\n"
            "    # 3. 创建训练器并训练\n"
            "    config = RollingTrainerConfig(n_epochs=20, weight_inheritance=True)\n"
            "    trainer = RollingDailyTrainer(model_factory, config)\n"
            "    trainer.fit(rolling_loaders)\n"
            + "=" * 70
        )


if __name__ == '__main__':
    # 测试DataManager
    print("=" * 80)
    print("DataManager 测试")
    print("=" * 80)
    
    # 创建配置
    config = DataConfig(
        base_dir='rq_data_parquet',
        data_file='train_data_final.parquet',
        window_size=40,
        batch_size=256,
        split_strategy='time_series',
        enable_validation=True,
        verbose=True
    )
    
    # 创建管理器
    manager = DataManager(config)
    
    try:
        # 运行完整流水线
        loaders = manager.run_full_pipeline()
        
        # 测试数据加载器
        print("\n测试数据加载器:")
        batch_x, batch_y = next(iter(loaders.train))
        print(f"  批次特征形状: {batch_x.shape}")
        print(f"  批次标签形状: {batch_y.shape}")
        
        print("\n✅ DataManager 测试完成")
        
    except FileNotFoundError:
        print("\n⚠️  测试数据文件不存在")
        print("✅ DataManager 类定义完成")
