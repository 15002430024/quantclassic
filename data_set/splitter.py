"""
DataSplitter - 数据划分器

实现多种数据划分策略：时间序列、分层、滚动窗口等
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from abc import ABC, abstractmethod
import logging
from .config import DataConfig


class DataSplitter(ABC):
    """数据划分器抽象基类"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """配置日志"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """划分数据为训练集、验证集、测试集"""
        pass


class TimeSeriesSplitter(DataSplitter):
    """时间序列划分器 - 按时间顺序划分"""
    
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        按时间序列划分数据
        
        Args:
            df: 输入数据
            
        Returns:
            (train_df, val_df, test_df)
        """
        self.logger.info("📅 时间序列划分...")
        
        # 按时间排序
        df = df.sort_values(self.config.time_col).reset_index(drop=True)
        
        # 获取唯一日期
        unique_dates = df[self.config.time_col].unique()
        n_dates = len(unique_dates)
        
        # 计算切点
        if self.config.train_end_date and self.config.val_end_date:
            # 使用指定日期
            train_end = pd.to_datetime(self.config.train_end_date)
            val_end = pd.to_datetime(self.config.val_end_date)
        else:
            # 使用比例
            train_idx = int(n_dates * self.config.train_ratio)
            val_idx = int(n_dates * (self.config.train_ratio + self.config.val_ratio))
            
            train_end = unique_dates[train_idx]
            val_end = unique_dates[val_idx]
        
        # 划分数据
        train_df = df[df[self.config.time_col] <= train_end].copy()
        val_df = df[(df[self.config.time_col] > train_end) & 
                    (df[self.config.time_col] <= val_end)].copy()
        test_df = df[df[self.config.time_col] > val_end].copy()
        
        # 输出统计
        self.logger.info(f"   训练集: {len(train_df):,} 行 "
                        f"({train_df[self.config.time_col].min()} ~ {train_df[self.config.time_col].max()})")
        self.logger.info(f"   验证集: {len(val_df):,} 行 "
                        f"({val_df[self.config.time_col].min()} ~ {val_df[self.config.time_col].max()})")
        self.logger.info(f"   测试集: {len(test_df):,} 行 "
                        f"({test_df[self.config.time_col].min()} ~ {test_df[self.config.time_col].max()})")
        
        return train_df, val_df, test_df


class StratifiedStockSplitter(DataSplitter):
    """分层股票划分器 - 按股票分层划分"""
    
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        按股票分层划分数据（确保每只股票都在各数据集中）
        
        Args:
            df: 输入数据
            
        Returns:
            (train_df, val_df, test_df)
        """
        self.logger.info("📊 分层股票划分...")
        
        # 按时间和股票排序
        df = df.sort_values([self.config.stock_col, self.config.time_col]).reset_index(drop=True)
        
        train_list, val_list, test_list = [], [], []
        
        # 对每只股票单独划分
        for stock_code, stock_df in df.groupby(self.config.stock_col):
            n = len(stock_df)
            
            # 计算切点
            train_end = int(n * self.config.train_ratio)
            val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
            
            # 划分
            train_list.append(stock_df.iloc[:train_end])
            val_list.append(stock_df.iloc[train_end:val_end])
            test_list.append(stock_df.iloc[val_end:])
        
        # 合并
        train_df = pd.concat(train_list, ignore_index=True)
        val_df = pd.concat(val_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)
        
        # 输出统计
        self.logger.info(f"   训练集: {len(train_df):,} 行, "
                        f"{train_df[self.config.stock_col].nunique()} 只股票")
        self.logger.info(f"   验证集: {len(val_df):,} 行, "
                        f"{val_df[self.config.stock_col].nunique()} 只股票")
        self.logger.info(f"   测试集: {len(test_df):,} 行, "
                        f"{test_df[self.config.stock_col].nunique()} 只股票")
        
        return train_df, val_df, test_df


class RollingWindowSplitter(DataSplitter):
    """滚动窗口划分器 - 用于时间序列交叉验证"""
    
    def split(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        滚动窗口划分（返回多个训练-测试对）
        
        Args:
            df: 输入数据
            
        Returns:
            [(train_df_1, test_df_1), (train_df_2, test_df_2), ...]
        """
        self.logger.info("🔄 滚动窗口划分...")
        
        # 按时间排序
        df = df.sort_values(self.config.time_col).reset_index(drop=True)
        
        # 【修复】确保数据中有有效标签（避免空窗口）
        if self.config.label_col in df.columns:
            original_len = len(df)
            df = df[df[self.config.label_col].notna()].copy()
            if len(df) < original_len:
                self.logger.info(f"   过滤标签缺失数据: {original_len:,} -> {len(df):,}")
        
        # 获取唯一日期
        unique_dates = sorted(df[self.config.time_col].unique())
        
        window_size = self.config.rolling_window_size
        step = self.config.rolling_step
        
        splits = []
        start_idx = 0
        
        while start_idx + window_size < len(unique_dates):
            # 训练窗口
            train_start = unique_dates[start_idx]
            train_end = unique_dates[start_idx + window_size - 1]
            
            # 测试窗口（下一个step期）
            test_start = unique_dates[start_idx + window_size]
            test_end_idx = min(start_idx + window_size + step, len(unique_dates) - 1)
            test_end = unique_dates[test_end_idx]
            
            # 划分数据
            train_df = df[(df[self.config.time_col] >= train_start) & 
                         (df[self.config.time_col] <= train_end)].copy()
            test_df = df[(df[self.config.time_col] >= test_start) & 
                        (df[self.config.time_col] <= test_end)].copy()
            
            splits.append((train_df, test_df))
            
            # 移动窗口
            start_idx += step
        
        self.logger.info(f"   生成 {len(splits)} 个滚动窗口")
        self.logger.info(f"   窗口大小: {window_size} 天, 步长: {step} 天")
        
        return splits


class RandomSplitter(DataSplitter):
    """随机划分器 - 传统机器学习划分（不推荐用于时序数据）"""
    
    def __init__(self, config: DataConfig, random_state: int = 42):
        super().__init__(config)
        self.random_state = random_state
    
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        随机划分数据
        
        Args:
            df: 输入数据
            
        Returns:
            (train_df, val_df, test_df)
        """
        self.logger.info("🎲 随机划分...")
        self.logger.warning("⚠️  警告: 随机划分不适合时序数据，可能导致数据泄漏")
        
        # 随机打乱
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        self.logger.info(f"   训练集: {len(train_df):,} 行")
        self.logger.info(f"   验证集: {len(val_df):,} 行")
        self.logger.info(f"   测试集: {len(test_df):,} 行")
        
        return train_df, val_df, test_df


def create_splitter(config: DataConfig) -> DataSplitter:
    """
    根据配置创建划分器
    
    Args:
        config: 配置对象
        
    Returns:
        DataSplitter实例
    """
    strategy = config.split_strategy.lower()
    
    if strategy == 'time_series':
        return TimeSeriesSplitter(config)
    elif strategy == 'stratified':
        return StratifiedStockSplitter(config)
    elif strategy == 'rolling':
        return RollingWindowSplitter(config)
    elif strategy == 'random':
        return RandomSplitter(config)
    else:
        raise ValueError(f"未知的划分策略: {strategy}")


if __name__ == '__main__':
    # 测试数据划分器
    from config import DataConfig
    
    print("=" * 80)
    print("DataSplitter 测试")
    print("=" * 80)
    
    # 创建模拟数据
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    stocks = ['000001.SZ', '000002.SZ', '600000.SH']
    
    data = []
    for stock in stocks:
        for date in dates:
            data.append({
                'ts_code': stock,
                'trade_date': date,
                'y_processed': np.random.randn(),
                'feature1': np.random.randn(),
            })
    
    df = pd.DataFrame(data)
    
    # 测试时间序列划分
    print("\n1. 时间序列划分:")
    config = DataConfig(split_strategy='time_series')
    splitter = TimeSeriesSplitter(config)
    train, val, test = splitter.split(df)
    
    # 测试分层划分
    print("\n2. 分层股票划分:")
    config = DataConfig(split_strategy='stratified')
    splitter = StratifiedStockSplitter(config)
    train, val, test = splitter.split(df)
    
    # 测试滚动窗口
    print("\n3. 滚动窗口划分:")
    config = DataConfig(split_strategy='rolling', rolling_window_size=100, rolling_step=50)
    splitter = RollingWindowSplitter(config)
    splits = splitter.split(df)
    
    print("\n✅ 数据划分器测试完成")
