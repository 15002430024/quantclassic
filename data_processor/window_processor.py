"""
窗口级数据处理模块

实现研报标准的价格对数变换和成交量标准化：
1. 价格对数变换：log(price_{t-i} / close_t)
2. 成交量标准化：volume_{t-i} / mean(volume_window)

特点：
- 支持在数据预处理阶段或Dataset阶段使用
- 与现有的 DataPreprocessor 管道无缝集成
- 保留原始列的同时生成变换后的新列（可选）

Author: QuantClassic
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, field
from tqdm.auto import tqdm
import logging

logger = logging.getLogger(__name__)


@dataclass
class WindowProcessConfig:
    """
    窗口处理配置
    
    Args:
        window_size: 窗口大小（交易日数）
        price_columns: 需要进行对数变换的价格列
        volume_columns: 需要进行均值标准化的成交量列
        close_column: 收盘价列名（作为价格变换的基准）
        stock_column: 股票代码列名（兼容 order_book_id/ts_code）
        time_column: 时间列名（兼容 trade_date/date）
        keep_original: 是否保留原始列（True则创建新列，False则覆盖）
        suffix: 变换后列名后缀（仅当keep_original=True时使用）
        min_window_ratio: 窗口内有效数据的最小比例（低于此比例则跳过）
        
    ⚠️ 重要提示（防止重复转换）:
        窗口转换可以在两个地方执行：
        1. data_processor/WindowProcessor（离线预处理）
        2. data_set/factory.py 的 TimeSeriesStockDataset（运行时转换）
        
        请确保只在其中一处执行，否则会导致特征变形！
        
        推荐方案：
        - 如果使用 DatasetFactory 创建 Dataset，设置 enable_window_transform=False
        - 或者不使用 WindowProcessor，让 Dataset 在运行时处理（更灵活）
    """
    window_size: int = 60
    price_columns: List[str] = field(default_factory=lambda: ['open', 'high', 'low', 'close', 'vwap'])
    volume_columns: List[str] = field(default_factory=lambda: ['vol', 'amount'])
    close_column: str = 'close'
    stock_column: str = 'order_book_id'  # 兼容 ts_code
    time_column: str = 'trade_date'  # 兼容 date
    keep_original: bool = False
    suffix: str = '_log'
    min_window_ratio: float = 0.8


# 全局标记：用于检测数据是否已经过窗口转换
_WINDOW_TRANSFORM_MARKER = '__window_transformed__'


class WindowProcessor:
    """
    窗口级数据处理器
    
    实现研报标准的数据变换方法：
    
    1. 价格对数变换：
       - 公式：log(price_{t-i} / close_t)
       - 含义：将窗口内所有价格除以窗口最后一天的收盘价，然后取对数
       - 效果：当天收盘价变为0，历史价格变为相对涨跌幅
       
    2. 成交量/成交额标准化：
       - 公式：volume_{t-i} / mean(volume_window)
       - 含义：将窗口内的成交量除以该窗口的平均成交量
       - 效果：数据变为倍数概念（如1.5倍均值）
    
    ⚠️ 防止重复转换：
        本类与 data_set.factory.TimeSeriesStockDataset 中的窗口转换功能重叠。
        请确保只使用其中一个！
        
        - 使用本类（离线模式）：适合固定窗口、一次性预处理
        - 使用 Dataset（运行时模式）：适合动态窗口、灵活实验
    
    使用场景：
    - 场景1：在Dataset的__getitem__中使用（推荐）
    - 场景2：预先处理整个数据集（仅用于固定窗口场景）
    
    示例：
        # 场景1：在Dataset中使用
        processor = WindowProcessor(config)
        window_data = processor.process_window(df_window)
        
        # 场景2：预处理整个数据集
        processor = WindowProcessor(config)
        df_processed = processor.process_dataset(df)
    """
    
    def __init__(self, config: Optional[WindowProcessConfig] = None):
        """
        初始化窗口处理器
        
        Args:
            config: 窗口处理配置，如果为None则使用默认配置
        """
        self.config = config or WindowProcessConfig()
        self._adapt_column_names()  # 自适应列名
        logger.info(f"初始化窗口处理器: window_size={self.config.window_size}")
    
    def _adapt_column_names(self, df: pd.DataFrame = None):
        """
        根据常见命名约定自适应列名
        
        Args:
            df: 可选的数据框，如果提供则根据实际列名适配
               如果不提供，仅在初始化时做基本校验
        
        Note:
            - 初始化时不传 df，仅保留默认配置
            - process_dataset 时传入 df，执行实际检测并更新 config
        """
        if df is None:
            # 初始化阶段：无数据，跳过检测
            return
        
        # 检测并适配股票列
        if self.config.stock_column not in df.columns:
            for col in ['order_book_id', 'ts_code', 'stock_code', 'symbol']:
                if col in df.columns:
                    logger.info(f"WindowProcessor: 股票列自适应 {self.config.stock_column} -> {col}")
                    self.config.stock_column = col
                    break
        
        # 检测并适配时间列
        if self.config.time_column not in df.columns:
            for col in ['trade_date', 'date', 'datetime', 'time']:
                if col in df.columns:
                    logger.info(f"WindowProcessor: 时间列自适应 {self.config.time_column} -> {col}")
                    self.config.time_column = col
                    break
    
    @staticmethod
    def is_transformed(df: pd.DataFrame) -> bool:
        """
        检查数据是否已经过窗口转换
        
        Args:
            df: 数据框
            
        Returns:
            True 如果数据已经过转换
        """
        return hasattr(df, 'attrs') and df.attrs.get(_WINDOW_TRANSFORM_MARKER, False)
    
    @staticmethod
    def mark_transformed(df: pd.DataFrame) -> pd.DataFrame:
        """
        标记数据已经过窗口转换
        
        Args:
            df: 数据框
            
        Returns:
            标记后的数据框
        """
        if not hasattr(df, 'attrs'):
            df.attrs = {}
        df.attrs[_WINDOW_TRANSFORM_MARKER] = True
        return df
    
    def process_window(
        self, 
        window_df: pd.DataFrame,
        inplace: bool = False,
        skip_if_transformed: bool = True
    ) -> pd.DataFrame:
        """
        处理单个窗口的数据
        
        这是核心方法，适合在Dataset的__getitem__中调用。
        
        Args:
            window_df: 单个窗口的数据（已按时间排序）
            inplace: 是否原地修改
            skip_if_transformed: 如果数据已转换，是否跳过（防止重复转换）
        
        Returns:
            处理后的窗口数据
            
        示例：
            # 在Dataset中使用
            class StockDataset(Dataset):
                def __getitem__(self, idx):
                    window = self.data.iloc[idx:idx+self.window_size]
                    window = self.processor.process_window(window)
                    return window
        """
        # 防止重复转换
        if skip_if_transformed and self.is_transformed(window_df):
            logger.debug("数据已经过窗口转换，跳过处理")
            return window_df
        
        if not inplace:
            window_df = window_df.copy()
        
        # 获取窗口最后一天的收盘价（作为基准）
        close_t = window_df[self.config.close_column].iloc[-1]
        
        if pd.isna(close_t) or close_t == 0:
            logger.warning(f"窗口最后一天收盘价无效: {close_t}")
            return window_df
        
        # 1. 价格对数变换：log(price / close_t)
        for col in self.config.price_columns:
            if col not in window_df.columns:
                continue
            
            target_col = f"{col}{self.config.suffix}" if self.config.keep_original else col
            
            with np.errstate(divide='ignore', invalid='ignore'):
                window_df[target_col] = np.log(window_df[col] / close_t)
            
            # 处理无效值
            window_df[target_col] = window_df[target_col].replace([np.inf, -np.inf], np.nan)
        
        # 2. 成交量/成交额标准化：volume / mean(volume)
        for col in self.config.volume_columns:
            if col not in window_df.columns:
                continue
            
            target_col = f"{col}{self.config.suffix}" if self.config.keep_original else col
            
            col_mean = window_df[col].mean()
            
            if pd.notna(col_mean) and col_mean != 0:
                window_df[target_col] = window_df[col] / col_mean
            else:
                window_df[target_col] = np.nan
        
        # 标记已转换
        window_df = self.mark_transformed(window_df)
        
        return window_df
    
    def process_dataset(
        self,
        df: pd.DataFrame,
        show_progress: bool = True,
        skip_if_transformed: bool = True
    ) -> pd.DataFrame:
        """
        处理整个数据集（按股票分组，滚动窗口处理）
        
        ⚠️ 重要提示：
            此方法与 data_set.factory.TimeSeriesStockDataset 中的窗口转换功能重叠。
            如果您使用 DatasetFactory 创建 Dataset 并启用了 enable_window_transform，
            请不要使用此方法，否则会导致重复转换！
        
        注意：此方法会为每个时间点生成基于其过去window_size天的变换结果。
        这意味着：
        - 同一天的不同股票有不同的基准价格
        - 每一行的变换结果只依赖于其历史数据
        
        Args:
            df: 完整数据集（必须包含stock_column和time_column）
            show_progress: 是否显示进度条
            skip_if_transformed: 如果数据已转换，是否跳过
        
        Returns:
            处理后的数据集
            
        说明：
            对于每个股票的每个时间点t，使用[t-window_size+1, t]的窗口数据，
            以t时刻的close作为基准进行变换。
        """
        print("\n" + "=" * 80)
        print("📊 窗口级数据处理")
        print("=" * 80)
        print(f"  窗口大小: {self.config.window_size}")
        print(f"  价格列: {self.config.price_columns}")
        print(f"  成交量列: {self.config.volume_columns}")
        print(f"  保留原始列: {self.config.keep_original}")
        
        # 防止重复转换
        if skip_if_transformed and self.is_transformed(df):
            print("  ⚠️ 警告: 数据已经过窗口转换，跳过处理以防止重复转换")
            logger.warning("数据已经过窗口转换，跳过 process_dataset 以防止重复转换")
            return df
        
        df = df.copy()
        
        # 🆕 使用统一的列名自适应方法（会更新 config）
        self._adapt_column_names(df)
        stock_col = self.config.stock_column
        time_col = self.config.time_column
        
        if stock_col in df.columns:
            print(f"  📝 股票列: {stock_col}")
        if time_col in df.columns:
            print(f"  📝 时间列: {time_col}")
        
        # 确保按股票和时间排序
        df = df.sort_values([stock_col, time_col])
        
        # 获取所有股票
        stocks = df[stock_col].unique()
        print(f"  股票数量: {len(stocks)}")
        
        # 结果存储
        results = []
        
        # 按股票分组处理
        stock_iter = tqdm(stocks, desc="处理股票", unit="只") if show_progress else stocks
        
        for stock in stock_iter:
            stock_df = df[df[stock_col] == stock].copy()
            stock_df = stock_df.reset_index(drop=True)
            
            n_rows = len(stock_df)
            window_size = self.config.window_size
            
            # 对每个有效时间点进行窗口处理
            for i in range(n_rows):
                # 窗口起始位置
                start_idx = max(0, i - window_size + 1)
                end_idx = i + 1
                
                # 获取窗口数据
                window = stock_df.iloc[start_idx:end_idx].copy()
                
                # 检查窗口是否足够大
                actual_window_size = len(window)
                if actual_window_size < window_size * self.config.min_window_ratio:
                    # 窗口太小，跳过变换，保留原值
                    continue
                
                # 获取当前行的收盘价作为基准
                close_t = stock_df.iloc[i][self.config.close_column]
                
                if pd.isna(close_t) or close_t == 0:
                    continue
                
                # 对当前行进行变换
                current_row = stock_df.iloc[i:i+1].copy()
                
                # 1. 价格对数变换
                for col in self.config.price_columns:
                    if col not in current_row.columns:
                        continue
                    
                    target_col = f"{col}{self.config.suffix}" if self.config.keep_original else col
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        current_row[target_col] = np.log(current_row[col] / close_t)
                    
                    current_row[target_col] = current_row[target_col].replace([np.inf, -np.inf], np.nan)
                
                # 2. 成交量标准化（使用窗口均值）
                for col in self.config.volume_columns:
                    if col not in window.columns:
                        continue
                    
                    target_col = f"{col}{self.config.suffix}" if self.config.keep_original else col
                    
                    col_mean = window[col].mean()
                    
                    if pd.notna(col_mean) and col_mean != 0:
                        current_row[target_col] = current_row[col] / col_mean
                    else:
                        current_row[target_col] = np.nan
                
                # 更新原数据
                for col in self.config.price_columns + self.config.volume_columns:
                    if col in current_row.columns:
                        target_col = f"{col}{self.config.suffix}" if self.config.keep_original else col
                        if target_col in current_row.columns:
                            stock_df.loc[i, target_col] = current_row[target_col].iloc[0]
            
            results.append(stock_df)
        
        # 合并结果
        df_processed = pd.concat(results, ignore_index=True)
        
        print(f"\n✅ 窗口处理完成!")
        print(f"  处理后形状: {df_processed.shape}")
        
        # 统计信息
        if not self.config.keep_original:
            print(f"\n【处理后统计】")
            all_cols = self.config.price_columns + self.config.volume_columns
            valid_cols = [c for c in all_cols if c in df_processed.columns]
            if valid_cols:
                print(df_processed[valid_cols].describe())
        
        print("=" * 80)
        
        return df_processed
    
    def process_dataset_vectorized(
        self,
        df: pd.DataFrame,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        向量化处理整个数据集（更高效的版本）
        
        使用向量化操作代替显式循环，适合大规模数据。
        
        原理：
        - 价格变换：每行除以自己的close，然后取对数
        - 成交量变换：使用滚动均值
        
        Args:
            df: 完整数据集
            show_progress: 是否显示进度条
        
        Returns:
            处理后的数据集
        """
        print("\n" + "=" * 80)
        print("📊 窗口级数据处理（向量化版本）")
        print("=" * 80)
        print(f"  窗口大小: {self.config.window_size}")
        print(f"  价格列: {self.config.price_columns}")
        print(f"  成交量列: {self.config.volume_columns}")
        
        df = df.copy()
        
        # 确保按股票和时间排序
        df = df.sort_values([self.config.stock_column, self.config.time_column])
        
        # 1. 价格对数变换：log(price / close)
        # 注意：这里每行用自己的close作为基准（适用于预处理场景）
        print("\n【价格对数变换】")
        close_values = df[self.config.close_column].values
        
        for col in self.config.price_columns:
            if col not in df.columns:
                print(f"  ⚠️ 列 {col} 不存在，跳过")
                continue
            
            target_col = f"{col}{self.config.suffix}" if self.config.keep_original else col
            
            with np.errstate(divide='ignore', invalid='ignore'):
                df[target_col] = np.log(df[col].values / close_values)
            
            df[target_col] = df[target_col].replace([np.inf, -np.inf], np.nan)
            
            valid_count = df[target_col].notna().sum()
            print(f"  ✓ {col} -> {target_col}: 有效值 {valid_count}/{len(df)}")
        
        # 2. 成交量标准化：使用滚动均值
        print("\n【成交量标准化】")
        
        for col in self.config.volume_columns:
            if col not in df.columns:
                print(f"  ⚠️ 列 {col} 不存在，跳过")
                continue
            
            target_col = f"{col}{self.config.suffix}" if self.config.keep_original else col
            
            # 按股票分组计算滚动均值
            rolling_mean = df.groupby(self.config.stock_column)[col].transform(
                lambda x: x.rolling(window=self.config.window_size, min_periods=1).mean()
            )
            
            # 标准化
            with np.errstate(divide='ignore', invalid='ignore'):
                df[target_col] = df[col] / rolling_mean
            
            df[target_col] = df[target_col].replace([np.inf, -np.inf], np.nan)
            
            valid_count = df[target_col].notna().sum()
            print(f"  ✓ {col} -> {target_col}: 有效值 {valid_count}/{len(df)}")
        
        print(f"\n✅ 向量化处理完成!")
        print(f"  处理后形状: {df.shape}")
        
        # 统计信息
        print(f"\n【处理后统计】")
        all_cols = self.config.price_columns + self.config.volume_columns
        if self.config.keep_original:
            all_cols = [f"{c}{self.config.suffix}" for c in all_cols]
        valid_cols = [c for c in all_cols if c in df.columns]
        if valid_cols:
            print(df[valid_cols].describe())
        
        print("=" * 80)
        
        return df


# 便捷函数
def process_price_log_transform(
    df: pd.DataFrame,
    price_columns: List[str] = None,
    close_column: str = 'close',
    stock_column: str = 'order_book_id',
    keep_original: bool = False,
    suffix: str = '_log'
) -> pd.DataFrame:
    """
    便捷函数：对价格列进行对数变换
    
    公式：log(price / close)
    
    Args:
        df: 数据框
        price_columns: 价格列列表，如果为None则使用默认值
        close_column: 收盘价列名
        stock_column: 股票代码列名
        keep_original: 是否保留原始列
        suffix: 变换后列名后缀
    
    Returns:
        变换后的数据框
    """
    if price_columns is None:
        price_columns = ['open', 'high', 'low', 'close', 'vwap']
    
    df = df.copy()
    close_values = df[close_column].values
    
    for col in price_columns:
        if col not in df.columns:
            continue
        
        target_col = f"{col}{suffix}" if keep_original else col
        
        with np.errstate(divide='ignore', invalid='ignore'):
            df[target_col] = np.log(df[col].values / close_values)
        
        df[target_col] = df[target_col].replace([np.inf, -np.inf], np.nan)
    
    return df


def process_volume_normalize(
    df: pd.DataFrame,
    volume_columns: List[str] = None,
    stock_column: str = 'order_book_id',
    window_size: int = 60,
    keep_original: bool = False,
    suffix: str = '_norm'
) -> pd.DataFrame:
    """
    便捷函数：对成交量列进行均值标准化
    
    公式：volume / rolling_mean(volume, window_size)
    
    Args:
        df: 数据框
        volume_columns: 成交量列列表，如果为None则使用默认值
        stock_column: 股票代码列名
        window_size: 滚动窗口大小
        keep_original: 是否保留原始列
        suffix: 变换后列名后缀
    
    Returns:
        变换后的数据框
    """
    if volume_columns is None:
        volume_columns = ['vol', 'amount']
    
    df = df.copy()
    
    for col in volume_columns:
        if col not in df.columns:
            continue
        
        target_col = f"{col}{suffix}" if keep_original else col
        
        # 按股票分组计算滚动均值
        rolling_mean = df.groupby(stock_column)[col].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean()
        )
        
        # 标准化
        with np.errstate(divide='ignore', invalid='ignore'):
            df[target_col] = df[col] / rolling_mean
        
        df[target_col] = df[target_col].replace([np.inf, -np.inf], np.nan)
    
    return df
