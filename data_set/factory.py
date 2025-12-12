"""
DatasetFactory - 数据集工厂

创建不同类型的数据集对象
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List, Tuple, Optional, Dict, Any, Iterator
from dataclasses import dataclass
import logging
from collections import defaultdict
from .config import DataConfig


# =============================================================================
# 截面批采样器 - 解决 IC Loss 的时间混乱问题
# =============================================================================
class CrossSectionalBatchSampler(Sampler):
    """
    截面批采样器 - 保证每个 batch 来自同一交易日
    
    🔴 核心修复：
    传统的 DataLoader(shuffle=True) 会将不同日期的样本混合在一个 batch 中。
    这会导致 IC Loss 计算的是"跨时间"的排序，毫无金融意义。
    
    本采样器确保：
    1. 每个 batch 内的样本来自同一交易日（截面数据）
    2. 日期顺序随机打乱，避免模型学习时间趋势
    3. 每天的股票顺序随机打乱，增加样本多样性
    
    使用方式：
        sampler = CrossSectionalBatchSampler(dataset, batch_size=256)
        loader = DataLoader(dataset, batch_sampler=sampler)
    """
    
    def __init__(self, dataset: 'TimeSeriesStockDatasetWithDate', 
                 batch_size: int = 256,
                 shuffle_dates: bool = True,
                 drop_last: bool = False):
        """
        Args:
            dataset: 必须是 TimeSeriesStockDatasetWithDate 类型
            batch_size: 每个 batch 的最大样本数
            shuffle_dates: 是否打乱日期顺序
            drop_last: 是否丢弃每天最后一个不足 batch_size 的 batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_dates = shuffle_dates
        self.drop_last = drop_last
        
        # 构建日期 -> 样本索引的映射
        self.date_to_indices = self._build_date_index()
        self.dates = list(self.date_to_indices.keys())
        
    def _build_date_index(self) -> Dict[Any, List[int]]:
        """构建日期到样本索引的映射"""
        date_to_indices = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            stock_idx, time_idx = self.dataset.sample_index[idx]
            stock_info = self.dataset.stock_data[stock_idx]
            # 标签对应的日期是 time_idx + 1
            date = stock_info['dates'][time_idx + 1]
            date_to_indices[date].append(idx)
        
        return dict(date_to_indices)
    
    def __iter__(self) -> Iterator[List[int]]:
        """生成批次"""
        dates = self.dates.copy()
        
        if self.shuffle_dates:
            np.random.shuffle(dates)
        
        for date in dates:
            indices = self.date_to_indices[date].copy()
            np.random.shuffle(indices)  # 打乱同一天内的股票顺序
            
            # 按 batch_size 分割
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                    
                yield batch
    
    def __len__(self) -> int:
        """返回总批次数"""
        total = 0
        for date, indices in self.date_to_indices.items():
            n_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size > 0:
                n_batches += 1
            total += n_batches
        return total


class TimeSeriesStockDatasetWithDate(Dataset):
    """
    时序股票数据集 - 增强版（返回日期信息 + 窗口级变换）
    
    🔴 关键增强：
    1. __getitem__ 返回 (X, y, date_idx) 三元组
    2. date_idx 用于在 Loss 计算时识别同一截面的样本
    3. 提供 get_date_for_idx() 方法获取具体日期
    4. 🆕 窗口级数据变换（研报标准）：
       - 价格对数变换: log(price / close_t)
       - 成交量标准化: volume / mean(volume_in_window)
    5. 🆕 valid_label_start_date: 只为该日期之后的标签生成样本（解决测试集历史数据问题）
    """
    
    def __init__(self, df: pd.DataFrame, feature_cols: List[str],
                 label_col: str, window_size: int, stock_col: str = 'ts_code',
                 time_col: str = 'trade_date', return_date: bool = False,
                 return_stock_id: bool = False,
                 # 🆕 窗口变换配置
                 enable_window_transform: bool = False,
                 window_price_log: bool = False,
                 window_volume_norm: bool = False,
                 price_cols: Optional[List[str]] = None,
                 close_col: str = 'close',
                 volume_cols: Optional[List[str]] = None,
                 # 🆕 标签窗口级排名标准化
                 label_rank_normalize: bool = False,
                 label_rank_output_range: Tuple[float, float] = (-1, 1),
                 # 🆕 全局股票映射
                 stock_map: Optional[Dict[str, int]] = None,
                 # 🆕 有效标签起始日期（只为该日期之后的标签生成样本）
                 valid_label_start_date: Optional[pd.Timestamp] = None):
        """
        Args:
            df: 数据DataFrame
            feature_cols: 特征列列表
            label_col: 标签列
            window_size: 时间窗口大小
            stock_col: 股票代码列
            time_col: 时间列
            return_date: 是否在 __getitem__ 中返回日期索引
            return_stock_id: 是否在 __getitem__ 中返回股票ID
            enable_window_transform: 是否启用窗口级变换
            window_price_log: 是否对价格做对数变换 log(price/close_t)
            window_volume_norm: 是否对成交量做窗口内均值标准化
            price_cols: 价格列名列表
            close_col: 基准收盘价列名
            volume_cols: 成交量列名列表
            stock_map: 股票代码到ID的映射字典 (可选，用于统一全局ID)
            label_rank_normalize: 是否对标签做窗口内时序排名标准化
            label_rank_output_range: 排名标准化输出范围，默认(-1, 1)
            valid_label_start_date: 🆕 只为该日期之后的标签生成样本（用于测试集包含历史数据的情况）
        """
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.stock_col = stock_col
        self.time_col = time_col
        self.return_date = return_date
        self.return_stock_id = return_stock_id
        self.stock_map = stock_map
        
        # 🆕 有效标签起始日期
        self.valid_label_start_date = valid_label_start_date
        
        # 🆕 标签窗口级排名标准化配置
        self.label_rank_normalize = label_rank_normalize
        self.label_rank_output_range = label_rank_output_range
        
        # 🆕 预计算的标签排名（在 _build_sample_index 中填充）
        self._precomputed_label_ranks = {}  # stock_idx -> np.ndarray
        
        # 🆕 窗口变换配置
        self.enable_window_transform = enable_window_transform
        self.window_price_log = window_price_log
        self.window_volume_norm = window_volume_norm
        self.price_cols = price_cols or ['open', 'high', 'low', 'close', 'vwap']
        self.close_col = close_col
        self.volume_cols = volume_cols or ['vol', 'amount']
        
        # 计算价格和成交量列在 feature_cols 中的索引位置
        self._price_indices = []
        self._close_index = None
        self._volume_indices = []
        
        if self.enable_window_transform:
            for i, col in enumerate(feature_cols):
                if col in self.price_cols:
                    self._price_indices.append(i)
                    if col == self.close_col:
                        self._close_index = i
                if col in self.volume_cols:
                    self._volume_indices.append(i)
        
        self._build_sample_index(df)
    
    def _build_sample_index(self, df: pd.DataFrame):
        """预先构建样本索引"""
        import logging
        logger = logging.getLogger(__name__)
        
        df = df.copy()
        df = df.dropna(subset=self.feature_cols + [self.label_col])
        df = df.sort_values([self.stock_col, self.time_col]).reset_index(drop=True)
        
        self.stock_data = {}
        self.sample_index = []
        
        # 构建日期到索引的映射
        all_dates = sorted(df[self.time_col].unique())
        self.date_to_idx = {date: idx for idx, date in enumerate(all_dates)}
        self.idx_to_date = {idx: date for date, idx in self.date_to_idx.items()}
        
        # 🆕 记录数据集中的股票数量（用于后续检查）
        all_stocks_in_data = set(df[self.stock_col].unique())
        
        # 🆕 如果没有提供 stock_map，则构建局部映射
        if self.stock_map is None:
            # 按字母顺序排序股票代码
            all_stocks = sorted(all_stocks_in_data)
            self.stock_map = {stock: i for i, stock in enumerate(all_stocks)}
            
        # 🆕 记录被丢弃的股票
        skipped_stocks_not_in_map = []
        skipped_stocks_insufficient_data = []
        samples_filtered_by_date = 0
        total_potential_samples = 0
        
        # 遍历每只股票
        for ts_code, stock_df in df.groupby(self.stock_col, observed=False):
            # 获取全局ID
            if ts_code not in self.stock_map:
                skipped_stocks_not_in_map.append(ts_code)  # 🆕 记录被跳过的股票
                continue # 跳过不在映射中的股票
                
            stock_idx = self.stock_map[ts_code]
            n = len(stock_df)
            
            if n < self.window_size + 1:
                skipped_stocks_insufficient_data.append(ts_code)  # 🆕 记录数据不足的股票
                continue
            
            features = stock_df[self.feature_cols].values.astype(np.float32)
            labels = stock_df[self.label_col].values.astype(np.float32)
            dates = stock_df[self.time_col].values
            
            self.stock_data[stock_idx] = {
                'ts_code': ts_code,
                'features': features,
                'labels': labels,
                'dates': dates,
                'n': n
            }
            
            # 🆕 预计算标签的窗口级排名（避免运行时计算开销）
            if self.label_rank_normalize:
                self._precomputed_label_ranks[stock_idx] = self._precompute_label_ranks(
                    labels, self.window_size
                )
            
            # 🆕 考虑 valid_label_start_date：只为有效日期范围内的标签生成样本
            for t in range(self.window_size - 1, n - 1):
                total_potential_samples += 1  # 🆕 记录总潜在样本数
                
                # 标签对应的日期是 dates[t + 1]
                label_date = dates[t + 1]
                
                # 如果设置了有效标签起始日期，跳过该日期之前的样本
                if self.valid_label_start_date is not None:
                    # 将 numpy datetime64 转换为 pandas Timestamp 进行比较
                    label_date_ts = pd.Timestamp(label_date)
                    if label_date_ts < self.valid_label_start_date:
                        samples_filtered_by_date += 1  # 🆕 记录被日期过滤掉的样本
                        continue
                
                self.sample_index.append((stock_idx, t))
        
        # 🆕 输出数据集构建统计信息
        logger.info(f"\n====== 数据集构建统计 ======")
        logger.info(f"数据中总股票数: {len(all_stocks_in_data)}")
        logger.info(f"成功加载股票数: {len(self.stock_data)}")
        logger.info(f"生成样本数: {len(self.sample_index):,}")
        
        # 🆕 警告：stock_map 覆盖率
        if skipped_stocks_not_in_map:
            logger.warning(
                f"\u26a0\ufe0f {len(skipped_stocks_not_in_map)} 只股票因不在 stock_map 中而被跳过。"
                f"\n   示例: {skipped_stocks_not_in_map[:5]}"
            )
        
        # 🆕 警告：数据不足
        if skipped_stocks_insufficient_data:
            logger.info(
                f"{len(skipped_stocks_insufficient_data)} 只股票因数据点不足 (<{self.window_size + 1}) 而被跳过。"
            )
        
        # 🆕 警告：valid_label_start_date 过滤
        if self.valid_label_start_date is not None:
            logger.info(
                f"valid_label_start_date 过滤: {samples_filtered_by_date:,} / {total_potential_samples:,} "
                f"({100 * samples_filtered_by_date / total_potential_samples:.1f}%) 样本被过滤"
            )
            
            if len(self.sample_index) == 0:
                logger.error(
                    f"\u274c 数据集为空！valid_label_start_date={self.valid_label_start_date} "
                    f"过滤掉了所有样本。请检查配置。"
                )
            elif len(self.sample_index) < 100:
                logger.warning(
                    f"\u26a0\ufe0f 数据集样本量过小 ({len(self.sample_index)} 个)，可能影响训练效果。"
                )
        
        logger.info("========================\n")
    
    def _precompute_label_ranks(self, labels: np.ndarray, window_size: int) -> np.ndarray:
        """
        预计算所有位置的窗口级标签排名
        
        对于每个位置 t，计算 labels[t] 在窗口 [t-window_size+1, t] 内的排名
        
        Args:
            labels: 该股票的全部标签序列
            window_size: 窗口大小
            
        Returns:
            ranks: 与 labels 等长的数组，每个位置存储该位置标签在窗口内的归一化排名 [0, 1]
        """
        n = len(labels)
        ranks = np.full(n, 0.5, dtype=np.float32)  # 默认值 0.5 (中间值)
        
        low, high = self.label_rank_output_range
        
        for t in range(window_size, n):
            # 窗口范围: [t - window_size + 1, t] (包含 t)
            # 但标签预测的是 t+1 时刻，所以我们需要用 [start, t] 内的标签计算 t 位置的排名
            # 实际上，对于 sample_index 中的 (stock_idx, time_idx)：
            #   - 输入窗口是 [time_idx - window_size + 1, time_idx]
            #   - 标签是 labels[time_idx + 1]
            # 所以预计算时，对于 labels[t]，我们需要用 [t - window_size, t-1] 窗口
            # 即：比较的是 labels[t] 与其之前 window_size 个历史标签
            
            start = t - window_size
            window_labels = labels[start:t]  # 历史窗口（不含当前）
            target_label = labels[t]
            
            # 处理 NaN
            valid_mask = ~np.isnan(window_labels)
            if not np.any(valid_mask) or np.isnan(target_label):
                ranks[t] = (low + high) / 2  # 中间值
                continue
            
            valid_labels = window_labels[valid_mask]
            n_valid = len(valid_labels)
            
            if n_valid == 0:
                ranks[t] = (low + high) / 2
                continue
            
            # 使用 searchsorted 计算排名
            sorted_labels = np.sort(valid_labels)
            left_pos = np.searchsorted(sorted_labels, target_label, side='left')
            right_pos = np.searchsorted(sorted_labels, target_label, side='right')
            
            # 平均排名，归一化到 [0, 1]
            rank = (left_pos + right_pos) / 2.0
            rank_normalized = rank / n_valid if n_valid > 0 else 0.5
            
            # 映射到目标范围
            ranks[t] = low + rank_normalized * (high - low)
        
        return ranks
    
    def __len__(self):
        return len(self.sample_index)
    
    def __getitem__(self, idx: int):
        stock_idx, time_idx = self.sample_index[idx]
        stock_info = self.stock_data[stock_idx]
        
        start_idx = time_idx - (self.window_size - 1)
        end_idx = time_idx + 1
        
        # 获取窗口数据（复制以避免修改原始数据）
        X_seq = stock_info['features'][start_idx:end_idx].copy()
        y = stock_info['labels'][time_idx + 1]
        
        # 🆕 窗口级变换（研报标准）
        if self.enable_window_transform:
            X_seq = self._apply_window_transform(X_seq)
        
        # 🆕 标签窗口内时序排名标准化（使用预计算的排名，O(1) 查表）
        if self.label_rank_normalize:
            y = self._precomputed_label_ranks[stock_idx][time_idx + 1]
        
        X_tensor = torch.from_numpy(X_seq)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        # 构建返回元组
        result = [X_tensor, y_tensor]
        
        # 附加信息（顺序：X, y, [date_idx], [stock_idx]）
        # 注意：如果同时启用，顺序很重要，需要在 Trainer 中对应解包
        
        if self.return_date:
            # 返回标签对应日期的索引
            label_date = stock_info['dates'][time_idx + 1]
            date_idx = self.date_to_idx.get(label_date, -1)
            result.append(torch.tensor(date_idx, dtype=torch.long))
            
        if self.return_stock_id:
            # 返回股票ID（全局索引）
            result.append(torch.tensor(stock_idx, dtype=torch.long))
            
        return tuple(result)
    
    def _apply_window_transform(self, X_seq: np.ndarray) -> np.ndarray:
        """
        对窗口数据进行研报标准的变换
        
        Args:
            X_seq: 窗口特征序列, shape=(window_size, num_features)
            
        Returns:
            变换后的特征序列
            
        研报标准：
        1. 价格对数变换: log(price_{t-i} / close_t)
           - 将窗口内所有价格除以窗口末端的收盘价
           - 然后取对数
           - 结果：close_t = 0, 其他价格为相对偏差
           
        2. 成交量标准化: volume_{t-i} / mean(volume_in_window)
           - 将窗口内的成交量除以该窗口的平均成交量
           - 结果：均值附近 ≈ 1.0
        """
        # 1. 价格对数变换
        if self.window_price_log and self._close_index is not None:
            # 获取窗口末端（当前时刻）的收盘价作为基准
            close_t = X_seq[-1, self._close_index]
            
            # 🆕 修复：跳过 close_t <= 0 的窗口，避免量纲不一致
            if close_t > 0 and not np.isnan(close_t):
                for col_idx in self._price_indices:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        X_seq[:, col_idx] = np.log(X_seq[:, col_idx] / close_t)
                    # 处理无效值
                    X_seq[:, col_idx] = np.nan_to_num(
                        X_seq[:, col_idx], 
                        nan=0.0, posinf=0.0, neginf=0.0
                    )
            else:
                # 🆕 close_t <= 0 或 NaN，跳过价格变换，保持原始值
                # 注意：这会导致该样本的价格特征与其他样本量纲不同
                # 建议在数据预处理阶段过滤掉 close <= 0 的记录
                if not hasattr(self, '_close_t_warning_count'):
                    self._close_t_warning_count = 0
                self._close_t_warning_count += 1
                
                # 只输出前几次警告，避免日志洪水
                if self._close_t_warning_count <= 5:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"\u26a0\ufe0f 窗口末端 close_t={close_t:.4f} <= 0 或 NaN，跳过价格对数变换。"
                        f"\n   建议在数据预处理阶段过滤掉 close <= 0 的记录。"
                    )
                    if self._close_t_warning_count == 5:
                        logger.warning("   (后续相同警告将不再显示)")
                pass  # 保持原始特征值
        
        # 2. 成交量标准化
        if self.window_volume_norm and len(self._volume_indices) > 0:
            for col_idx in self._volume_indices:
                vol_window = X_seq[:, col_idx]
                vol_mean = np.nanmean(vol_window)
                
                if vol_mean > 0 and not np.isnan(vol_mean):
                    X_seq[:, col_idx] = vol_window / vol_mean
                    # 处理无效值
                    X_seq[:, col_idx] = np.nan_to_num(
                        X_seq[:, col_idx],
                        nan=1.0, posinf=1.0, neginf=1.0
                    )
        
        return X_seq
    
    def _apply_label_rank_normalize(self, labels: np.ndarray, 
                                    start_idx: int, end_idx: int, 
                                    target_idx: int) -> float:
        """
        对标签进行窗口内时序排名标准化
        
        🔴 关键：避免未来信息泄露
        - 只使用 [start_idx, target_idx] 范围内的历史标签进行排名
        - 不使用 target_idx 之后的任何数据
        
        Args:
            labels: 该股票的全部标签序列
            start_idx: 窗口起始索引
            end_idx: 窗口结束索引（不含）
            target_idx: 目标标签的索引
            
        Returns:
            标准化后的标签值，范围为 label_rank_output_range
            
        算法:
        1. 取窗口内的历史标签 labels[start_idx:target_idx+1]
        2. 对目标标签在这个历史序列中进行排名
        3. 将排名映射到 output_range (默认 -1 到 1)
        """
        # 取窗口内的标签（包含当前目标标签）
        # 注意: 这里我们用的是历史窗口内的标签来计算rank，确保无未来信息泄露
        window_labels = labels[start_idx:target_idx + 1]
        target_label = labels[target_idx]
        
        # 处理 NaN 值
        valid_mask = ~np.isnan(window_labels)
        if not np.any(valid_mask) or np.isnan(target_label):
            return 0.0  # 默认返回中间值
        
        valid_labels = window_labels[valid_mask]
        n_valid = len(valid_labels)
        
        if n_valid <= 1:
            return 0.0  # 只有一个有效值，返回中间值
        
        # 计算排名（从小到大排序）
        # 使用 scipy.stats.rankdata 风格的排名计算
        # 对于重复值使用平均排名
        sorted_labels = np.sort(valid_labels)
        
        # 找到 target_label 的排名位置
        # 使用二分查找找左边界和右边界来处理重复值
        left_pos = np.searchsorted(sorted_labels, target_label, side='left')
        right_pos = np.searchsorted(sorted_labels, target_label, side='right')
        
        # 平均排名（处理重复值）
        rank = (left_pos + right_pos) / 2.0  # 范围 [0, n_valid-1]
        
        # 归一化到 [0, 1]
        rank_normalized = rank / (n_valid - 1) if n_valid > 1 else 0.5
        
        # 映射到目标范围 [low, high]
        low, high = self.label_rank_output_range
        result = low + rank_normalized * (high - low)
        
        return float(result)
    
    def get_date_for_idx(self, date_idx: int):
        """根据日期索引获取实际日期"""
        return self.idx_to_date.get(date_idx, None)
    
    def get_num_dates(self) -> int:
        """获取不同日期的数量"""
        return len(self.date_to_idx)


# 为了向后兼容，保留原来的 TimeSeriesStockDataset 名称
TimeSeriesStockDataset = TimeSeriesStockDatasetWithDate


@dataclass
class DatasetCollection:
    """数据集集合"""
    train: Dataset
    val: Dataset
    test: Dataset
    metadata: Dict[str, Any]
    
    def get_loaders(self, batch_size: Optional[int] = None,
                   num_workers: Optional[int] = None,
                   shuffle_train: Optional[bool] = None,
                   use_cross_sectional: bool = False) -> 'LoaderCollection':
        """
        创建数据加载器
        
        Args:
            batch_size: 批次大小
            num_workers: 工作进程数
            shuffle_train: 是否打乱训练数据（传统模式）
            use_cross_sectional: 🔴 是否使用截面批采样（IC Loss 必须开启）
        """
        bs = batch_size or 256
        nw = num_workers or 0
        
        if use_cross_sectional and isinstance(self.train, TimeSeriesStockDatasetWithDate):
            # 使用截面批采样器
            train_sampler = CrossSectionalBatchSampler(
                self.train, batch_size=bs, shuffle_dates=True
            )
            train_loader = DataLoader(
                self.train,
                batch_sampler=train_sampler,
                num_workers=nw
            )
        else:
            # 传统随机采样
            train_loader = DataLoader(
                self.train,
                batch_size=bs,
                shuffle=shuffle_train if shuffle_train is not None else True,
                num_workers=nw
            )
        
        return LoaderCollection(
            train=train_loader,
            val=DataLoader(
                self.val,
                batch_size=bs,
                shuffle=False,
                num_workers=nw
            ),
            test=DataLoader(
                self.test,
                batch_size=bs,
                shuffle=False,
                num_workers=nw
            ),
            metadata=self.metadata
        )


@dataclass
class LoaderCollection:
    """数据加载器集合"""
    train: DataLoader
    val: DataLoader
    test: DataLoader
    metadata: Dict[str, Any]


class InferenceDataset(Dataset):
    """推理数据集（仅特征，无标签）"""
    
    def __init__(self, df: pd.DataFrame, feature_cols: List[str],
                 window_size: int, stock_col: str = 'ts_code',
                 time_col: str = 'trade_date'):
        """
        Args:
            df: 数据DataFrame
            feature_cols: 特征列列表
            window_size: 时间窗口大小
            stock_col: 股票代码列
            time_col: 时间列
        """
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.stock_col = stock_col
        self.time_col = time_col
        
        self._build_sample_index(df)
    
    def _build_sample_index(self, df: pd.DataFrame):
        """预先构建样本索引"""
        df = df.copy()
        df = df.dropna(subset=self.feature_cols)
        df = df.sort_values([self.stock_col, self.time_col]).reset_index(drop=True)
        
        self.stock_data = {}
        self.sample_index = []
        self.sample_info = []  # 存储样本元信息
        
        stock_idx = 0
        for ts_code, stock_df in df.groupby(self.stock_col, observed=False):
            n = len(stock_df)
            
            if n < self.window_size:
                continue
            
            features = stock_df[self.feature_cols].values.astype(np.float32)
            dates = stock_df[self.time_col].values
            
            self.stock_data[stock_idx] = {
                'ts_code': ts_code,
                'features': features,
                'dates': dates,
                'n': n
            }
            
            # 构建样本索引
            for t in range(self.window_size - 1, n):
                self.sample_index.append((stock_idx, t))
                self.sample_info.append({
                    'ts_code': ts_code,
                    'date': dates[t]
                })
            
            stock_idx += 1
    
    def __len__(self):
        return len(self.sample_index)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        stock_idx, time_idx = self.sample_index[idx]
        stock_info = self.stock_data[stock_idx]
        
        start_idx = time_idx - (self.window_size - 1)
        end_idx = time_idx + 1
        
        X_seq = stock_info['features'][start_idx:end_idx]
        
        return torch.from_numpy(X_seq)
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """获取样本元信息"""
        return self.sample_info[idx]


class DatasetFactory:
    """数据集工厂"""
    
    def __init__(self, config: DataConfig):
        """
        Args:
            config: DataConfig配置对象
        """
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """配置日志"""
        logger = logging.getLogger('DatasetFactory')
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_datasets(self,
                       train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       feature_cols: List[str],
                       test_valid_label_start_date: Optional[pd.Timestamp] = None) -> DatasetCollection:
        """
        创建训练、验证、测试数据集
        
        Args:
            train_df: 训练数据
            val_df: 验证数据
            test_df: 测试数据
            feature_cols: 特征列列表
            test_valid_label_start_date: 🆕 测试集的有效标签起始日期
                                         如果设置，只为该日期之后的标签生成样本
            
        Returns:
            DatasetCollection对象
        """
        self.logger.info("🏭 创建数据集...")
        
        # 🆕 从配置中获取窗口变换参数
        enable_wt = getattr(self.config, 'enable_window_transform', False)
        price_log = getattr(self.config, 'window_price_log', False)
        vol_norm = getattr(self.config, 'window_volume_norm', False)
        price_cols = getattr(self.config, 'price_cols', ['open', 'high', 'low', 'close', 'vwap'])
        close_col = getattr(self.config, 'close_col', 'close')
        volume_cols = getattr(self.config, 'volume_cols', ['vol', 'amount'])
        
        if enable_wt:
            self.logger.info(f"   🔄 启用窗口级变换:")
            self.logger.info(f"      价格对数变换: {price_log} ({price_cols})")
            self.logger.info(f"      成交量标准化: {vol_norm} ({volume_cols})")
        
        # 创建数据集
        train_dataset = TimeSeriesStockDataset(
            train_df, feature_cols, self.config.label_col,
            self.config.window_size, self.config.stock_col, self.config.time_col,
            # 🆕 传递窗口变换配置
            enable_window_transform=enable_wt,
            window_price_log=price_log,
            window_volume_norm=vol_norm,
            price_cols=price_cols,
            close_col=close_col,
            volume_cols=volume_cols
        )
        
        val_dataset = TimeSeriesStockDataset(
            val_df, feature_cols, self.config.label_col,
            self.config.window_size, self.config.stock_col, self.config.time_col,
            enable_window_transform=enable_wt,
            window_price_log=price_log,
            window_volume_norm=vol_norm,
            price_cols=price_cols,
            close_col=close_col,
            volume_cols=volume_cols
        )
        
        # 🆕 测试集：传递有效标签起始日期
        test_dataset = TimeSeriesStockDataset(
            test_df, feature_cols, self.config.label_col,
            self.config.window_size, self.config.stock_col, self.config.time_col,
            enable_window_transform=enable_wt,
            window_price_log=price_log,
            window_volume_norm=vol_norm,
            price_cols=price_cols,
            close_col=close_col,
            volume_cols=volume_cols,
            valid_label_start_date=test_valid_label_start_date  # 🆕 只为该日期后的标签生成样本
        )
        
        if test_valid_label_start_date is not None:
            self.logger.info(f"   🔴 测试集有效标签起始: {test_valid_label_start_date}")
        
        # 收集元数据
        metadata = {
            'feature_cols': feature_cols,
            'num_features': len(feature_cols),
            'window_size': self.config.window_size,
            'label_col': self.config.label_col,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'test_valid_label_start_date': test_valid_label_start_date,  # 🆕
        }
        
        self.logger.info(f"   训练集: {len(train_dataset):,} 样本")
        self.logger.info(f"   验证集: {len(val_dataset):,} 样本")
        self.logger.info(f"   测试集: {len(test_dataset):,} 样本")
        
        return DatasetCollection(
            train=train_dataset,
            val=val_dataset,
            test=test_dataset,
            metadata=metadata
        )
    
    def create_inference_dataset(self, df: pd.DataFrame,
                                feature_cols: List[str]) -> InferenceDataset:
        """
        创建推理数据集
        
        Args:
            df: 数据DataFrame
            feature_cols: 特征列列表
            
        Returns:
            InferenceDataset对象
        """
        self.logger.info("🔮 创建推理数据集...")
        
        dataset = InferenceDataset(
            df, feature_cols, self.config.window_size,
            self.config.stock_col, self.config.time_col
        )
        
        self.logger.info(f"   推理样本: {len(dataset):,}")
        
        return dataset


if __name__ == '__main__':
    # 测试数据集工厂
    from config import DataConfig
    
    print("=" * 80)
    print("DatasetFactory 测试")
    print("=" * 80)
    
    # 创建配置
    config = DataConfig(window_size=40)
    
    # 创建工厂
    factory = DatasetFactory(config)
    
    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200)
    stocks = ['000001.SZ', '000002.SZ']
    
    data = []
    for stock in stocks:
        for date in dates:
            data.append({
                'ts_code': stock,
                'trade_date': date,
                'y_processed': np.random.randn(),
                'feature1': np.random.randn(),
                'feature2': np.random.randn(),
            })
    
    df = pd.DataFrame(data)
    
    # 划分数据
    n = len(dates)
    train_end = dates[int(n * 0.7)]
    val_end = dates[int(n * 0.85)]
    
    train_df = df[df['trade_date'] <= train_end]
    val_df = df[(df['trade_date'] > train_end) & (df['trade_date'] <= val_end)]
    test_df = df[df['trade_date'] > val_end]
    
    # 创建数据集
    datasets = factory.create_datasets(
        train_df, val_df, test_df,
        feature_cols=['feature1', 'feature2']
    )
    
    print(f"\n数据集元数据:")
    for key, value in datasets.metadata.items():
        print(f"  {key}: {value}")
    
    # 测试数据加载器
    print(f"\n创建数据加载器...")
    loaders = datasets.get_loaders(batch_size=32)
    
    # 测试一个批次
    batch_x, batch_y = next(iter(loaders.train))
    print(f"  批次特征形状: {batch_x.shape}")
    print(f"  批次标签形状: {batch_y.shape}")
    
    # 测试推理数据集
    print(f"\n创建推理数据集...")
    inference_dataset = factory.create_inference_dataset(
        test_df, feature_cols=['feature1', 'feature2']
    )
    
    sample = inference_dataset[0]
    info = inference_dataset.get_sample_info(0)
    print(f"  推理样本形状: {sample.shape}")
    print(f"  样本信息: {info}")
    
    print("\n✅ 数据集工厂测试完成")
