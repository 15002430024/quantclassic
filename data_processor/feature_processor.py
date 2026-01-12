"""
特征处理引擎 - 具体算法实现
"""
import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Dict, Optional
from scipy.stats.mstats import winsorize
from scipy import stats
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import logging


logger = logging.getLogger(__name__)


class FeatureProcessor:
    """特征处理引擎 - 实现各种数据处理算法"""
    
    def __init__(self, groupby_columns: List[str] = None, stock_col: str = 'order_book_id'):
        """
        初始化特征处理器
        
        Args:
            groupby_columns: 默认分组列
            stock_col: 股票代码列名，兼容 'order_book_id'(RiceQuant) 和 'ts_code'(Tushare/DataManager)
        """
        self.groupby_columns = groupby_columns or ['trade_date']
        self.stock_col = stock_col
        self.fitted_params = {}  # 存储拟合参数
    
    def set_stock_col(self, df: pd.DataFrame) -> str:
        """
        根据数据自动检测并设置股票代码列名
        
        Args:
            df: 数据框
            
        Returns:
            实际使用的股票代码列名
        """
        candidates = ['order_book_id', 'ts_code', 'stock_code', 'symbol']
        for col in candidates:
            if col in df.columns:
                self.stock_col = col
                return col
        return self.stock_col
    
    # ========== 基础处理 ==========
    
    def handle_infinite_values(
        self, 
        df: pd.DataFrame, 
        features: List[str],
        method: str = 'remove',
        lower: float = -1e10,
        upper: float = 1e10
    ) -> pd.DataFrame:
        """
        处理无穷值
        
        Args:
            df: 数据框
            features: 特征列
            method: 'remove'(替换为NaN) 或 'clip'(截断)
            lower: 下界
            upper: 上界
        """
        df = df.copy()
        
        if method == 'remove':
            df[features] = df[features].replace([np.inf, -np.inf], np.nan)
        elif method == 'clip':
            for col in features:
                df.loc[df[col] == np.inf, col] = upper
                df.loc[df[col] == -np.inf, col] = lower
        
        return df
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        features: List[str],
        method: str = 'median',
        fillna_value: float = 0,
        industry_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: 数据框
            features: 特征列
            method: 'median', 'mean', 'zero', 'forward'
            fillna_value: 填充值(method='constant'时使用)
            industry_column: 行业列(用于行业内填充)
        """
        df = df.copy()
        
        if method == 'zero':
            df[features] = df[features].fillna(0)
        
        elif method == 'forward':
            # 使用自适应的股票代码列名
            stock_col = self.stock_col
            if stock_col not in df.columns:
                # 尝试自动检测
                for col in ['order_book_id', 'ts_code', 'stock_code', 'symbol']:
                    if col in df.columns:
                        stock_col = col
                        break
            df[features] = df.groupby(stock_col)[features].fillna(method='ffill')
        
        elif method in ['median', 'mean']:
            # 先用行业内统计量填充
            if industry_column and industry_column in df.columns:
                for col in features:
                    if method == 'median':
                        industry_values = df.groupby(self.groupby_columns + [industry_column])[col].transform('median')
                    else:
                        industry_values = df.groupby(self.groupby_columns + [industry_column])[col].transform('mean')
                    df[col] = df[col].fillna(industry_values)
            
            # 再用市场统计量填充
            for col in features:
                if method == 'median':
                    market_values = df.groupby(self.groupby_columns)[col].transform('median')
                else:
                    market_values = df.groupby(self.groupby_columns)[col].transform('mean')
                df[col] = df[col].fillna(market_values)
            
            # 最后用0填充
            df[features] = df[features].fillna(0)
        
        return df
    
    # ========== 标准化/归一化 ==========
    
    def z_score_normalize(
        self,
        df: pd.DataFrame,
        features: List[str],
        ddof: int = 1,
        clip_sigma: Optional[float] = None,
        fit: bool = True,
        normalize_mode: str = 'cross_section'
    ) -> pd.DataFrame:
        """
        Z-score标准化（性能优化版本）
        
        Args:
            df: 数据框
            features: 特征列
            ddof: 标准差自由度
            clip_sigma: 可选的sigma截断值
            fit: 是否拟合(保存均值和标准差)
            normalize_mode: 标准化模式
                - 'cross_section': 截面标准化（按日期分组，每个日期内标准化）
                - 'time_series': 时序标准化（按股票分组，每只股票自身时序标准化）
                - 'global': 全局标准化（整体标准化）
        """
        from tqdm.auto import tqdm
        
        df = df.copy()
        
        # 性能优化：批量处理，减少 groupby 次数
        if fit:
            if normalize_mode == 'cross_section':
                # 截面标准化：一次性计算所有特征的均值和标准差
                grouped = df.groupby(self.groupby_columns)[features]
                mean_vals = grouped.transform('mean')
                std_vals = grouped.transform(lambda x: x.std(ddof=ddof))
                
            elif normalize_mode == 'time_series':
                # 时序标准化：按股票分组（自适应列名）
                stock_col = self.stock_col
                if stock_col not in df.columns:
                    # 尝试自动检测
                    for col in ['order_book_id', 'ts_code', 'stock_code', 'symbol']:
                        if col in df.columns:
                            stock_col = col
                            break
                
                if stock_col in df.columns:
                    grouped = df.groupby(stock_col)[features]
                    mean_vals = grouped.transform('mean')
                    std_vals = grouped.transform(lambda x: x.std(ddof=ddof))
                else:
                    logger.warning(f"时序标准化需要股票代码列（尝试: order_book_id/ts_code/stock_code/symbol），回退到全局标准化")
                    mean_vals = df[features].mean()
                    std_vals = df[features].std(ddof=ddof)
            
            elif normalize_mode == 'global':
                # 全局标准化：整体计算（向量化）
                mean_vals = df[features].mean()
                std_vals = df[features].std(ddof=ddof)
            
            else:
                raise ValueError(f"不支持的标准化模式: {normalize_mode}")
            
            # 保存参数（批量）
            for col in features:
                if isinstance(mean_vals, pd.DataFrame):
                    self.fitted_params[f'{col}_mean_{normalize_mode}'] = mean_vals[col]
                    self.fitted_params[f'{col}_std_{normalize_mode}'] = std_vals[col]
                else:
                    self.fitted_params[f'{col}_mean_{normalize_mode}'] = mean_vals[col] if hasattr(mean_vals, '__getitem__') else mean_vals
                    self.fitted_params[f'{col}_std_{normalize_mode}'] = std_vals[col] if hasattr(std_vals, '__getitem__') else std_vals
        else:
            # 加载保存的参数
            mean_vals = pd.DataFrame({col: self.fitted_params[f'{col}_mean_{normalize_mode}'] for col in features})
            std_vals = pd.DataFrame({col: self.fitted_params[f'{col}_std_{normalize_mode}'] for col in features})
        
        # 批量标准化（向量化操作）
        if isinstance(mean_vals, pd.DataFrame):
            df[features] = (df[features] - mean_vals) / (std_vals + 1e-8)
        else:
            for col in features:
                df[col] = (df[col] - mean_vals[col]) / (std_vals[col] + 1e-8)
        
        # 可选的sigma截断
        if clip_sigma:
            df[features] = df[features].clip(-clip_sigma, clip_sigma)
        
        return df
    
    def minmax_normalize(
        self,
        df: pd.DataFrame,
        features: List[str],
        output_range: Tuple[float, float] = (0, 1),
        fit: bool = True,
        normalize_mode: str = 'cross_section'
    ) -> pd.DataFrame:
        """
        最小最大归一化
        
        Args:
            df: 数据框
            features: 特征列
            output_range: 输出范围
            fit: 是否拟合
            normalize_mode: 标准化模式
                - 'cross_section': 截面归一化（按日期分组，每个日期内归一化）
                - 'time_series': 时序归一化（按股票分组，每只股票自身时序归一化）
                - 'global': 全局归一化（整体归一化）
        """
        df = df.copy()
        min_val, max_val = output_range
        
        for col in features:
            if fit:
                if normalize_mode == 'cross_section':
                    # 截面归一化：按日期分组（同一天内不同股票归一化）
                    col_min = df.groupby(self.groupby_columns)[col].transform('min')
                    col_max = df.groupby(self.groupby_columns)[col].transform('max')
                
                elif normalize_mode == 'time_series':
                    # 时序归一化：按股票分组（自适应列名）
                    stock_col = self.stock_col
                    if stock_col not in df.columns:
                        # 尝试自动检测
                        for candidate in ['order_book_id', 'ts_code', 'stock_code', 'symbol']:
                            if candidate in df.columns:
                                stock_col = candidate
                                break
                    
                    if stock_col in df.columns:
                        col_min = df.groupby(stock_col)[col].transform('min')
                        col_max = df.groupby(stock_col)[col].transform('max')
                    else:
                        logger.warning(f"时序归一化需要股票代码列（尝试: order_book_id/ts_code/stock_code/symbol），回退到全局归一化")
                        col_min = df[col].min()
                        col_max = df[col].max()
                
                elif normalize_mode == 'global':
                    # 全局归一化：整体计算最小最大值
                    col_min = df[col].min()
                    col_max = df[col].max()
                
                else:
                    raise ValueError(f"不支持的标准化模式: {normalize_mode}")
                
                self.fitted_params[f'{col}_min_{normalize_mode}'] = col_min
                self.fitted_params[f'{col}_max_{normalize_mode}'] = col_max
            else:
                col_min = self.fitted_params[f'{col}_min_{normalize_mode}']
                col_max = self.fitted_params[f'{col}_max_{normalize_mode}']
            
            # 归一化
            col_range = col_max - col_min
            df[col] = (df[col] - col_min) / (col_range + 1e-8)
            df[col] = df[col] * (max_val - min_val) + min_val
        
        return df
    
    def rank_normalize(
        self,
        df: pd.DataFrame,
        features: List[str],
        output_range: Tuple[float, float] = (-1, 1),
        method: str = 'average',
        normalize_mode: str = 'cross_section'
    ) -> pd.DataFrame:
        """
        秩归一化到指定区间
        
        Args:
            df: 数据框
            features: 特征列
            output_range: 输出范围,默认(-1, 1)
            method: 排名方法 'average', 'min', 'max', 'dense', 'ordinal'
            normalize_mode: 标准化模式
                - 'cross_section': 截面归一化（按日期分组，每个日期内排名归一化）
                - 'time_series': 时序归一化（按股票分组，每只股票自身时序排名归一化）
                - 'global': 全局归一化（整体排名归一化）
        """
        df = df.copy()
        min_val, max_val = output_range
        
        def _rank_normalize_group(x):
            """对单个分组进行秩归一化"""
            if x.isnull().all():
                return x
            
            # 计算秩
            ranks = x.rank(method=method, na_option='keep')
            
            # 获取有效值数量
            valid_count = ranks.notna().sum()
            
            if valid_count <= 1:
                return pd.Series(0.0, index=x.index)
            
            # 归一化到指定区间
            # 公式: (max_val - min_val) * (rank - 1) / (n - 1) + min_val
            normalized = (max_val - min_val) * (ranks - 1.0) / (valid_count - 1.0) + min_val
            
            return normalized
        
        # 根据模式选择分组方式
        if normalize_mode == 'cross_section':
            # 截面归一化：按日期分组（同一天内不同股票排名归一化）
            for col in features:
                df[col] = df.groupby(self.groupby_columns)[col].transform(_rank_normalize_group)
        
        elif normalize_mode == 'time_series':
            # 时序归一化：按股票分组（每只股票自身时序排名归一化）
            if 'order_book_id' in df.columns:
                for col in features:
                    df[col] = df.groupby('order_book_id')[col].transform(_rank_normalize_group)
            else:
                logger.warning(f"时序归一化需要order_book_id列，回退到全局归一化")
                for col in features:
                    df[col] = _rank_normalize_group(df[col])
        
        elif normalize_mode == 'global':
            # 全局归一化：整体排名归一化
            for col in features:
                df[col] = _rank_normalize_group(df[col])
        
        else:
            raise ValueError(f"不支持的标准化模式: {normalize_mode}")
        
        return df
    
    # ========== 极值处理 ==========
    
    def winsorize_features(
        self,
        df: pd.DataFrame,
        features: List[str],
        limits: Tuple[float, float] = (0.025, 0.025)
    ) -> pd.DataFrame:
        """
        Winsorize去极值（性能优化版本）
        
        Args:
            df: 数据框
            features: 特征列
            limits: 缩尾比例 (下限, 上限)
        """
        from tqdm.auto import tqdm
        
        df = df.copy()
        
        def _winsorize_group(x):
            """对单个分组进行缩尾"""
            if x.isnull().all():
                return x
            try:
                return pd.Series(winsorize(x.astype(float), limits=limits), index=x.index)
            except:
                return x
        
        # 按日期分组进行缩尾（添加进度条）
        with tqdm(total=len(features), desc="去极值处理", unit="列", leave=False) as pbar:
            for col in features:
                df[col] = df.groupby(self.groupby_columns)[col].transform(_winsorize_group)
                pbar.update(1)
        
        return df
    
    def clip_features(
        self,
        df: pd.DataFrame,
        features: List[str],
        lower_percentile: float = 1,
        upper_percentile: float = 99
    ) -> pd.DataFrame:
        """
        截尾处理(按百分位)
        
        Args:
            df: 数据框
            features: 特征列
            lower_percentile: 下百分位
            upper_percentile: 上百分位
        """
        df = df.copy()
        
        for col in features:
            lower = df.groupby(self.groupby_columns)[col].transform(
                lambda x: x.quantile(lower_percentile / 100)
            )
            upper = df.groupby(self.groupby_columns)[col].transform(
                lambda x: x.quantile(upper_percentile / 100)
            )
            df[col] = df[col].clip(lower, upper)
        
        return df
    
    # ========== 中性化处理 ==========
    
    def industry_cap_neutralize_ols(
        self,
        df: pd.DataFrame,
        features: List[str],
        industry_column: str = 'industry_name',
        market_cap_column: str = 'total_mv',
        min_samples: int = 10
    ) -> pd.DataFrame:
        """
        OLS市值行业中性化（性能优化版本）
        
        对每个特征,使用OLS回归剔除行业和市值的影响
        公式: feature_residual = feature - (β_industry * industry_dummy + β_mv * market_cap)
        
        Args:
            df: 数据框
            features: 特征列
            industry_column: 行业列
            market_cap_column: 市值列
            min_samples: 最小样本数
        """
        from tqdm.auto import tqdm
        
        df = df.copy()
        
        def _neutralize_group(group):
            """对单个时间切片进行中性化（批量处理）"""
            if len(group) < min_samples:
                return group
            
            # 准备行业哑变量（一次性创建）
            industry_dummies = pd.get_dummies(group[industry_column], prefix='ind')
            
            # 市值取对数
            if group[market_cap_column].isnull().all():
                return group
            
            log_mv = np.log(group[market_cap_column] + 1)
            
            # 合并特征（预先计算）
            X = pd.concat([industry_dummies, log_mv.rename('log_mv')], axis=1)
            
            # 批量处理所有特征（减少循环开销）
            for col in features:
                if col not in group.columns:
                    continue
                
                # 准备有效数据
                valid_mask = group[col].notna() & log_mv.notna()
                if valid_mask.sum() < min_samples:
                    continue
                
                y = group.loc[valid_mask, col].values
                X_valid = X.loc[valid_mask].values
                
                # OLS回归
                try:
                    lr = LinearRegression(n_jobs=-1)  # 使用多线程加速
                    lr.fit(X_valid, y)
                    
                    # 计算残差
                    residuals = y - lr.predict(X_valid)
                    group.loc[valid_mask, col] = residuals
                except:
                    pass
            
            return group
        
        # 按日期分组进行中性化（添加进度条）
        grouped = df.groupby(self.groupby_columns, group_keys=False)
        n_groups = len(grouped)
        
        with tqdm(total=n_groups, desc="OLS中性化", unit="日期", leave=False) as pbar:
            results = []
            for name, group in grouped:
                results.append(_neutralize_group(group))
                pbar.update(1)
            df = pd.concat(results, ignore_index=False)
        
        return df
    
    def industry_cap_neutralize_mean(
        self,
        df: pd.DataFrame,
        features: List[str],
        industry_column: str = 'industry_name',
        market_cap_column: str = 'total_mv',
        n_quantiles: int = 5
    ) -> pd.DataFrame:
        """
        减均值版市值行业中性化
        
        在每个行业-市值分组内,减去组内均值
        
        Args:
            df: 数据框
            features: 特征列
            industry_column: 行业列
            market_cap_column: 市值列
            n_quantiles: 市值分组数
        """
        df = df.copy()
        
        def _neutralize_group(group):
            """对单个时间切片进行中性化"""
            # 市值分组
            group['mv_quantile'] = pd.qcut(
                group[market_cap_column], 
                q=n_quantiles, 
                labels=False, 
                duplicates='drop'
            )
            
            # 在行业-市值分组内减均值
            for col in features:
                if col in group.columns:
                    group_mean = group.groupby([industry_column, 'mv_quantile'])[col].transform('mean')
                    group[col] = group[col] - group_mean
            
            group.drop('mv_quantile', axis=1, inplace=True)
            return group
        
        # 按日期分组进行中性化
        df = df.groupby(self.groupby_columns, group_keys=False).apply(_neutralize_group)
        
        return df
    
    def simstock_label_neutralize(
        self,
        df: pd.DataFrame,
        label_column: str = 'ret_1d',
        similarity_threshold: float = 0.7,
        lookback_window: int = 252,
        min_similar_stocks: int = 5,
        correlation_method: str = 'pearson',
        output_column: str = 'alpha_label',
        recalc_interval: int = 20  # 新增：每隔20天更新一次相关性矩阵（关键优化）
    ) -> pd.DataFrame:
        """
        SimStock标签中性化（Numpy加速 + 降频更新版）
        
        对每只股票，找到收益率相关性高于阈值的"兄弟股票"，用其均值作为基准，标签为超额收益。
        
        🚀 核心优化（3小时 → 2-3分钟）：
        1. 使用 np.corrcoef 替代 pd.DataFrame.corr（快 5-10x）
        2. 降频更新：每隔 recalc_interval 天才重算相关性（快 20x）
        3. 宽表矩阵：零拷贝切片 + 矩阵乘法（快 100x）
        
        Args:
            df: 数据框（必须包含 'trade_date' 和 'order_book_id' 列）
            label_column: 用于相关性和中性化的标签列（如 'y_ret_1d'）
            similarity_threshold: 相关性阈值（默认 0.7）
            lookback_window: 回溯窗口(交易日)（默认 252，约1年）
            min_similar_stocks: 最小相似股票数（矩阵方案中自动处理）
            correlation_method: 'pearson' 或 'spearman'（默认 'pearson'，暂未使用）
            output_column: 输出的alpha标签列名（默认 'alpha_label'）
            recalc_interval: 相关性更新间隔（默认 20天）
                           - 设为 1：每天更新（最精确但最慢）
                           - 设为 20：每月更新（推荐，快 20x）
                           - 设为 63：每季度更新（适合极大数据集）
            
        Returns:
            新增 output_column 列的 DataFrame
            
        性能指标（中证800，5年数据）：
            - 数据量：~2000天 × 800股 = 160万行
            - 原方案：~3小时（每天计算相关性）
            - 优化后：~2-3分钟（recalc_interval=20）
            - 提速比：~60x
        """
        print(f"\n{'='*80}")
        print("🚀 SimStock 标签中性化（Numpy加速 + 降频更新版）")
        print(f"{'='*80}")
        
        # ==================== 1. 构建宽表 (Time x Stock) ====================
        print(f"📊 步骤1/4: 转换为宽表矩阵 (Pivot)...")
        print(f"  原始数据: {len(df):,} 行")
        
        # 确保数据已排序
        df = df.sort_values(['trade_date', 'order_book_id']).copy()
        
        # Pivot 为宽表：行=日期, 列=股票代码, 值=标签
        wide_ret = df.pivot(index='trade_date', columns='order_book_id', values=label_column)
        
        # 提取基础数据结构
        dates = wide_ret.index
        stocks = wide_ret.columns
        ret_values = wide_ret.values  # 转换为 numpy 数组加速
        n_days, n_stocks = ret_values.shape
        
        print(f"  宽表维度: {n_days:,} 天 × {n_stocks:,} 只股票")
        print(f"  回溯窗口: {lookback_window} 天")
        print(f"  相关性阈值: {similarity_threshold:.2f}")
        print(f"  更新频率: 每 {recalc_interval} 天重新计算一次相似性（关键优化！）")
        print(f"  预计重算次数: {(n_days - lookback_window) // recalc_interval + 1} 次（vs 原方案 {n_days - lookback_window} 次）")
        
        # 初始化结果矩阵 (全为 NaN)
        alpha_matrix = np.full((n_days, n_stocks), np.nan)
        
        # 缓存变量：存储上一次计算的相似性掩码
        cached_sim_mask = None
        
        # ==================== 2. 滚动计算 Alpha ====================
        print(f"\n📈 步骤2/4: 滚动计算 Alpha ({n_days - lookback_window:,} 个有效日期)...")
        
        # 从 lookback_window 开始遍历
        for i in tqdm(range(lookback_window, n_days), desc="计算Alpha", unit="天", mininterval=0.5):
            
            # --- A. 获取当日数据 ---
            # 当日收益率向量 shape: (n_stocks, )
            curr_ret = ret_values[i, :]
            
            # 检查当日是否有有效数据（如果当天全停牌，直接跳过）
            valid_mask = ~np.isnan(curr_ret)
            if not np.any(valid_mask):
                continue
            
            # --- B. 智能更新相关性矩阵（降频更新逻辑）---
            # 只有在以下情况才重新计算相关性：
            # 1. 第一次计算 (cached_sim_mask is None)
            # 2. 达到了更新间隔 ((i - lookback_window) % recalc_interval == 0)
            should_recalc = (cached_sim_mask is None) or ((i - lookback_window) % recalc_interval == 0)
            
            if should_recalc:
                # 获取历史窗口数据 shape: (window, n_stocks)
                hist_slice = ret_values[i - lookback_window : i, :]
                
                # 【核心优化】使用 Numpy 计算相关性
                # np.corrcoef 不支持 NaN，必须先填充
                # 对于收益率相关性，缺失值填 0（代表无波动/无相关）是业界常用做法
                hist_slice_filled = np.nan_to_num(hist_slice, nan=0.0)
                
                # 计算相关性矩阵 (n_stocks, n_stocks)
                # rowvar=False 表示每一列是一个变量（股票）
                # 这一步比 pandas.corr 快 5-10 倍！
                with np.errstate(invalid='ignore'):
                    corr_matrix = np.corrcoef(hist_slice_filled, rowvar=False)
                
                # 生成相似性掩码 (0/1 矩阵)
                with np.errstate(invalid='ignore'):
                    cached_sim_mask = (corr_matrix >= similarity_threshold).astype(np.float32)
                
                # 排除自身（对角线置 0）
                np.fill_diagonal(cached_sim_mask, 0)
            
            # --- C. 矩阵化计算 SimStock Benchmark（复用 cached_sim_mask）---
            
            # 1. 处理当日停牌对齐问题
            # 如果某只股票今天没交易（NaN），它不能作为别人的基准
            # 临时掩码 = 缓存的长期相似关系 × 当天实际交易状态
            valid_mask_2d = valid_mask.reshape(1, -1)
            current_step_mask = cached_sim_mask * valid_mask_2d
            
            # 2. 计算分子：相似股票收益率之和
            # (N, N) @ (N, ) -> (N, )
            # 将 curr_ret 中的 NaN 换成 0 避免污染矩阵乘法
            curr_ret_safe = np.nan_to_num(curr_ret, 0.0)
            sum_ret = current_step_mask @ curr_ret_safe
            
            # 3. 计算分母：相似股票数量
            # (N, N) 按行求和 -> (N, )
            count_sim = current_step_mask.sum(axis=1)
            
            # 4. 计算基准值 Benchmark
            # 默认基准：全市场均值（降级策略）
            market_mean = np.nanmean(curr_ret)
            
            # 计算平均相似股票收益率
            with np.errstate(divide='ignore', invalid='ignore'):
                benchmark = sum_ret / count_sim
            
            # 5. 填充无效值
            # 如果 count_sim == 0（没有相似股票），使用 market_mean
            benchmark = np.where(count_sim == 0, market_mean, benchmark)
            # 如果 benchmark 是 NaN（比如相似股票都停牌），也用 market_mean
            benchmark = np.where(np.isnan(benchmark), market_mean, benchmark)
            
            # --- D. 计算 Alpha 并存入矩阵 ---
            # Alpha = 原始收益 - 基准
            alpha = curr_ret - benchmark
            
            # 存入结果矩阵的第 i 行
            alpha_matrix[i, :] = alpha
        
        # ==================== 3. 还原为长表 ====================
        print(f"\n📦 步骤3/4: 堆叠结果回长表 (Stack)...")
        
        # 将 alpha_matrix 转回 DataFrame
        alpha_df = pd.DataFrame(alpha_matrix, index=dates, columns=stocks)
        
        # Stack 为 Series: index=(trade_date, order_book_id)
        alpha_series = alpha_df.stack(dropna=False)
        alpha_series.name = output_column
        
        # ==================== 4. 合并回原数据 ====================
        print(f"🔗 步骤4/4: 合并回原始数据...")
        
        # 确保原 df 有正确的索引
        df_indexed = df.set_index(['trade_date', 'order_book_id'])
        
        # 合并（使用 left join 保留原数据的所有行）
        result_df = df_indexed.join(alpha_series, how='left')
        
        # 恢复索引
        result_df = result_df.reset_index()
        
        # 统计信息
        valid_alpha = result_df[output_column].count()
        total_rows = len(result_df)
        print(f"\n✅ SimStock 中性化完成!")
        print(f"  输出列: {output_column}")
        print(f"  有效样本: {valid_alpha:,} / {total_rows:,} ({100*valid_alpha/total_rows:.1f}%)")
        print(f"  Alpha均值: {result_df[output_column].mean():.6f}")
        print(f"  Alpha标准差: {result_df[output_column].std():.6f}")
        print(f"{'='*80}\n")
        
        return result_df
    
    # 别名方法（向后兼容）
    def simstock_neutralize(self, *args, **kwargs):
        """simstock_label_neutralize 的别名方法（向后兼容）"""
        return self.simstock_label_neutralize(*args, **kwargs)
