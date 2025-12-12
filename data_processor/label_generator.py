"""
标签生成器 - LabelGenerator (重构版)

修复分类和排名标签忽略 entry_offset 和 return_periods 的问题
统一所有标签类型的收益率计算逻辑

支持：
1. 未来收益率标签（回归任务）
2. 分类标签（涨跌、多分类）
3. 排名标签（分位数）
4. SimStock中性化标签（Alpha）
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union, Literal
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LabelConfig:
    """标签生成配置
    
    用于配置标签生成器的参数，支持多种标签类型和生成策略。
    
    Args:
        stock_col (str): 股票代码列名。
            默认值: 'order_book_id'
            
        time_col (str): 时间列名（交易日期）。
            默认值: 'trade_date'
            
        price_col (str): 价格列名，用于计算收益率/标签。
            默认值: 'close'
            
        base_price_col (str): 基准价格列名，用于标签计算的分母。
            - None: 使用 price_col (传统方式: price_t+n / price_t - 1)
            - 'vwap': 使用次日 VWAP (研报标准: price_t+n / vwap_t+1 - 1)
            - 'close': 使用次日收盘价 (研报标准: price_t+n / close_t+1 - 1)
            - 'open': 使用次日开盘价 (price_t+n / open_t+1 - 1)
            默认值: None (向后兼容)
            
        entry_offset (int): 建仓价相对当前行的时间偏移。
            - 0: 使用 T 日价格（传统模式）
            - 1: 使用 T+1 日价格（研报模式，买入 T+1，卖出 T+period+1）
            默认值: 1（1日标签 = price_t+2 / price_t+1 - 1）
            
        entry_price_col (str): 建仓价使用的列。
            - None: 自动回退到 base_price_col，否则使用 price_col
            - 也可指定 'vwap'、'open' 等列
            默认值: None
            
        label_type (str): 标签类型，可选值:
            - 'return': 回归标签（未来收益率），使用shift(-period)获取未来价格
            - 'classification': 分类标签（涨/平/跌或多分类）
            - 'rank': 排名标签（分位数排名）
            默认值: 'return'
            
        return_periods (List[int]): 回归标签的期数列表。
            指定要生成哪些周期的未来收益率标签。
            例如: [1, 5, 10, 20] 生成 y_ret_1d, y_ret_5d, y_ret_10d, y_ret_20d
            (注: 建议使用 'y_' 前缀区分标签列和特征列中的 ret_1d)
            使用shift(-period)实现向前看，获取未来价格。
            最后N行为NaN（无未来数据）。
            默认值: [1, 5, 10, 20]
            
        return_method (str): 收益率计算方法，可选值:
            - 'simple': 简单收益率 (未来价-当前价)/当前价
            - 'log': 对数收益率 ln(未来价/当前价)
            默认值: 'simple'
            
        n_classes (int): 分类标签的类别数（仅当label_type='classification'时使用）。
            默认值: 3（涨/平/跌）
            
        class_method (str): 分类方法，可选值:
            - 'quantile': 按分位数划分（自适应）
            - 'threshold': 按固定阈值划分（需指定thresholds参数）
            默认值: 'quantile'
            
        thresholds (List[float]): 分类阈值列表（仅当class_method='threshold'时使用）。
            例如: [-0.02, 0.02] 表示 <=−2% 为跌, −2%~2% 为平, >=2% 为涨
            默认值: [-0.02, 0.02]
            
        rank_method (str): 排名方法（仅当label_type='rank'时使用），可选值:
            - 'quantile': 按分位数排名
            - 'percentile': 按百分比排名
            默认值: 'quantile'
            
        n_quantiles (int): 分位数数量（仅当label_type='rank'时使用）。
            例如: 10 表示分为10个分位组（0-9）
            默认值: 10
            
        neutralize (bool): 是否对标签进行中性化处理。
            默认值: False
            
        neutralize_method (str): 中性化方法，可选值:
            - 'market': 市场中性化
            - 'industry': 行业中性化
            - 'simstock': SimStock相似股票中性化
            默认值: 'market'
            
        fillna_method (str): 缺失值处理方法，可选值:
            - 'drop': 删除包含NaN的行
            - 'zero': 用0填充
            - 'forward': 向前填充
            默认值: 'drop'
    """
    # 基础配置
    stock_col: str = 'order_book_id'
    time_col: str = 'trade_date'
    price_col: str = 'close'  # 未来价格列（分子/卖出价）
    base_price_col: Optional[str] = None  # 基准价格列（分母，None表示使用T日价格）
    
    # 研报式收益率配置
    entry_offset: int = 1  # 默认采取T+1作为建仓价，保证所有周期遵循研报定义
    entry_price_col: Optional[str] = None  # 建仓价格列（None则使用base_price_col或price_col）
    
    # 标签类型
    label_type: Literal['return', 'classification', 'rank'] = 'return'
    
    # 收益率配置 - 使用 field(default_factory) 避免可变默认参数陷阱
    return_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    return_method: Literal['simple', 'log'] = 'simple'
    
    # 分类配置
    n_classes: int = 3  # 涨/平/跌
    class_method: Literal['quantile', 'threshold'] = 'quantile'
    thresholds: List[float] = field(default_factory=lambda: [-0.02, 0.02])
    
    # 排名配置
    rank_method: Literal['quantile', 'percentile'] = 'quantile'
    n_quantiles: int = 10
    
    # 中性化配置
    neutralize: bool = False
    neutralize_method: Literal['market', 'industry', 'simstock'] = 'market'
    
    # 缺失值处理
    fillna_method: Literal['drop', 'zero', 'forward'] = 'drop'


class LabelGenerator:
    """标签生成器 - 提供多种标签生成策略
    
    重构版本：统一所有标签类型的收益率计算逻辑，
    确保分类/排名标签也遵循 entry_offset 和 return_periods 配置。
    """
    
    def __init__(self, config: LabelConfig = None):
        """
        初始化标签生成器
        
        Args:
            config: 标签配置对象
        """
        self.config = config or LabelConfig()
        self.logger = logging.getLogger(__name__)
    
    def generate_labels(
        self,
        df: pd.DataFrame,
        label_name: str = 'label'
    ) -> pd.DataFrame:
        """
        生成标签（主入口）
        
        重构逻辑：
        1. 先统一计算所有周期的未来收益率（临时列）
        2. 根据 label_type 决定如何处理这些收益率
        3. 清理临时列，应用中性化和缺失值处理
        
        Args:
            df: 输入数据框
            label_name: 标签列名
            
        Returns:
            添加标签列的数据框
        """
        df = df.copy()
        
        self.logger.info(f"开始生成标签 (类型: {self.config.label_type}, 周期: {self.config.return_periods})...")
        
        # 确保数据按股票和时间排序
        df = df.sort_values([self.config.stock_col, self.config.time_col]).reset_index(drop=True)
        
        # ============ 第1步：统一计算所有周期的未来收益率 ============
        temp_ret_cols = {}
        for period in self.config.return_periods:
            temp_col = f"_temp_ret_{period}d"
            df = self._calculate_single_period_return(df, period, temp_col)
            temp_ret_cols[period] = temp_col
        
        # ============ 第2步：根据标签类型处理收益率 ============
        generated_cols = []
        
        for period, temp_col in temp_ret_cols.items():
            # 确定最终列名
            final_col = f"{label_name}_{period}d" if len(self.config.return_periods) > 1 else label_name
            
            if self.config.label_type == 'return':
                # 回归任务：直接使用收益率
                df[final_col] = df[temp_col]
                
            elif self.config.label_type == 'classification':
                # 分类任务：对收益率进行离散化
                df = self._apply_classification(df, temp_col, final_col)
                
            elif self.config.label_type == 'rank':
                # 排名任务：对收益率进行排名
                df = self._apply_ranking(df, temp_col, final_col)
            
            else:
                raise ValueError(f"不支持的标签类型: {self.config.label_type}")
            
            generated_cols.append(final_col)
        
        # ============ 第3步：清理临时列 ============
        df = df.drop(columns=list(temp_ret_cols.values()))
        
        # ============ 第4步：中性化处理 ============
        if self.config.neutralize:
            for col in generated_cols:
                df = self._neutralize_column(df, col)
        
        # ============ 第5步：处理缺失值 ============
        df = self._handle_missing_labels(df, generated_cols)
        
        # 统计生成的标签列
        total_valid = sum(df[col].count() for col in generated_cols if col in df.columns)
        self.logger.info(f"标签生成完成: {len(generated_cols)} 个标签列, 共 {total_valid} 个有效值")
        
        return df
    
    def _calculate_single_period_return(
        self,
        df: pd.DataFrame,
        period: int,
        output_col: str
    ) -> pd.DataFrame:
        """
        计算单周期的未来收益率（核心逻辑，支持 entry_offset）
        
        这是所有标签类型共用的收益率计算方法，确保一致性。
        
        Args:
            df: 数据框（已排序）
            period: 持有期
            output_col: 输出列名
            
        Returns:
            添加收益率列的数据框
        """
        # 确定建仓价格列
        # 优先级: entry_price_col > base_price_col > price_col
        if self.config.entry_price_col is not None:
            entry_col = self.config.entry_price_col
        elif self.config.base_price_col is not None:
            entry_col = self.config.base_price_col
        else:
            entry_col = self.config.price_col
        
        entry_offset = self.config.entry_offset
        
        # 按股票分组
        grouped = df.groupby(self.config.stock_col)
        
        # 1. 获取建仓价格 P_entry
        # entry_offset=1 (研报模式): 使用 T+1 的价格作为分母
        # entry_offset=0 (传统模式): 使用 T 的价格作为分母
        if entry_offset > 0:
            entry_price = grouped[entry_col].shift(-entry_offset)
        else:
            entry_price = df[entry_col]
        
        # 2. 获取卖出价格 P_exit
        # 研报模式: T+1 买入，持有 period 天，即 T+1+period 卖出
        # 传统模式: T 买入，持有 period 天，即 T+period 卖出
        # shift 应该是 -(entry_offset + period)
        exit_shift = -(entry_offset + period)
        exit_price = grouped[self.config.price_col].shift(exit_shift)
        
        # 3. 计算收益率
        if self.config.return_method == 'simple':
            # 简单收益率: (P_exit - P_entry) / P_entry
            df[output_col] = (exit_price - entry_price) / entry_price
        else:
            # 对数收益率: log(P_exit / P_entry)
            df[output_col] = np.log(exit_price / entry_price)
        
        # 构建公式描述用于日志
        if entry_offset > 0:
            formula = f"{self.config.price_col}[t+{period + entry_offset}] / {entry_col}[t+{entry_offset}] - 1"
        else:
            formula = f"{self.config.price_col}[t+{period}] / {entry_col}[t] - 1"
        
        valid_count = df[output_col].count()
        self.logger.info(f"  计算 {period}日收益率: {valid_count} 个有效值 (公式: {formula})")
        
        return df
    
    def _apply_classification(
        self,
        df: pd.DataFrame,
        source_col: str,
        target_col: str
    ) -> pd.DataFrame:
        """
        基于源列（收益率）生成分类标签
        
        Args:
            df: 数据框
            source_col: 源列名（收益率）
            target_col: 目标列名（分类标签）
            
        Returns:
            添加分类标签的数据框
        """
        if self.config.class_method == 'quantile':
            # 按时间横截面分组进行分位数划分
            df[target_col] = df.groupby(self.config.time_col)[source_col].transform(
                lambda x: pd.qcut(x, q=self.config.n_classes, labels=False, duplicates='drop')
                if x.notna().sum() >= self.config.n_classes else np.nan
            )
            
        elif self.config.class_method == 'threshold':
            # 使用阈值划分类别
            df[target_col] = np.nan  # 初始化
            valid_mask = df[source_col].notna()
            
            if self.config.n_classes == 3:
                # 三分类: 跌(0) / 平(1) / 涨(2)
                lower, upper = self.config.thresholds
                df.loc[valid_mask, target_col] = 1  # 默认中间类
                df.loc[valid_mask & (df[source_col] < lower), target_col] = 0
                df.loc[valid_mask & (df[source_col] > upper), target_col] = 2
                
            elif self.config.n_classes == 2:
                # 二分类: 跌(0) / 涨(1)
                threshold = self.config.thresholds[0]
                df.loc[valid_mask, target_col] = (df.loc[valid_mask, source_col] > threshold).astype(int)
        
        # 日志
        if target_col in df.columns:
            dist = df[target_col].value_counts().sort_index()
            self.logger.info(f"  分类标签 {target_col}: {dict(dist)}")
        
        return df
    
    def _apply_ranking(
        self,
        df: pd.DataFrame,
        source_col: str,
        target_col: str
    ) -> pd.DataFrame:
        """
        基于源列（收益率）生成排名标签
        
        Args:
            df: 数据框
            source_col: 源列名（收益率）
            target_col: 目标列名（排名标签）
            
        Returns:
            添加排名标签的数据框
        """
        if self.config.rank_method == 'quantile':
            # 按时间横截面分组进行分位数划分
            df[target_col] = df.groupby(self.config.time_col)[source_col].transform(
                lambda x: pd.qcut(x, q=self.config.n_quantiles, labels=False, duplicates='drop')
                if x.notna().sum() >= self.config.n_quantiles else np.nan
            )
            
        elif self.config.rank_method == 'percentile':
            # 百分比排名
            df[target_col] = df.groupby(self.config.time_col)[source_col].rank(pct=True)
        
        # 日志
        if target_col in df.columns:
            self.logger.info(f"  排名标签 {target_col}: 范围 [{df[target_col].min()}, {df[target_col].max()}]")
        
        return df
    
    def _neutralize_column(
        self,
        df: pd.DataFrame,
        col_name: str
    ) -> pd.DataFrame:
        """
        单列中性化
        
        Args:
            df: 数据框
            col_name: 要中性化的列名
            
        Returns:
            中性化后的数据框
        """
        self.logger.info(f"  中性化 {col_name} (方法: {self.config.neutralize_method})...")
        
        if self.config.neutralize_method == 'market':
            # 减去市场平均收益
            market_mean = df.groupby(self.config.time_col)[col_name].transform('mean')
            df[col_name] = df[col_name] - market_mean
            
        elif self.config.neutralize_method == 'industry':
            # 减去行业平均收益（需要 industry 列）
            ind_col = next(
                (c for c in ['industry_name', 'industry_code', 'citic_industry'] if c in df.columns),
                None
            )
            if ind_col:
                ind_mean = df.groupby([self.config.time_col, ind_col])[col_name].transform('mean')
                df[col_name] = df[col_name] - ind_mean
            else:
                self.logger.warning("未找到行业列，跳过行业中性化")
                
        elif self.config.neutralize_method == 'simstock':
            self.logger.warning("SimStock中性化需要使用 FeatureProcessor.simstock_label_neutralize")
        
        return df
    
    def _handle_missing_labels(
        self,
        df: pd.DataFrame,
        label_cols: List[str]
    ) -> pd.DataFrame:
        """
        处理缺失标签
        
        Args:
            df: 数据框
            label_cols: 标签列名列表
            
        Returns:
            处理后的数据框
        """
        # 过滤实际存在的列
        existing_cols = [c for c in label_cols if c in df.columns]
        if not existing_cols:
            return df
        
        missing_count = df[existing_cols].isnull().sum().sum()
        if missing_count == 0:
            return df
        
        self.logger.info(f"处理缺失标签 (方法: {self.config.fillna_method}, 缺失数: {missing_count})...")
        
        if self.config.fillna_method == 'drop':
            df = df.dropna(subset=existing_cols)
            
        elif self.config.fillna_method == 'zero':
            df[existing_cols] = df[existing_cols].fillna(0)
            
        elif self.config.fillna_method == 'forward':
            # 注意：标签的 forward fill 比较危险，通常建议 drop
            for col in existing_cols:
                df[col] = df.groupby(self.config.stock_col)[col].ffill()
        
        return df
    
    # ========== 便捷方法 ==========
    
    def generate_multi_period_returns(
        self,
        df: pd.DataFrame,
        periods: List[int] = None,
        prefix: str = 'ret'
    ) -> pd.DataFrame:
        """
        快速生成多周期收益率
        
        Args:
            df: 数据框
            periods: 周期列表
            prefix: 列名前缀
            
        Returns:
            添加多周期收益率的数据框
        """
        periods = periods or [1, 5, 10, 20]
        
        # 临时修改配置
        old_periods = self.config.return_periods
        old_type = self.config.label_type
        
        self.config.return_periods = periods
        self.config.label_type = 'return'
        
        # 生成标签
        df = self.generate_labels(df, label_name=prefix)
        
        # 恢复配置
        self.config.return_periods = old_periods
        self.config.label_type = old_type
        
        return df
    
    def generate_binary_labels(
        self,
        df: pd.DataFrame,
        threshold: float = 0.0,
        label_name: str = 'label'
    ) -> pd.DataFrame:
        """
        快速生成二分类标签（涨/跌）
        
        Args:
            df: 数据框
            threshold: 阈值
            label_name: 标签列名
            
        Returns:
            添加二分类标签的数据框
        """
        # 临时修改配置
        old_type = self.config.label_type
        old_classes = self.config.n_classes
        old_thresholds = self.config.thresholds
        old_method = self.config.class_method
        
        self.config.label_type = 'classification'
        self.config.n_classes = 2
        self.config.thresholds = [threshold]
        self.config.class_method = 'threshold'
        
        # 生成标签
        df = self.generate_labels(df, label_name=label_name)
        
        # 恢复配置
        self.config.label_type = old_type
        self.config.n_classes = old_classes
        self.config.thresholds = old_thresholds
        self.config.class_method = old_method
        
        return df
    
    def get_label_statistics(
        self,
        df: pd.DataFrame,
        label_cols: Union[str, List[str]]
    ) -> pd.DataFrame:
        """
        获取标签统计信息
        
        Args:
            df: 数据框
            label_cols: 标签列名或列名列表
            
        Returns:
            统计信息DataFrame
        """
        if isinstance(label_cols, str):
            label_cols = [label_cols]
        
        stats = []
        for col in label_cols:
            if col not in df.columns:
                continue
            
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            
            stats.append({
                '标签列': col,
                '有效样本数': len(col_data),
                '缺失样本数': df[col].isnull().sum(),
                '均值': col_data.mean(),
                '标准差': col_data.std(),
                '最小值': col_data.min(),
                '25分位': col_data.quantile(0.25),
                '中位数': col_data.median(),
                '75分位': col_data.quantile(0.75),
                '最大值': col_data.max(),
            })
        
        return pd.DataFrame(stats)


# ========== 便捷函数 ==========

def generate_future_returns(
    df: pd.DataFrame,
    stock_col: str = 'order_book_id',
    time_col: str = 'trade_date',
    price_col: str = 'close',
    periods: List[int] = None,
    method: str = 'simple',
    entry_offset: int = 1
) -> pd.DataFrame:
    """
    快速生成未来收益率（便捷函数）
    
    Args:
        df: 数据框
        stock_col: 股票列名
        time_col: 时间列名
        price_col: 价格列名
        periods: 周期列表
        method: 'simple' 或 'log'
        entry_offset: 建仓价偏移量（默认1为研报模式）
        
    Returns:
        添加收益率列的数据框
    """
    config = LabelConfig(
        stock_col=stock_col,
        time_col=time_col,
        price_col=price_col,
        label_type='return',
        return_periods=periods or [1, 5, 10, 20],
        return_method=method,
        entry_offset=entry_offset
    )
    
    generator = LabelGenerator(config)
    return generator.generate_labels(df, label_name='label')


if __name__ == '__main__':
    # 测试标签生成器
    print("=" * 80)
    print("LabelGenerator 测试 (重构版)")
    print("=" * 80)
    
    # 创建测试数据（线性上涨，便于验证）
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=30)
    stocks = ['000001.SZ', '000002.SZ', '000003.SZ']
    
    data = []
    for stock in stocks:
        base_price = 10
        for i, date in enumerate(dates):
            data.append({
                'order_book_id': stock,
                'trade_date': date,
                'close': base_price + i * 0.5,  # 线性上涨
            })
    
    df = pd.DataFrame(data)
    
    # ============ 测试1: 研报模式收益率 ============
    print("\n【测试1: 研报模式 (entry_offset=1) 收益率】")
    config1 = LabelConfig(label_type='return', return_periods=[1, 5, 10], entry_offset=1)
    gen1 = LabelGenerator(config1)
    df1 = gen1.generate_labels(df.copy())
    print(f"公式: P[t+period+1] / P[t+1] - 1")
    print(df1[df1['order_book_id'] == '000001.SZ'][['trade_date', 'close', 'label_1d', 'label_5d', 'label_10d']].head(10))
    
    # 手动验证
    print("\n【手动验证 entry_offset=1】")
    test_df = df[df['order_book_id'] == '000001.SZ'].reset_index(drop=True)
    # 第0行: label_10d = close[11] / close[1] - 1 = 15.5 / 10.5 - 1 = 0.4762
    expected_10d = (test_df.loc[11, 'close'] / test_df.loc[1, 'close']) - 1
    actual_10d = df1[df1['order_book_id'] == '000001.SZ'].iloc[0]['label_10d']
    print(f"  第0行 label_10d: 预期={expected_10d:.6f}, 实际={actual_10d:.6f}, 匹配={np.isclose(expected_10d, actual_10d)}")
    
    # 第0行: label_1d = close[2] / close[1] - 1 = 11.0 / 10.5 - 1 = 0.0476
    expected_1d = (test_df.loc[2, 'close'] / test_df.loc[1, 'close']) - 1
    actual_1d = df1[df1['order_book_id'] == '000001.SZ'].iloc[0]['label_1d']
    print(f"  第0行 label_1d: 预期={expected_1d:.6f}, 实际={actual_1d:.6f}, 匹配={np.isclose(expected_1d, actual_1d)}")
    
    # ============ 测试2: 传统模式收益率 ============
    print("\n【测试2: 传统模式 (entry_offset=0) 收益率】")
    config2 = LabelConfig(label_type='return', return_periods=[1, 5, 10], entry_offset=0)
    gen2 = LabelGenerator(config2)
    df2 = gen2.generate_labels(df.copy())
    print(f"公式: P[t+period] / P[t] - 1")
    print(df2[df2['order_book_id'] == '000001.SZ'][['trade_date', 'close', 'label_1d', 'label_5d', 'label_10d']].head(10))
    
    # ============ 测试3: 分类标签 + 研报模式 ============
    print("\n【测试3: 分类标签 + 研报模式 (entry_offset=1)】")
    config3 = LabelConfig(
        label_type='classification',
        return_periods=[10],
        entry_offset=1,
        n_classes=2,
        class_method='threshold',
        thresholds=[0]
    )
    gen3 = LabelGenerator(config3)
    df3 = gen3.generate_labels(df.copy())
    
    # 因为价格线性上涨，所有收益率应该 > 0，分类应该全部是 1
    print(f"分类分布 (预期全为1，因为价格上涨):\n{df3['label'].value_counts()}")
    
    # 验证分类和回归的一致性
    print("\n【验证分类与回归的一致性】")
    # 重新生成回归标签，只看10日
    config_ret = LabelConfig(label_type='return', return_periods=[10], entry_offset=1)
    gen_ret = LabelGenerator(config_ret)
    df_ret = gen_ret.generate_labels(df.copy())
    
    # 取 label，它应该都是正的
    positive_count = (df_ret['label'] > 0).sum()
    class_1_count = (df3['label'] == 1).sum()
    print(f"  回归标签 > 0 的数量: {positive_count}")
    print(f"  分类标签 = 1 的数量: {class_1_count}")
    print(f"  一致性检查: {'通过 ✓' if positive_count == class_1_count else '失败 ✗'}")
    
    # ============ 测试4: 排名标签 + 研报模式 ============
    print("\n【测试4: 排名标签 + 研报模式 (entry_offset=1)】")
    config4 = LabelConfig(
        label_type='rank',
        return_periods=[5],
        entry_offset=1,
        n_quantiles=3
    )
    gen4 = LabelGenerator(config4)
    df4 = gen4.generate_labels(df.copy())
    print(f"排名分布:\n{df4['label'].value_counts().sort_index()}")
    
    # ============ 测试5: 统计信息 ============
    print("\n【测试5: 标签统计 (研报模式)】")
    stats = gen1.get_label_statistics(df1, ['label_1d', 'label_5d', 'label_10d'])
    print(stats.to_string())
    
    print("\n" + "=" * 80)
    print("✅ LabelGenerator 重构版测试完成")
    print("=" * 80)
