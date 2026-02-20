"""
回测系统配置模块
包含所有回测相关的配置参数
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch
import yaml

# 导入 BaseConfig 基类
try:
    from ..config.base_config import BaseConfig
except ImportError:
    from config.base_config import BaseConfig


@dataclass
class BacktestConfig(BaseConfig):
    """
    回测系统配置类

    管理回测所需的所有参数：数据路径、因子处理、组合构建、调仓频率、基准设置、成本与可视化等。

    Args:
        data_dir (str): 数据目录，默认 'output'。
        output_dir (str): 回测结果输出目录，默认 'output/backtest'。
        model_path (Optional[str]): 预训练模型路径，若为 None 则不加载。
        batch_size (int): 批处理大小，默认 256。
        window_size (int): 时间窗口大小（序列长度），默认 40。
        device (str): 运行设备，例如 'cuda' 或 'cpu'。
        feature_cols (Optional[List[str]]): 特征列列表，None 时自动推断。
        label_col (str): 标签列名，默认 'y_processed'。
        winsorize_method (str): 去极值方法: 'quantile'/'mad'/'std'。
        standardize_method (str): 标准化方法: 'zscore'/'minmax'/'rank'。
        n_groups (int): 分组数量（用于分组多空），默认 10。
        rebalance_freq (str): 换仓频率: 'daily'/'weekly'/'biweekly'/'monthly'。
        rebalance_day (str): 换仓日: 'first'/'last'/'middle'。
        benchmark_index (Optional[str]): 基准指数标识: 'hs300'/'zz500'/'zz800' 或 None。
        custom_benchmark_col (Optional[str]): 自定义基准列名（当 benchmark_index='custom' 时使用）。
        stock_universe (Optional[str]): 回测股票池限制: 'hs300' 等。
        weight_method (str): 权重分配方法: 'equal'/'value_weight'/'factor_weight'。
        long_ratio (float): 做多比例，默认 0.2。
        short_ratio (float): 做空比例，默认 0.2。
        commission_rate (float): 交易佣金率，默认 0.0003（3个基点）。
        slippage_rate (float): 滑点率，默认 0.001（10个基点）。
        plot_style (str): 绘图样式，默认 'seaborn'。
        figure_size (tuple): 图表尺寸，默认 (12,6)。
        generate_excel (bool): 是否生成 Excel 报告，默认 True。
    """
    
    # =====================================================================
    # 数据路径配置
    # =====================================================================
    data_dir: str = 'output'
    output_dir: str = 'output/backtest'
    model_path: Optional[str] = None
    
    # =====================================================================
    # 因子生成配置
    # =====================================================================
    # 批次大小
    batch_size: int = 256
    # 滑动窗口大小
    window_size: int = 40
    # 设备
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 特征列（留空则自动识别）
    feature_cols: Optional[List[str]] = None
    # 标签列
    label_col: str = 'y_processed'
    
    # =====================================================================
    # 因子处理配置
    # =====================================================================
    # 去极值方法: 'quantile', 'mad', 'std'
    winsorize_method: str = 'quantile'
    # 去极值百分位 (仅用于quantile方法)
    winsorize_quantiles: tuple = (0.025, 0.975)
    # MAD倍数 (仅用于mad方法)
    mad_threshold: float = 3.0
    # 标准差倍数 (仅用于std方法)
    std_threshold: float = 3.0
    
    # 标准化方法: 'zscore', 'minmax', 'rank'
    standardize_method: str = 'zscore'
    
    # 是否进行行业中性化
    industry_neutral: bool = False
    # 行业列名
    industry_col: str = 'industry_name'
    
    # 是否进行市值中性化
    market_value_neutral: bool = False
    # 市值列名
    market_value_col: str = 'market_value'
    
    # 缺失值填充方法: 'mean', 'median', 'forward', 'zero'
    fillna_method: str = 'median'
    
    # =====================================================================
    # 组合构建配置
    # =====================================================================
    # 分组数量
    n_groups: int = 10
    # 换仓频率: 'daily', 'weekly', 'biweekly', 'monthly'
    rebalance_freq: str = 'monthly'
    # 换仓日: 'first', 'last', 'middle'
    rebalance_day: str = 'last'
    
    # =====================================================================
    # 基准指数配置
    # =====================================================================
    # 基准指数: None, 'hs300', 'zz500', 'zz800', 'custom'
    benchmark_index: Optional[str] = None
    # 自定义基准收益率列名（当 benchmark_index='custom' 时使用）
    custom_benchmark_col: Optional[str] = None
    # 基准股票池（用于策略股票池限制）: None, 'hs300', 'zz500', 'zz800'
    stock_universe: Optional[str] = None
    
    # 权重分配方法: 'equal', 'value_weight', 'factor_weight'
    weight_method: str = 'equal'
    
    # 多空组合配置
    long_ratio: float = 0.2  # 做多前20%
    short_ratio: float = 0.2  # 做空后20%
    
    # 个股权重约束
    max_stock_weight: float = 0.1  # 单股最大权重10%
    min_stock_weight: float = 0.0  # 单股最小权重0%
    
    # 行业权重约束
    max_industry_weight: float = 0.3  # 单行业最大权重30%
    
    # =====================================================================
    # IC分析配置
    # =====================================================================
    # IC计算方法: 'pearson', 'spearman'
    ic_method: str = 'spearman'
    # 多期IC分析的持有期 (天数)
    holding_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    # IC显著性检验水平
    ic_significance_level: float = 0.05
    
    # =====================================================================
    # 绩效评估配置
    # =====================================================================
    # 年化因子 (默认252个交易日)
    annual_factor: int = 252
    # 无风险利率 (年化)
    risk_free_rate: float = 0.03
    # 基准收益列名
    benchmark_col: Optional[str] = None
    
    # 绩效分析窗口 (天数)
    rolling_window: int = 60
    
    # =====================================================================
    # 回测引擎配置
    # =====================================================================
    # 回测引擎: 'general_backtest' (GeneralBacktest, 内嵌)
    engine: str = 'general_backtest'
    # 买入价格字段 (仅 general_backtest 引擎使用): 'open', 'close'
    buy_price: str = 'open'
    # 卖出价格字段 (仅 general_backtest 引擎使用): 'open', 'close'
    sell_price: str = 'close'
    # GeneralBacktest 引擎专用配置
    general_backtest_options: Dict[str, Any] = field(default_factory=lambda: {
        'rebalance_threshold': 0.005,   # 调仓阈值
        'initial_capital': 1.0,         # 初始资金
        'adj_factor_col': 'adj_factor', # 复权因子列名
        'close_price_col': 'close',     # 收盘价列名
    })

    # =====================================================================
    # 交易成本配置
    # =====================================================================
    # 是否考虑交易成本
    consider_cost: bool = False
    # 交易佣金率
    commission_rate: float = 0.0003
    # 印花税率
    stamp_tax_rate: float = 0.001
    # 冲击成本率
    slippage_rate: float = 0.001
    
    # =====================================================================
    # 可视化配置
    # =====================================================================
    # 图表样式: 'seaborn', 'ggplot', 'default'
    plot_style: str = 'seaborn'
    # 图表尺寸
    figure_size: tuple = (12, 6)
    # DPI
    dpi: int = 100
    # 是否保存图表
    save_plots: bool = True
    # 图表格式
    plot_format: str = 'png'
    
    # =====================================================================
    # 报告配置
    # =====================================================================
    # 是否生成PDF报告
    generate_pdf: bool = False
    # 是否生成Excel报告
    generate_excel: bool = True
    # 报告标题
    report_title: str = "因子回测报告"
    # 报告作者
    report_author: str = "QuantClassic"
    
    # =====================================================================
    # 性能优化配置
    # =====================================================================
    # 并行计算进程数 (0表示不使用并行)
    n_jobs: int = 0
    # 是否使用缓存
    use_cache: bool = False
    # 缓存目录
    cache_dir: str = 'cache/backtest'
    
    # =====================================================================
    # 日志配置
    # =====================================================================
    # 日志级别: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    log_level: str = 'INFO'
    # 日志文件路径
    log_file: Optional[str] = None
    # 是否在控制台显示日志
    console_log: bool = True
    
    def __post_init__(self):
        """初始化后的验证"""
        # 验证配置合理性
        assert 0 <= self.long_ratio <= 1, "long_ratio必须在[0, 1]之间"
        assert 0 <= self.short_ratio <= 1, "short_ratio必须在[0, 1]之间"
        assert self.n_groups >= 2, "n_groups必须>=2"
        assert self.winsorize_method in ['quantile', 'mad', 'std'], \
            "winsorize_method必须是'quantile', 'mad'或'std'"
        assert self.standardize_method in ['zscore', 'minmax', 'rank'], \
            "standardize_method必须是'zscore', 'minmax'或'rank'"
        assert self.rebalance_freq in ['daily', 'weekly', 'biweekly', 'monthly'], \
            "rebalance_freq必须是'daily', 'weekly', 'biweekly'或'monthly'"
        assert self.weight_method in ['equal', 'value_weight', 'factor_weight'], \
            "weight_method必须是'equal', 'value_weight'或'factor_weight'"
        assert self.ic_method in ['pearson', 'spearman'], \
            "ic_method必须是'pearson'或'spearman'"
        assert self.engine in ['general_backtest'], \
            "engine 仅支持 'general_backtest'"
        assert self.buy_price in ['open', 'close'], \
            "buy_price必须是'open'或'close'"
        assert self.sell_price in ['open', 'close'], \
            "sell_price必须是'open'或'close'"
        
        # 验证基准指数配置
        if self.benchmark_index is not None:
            assert self.benchmark_index in ['hs300', 'zz500', 'zz800', 'custom'], \
                "benchmark_index必须是None, 'hs300', 'zz500', 'zz800'或'custom'"
        
        if self.stock_universe is not None:
            assert self.stock_universe in ['hs300', 'zz500', 'zz800'], \
                "stock_universe必须是None, 'hs300', 'zz500'或'zz800'"
    
    def validate(self) -> bool:
        """
        验证配置有效性 (重写 BaseConfig.validate)
        
        Returns:
            是否有效
        """
        # 调用 __post_init__ 中的断言逻辑已在初始化时执行
        return True


@dataclass
class FactorConfig:
    """因子特定配置 (用于管理多个因子)"""
    
    # 因子名称
    name: str = 'factor'
    # 因子描述
    description: str = ''
    # 因子类型: 'raw', 'processed', 'combined'
    factor_type: str = 'raw'
    # 因子方向: 1表示正向，-1表示反向
    direction: int = 1
    # 是否启用
    enabled: bool = True
    
    def __post_init__(self):
        assert self.direction in [1, -1], "direction必须是1或-1"

        assert self.factor_type in ['raw', 'processed', 'combined'], \
            "factor_type必须是'raw', 'processed'或'combined'"


# 预设配置模板
class ConfigTemplates:
    """配置模板库"""
    
    @staticmethod
    def default() -> BacktestConfig:
        """默认配置"""
        return BacktestConfig()
    
    @staticmethod
    def fast_test() -> BacktestConfig:
        """快速测试配置（性能优先）"""
        return BacktestConfig(
            batch_size=512,
            n_groups=5,
            rebalance_freq='monthly',
            holding_periods=[1, 5],
            save_plots=False,
            generate_pdf=False,
            n_jobs=4
        )
    
    @staticmethod
    def detailed_analysis() -> BacktestConfig:
        """详细分析配置（全面性优先）"""
        return BacktestConfig(
            n_groups=10,
            holding_periods=[1, 5, 10, 20, 60],
            industry_neutral=True,
            market_value_neutral=True,
            consider_cost=True,
            rolling_window=120,
            save_plots=True,
            generate_pdf=True,
            generate_excel=True
        )
    
    @staticmethod
    def production() -> BacktestConfig:
        """生产环境配置"""
        return BacktestConfig(
            batch_size=256,
            n_groups=10,
            consider_cost=True,
            commission_rate=0.0003,
            slippage_rate=0.001,
            use_cache=True,
            log_level='INFO',
            save_plots=True
        )
