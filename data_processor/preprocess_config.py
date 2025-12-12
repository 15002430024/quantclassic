"""
数据预处理配置模块 - 使用面向对象配置
"""
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.base_config import BaseConfig


class ProcessMethod(Enum):
    """
    处理方法枚举
    
    定义预处理管道中所有支持的处理方法。每种方法对应不同的数据处理操作。
    使用 add_step() 时，直接传入对应的参数即可。
    
    🎯 1️⃣ 标签生成
    GENERATE_LABELS: 生成多周期收益率标签
        用途: 在预处理管道中自动生成未来收益率标签
        
        Args（直接传入 add_step）:
            stock_col (str): 股票代码列名。默认 'order_book_id'
            time_col (str): 时间列名。默认 'trade_date'
            price_col (str): 价格列（分子：未来价格）。默认 'close'
            base_price_col (str|None): 基准价格列（分母）。
                - None: T日基准（传统方式）
                - 'close': T+1基准（研报标准，推荐）
            label_type (str): 标签类型。默认 'return'
            return_periods (List[int]): 收益率周期列表。默认 [1, 5, 10]
            return_method (str): 计算方法 'simple'|'log'。默认 'simple'
            label_prefix (str): 标签前缀。默认 'y_ret' → y_ret_1d, y_ret_5d
        
        示例:config.add_step(
                    name='生成标签',
                    method=ProcessMethod.GENERATE_LABELS,
                    stock_col='order_book_id',
                    time_col='trade_date',
                    price_col='close',
                    base_price_col='close',      # 研报标准
                    return_periods=[1, 5, 10],
                    label_prefix='y_ret'
            )
    
    2、标准化/归一化
  
    Z_SCORE: 标准正态分布标准化
        公式: (x - 均值) / 标准差
        结果: 均值=0, 标准差=1
        
        Args:
            normalize_mode (str): 标准化模式。默认 'cross_section'
                - 'cross_section': 截面标准化（同一时间点所有股票）
                - 'time_series': 时序标准化（同一股票历史数据）
                - 'global': 全局标准化（所有数据）
        
        示例:
            config.add_step('截面标准化', ProcessMethod.Z_SCORE, 
                           normalize_mode='cross_section')
    
    
    MINMAX: 最小最大标准化
        公式: (x - min) / (max - min)
        结果: 所有值在指定区间内
        
        Args:
            normalize_mode (str): 标准化模式。默认 'cross_section'
                - 'cross_section' | 'time_series' | 'global'
            output_range (tuple): 输出区间。默认 (0, 1)
        
        示例:
            config.add_step('MinMax标准化', ProcessMethod.MINMAX,
                           normalize_mode='cross_section',
                           output_range=(0, 1))
    
    
    RANK: 排名标准化
        公式: (rank - 1) / (n - 1) 映射到指定区间
        用途: 对分布鲁棒，处理异常值敏感场景
        
        Args:
            normalize_mode (str): 标准化模式。默认 'cross_section'
                - 'cross_section' | 'time_series' | 'global'
            output_range (tuple): 输出区间。默认 (-1, 1)
            rank_method (str): 排名方法。默认 'average'
                - 'average': 相同值取平均排名
                - 'min': 相同值取最小排名
                - 'max': 相同值取最大排名
                - 'first': 按出现顺序排名
                - 'dense': 密集排名（无跳跃）
        
        示例:
            config.add_step('排名标准化', ProcessMethod.RANK,
                           normalize_mode='cross_section',
                           output_range=(-1, 1),
                           rank_method='average')
    

    3️⃣ 中性化（因子正交化）
    
    SIMSTOCK_LABEL_NEUTRALIZE: SimStock相似股票标签中性化
        方法: 用相似股票的因子值中性化标签
        用途: 标签中性化（专用于标签工程）
        
        Args:
            label_column (str): 输入标签列名。默认 'y_ret_1d'
            output_column (str): 输出列名。默认 'alpha_label'
            similarity_threshold (float): 相似度阈值 [0,1]。默认 0.7
                - 0.5: 宽松，选择较多相似股票
                - 0.7: 平衡（推荐）
                - 0.8: 严格
            lookback_window (int): 回看窗口（交易日）。默认 252
                - 60: 约3个月
                - 252: 约1年（推荐）
                - 504: 约2年
            min_similar_stocks (int): 最少相似股票数。默认 5
            recalc_interval (int): 相关性矩阵重算间隔（交易日）。默认 20
                - 1: 每天重算（最精确，最慢）
                - 20: 每月重算（推荐，平衡精度与速度）
                - 60: 每季度重算（最快）
            correlation_method (str): 相关性方法。默认 'pearson'
                - 'pearson': 皮尔逊相关（对异常值敏感）
                - 'spearman': 斯皮尔曼相关（对异常值鲁棒）

    MEAN_NEUTRALIZE: 平均值中性化
        方法: 按行业、市值分位数计算均值中性
        用途: 快速中性化（比OLS更快）
        
        Args:
            industry_column (str): 行业列名。默认 'industry_name'
            market_cap_column (str): 市值列名。默认 'total_mv'
        
        示例:
            config.add_step('均值中性化', ProcessMethod.MEAN_NEUTRALIZE,
                           industry_column='industry_name')

    OLS_NEUTRALIZE: OLS回归中性化
        方法: 用OLS回归移除市值、行业因素影响
        用途: 特征中性化
        
        Args:
            industry_column (str): 行业列名。默认 'industry_name'
            market_cap_column (str): 市值列名。默认 'total_mv'
            min_samples (int): 最小样本数。默认 10
        
        示例:
            config.add_step('特征中性化', ProcessMethod.OLS_NEUTRALIZE,
                           industry_column='industry_name',
                           market_cap_column='total_mv',
                           min_samples=10)
    4️⃣ 极值处理

    WINSORIZE: 百分位截断（温和处理）
        方法: 上下两端各截断一定比例
        用途: 保留值域范围，温和处理异常值
        
        Args:
            limits (list|tuple): 截断比例 [下界, 上界]。默认 [0.025, 0.025]
                - [0.01, 0.01]: 上下各截1%
                - [0.025, 0.025]: 上下各截2.5%（推荐）
                - [0.05, 0.05]: 上下各截5%
        
        示例:
            config.add_step('去极值', ProcessMethod.WINSORIZE,
                           limits=[0.025, 0.025])
    
    
    CLIP: 固定百分位截断（激进处理）
        方法: 按固定百分位截断到边界值
        用途: 激进处理异常值
        
        Args:
            lower_percentile (float): 下界百分位。默认 0.01 (1%)
            upper_percentile (float): 上界百分位。默认 0.99 (99%)
            lower (float): 直接指定下界值（可选，优先于百分位）
            upper (float): 直接指定上界值（可选，优先于百分位）
        
        示例:
            # 按百分位截断
            config.add_step('Clip截断', ProcessMethod.CLIP,
                           lower_percentile=0.01,
                           upper_percentile=0.99)
            
            # 按固定值截断
            config.add_step('Clip截断', ProcessMethod.CLIP,
                           lower=-1e10, upper=1e10)
    
  
    🎯 5️⃣ 缺失值处理

    FILLNA_MEDIAN: 中位数填充
        用途: 保留分布特征，对异常值稳健
        Args: 无额外参数
        示例: config.add_step('填充缺失', ProcessMethod.FILLNA_MEDIAN)
    
    FILLNA_MEAN: 均值填充
        用途: 快速填充，基础方法
        Args: 无额外参数
        示例: config.add_step('填充缺失', ProcessMethod.FILLNA_MEAN)
    
    FILLNA_FORWARD: 向前填充
        用途: 时间序列数据填充（用前一个有效值填充）
        Args: 无额外参数
        示例: config.add_step('向前填充', ProcessMethod.FILLNA_FORWARD)
    
    FILLNA_ZERO: 零值填充
        用途: 特殊场景（如交易量缺失表示无交易）
        Args: 无额外参数
        示例: config.add_step('零值填充', ProcessMethod.FILLNA_ZERO)
    
  
    📊 完整流程示例（研报标准6步）
    
        config = PreprocessConfig()
        
        # 1. 生成标签
        config.add_step('生成标签', ProcessMethod.GENERATE_LABELS,
            base_price_col='close', return_periods=[1, 5, 10])
        
        # 2. 去极值
        config.add_step('去极值', ProcessMethod.WINSORIZE, limits=[0.025, 0.025])
        
        # 3. 截面标准化
        config.add_step('截面标准化', ProcessMethod.Z_SCORE, normalize_mode='cross_section')
        
        # 4. 特征OLS中性化
        config.add_step('特征中性化', ProcessMethod.OLS_NEUTRALIZE,
            industry_column='industry_name', market_cap_column='total_mv')
        
        # 5. 标签SimStock中性化
        config.add_step('标签中性化', ProcessMethod.SIMSTOCK_LABEL_NEUTRALIZE,
            lookback_window=252, recalc_interval=20)
        
        # 6. 缺失值填充
        config.add_step('填充缺失', ProcessMethod.FILLNA_MEDIAN)
    
    💡 最佳实践：
        推荐流程: 标签生成 → 去极值 → 标准化 → 中性化 → 填充缺失
        顺序说明:
          1. 生成标签（必须第一步，因为需要未来价格）
          2. 去极值（标准化前进行，避免异常值影响）
          3. 标准化（中性化前进行）
          4. 中性化（可选，用于因子工程）
          5. 填充缺失（最后进行，避免填充值被处理）
    """
    # 标签生成
    GENERATE_LABELS = "generate_labels"
    
    # 标准化/归一化
    Z_SCORE = "z_score"
    MINMAX = "minmax"
    RANK = "rank"
    
    # 中性化
    OLS_NEUTRALIZE = "ols_neutralize"
    MEAN_NEUTRALIZE = "mean_neutralize"
    SIMSTOCK_LABEL_NEUTRALIZE = "simstock_label_neutralize"  # 专注于标签工程
    
    # 极值处理
    WINSORIZE = "winsorize"
    CLIP = "clip"
    
    # 缺失值处理
    FILLNA_MEDIAN = "fillna_median"
    FILLNA_MEAN = "fillna_mean"
    FILLNA_FORWARD = "fillna_forward"
    FILLNA_ZERO = "fillna_zero"


@dataclass
class ProcessingStep:
    """
    单个处理步骤配置

    定义数据处理管道中的单个步骤，包括处理方法、目标列、参数等。

    Args:
        name (str): 步骤名称，用于标识和日志记录。
            例如: '去极值处理', '截面标准化', '特征中性化', '缺失值填充'等
            
        method (ProcessMethod): 处理方法，可选值:
            **标准化/归一化**:
            - 'z_score': 标准正态分布标准化（均值0，标准差1）
            - 'minmax': 最小最大标准化到[0,1]区间
            - 'rank': 排名标准化到[0,1]区间
            
            **中性化**:
            - 'ols_neutralize': OLS回归中性化（对标准差、行业中性化）
            - 'mean_neutralize': 平均值中性化
            - 'simstock_label_neutralize': SimStock相似股票标签中性化（专用于标签工程）
            
            **极值处理**:
            - 'winsorize': 百分位截断（上下两端各截断一定比例）
            - 'clip': 固定值截断
            
            **缺失值处理**:
            - 'fillna_median': 用中位数填充缺失值
            - 'fillna_mean': 用平均值填充缺失值
            - 'fillna_forward': 向前填充缺失值
            - 'fillna_zero': 用0填充缺失值
            
        features (Union[str, List[str], None]): 要处理的列名，可选值:
            - 单列名字符串: 'close' 只处理该列
            - 列名列表: ['close', 'volume', 'high', 'low'] 处理多列
            - None: 处理所有数值特征（默认行为）
            
        enabled (bool): 是否启用该处理步骤。
            - True: 执行该步骤（默认）
            - False: 跳过该步骤（用于临时禁用）
            
        params (Dict[str, Any]): 方法特定参数字典。
            常见参数:
            - 'limits': [0.025, 0.025] (Winsorize两端截断比例)
            - 'normalize_mode': 'cross_section' (z_score的标准化模式)
            - 'industry_column': 'industry_name' (OLS中性化的行业列)
            - 'market_cap_column': 'market_cap' (OLS中性化的市值列)
    """
    name: str                                    # 步骤名称
    method: ProcessMethod                        # 处理方法（SimStock仅用于标签工程）
    features: Union[str, List[str], None] = None # 要处理的特征列(None表示所有特征)
    enabled: bool = True                         # 是否启用
    params: Dict[str, Any] = field(default_factory=dict)  # 方法特定参数
    
    def __post_init__(self):
        """验证配置"""
        if isinstance(self.method, str):
            self.method = ProcessMethod(self.method)


@dataclass
class LabelGeneratorConfig(BaseConfig):
    """
    标签生成配置
    
    🎯 功能：
        在预处理管道中自动生成多周期未来收益率标签，支持研报标准和传统标准。
        支持按股票分组、自定义周期、自定义前缀等灵活配置。
    
    📋 Args:
        enabled (bool): 是否启用标签生成。
            默认: True
            取值: True | False
            说明: 
              - True: 在预处理管道中执行标签生成
              - False: 跳过标签生成步骤
            
        stock_col (str): 股票代码列名。
            默认: 'order_book_id'
            类型: str
            说明: 用于分组计算各股票的时间序列标签
            常见值: 'order_book_id', 'stock_code', 'symbol'
            
        time_col (str): 时间列名。
            默认: 'trade_date'
            类型: str
            说明: 用于时间序列排序和未来价格偏移计算
            常见值: 'trade_date', 'date', 'datetime'
            
        price_col (str): 价格列名（分子，未来价格）。
            默认: 'close'
            类型: str
            说明: 计算收益率使用的未来价格列（price_{t+n}）
            常见值: 'close', 'vwap', 'open', 'high', 'low'
            使用场景:
              - 'close': 使用收盘价计算收益率
              - 'vwap': 使用成交量加权平均价（更接近真实交易）
            
        base_price_col (Optional[str]): 基准价格列名（分母，基准价格）。
            默认: None
            类型: Optional[str]
            说明: 计算收益率使用的基准价格列（分母）
            取值:
              - None: 使用 T 日价格作为基准（传统方式）
                公式: label_t = (price_{t+n} / price_t) - 1
                含义: 在T日收盘预测并交易（不可能）
              - 'close': 使用 T+1 日价格作为基准（研报标准）
                公式: label_t = (price_{t+n} / price_{t+1}) - 1
                含义: 在T日收盘预测，T+1日开盘交易
            推荐: 使用 'close' 实现研报标准
            
        label_type (str): 标签类型。
            默认: 'return'
            类型: str
            说明: 生成的标签类型
            取值:
              - 'return': 生成收益率标签（当前支持）
              - 'class': 生成分类标签（未来支持）
            
        return_periods (List[int]): 收益率周期列表（单位：交易日）。
            默认: [1, 5, 10]
            类型: List[int]
            说明: 生成多个周期的收益率标签
            示例:
              - [1]: 仅生成1日收益率 → y_ret_1d
              - [1, 5, 10]: 生成三个周期 → y_ret_1d, y_ret_5d, y_ret_10d
              - [1, 2, 3, 5, 10, 20]: 生成多个周期
            推荐范围: [1, 5, 10, 20] 或自定义
            
        return_method (str): 收益率计算方法。
            默认: 'simple'
            类型: str
            说明: 如何计算收益率
            取值:
              - 'simple': 简单收益率 (price/base_price - 1)
                公式: r = (P_{t+n} / P_{t+1}) - 1
                特点: 直观易懂，易于解释
              - 'log': 对数收益率 log(price/base_price)
                公式: r = ln(P_{t+n} / P_{t+1})
                特点: 数学上更严谨，适合统计分析
            推荐: 'simple' 用于初始分析，'log' 用于统计建模
            
        label_prefix (str): 标签列名前缀。
            默认: 'y_ret'
            类型: str
            说明: 生成的标签列名格式为 {label_prefix}_{period}d
            示例:
              - 前缀='y_ret' → y_ret_1d, y_ret_5d, y_ret_10d
              - 前缀='ret' → ret_1d, ret_5d, ret_10d
              - 前缀='future_ret' → future_ret_1d, future_ret_5d
            建议:
              - 使用 'y_ret' 或 'y_' 前缀区分标签和特征
              - 避免使用 'ret' 避免与历史收益特征混淆
            
        neutralize (bool): 是否对生成的标签进行中性化。
            默认: False
            类型: bool
            说明: 是否立即进行中性化处理
            取值:
              - False: 不进行中性化（推荐，在后续步骤中处理）
              - True: 在生成时进行中性化
            推荐: False，在后续的管道步骤中配置中性化处理
    
    💡 使用示例：
        
        # 研报标准（推荐）
        config = LabelGeneratorConfig(
            enabled=True,
            stock_col='order_book_id',
            time_col='trade_date',
            price_col='close',
            base_price_col='close',  # T+1 基准
            return_periods=[1, 5, 10],
            label_prefix='y_ret'
        )
        
        # 传统标准
        config = LabelGeneratorConfig(
            enabled=True,
            price_col='close',
            base_price_col=None,  # T 基准
            return_periods=[1, 5, 10]
        )
        
        # 使用 VWAP 价格
        config = LabelGeneratorConfig(
            price_col='vwap',
            base_price_col='vwap',
            label_prefix='y_vwap_ret'
        )
    
    🔗 相关配置：
        - PreprocessConfig.label_config: 在预处理配置中使用
        - ProcessMethod.GENERATE_LABELS: 对应的处理方法
        - DataPreprocessor: 执行标签生成的处理器
    """
    enabled: bool = True
    stock_col: str = 'order_book_id'
    time_col: str = 'trade_date'
    price_col: str = 'close'
    base_price_col: Optional[str] = None  # None=T日（传统），'close'=T+1日（研报标准）
    label_type: str = 'return'
    return_periods: List[int] = field(default_factory=lambda: [1, 5, 10])
    return_method: str = 'simple'
    label_prefix: str = 'y_ret'  # 建议使用 y_ret 前缀
    neutralize: bool = False


@dataclass
class NeutralizeConfig(BaseConfig):
    """
    中性化配置
    
    🎯 功能：
        控制不同中性化方法的参数，包括 OLS 中性化、市值/行业中性化、
        SimStock 标签中性化等。
    
    📋 Args:
        industry_column (str): 行业列名。
            默认: 'industry_name'
            类型: str
            说明: 用于 OLS 和其他中性化方法中的行业因子
            常见值: 'industry_name', 'industry_code', 'sector'
            用途: 区分不同行业的特征差异
            
        market_cap_column (str): 市值列名。
            默认: 'total_mv'
            类型: str
            说明: 用于中性化中的市值因子（规模因子）
            常见值: 'total_mv', 'market_cap', 'market_value'
            用途: 控制规模效应影响
            
        min_samples (int): OLS 回归最小样本数量。
            默认: 10
            类型: int
            说明: 当截面样本数小于此值时，跳过中性化处理
            用途: 避免样本过少导致的不稳定估计
            推荐范围: [5, 20]
            
        label_column (str): SimStock 中性化的输入标签列名。
            默认: 'y_ret_1d'
            类型: str
            说明: 指定要进行 SimStock 中性化的原始标签列
            常见值: 'y_ret_1d', 'y_ret_5d', 'ret_1d'
            用途: 用于标签工程中的中性化处理
            重要: 应与标签生成时的 label_prefix 配合使用
            
        similarity_threshold (float): SimStock 相似度阈值。
            默认: 0.7
            类型: float
            范围: [0.0, 1.0]
            说明: 相关系数超过此阈值的股票视为相似
            取值说明:
              - 0.5: 范围广，选择较多相似股票
              - 0.7: 平衡，中等严格程度（推荐）
              - 0.8: 严格，选择更相似的股票
              - 0.9: 非常严格，只选择最相似的股票
            用途: 控制相似股票的筛选严格程度
            
        lookback_window (int): SimStock 计算的历史回溯窗口（交易日）。
            默认: 252
            类型: int
            说明: 使用过去 N 个交易日的数据计算相似度
            常见值:
              - 60: 约3个月
              - 120: 约6个月
              - 252: 约1年（推荐，标准年度数据）
              - 504: 约2年
            用途: 历史时间范围越长，相似度计算越稳定
            
        min_similar_stocks (int): SimStock 最少相似股票数量。
            默认: 5
            类型: int
            说明: 当相似股票少于此数时，可能跳过或降低阈值
            推荐范围: [3, 10]
            用途: 确保中性化有足够的对标股票
            
        recalc_interval (int): 相关性矩阵重计算间隔（交易日）。
            默认: 20
            类型: int
            说明: 每隔 N 个交易日重新计算一次相关性矩阵
            常见值:
              - 1: 每天重算（最精确，但计算量大）
              - 5: 每周重算
              - 20: 每月重算（推荐，平衡精度与速度）
              - 60: 每季度重算（快速模式）
            用途: 加速 SimStock 计算，避免每天重算相关性矩阵
            权衡: 间隔越大计算越快，但相似股票可能略过时
            
        correlation_method (str): 相关性计算方法。
            默认: 'pearson'
            类型: str
            取值:
              - 'pearson': 皮尔逊相关系数（线性相关）
                适用: 正态分布的特征
                敏感性: 对异常值敏感
                
              - 'spearman': 斯皮尔曼等级相关系数（排名相关）
                适用: 任意分布
                敏感性: 对异常值不敏感
            推荐: 'pearson' 用于初始分析，'spearman' 用于鲁棒分析
            
        output_column (str): 中性化后输出列名。
            默认: 'alpha_label'
            类型: str
            说明: SimStock 中性化完成后的结果保存到此列
            常见值: 'alpha_label', 'neutral_label', 'adjusted_label'
            用途: 标记中性化后的标签列
            建议: 使用明确的命名表示已中性化
    
    💡 使用示例：
        
        # 基础配置（推荐）
        config = NeutralizeConfig(
            industry_column='industry_name',
            market_cap_column='total_mv',
            min_samples=10,
            label_column='y_ret_1d',
            similarity_threshold=0.7,
            lookback_window=252,
            min_similar_stocks=5,
            recalc_interval=20,
            correlation_method='pearson',
            output_column='alpha_label'
        )
        
        # 严格配置（选择更相似的股票）
        config = NeutralizeConfig(
            similarity_threshold=0.8,
            min_similar_stocks=10
        )
        
        # 鲁棒配置（使用等级相关）
        config = NeutralizeConfig(
            correlation_method='spearman'
        )
    
    🔗 相关配置：
        - PreprocessConfig.neutralize_config: 在预处理配置中使用
        - ProcessMethod.OLS_NEUTRALIZE: OLS中性化方法
        - ProcessMethod.SIMSTOCK_LABEL_NEUTRALIZE: 标签中性化方法
        - LabelGeneratorConfig.label_prefix: 与标签生成配置配合
    
    📊 推荐参数组合：
        
        场景1: 基础因子工程
          - similarity_threshold: 0.7
          - lookback_window: 252
          - min_similar_stocks: 5
        
        场景2: 严格因子工程
          - similarity_threshold: 0.8
          - lookback_window: 252
          - min_similar_stocks: 10
        
        场景3: 快速迭代
          - similarity_threshold: 0.6
          - lookback_window: 120
          - min_similar_stocks: 3
    """
    # OLS中性化参数
    industry_column: str = 'industry_name'
    market_cap_column: str = 'total_mv'
    min_samples: int = 10
    
    # SimStock标签中性化参数（专注于标签工程）
    label_column: str = 'y_ret_1d'  # 只对标签做中性化（未来1日收益率标签）
    similarity_threshold: float = 0.7
    lookback_window: int = 252
    min_similar_stocks: int = 5
    recalc_interval: int = 20  # 每20天重算相关性矩阵（加速计算）
    correlation_method: str = 'pearson'  # 'pearson', 'spearman'
    output_column: str = 'alpha_label'   # 输出的alpha标签名


@dataclass
class PreprocessConfig(BaseConfig):
    """
    预处理总配置
    
    🎯 功能：
        管道式管理一系列数据处理步骤，包括标签生成、去极值、标准化、中性化、
        缺失值填充等。支持灵活的步骤组合和参数配置。
    
    📋 Args:
        pipeline_steps (List[ProcessingStep]): 处理步骤列表，按添加顺序依次执行。
        column_mapping (Dict[str, str]): 列名映射字典，用于统一字段名。
        groupby_columns (List[str]): 分组列列表，用于截面处理（默认：['trade_date']）。
        id_columns (List[str]): ID列列表，不参与处理但保留在输出中（默认：['order_book_id', 'trade_date']）。
        label_config (LabelGeneratorConfig): 标签生成配置对象（通过 add_step 自动更新）。
        neutralize_config (NeutralizeConfig): 中性化配置对象（通过 add_step 自动更新）。
        save_intermediate (bool): 是否保存每一步的中间结果（默认：False）。
        intermediate_dir (str): 中间结果保存目录（默认：'intermediate_results'）。
        validate_each_step (bool): 是否验证每一步的数据完整性（默认：True）。
        verbose (bool): 是否打印详细处理日志（默认：True）。
    
    💡 使用示例（推荐：参数直接在 add_step 中配置）：
        
        config = PreprocessConfig()
        
        # 步骤1: 生成标签（参数直接传入，自动更新 label_config）
        config.add_step(
            name='生成多周期标签', 
            method=ProcessMethod.GENERATE_LABELS,
            stock_col='order_book_id',
            time_col='trade_date',
            price_col='close',
            base_price_col='close',      # 研报标准：T+1基准
            return_periods=[1, 5, 10],
            label_prefix='y_ret'
        )
        
        # 步骤2: 去极值
        config.add_step('去极值', ProcessMethod.WINSORIZE, limits=[0.025, 0.025])
        
        # 步骤3: 截面标准化
        config.add_step('截面标准化', ProcessMethod.Z_SCORE, normalize_mode='cross_section')
        
        # 步骤4: 特征OLS中性化
        config.add_step('特征中性化', ProcessMethod.OLS_NEUTRALIZE,
            industry_column='industry_name',
            market_cap_column='total_mv'
        )
        
        # 步骤5: 标签SimStock中性化（参数直接传入，自动更新 neutralize_config）
        config.add_step(
            name='标签SimStock中性化', 
            method=ProcessMethod.SIMSTOCK_LABEL_NEUTRALIZE,
            label_column='y_ret_1d',
            output_column='alpha_label',
            similarity_threshold=0.7,
            lookback_window=252,
            min_similar_stocks=5,
            recalc_interval=20,
            correlation_method='pearson'
        )
        
        # 步骤6: 缺失值填充
        config.add_step('填充缺失', ProcessMethod.FILLNA_MEDIAN)
        
        # 执行预处理
        processor = DataPreprocessor(config)
        df_processed = processor.fit_transform(df_raw, target_column='y_ret_1d')
    
    🔗 相关类：
        - LabelGeneratorConfig: 标签生成配置
        - NeutralizeConfig: 中性化配置
        - ProcessingStep: 单个处理步骤
        - ProcessMethod: 处理方法枚举
        - DataPreprocessor: 执行预处理的处理器
    
    📊 推荐配置模板：
        
        # 基础流程（3步）
        config = PreprocessConfig()
        config.add_step('生成标签', ProcessMethod.GENERATE_LABELS, 
                        base_price_col='close', return_periods=[1, 5, 10])
        config.add_step('去极值', ProcessMethod.WINSORIZE, limits=[0.025, 0.025])
        config.add_step('填充缺失', ProcessMethod.FILLNA_MEDIAN)
        
        # 完整流程（6步，含中性化）
        config = PreprocessConfig()
        config.add_step('生成标签', ProcessMethod.GENERATE_LABELS, 
                        base_price_col='close', return_periods=[1, 5, 10])
        config.add_step('去极值', ProcessMethod.WINSORIZE, limits=[0.025, 0.025])
        config.add_step('截面标准化', ProcessMethod.Z_SCORE, normalize_mode='cross_section')
        config.add_step('特征中性化', ProcessMethod.OLS_NEUTRALIZE)
        config.add_step('标签中性化', ProcessMethod.SIMSTOCK_LABEL_NEUTRALIZE,
                        lookback_window=252, recalc_interval=20)
        config.add_step('填充缺失', ProcessMethod.FILLNA_MEDIAN)
    
    💾 保存和加载：
        
        # 保存配置到YAML
        config.to_yaml('preprocess_config.yaml')
        
        # 从YAML加载配置
        config = PreprocessConfig.from_yaml('preprocess_config.yaml')
    """
    # 处理步骤
    pipeline_steps: List[ProcessingStep] = field(default_factory=list)
    
    # 字段映射
    column_mapping: Dict[str, str] = field(default_factory=dict)
    
    # 分组配置
    groupby_columns: List[str] = field(default_factory=lambda: ['trade_date'])
    
    # ID列(不进行处理)
    id_columns: List[str] = field(default_factory=lambda: ['order_book_id', 'trade_date'])
    
    # 标签生成配置
    label_config: LabelGeneratorConfig = field(default_factory=LabelGeneratorConfig)
    
    # 中性化配置
    neutralize_config: NeutralizeConfig = field(default_factory=NeutralizeConfig)
    
    # 保存选项
    save_intermediate: bool = False
    intermediate_dir: str = 'intermediate_results'
    
    # 验证选项
    validate_each_step: bool = True
    verbose: bool = True
    
    def validate(self) -> bool:
        """验证配置"""
        # 验证处理步骤
        for step in self.pipeline_steps:
            if not isinstance(step, ProcessingStep):
                raise ValueError(f"pipeline_steps 中的元素必须是 ProcessingStep 类型")
        
        return True
    
    def add_step(self, name: str, method: Union[str, ProcessMethod], 
                 features: Union[str, List[str], None] = None,
                 enabled: bool = True, **params):
        """
        添加处理步骤
        
        对于 GENERATE_LABELS 方法，params 会自动更新 label_config。
        对于 SIMSTOCK_LABEL_NEUTRALIZE 方法，params 会自动更新 neutralize_config。
        
        Args:
            name: 步骤名称
            method: 处理方法
            features: 要处理的特征列
            enabled: 是否启用
            **params: 方法特定参数，会根据方法类型自动更新相关配置
            
        Returns:
            self: 支持链式调用
            
        示例:
            # 方式1：直接在 add_step 中配置（推荐，更简洁）
            config.add_step(
                name='生成多周期标签',
                method=ProcessMethod.GENERATE_LABELS,
                stock_col='order_book_id',
                time_col='trade_date',
                price_col='close',
                base_price_col='close',  # 研报标准
                return_periods=[1, 5, 10],
                label_prefix='y_ret'
            )
            
            # 方式2：先配置 label_config，再 add_step（兼容旧方式）
            config.label_config.base_price_col = 'close'
            config.add_step('生成标签', ProcessMethod.GENERATE_LABELS)
        """
        method_enum = method if isinstance(method, ProcessMethod) else ProcessMethod(method)
        
        # 对于 GENERATE_LABELS，将 params 更新到 label_config
        if method_enum == ProcessMethod.GENERATE_LABELS and params:
            label_params = {}
            for key in list(params.keys()):
                if hasattr(self.label_config, key):
                    label_params[key] = params.pop(key)
            if label_params:
                for k, v in label_params.items():
                    setattr(self.label_config, k, v)
                self.label_config.enabled = True
        
        # 对于 SIMSTOCK_LABEL_NEUTRALIZE，将 params 更新到 neutralize_config
        if method_enum == ProcessMethod.SIMSTOCK_LABEL_NEUTRALIZE and params:
            neutralize_params = {}
            for key in list(params.keys()):
                if hasattr(self.neutralize_config, key):
                    neutralize_params[key] = params.pop(key)
            if neutralize_params:
                for k, v in neutralize_params.items():
                    setattr(self.neutralize_config, k, v)
        
        step = ProcessingStep(
            name=name,
            method=method_enum,
            features=features,
            enabled=enabled,
            params=params
        )
        self.pipeline_steps.append(step)
        return self
    
    # 继承自 BaseConfig 的方法：
    # - from_yaml(yaml_path)
    # - to_yaml(yaml_path)
    # - from_dict(config_dict)
    # - update(**kwargs)
    
    def to_dict(self) -> Dict:
        """转换为字典（覆盖基类方法以正确处理 ProcessingStep）"""
        return {
            'pipeline_steps': [
                {
                    'name': step.name,
                    'method': step.method.value,
                    'features': step.features,
                    'enabled': step.enabled,
                    'params': step.params
                }
                for step in self.pipeline_steps
            ],
            'column_mapping': self.column_mapping,
            'groupby_columns': self.groupby_columns,
            'id_columns': self.id_columns,
            'label_config': self.label_config.to_dict() if isinstance(self.label_config, BaseConfig) else vars(self.label_config),
            'neutralize_config': self.neutralize_config.to_dict() if isinstance(self.neutralize_config, BaseConfig) else vars(self.neutralize_config),
            'save_intermediate': self.save_intermediate,
            'intermediate_dir': self.intermediate_dir,
            'validate_each_step': self.validate_each_step,
            'verbose': self.verbose
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'PreprocessConfig':
        """从字典创建配置"""
        config = cls()
        
        # 加载处理步骤
        if 'pipeline_steps' in config_dict:
            config.pipeline_steps = [
                ProcessingStep(
                    name=step['name'],
                    method=ProcessMethod(step['method']),
                    features=step.get('features'),
                    enabled=step.get('enabled', True),
                    params=step.get('params', {})
                )
                for step in config_dict['pipeline_steps']
            ]
        
        # 加载其他配置
        config.column_mapping = config_dict.get('column_mapping', {})
        config.groupby_columns = config_dict.get('groupby_columns', ['trade_date'])
        config.id_columns = config_dict.get('id_columns', ['order_book_id', 'trade_date'])
        config.save_intermediate = config_dict.get('save_intermediate', False)
        config.intermediate_dir = config_dict.get('intermediate_dir', 'intermediate_results')
        config.validate_each_step = config_dict.get('validate_each_step', True)
        config.verbose = config_dict.get('verbose', True)
        
        # 加载标签生成配置
        if 'label_config' in config_dict:
            lc = config_dict['label_config']
            config.label_config = LabelGeneratorConfig(**lc)
        
        # 加载中性化配置
        if 'neutralize_config' in config_dict:
            nc = config_dict['neutralize_config']
            config.neutralize_config = NeutralizeConfig(**nc)
        
        return config


# 预定义的配置模板
class PreprocessTemplates:
    """预处理配置模板"""
    
    @staticmethod
    def basic_pipeline() -> PreprocessConfig:
        """基础处理流程"""
        config = PreprocessConfig()
        config.add_step('处理无穷值', ProcessMethod.CLIP, params={'lower': -1e10, 'upper': 1e10})
        config.add_step('填充缺失值', ProcessMethod.FILLNA_MEDIAN)
        config.add_step('去极值', ProcessMethod.WINSORIZE, params={'limits': [0.025, 0.025]})
        config.add_step('标准化', ProcessMethod.Z_SCORE)
        return config
    
    @staticmethod
    def advanced_pipeline() -> PreprocessConfig:
        """高级处理流程(包含中性化)"""
        config = PreprocessConfig()
        config.add_step('处理无穷值', ProcessMethod.CLIP, params={'lower': -1e10, 'upper': 1e10})
        config.add_step('填充缺失值', ProcessMethod.FILLNA_MEDIAN)
        config.add_step('去极值', ProcessMethod.WINSORIZE, params={'limits': [0.025, 0.025]})
        config.add_step('市值行业中性化', ProcessMethod.OLS_NEUTRALIZE)
        config.add_step('秩归一化', ProcessMethod.RANK, params={'output_range': (-1, 1)})
        return config
    
    @staticmethod
    def alpha_pipeline() -> PreprocessConfig:
        """Alpha因子处理流程"""
        config = PreprocessConfig()
        config.add_step('处理无穷值', ProcessMethod.CLIP, params={'lower': -1e10, 'upper': 1e10})
        config.add_step('填充缺失值', ProcessMethod.FILLNA_MEDIAN)
        config.add_step('去极值', ProcessMethod.WINSORIZE, params={'limits': [0.01, 0.01]})
        config.add_step('标签中性化', ProcessMethod.SIMSTOCK_LABEL_NEUTRALIZE)
        config.add_step('秩归一化', ProcessMethod.RANK, params={'output_range': (-1, 1)})
        return config
