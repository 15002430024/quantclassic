"""
DataConfig - 数据管理配置类

使用面向对象的配置替代字典配置
统一管理所有数据相关的配置参数
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import yaml

# 使用相对导入（相对于 quantclassic 包）
try:
    from ..config.base_config import BaseConfig
except ImportError:
    # 直接运行脚本时的后备导入
    from config.base_config import BaseConfig


@dataclass
class DataConfig(BaseConfig):
    """
    数据管理核心配置类

    管理数据路径、特征工程、预处理、加载与缓存等与数据相关的全局参数。

    Args:
        base_dir (str): 数据文件根目录，默认 'rq_data_parquet'。
            数据加载器查找数据文件的基础路径。
            
        cache_dir (str): 缓存目录，默认 'cache/data_set'。
            处理后的特征工程结果、拆分数据等会缓存到此处。
            
        output_dir (str): 输出目录，默认 'output'。
            模型、图表、报告等输出到此目录。
            
        data_file (str): 数据文件名，默认 'train_data_final.parquet'。
            从 base_dir 中加载的具体文件名。
            
        data_format (str): 数据格式，可选值:
            - 'parquet': Parquet 格式（默认，快速高效）
            - 'csv': CSV 格式（易于查看和编辑）
            - 'hdf5': HDF5 格式（大文件优化）
            
        window_size (int): 序列窗口长度（时间步数），默认 40。
            用于创建滑动窗口数据用于 RNN 模型训练。
            每个样本包含过去 40 个交易日的数据。
            
        label_col (str): 标签列名，默认 'y_processed'。
            数据框中标签列的列名。
            
        stock_col (str): 股票代码列名，默认 'ts_code'。
            唯一标识每只股票的列名。
            
        time_col (str): 时间列名，默认 'trade_date'。
            交易日期列的列名，用于时间序列拆分。
            
        exclude_cols (List[str]): 要排除的列名列表。
            这些列不会被用作特征（通常是 ID 列、目标列等）。
            
        feature_cols (Optional[List[str]]): 特征列列表，默认 None。
            - None: 自动检测（除 exclude_cols 外的所有数值列）
            - 列表: 手动指定特征列
            
        standardize_method (str): 标准化方法，可选值:
            - 'zscore': 标准正态分布标准化（默认）
            - 'minmax': 最小最大标准化
            - 'robust': 鲁棒标准化（对异常值不敏感）
            - 'none': 不进行标准化
            
        fill_na_method (str): 缺失值填充方法，可选值:
            - 'forward': 向前填充（默认）
            - 'backward': 向后填充
            - 'mean': 用平均值填充
            - 'zero': 用 0 填充
            
        winsorize_limits (List[float]): 去极值截断比例，默认 [0.01, 0.01]。
            [下限, 上限]，表示下端和上端各截断的百分比。
            
        split_strategy (str): 数据拆分策略，可选值:
            - 'time_series': 时间序列拆分（默认，按时间顺序）
            - 'stratified': 分层拆分
            - 'random': 随机拆分
            - 'rolling': 滚动窗口拆分
            
        train_ratio (float): 训练集比例，默认 0.7（70%）。
        val_ratio (float): 验证集比例，默认 0.15（15%）。
        test_ratio (float): 测试集比例，默认 0.15（15%）。
            三者应相加为 1.0。
            
        train_end_date (Optional[str]): 训练集结束日期，格式 'YYYY-MM-DD'。
            当 split_strategy='time_series' 时指定训练集的时间边界。
            
        val_end_date (Optional[str]): 验证集结束日期，格式 'YYYY-MM-DD'。
            当 split_strategy='time_series' 时指定验证集的时间边界。
            
        rolling_window_size (int): 滚动窗口大小（交易日数），默认 252。
            当 split_strategy='rolling' 时使用。
            
        rolling_step (int): 滚动步长（交易日数），默认 63。
            每次滚动前进的天数。
            
        batch_size (int): 数据加载批次大小，默认 256。
            每个批次包含的样本数。
            
        num_workers (int): 数据加载进程数，默认 0。
            - 0: 主进程加载（默认）
            - >0: 使用多进程并行加载（加快速度）
            
        pin_memory (bool): 是否将数据固定在 GPU 内存，默认 False。
            仅对 GPU 训练有效，可加速数据传输。
            
        shuffle_train (bool): 是否打乱训练数据，默认 True。
            帮助模型更好地泛化。
            
        use_dtype_optimization (bool): 是否使用 dtype 优化，默认 True。
            将 float64 转换为 float32 以节省内存。
            
        chunk_size (Optional[int]): 分块加载大小，默认 None。
            用于大文件分块读取。
            
        enable_cache (bool): 是否启用缓存，默认 True。
            缓存处理后的数据以加快加载速度。
            
        cache_feature_engineering (bool): 是否缓存特征工程结果，默认 True。
        cache_split_data (bool): 是否缓存拆分后的数据，默认 True。
        cache_expire_hours (int): 缓存过期时间（小时），默认 24。
            超过此时间的缓存会被重新计算。
            
        enable_validation (bool): 是否启用数据验证，默认 True。
            验证缺失值、异常值、样本数量等。
            
        max_na_ratio (float): 最大缺失值比例，默认 0.3。
            超过此比例的列会被警告或删除。
            
        min_samples_per_stock (int): 每只股票最小样本数，默认 60。
            样本不足的股票会被过滤。
            
        detect_outliers (bool): 是否检测异常值，默认 True。
        outlier_std_threshold (float): 异常值标准差阈值，默认 5.0。
            超过该倍数标准差的值被认为是异常值。
        
        enable_window_transform (bool): 是否启用窗口级变换，默认 True。
            在 Dataset.__getitem__ 中对每个窗口进行实时变换（研报标准）。
            
        window_price_log (bool): 是否启用价格对数变换，默认 True。
            公式: log(price_{t-i} / close_t)
            将窗口内所有价格除以窗口末端的收盘价，然后取对数。
            效果: close_t = 0, 其他价格为相对偏差。
            
        price_cols (List[str]): 需要进行对数变换的价格列。
            默认 ['open', 'high', 'low', 'close', 'vwap']。
            
        close_col (str): 基准收盘价列名，默认 'close'。
            用作价格对数变换的分母。
            
        window_volume_norm (bool): 是否启用成交量标准化，默认 True。
            公式: volume_{t-i} / mean(volume_in_window)
            将窗口内的成交量除以该窗口的平均成交量。
            效果: 均值附近 ≈ 1.0。
            
        volume_cols (List[str]): 需要进行标准化的成交量列。
            默认 ['vol', 'amount']。

        label_rank_normalize (bool): 是否对标签做截面排名标准化，默认 False。
            每个交易日内对标签进行排名，再映射到指定输出范围。

        label_rank_output_range (Tuple[float, float]): 截面排名输出范围，默认 (-1, 1)。
            例如 (-1, 1) 映射为对称区间，(0, 1) 映射为百分比分数。

        use_daily_batch (bool): 是否启用日批次模式，默认 False。
            开启后 __getitem__ 返回当日所有股票，适配动态图训练。

        graph_builder_config (Optional[Dict[str, Any]]): 图构建器配置字典（默认 None）。
            建议传入 GraphBuilderConfig.to_dict() 的结果。

        shuffle_dates (bool): 日批次模式下是否打乱交易日顺序，默认 True。
            训练时建议 True，验证/测试建议 False。
            
        verbose (bool): 是否打印详细日志，默认 True。
        log_level (str): 日志级别，可选值: 'DEBUG'/'INFO'/'WARNING'/'ERROR'。
        save_data_report (bool): 是否保存数据报告，默认 True。
    """
    
    # ==================== 数据路径配置 ====================
    base_dir: str = 'rq_data_parquet'
    cache_dir: str = 'cache/data_set'
    output_dir: str = 'output'
    
    # ==================== 数据文件配置 ====================
    data_file: str = 'train_data_final.parquet'
    data_format: str = 'parquet'  # 'parquet', 'csv', 'hdf5'
    
    # ==================== 特征工程参数 ====================
    window_size: int = 40
    label_col: str = 'y_processed'
    stock_col: str = 'ts_code'
    time_col: str = 'trade_date'
    
    # 需要排除的列
    exclude_cols: List[str] = field(default_factory=lambda: [
        'ts_code', 'trade_date', 'y_processed', 'y_raw', 
        'y_winsorized', 'industry_name'
    ])
    
    # 特征列（自动检测或手动指定）
    feature_cols: Optional[List[str]] = None
    
    # ==================== 标准化和预处理 ====================
    standardize_method: str = 'zscore'  # 'zscore', 'minmax', 'robust', 'none'
    fill_na_method: str = 'forward'  # 'forward', 'backward', 'mean', 'zero'
    winsorize_limits: List[float] = field(default_factory=lambda: [0.01, 0.01])
    
    # ==================== 数据划分策略 ====================
    split_strategy: str = 'time_series'  # 'time_series', 'stratified', 'random', 'rolling'
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # 时间序列划分的日期切点（可选）
    train_end_date: Optional[str] = None
    val_end_date: Optional[str] = None
    
    # 滚动窗口参数
    rolling_window_size: int = 252  # 交易日数
    rolling_step: int = 63  # 滚动步长
    
    # ==================== 数据加载参数 ====================
    batch_size: int = 256
    num_workers: int = 0
    # GPU训练时可开启以加速数据传输
    pin_memory: bool = False
    # 是否打乱训练集
    shuffle_train: bool = True
    
    # 内存优化
    use_dtype_optimization: bool = True  # 使用 float32 替代 float64
    chunk_size: Optional[int] = None  # 分块加载大小
    
    # ==================== 缓存策略 ====================
    enable_cache: bool = True
    # 是否缓存特征工程结果
    cache_feature_engineering: bool = True
    # 是否缓存拆分后的数据
    cache_split_data: bool = True
    # 缓存有效期（小时）
    cache_expire_hours: int = 24
    
    # ==================== 数据验证参数 ====================
    enable_validation: bool = True
    max_na_ratio: float = 0.3  # 最大缺失值比例
    min_samples_per_stock: int = 60  # 每只股票最小样本数
    detect_outliers: bool = True
    outlier_std_threshold: float = 5.0
    
    # ==================== 窗口级变换配置（研报标准）====================
    # 在 Dataset.__getitem__ 中对每个窗口进行实时变换
    enable_window_transform: bool = True  # 是否启用窗口级变换
    
    # 价格对数变换: log(price / close_t)
    # 将窗口内所有价格除以窗口末端的收盘价，然后取对数
    window_price_log: bool = True  # 是否启用价格对数变换
    price_cols: List[str] = field(default_factory=lambda: ['open', 'high', 'low', 'close', 'vwap'])
    close_col: str = 'close'  # 基准收盘价列名
    
    # 成交量标准化: volume / mean(volume_in_window)
    # 将窗口内的成交量除以该窗口的平均成交量
    window_volume_norm: bool = True  # 是否启用成交量标准化
    volume_cols: List[str] = field(default_factory=lambda: ['vol', 'amount'])
    
    # 🆕 标签窗口级时序排名标准化
    # 在每个时间窗口内，对标签进行排名并映射到指定范围
    # 避免使用未来信息（只使用历史窗口内的标签计算排名）
    label_rank_normalize: bool = False  # 是否启用标签窗口级排名标准化
    label_rank_output_range: Tuple[float, float] = (-1.0, 1.0)  # 排名标准化输出范围
    
    # ==================== 🆕 日批次模式配置 ====================
    # 日批次模式：每个 batch 是一个交易日的所有股票，适用于 GNN 动态图训练
    use_daily_batch: bool = False  # 是否启用日批次模式
    
    # 动态图构建配置（仅在 use_daily_batch=True 时生效）
    graph_builder_config: Optional[Dict[str, Any]] = None  # 图构建器配置
    # 示例配置：
    # graph_builder_config = {
    #     'type': 'hybrid',
    #     'alpha': 0.7,
    #     'corr_method': 'cosine',
    #     'top_k': 10,
    #     'industry_col': 'industry_name',
    # }
    
    shuffle_dates: bool = True  # 是否打乱日期顺序（日批次模式）
    
    # ==================== 日志和调试 ====================
    verbose: bool = True
    log_level: str = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    save_data_report: bool = True
    
    def __post_init__(self):
        """初始化后处理"""
        # 创建必要的目录
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 调用验证
        self.validate()
    
    def validate(self) -> bool:
        """验证配置"""
        # 验证比例总和
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total_ratio <= 1.01):
            raise ValueError(f"数据划分比例总和应为1.0，当前为{total_ratio}")
        
        # 验证窗口大小
        if self.window_size <= 0:
            raise ValueError("window_size 必须大于 0")
        
        # 验证批次大小
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        
        return True
    
    @property
    def data_path(self) -> str:
        """完整数据文件路径"""
        return os.path.join(self.base_dir, self.data_file)
    
    # 继承自 BaseConfig 的方法：
    # - from_yaml(yaml_path)
    # - to_yaml(yaml_path)
    # - from_dict(config_dict)
    # - to_dict()
    # - update(**kwargs)
    # - validate()


# 预定义配置模板
class ConfigTemplates:
    """预定义的配置模板"""
    
    @staticmethod
    def default() -> DataConfig:
        """默认配置"""
        return DataConfig()
    
    @staticmethod
    def quick_test() -> DataConfig:
        """快速测试配置（小数据集）"""
        return DataConfig(
            window_size=20,
            batch_size=128,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            enable_cache=False,
        )
    
    @staticmethod
    def production() -> DataConfig:
        """生产环境配置（高性能）"""
        return DataConfig(
            batch_size=512,
            num_workers=4,
            pin_memory=True,
            enable_cache=True,
            cache_feature_engineering=True,
            use_dtype_optimization=True,
        )
    
    @staticmethod
    def backtest() -> DataConfig:
        """回测配置（滚动窗口）"""
        return DataConfig(
            split_strategy='rolling',
            rolling_window_size=252,
            rolling_step=21,
            enable_cache=True,
        )


if __name__ == '__main__':
    # 测试配置类
    print("=" * 80)
    print("DataConfig 测试")
    print("=" * 80)
    
    # 创建默认配置
    config = DataConfig()
    print("\n1. 默认配置:")
    print(config)
    
    # 测试配置更新
    print("\n2. 更新配置:")
    config.update(batch_size=512, window_size=60)
    print(f"  batch_size: {config.batch_size}")
    print(f"  window_size: {config.window_size}")
    
    # 测试保存和加载
    print("\n3. 保存配置到YAML:")
    yaml_path = 'cache/test_config.yaml'
    os.makedirs('cache', exist_ok=True)
    config.to_yaml(yaml_path)
    print(f"  已保存到: {yaml_path}")
    
    # 测试模板
    print("\n4. 配置模板:")
    print(f"  快速测试: batch_size={ConfigTemplates.quick_test().batch_size}")
    print(f"  生产环境: num_workers={ConfigTemplates.production().num_workers}")
    print(f"  回测: split_strategy={ConfigTemplates.backtest().split_strategy}")
    
    print("\n✅ 配置类测试完成")
