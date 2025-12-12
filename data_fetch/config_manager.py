"""
配置管理模块
负责管理数据获取流程中的所有配置参数
"""
import os
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TimeConfig:
    """
    时间配置

    管理数据时间范围和频率参数。

    Args:
        start_date (str): 数据开始日期，格式 'YYYY-MM-DD'，默认 '2015-01-01'。
            数据提取的起始日期。
            
        end_date (str): 数据结束日期，格式 'YYYY-MM-DD'，默认 '2024-12-31'。
            数据提取的结束日期（包含该日期）。
            
        frequency (str): 数据频率，可选值:
            - '1d': 日频（默认，最常用）
            - '1w': 周频
            - '1m': 月频
    """
    start_date: str = '2015-01-01'
    end_date: str = '2024-12-31'
    frequency: str = '1d'  # 数据频率: 1d-日频, 1w-周频, 1m-月频
    
    def validate(self):
        """验证时间配置"""
        try:
            datetime.strptime(self.start_date, '%Y-%m-%d')
            datetime.strptime(self.end_date, '%Y-%m-%d')
            assert self.start_date <= self.end_date, "开始日期不能晚于结束日期"
        except ValueError as e:
            raise ValueError(f"日期格式错误: {e}")


@dataclass
class DataSourceConfig:
    """
    数据源配置

    指定数据来源、市场、认证方式等信息。

    Args:
        source (str): 数据源标识，可选值:
            - 'rq': 米筐（推荐，国内最全）
            - 'clickhouse': ClickHouse 数据库（ETF 数据）
            - 'tushare': Tushare（免费，但数据不如米筐全）
            - 'baostock': BaoStock（免费）
            默认: 'rq'
            
        market (str): 市场标识，可选值:
            - 'cn': 中国 A 股市场（默认）
            - 'us': 美国股市
            
        auth_method (str): 认证方式，可选值:
            - 'config_file': 使用 ~/.rqdatac/config.yaml 或指定配置文件（默认）
            - 'account': 账号+密码认证（与 rqdatac.init(user, pass) 相同）
            - 'token': 直接 Token 认证（rqdatac.init(token='xxx')）
            
        username (Optional[str]): 米筐账号（auth_method='account' 时必填）
        password (Optional[str]): 米筐密码（auth_method='account' 时必填）
        token (Optional[str]): 米筐 Token（auth_method='token' 时必填）
        config_path (Optional[str]): 自定义 rqdatac 配置文件路径（auth_method='config_file' 可选）

        clickhouse_config (Optional[Dict]): ClickHouse 连接配置
            当 source='clickhouse' 时必需，包含:
            - host: 主机地址
            - port: 端口号
            - user: 用户名
            - password: 密码
            - database: 数据库名
    """
    source: str = 'rq'  # 数据源: rq-米筐, clickhouse-ClickHouse, tushare-Tushare
    market: str = 'cn'  # 市场: cn-中国, us-美国
    auth_method: str = 'config_file'  # 认证方式
    username: Optional[str] = None  # 米筐账号
    password: Optional[str] = None  # 米筐密码
    token: Optional[str] = None     # 米筐Token
    config_path: Optional[str] = None  # rqdatac配置文件路径
    clickhouse_config: Optional[Dict] = None  # ClickHouse 配置
    
    def validate(self):
        """验证数据源配置"""
        valid_sources = ['rq', 'clickhouse', 'tushare', 'baostock']
        if self.source not in valid_sources:
            raise ValueError(f"不支持的数据源: {self.source}, 仅支持: {valid_sources}")
        
        # 验证认证参数
        if self.source == 'rq':
            if self.auth_method == 'account':
                if not self.username or not self.password:
                    raise ValueError("auth_method='account' 需要提供 username 与 password")
            elif self.auth_method == 'token':
                if not self.token:
                    raise ValueError("auth_method='token' 需要提供 token")
            elif self.auth_method == 'config_file':
                # 可选 config_path，若为空则使用默认路径
                pass
            else:
                raise ValueError("auth_method 仅支持 'config_file'、'account'、'token'")

        # 验证 ClickHouse 配置
        if self.source == 'clickhouse':
            if self.clickhouse_config is None:
                raise ValueError("使用 ClickHouse 数据源时必须提供 clickhouse_config")
            required_keys = ['host', 'port', 'user', 'password', 'database']
            for key in required_keys:
                if key not in self.clickhouse_config:
                    raise ValueError(f"ClickHouse 配置缺少必需参数: {key}")


@dataclass
class UniverseConfig:
    """
    股票池配置

    管理因子分析所涉及的股票范围和筛选规则。

    Args:
        universe_type (str): 股票池类型，可选值:
            - 'csi800': 中证800 指数（默认，大中型股）
            - 'csi300': 沪深 300 指数（大盘蓝筹）
            - 'csi500': 中证 500 指数（中盘）
            - 'all_a': 全部 A 股
            - 'custom': 自定义股票列表
            
        custom_stocks (Optional[List[str]]): 自定义股票列表。
            当 universe_type='custom' 时使用，提供具体股票代码列表。
            
        filters (List[str]): 默认筛选规则列表。
            应用的筛选条件名称列表。
            
        exclude_st (bool): 是否排除 ST 股票，默认 True。
            ST 股票风险较高，多数策略选择排除。
            
        main_board_only (bool): 是否仅保留主板，默认 False。
            为 True 时只包含沪深京三家交易所的主板股票。
            
        min_list_days (int): 最小上市天数，默认 0。
            只包含上市至少此天数的股票（新股除外）。
    """
    universe_type: str = 'csi800'  # csi800, csi300, csi500, all_a, custom
    custom_stocks: Optional[List[str]] = None  # 自定义股票列表
    filters: List[str] = field(default_factory=lambda: ['exclude_st', 'main_board_only'])
    
    # 筛选规则
    exclude_st: bool = True  # 排除ST股票
    main_board_only: bool = False  # 仅主板
    min_list_days: int = 0  # 最小上市天数
    
    def validate(self):
        """验证股票池配置"""
        valid_types = ['csi800', 'csi300', 'csi500', 'all_a', 'custom']
        if self.universe_type not in valid_types:
            raise ValueError(f"不支持的股票池类型: {self.universe_type}")
        
        if self.universe_type == 'custom' and not self.custom_stocks:
            raise ValueError("自定义股票池需要提供股票列表")


@dataclass
class DataFieldsConfig:
    """
    数据字段配置

    Args:
        price_fields (List[str]): 行情字段列表（如 open/high/low/close/volume）
        valuation_fields (List[str]): 估值字段列表（如 pe/pb/market_cap）
        share_fields (List[str]): 股本相关字段
        industry_source (str): 行业来源（如 'sws'、'citics'），默认 'sws'
        industry_level (int): 行业级别，默认 1
    """
    # 行情数据字段
    price_fields: List[str] = field(default_factory=lambda: [
        'open', 'high', 'low', 'close', 'volume', 'total_turnover',
        'limit_up', 'limit_down', 'num_trades'
    ])
    
    # VWAP（成交量加权平均价格）- 用于回测真实成交价格
    include_vwap: bool = True  # 是否获取 VWAP 数据
    
    # 估值数据字段
    valuation_fields: List[str] = field(default_factory=lambda: [
        'pe_ratio', 'pe_ratio_ttm', 'pb_ratio', 'ps_ratio', 'ps_ratio_ttm',
        'market_cap', 'a_share_market_val_in_circulation'
    ])
    
    # 股本数据字段
    share_fields: List[str] = field(default_factory=lambda: [
        'total_shares', 'shares_outstanding', 'float_a_share_quantity'
    ])
    
    # 行业分类
    industry_source: str = 'sws'  # sws-申万, citics-中信, zjh-证监会
    industry_level: int = 1  # 行业分类级别


@dataclass
class StorageConfig:
    """
    存储配置

    Args:
        save_dir (str): 数据保存根目录，默认 'rq_data_parquet'
        file_format (str): 文件格式，'parquet','csv','hdf5'，默认 'parquet'
        compression (Optional[str]): 压缩方式（如 'gzip','snappy'）
        basic_data_dir (str): 基础数据子目录名，默认 'basic_data'
        daily_data_dir (str): 日线数据子目录名，默认 'daily_data'
    """
    save_dir: str = 'rq_data_parquet'
    file_format: str = 'parquet'  # parquet, csv, hdf5
    compression: Optional[str] = None  # gzip, snappy, None
    
    # 子目录结构
    basic_data_dir: str = 'basic_data'
    daily_data_dir: str = 'daily_data'
    
    def get_full_path(self, subdir: str, filename: str) -> str:
        """获取完整文件路径"""
        return os.path.join(self.save_dir, subdir, filename)
    
    def ensure_dirs(self):
        """确保目录存在"""
        os.makedirs(os.path.join(self.save_dir, self.basic_data_dir), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, self.daily_data_dir), exist_ok=True)


@dataclass
class ProcessConfig:
    """
    处理流程配置

    Args:
        batch_size (int): 批量处理大小，默认 500
        retry_times (int): 重试次数，默认 3
        sleep_interval (float): 请求间隔秒数，默认 0.1
        timeout (int): 请求超时（秒），默认 30
        use_multiprocessing (bool): 是否使用并行，默认 False
        max_workers (int): 最大并发工作线程数，默认 4
    """
    batch_size: int = 500  # 批处理大小
    retry_times: int = 3  # 重试次数
    sleep_interval: float = 0.1  # 请求间隔(秒)
    timeout: int = 30  # 超时时间(秒)
    
    # 并行处理
    use_multiprocessing: bool = False
    max_workers: int = 4

    # 断点续传
    resume_enabled: bool = True
    resume_entities: List[str] = field(default_factory=lambda: ['price', 'valuation', 'share'])


@dataclass
class FeatureConfig:
    """
    特征工程配置

    Args:
        technical_indicators (List[str]): 技术指标列表（如 sma, volatility）
        lag_periods (List[int]): 特征滞后期列表
        ma_windows (List[int]): 移动平均窗口列表
        return_periods (List[int]): 收益率计算周期列表
        volatility_window (int): 波动率计算窗口大小
    """
    # 技术指标
    technical_indicators: List[str] = field(default_factory=lambda: [
        'sma', 'volatility', 'momentum', 'volume_ratio'
    ])
    
    # 滞后期数
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])
    
    # 移动平均周期
    ma_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    
    # 收益率计算周期
    return_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    
    # 波动率计算窗口
    volatility_window: int = 20


class ConfigManager:
    """
    配置管理器

    管理项目中各个子配置（时间、数据源、股票池、字段、存储、处理、特征）。

    Args:
        config_path (Optional[str]): 可选 YAML 配置文件路径，若提供则从文件加载并覆盖默认配置。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径(YAML格式)
        """
        self.time = TimeConfig()
        self.data_source = DataSourceConfig()
        self.universe = UniverseConfig()
        self.fields = DataFieldsConfig()
        self.storage = StorageConfig()
        self.process = ProcessConfig()
        self.feature = FeatureConfig()
        
        if config_path and os.path.exists(config_path):
            self.load_from_yaml(config_path)
        
        self.validate_all()
    
    def load_from_yaml(self, config_path: str):
        """从YAML文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        if not config_dict:
            return
        
        # 加载各个配置模块
        if 'time_settings' in config_dict:
            self._update_dataclass(self.time, config_dict['time_settings'])
        
        if 'data_source' in config_dict:
            self._update_dataclass(self.data_source, config_dict['data_source'])
        
        if 'universe' in config_dict:
            self._update_dataclass(self.universe, config_dict['universe'])
        
        if 'fields' in config_dict:
            self._update_dataclass(self.fields, config_dict['fields'])
        
        if 'storage' in config_dict:
            self._update_dataclass(self.storage, config_dict['storage'])
        
        if 'process' in config_dict:
            self._update_dataclass(self.process, config_dict['process'])
        
        if 'features' in config_dict:
            self._update_dataclass(self.feature, config_dict['features'])
    
    def _update_dataclass(self, obj: Any, updates: Dict):
        """更新dataclass对象的属性"""
        for key, value in updates.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def validate_all(self):
        """验证所有配置"""
        self.time.validate()
        self.data_source.validate()
        self.universe.validate()
        self.storage.ensure_dirs()
    
    def to_dict(self) -> Dict:
        """导出为字典"""
        return {
            'time_settings': vars(self.time),
            'data_source': vars(self.data_source),
            'universe': vars(self.universe),
            'fields': vars(self.fields),
            'storage': vars(self.storage),
            'process': vars(self.process),
            'features': vars(self.feature)
        }
    
    def save_to_yaml(self, save_path: str):
        """保存配置到YAML文件"""
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"ConfigManager(source={self.data_source.source}, " \
               f"universe={self.universe.universe_type}, " \
               f"period={self.time.start_date}~{self.time.end_date})"
