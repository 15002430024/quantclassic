"""
量化数据获取流水线
整合所有模块,提供统一的执行接口
"""
import os
import logging
from typing import Dict, List, Optional
import pandas as pd

from .config_manager import ConfigManager
from .data_fetcher import DataFetcher
from .data_processor import DataProcessor
from .data_validator import DataValidator
from .resume_manager import ResumeManager


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class QuantDataPipeline:
    """量化数据获取流水线"""
    
    def __init__(
        self, 
        config: Optional[ConfigManager] = None, 
        config_path: Optional[str] = None,
        time_config=None,
        source_config=None,
        universe_config=None,
        fields_config=None,
        storage_config=None,
        feature_config=None,
        process_config=None
    ):
        """
        初始化数据流水线
        
        Args:
            config: 配置管理器实例(优先)
            config_path: 配置文件路径(如果config为None)
            time_config: 时间配置对象（替代方案）
            source_config: 数据源配置对象（替代方案）
            universe_config: 股票池配置对象（替代方案）
            fields_config: 字段配置对象（替代方案）
            storage_config: 存储配置对象（替代方案）
            feature_config: 特征配置对象（替代方案）
        """
        # 方式1: 使用 ConfigManager（原有方式）
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = ConfigManager(config_path)
        # 方式2: 使用分离的配置对象（新方式）
        elif time_config is not None:
            # 创建ConfigManager并设置各项配置
            self.config = ConfigManager()
            if time_config is not None:
                self.config.time = time_config
            if source_config is not None:
                self.config.data_source = source_config
            if universe_config is not None:
                self.config.universe = universe_config
            if fields_config is not None:
                self.config.fields = fields_config
            if storage_config is not None:
                self.config.storage = storage_config
            if feature_config is not None:
                self.config.feature = feature_config
            if process_config is not None:
                self.config.process = process_config
        else:
            self.config = ConfigManager()
        
        logger.info(f"初始化数据流水线: {self.config}")
        
        # 初始化各个模块
        self.resume_manager = ResumeManager(
            self.config.storage,
            enabled=self.config.process.resume_enabled
        )
        self.fetcher = DataFetcher(self.config, resume_manager=self.resume_manager)
        self.processor = DataProcessor(self.config)
        self.validator = DataValidator(self.config)
        
        # 数据缓存
        self.data = {
            'stocks': None,
            'calendar': None,
            'industry': None,
            'price': None,
            'valuation': None,
            'share': None,
            'features': None
        }
    
    def run_full_pipeline(
        self, 
        steps: Optional[List[str]] = None,
        save_intermediate: bool = True,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        执行完整数据流水线
        
        Args:
            steps: 要执行的步骤列表(如果为None则执行全部)
                   可选: ['fetch_basic', 'fetch_daily', 'merge', 'features', 'validate', 'save']
            save_intermediate: 是否保存中间结果
            validate: 是否执行数据验证
            
        Returns:
            最终特征矩阵DataFrame
        """
        logger.info("\n" + "=" * 80)
        logger.info("开始执行完整数据流水线")
        logger.info("=" * 80)
        logger.info(f"配置: {self.config}")
        logger.info(f"保存中间结果: {save_intermediate}")
        logger.info(f"执行验证: {validate}")
        logger.info("=" * 80 + "\n")
        
        # 默认执行所有步骤
        if steps is None:
            steps = ['fetch_basic', 'fetch_daily', 'merge', 'features', 'validate', 'save']
        
        # 1. 获取基础数据
        if 'fetch_basic' in steps:
            self._fetch_basic_data(save=save_intermediate)
        
        # 2. 获取日频数据
        if 'fetch_daily' in steps:
            self._fetch_daily_data(save=save_intermediate)
        
        # 3. 合并数据
        if 'merge' in steps:
            self._merge_data()
        
        # 4. 特征工程
        if 'features' in steps:
            self._build_features()
        
        # 5. 数据验证
        if validate and 'validate' in steps:
            self._validate_data()
        
        # 6. 保存最终结果
        if 'save' in steps:
            self._save_final_data()
        
        logger.info("\n" + "=" * 80)
        logger.info("数据流水线执行完成!")
        logger.info("=" * 80 + "\n")
        
        return self.data['features']
    
    def _fetch_basic_data(self, save: bool = True):
        """获取基础数据"""
        logger.info("\n" + "=" * 80)
        logger.info("步骤1: 获取基础数据")
        logger.info("=" * 80 + "\n")
        
        # 获取股票列表
        self.data['stocks'] = self.fetcher.get_stock_list()
        
        # 获取交易日历
        self.data['calendar'] = self.fetcher.get_trading_calendar()
        
        # 获取行业分类
        order_book_ids = self.data['stocks']['order_book_id'].tolist()
        self.data['industry'] = self.fetcher.get_industry_data(order_book_ids)
        
        # 保存基础数据
        if save:
            self.fetcher.save_data(
                self.data['stocks'], 
                self.config.storage.basic_data_dir, 
                'stock_basic'
            )
            self.fetcher.save_data(
                self.data['calendar'], 
                self.config.storage.basic_data_dir, 
                'trade_calendar'
            )
            self.fetcher.save_data(
                self.data['industry'], 
                self.config.storage.basic_data_dir, 
                'industry_classify'
            )
        
        logger.info(f"✓ 基础数据获取完成")
    
    def _fetch_daily_data(self, save: bool = True):
        """获取日频数据"""
        logger.info("\n" + "=" * 80)
        logger.info("步骤2: 获取日频数据")
        logger.info("=" * 80 + "\n")
        
        if self.data['stocks'] is None:
            logger.warning("股票列表未加载,先加载基础数据...")
            self._fetch_basic_data(save=False)
        
        order_book_ids = self.data['stocks']['order_book_id'].tolist()

        existing_price = self._load_existing_daily('daily_price')
        existing_valuation = self._load_existing_daily('daily_valuation')
        existing_share = self._load_existing_daily('daily_share')
        
        # 获取价格数据
        self.data['price'] = self.fetcher.get_price_data(order_book_ids, existing_price)
        
        # 获取估值数据
        self.data['valuation'] = self.fetcher.get_valuation_data(order_book_ids, existing_valuation)
        
        # 获取股本数据
        self.data['share'] = self.fetcher.get_share_data(order_book_ids, existing_share)
        
        # 保存日频数据
        if save:
            if self.data['price'] is not None and not self.data['price'].empty:
                self.fetcher.save_data(
                    self.data['price'], 
                    self.config.storage.daily_data_dir, 
                    'daily_price'
                )
            if self.data['valuation'] is not None and not self.data['valuation'].empty:
                self.fetcher.save_data(
                    self.data['valuation'], 
                    self.config.storage.daily_data_dir, 
                    'daily_valuation'
                )
            if self.data['share'] is not None and not self.data['share'].empty:
                self.fetcher.save_data(
                    self.data['share'], 
                    self.config.storage.daily_data_dir, 
                    'daily_share'
                )
        
        logger.info(f"✓ 日频数据获取完成")
    
    def _merge_data(self):
        """合并数据"""
        logger.info("\n" + "=" * 80)
        logger.info("步骤3: 合并数据")
        logger.info("=" * 80 + "\n")
        
        if self.data['price'] is None:
            raise ValueError("价格数据未加载,请先执行fetch_daily步骤")
        
        # 合并日频数据
        df_merged = self.processor.merge_daily_data(
            self.data['price'],
            self.data['valuation'],
            self.data['share']
        )
        
        # 合并行业数据（关键步骤）
        if self.data['industry'] is not None and not self.data['industry'].empty:
            logger.info(f"合并行业数据...")
            df_industry = self.data['industry'].copy()
            # 行业数据通常按 order_book_id 存储，与每日数据进行左连接
            df_merged = pd.merge(
                df_merged,
                df_industry,
                on='order_book_id',
                how='left'
            )
            logger.info(f"合并行业后: {df_merged.shape}")
        else:
            logger.warning("⚠️  行业数据未加载，将跳过行业相关特征")
        
        # 计算基础字段
        df_merged = self.processor.calculate_basic_fields(df_merged)
        
        self.data['features'] = df_merged
        logger.info(f"✓ 数据合并完成: {df_merged.shape}")
    
    def _build_features(self):
        """构建特征"""
        logger.info("\n" + "=" * 80)
        logger.info("步骤4: 特征工程")
        logger.info("=" * 80 + "\n")
        
        if self.data['features'] is None:
            raise ValueError("数据未合并,请先执行merge步骤")
        
        # 执行特征工程
        self.data['features'] = self.processor.build_features(self.data['features'])
        
        logger.info(f"✓ 特征工程完成: {self.data['features'].shape}")
    
    def _validate_data(self):
        """验证数据"""
        logger.info("\n" + "=" * 80)
        logger.info("步骤5: 数据验证")
        logger.info("=" * 80 + "\n")
        
        if self.data['features'] is None:
            raise ValueError("特征数据未生成,请先执行features步骤")
        
        # 运行完整验证
        report = self.validator.run_full_validation(self.data['features'])
        
        # 保存验证报告
        report_path = os.path.join(
            self.config.storage.save_dir, 
            'data_quality_report.txt'
        )
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"数据质量报告\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"生成时间: {report['timestamp']}\n")
            f.write(f"总体状态: {report['overall_status']}\n\n")
            
            for check_name, check_result in report['checks'].items():
                f.write(f"\n{check_name}:\n")
                f.write(f"{'-' * 80}\n")
                f.write(f"{check_result}\n")
        
        logger.info(f"✓ 验证报告已保存: {report_path}")
    
    def _save_final_data(self):
        """保存最终数据"""
        logger.info("\n" + "=" * 80)
        logger.info("步骤6: 保存最终数据")
        logger.info("=" * 80 + "\n")
        
        if self.data['features'] is None:
            raise ValueError("特征数据未生成")
        
        # 保存特征矩阵
        filepath = os.path.join(
            self.config.storage.save_dir,
            f'features_raw.{self.config.storage.file_format}'
        )
        
        if self.config.storage.file_format == 'parquet':
            self.data['features'].to_parquet(filepath, index=False)
        elif self.config.storage.file_format == 'csv':
            self.data['features'].to_csv(filepath, index=False)
        
        logger.info(f"✓ 最终数据已保存: {filepath}")
        
        # 保存特征列名
        feature_info = self.processor.get_feature_names(self.data['features'])
        feature_cols_path = os.path.join(
            self.config.storage.save_dir,
            'feature_columns.txt'
        )
        
        with open(feature_cols_path, 'w', encoding='utf-8') as f:
            f.write("特征列名清单\n")
            f.write("=" * 80 + "\n\n")
            
            for category, cols in feature_info.items():
                f.write(f"\n{category.upper()} ({len(cols)} 列):\n")
                f.write("-" * 80 + "\n")
                for col in cols:
                    f.write(f"  - {col}\n")
        
        logger.info(f"✓ 特征列名已保存: {feature_cols_path}")
    
    def run_incremental_update(self, update_date: str):
        """
        增量更新模式(仅更新指定日期的数据)
        
        Args:
            update_date: 更新日期(格式: YYYY-MM-DD)
        """
        logger.info(f"增量更新模式: {update_date}")
        
        # 临时修改配置
        original_start = self.config.time.start_date
        self.config.time.start_date = update_date
        
        # 执行数据获取
        self.run_full_pipeline(
            steps=['fetch_daily', 'merge', 'features', 'save'],
            save_intermediate=False,
            validate=False
        )
        
        # 恢复配置
        self.config.time.start_date = original_start
        
        logger.info("✓ 增量更新完成")
    
    def run_custom_universe(self, custom_stocks: List[str]):
        """
        自定义股票池运行
        
        Args:
            custom_stocks: 自定义股票列表
        """
        logger.info(f"自定义股票池模式: {len(custom_stocks)} 只股票")
        
        # 临时修改配置
        original_type = self.config.universe.universe_type
        original_stocks = self.config.universe.custom_stocks
        
        self.config.universe.universe_type = 'custom'
        self.config.universe.custom_stocks = custom_stocks
        
        # 执行完整流水线
        self.run_full_pipeline()
        
        # 恢复配置
        self.config.universe.universe_type = original_type
        self.config.universe.custom_stocks = original_stocks
        
        logger.info("✓ 自定义股票池处理完成")
    
    def load_existing_data(self) -> pd.DataFrame:
        """
        加载已存在的数据
        
        Returns:
            特征矩阵DataFrame
        """
        filepath = os.path.join(
            self.config.storage.save_dir,
            f'features_raw.{self.config.storage.file_format}'
        )
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"数据文件不存在: {filepath}")
        
        logger.info(f"加载数据: {filepath}")
        
        if self.config.storage.file_format == 'parquet':
            df = pd.read_parquet(filepath)
        elif self.config.storage.file_format == 'csv':
            df = pd.read_csv(filepath, parse_dates=['trade_date'])
        else:
            raise ValueError(f"不支持的文件格式: {self.config.storage.file_format}")
        
        self.data['features'] = df
        logger.info(f"✓ 数据加载完成: {df.shape}")
        
        return df
    
    def get_data_summary(self) -> Dict:
        """
        获取数据摘要
        
        Returns:
            数据摘要字典
        """
        if self.data['features'] is None:
            raise ValueError("特征数据未生成")
        
        df = self.data['features']
        
        summary = {
            'shape': df.shape,
            'stocks': df['order_book_id'].nunique(),
            'date_range': (
                str(df['trade_date'].min().date()),
                str(df['trade_date'].max().date())
            ),
            'features': len(df.columns),
            'missing_ratio': df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        
        return summary

    def _load_existing_daily(self, filename: str) -> Optional[pd.DataFrame]:
        full_path = self.config.storage.get_full_path(
            self.config.storage.daily_data_dir,
            f"{filename}.{self.config.storage.file_format}"
        )
        if not os.path.exists(full_path):
            return None
        try:
            if self.config.storage.file_format == 'parquet':
                df = pd.read_parquet(full_path)
            elif self.config.storage.file_format == 'csv':
                df = pd.read_csv(full_path)
            elif self.config.storage.file_format == 'hdf5':
                df = pd.read_hdf(full_path, key='data')
            else:
                logger.warning("不支持的文件格式, 无法加载: %s", self.config.storage.file_format)
                return None
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
            logger.info("断点续传: 加载已有文件 %s", full_path)
            return df
        except Exception as exc:
            logger.warning("加载已有数据失败, 忽略 %s: %s", full_path, exc)
            return None


# 兼容旧接口: 导出 DataPipeline 名称以便外部直接导入
DataPipeline = QuantDataPipeline
