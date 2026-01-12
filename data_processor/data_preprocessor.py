"""
数据预处理管道 - 主要编排器
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
import pickle
import logging
from pathlib import Path
import yaml
from tqdm.auto import tqdm

from .preprocess_config import PreprocessConfig, ProcessingStep, ProcessMethod
from .feature_processor import FeatureProcessor
from .label_generator import LabelGenerator, LabelConfig
from .window_processor import WindowProcessor, WindowProcessConfig


logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    数据预处理管道
    
    功能:
    1. 配置驱动的多步骤处理流程
    2. fit_transform/transform分离(训练/推理)
    3. 状态保存和加载
    4. 灵活的字段选择和操作配置
    5. 窗口级数据处理（研报标准）
    """
    
    def __init__(self, config: Union[PreprocessConfig, str, dict] = None):
        """
        初始化预处理器
        
        Args:
            config: 配置对象、配置文件路径或字典
        """
        # 加载配置
        if isinstance(config, str):
            self.config = self._load_config_from_file(config)
        elif isinstance(config, dict):
            self.config = PreprocessConfig(**config)
        elif isinstance(config, PreprocessConfig):
            self.config = config
        else:
            self.config = PreprocessConfig()
        
        # 初始化特征处理器（传入 stock_col 以支持列名适配）
        self.feature_processor = FeatureProcessor(
            groupby_columns=self.config.groupby_columns,
            stock_col=self.config.stock_col
        )
        
        # 是否已拟合
        self.is_fitted = False
        
        # 保存的列名(用于验证)
        self.feature_columns = []
        self.id_columns = []
        
        # 实际使用的列名映射（在 fit_transform 时确定）
        self._actual_stock_col = None
        self._actual_time_col = None
        
        logger.info(f"初始化预处理器,管道步骤数: {len(self.config.pipeline_steps)}")
    
    def _load_config_from_file(self, config_path: str) -> PreprocessConfig:
        """从YAML文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return PreprocessConfig(**config_dict)
    
    def set_pipeline_steps(self, steps: List[ProcessingStep]):
        """设置处理流程"""
        self.config.pipeline_steps = steps
        logger.info(f"更新管道步骤数: {len(steps)}")
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        拟合并转换数据(训练模式)
        
        Args:
            df: 输入数据框
            feature_columns: 特征列(如果为None,自动推断)
            target_column: 目标列(用于SimStock等需要收益率的方法)
        
        Returns:
            处理后的数据框
            
        Note:
            自动兼容 DataManager 输出的 ts_code 列和传统的 order_book_id 列。
            如果配置了 auto_adapt_columns=True（默认），会自动检测并适配列名。
        """
        df = df.copy()
        
        # ========== 列名自适应（解决 ts_code vs order_book_id 问题）==========
        if self.config.auto_adapt_columns:
            df, col_mapping = self.config.adapt_columns(df)
            self._actual_stock_col = col_mapping['stock_col']
            self._actual_time_col = col_mapping['time_col']
            # 同步更新 FeatureProcessor 的 stock_col
            self.feature_processor.stock_col = self._actual_stock_col
            logger.info(f"列名适配完成: stock_col='{self._actual_stock_col}', time_col='{self._actual_time_col}'")
        else:
            self._actual_stock_col = self.config.stock_col
            self._actual_time_col = self.config.time_col
        
        # 自动推断特征列
        if feature_columns is None:
            feature_columns = self._infer_feature_columns(df)
        
        self.feature_columns = feature_columns
        
        # 推断ID列（使用实际存在的列）
        self.id_columns = [col for col in self.config.id_columns if col in df.columns]
        if not self.id_columns:
            self.id_columns = [self._actual_stock_col, self._actual_time_col]
        
        logger.info(f"开始fit_transform, 特征数: {len(feature_columns)}, 样本数: {len(df)}")
        
        # 执行管道步骤（添加进度条）
        enabled_steps = [step for step in self.config.pipeline_steps if step.enabled]
        
        with tqdm(total=len(enabled_steps), desc="预处理流程", unit="步骤") as pbar:
            for step in self.config.pipeline_steps:
                if not step.enabled:
                    logger.debug(f"跳过步骤: {step.name}")
                    continue
                
                # 更新进度条描述
                pbar.set_description(f"处理: {step.name}")
                logger.info(f"执行步骤: {step.name} ({step.method.value})")
                
                # 确定处理的特征
                process_features = self._get_process_features(step, feature_columns, df)
                if not process_features:
                    logger.warning(f"步骤 {step.name} 无有效特征,跳过")
                    pbar.update(1)
                    continue
                
                # 执行处理
                df = self._apply_step(df, step, process_features, target_column, fit=True)
                
                logger.info(f"完成步骤: {step.name}, 当前样本数: {len(df)}")
                pbar.update(1)
        
        self.is_fitted = True
        logger.info("fit_transform完成")
        
        return df
    
    def transform(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        转换数据(推理模式,使用已拟合的参数)
        
        Args:
            df: 输入数据框
            target_column: 目标列
        
        Returns:
            处理后的数据框
        """
        if not self.is_fitted:
            raise ValueError("预处理器尚未拟合,请先调用fit_transform()")
        
        df = df.copy()
        
        logger.info(f"开始transform, 样本数: {len(df)}")
        
        # 执行管道步骤
        for step in self.config.pipeline_steps:
            if not step.enabled:
                continue
            
            logger.info(f"执行步骤: {step.name} ({step.method.value})")
            
            # 确定处理的特征
            process_features = self._get_process_features(step, self.feature_columns, df)
            if not process_features:
                continue
            
            # 执行处理(不拟合)
            df = self._apply_step(df, step, process_features, target_column, fit=False)
        
        logger.info("transform完成")
        
        return df
    
    def _infer_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        智能推断特征列
        
        Args:
            df: 数据框
        
        Returns:
            特征列列表
        
        说明:
            • 仅选择数值类型的列
            • 排除ID列（order_book_id, trade_date等）
            • 排除标签列（y_*, label_*, alpha_*）
            • 排除辅助列（industry_name, sector等）
            • 保留所有可以进行数学运算的列
        """
        # 排除ID列和已知非特征列
        exclude_cols = set(self.config.id_columns + 
                          ['industry_name', 'industry_code', 'sector_name', 'ts_code'])
        
        # 添加常见的非特征列名模式
        exclude_patterns = ['y_', 'label_', 'alpha_', 'target']
        
        # 数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 过滤掉ID列和标签列
        feature_cols = []
        for col in numeric_cols:
            # 如果列名在排除列表中，跳过
            if col in exclude_cols:
                continue
            # 如果列名包含排除模式，跳过
            if any(pattern in col for pattern in exclude_patterns):
                continue
            feature_cols.append(col)
        
        logger.info(f"自动推断特征列数: {len(feature_cols)}")
        if len(feature_cols) > 0:
            logger.debug(f"特征列示例（前10个）: {feature_cols[:10]}")
        
        return feature_cols
    
    def _get_required_columns(self, method: ProcessMethod, params: Dict) -> List[str]:
        """
        获取处理方法需要的必需列
        
        Args:
            method: 处理方法
            params: 参数字典
        
        Returns:
            必需列列表
        
        说明:
            • 不同处理方法需要不同的必需列
            • OLS中性化: 需要industry_column和market_cap_column
            • MEAN中性化: 需要industry_column和market_cap_column
            • SimStock: 需要label_column
            • 标签生成: 需要stock_col, time_col, price_col
        """
        required = []
        
        if method == ProcessMethod.OLS_NEUTRALIZE:
            required.extend([
                params.get('industry_column', 'industry_name'),
                params.get('market_cap_column', 'total_mv')
            ])
        
        elif method == ProcessMethod.MEAN_NEUTRALIZE:
            required.extend([
                params.get('industry_column', 'industry_name'),
                params.get('market_cap_column', 'total_mv')
            ])
        
        elif method == ProcessMethod.SIMSTOCK_LABEL_NEUTRALIZE:
            required.append(params.get('label_column', 'y_ret_1d'))
        
        elif method == ProcessMethod.GENERATE_LABELS:
            required.extend([
                params.get('stock_col', 'order_book_id'),
                params.get('time_col', 'trade_date'),
                params.get('price_col', 'close')
            ])
            if params.get('base_price_col'):
                required.append(params['base_price_col'])
        
        elif method in [ProcessMethod.FILLNA_MEDIAN, ProcessMethod.FILLNA_MEAN]:
            # 这些方法可能使用industry_column，但不是必需的
            if 'industry_column' in params and params['industry_column']:
                required.append(params['industry_column'])
        
        return required
    
    def _get_process_features(
        self, 
        step: ProcessingStep, 
        all_features: List[str],
        df: pd.DataFrame
    ) -> List[str]:
        """获取步骤要处理的特征列"""
        # 列名映射
        if self.config.column_mapping:
            step_features = [self.config.column_mapping.get(f, f) for f in step.features]
        else:
            step_features = step.features
        
        # 如果未指定特征,使用所有特征
        if not step_features:
            step_features = all_features
        
        # 过滤存在的列
        valid_features = [f for f in step_features if f in df.columns]
        
        return valid_features
    
    def _apply_step(
        self,
        df: pd.DataFrame,
        step: ProcessingStep,
        features: List[str],
        target_column: Optional[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        应用单个处理步骤
        
        Args:
            df: 数据框
            step: 处理步骤
            features: 特征列
            target_column: 目标列
            fit: 是否拟合（训练时True，推理时False）
        
        说明:
            • 自动验证必需列是否存在
            • 缺失必需列时自动跳过该步骤
            • 仅处理数值类型的特征列
            • 保留ID列和分组列不处理
        """
        params = step.params or {}
        method = step.method
        
        # ========== 列存在性验证 ==========
        required_columns = self._get_required_columns(method, params)
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(
                f"步骤 '{step.name}' 缺失必需列: {missing_columns}, 跳过此步骤。"
                f"\n  提示: 这些列可能在数据提取时未包含，或被特征筛选过滤。"
                f"\n  建议: 检查数据源配置或调整特征筛选策略。"
            )
            return df
        
        # ========== 特征类型验证 ==========
        # 过滤出只有数值类型的特征（避免处理字符串列导致错误）
        valid_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
        
        if not valid_features and method not in [ProcessMethod.GENERATE_LABELS, ProcessMethod.SIMSTOCK_LABEL_NEUTRALIZE]:
            logger.warning(f"步骤 '{step.name}' 无有效数值特征，跳过")
            return df
        
        # 替换原features为验证后的有效特征
        features = valid_features if valid_features else features
        
        # 先处理无穷值(所有方法之前，除了标签生成)
        if method not in [ProcessMethod.GENERATE_LABELS]:
            df = self.feature_processor.handle_infinite_values(df, features)
        
        # 根据方法调用对应函数
        
        # ========== 标签生成 ==========
        if method == ProcessMethod.GENERATE_LABELS:
            # 标签生成（使用config中的LabelConfig或step的params）
            label_config = None
            if hasattr(self.config, 'label_config') and self.config.label_config.enabled:
                # 从总配置中获取
                label_config = LabelConfig(
                    stock_col=self.config.label_config.stock_col,
                    time_col=self.config.label_config.time_col,
                    price_col=self.config.label_config.price_col,
                    base_price_col=self.config.label_config.base_price_col,
                    label_type=self.config.label_config.label_type,
                    return_periods=self.config.label_config.return_periods,
                    return_method=self.config.label_config.return_method,
                    neutralize=self.config.label_config.neutralize
                )
            elif params:
                # 从step的params中获取
                label_config = LabelConfig(
                    stock_col=params.get('stock_col', 'order_book_id'),
                    time_col=params.get('time_col', 'trade_date'),
                    price_col=params.get('price_col', 'close'),
                    base_price_col=params.get('base_price_col'),
                    label_type=params.get('label_type', 'return'),
                    return_periods=params.get('return_periods', [1, 5, 10]),
                    return_method=params.get('return_method', 'simple'),
                    neutralize=params.get('neutralize', False)
                )
            
            if label_config:
                label_generator = LabelGenerator(label_config)
                label_prefix = params.get('label_prefix', self.config.label_config.label_prefix if hasattr(self.config, 'label_config') else 'label')
                
                logger.info(f"生成标签: 周期={label_config.return_periods}, 前缀={label_prefix}")
                df = label_generator.generate_labels(df, label_name=label_prefix)
                
                # 重命名为 y_ret_* 格式（如果使用默认'label'前缀）
                if label_prefix == 'label':
                    rename_map = {}
                    for period in label_config.return_periods:
                        old_col = f'label_{period}d'
                        new_col = f'{self.config.label_config.label_prefix}_{period}d'
                        if old_col in df.columns:
                            rename_map[old_col] = new_col
                    if rename_map:
                        df = df.rename(columns=rename_map)
                        logger.info(f"重命名标签列: {list(rename_map.keys())} → {list(rename_map.values())}")
            else:
                logger.warning("标签生成配置为空，跳过标签生成")
        
        elif method == ProcessMethod.Z_SCORE:
            df = self.feature_processor.z_score_normalize(
                df, features,
                ddof=params.get('ddof', 1),
                clip_sigma=params.get('clip_sigma'),
                fit=fit,
                normalize_mode=params.get('normalize_mode', 'cross_section')
            )
        
        elif method == ProcessMethod.MINMAX:
            df = self.feature_processor.minmax_normalize(
                df, features,
                output_range=tuple(params.get('output_range', (0, 1))),
                fit=fit,
                normalize_mode=params.get('normalize_mode', 'cross_section')
            )
        
        elif method == ProcessMethod.RANK:
            df = self.feature_processor.rank_normalize(
                df, features,
                output_range=tuple(params.get('output_range', (-1, 1))),
                method=params.get('rank_method', 'average'),
                normalize_mode=params.get('normalize_mode', 'cross_section')
            )
        
        elif method == ProcessMethod.WINSORIZE:
            df = self.feature_processor.winsorize_features(
                df, features,
                limits=tuple(params.get('limits', (0.025, 0.025)))
            )
        
        elif method == ProcessMethod.CLIP:
            df = self.feature_processor.clip_features(
                df, features,
                lower_percentile=params.get('lower_percentile', 1),
                upper_percentile=params.get('upper_percentile', 99)
            )
        
        elif method == ProcessMethod.OLS_NEUTRALIZE:
            df = self.feature_processor.industry_cap_neutralize_ols(
                df, features,
                industry_column=params.get('industry_column', 'industry_name'),
                market_cap_column=params.get('market_cap_column', 'total_mv'),
                min_samples=params.get('min_samples', 10)
            )
        
        elif method == ProcessMethod.MEAN_NEUTRALIZE:
            df = self.feature_processor.industry_cap_neutralize_mean(
                df, features,
                industry_column=params.get('industry_column', 'industry_name'),
                market_cap_column=params.get('market_cap_column', 'total_mv'),
                n_quantiles=params.get('n_quantiles', 5)
            )
        
        elif method == ProcessMethod.SIMSTOCK_LABEL_NEUTRALIZE:
            # SimStock标签中性化
            if not target_column:
                logger.warning("SimStock标签中性化需要target_column,跳过")
            else:
                label_col = params.get('label_column', 'y_ret_1d')
                if label_col not in df.columns:
                    logger.warning(f"标签列 '{label_col}' 不存在，跳过SimStock标签中性化")
                else:
                    df = self.feature_processor.simstock_label_neutralize(
                        df, label_column=label_col,
                        output_column=params.get('output_column', 'alpha_label'),
                        similarity_threshold=params.get('similarity_threshold', 0.7),
                        lookback_window=params.get('lookback_window', 252),
                        min_similar_stocks=params.get('min_similar_stocks', 5),
                        correlation_method=params.get('correlation_method', 'pearson')
                    )
        
        elif method in [ProcessMethod.FILLNA_MEDIAN, ProcessMethod.FILLNA_MEAN, 
                       ProcessMethod.FILLNA_ZERO, ProcessMethod.FILLNA_FORWARD]:
            # 缺失值处理
            fillna_map = {
                ProcessMethod.FILLNA_MEDIAN: 'median',
                ProcessMethod.FILLNA_MEAN: 'mean',
                ProcessMethod.FILLNA_ZERO: 'zero',
                ProcessMethod.FILLNA_FORWARD: 'forward'
            }
            df = self.feature_processor.handle_missing_values(
                df, features,
                method=fillna_map[method],
                industry_column=params.get('industry_column', 'industry_name')
            )
        
        else:
            logger.warning(f"未知处理方法: {method}")
        
        return df
    
    # ========== 状态管理 ==========
    
    def save(self, save_path: str):
        """
        保存预处理器状态
        
        Args:
            save_path: 保存路径(pickle文件)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'config': self.config,
            'feature_processor_params': self.feature_processor.fitted_params,
            'is_fitted': self.is_fitted,
            'feature_columns': self.feature_columns,
            'id_columns': self.id_columns,
            'groupby_columns': self.feature_processor.groupby_columns
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"预处理器已保存至: {save_path}")
    
    @classmethod
    def load(cls, load_path: str) -> 'DataPreprocessor':
        """
        加载预处理器状态
        
        Args:
            load_path: 保存路径
        
        Returns:
            加载的预处理器
        """
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
        # 重建预处理器
        preprocessor = cls(config=state['config'])
        preprocessor.feature_processor.fitted_params = state['feature_processor_params']
        preprocessor.is_fitted = state['is_fitted']
        preprocessor.feature_columns = state['feature_columns']
        preprocessor.id_columns = state['id_columns']
        preprocessor.feature_processor.groupby_columns = state['groupby_columns']
        
        logger.info(f"预处理器已从 {load_path} 加载")
        
        return preprocessor
    
    # ========== 便捷方法 ==========
    
    def get_pipeline_summary(self) -> pd.DataFrame:
        """获取管道步骤摘要"""
        summary_data = []
        for step in self.config.pipeline_steps:
            summary_data.append({
                '步骤名称': step.name,
                '处理方法': step.method.value,
                '特征数': len(step.features) if step.features else 'All',
                '是否启用': '✓' if step.enabled else '✗',
                '参数': str(step.params) if step.params else 'Default'
            })
        
        return pd.DataFrame(summary_data)
    
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """验证数据质量"""
        report = {
            'shape': df.shape,
            'missing_ratio': df[self.feature_columns].isnull().mean().to_dict(),
            'infinite_count': np.isinf(df[self.feature_columns].values).sum(axis=0).tolist(),
            'zero_std_features': []
        }
        
        # 检查标准差为0的特征
        for col in self.feature_columns:
            if df[col].std() == 0:
                report['zero_std_features'].append(col)
        
        return report
    
    def __repr__(self) -> str:
        status = "已拟合" if self.is_fitted else "未拟合"
        n_steps = len([s for s in self.config.pipeline_steps if s.enabled])
        return f"DataPreprocessor(状态={status}, 活跃步骤={n_steps}, 特征数={len(self.feature_columns)})"
