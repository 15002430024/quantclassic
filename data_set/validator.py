"""
DataValidator - 数据验证器

提供数据质量监控和验证功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from .config import DataConfig


@dataclass
class ValidationReport:
    """数据验证报告"""
    is_valid: bool
    warnings: List[str]
    errors: List[str]
    stats: Dict[str, Any]
    
    def print_report(self):
        """打印验证报告"""
        print("\n" + "=" * 80)
        print("📋 数据验证报告")
        print("=" * 80)
        
        print(f"\n状态: {'✅ 通过' if self.is_valid else '❌ 失败'}")
        
        if self.errors:
            print(f"\n错误 ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. ❌ {error}")
        
        if self.warnings:
            print(f"\n警告 ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. ⚠️  {warning}")
        
        if self.stats:
            print(f"\n统计信息:")
            for key, value in self.stats.items():
                print(f"  {key}: {value}")
        
        print("=" * 80)


class DataValidator:
    """数据验证器 - 数据质量监控"""
    
    def __init__(self, config: DataConfig):
        """
        Args:
            config: DataConfig配置对象
        """
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """配置日志"""
        logger = logging.getLogger('DataValidator')
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def validate(self, df: pd.DataFrame, 
                feature_cols: Optional[List[str]] = None) -> ValidationReport:
        """
        全面验证数据质量
        
        Args:
            df: 待验证的数据
            feature_cols: 特征列列表
            
        Returns:
            ValidationReport对象
        """
        self.logger.info("🔍 开始数据验证...")
        
        errors = []
        warnings = []
        stats = {}
        
        # 1. 基础验证
        errors.extend(self._validate_basic(df))
        
        # 2. 缺失值检查
        missing_warnings = self._check_missing_values(df, feature_cols)
        warnings.extend(missing_warnings)
        
        # 3. 时序连续性检查
        continuity_warnings = self._check_time_continuity(df)
        warnings.extend(continuity_warnings)
        
        # 4. 异常值检测
        if self.config.detect_outliers:
            outlier_stats = self._detect_outliers(df, feature_cols)
            stats['outliers'] = outlier_stats
        
        # 5. 股票样本数检查
        stock_warnings = self._check_stock_samples(df)
        warnings.extend(stock_warnings)
        
        # 6. 数据泄漏检测
        leakage_warnings = self._check_data_leakage(df)
        warnings.extend(leakage_warnings)
        
        # 生成统计
        stats.update(self._compute_validation_stats(df, feature_cols))
        
        # 判断是否通过验证
        is_valid = len(errors) == 0
        
        self.logger.info(f"✅ 验证完成: {len(errors)} 错误, {len(warnings)} 警告")
        
        return ValidationReport(
            is_valid=is_valid,
            warnings=warnings,
            errors=errors,
            stats=stats
        )
    
    def _validate_basic(self, df: pd.DataFrame) -> List[str]:
        """基础验证"""
        errors = []
        
        # 检查是否为空
        if df.empty:
            errors.append("数据为空")
            return errors
        
        # 检查必需列
        required_cols = [
            self.config.stock_col,
            self.config.time_col,
            self.config.label_col
        ]
        
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"缺少必需列: {col}")
        
        return errors
    
    def _check_missing_values(self, df: pd.DataFrame, 
                             feature_cols: Optional[List[str]]) -> List[str]:
        """检查缺失值"""
        warnings = []
        
        if feature_cols is None:
            check_cols = df.columns
        else:
            check_cols = feature_cols
        
        for col in check_cols:
            if col not in df.columns:
                continue
            
            na_count = df[col].isnull().sum()
            na_ratio = na_count / len(df)
            
            if na_ratio > self.config.max_na_ratio:
                warnings.append(
                    f"列 '{col}' 缺失值过高: {na_ratio*100:.2f}% ({na_count}/{len(df)})"
                )
        
        return warnings
    
    def _check_time_continuity(self, df: pd.DataFrame) -> List[str]:
        """检查时序连续性"""
        warnings = []
        
        if self.config.time_col not in df.columns:
            return warnings
        
        # 按股票分组检查
        for stock_code, stock_df in df.groupby(self.config.stock_col):
            dates = pd.to_datetime(stock_df[self.config.time_col]).sort_values()
            
            # 检查是否有重复日期
            duplicates = dates.duplicated().sum()
            if duplicates > 0:
                warnings.append(
                    f"股票 {stock_code} 存在 {duplicates} 个重复日期"
                )
            
            # 检查时间间隔（简单检查是否单调递增）
            if not dates.is_monotonic_increasing:
                warnings.append(
                    f"股票 {stock_code} 时间序列不是单调递增"
                )
        
        return warnings
    
    def _detect_outliers(self, df: pd.DataFrame, 
                        feature_cols: Optional[List[str]]) -> Dict[str, int]:
        """检测异常值"""
        outlier_stats = {}
        
        if feature_cols is None:
            return outlier_stats
        
        threshold = self.config.outlier_std_threshold
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            mean = df[col].mean()
            std = df[col].std()
            
            outliers = np.abs((df[col] - mean) / std) > threshold
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                outlier_stats[col] = int(outlier_count)
        
        return outlier_stats
    
    def _check_stock_samples(self, df: pd.DataFrame) -> List[str]:
        """检查每只股票的样本数"""
        warnings = []
        
        if self.config.stock_col not in df.columns:
            return warnings
        
        stock_counts = df[self.config.stock_col].value_counts()
        
        low_sample_stocks = stock_counts[
            stock_counts < self.config.min_samples_per_stock
        ]
        
        if len(low_sample_stocks) > 0:
            warnings.append(
                f"{len(low_sample_stocks)} 只股票样本数少于 {self.config.min_samples_per_stock}"
            )
        
        return warnings
    
    def _check_data_leakage(self, df: pd.DataFrame) -> List[str]:
        """检查数据泄漏（简单检查）"""
        warnings = []
        
        # 检查是否有未来数据（时间戳）
        # 这里只做简单示例，实际应用中需要更复杂的检查
        
        return warnings
    
    def _compute_validation_stats(self, df: pd.DataFrame, 
                                  feature_cols: Optional[List[str]]) -> Dict[str, Any]:
        """计算验证统计信息"""
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
        }
        
        if self.config.stock_col in df.columns:
            stats['num_stocks'] = df[self.config.stock_col].nunique()
        
        if self.config.time_col in df.columns:
            stats['date_range'] = (
                str(df[self.config.time_col].min()),
                str(df[self.config.time_col].max())
            )
        
        if feature_cols:
            stats['num_features'] = len(feature_cols)
            
            # 特征类型分布
            numeric_features = sum(
                1 for col in feature_cols 
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
            )
            stats['numeric_features'] = numeric_features
        
        return stats


if __name__ == '__main__':
    # 测试数据验证器
    from config import DataConfig
    
    print("=" * 80)
    print("DataValidator 测试")
    print("=" * 80)
    
    # 创建配置
    config = DataConfig(
        max_na_ratio=0.3,
        min_samples_per_stock=50,
        detect_outliers=True,
        outlier_std_threshold=5.0
    )
    
    # 创建验证器
    validator = DataValidator(config)
    
    # 创建测试数据
    np.random.seed(42)
    df = pd.DataFrame({
        'ts_code': ['000001.SZ'] * 100 + ['000002.SZ'] * 100,
        'trade_date': pd.date_range('2020-01-01', periods=100).tolist() * 2,
        'y_processed': np.random.randn(200),
        'feature1': np.random.randn(200),
        'feature2': [np.nan] * 50 + list(np.random.randn(150)),  # 25% 缺失
        'feature3': np.concatenate([np.random.randn(190), [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]]),  # 异常值
    })
    
    # 执行验证
    report = validator.validate(df, feature_cols=['feature1', 'feature2', 'feature3'])
    report.print_report()
    
    print("\n✅ 数据验证器测试完成")
