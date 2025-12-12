"""
FeatureEngineer - 特征工程师

提供特征选择、特征缓存和自动筛选功能
"""

import os
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging
import pickle
from pathlib import Path
from .config import DataConfig


class FeatureEngineer:
    """特征工程师 - 管理特征选择和处理"""
    
    def __init__(self, config: DataConfig):
        """
        Args:
            config: DataConfig配置对象
        """
        self.config = config
        self.logger = self._setup_logger()
        self.feature_cols: Optional[List[str]] = None
        self.feature_stats: Optional[Dict[str, Any]] = None
    
    def _setup_logger(self) -> logging.Logger:
        """配置日志"""
        logger = logging.getLogger('FeatureEngineer')
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def select_features(self, df: pd.DataFrame, 
                       auto_select: bool = True) -> List[str]:
        """
        选择特征列
        
        Args:
            df: 数据DataFrame
            auto_select: 是否自动选择特征
            
        Returns:
            特征列列表
        """
        # 如果配置中已指定特征列
        if self.config.feature_cols is not None:
            self.feature_cols = self.config.feature_cols
            self.logger.info(f"✅ 使用配置的特征列: {len(self.feature_cols)} 列")
            return self.feature_cols
        
        if not auto_select:
            raise ValueError("未指定特征列且auto_select=False")
        
        self.logger.info("🔍 自动检测特征列...")
        
        # 排除列（包括配置的排除列 + 系统列）
        exclude = set(self.config.exclude_cols)
        
        # 【修复】强制排除系统列（stock_col, time_col, label_col）
        system_cols = {
            self.config.stock_col,
            self.config.time_col,
            self.config.label_col
        }
        exclude.update(system_cols)
        
        # 选择数值型列
        feature_cols = [
            col for col in df.columns
            if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        self.feature_cols = feature_cols
        self.logger.info(f"✅ 自动选择特征列: {len(feature_cols)} 列")
        
        return feature_cols
    
    def compute_feature_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算特征统计信息
        
        Args:
            df: 数据DataFrame
            
        Returns:
            特征统计字典
        """
        if self.feature_cols is None:
            self.select_features(df)
        
        self.logger.info("📊 计算特征统计信息...")
        
        stats = {}
        for col in self.feature_cols:
            col_data = df[col].dropna()
            
            stats[col] = {
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'q25': float(col_data.quantile(0.25)),
                'q50': float(col_data.quantile(0.50)),
                'q75': float(col_data.quantile(0.75)),
                'missing_ratio': float(df[col].isnull().sum() / len(df)),
                'unique_ratio': float(df[col].nunique() / len(df)),
            }
        
        self.feature_stats = stats
        return stats
    
    def filter_features(self, df: pd.DataFrame,
                       min_variance: float = 1e-6,
                       max_missing_ratio: float = 0.5,
                       max_correlation: float = 0.95) -> List[str]:
        """
        过滤低质量特征
        
        Args:
            df: 数据DataFrame
            min_variance: 最小方差阈值
            max_missing_ratio: 最大缺失率
            max_correlation: 最大相关性阈值
            
        Returns:
            过滤后的特征列列表
        """
        if self.feature_cols is None:
            self.select_features(df)
        
        self.logger.info("🔧 过滤低质量特征...")
        
        filtered_cols = self.feature_cols.copy()
        remove_reasons = {}
        
        # 1. 过滤低方差特征
        for col in self.feature_cols:
            if df[col].var() < min_variance:
                if col in filtered_cols:
                    filtered_cols.remove(col)
                    remove_reasons[col] = 'low_variance'
        
        # 2. 过滤高缺失率特征
        for col in self.feature_cols:
            missing_ratio = df[col].isnull().sum() / len(df)
            if missing_ratio > max_missing_ratio:
                if col in filtered_cols:
                    filtered_cols.remove(col)
                    remove_reasons[col] = 'high_missing'
        
        # 3. 过滤高相关性特征
        if len(filtered_cols) > 1:
            corr_matrix = df[filtered_cols].corr().abs()
            
            # 找到高相关性特征对
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > max_correlation:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
            
            # 移除高相关性特征（保留方差较大的）
            for col1, col2, corr_val in high_corr_pairs:
                if col1 in filtered_cols and col2 in filtered_cols:
                    var1, var2 = df[col1].var(), df[col2].var()
                    remove_col = col1 if var1 < var2 else col2
                    filtered_cols.remove(remove_col)
                    remove_reasons[remove_col] = f'high_correlation_with_{col2 if remove_col==col1 else col1}'
        
        # 输出过滤结果
        removed_count = len(self.feature_cols) - len(filtered_cols)
        self.logger.info(f"   移除 {removed_count} 个特征:")
        self.logger.info(f"   - 低方差: {sum(1 for v in remove_reasons.values() if v=='low_variance')}")
        self.logger.info(f"   - 高缺失: {sum(1 for v in remove_reasons.values() if v=='high_missing')}")
        self.logger.info(f"   - 高相关: {sum(1 for v in remove_reasons.values() if 'correlation' in v)}")
        self.logger.info(f"✅ 保留 {len(filtered_cols)} 个特征")
        
        self.feature_cols = filtered_cols
        return filtered_cols
    
    def save_feature_info(self, save_dir: Optional[str] = None):
        """保存特征信息"""
        if save_dir is None:
            save_dir = self.config.output_dir
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # 保存特征列表
        feature_list_path = os.path.join(save_dir, 'feature_columns.txt')
        with open(feature_list_path, 'w') as f:
            f.write('\n'.join(self.feature_cols))
        
        # 保存特征统计
        if self.feature_stats:
            stats_path = os.path.join(save_dir, 'feature_stats.pkl')
            with open(stats_path, 'wb') as f:
                pickle.dump(self.feature_stats, f)
        
        self.logger.info(f"💾 特征信息已保存到: {save_dir}")
    
    def load_feature_info(self, load_dir: Optional[str] = None):
        """加载特征信息"""
        if load_dir is None:
            load_dir = self.config.output_dir
        
        # 加载特征列表
        feature_list_path = os.path.join(load_dir, 'feature_columns.txt')
        if os.path.exists(feature_list_path):
            with open(feature_list_path, 'r') as f:
                self.feature_cols = [line.strip() for line in f]
        
        # 加载特征统计
        stats_path = os.path.join(load_dir, 'feature_stats.pkl')
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                self.feature_stats = pickle.load(f)
        
        self.logger.info(f"📁 特征信息已加载: {len(self.feature_cols)} 列")


if __name__ == '__main__':
    # 测试特征工程师
    from config import DataConfig
    
    print("=" * 80)
    print("FeatureEngineer 测试")
    print("=" * 80)
    
    # 创建配置
    config = DataConfig()
    
    # 创建特征工程师
    engineer = FeatureEngineer(config)
    
    # 模拟数据
    np.random.seed(42)
    df = pd.DataFrame({
        'ts_code': ['000001.SZ'] * 100,
        'trade_date': pd.date_range('2020-01-01', periods=100),
        'y_processed': np.random.randn(100),
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100) * 0.0001,  # 低方差
        'feature4': [np.nan] * 60 + list(np.random.randn(40)),  # 高缺失
    })
    
    # 测试特征选择
    features = engineer.select_features(df)
    print(f"\n1. 自动选择特征: {len(features)} 列")
    print(f"   {features}")
    
    # 测试特征过滤
    filtered = engineer.filter_features(df, min_variance=1e-4, max_missing_ratio=0.5)
    print(f"\n2. 过滤后特征: {len(filtered)} 列")
    print(f"   {filtered}")
    
    # 测试统计计算
    stats = engineer.compute_feature_stats(df)
    print(f"\n3. 特征统计: {len(stats)} 列")
    
    print("\n✅ 特征工程师测试完成")
