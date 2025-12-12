"""
DataLoader - 数据加载引擎

支持多种数据格式的加载、验证和内存优化
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
from .config import DataConfig


class DataLoaderEngine:
    """数据加载引擎 - 支持多种格式和优化策略"""
    
    def __init__(self, config: DataConfig):
        """
        Args:
            config: DataConfig配置对象
        """
        self.config = config
        self.logger = self._setup_logger()
        self._data_cache: Optional[pd.DataFrame] = None
    
    def _setup_logger(self) -> logging.Logger:
        """配置日志"""
        logger = logging.getLogger('DataLoader')
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_data(self, file_path: Optional[str] = None, 
                  use_cache: bool = True) -> pd.DataFrame:
        """
        加载数据主方法
        
        Args:
            file_path: 数据文件路径（None则使用config中的路径）
            use_cache: 是否使用缓存
            
        Returns:
            加载的DataFrame
        """
        # 使用缓存
        if use_cache and self._data_cache is not None:
            self.logger.info("📦 使用缓存数据")
            return self._data_cache.copy()
        
        # 确定文件路径
        if file_path is None:
            file_path = self.config.data_path
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        self.logger.info(f"📁 加载数据: {file_path}")
        
        # 根据格式选择加载方法
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.parquet':
            df = self._load_parquet(file_path)
        elif file_ext == '.csv':
            df = self._load_csv(file_path)
        elif file_ext in ['.h5', '.hdf5']:
            df = self._load_hdf5(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
        
        # 数据类型优化
        if self.config.use_dtype_optimization:
            df = self._optimize_dtypes(df)
        
        # 基础验证
        self._validate_data(df)
        
        # 缓存数据
        if use_cache:
            self._data_cache = df.copy()
        
        self.logger.info(f"✅ 数据加载完成: {len(df):,} 行, {len(df.columns)} 列")
        
        return df
    
    def _load_parquet(self, file_path: str) -> pd.DataFrame:
        """加载Parquet文件"""
        if self.config.chunk_size:
            # 分块加载
            chunks = []
            for chunk in pd.read_parquet(file_path, chunksize=self.config.chunk_size):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_parquet(file_path)
        
        return df
    
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """加载CSV文件"""
        if self.config.chunk_size:
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=self.config.chunk_size):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(file_path)
        
        return df
    
    def _load_hdf5(self, file_path: str) -> pd.DataFrame:
        """加载HDF5文件"""
        df = pd.read_hdf(file_path)
        return df
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化数据类型以节省内存"""
        self.logger.info("🔧 优化数据类型...")
        
        original_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            # 数值类型转换为float32
            if col_type == 'float64':
                df[col] = df[col].astype('float32')
            
            # 整数类型优化
            elif col_type == 'int64':
                col_min = df[col].min()
                col_max = df[col].max()
                
                if col_min >= 0:
                    if col_max < 255:
                        df[col] = df[col].astype('uint8')
                    elif col_max < 65535:
                        df[col] = df[col].astype('uint16')
                    elif col_max < 4294967295:
                        df[col] = df[col].astype('uint32')
                else:
                    if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype('int8')
                    elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype('int16')
                    elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype('int32')
            
            # 字符串类型转换为category
            elif col_type == 'object':
                num_unique = df[col].nunique()
                num_total = len(df[col])
                if num_unique / num_total < 0.5:  # 如果唯一值少于50%
                    df[col] = df[col].astype('category')
        
        optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
        reduction = (1 - optimized_memory / original_memory) * 100
        
        self.logger.info(f"   内存优化: {original_memory:.2f}MB → {optimized_memory:.2f}MB "
                        f"(减少 {reduction:.1f}%)")
        
        return df
    
    def _validate_data(self, df: pd.DataFrame):
        """基础数据验证"""
        # 检查是否为空
        if df.empty:
            raise ValueError("数据为空")
        
        # 检查必需列
        required_cols = [self.config.stock_col, self.config.time_col, self.config.label_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"缺少必需列: {missing_cols}")
        
        # 检查缺失值
        na_counts = df.isnull().sum()
        na_ratios = na_counts / len(df)
        
        high_na_cols = na_ratios[na_ratios > self.config.max_na_ratio].index.tolist()
        if high_na_cols:
            self.logger.warning(f"⚠️  以下列缺失值超过{self.config.max_na_ratio*100}%: {high_na_cols}")
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取数据信息摘要"""
        info = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'num_stocks': df[self.config.stock_col].nunique() if self.config.stock_col in df.columns else 0,
            'date_range': (
                df[self.config.time_col].min(), 
                df[self.config.time_col].max()
            ) if self.config.time_col in df.columns else None,
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.value_counts().to_dict(),
        }
        
        return info
    
    def print_data_summary(self, df: pd.DataFrame):
        """打印数据摘要"""
        info = self.get_data_info(df)
        
        print("\n" + "=" * 80)
        print("📊 数据摘要")
        print("=" * 80)
        print(f"形状: {info['shape'][0]:,} 行 × {info['shape'][1]} 列")
        print(f"内存占用: {info['memory_usage_mb']:.2f} MB")
        
        if info['num_stocks'] > 0:
            print(f"股票数量: {info['num_stocks']:,}")
        
        if info['date_range']:
            print(f"时间范围: {info['date_range'][0]} ~ {info['date_range'][1]}")
        
        print(f"\n数据类型分布:")
        for dtype, count in info['dtypes'].items():
            print(f"  {dtype}: {count} 列")
        
        # 缺失值统计
        missing = {k: v for k, v in info['missing_values'].items() if v > 0}
        if missing:
            print(f"\n缺失值 (前10列):")
            for col, count in list(missing.items())[:10]:
                ratio = count / info['shape'][0] * 100
                print(f"  {col}: {count} ({ratio:.2f}%)")
        
        print("=" * 80)
    
    def clear_cache(self):
        """清除缓存"""
        self._data_cache = None
        self.logger.info("🗑️  缓存已清除")


if __name__ == '__main__':
    # 测试数据加载器
    from config import DataConfig
    
    print("=" * 80)
    print("DataLoader 测试")
    print("=" * 80)
    
    # 创建配置
    config = DataConfig(
        base_dir='rq_data_parquet',
        data_file='train_data_final.parquet',
        use_dtype_optimization=True
    )
    
    # 创建加载器
    loader = DataLoaderEngine(config)
    
    # 加载数据
    try:
        df = loader.load_data()
        loader.print_data_summary(df)
        
        print("\n✅ 数据加载器测试完成")
    except FileNotFoundError:
        print("\n⚠️  测试数据文件不存在，跳过实际加载测试")
        print("✅ 数据加载器类定义完成")
