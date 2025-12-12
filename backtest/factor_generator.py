"""
因子生成器模块
基于训练好的模型生成原始因子
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm

from .backtest_config import BacktestConfig


class FactorDataset(Dataset):
    """因子生成专用数据集"""
    
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], 
                 window_size: int = 40):
        """
        初始化数据集
        
        Args:
            df: 原始数据DataFrame
            feature_cols: 特征列名列表
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self.feature_cols = feature_cols
        
        # 确保有ts_code列
        self._ensure_ts_code_column(df)
        
        # 准备数据
        self._prepare_data(df)
    
    def _ensure_ts_code_column(self, df: pd.DataFrame) -> None:
        """确保DataFrame中有ts_code列"""
        if 'ts_code' in df.columns:
            df['ts_code'] = df['ts_code'].astype(str).str.strip()
            return
        
        # 尝试从其他列推断
        candidates = ['order_book_id', 'code', 'symbol', 'stock_code', 'ticker']
        for c in candidates:
            if c in df.columns:
                df['ts_code'] = df[c].astype(str).str.strip()
                return
        
        raise ValueError("未找到股票代码列，请确保数据中包含ts_code列")
    
    def _prepare_data(self, df: pd.DataFrame):
        """准备数据"""
        # 清理数据
        df = df.dropna(subset=self.feature_cols)
        df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        
        self.samples = []
        self.metadata = []
        
        # 按股票分组构建样本
        for ts_code, stock_df in df.groupby('ts_code'):
            stock_df = stock_df.sort_values('trade_date').reset_index(drop=True)
            
            # 构建滑动窗口
            for i in range(len(stock_df) - self.window_size + 1):
                window_data = stock_df.iloc[i:i + self.window_size]
                
                # 提取特征
                X = window_data[self.feature_cols].values.astype(np.float32)
                
                # 检查是否有缺失值
                if np.isnan(X).any():
                    continue
                
                self.samples.append(X)
                
                # 记录元数据
                self.metadata.append({
                    'ts_code': ts_code,
                    'trade_date': window_data.iloc[-1]['trade_date'],
                    'index': i + self.window_size - 1
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        X = self.samples[idx]
        return torch.FloatTensor(X)
    
    def get_metadata(self, idx):
        """获取样本的元数据"""
        return self.metadata[idx]


class FactorGenerator:
    """因子生成器 - 基于VAE模型生成因子"""
    
    def __init__(self, model: torch.nn.Module, config: BacktestConfig):
        """
        初始化因子生成器
        
        Args:
            model: 训练好的模型
            config: 回测配置
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(__name__)
        
        # 将模型移到指定设备并设置为评估模式
        self.model.to(self.device)
        self.model.eval()
    
    def generate_factors(self, 
                        df: pd.DataFrame, 
                        feature_cols: Optional[List[str]] = None,
                        return_metadata: bool = True,
                        mode: str = 'prediction') -> pd.DataFrame:
        """
        生成因子
        
        Args:
            df: 原始数据DataFrame
            feature_cols: 特征列名列表（None时自动识别）
            return_metadata: 是否返回元数据
            mode: 因子提取模式
                - 'prediction': 提取模型的Alpha预测值（y_pred），用于端到端监督学习
                - 'latent': 提取VAE的隐变量均值（mu），用于无监督特征学习
            
        Returns:
            包含因子值的DataFrame
        """
        self.logger.info(f"开始生成因子 (模式: {mode})...")
        
        # 自动识别特征列
        if feature_cols is None:
            feature_cols = self._auto_detect_features(df)
        
        # 创建数据集和数据加载器
        dataset = FactorDataset(df, feature_cols, self.config.window_size)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False
        )
        
        # 生成因子
        factors_list = []
        metadata_list = []
        
        with torch.no_grad():
            for batch_idx, X in enumerate(tqdm(dataloader, desc="生成因子")):
                X = X.to(self.device)
                
                # 提取因子（预测值或隐变量）
                try:
                    # 根据mode选择提取方式
                    if mode == 'prediction':
                        # 研报模式：提取Alpha预测值
                        # VAEModel包装类：self.model.model(X) -> (x_recon, y_pred, mu, logvar, z)
                        # 直接调用网络：self.model(X) -> 同上
                        if hasattr(self.model, 'model'):
                            # VAEModel包装类
                            outputs = self.model.model(X)
                        else:
                            # 直接是VAENet
                            outputs = self.model(X)
                        
                        # 提取y_pred（第2个返回值，索引为1）
                        data = outputs[1]
                        
                    elif mode == 'latent':
                        # 标准VAE模式：提取隐变量
                        if hasattr(self.model, 'encode'):
                            mu, logvar = self.model.encode(X)
                            data = mu
                        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'encode'):
                            mu, logvar = self.model.model.encode(X)
                            data = mu
                        else:
                            # 从forward输出提取mu（第3个返回值，索引为2）
                            if hasattr(self.model, 'model'):
                                outputs = self.model.model(X)
                            else:
                                outputs = self.model(X)
                            data = outputs[2]
                    else:
                        raise ValueError(f"未知的模式: {mode}，仅支持 'prediction' 或 'latent'")
                    
                    # 转换为numpy并处理维度
                    if isinstance(data, torch.Tensor):
                        data = data.cpu().numpy()
                    
                    # 确保维度是 (Batch, N)，如果是预测值 (Batch,) 则升维
                    if data.ndim == 1:
                        data = data[:, np.newaxis]
                    
                    factors_list.append(data)
                    
                    # 收集元数据
                    if return_metadata:
                        start_idx = batch_idx * self.config.batch_size
                        end_idx = min(start_idx + len(X), len(dataset))
                        for i in range(start_idx, end_idx):
                            metadata_list.append(dataset.get_metadata(i))
                
                except Exception as e:
                    self.logger.error(f"批次 {batch_idx} 生成因子失败: {str(e)}")
                    continue
        
        # 合并结果
        if len(factors_list) == 0:
            raise ValueError("未能生成任何因子")
        
        factors_array = np.vstack(factors_list)
        
        # 构建DataFrame - 根据mode动态生成列名
        dim = factors_array.shape[1]
        if mode == 'prediction':
            # 预测模式通常只有1列Alpha预测值
            factor_cols = ['pred_alpha'] if dim == 1 else [f'pred_alpha_{i}' for i in range(dim)]
        else:
            # 隐变量模式
            factor_cols = [f'latent_{i}' for i in range(dim)]
        
        factor_df = pd.DataFrame(factors_array, columns=factor_cols)
        
        # 添加元数据
        if return_metadata and len(metadata_list) > 0:
            metadata_df = pd.DataFrame(metadata_list)
            factor_df = pd.concat([metadata_df, factor_df], axis=1)
        
        self.logger.info(f"因子生成完成: {len(factor_df)} 条记录, {dim} 个因子维度 (模式: {mode})")
        
        # 统计信息
        self._print_factor_statistics(factor_df, factor_cols)
        
        return factor_df
    
    def _auto_detect_features(self, df: pd.DataFrame) -> List[str]:
        """自动识别特征列"""
        exclude_cols = [
            'ts_code', 'trade_date', 'y_processed', 'y_raw', 
            'y_winsorized', 'industry_name', 'order_book_id',
            'code', 'symbol', 'stock_code', 'ticker'
        ]
        
        feature_cols = [
            col for col in df.columns 
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        self.logger.info(f"自动识别特征列: {len(feature_cols)} 个")
        return feature_cols
    
    def _print_factor_statistics(self, factor_df: pd.DataFrame, factor_cols: List[str]):
        """打印因子统计信息"""
        self.logger.info("\n因子统计信息:")
        for col in factor_cols:
            values = factor_df[col].values
            self.logger.info(
                f"  {col}: "
                f"均值={np.mean(values):.4f}, "
                f"标准差={np.std(values):.4f}, "
                f"最小值={np.min(values):.4f}, "
                f"最大值={np.max(values):.4f}, "
                f"缺失值={np.isnan(values).sum()}"
            )
    
    def generate_single_factor(self, 
                              df: pd.DataFrame,
                              feature_cols: Optional[List[str]] = None,
                              aggregation: str = 'mean',
                              mode: str = 'prediction') -> pd.DataFrame:
        """
        生成单一因子（将多维潜在向量聚合为一维）
        
        Args:
            df: 原始数据DataFrame
            feature_cols: 特征列名列表
            aggregation: 聚合方法 ('mean', 'sum', 'first', 'pca')
            mode: 因子提取模式 ('prediction' 或 'latent')
            
        Returns:
            包含单一因子的DataFrame
        """
        # 先生成多维因子
        factor_df = self.generate_factors(df, feature_cols, return_metadata=True, mode=mode)
        
        # 提取因子列 - 根据mode识别
        if mode == 'prediction':
            factor_cols = [col for col in factor_df.columns if col.startswith('pred_')]
        else:
            factor_cols = [col for col in factor_df.columns if col.startswith('latent_')]
        
        # 如果已经是单一因子，直接返回
        if len(factor_cols) == 1:
            result_df = factor_df[['ts_code', 'trade_date']].copy()
            result_df['factor_raw'] = factor_df[factor_cols[0]]
            self.logger.info(f"因子已经是单维度，直接返回")
            return result_df
        
        factor_values = factor_df[factor_cols].values
        
        # 聚合
        if aggregation == 'mean':
            single_factor = np.mean(factor_values, axis=1)
        elif aggregation == 'sum':
            single_factor = np.sum(factor_values, axis=1)
        elif aggregation == 'first':
            single_factor = factor_values[:, 0]
        elif aggregation == 'pca':
            # 简单PCA（取第一主成分）
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            single_factor = pca.fit_transform(factor_values).flatten()
        else:
            raise ValueError(f"未知的聚合方法: {aggregation}")
        
        # 构建结果
        result_df = factor_df[['ts_code', 'trade_date']].copy()
        result_df['factor_raw'] = single_factor
        
        self.logger.info(f"生成单一因子完成 (聚合方法: {aggregation})")
        
        return result_df
    
    def batch_generate_factors(self, 
                               df_dict: Dict[str, pd.DataFrame],
                               feature_cols: Optional[List[str]] = None,
                               mode: str = 'prediction') -> Dict[str, pd.DataFrame]:
        """
        批量生成因子（用于多个时间段或多个数据集）
        
        Args:
            df_dict: 数据集字典 {名称: DataFrame}
            feature_cols: 特征列名列表
            mode: 因子提取模式 ('prediction' 或 'latent')
            
        Returns:
            因子DataFrame字典
        """
        result_dict = {}
        
        for name, df in df_dict.items():
            self.logger.info(f"处理数据集: {name}")
            try:
                factor_df = self.generate_factors(df, feature_cols, mode=mode)
                result_dict[name] = factor_df
            except Exception as e:
                self.logger.error(f"数据集 {name} 生成因子失败: {str(e)}")
                continue
        
        return result_dict
    
    def save_factors(self, factor_df: pd.DataFrame, output_path: str):
        """保存因子到文件"""
        if output_path.endswith('.parquet'):
            factor_df.to_parquet(output_path, index=False)
        elif output_path.endswith('.csv'):
            factor_df.to_csv(output_path, index=False)
        else:
            raise ValueError("仅支持 .parquet 和 .csv 格式")
        
        self.logger.info(f"因子已保存到: {output_path}")
    
    @staticmethod
    def load_factors(input_path: str) -> pd.DataFrame:
        """从文件加载因子"""
        if input_path.endswith('.parquet'):
            return pd.read_parquet(input_path)
        elif input_path.endswith('.csv'):
            return pd.read_csv(input_path)
        else:
            raise ValueError("仅支持 .parquet 和 .csv 格式")
