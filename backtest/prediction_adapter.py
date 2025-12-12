"""
prediction_adapter.py - 预测结果适配器

将 RollingWindowTrainer 的多因子预测结果转换为 backtest 标准格式。
支持多因子集成策略：简单平均、IC加权、最佳因子选择等。

Usage:
    from quantclassic.backtest.prediction_adapter import PredictionAdapter
    
    adapter = PredictionAdapter(config)
    backtest_df = adapter.adapt(
        rolling_predictions,
        stock_col='order_book_id',
        time_col='trade_date',
        label_col='y_ret_10d'
    )
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.stats import spearmanr
import logging
from pathlib import Path

from .backtest_config import BacktestConfig


class PredictionAdapter:
    """
    预测结果适配器
    
    将 RollingWindowTrainer.predict_all_windows() 的输出转换为 backtest 的标准格式。
    支持多因子集成与单因子模式。
    
    Features:
        - 自动识别多因子/单因子输出
        - 多因子集成策略：mean, ic_weighted, best, custom_weights
        - 列名映射与格式转换
        - 数据质量检查与修复
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        初始化适配器
        
        Args:
            config: 回测配置（可选）
        """
        self.config = config or BacktestConfig()
        self.logger = logging.getLogger(__name__)
        
        # 多因子集成结果缓存
        self._factor_ics: Dict[str, float] = {}
        self._ensemble_weights: Dict[str, float] = {}
    
    def adapt(
        self,
        predictions_df: pd.DataFrame,
        stock_col: str = 'order_book_id',
        time_col: str = 'trade_date',
        label_col: str = 'y_ret_10d',
        ensemble_method: str = 'mean',
        custom_weights: Optional[Dict[str, float]] = None,
        output_factor_col: str = 'factor_value'
    ) -> pd.DataFrame:
        """
        适配预测结果为 backtest 标准格式
        
        Args:
            predictions_df: RollingWindowTrainer 的预测结果 DataFrame
            stock_col: 原始股票列名
            time_col: 原始时间列名
            label_col: 原始标签列名
            ensemble_method: 多因子集成方法
                - 'mean': 简单平均（默认）
                - 'ic_weighted': IC 加权平均
                - 'best': 选择 IC 最高的因子
                - 'custom': 使用自定义权重
            custom_weights: 自定义权重字典（仅当 ensemble_method='custom' 时使用）
            output_factor_col: 输出因子列名
            
        Returns:
            适配后的 DataFrame，包含标准列名：
            - ts_code: 股票代码
            - trade_date: 交易日期
            - factor_raw / factor_value: 原始因子值
            - y_true / y_processed: 真实标签
            - 以及可选的 pred_factor_* 列
        """
        self.logger.info("=" * 60)
        self.logger.info("🔄 开始适配预测结果")
        self.logger.info("=" * 60)
        
        df = predictions_df.copy()
        
        # 1. 检测因子列
        factor_cols = self._detect_factor_cols(df)
        self.logger.info(f"  检测到因子列: {len(factor_cols)} 个")
        
        # 2. 多因子集成
        if len(factor_cols) > 1:
            self.logger.info(f"  多因子模式，使用 {ensemble_method} 集成")
            df = self._ensemble_factors(
                df, factor_cols, label_col, time_col,
                method=ensemble_method,
                custom_weights=custom_weights,
                output_col=output_factor_col
            )
        elif len(factor_cols) == 1:
            self.logger.info(f"  单因子模式")
            df[output_factor_col] = df[factor_cols[0]]
        else:
            # 尝试查找 pred_alpha
            if 'pred_alpha' in df.columns:
                df[output_factor_col] = df['pred_alpha']
            else:
                raise ValueError("未找到因子列（pred_factor_* 或 pred_alpha）")
        
        # 3. 列名映射
        df = self._rename_columns(df, stock_col, time_col, label_col, output_factor_col)
        
        # 4. 数据质量检查
        df = self._quality_check(df)
        
        # 5. 打印汇总
        self._print_summary(df)
        
        return df
    
    def _detect_factor_cols(self, df: pd.DataFrame) -> List[str]:
        """检测多因子列"""
        return [col for col in df.columns if col.startswith('pred_factor_')]
    
    def _ensemble_factors(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        label_col: str,
        time_col: str,
        method: str = 'mean',
        custom_weights: Optional[Dict[str, float]] = None,
        output_col: str = 'factor_value'
    ) -> pd.DataFrame:
        """
        多因子集成
        
        Args:
            df: 数据
            factor_cols: 因子列名列表
            label_col: 标签列名
            time_col: 时间列名（用于计算 IC）
            method: 集成方法
            custom_weights: 自定义权重
            output_col: 输出列名
            
        Returns:
            添加集成因子后的 DataFrame
        """
        if method == 'mean':
            # 简单平均
            df[output_col] = df[factor_cols].mean(axis=1)
            self._ensemble_weights = {col: 1.0/len(factor_cols) for col in factor_cols}
            
        elif method == 'ic_weighted':
            # IC 加权平均
            weights = self._calculate_ic_weights(df, factor_cols, label_col, time_col)
            df[output_col] = (df[factor_cols].values * weights).sum(axis=1)
            self._ensemble_weights = dict(zip(factor_cols, weights))
            
        elif method == 'best':
            # 选择最佳因子
            ics = self._calculate_factor_ics(df, factor_cols, label_col, time_col)
            best_col = max(ics, key=ics.get)
            df[output_col] = df[best_col]
            self._ensemble_weights = {best_col: 1.0}
            self.logger.info(f"    最佳因子: {best_col} (IC={ics[best_col]:.4f})")
            
        elif method == 'custom':
            if custom_weights is None:
                raise ValueError("custom 方法需要提供 custom_weights")
            weights = np.array([custom_weights.get(col, 0) for col in factor_cols])
            # 防御性检查：如果权重之和为 0，回退到等权重
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                self.logger.warning("    自定义权重之和为 0，回退到等权重")
                weights = np.ones(len(factor_cols)) / len(factor_cols)
            df[output_col] = (df[factor_cols].values * weights).sum(axis=1)
            self._ensemble_weights = dict(zip(factor_cols, weights))
            
        else:
            raise ValueError(f"未知的集成方法: {method}")
        
        # 同时保留各因子的 IC 加权版本（供后续分析）
        if method != 'mean':
            df[f'{output_col}_mean'] = df[factor_cols].mean(axis=1)
        
        return df
    
    def _calculate_factor_ics(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        label_col: str,
        time_col: str
    ) -> Dict[str, float]:
        """计算各因子的 IC"""
        ics = {}
        
        for col in factor_cols:
            ic_values = []
            for date, group in df.groupby(time_col):
                if len(group) < 10:
                    continue
                pred = group[col].values
                label = group[label_col].values
                valid = ~(np.isnan(pred) | np.isnan(label))
                if valid.sum() < 10:
                    continue
                ic, _ = spearmanr(pred[valid], label[valid])
                if not np.isnan(ic):
                    ic_values.append(ic)
            
            ics[col] = np.mean(ic_values) if ic_values else 0.0
            self.logger.info(f"    {col}: IC = {ics[col]:.4f}")
        
        self._factor_ics = ics
        return ics
    
    def _calculate_ic_weights(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        label_col: str,
        time_col: str
    ) -> np.ndarray:
        """计算 IC 加权权重"""
        ics = self._calculate_factor_ics(df, factor_cols, label_col, time_col)
        
        # 只使用正 IC 作为权重
        weights = np.array([max(ics[col], 0) for col in factor_cols])
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(factor_cols)) / len(factor_cols)
        
        self.logger.info(f"    IC权重: {dict(zip(factor_cols, weights.round(3)))}")
        return weights
    
    def _rename_columns(
        self,
        df: pd.DataFrame,
        stock_col: str,
        time_col: str,
        label_col: str,
        factor_col: str
    ) -> pd.DataFrame:
        """列名映射到 backtest 标准格式"""
        rename_map = {}
        
        # 股票列
        if stock_col in df.columns and stock_col != 'ts_code':
            rename_map[stock_col] = 'ts_code'
        
        # 时间列
        if time_col in df.columns and time_col != 'trade_date':
            rename_map[time_col] = 'trade_date'
        
        # 标签列
        if label_col in df.columns:
            rename_map[label_col] = 'y_true'
            df['y_processed'] = df[label_col]  # 保留两个版本
        
        # 因子列
        if factor_col in df.columns:
            df['factor_raw'] = df[factor_col]
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        return df
    
    def _quality_check(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据质量检查与修复"""
        original_len = len(df)
        
        # 确保日期格式
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 检查必需列
        required = ['ts_code', 'trade_date', 'factor_raw']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"缺少必需列: {missing}")
        
        # 移除因子缺失的行
        df = df.dropna(subset=['factor_raw'])
        
        dropped = original_len - len(df)
        if dropped > 0:
            self.logger.warning(f"  移除 {dropped} 行缺失数据 ({dropped/original_len:.1%})")
        
        return df
    
    def _print_summary(self, df: pd.DataFrame):
        """打印适配结果汇总"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("✅ 适配完成")
        self.logger.info("=" * 60)
        self.logger.info(f"  数据形状: {df.shape}")
        self.logger.info(f"  时间范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
        self.logger.info(f"  股票数量: {df['ts_code'].nunique()}")
        self.logger.info(f"  输出列: {list(df.columns)}")
    
    def get_factor_ics(self) -> Dict[str, float]:
        """获取各因子的 IC"""
        return self._factor_ics
    
    def get_ensemble_weights(self) -> Dict[str, float]:
        """获取集成权重"""
        return self._ensemble_weights
    
    def save_adapted_data(
        self,
        df: pd.DataFrame,
        output_path: str,
        format: str = 'parquet'
    ):
        """保存适配后的数据"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'parquet':
            df.to_parquet(output_path, index=False)
        elif format == 'csv':
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        self.logger.info(f"  💾 已保存: {output_path}")


# ==================== 便捷函数 ====================

def adapt_predictions(
    predictions_df: pd.DataFrame,
    stock_col: str = 'order_book_id',
    time_col: str = 'trade_date',
    label_col: str = 'y_ret_10d',
    ensemble_method: str = 'mean'
) -> pd.DataFrame:
    """
    便捷函数：快速适配预测结果
    
    Example:
        from quantclassic.backtest.prediction_adapter import adapt_predictions
        
        backtest_df = adapt_predictions(
            rolling_predictions,
            stock_col='order_book_id',
            time_col='trade_date',
            label_col='y_ret_10d'
        )
    """
    adapter = PredictionAdapter()
    return adapter.adapt(
        predictions_df,
        stock_col=stock_col,
        time_col=time_col,
        label_col=label_col,
        ensemble_method=ensemble_method
    )
