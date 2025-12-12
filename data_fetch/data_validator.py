"""
数据验证器模块
负责数据质量检查与验证
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from .config_manager import ConfigManager


logger = logging.getLogger(__name__)


class DataValidator:
    """数据验证器"""
    
    def __init__(self, config: ConfigManager):
        """
        初始化数据验证器
        
        Args:
            config: 配置管理器实例
        """
        self.config = config
        self.validation_results = {}
    
    def validate_data_integrity(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        验证数据完整性
        
        Args:
            df: 待验证的DataFrame
            
        Returns:
            验证结果字典
        """
        logger.info("=" * 80)
        logger.info("数据完整性检查")
        logger.info("=" * 80)
        
        results = {
            'total_rows': len(df),
            'total_stocks': df['order_book_id'].nunique() if 'order_book_id' in df.columns else 0,
            'date_range': None,
            'missing_values': {},
            'duplicate_rows': 0,
            'data_types': {},
            'status': 'PASSED'
        }
        
        # 1. 检查日期范围
        if 'trade_date' in df.columns:
            min_date = df['trade_date'].min()
            max_date = df['trade_date'].max()
            results['date_range'] = (str(min_date.date()), str(max_date.date()))
            logger.info(f"✓ 日期范围: {results['date_range'][0]} ~ {results['date_range'][1]}")
        
        # 2. 检查缺失值
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        results['missing_values'] = {
            col: {'count': int(missing[col]), 'percentage': float(missing_pct[col])}
            for col in df.columns if missing[col] > 0
        }
        
        if results['missing_values']:
            logger.warning(f"⚠ 发现缺失值:")
            for col, info in results['missing_values'].items():
                logger.warning(f"  - {col}: {info['count']} ({info['percentage']}%)")
        else:
            logger.info("✓ 无缺失值")
        
        # 3. 检查重复行
        duplicates = df.duplicated(subset=['order_book_id', 'trade_date']).sum()
        results['duplicate_rows'] = int(duplicates)
        
        if duplicates > 0:
            logger.warning(f"⚠ 发现 {duplicates} 行重复数据")
            results['status'] = 'WARNING'
        else:
            logger.info("✓ 无重复数据")
        
        # 4. 检查数据类型
        results['data_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        logger.info(f"\n数据概况:")
        logger.info(f"  - 总行数: {results['total_rows']:,}")
        logger.info(f"  - 股票数: {results['total_stocks']:,}")
        logger.info(f"  - 特征数: {len(df.columns)}")
        
        self.validation_results['integrity'] = results
        return results
    
    def check_data_leakage(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        检查数据泄漏
        
        Args:
            df: 待检查的DataFrame
            
        Returns:
            检查结果
        """
        logger.info("=" * 80)
        logger.info("数据泄漏检查")
        logger.info("=" * 80)
        
        results = {
            'lag_features_valid': True,
            'future_data_detected': False,
            'issues': [],
            'status': 'PASSED'
        }
        
        # 1. 检查滞后特征是否正确
        lag_features = [col for col in df.columns if 'lag' in col.lower()]
        
        if lag_features:
            logger.info(f"检查 {len(lag_features)} 个滞后特征...")
            
            # 随机抽取一只股票验证
            sample_stock = df['order_book_id'].iloc[0]
            sample_df = df[df['order_book_id'] == sample_stock].head(30)
            
            # 检查 close_lag_1 是否等于前一天的 close
            if 'close_lag_1' in df.columns and 'close' in df.columns:
                close_shifted = sample_df['close'].shift(1)
                close_lag_1 = sample_df['close_lag_1']
                
                # 允许微小的浮点误差
                is_valid = np.allclose(close_shifted.dropna(), close_lag_1.dropna(), 
                                      rtol=1e-9, equal_nan=True)
                
                if is_valid:
                    logger.info("✓ close_lag_1 验证通过 (与shift(1)一致)")
                else:
                    logger.error("✗ close_lag_1 验证失败 (与shift(1)不一致)")
                    results['lag_features_valid'] = False
                    results['issues'].append("close_lag_1 计算错误")
                    results['status'] = 'FAILED'
            
            logger.info(f"✓ 滞后特征检查完成")
        
        # 2. 检查是否使用了未来数据
        # 这里可以添加更复杂的逻辑来检测未来数据泄漏
        
        logger.info(f"\n数据泄漏检查结果: {results['status']}")
        
        self.validation_results['leakage'] = results
        return results
    
    def sample_verification(
        self, 
        df: pd.DataFrame, 
        stock_code: Optional[str] = None,
        n_rows: int = 10
    ) -> pd.DataFrame:
        """
        样本验证
        
        Args:
            df: 待验证的DataFrame
            stock_code: 股票代码(如果为None则随机选择)
            n_rows: 显示行数
            
        Returns:
            样本数据
        """
        logger.info("=" * 80)
        logger.info("样本数据验证")
        logger.info("=" * 80)
        
        if stock_code is None:
            stock_code = df['order_book_id'].iloc[0]
        
        sample_df = df[df['order_book_id'] == stock_code].head(n_rows)
        
        logger.info(f"\n股票: {stock_code}")
        logger.info(f"样本行数: {len(sample_df)}")
        
        # 显示关键字段
        key_cols = ['order_book_id', 'trade_date', 'close', 'vol', 'ret_1d']
        display_cols = [col for col in key_cols if col in sample_df.columns]
        
        logger.info(f"\n关键字段预览:")
        logger.info(f"\n{sample_df[display_cols].to_string()}")
        
        # 显示滞后特征
        lag_cols = [col for col in sample_df.columns if 'lag' in col.lower()]
        if lag_cols:
            logger.info(f"\n滞后特征预览:")
            display_lag = ['trade_date', 'close'] + lag_cols[:5]
            display_lag = [col for col in display_lag if col in sample_df.columns]
            logger.info(f"\n{sample_df[display_lag].to_string()}")
        
        return sample_df
    
    def generate_quality_report(self) -> Dict[str, any]:
        """
        生成数据质量报告
        
        Returns:
            质量报告字典
        """
        logger.info("=" * 80)
        logger.info("生成数据质量报告")
        logger.info("=" * 80)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'PASSED',
            'checks': self.validation_results,
            'summary': {}
        }
        
        # 汇总状态
        statuses = [v.get('status', 'PASSED') for v in self.validation_results.values()]
        if 'FAILED' in statuses:
            report['overall_status'] = 'FAILED'
        elif 'WARNING' in statuses:
            report['overall_status'] = 'WARNING'
        
        # 汇总关键指标
        if 'integrity' in self.validation_results:
            integrity = self.validation_results['integrity']
            report['summary'] = {
                'total_rows': integrity.get('total_rows', 0),
                'total_stocks': integrity.get('total_stocks', 0),
                'date_range': integrity.get('date_range'),
                'missing_ratio': sum(
                    info['percentage'] for info in integrity.get('missing_values', {}).values()
                ) / max(1, len(integrity.get('missing_values', {})))
            }
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"数据质量报告")
        logger.info(f"{'=' * 80}")
        logger.info(f"总体状态: {report['overall_status']}")
        logger.info(f"检查项数: {len(self.validation_results)}")
        
        for check_name, check_result in self.validation_results.items():
            status = check_result.get('status', 'UNKNOWN')
            logger.info(f"  - {check_name}: {status}")
        
        logger.info(f"{'=' * 80}\n")
        
        return report
    
    def validate_feature_distribution(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """
        验证特征分布
        
        Args:
            df: DataFrame
            feature_cols: 特征列名列表
            
        Returns:
            分布统计结果
        """
        logger.info("=" * 80)
        logger.info("特征分布验证")
        logger.info("=" * 80)
        
        results = {}
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            stats = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
                'null_count': int(df[col].isnull().sum()),
                'inf_count': int(np.isinf(df[col]).sum()) if df[col].dtype in [np.float64, np.float32] else 0
            }
            
            results[col] = stats
            
            # 检查异常值
            if stats['inf_count'] > 0:
                logger.warning(f"⚠ {col}: 发现 {stats['inf_count']} 个无穷值")
            
            if abs(stats['mean']) > 1e6:
                logger.warning(f"⚠ {col}: 均值过大 ({stats['mean']:.2e})")
        
        logger.info(f"✓ 已验证 {len(results)} 个特征的分布")
        
        self.validation_results['distribution'] = results
        return results
    
    def check_data_consistency(self, df: pd.DataFrame) -> Dict:
        """
        检查数据一致性
        
        Args:
            df: DataFrame
            
        Returns:
            一致性检查结果
        """
        logger.info("=" * 80)
        logger.info("数据一致性检查")
        logger.info("=" * 80)
        
        results = {
            'issues': [],
            'status': 'PASSED'
        }
        
        # 1. 检查价格逻辑
        if all(col in df.columns for col in ['high', 'low', 'close', 'open']):
            # high应该 >= low
            invalid_high_low = (df['high'] < df['low']).sum()
            if invalid_high_low > 0:
                results['issues'].append(f"发现 {invalid_high_low} 行高价 < 低价")
                results['status'] = 'WARNING'
            
            # close应该在high和low之间
            invalid_close = ((df['close'] > df['high']) | (df['close'] < df['low'])).sum()
            if invalid_close > 0:
                results['issues'].append(f"发现 {invalid_close} 行收盘价超出高低价范围")
                results['status'] = 'WARNING'
            
            logger.info(f"✓ 价格逻辑检查: {len(results['issues'])} 个问题")
        
        # 2. 检查成交量为负
        if 'vol' in df.columns:
            negative_vol = (df['vol'] < 0).sum()
            if negative_vol > 0:
                results['issues'].append(f"发现 {negative_vol} 行负成交量")
                results['status'] = 'WARNING'
            
            logger.info(f"✓ 成交量检查完成")
        
        # 3. 检查时间序列连续性
        if all(col in df.columns for col in ['order_book_id', 'trade_date']):
            # 随机抽取几只股票检查
            sample_stocks = df['order_book_id'].drop_duplicates().sample(min(5, df['order_book_id'].nunique()))
            
            for stock in sample_stocks:
                stock_df = df[df['order_book_id'] == stock].sort_values('trade_date')
                date_diff = stock_df['trade_date'].diff().dt.days
                
                # 检查是否有异常的日期跳跃(超过30天)
                large_gaps = (date_diff > 30).sum()
                if large_gaps > 0:
                    results['issues'].append(f"{stock}: 发现 {large_gaps} 个大时间间隔")
            
            logger.info(f"✓ 时间序列检查完成")
        
        if results['issues']:
            logger.warning(f"⚠ 发现 {len(results['issues'])} 个一致性问题:")
            for issue in results['issues']:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✓ 数据一致性验证通过")
        
        self.validation_results['consistency'] = results
        return results
    
    def run_full_validation(self, df: pd.DataFrame) -> Dict:
        """
        运行完整的验证流程
        
        Args:
            df: 待验证的DataFrame
            
        Returns:
            完整验证报告
        """
        logger.info("\n" + "=" * 80)
        logger.info("开始完整数据验证")
        logger.info("=" * 80 + "\n")
        
        # 1. 数据完整性检查
        self.validate_data_integrity(df)
        
        # 2. 数据泄漏检查
        self.check_data_leakage(df)
        
        # 3. 数据一致性检查
        self.check_data_consistency(df)
        
        # 4. 样本验证
        self.sample_verification(df)
        
        # 5. 生成报告
        report = self.generate_quality_report()
        
        return report
