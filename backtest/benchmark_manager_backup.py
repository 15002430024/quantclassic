"""
基准指数管理器
用于获取和计算基准指数收益率
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class BenchmarkManager:
    """基准指数管理器 - 管理各类基准指数数据"""
    
    # 指数代码映射
    INDEX_MAPPING = {
        'hs300': '000300.SH',     # 沪深300
        'zz500': '000905.SH',     # 中证500
        'zz800': '000906.SH',     # 中证800
        'sz50': '000016.SH',      # 上证50
        'zz1000': '000852.SH',    # 中证1000
        'csi2000': '932000.CSI',  # 中证2000
    }
    
    # 股票池成分映射（代码前缀）
    UNIVERSE_MAPPING = {
        'hs300': ['000300'],
        'zz500': ['000905'],
        'zz800': ['000906'],
    }
    
    def __init__(self):
        """初始化基准管理器"""
        self.logger = logging.getLogger(__name__)
        self.index_data = {}  # 缓存指数数据
    
    def get_benchmark_returns(
        self,
        benchmark_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        data_source: str = 'auto'
    ) -> pd.Series:
        """
        获取基准指数收益率
        
        Args:
            benchmark_name: 基准名称 ('hs300', 'zz500', 'zz800', 等)
            start_date: 起始日期
            end_date: 结束日期
            data_source: 数据源 ('auto', 'rqdatac', 'tushare', 'akshare', 'file')
            
        Returns:
            基准收益率序列（索引为日期）
        """
        if benchmark_name not in self.INDEX_MAPPING:
            raise ValueError(f"不支持的基准指数: {benchmark_name}，支持: {list(self.INDEX_MAPPING.keys())}")
        
        index_code = self.INDEX_MAPPING[benchmark_name]
        
        self.logger.info(f"获取基准指数收益率: {benchmark_name} ({index_code})")
        
        # 根据数据源获取数据
        if data_source == 'rqdatac':
            return self._get_from_rqdatac(index_code, start_date, end_date)
        elif data_source == 'tushare':
            return self._get_from_tushare(index_code, start_date, end_date)
        elif data_source == 'akshare':
            return self._get_from_akshare(index_code, start_date, end_date)
        elif data_source == 'file':
            return self._get_from_file(benchmark_name, start_date, end_date)
        else:  # auto
            # 自动尝试各种数据源
            for source in ['rqdatac', 'akshare', 'tushare', 'file']:
                try:
                    return self.get_benchmark_returns(benchmark_name, start_date, end_date, source)
                except Exception as e:
                    self.logger.debug(f"数据源 {source} 失败: {e}")
                    continue
            
            # 如果都失败，返回零收益率（警告）
            self.logger.warning(f"无法获取基准指数数据，返回零收益率")
            return self._get_zero_returns(start_date, end_date)
    
    def _get_from_rqdatac(
        self,
        index_code: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.Series:
        """从米筐获取指数数据"""
        try:
            import rqdatac as rq
            
            # 获取指数行情
            df = rq.get_price(
                index_code,
                start_date=start_date,
                end_date=end_date,
                frequency='1d',
                fields=['close']
            )
            
            # 计算收益率
            returns = df['close'].pct_change().fillna(0)
            returns.index.name = 'trade_date'
            
            self.logger.info(f"从米筐获取指数数据成功: {len(returns)} 条")
            return returns
        
        except ImportError:
            raise ImportError("需要安装 rqdatac: pip install rqdatac")
        except Exception as e:
            raise RuntimeError(f"米筐数据获取失败: {e}")
    
    def _get_from_tushare(
        self,
        index_code: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.Series:
        """从Tushare获取指数数据"""
        try:
            import tushare as ts
            
            # Tushare代码格式转换
            ts_code = index_code.replace('.SH', '.SH').replace('.SZ', '.SZ')
            
            pro = ts.pro_api()
            df = pro.index_daily(
                ts_code=ts_code,
                start_date=start_date.replace('-', '') if start_date else None,
                end_date=end_date.replace('-', '') if end_date else None
            )
            
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            
            # 计算收益率
            df['return'] = df['close'].pct_change().fillna(0)
            
            returns = df.set_index('trade_date')['return']
            
            self.logger.info(f"从Tushare获取指数数据成功: {len(returns)} 条")
            return returns
        
        except ImportError:
            raise ImportError("需要安装 tushare: pip install tushare")
        except Exception as e:
            raise RuntimeError(f"Tushare数据获取失败: {e}")
    
    def _get_from_akshare(
        self,
        index_code: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.Series:
        """从AkShare获取指数数据"""
        try:
            import akshare as ak
            
            # AkShare代码格式
            symbol = index_code.split('.')[0]
            
            df = ak.stock_zh_index_daily(symbol=symbol)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # 日期过滤
            if start_date:
                df = df[df['date'] >= start_date]
            if end_date:
                df = df[df['date'] <= end_date]
            
            # 计算收益率
            df['return'] = df['close'].pct_change().fillna(0)
            
            returns = df.set_index('date')['return']
            returns.index.name = 'trade_date'
            
            self.logger.info(f"从AkShare获取指数数据成功: {len(returns)} 条")
            return returns
        
        except ImportError:
            raise ImportError("需要安装 akshare: pip install akshare")
        except Exception as e:
            raise RuntimeError(f"AkShare数据获取失败: {e}")
    
    def _get_from_file(
        self,
        benchmark_name: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.Series:
        """从本地文件获取指数数据"""
        # 尝试多个可能的文件路径
        possible_paths = [
            f'data/benchmark/{benchmark_name}.csv',
            f'data/benchmark/{benchmark_name}.parquet',
            f'output/benchmark/{benchmark_name}.csv',
            f'output/benchmark/{benchmark_name}.parquet',
        ]
        
        for path in possible_paths:
            try:
                if path.endswith('.csv'):
                    df = pd.read_csv(path, parse_dates=['trade_date'])
                elif path.endswith('.parquet'):
                    df = pd.read_parquet(path)
                
                df = df.sort_values('trade_date')
                
                # 日期过滤
                if start_date:
                    df = df[df['trade_date'] >= start_date]
                if end_date:
                    df = df[df['trade_date'] <= end_date]
                
                # 计算或读取收益率
                if 'return' in df.columns:
                    returns = df.set_index('trade_date')['return']
                elif 'close' in df.columns:
                    returns = df.set_index('trade_date')['close'].pct_change().fillna(0)
                else:
                    raise ValueError(f"文件缺少 'return' 或 'close' 列")
                
                self.logger.info(f"从文件获取指数数据成功: {path}, {len(returns)} 条")
                return returns
            
            except Exception as e:
                self.logger.debug(f"文件 {path} 读取失败: {e}")
                continue
        
        raise FileNotFoundError(f"未找到基准数据文件: {benchmark_name}")
    
    def _get_zero_returns(
        self,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.Series:
        """生成零收益率序列（回退方案）"""
        # 生成日期范围
        if start_date and end_date:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
        else:
            # 默认最近一年
            dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq='D')
        
        returns = pd.Series(0.0, index=dates)
        returns.index.name = 'trade_date'
        
        return returns
    
    def calculate_excess_returns(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> pd.Series:
        """
        计算超额收益率
        
        Args:
            portfolio_returns: 组合收益率序列
            benchmark_returns: 基准收益率序列
            
        Returns:
            超额收益率序列
        """
        # 对齐日期
        aligned_portfolio, aligned_benchmark = portfolio_returns.align(
            benchmark_returns, 
            join='inner'
        )
        
        excess_returns = aligned_portfolio - aligned_benchmark
        
        self.logger.info(f"计算超额收益率: {len(excess_returns)} 条")
        
        return excess_returns
    
    def get_universe_stocks(
        self,
        universe_name: str,
        date: Optional[str] = None
    ) -> List[str]:
        """
        获取指数成分股列表
        
        Args:
            universe_name: 股票池名称 ('hs300', 'zz500', 'zz800')
            date: 日期（可选，默认最新）
            
        Returns:
            股票代码列表
        """
        if universe_name not in self.UNIVERSE_MAPPING:
            raise ValueError(f"不支持的股票池: {universe_name}")
        
        self.logger.info(f"获取股票池成分: {universe_name}")
        
        # 尝试从各种数据源获取
        try:
            return self._get_universe_from_rqdatac(universe_name, date)
        except:
            pass
        
        try:
            return self._get_universe_from_tushare(universe_name, date)
        except:
            pass
        
        # 如果都失败，返回空列表（或根据代码前缀筛选）
        self.logger.warning(f"无法获取股票池成分，返回空列表")
        return []
    
    def _get_universe_from_rqdatac(
        self,
        universe_name: str,
        date: Optional[str]
    ) -> List[str]:
        """从米筐获取成分股"""
        import rqdatac as rq
        
        index_code = self.INDEX_MAPPING[universe_name]
        
        if date:
            stocks = rq.index_components(index_code, date=date)
        else:
            stocks = rq.index_components(index_code)
        
        return list(stocks)
    
    def _get_universe_from_tushare(
        self,
        universe_name: str,
        date: Optional[str]
    ) -> List[str]:
        """从Tushare获取成分股"""
        import tushare as ts
        
        index_code = self.INDEX_MAPPING[universe_name]
        ts_code = index_code.replace('.SH', '.SH').replace('.SZ', '.SZ')
        
        pro = ts.pro_api()
        df = pro.index_weight(
            index_code=ts_code,
            trade_date=date.replace('-', '') if date else None
        )
        
        return df['con_code'].tolist()


# 便捷函数
def get_benchmark_returns(
    benchmark_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.Series:
    """
    快速获取基准收益率（便捷函数）
    
    Args:
        benchmark_name: 基准名称
        start_date: 起始日期
        end_date: 结束日期
        
    Returns:
        基准收益率序列
    """
    manager = BenchmarkManager()
    return manager.get_benchmark_returns(benchmark_name, start_date, end_date)


if __name__ == '__main__':
    # 测试基准管理器
    print("=" * 80)
    print("BenchmarkManager 测试")
    print("=" * 80)
    
    manager = BenchmarkManager()
    
    # 测试获取基准数据
    print("\n【测试: 获取沪深300指数收益率】")
    try:
        hs300_returns = manager.get_benchmark_returns(
            'hs300',
            start_date='2024-01-01',
            end_date='2024-12-31'
        )
        print(f"数据长度: {len(hs300_returns)}")
        print(f"平均收益率: {hs300_returns.mean():.4%}")
        print(f"累计收益率: {(1 + hs300_returns).prod() - 1:.4%}")
        print(hs300_returns.head())
    except Exception as e:
        print(f"获取失败: {e}")
    
    print("\n✅ 测试完成")
