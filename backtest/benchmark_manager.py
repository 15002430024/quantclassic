"""
基准指数管理器（带智能缓存）
用于获取和计算基准指数收益率，支持数据缓存和增量更新
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging
import os
import json
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BenchmarkCache:
    """基准数据缓存管理器"""
    
    def __init__(self, cache_dir: str = "cache/benchmark"):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.BenchmarkCache")
        
    def get_cache_path(self, index_code: str) -> Path:
        """获取指定指数的缓存文件路径"""
        # 将指数代码中的点替换为下划线，避免文件名问题
        safe_code = index_code.replace('.', '_')
        return self.cache_dir / f"{safe_code}.parquet"
    
    def get_metadata_path(self, index_code: str) -> Path:
        """获取指定指数的元数据文件路径"""
        safe_code = index_code.replace('.', '_')
        return self.cache_dir / f"{safe_code}_meta.json"
    
    def load_cache(self, index_code: str) -> Optional[pd.DataFrame]:
        """
        加载缓存的指数数据
        
        Args:
            index_code: 指数代码
            
        Returns:
            缓存的数据框，如果不存在则返回None
        """
        cache_path = self.get_cache_path(index_code)
        
        if not cache_path.exists():
            self.logger.debug(f"缓存文件不存在: {cache_path}")
            return None
        
        try:
            df = pd.read_parquet(cache_path)
            self.logger.info(f"成功加载缓存数据: {index_code}, {len(df)} 条记录")
            return df
        except Exception as e:
            self.logger.warning(f"加载缓存失败: {e}")
            return None
    
    def save_cache(self, index_code: str, df: pd.DataFrame) -> None:
        """
        保存指数数据到缓存
        
        Args:
            index_code: 指数代码
            df: 要保存的数据框（必须包含 trade_date 和 close 列）
        """
        cache_path = self.get_cache_path(index_code)
        
        try:
            # 确保数据按日期排序
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            # 保存数据
            df.to_parquet(cache_path, index=False)
            
            # 保存元数据
            metadata = {
                'index_code': index_code,
                'start_date': df['trade_date'].min().strftime('%Y-%m-%d'),
                'end_date': df['trade_date'].max().strftime('%Y-%m-%d'),
                'record_count': len(df),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            meta_path = self.get_metadata_path(index_code)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"成功保存缓存: {index_code}, {len(df)} 条记录 "
                           f"[{metadata['start_date']} ~ {metadata['end_date']}]")
        
        except Exception as e:
            self.logger.error(f"保存缓存失败: {e}")
    
    def get_cache_range(self, index_code: str) -> Optional[Tuple[str, str]]:
        """
        获取缓存的日期范围
        
        Args:
            index_code: 指数代码
            
        Returns:
            (start_date, end_date) 元组，如果不存在则返回None
        """
        meta_path = self.get_metadata_path(index_code)
        
        if not meta_path.exists():
            return None
        
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            return (metadata['start_date'], metadata['end_date'])
        
        except Exception as e:
            self.logger.warning(f"读取元数据失败: {e}")
            return None
    
    def check_cache_coverage(
        self,
        index_code: str,
        start_date: str,
        end_date: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        检查缓存是否覆盖所需的日期范围
        
        Args:
            index_code: 指数代码
            start_date: 需要的起始日期
            end_date: 需要的结束日期
            
        Returns:
            (is_covered, fetch_start, fetch_end)
            - is_covered: 是否完全覆盖
            - fetch_start: 如果需要获取，从哪个日期开始
            - fetch_end: 如果需要获取，到哪个日期结束
        """
        cache_range = self.get_cache_range(index_code)
        
        if cache_range is None:
            # 缓存不存在，需要完整下载
            self.logger.info(f"缓存不存在，需要下载完整范围: {start_date} ~ {end_date}")
            return False, start_date, end_date
        
        cache_start, cache_end = cache_range
        
        # 转换为日期对象进行比较
        cache_start_dt = pd.to_datetime(cache_start)
        cache_end_dt = pd.to_datetime(cache_end)
        request_start_dt = pd.to_datetime(start_date)
        request_end_dt = pd.to_datetime(end_date)
        
        # 检查是否完全覆盖
        if cache_start_dt <= request_start_dt and cache_end_dt >= request_end_dt:
            self.logger.info(f"缓存完全覆盖所需范围: [{cache_start} ~ {cache_end}] "
                           f"包含 [{start_date} ~ {end_date}]")
            return True, None, None
        
        # 需要扩展缓存
        fetch_start = min(cache_start_dt, request_start_dt).strftime('%Y-%m-%d')
        fetch_end = max(cache_end_dt, request_end_dt).strftime('%Y-%m-%d')
        
        self.logger.info(f"缓存范围不足，需要扩展: "
                       f"当前 [{cache_start} ~ {cache_end}] -> "
                       f"扩展到 [{fetch_start} ~ {fetch_end}]")
        
        return False, fetch_start, fetch_end
    
    def merge_with_cache(
        self,
        index_code: str,
        new_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        将新数据与缓存合并
        
        Args:
            index_code: 指数代码
            new_df: 新获取的数据
            
        Returns:
            合并后的数据框
        """
        cached_df = self.load_cache(index_code)
        
        if cached_df is None:
            return new_df
        
        # 合并并去重
        merged_df = pd.concat([cached_df, new_df], ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=['trade_date'], keep='last')
        merged_df = merged_df.sort_values('trade_date').reset_index(drop=True)
        
        self.logger.info(f"合并数据: 缓存 {len(cached_df)} + 新增 {len(new_df)} = 总计 {len(merged_df)}")
        
        return merged_df


class BenchmarkManager:
    """基准指数管理器 - 管理各类基准指数数据（支持智能缓存）"""
    
    # 指数代码映射（米筐格式）
    INDEX_MAPPING = {
        'hs300': '000300.XSHG',     # 沪深300
        'zz500': '000905.XSHG',     # 中证500
        'zz800': '000906.XSHG',     # 中证800
        'sz50': '000016.XSHG',      # 上证50
        'zz1000': '000852.XSHG',    # 中证1000
        'csi2000': '932000.CSI',    # 中证2000
        'szzs': '399001.XSHE',      # 深证成指
        'cybz': '399006.XSHE',      # 创业板指
    }
    
    # 股票池成分映射（代码前缀）
    UNIVERSE_MAPPING = {
        'hs300': ['000300'],
        'zz500': ['000905'],
        'zz800': ['000906'],
    }
    
    def __init__(self, cache_dir: str = "cache/benchmark"):
        """
        初始化基准管理器
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.logger = logging.getLogger(__name__)
        self.index_data = {}  # 内存缓存
        self.cache = BenchmarkCache(cache_dir)
    
    def get_benchmark_returns(
        self,
        benchmark_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        data_source: str = 'auto',
        use_cache: bool = True
    ) -> pd.Series:
        """
        获取基准指数收益率（支持智能缓存）
        
        Args:
            benchmark_name: 基准名称 ('hs300', 'zz500', 'zz800', 等)
            start_date: 起始日期
            end_date: 结束日期
            data_source: 数据源 ('auto', 'rqdatac', 'tushare', 'akshare', 'file')
            use_cache: 是否使用缓存
            
        Returns:
            基准收益率序列（索引为日期）
        """
        if benchmark_name not in self.INDEX_MAPPING:
            raise ValueError(f"不支持的基准指数: {benchmark_name}，支持: {list(self.INDEX_MAPPING.keys())}")
        
        index_code = self.INDEX_MAPPING[benchmark_name]
        
        self.logger.info(f"获取基准指数收益率: {benchmark_name} ({index_code})")
        
        # 根据数据源获取数据
        if data_source == 'rqdatac':
            return self._get_from_rqdatac(index_code, start_date, end_date, use_cache)
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
                    return self.get_benchmark_returns(
                        benchmark_name, start_date, end_date, source, use_cache
                    )
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
        end_date: Optional[str],
        use_cache: bool = True
    ) -> pd.Series:
        """
        从米筐获取指数数据（支持智能缓存）
        
        Args:
            index_code: 指数代码（米筐格式，如 000300.XSHG）
            start_date: 起始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
            
        Returns:
            收益率序列
        """
        try:
            import rqdatac as rq
            
            # 如果没有指定日期范围，使用默认范围
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # 检查缓存
            if use_cache:
                is_covered, fetch_start, fetch_end = self.cache.check_cache_coverage(
                    index_code, start_date, end_date
                )
                
                if is_covered:
                    # 从缓存加载
                    cached_df = self.cache.load_cache(index_code)
                    # 筛选日期范围
                    mask = (cached_df['trade_date'] >= start_date) & (cached_df['trade_date'] <= end_date)
                    df = cached_df[mask].copy()
                    self.logger.info(f"从缓存提取数据: {len(df)} 条 [{start_date} ~ {end_date}]")
                else:
                    # 需要从API获取更多数据
                    self.logger.info(f"从米筐API获取数据: {fetch_start} ~ {fetch_end}")
                    
                    # 初始化米筐（如果需要）
                    try:
                        rq.init()
                    except:
                        pass
                    
                    # 获取指数行情
                    price_df = rq.get_price(
                        index_code,
                        start_date=fetch_start,
                        end_date=fetch_end,
                        frequency='1d',
                        fields=['close']
                    )
                    
                    # 转换为DataFrame
                    new_df = pd.DataFrame({
                        'trade_date': price_df.index,
                        'close': price_df['close'].values
                    })
                    new_df['trade_date'] = pd.to_datetime(new_df['trade_date'])
                    
                    # 合并并保存到缓存
                    merged_df = self.cache.merge_with_cache(index_code, new_df)
                    self.cache.save_cache(index_code, merged_df)
                    
                    # 筛选所需日期范围
                    mask = (merged_df['trade_date'] >= start_date) & (merged_df['trade_date'] <= end_date)
                    df = merged_df[mask].copy()
            else:
                # 不使用缓存，直接从API获取
                self.logger.info(f"直接从米筐API获取数据（不使用缓存）: {start_date} ~ {end_date}")
                
                try:
                    rq.init()
                except:
                    pass
                
                price_df = rq.get_price(
                    index_code,
                    start_date=start_date,
                    end_date=end_date,
                    frequency='1d',
                    fields=['close']
                )
                
                df = pd.DataFrame({
                    'trade_date': price_df.index,
                    'close': price_df['close'].values
                })
                df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            # 计算收益率
            df = df.sort_values('trade_date')
            returns = df.set_index('trade_date')['close'].pct_change().fillna(0)
            
            self.logger.info(f"成功获取指数数据: {len(returns)} 条")
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
            
            # Tushare代码格式转换（米筐格式 -> Tushare格式）
            if index_code.endswith('.XSHG'):
                ts_code = index_code.replace('.XSHG', '.SH')
            elif index_code.endswith('.XSHE'):
                ts_code = index_code.replace('.XSHE', '.SZ')
            else:
                ts_code = index_code
            
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
            f'cache/benchmark/{benchmark_name}.parquet',
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
    
    def clear_cache(self, index_code: Optional[str] = None):
        """
        清除缓存
        
        Args:
            index_code: 指定清除某个指数的缓存，如果为None则清除所有
        """
        if index_code:
            # 清除指定指数
            cache_path = self.cache.get_cache_path(index_code)
            meta_path = self.cache.get_metadata_path(index_code)
            
            if cache_path.exists():
                cache_path.unlink()
                self.logger.info(f"已清除缓存: {cache_path}")
            
            if meta_path.exists():
                meta_path.unlink()
                self.logger.info(f"已清除元数据: {meta_path}")
        else:
            # 清除所有缓存
            import shutil
            if self.cache.cache_dir.exists():
                shutil.rmtree(self.cache.cache_dir)
                self.cache.cache_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"已清除所有缓存: {self.cache.cache_dir}")
    
    def get_cache_info(self) -> pd.DataFrame:
        """
        获取所有缓存的信息
        
        Returns:
            包含所有缓存信息的DataFrame
        """
        cache_info = []
        
        for benchmark_name, index_code in self.INDEX_MAPPING.items():
            meta_path = self.cache.get_metadata_path(index_code)
            
            if meta_path.exists():
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    cache_info.append({
                        'benchmark_name': benchmark_name,
                        'index_code': index_code,
                        'start_date': metadata['start_date'],
                        'end_date': metadata['end_date'],
                        'record_count': metadata['record_count'],
                        'last_updated': metadata['last_updated']
                    })
                except:
                    pass
        
        if cache_info:
            return pd.DataFrame(cache_info)
        else:
            return pd.DataFrame(columns=[
                'benchmark_name', 'index_code', 'start_date', 
                'end_date', 'record_count', 'last_updated'
            ])
    
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
        
        # 如果都失败，返回空列表
        self.logger.warning(f"无法获取股票池成分，返回空列表")
        return []
    
    def _get_universe_from_rqdatac(
        self,
        universe_name: str,
        date: Optional[str]
    ) -> List[str]:
        """从米筐获取成分股"""
        import rqdatac as rq
        
        try:
            rq.init()
        except:
            pass
        
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
        
        # 转换为Tushare格式
        if index_code.endswith('.XSHG'):
            ts_code = index_code.replace('.XSHG', '.SH')
        elif index_code.endswith('.XSHE'):
            ts_code = index_code.replace('.XSHE', '.SZ')
        else:
            ts_code = index_code
        
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
    end_date: Optional[str] = None,
    use_cache: bool = True
) -> pd.Series:
    """
    快速获取基准收益率（便捷函数）
    
    Args:
        benchmark_name: 基准名称
        start_date: 起始日期
        end_date: 结束日期
        use_cache: 是否使用缓存
        
    Returns:
        基准收益率序列
    """
    manager = BenchmarkManager()
    return manager.get_benchmark_returns(
        benchmark_name, start_date, end_date, use_cache=use_cache
    )


if __name__ == '__main__':
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 测试基准管理器
    print("=" * 80)
    print("BenchmarkManager 测试（带智能缓存）")
    print("=" * 80)
    
    manager = BenchmarkManager()
    
    # 测试1: 第一次获取数据（从API）
    print("\n【测试1: 第一次获取沪深300指数收益率 - 应该从API下载】")
    try:
        hs300_returns = manager.get_benchmark_returns(
            'hs300',
            start_date='2023-01-01',
            end_date='2023-12-31',
            data_source='rqdatac'
        )
        print(f"✓ 数据长度: {len(hs300_returns)}")
        print(f"✓ 日期范围: {hs300_returns.index[0]} ~ {hs300_returns.index[-1]}")
        print(f"✓ 平均收益率: {hs300_returns.mean():.4%}")
        print(f"✓ 累计收益率: {(1 + hs300_returns).prod() - 1:.4%}")
    except Exception as e:
        print(f"✗ 获取失败: {e}")
    
    # 测试2: 第二次获取相同范围（从缓存）
    print("\n【测试2: 第二次获取相同日期范围 - 应该从缓存加载】")
    try:
        hs300_returns_cached = manager.get_benchmark_returns(
            'hs300',
            start_date='2023-01-01',
            end_date='2023-12-31',
            data_source='rqdatac'
        )
        print(f"✓ 数据长度: {len(hs300_returns_cached)}")
        print(f"✓ 数据一致性: {hs300_returns.equals(hs300_returns_cached)}")
    except Exception as e:
        print(f"✗ 获取失败: {e}")
    
    # 测试3: 扩展日期范围（增量更新）
    print("\n【测试3: 扩展日期范围 - 应该增量下载并合并】")
    try:
        hs300_returns_extended = manager.get_benchmark_returns(
            'hs300',
            start_date='2022-01-01',
            end_date='2024-06-30',
            data_source='rqdatac'
        )
        print(f"✓ 数据长度: {len(hs300_returns_extended)}")
        print(f"✓ 日期范围: {hs300_returns_extended.index[0]} ~ {hs300_returns_extended.index[-1]}")
    except Exception as e:
        print(f"✗ 获取失败: {e}")
    
    # 测试4: 查看缓存信息
    print("\n【测试4: 查看缓存信息】")
    try:
        cache_info = manager.get_cache_info()
        if not cache_info.empty:
            print(cache_info.to_string(index=False))
        else:
            print("没有缓存数据")
    except Exception as e:
        print(f"✗ 查看失败: {e}")
    
    # 测试5: 测试中证800
    print("\n【测试5: 测试中证800指数】")
    try:
        zz800_returns = manager.get_benchmark_returns(
            'zz800',
            start_date='2023-01-01',
            end_date='2023-12-31',
            data_source='rqdatac'
        )
        print(f"✓ 数据长度: {len(zz800_returns)}")
        print(f"✓ 平均收益率: {zz800_returns.mean():.4%}")
        print(f"✓ 累计收益率: {(1 + zz800_returns).prod() - 1:.4%}")
    except Exception as e:
        print(f"✗ 获取失败: {e}")
    
    print("\n✅ 测试完成")
