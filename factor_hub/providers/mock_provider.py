"""
Mock Data Provider - 用于测试的模拟数据提供者

生成符合 StandardDataProtocol 的随机测试数据。
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Union

import pandas as pd
import numpy as np

from quantclassic.factor_hub.providers import BaseDataAdapter


class MockDataProvider(BaseDataAdapter):
    """
    模拟数据提供者
    
    用于测试和开发阶段，随机生成符合 StandardDataProtocol 的数据。
    
    Features:
    - 生成符合协议的 OHLCV 数据
    - 支持自定义股票池
    - 支持指定时间范围
    - 价格走势有一定的随机游走特性
    
    Example:
        >>> provider = MockDataProvider()
        >>> data = provider.get_history(
        ...     symbols=["000001.SZ", "600000.SH"],
        ...     start="2024-01-01",
        ...     end="2024-03-01"
        ... )
    """
    
    # 默认的模拟股票池
    DEFAULT_SYMBOLS = [
        "000001.SZ",  # 平安银行
        "000002.SZ",  # 万科A
        "600000.SH",  # 浦发银行
        "600519.SH",  # 贵州茅台
        "000858.SZ",  # 五粮液
    ]
    
    def __init__(
        self,
        default_symbols: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        初始化 Mock 数据提供者
        
        Args:
            default_symbols: 默认股票池，None 使用内置股票池
            seed: 随机种子，用于复现
        """
        self._default_symbols = default_symbols or self.DEFAULT_SYMBOLS
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    @property
    def name(self) -> str:
        return "MockDataProvider"
    
    @property
    def description(self) -> str:
        return "用于测试的模拟数据提供者，生成符合标准协议的随机 OHLCV 数据"
    
    def get_history(
        self,
        symbols: Union[str, List[str]],
        start: Union[str, datetime, pd.Timestamp],
        end: Union[str, datetime, pd.Timestamp],
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        生成模拟历史数据
        
        Args:
            symbols: 股票代码
            start: 起始日期
            end: 结束日期
            fields: 字段列表（忽略，始终返回全部字段）
            
        Returns:
            pd.DataFrame: 符合 StandardDataProtocol 的模拟数据
        """
        # 标准化参数
        symbols_list = self._normalize_symbols(symbols)
        start_ts, end_ts = self._normalize_dates(start, end)
        self._validate_date_range(start_ts, end_ts)
        
        # 生成交易日序列（排除周末）
        all_dates = pd.date_range(start=start_ts, end=end_ts, freq='B')  # 'B' = business day
        
        # 为每个股票生成数据
        all_data: List[pd.DataFrame] = []
        
        for symbol in symbols_list:
            df = self._generate_stock_data(symbol, all_dates)
            all_data.append(df)
        
        # 合并所有数据
        result = pd.concat(all_data, ignore_index=True)
        
        return result
    
    def _generate_stock_data(
        self,
        symbol: str,
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        为单个股票生成模拟数据
        
        使用几何布朗运动模拟价格走势
        """
        n_days = len(dates)
        
        # 设置初始价格（基于股票代码 hash 保证同一股票价格一致）
        symbol_hash = hash(symbol) % 1000
        initial_price = 10 + symbol_hash / 10  # 10 ~ 110
        
        # 几何布朗运动参数
        mu = 0.0001  # 日均收益率
        sigma = 0.02  # 日波动率
        
        # 生成收益率序列
        returns = np.random.normal(mu, sigma, n_days)
        
        # 计算价格序列
        price_multipliers = np.exp(np.cumsum(returns))
        close_prices = initial_price * price_multipliers
        
        # 生成 OHLC 数据
        # High/Low 在 Close 附近波动
        daily_range = np.random.uniform(0.01, 0.03, n_days)
        
        high_prices = close_prices * (1 + daily_range / 2)
        low_prices = close_prices * (1 - daily_range / 2)
        
        # Open 在前一天 Close 附近
        open_prices = np.roll(close_prices, 1) * np.random.uniform(0.99, 1.01, n_days)
        open_prices[0] = initial_price
        
        # 确保 High >= max(Open, Close), Low <= min(Open, Close)
        high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
        low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
        
        # 生成成交量（与价格变动相关）
        base_volume = 1e6 * (1 + symbol_hash / 500)  # 基础成交量
        volume_volatility = np.abs(returns) * 10 + 1  # 波动大时成交量放大
        volumes = base_volume * volume_volatility * np.random.uniform(0.8, 1.2, n_days)
        
        # 计算成交额
        vwap = (high_prices + low_prices + close_prices) / 3
        amounts = volumes * vwap
        
        # 构建 DataFrame
        df = pd.DataFrame({
            "symbol": symbol,
            "datetime": dates,
            "open": np.round(open_prices, 2),
            "high": np.round(high_prices, 2),
            "low": np.round(low_prices, 2),
            "close": np.round(close_prices, 2),
            "volume": np.round(volumes, 0),
            "amount": np.round(amounts, 2),
            "vwap": np.round(vwap, 2),
        })
        
        return df
    
    def get_latest(
        self,
        symbols: Union[str, List[str]],
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """获取最新数据（返回最近一个交易日的数据）"""
        today = pd.Timestamp.now().normalize()
        # 往前找最近的交易日
        if today.weekday() >= 5:  # 周末
            today = today - pd.Timedelta(days=today.weekday() - 4)
        
        return self.get_history(symbols, today, today, fields)
