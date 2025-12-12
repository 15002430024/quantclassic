"""
Demo Factors - 示例因子

提供一些简单的示例因子，用于测试和演示。
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

import pandas as pd
import numpy as np

from quantclassic.factor_hub.factors import BaseFactor, FactorMeta, factor_registry

if TYPE_CHECKING:
    from quantclassic.factor_hub.protocols import StandardDataProtocol


@factor_registry.register
class Return1DFactor(BaseFactor):
    """
    1日收益率因子
    
    计算公式: (close_t - close_{t-1}) / close_{t-1}
    """
    
    @property
    def meta(self) -> FactorMeta:
        return FactorMeta(
            name="return_1d",
            description="1日收益率",
            category="momentum",
            version="1.0.0",
        )
    
    def compute(self, data: "StandardDataProtocol") -> pd.Series:
        close = data.close
        # 按股票分组计算收益率
        returns = close.groupby(level="symbol").pct_change(1)
        returns.name = self.name
        return returns


@factor_registry.register
class Return5DFactor(BaseFactor):
    """
    5日收益率因子
    
    计算公式: (close_t - close_{t-5}) / close_{t-5}
    """
    
    @property
    def meta(self) -> FactorMeta:
        return FactorMeta(
            name="return_5d",
            description="5日收益率",
            category="momentum",
            version="1.0.0",
        )
    
    @property
    def default_params(self) -> Dict[str, Any]:
        return {"period": 5}
    
    def compute(self, data: "StandardDataProtocol") -> pd.Series:
        close = data.close
        period = self.params.get("period", 5)
        returns = close.groupby(level="symbol").pct_change(period)
        returns.name = self.name
        return returns


@factor_registry.register
class VolatilityFactor(BaseFactor):
    """
    波动率因子
    
    计算过去 N 日收益率的标准差
    """
    
    @property
    def meta(self) -> FactorMeta:
        return FactorMeta(
            name="volatility",
            description="N日波动率",
            category="risk",
            version="1.0.0",
        )
    
    @property
    def default_params(self) -> Dict[str, Any]:
        return {"window": 20}
    
    def compute(self, data: "StandardDataProtocol") -> pd.Series:
        close = data.close
        window = self.params.get("window", 20)
        
        # 计算日收益率
        returns = close.groupby(level="symbol").pct_change(1)
        
        # 计算滚动波动率
        volatility = returns.groupby(level="symbol").rolling(
            window=window, min_periods=1
        ).std().droplevel(0)
        
        volatility.name = self.name
        return volatility


@factor_registry.register
class TurnoverFactor(BaseFactor):
    """
    换手率因子 (成交量 / 基准成交量)
    
    使用成交量相对于其均值的比值
    """
    
    @property
    def meta(self) -> FactorMeta:
        return FactorMeta(
            name="turnover_ratio",
            description="换手率因子",
            category="liquidity",
            version="1.0.0",
        )
    
    @property
    def default_params(self) -> Dict[str, Any]:
        return {"window": 20}
    
    def compute(self, data: "StandardDataProtocol") -> pd.Series:
        volume = data.volume
        window = self.params.get("window", 20)
        
        # 计算成交量均值
        vol_mean = volume.groupby(level="symbol").rolling(
            window=window, min_periods=1
        ).mean().droplevel(0)
        
        # 计算换手率比值
        turnover_ratio = volume / vol_mean
        turnover_ratio.name = self.name
        return turnover_ratio


@factor_registry.register
class PriceRangeFactor(BaseFactor):
    """
    价格振幅因子
    
    计算公式: (high - low) / close
    """
    
    @property
    def meta(self) -> FactorMeta:
        return FactorMeta(
            name="price_range",
            description="日内价格振幅",
            category="volatility",
            version="1.0.0",
        )
    
    def compute(self, data: "StandardDataProtocol") -> pd.Series:
        high = data.high
        low = data.low
        close = data.close
        
        price_range = (high - low) / close
        price_range.name = self.name
        return price_range
