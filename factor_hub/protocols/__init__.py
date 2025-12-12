"""
Standard Data Protocol - 定义因子计算引擎接收的数据格式标准

该模块明确规定了数据流动的"标准语言"：
1. DataFrame 的 Index 结构要求
2. 必须包含的列名（symbol, datetime, open, high, low, close, volume 等）
3. 数据类型约束
4. 数据校验方法
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Union, TYPE_CHECKING
from datetime import datetime

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class DataColumnSpec:
    """数据列规范定义"""
    
    # 必须的列
    REQUIRED_COLUMNS: frozenset = field(default_factory=lambda: frozenset({
        "symbol",      # 股票代码
        "datetime",    # 日期时间
        "open",        # 开盘价
        "high",        # 最高价
        "low",         # 最低价
        "close",       # 收盘价
        "volume",      # 成交量
    }))
    
    # 可选的列
    OPTIONAL_COLUMNS: frozenset = field(default_factory=lambda: frozenset({
        "amount",      # 成交额
        "vwap",        # 成交量加权平均价
        "turnover",    # 换手率
        "adj_factor",  # 复权因子
        "total_shares",    # 总股本
        "float_shares",    # 流通股本
        "market_cap",      # 总市值
        "float_market_cap", # 流通市值
    }))
    
    # 列的数据类型约束
    COLUMN_DTYPES: dict = field(default_factory=lambda: {
        "symbol": "object",
        "datetime": "datetime64[ns]",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
        "amount": "float64",
        "vwap": "float64",
        "turnover": "float64",
        "adj_factor": "float64",
        "total_shares": "float64",
        "float_shares": "float64",
        "market_cap": "float64",
        "float_market_cap": "float64",
    })


# 全局列规范实例
COLUMN_SPEC = DataColumnSpec()


class DataValidationError(Exception):
    """数据校验异常"""
    pass


class StandardDataProtocol:
    """
    标准化数据协议类
    
    该类封装了符合标准协议的 DataFrame，提供：
    1. 数据格式校验
    2. 标准化的数据访问接口
    3. 便捷的数据操作方法
    
    Index 结构要求：
    - 使用 MultiIndex: (datetime, symbol)
    - 或者将 datetime 和 symbol 作为普通列
    
    Example:
        >>> data = StandardDataProtocol(df)
        >>> data.validate()
        >>> close_prices = data.close
        >>> symbols = data.symbols
    """
    
    def __init__(
        self, 
        data: pd.DataFrame,
        auto_validate: bool = True,
        auto_standardize: bool = True
    ) -> None:
        """
        初始化标准数据协议
        
        Args:
            data: 原始数据 DataFrame
            auto_validate: 是否自动进行数据校验
            auto_standardize: 是否自动标准化数据格式
        """
        self._raw_data = data.copy()
        self._data: pd.DataFrame = data.copy()
        
        if auto_standardize:
            self._standardize()
        
        if auto_validate:
            self.validate()
    
    def _standardize(self) -> None:
        """标准化数据格式"""
        # 如果 datetime 在 index 中，重置为列
        if "datetime" in self._data.index.names:
            self._data = self._data.reset_index()
        
        # 如果 symbol 在 index 中，重置为列
        if "symbol" in self._data.index.names:
            self._data = self._data.reset_index()
        
        # 确保 datetime 列是 datetime 类型
        if "datetime" in self._data.columns:
            self._data["datetime"] = pd.to_datetime(self._data["datetime"])
        
        # 确保 symbol 列是字符串类型
        if "symbol" in self._data.columns:
            self._data["symbol"] = self._data["symbol"].astype(str)
        
        # 设置 MultiIndex: (datetime, symbol)
        if "datetime" in self._data.columns and "symbol" in self._data.columns:
            self._data = self._data.set_index(["datetime", "symbol"]).sort_index()
    
    def validate(self) -> bool:
        """
        校验数据是否符合标准协议
        
        Returns:
            bool: 校验通过返回 True
            
        Raises:
            DataValidationError: 校验失败时抛出异常
        """
        errors: List[str] = []
        
        # 检查必须的列（考虑 index 和 columns）
        all_columns = set(self._data.columns.tolist() + list(self._data.index.names))
        missing_columns = COLUMN_SPEC.REQUIRED_COLUMNS - all_columns
        
        if missing_columns:
            errors.append(f"缺少必须的列: {missing_columns}")
        
        # 检查数据是否为空
        if self._data.empty:
            errors.append("数据为空")
        
        # 检查数值列是否包含无穷大
        numeric_cols = self._data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(self._data[col]).any():
                errors.append(f"列 '{col}' 包含无穷大值")
        
        # 汇总错误
        if errors:
            raise DataValidationError("\n".join(errors))
        
        return True
    
    @property
    def data(self) -> pd.DataFrame:
        """获取标准化后的 DataFrame"""
        return self._data
    
    @property
    def raw_data(self) -> pd.DataFrame:
        """获取原始 DataFrame"""
        return self._raw_data
    
    @property
    def symbols(self) -> List[str]:
        """获取所有股票代码"""
        if "symbol" in self._data.index.names:
            return self._data.index.get_level_values("symbol").unique().tolist()
        return self._data["symbol"].unique().tolist()
    
    @property
    def datetimes(self) -> pd.DatetimeIndex:
        """获取所有日期时间"""
        if "datetime" in self._data.index.names:
            return self._data.index.get_level_values("datetime").unique()
        return pd.DatetimeIndex(self._data["datetime"].unique())
    
    @property
    def start_date(self) -> pd.Timestamp:
        """获取起始日期"""
        return self.datetimes.min()
    
    @property
    def end_date(self) -> pd.Timestamp:
        """获取结束日期"""
        return self.datetimes.max()
    
    def __getattr__(self, name: str) -> pd.Series:
        """
        动态获取列数据
        
        Example:
            >>> data.close  # 获取收盘价
            >>> data.volume  # 获取成交量
        """
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        if name in self._data.columns:
            return self._data[name]
        
        raise AttributeError(f"列 '{name}' 不存在于数据中")
    
    def get_symbol_data(self, symbol: str) -> pd.DataFrame:
        """获取单个股票的数据"""
        if "symbol" in self._data.index.names:
            return self._data.xs(symbol, level="symbol")
        return self._data[self._data["symbol"] == symbol]
    
    def get_date_data(self, date: Union[str, datetime, pd.Timestamp]) -> pd.DataFrame:
        """获取单个日期的数据"""
        date = pd.Timestamp(date)
        if "datetime" in self._data.index.names:
            return self._data.xs(date, level="datetime")
        return self._data[self._data["datetime"] == date]
    
    def to_pivot(self, column: str) -> pd.DataFrame:
        """
        将数据转换为透视表格式
        
        Args:
            column: 要透视的列名
            
        Returns:
            DataFrame with index=datetime, columns=symbol
        """
        if "datetime" in self._data.index.names and "symbol" in self._data.index.names:
            return self._data[column].unstack(level="symbol")
        return self._data.pivot(index="datetime", columns="symbol", values=column)
    
    def __repr__(self) -> str:
        return (
            f"StandardDataProtocol(\n"
            f"  symbols={len(self.symbols)}, "
            f"  date_range=[{self.start_date.date()} ~ {self.end_date.date()}], "
            f"  rows={len(self._data)}\n"
            f")"
        )
    
    def info(self) -> None:
        """打印数据信息"""
        print("=" * 60)
        print("StandardDataProtocol - Data Info")
        print("=" * 60)
        print(f"  股票数量: {len(self.symbols)}")
        print(f"  日期范围: {self.start_date.date()} ~ {self.end_date.date()}")
        print(f"  数据行数: {len(self._data)}")
        print(f"  股票列表: {self.symbols[:5]}{'...' if len(self.symbols) > 5 else ''}")
        print("-" * 60)
        print("DataFrame Info:")
        self._data.info()
        print("=" * 60)
