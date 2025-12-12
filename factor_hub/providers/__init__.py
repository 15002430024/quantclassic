"""
Data Provider 抽象基类和相关接口

该模块定义了数据获取的标准接口，使用适配器模式设计，
方便未来对接不同的数据源（如 datafetch 外部包）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Union, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from factor_hub.protocols import StandardDataProtocol


class IDataProvider(ABC):
    """
    数据提供者抽象基类 (Interface)
    
    使用适配器模式设计，定义了获取数据的标准接口。
    所有数据源适配器都必须实现这个接口。
    
    设计原则：
    1. 依赖倒置：高层模块不依赖低层模块，都依赖抽象
    2. 开闭原则：对扩展开放，对修改关闭
    3. 单一职责：只负责数据获取
    
    Example:
        >>> class MyDataProvider(IDataProvider):
        ...     def get_history(self, symbols, start, end, fields):
        ...         # 实现数据获取逻辑
        ...         pass
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """数据源名称"""
        pass
    
    @property
    def description(self) -> str:
        """数据源描述"""
        return ""
    
    @abstractmethod
    def get_history(
        self,
        symbols: Union[str, List[str]],
        start: Union[str, datetime, pd.Timestamp],
        end: Union[str, datetime, pd.Timestamp],
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        获取历史数据
        
        Args:
            symbols: 股票代码，单个或列表
            start: 起始日期
            end: 结束日期
            fields: 需要的字段列表，None 表示获取全部字段
            
        Returns:
            pd.DataFrame: 包含指定字段的数据，格式需符合 StandardDataProtocol
            
        Raises:
            DataFetchError: 数据获取失败时抛出
        """
        pass
    
    def get_latest(
        self,
        symbols: Union[str, List[str]],
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        获取最新数据（可选实现）
        
        Args:
            symbols: 股票代码
            fields: 需要的字段列表
            
        Returns:
            pd.DataFrame: 最新一期数据
        """
        raise NotImplementedError("该数据源不支持获取最新数据")
    
    def is_available(self) -> bool:
        """
        检查数据源是否可用
        
        Returns:
            bool: 数据源可用返回 True
        """
        return True
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class DataFetchError(Exception):
    """数据获取异常"""
    pass


class BaseDataAdapter(IDataProvider):
    """
    数据适配器基类
    
    提供一些通用的数据处理方法，子类可以复用。
    """
    
    def _normalize_symbols(
        self, 
        symbols: Union[str, List[str]]
    ) -> List[str]:
        """标准化股票代码格式"""
        if isinstance(symbols, str):
            return [symbols]
        return list(symbols)
    
    def _normalize_dates(
        self,
        start: Union[str, datetime, pd.Timestamp],
        end: Union[str, datetime, pd.Timestamp],
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """标准化日期格式"""
        return pd.Timestamp(start), pd.Timestamp(end)
    
    def _validate_date_range(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> None:
        """校验日期范围"""
        if start > end:
            raise ValueError(f"起始日期 {start} 不能晚于结束日期 {end}")
