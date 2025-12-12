"""
Factor Base Module - 因子基类与注册机制

该模块提供：
1. BaseFactor 抽象基类 - 所有因子的基类
2. FactorRegistry - 因子注册中心
3. 装饰器注册机制
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from quantclassic.factor_hub.protocols import StandardDataProtocol


@dataclass
class FactorMeta:
    """因子元数据"""
    name: str
    description: str = ""
    category: str = "default"
    params: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    author: str = ""


class BaseFactor(ABC):
    """
    因子抽象基类
    
    所有因子都必须继承此类并实现 compute 方法。
    
    Attributes:
        meta: 因子元数据（名称、描述、参数等）
        
    Example:
        >>> class ReturnFactor(BaseFactor):
        ...     @property
        ...     def meta(self) -> FactorMeta:
        ...         return FactorMeta(
        ...             name="return_1d",
        ...             description="1日收益率",
        ...             category="momentum"
        ...         )
        ...     
        ...     def compute(self, data: StandardDataProtocol) -> pd.Series:
        ...         close = data.close
        ...         return close.groupby(level="symbol").pct_change(1)
    """
    
    def __init__(self, **params: Any) -> None:
        """
        初始化因子
        
        Args:
            **params: 因子参数，会覆盖默认参数
        """
        self._params = {**self.default_params, **params}
    
    @property
    @abstractmethod
    def meta(self) -> FactorMeta:
        """
        因子元数据
        
        子类必须实现此属性，返回因子的元数据信息
        """
        pass
    
    @property
    def name(self) -> str:
        """因子名称"""
        return self.meta.name
    
    @property
    def description(self) -> str:
        """因子描述"""
        return self.meta.description
    
    @property
    def category(self) -> str:
        """因子类别"""
        return self.meta.category
    
    @property
    def default_params(self) -> Dict[str, Any]:
        """
        默认参数
        
        子类可以重写此属性提供默认参数
        """
        return {}
    
    @property
    def params(self) -> Dict[str, Any]:
        """当前参数"""
        return self._params
    
    @abstractmethod
    def compute(self, data: "StandardDataProtocol") -> pd.Series:
        """
        计算因子值
        
        Args:
            data: 符合 StandardDataProtocol 的数据
            
        Returns:
            pd.Series: 因子值，index 应与输入数据对齐
                      通常为 MultiIndex (datetime, symbol)
        """
        pass
    
    def validate_output(self, result: pd.Series) -> bool:
        """
        验证输出结果
        
        Args:
            result: 因子计算结果
            
        Returns:
            bool: 结果有效返回 True
        """
        if not isinstance(result, pd.Series):
            return False
        if result.empty:
            return False
        return True
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"


class FactorRegistry:
    """
    因子注册中心
    
    使用装饰器模式自动注册因子类，支持通过名称获取因子。
    
    Example:
        >>> registry = FactorRegistry()
        >>> 
        >>> @registry.register
        >>> class MyFactor(BaseFactor):
        ...     pass
        >>> 
        >>> # 获取因子类
        >>> factor_cls = registry.get("my_factor")
        >>> factor = factor_cls()
    """
    
    def __init__(self) -> None:
        self._registry: Dict[str, Type[BaseFactor]] = {}
    
    def register(
        self,
        cls: Optional[Type[BaseFactor]] = None,
        *,
        name: Optional[str] = None,
    ):
        """
        注册因子的装饰器
        
        可以直接使用 @registry.register 或 @registry.register(name="custom_name")
        
        Args:
            cls: 因子类
            name: 自定义注册名称，None 则使用因子的 meta.name
        """
        def decorator(factor_cls: Type[BaseFactor]) -> Type[BaseFactor]:
            # 实例化一次以获取 meta
            try:
                instance = factor_cls.__new__(factor_cls)
                # 初始化 _params 属性
                instance._params = {}
                factor_name = name or instance.meta.name
            except Exception:
                # 如果无法实例化，使用类名
                factor_name = name or factor_cls.__name__.lower()
            
            if factor_name in self._registry:
                raise ValueError(f"因子 '{factor_name}' 已经注册")
            
            self._registry[factor_name] = factor_cls
            return factor_cls
        
        # 支持 @register 和 @register(name="xxx") 两种方式
        if cls is not None:
            return decorator(cls)
        return decorator
    
    def get(self, name: str) -> Type[BaseFactor]:
        """
        获取因子类
        
        Args:
            name: 因子名称
            
        Returns:
            因子类
            
        Raises:
            KeyError: 因子不存在
        """
        if name not in self._registry:
            raise KeyError(f"因子 '{name}' 未注册。已注册的因子: {list(self._registry.keys())}")
        return self._registry[name]
    
    def create(self, name: str, **params: Any) -> BaseFactor:
        """
        创建因子实例
        
        Args:
            name: 因子名称
            **params: 因子参数
            
        Returns:
            因子实例
        """
        factor_cls = self.get(name)
        return factor_cls(**params)
    
    def list_factors(self) -> List[str]:
        """列出所有已注册的因子名称"""
        return list(self._registry.keys())
    
    def list_factors_by_category(self) -> Dict[str, List[str]]:
        """按类别列出因子"""
        result: Dict[str, List[str]] = {}
        for name, cls in self._registry.items():
            try:
                instance = cls.__new__(cls)
                instance._params = {}
                category = instance.category
            except Exception:
                category = "unknown"
            
            if category not in result:
                result[category] = []
            result[category].append(name)
        
        return result
    
    def __contains__(self, name: str) -> bool:
        return name in self._registry
    
    def __len__(self) -> int:
        return len(self._registry)
    
    def __repr__(self) -> str:
        return f"FactorRegistry(factors={list(self._registry.keys())})"


# 全局因子注册中心
factor_registry = FactorRegistry()
