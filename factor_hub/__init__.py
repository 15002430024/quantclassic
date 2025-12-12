"""
FactorHub - A Quantitative Factor Computation Framework

This framework provides an end-to-end pipeline for factor computation:
DataProvider Adapter → Standardized Data Protocol → Factor Calculation Engine → Factor Output

Author: Quant Architect
Version: 1.0.0
"""

from factor_hub.protocols.data_protocol import StandardDataProtocol
from factor_hub.providers.base import IDataProvider
from factor_hub.providers.mock_provider import MockDataProvider
from factor_hub.factors.base import BaseFactor
from factor_hub.factors.registry import FactorRegistry, factor_registry
from factor_hub.engine.factor_engine import FactorEngine
from factor_hub.io.writers import IFactorWriter, CSVWriter, ParquetWriter

__version__ = "1.0.0"

__all__ = [
    # Protocols
    "StandardDataProtocol",
    # Providers
    "IDataProvider",
    "MockDataProvider",
    # Factors
    "BaseFactor",
    "FactorRegistry",
    "factor_registry",
    # Engine
    "FactorEngine",
    # IO
    "IFactorWriter",
    "CSVWriter",
    "ParquetWriter",
]
