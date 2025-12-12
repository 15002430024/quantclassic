"""
FactorHub - A Quantitative Factor Computation Framework

This framework provides an end-to-end pipeline for factor computation:
DataProvider Adapter → Standardized Data Protocol → Factor Calculation Engine → Factor Output

Author: Quant Architect
Version: 1.0.0
"""

from quantclassic.factor_hub.protocols import (
    StandardDataProtocol,
    DataValidationError,
    DataColumnSpec,
    COLUMN_SPEC,
)
from quantclassic.factor_hub.providers import (
    IDataProvider,
    BaseDataAdapter,
    DataFetchError,
)
from quantclassic.factor_hub.providers.mock_provider import MockDataProvider
from quantclassic.factor_hub.factors import (
    BaseFactor,
    FactorMeta,
    FactorRegistry,
    factor_registry,
)
from quantclassic.factor_hub.engine.factor_engine import FactorEngine
from quantclassic.factor_hub.io.writers import IFactorWriter, CSVWriter, ParquetWriter

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
