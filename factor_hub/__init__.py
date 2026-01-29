"""
FactorHub - A Quantitative Factor Computation Framework

⚠️ **实验性模块，不用于生产** ⚠️

本模块为因子计算原型/实验平台，不建议用于生产环境。
生产级因子生成/处理/回测请使用 `quantclassic.backtest` 模块：
- MultiFactorBacktest（推荐，多因子）
- FactorBacktestSystem（端到端模型推理）

迁移指南：
- FactorEngine → backtest.FactorGenerator + backtest.FactorProcessor
- 因子计算结果可直接传给 MultiFactorBacktest.run()

This framework provides an end-to-end pipeline for factor computation:
DataProvider Adapter → Standardized Data Protocol → Factor Calculation Engine → Factor Output

Author: Quant Architect
Version: 1.0.0 (Experimental)
"""

import warnings

warnings.warn(
    "factor_hub 为实验性模块，不建议用于生产。"
    "请使用 quantclassic.backtest 作为生产因子回测入口。",
    category=FutureWarning,
    stacklevel=2
)

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
