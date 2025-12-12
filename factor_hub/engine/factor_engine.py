"""
Factor Engine - 因子计算引擎核心模块

该模块是 FactorHub 的核心调度器，负责：
1. 连接数据提供者和因子算法
2. 执行因子计算流水线
3. 处理异常和错误恢复
4. 拼接和输出因子结果
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass, field

import pandas as pd

from quantclassic.factor_hub.protocols import StandardDataProtocol, DataValidationError
from quantclassic.factor_hub.providers import IDataProvider
from quantclassic.factor_hub.factors import BaseFactor, factor_registry

if TYPE_CHECKING:
    from quantclassic.factor_hub.io.writers import IFactorWriter


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FactorComputeResult:
    """因子计算结果"""
    factor_name: str
    success: bool
    data: Optional[pd.Series] = None
    error: Optional[str] = None
    compute_time: float = 0.0


@dataclass
class EngineRunResult:
    """引擎运行结果"""
    success: bool
    factors_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    factor_results: List[FactorComputeResult] = field(default_factory=list)
    total_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    @property
    def successful_factors(self) -> List[str]:
        """成功计算的因子列表"""
        return [r.factor_name for r in self.factor_results if r.success]
    
    @property
    def failed_factors(self) -> List[str]:
        """失败的因子列表"""
        return [r.factor_name for r in self.factor_results if not r.success]


class FactorEngine:
    """
    因子计算引擎
    
    核心功能：
    1. 接收 IDataProvider 实例（依赖注入）
    2. 从 Registry 获取因子类
    3. 执行因子计算流水线
    4. 处理异常，防止单个因子报错中断整个流程
    5. 拼接因子结果
    
    Example:
        >>> from factor_hub import FactorEngine, MockDataProvider
        >>> 
        >>> provider = MockDataProvider()
        >>> engine = FactorEngine(provider)
        >>> 
        >>> result = engine.run(
        ...     symbols=["000001.SZ", "600000.SH"],
        ...     factor_names=["return_1d", "volatility"],
        ...     start="2024-01-01",
        ...     end="2024-03-01"
        ... )
        >>> print(result.factors_data)
    """
    
    def __init__(
        self,
        data_provider: IDataProvider,
        continue_on_error: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        初始化因子计算引擎
        
        Args:
            data_provider: 数据提供者实例（依赖注入）
            continue_on_error: 单个因子报错时是否继续执行其他因子
            verbose: 是否打印详细日志
        """
        self._provider = data_provider
        self._continue_on_error = continue_on_error
        self._verbose = verbose
        self._writer: Optional["IFactorWriter"] = None
        
        # 验证 provider
        if not isinstance(data_provider, IDataProvider):
            raise TypeError(
                f"data_provider 必须是 IDataProvider 的实例，"
                f"当前类型: {type(data_provider)}"
            )
    
    @property
    def provider(self) -> IDataProvider:
        """数据提供者"""
        return self._provider
    
    def set_writer(self, writer: "IFactorWriter") -> None:
        """
        设置因子输出写入器
        
        Args:
            writer: 实现 IFactorWriter 接口的写入器
        """
        self._writer = writer
    
    def run(
        self,
        symbols: Union[str, List[str]],
        factor_names: List[str],
        start: Union[str, datetime, pd.Timestamp],
        end: Union[str, datetime, pd.Timestamp],
        factor_params: Optional[Dict[str, Dict[str, Any]]] = None,
        save_path: Optional[str] = None,
    ) -> EngineRunResult:
        """
        执行因子计算流水线
        
        Args:
            symbols: 股票代码列表
            factor_names: 因子名称列表
            start: 起始日期
            end: 结束日期
            factor_params: 因子参数，格式 {factor_name: {param_name: value}}
            save_path: 保存路径，None 则不保存
            
        Returns:
            EngineRunResult: 包含计算结果和状态信息
        """
        import time
        start_time = time.time()
        
        result = EngineRunResult(success=True)
        factor_params = factor_params or {}
        
        self._log(f"=" * 60)
        self._log(f"FactorEngine 开始运行")
        self._log(f"=" * 60)
        
        # Step A: 调用 Provider 获取 Raw Data
        self._log(f"\n[Step A] 获取原始数据...")
        try:
            raw_data = self._fetch_data(symbols, start, end)
            self._log(f"    ✓ 获取 {len(raw_data)} 行数据")
        except Exception as e:
            error_msg = f"数据获取失败: {e}"
            self._log(f"    ✗ {error_msg}")
            result.success = False
            result.errors.append(error_msg)
            return result
        
        # Step B: 校验数据格式
        self._log(f"\n[Step B] 校验数据格式...")
        try:
            std_data = self._validate_data(raw_data)
            self._log(f"    ✓ 数据校验通过")
            self._log(f"    ✓ 股票数量: {len(std_data.symbols)}")
            self._log(f"    ✓ 日期范围: {std_data.start_date.date()} ~ {std_data.end_date.date()}")
        except DataValidationError as e:
            error_msg = f"数据校验失败: {e}"
            self._log(f"    ✗ {error_msg}")
            result.success = False
            result.errors.append(error_msg)
            return result
        
        # Step C & D: 遍历因子，实例化并计算
        self._log(f"\n[Step C & D] 计算因子...")
        factor_series_list: List[pd.Series] = []
        
        for factor_name in factor_names:
            self._log(f"\n    计算因子: {factor_name}")
            
            factor_result = self._compute_single_factor(
                factor_name=factor_name,
                data=std_data,
                params=factor_params.get(factor_name, {}),
            )
            
            result.factor_results.append(factor_result)
            
            if factor_result.success and factor_result.data is not None:
                self._log(f"        ✓ 成功，非空值数量: {factor_result.data.notna().sum()}")
                factor_series_list.append(factor_result.data)
            else:
                self._log(f"        ✗ 失败: {factor_result.error}")
                result.errors.append(f"{factor_name}: {factor_result.error}")
                
                if not self._continue_on_error:
                    result.success = False
                    break
        
        # Step E: 拼接所有因子结果
        self._log(f"\n[Step E] 拼接因子结果...")
        if factor_series_list:
            result.factors_data = self._concat_factors(factor_series_list)
            self._log(f"    ✓ 结果 DataFrame shape: {result.factors_data.shape}")
        else:
            self._log(f"    ⚠ 没有成功计算的因子")
        
        # 可选：保存结果
        if save_path and not result.factors_data.empty:
            self._log(f"\n[Step F] 保存结果...")
            try:
                self._save_result(result.factors_data, save_path)
                self._log(f"    ✓ 结果已保存到: {save_path}")
            except Exception as e:
                self._log(f"    ✗ 保存失败: {e}")
                result.errors.append(f"保存失败: {e}")
        
        # 汇总
        result.total_time = time.time() - start_time
        result.success = len(result.failed_factors) == 0
        
        self._log(f"\n" + "=" * 60)
        self._log(f"运行完成!")
        self._log(f"    成功因子: {result.successful_factors}")
        self._log(f"    失败因子: {result.failed_factors}")
        self._log(f"    总耗时: {result.total_time:.2f}s")
        self._log(f"=" * 60)
        
        return result
    
    def _fetch_data(
        self,
        symbols: Union[str, List[str]],
        start: Union[str, datetime, pd.Timestamp],
        end: Union[str, datetime, pd.Timestamp],
    ) -> pd.DataFrame:
        """获取原始数据"""
        return self._provider.get_history(symbols, start, end)
    
    def _validate_data(self, raw_data: pd.DataFrame) -> StandardDataProtocol:
        """校验并标准化数据"""
        return StandardDataProtocol(raw_data, auto_validate=True, auto_standardize=True)
    
    def _compute_single_factor(
        self,
        factor_name: str,
        data: StandardDataProtocol,
        params: Dict[str, Any],
    ) -> FactorComputeResult:
        """
        计算单个因子
        
        使用 try-catch 防止单个因子报错影响其他因子
        """
        import time
        start_time = time.time()
        
        try:
            # 从 Registry 获取因子类
            factor_cls = factor_registry.get(factor_name)
            
            # 实例化因子
            factor = factor_cls(**params)
            
            # 执行计算
            factor_values = factor.compute(data)
            
            # 确保结果有正确的名称
            factor_values.name = factor_name
            
            compute_time = time.time() - start_time
            
            return FactorComputeResult(
                factor_name=factor_name,
                success=True,
                data=factor_values,
                compute_time=compute_time,
            )
            
        except KeyError as e:
            return FactorComputeResult(
                factor_name=factor_name,
                success=False,
                error=f"因子未注册: {e}",
            )
        except Exception as e:
            return FactorComputeResult(
                factor_name=factor_name,
                success=False,
                error=str(e),
            )
    
    def _concat_factors(self, factor_series_list: List[pd.Series]) -> pd.DataFrame:
        """拼接所有因子结果"""
        if not factor_series_list:
            return pd.DataFrame()
        
        # 使用 concat 拼接，每个 Series 成为一列
        result = pd.concat(factor_series_list, axis=1)
        
        return result
    
    def _save_result(self, data: pd.DataFrame, save_path: str) -> None:
        """保存结果"""
        if self._writer is not None:
            self._writer.write(data, save_path)
        else:
            # 默认根据扩展名选择格式
            if save_path.endswith('.parquet'):
                data.to_parquet(save_path)
            elif save_path.endswith('.csv'):
                data.to_csv(save_path)
            else:
                data.to_csv(save_path + '.csv')
    
    def _log(self, message: str) -> None:
        """打印日志"""
        if self._verbose:
            print(message)
    
    def list_available_factors(self) -> List[str]:
        """列出所有可用的因子"""
        return factor_registry.list_factors()
    
    def __repr__(self) -> str:
        return (
            f"FactorEngine(\n"
            f"  provider={self._provider},\n"
            f"  available_factors={self.list_available_factors()}\n"
            f")"
        )
