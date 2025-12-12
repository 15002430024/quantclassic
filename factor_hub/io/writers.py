"""
Factor Writers - 因子结果输出写入器

提供多种格式的因子结果输出支持。
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class IFactorWriter(ABC):
    """
    因子输出写入器接口
    
    所有写入器都必须实现这个接口。
    """
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """输出格式名称"""
        pass
    
    @property
    def file_extension(self) -> str:
        """文件扩展名"""
        return ""
    
    @abstractmethod
    def write(
        self,
        data: pd.DataFrame,
        path: str,
        **kwargs: Any,
    ) -> str:
        """
        写入因子数据
        
        Args:
            data: 因子数据 DataFrame
            path: 输出路径
            **kwargs: 额外参数
            
        Returns:
            str: 实际保存的路径
        """
        pass
    
    def _ensure_dir(self, path: str) -> None:
        """确保目录存在"""
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
    
    def _add_extension(self, path: str) -> str:
        """添加文件扩展名（如果没有）"""
        if self.file_extension and not path.endswith(self.file_extension):
            return path + self.file_extension
        return path


class CSVWriter(IFactorWriter):
    """
    CSV 格式写入器
    
    Example:
        >>> writer = CSVWriter()
        >>> writer.write(df, "/path/to/factors.csv")
    """
    
    def __init__(
        self,
        index: bool = True,
        encoding: str = "utf-8",
        float_format: Optional[str] = "%.6f",
    ) -> None:
        """
        初始化 CSV 写入器
        
        Args:
            index: 是否写入索引
            encoding: 文件编码
            float_format: 浮点数格式
        """
        self._index = index
        self._encoding = encoding
        self._float_format = float_format
    
    @property
    def format_name(self) -> str:
        return "CSV"
    
    @property
    def file_extension(self) -> str:
        return ".csv"
    
    def write(
        self,
        data: pd.DataFrame,
        path: str,
        **kwargs: Any,
    ) -> str:
        """写入 CSV 文件"""
        path = self._add_extension(path)
        self._ensure_dir(path)
        
        data.to_csv(
            path,
            index=self._index,
            encoding=self._encoding,
            float_format=self._float_format,
            **kwargs,
        )
        
        return path


class ParquetWriter(IFactorWriter):
    """
    Parquet 格式写入器
    
    Parquet 是一种高效的列式存储格式，适合大数据量场景。
    
    Example:
        >>> writer = ParquetWriter()
        >>> writer.write(df, "/path/to/factors.parquet")
    """
    
    def __init__(
        self,
        compression: str = "snappy",
        index: bool = True,
    ) -> None:
        """
        初始化 Parquet 写入器
        
        Args:
            compression: 压缩算法 (snappy, gzip, brotli, None)
            index: 是否写入索引
        """
        self._compression = compression
        self._index = index
    
    @property
    def format_name(self) -> str:
        return "Parquet"
    
    @property
    def file_extension(self) -> str:
        return ".parquet"
    
    def write(
        self,
        data: pd.DataFrame,
        path: str,
        **kwargs: Any,
    ) -> str:
        """写入 Parquet 文件"""
        path = self._add_extension(path)
        self._ensure_dir(path)
        
        data.to_parquet(
            path,
            compression=self._compression,
            index=self._index,
            **kwargs,
        )
        
        return path


class PickleWriter(IFactorWriter):
    """
    Pickle 格式写入器
    
    适合保存 Python 对象的完整状态。
    """
    
    @property
    def format_name(self) -> str:
        return "Pickle"
    
    @property
    def file_extension(self) -> str:
        return ".pkl"
    
    def write(
        self,
        data: pd.DataFrame,
        path: str,
        **kwargs: Any,
    ) -> str:
        """写入 Pickle 文件"""
        path = self._add_extension(path)
        self._ensure_dir(path)
        
        data.to_pickle(path, **kwargs)
        
        return path


class FactorWriterFactory:
    """
    写入器工厂
    
    根据格式名称或文件扩展名创建对应的写入器。
    
    Example:
        >>> writer = FactorWriterFactory.create("csv")
        >>> writer = FactorWriterFactory.from_path("/path/to/file.parquet")
    """
    
    _writers = {
        "csv": CSVWriter,
        "parquet": ParquetWriter,
        "pickle": PickleWriter,
        "pkl": PickleWriter,
    }
    
    @classmethod
    def create(cls, format_name: str, **kwargs: Any) -> IFactorWriter:
        """
        根据格式名称创建写入器
        
        Args:
            format_name: 格式名称 (csv, parquet, pickle)
            **kwargs: 传递给写入器的参数
        """
        format_name = format_name.lower()
        
        if format_name not in cls._writers:
            raise ValueError(
                f"不支持的格式: {format_name}。"
                f"支持的格式: {list(cls._writers.keys())}"
            )
        
        return cls._writers[format_name](**kwargs)
    
    @classmethod
    def from_path(cls, path: str, **kwargs: Any) -> IFactorWriter:
        """
        根据文件路径（扩展名）创建写入器
        
        Args:
            path: 文件路径
            **kwargs: 传递给写入器的参数
        """
        ext = Path(path).suffix.lower().lstrip(".")
        
        if not ext:
            # 默认使用 CSV
            ext = "csv"
        
        return cls.create(ext, **kwargs)
    
    @classmethod
    def register(cls, format_name: str, writer_class: type) -> None:
        """注册新的写入器"""
        cls._writers[format_name.lower()] = writer_class
