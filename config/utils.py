"""
配置工具函数 - 提供类似Qlib的配置实例化功能
"""

import importlib
import re
import sys
from types import ModuleType
from typing import Any, Dict, Tuple, Union


def get_module_by_module_path(module_path: Union[str, ModuleType]):
    """根据模块路径加载模块"""
    if module_path is None:
        raise ModuleNotFoundError("None is passed in as parameters as module_path")
    
    if isinstance(module_path, ModuleType):
        return module_path
    
    if module_path.endswith(".py"):
        module_name = re.sub("^[^a-zA-Z_]+", "", 
                           re.sub("[^0-9a-zA-Z_]", "", 
                                 module_path[:-3].replace("/", "_")))
        module_spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)
    
    return module


def split_module_path(module_path: str) -> Tuple[str, str]:
    """分割模块路径和类名"""
    *m_path, cls = module_path.split(".")
    m_path = ".".join(m_path)
    return m_path, cls


def get_callable_kwargs(config: Union[dict, str], 
                       default_module: Union[str, ModuleType] = None) -> Tuple[type, dict]:
    """从配置中提取可调用对象和参数"""
    if isinstance(config, dict):
        key = "class" if "class" in config else "func"
        
        if isinstance(config[key], str):
            m_path, cls = split_module_path(config[key])
            if m_path == "":
                m_path = config.get("module_path", default_module)
            
            module = get_module_by_module_path(m_path)
            _callable = getattr(module, cls)
        else:
            _callable = config[key]
        
        kwargs = config.get("kwargs", {})
        
    elif isinstance(config, str):
        m_path, cls = split_module_path(config)
        module = get_module_by_module_path(default_module if m_path == "" else m_path)
        _callable = getattr(module, cls)
        kwargs = {}
    else:
        raise NotImplementedError(f"This type of input is not supported: {type(config)}")
    
    return _callable, kwargs


def init_instance_by_config(config: Union[dict, str],
                           default_module: Union[str, ModuleType] = None,
                           accept_types: Union[type, Tuple[type]] = (),
                           **kwargs) -> Any:
    """根据配置初始化实例"""
    if accept_types and isinstance(config, accept_types):
        return config
    
    _callable, _kwargs = get_callable_kwargs(config, default_module)
    _kwargs.update(kwargs)
    
    try:
        return _callable(**_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize {_callable} with kwargs {_kwargs}: {e}"
        ) from e


def update_config(base_config: dict, custom_config: dict) -> dict:
    """递归更新配置（深度合并）"""
    import copy
    result = copy.deepcopy(base_config)
    
    for key, value in custom_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = update_config(result[key], value)
        else:
            result[key] = value
    
    return result
