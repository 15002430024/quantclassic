"""
ConfigLoader - 统一配置加载器

支持：
1. YAML 文件加载
2. 面向对象配置类
3. 字典配置（向后兼容）
4. 配置继承 (BASE_CONFIG_PATH)
5. 环境变量替换
6. 配置验证
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type
import yaml

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from base_config import BaseConfig


class ConfigLoader:
    """
    统一配置加载器
    
    特性：
    1. 自动识别配置类型（面向对象 vs 字典）
    2. 支持配置继承和合并
    3. 支持环境变量替换
    4. 支持配置验证
    
    Example:
        # 方式1: 加载为配置对象
        from model.model_config import VAEConfig
        config = ConfigLoader.load('config.yaml', VAEConfig)
        
        # 方式2: 加载为字典（向后兼容）
        config_dict = ConfigLoader.load('config.yaml')
    """
    
    @staticmethod
    def load(
        config_path: str,
        config_class: Optional[Type[BaseConfig]] = None,
        return_dict: bool = False
    ) -> Union[BaseConfig, Dict[str, Any]]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            config_class: 配置类（如果指定，返回配置对象；否则返回字典）
            return_dict: 是否强制返回字典（用于向后兼容）
            
        Returns:
            配置对象或配置字典
            
        Example:
            # 加载为对象
            from model.model_config import VAEConfig
            config = ConfigLoader.load('config.yaml', VAEConfig)
            
            # 加载为字典
            config_dict = ConfigLoader.load('config.yaml', return_dict=True)
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 加载 YAML
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            config_dict = {}
        
        # 处理基础配置继承
        base_config_path = config_dict.get('BASE_CONFIG_PATH')
        if base_config_path:
            config_dict = ConfigLoader._load_with_base(config_path, base_config_path, config_dict)
        
        # 环境变量替换
        config_dict = ConfigLoader._replace_env_vars(config_dict)
        
        # 决定返回类型
        if return_dict or config_class is None:
            # 返回字典（向后兼容）
            return config_dict
        else:
            # 返回配置对象
            if not issubclass(config_class, BaseConfig):
                raise TypeError(f"{config_class} 必须继承自 BaseConfig")
            
            return config_class.from_dict(config_dict)
    
    @staticmethod
    def _load_with_base(config_path: Path, base_config_path: str, config: dict) -> dict:
        """
        加载并合并基础配置
        
        Args:
            config_path: 当前配置文件路径
            base_config_path: 基础配置路径（相对或绝对）
            config: 当前配置
            
        Returns:
            合并后的配置
        """
        from .utils import update_config
        
        base_path = Path(base_config_path)
        
        # 尝试绝对路径
        if base_path.exists():
            path = base_path
        else:
            # 尝试相对于配置文件的路径
            relative_path = config_path.parent / base_path
            if relative_path.exists():
                path = relative_path
            else:
                raise FileNotFoundError(
                    f"找不到基础配置文件: {base_config_path}\n"
                    f"已尝试: {base_path}, {relative_path}"
                )
        
        # 加载基础配置
        with open(path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        
        # 合并配置（当前配置覆盖基础配置）
        merged_config = update_config(base_config, config)
        
        # 移除BASE_CONFIG_PATH字段
        merged_config.pop('BASE_CONFIG_PATH', None)
        
        return merged_config
    
    @staticmethod
    def _replace_env_vars(config: Any) -> Any:
        """
        递归替换配置中的环境变量
        
        支持的格式:
            - ${ENV_VAR}
            - ${ENV_VAR:default_value}
        
        Args:
            config: 配置对象（dict, list, str等）
            
        Returns:
            替换后的配置
        """
        import re
        
        if isinstance(config, dict):
            return {k: ConfigLoader._replace_env_vars(v) for k, v in config.items()}
        
        elif isinstance(config, list):
            return [ConfigLoader._replace_env_vars(item) for item in config]
        
        elif isinstance(config, str):
            # 匹配 ${VAR} 或 ${VAR:default}
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
            
            def replacer(match):
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) is not None else ''
                return os.environ.get(var_name, default_value)
            
            return re.sub(pattern, replacer, config)
        
        else:
            return config
    
    @staticmethod
    def validate(config: Dict[str, Any]) -> bool:
        """
        验证配置的完整性
        
        Args:
            config: 配置字典
            
        Returns:
            是否有效
        """
        # 必需的顶层字段
        if 'task' not in config:
            raise ValueError("配置中缺少 'task' 字段")
        
        task = config['task']
        
        # 检查task的基本结构
        if 'model' not in task and 'dataset' not in task:
            raise ValueError("task中至少需要包含 'model' 或 'dataset'")
        
        return True
    
    @staticmethod
    def save(
        config: Union[BaseConfig, Dict[str, Any]],
        save_path: str
    ):
        """
        保存配置到 YAML 文件
        
        支持保存配置对象或字典
        
        Args:
            config: 配置对象或配置字典
            save_path: 保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为字典
        if isinstance(config, BaseConfig):
            config_dict = config.to_dict()
        elif hasattr(config, 'to_dict'):
            # 鸭子类型：如果有 to_dict 方法就调用
            config_dict = config.to_dict()
        else:
            config_dict = config
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config_dict, f, 
                         default_flow_style=False, 
                         allow_unicode=True,
                         sort_keys=False)


if __name__ == "__main__":
    # 测试 ConfigLoader
    print("=== 测试 ConfigLoader ===")
    
    # 创建测试配置
    test_config = {
        'quantclassic_init': {
            'log_level': 'INFO'
        },
        'task': {
            'model': {
                'class': 'LSTM',
                'module_path': 'quantclassic.model.pytorch_models',
                'kwargs': {
                    'd_feat': 20,
                    'hidden_size': 64
                }
            }
        }
    }
    
    # 保存测试配置
    test_file = 'output/test_config_loader.yaml'
    ConfigLoader.save(test_config, test_file)
    print(f"✅ 保存测试配置到: {test_file}")
    
    # 测试1: 加载为字典（向后兼容）
    loaded_dict = ConfigLoader.load(test_file, return_dict=True)
    print(f"✅ 加载为字典成功: {type(loaded_dict)}")
    
    # 测试2: 使用面向对象配置
    from model.model_config import VAEConfig
    
    vae_config = VAEConfig(
        hidden_dim=128,
        latent_dim=16,
        n_epochs=100,
    )
    
    vae_file = 'output/test_vae_config.yaml'
    ConfigLoader.save(vae_config, vae_file)
    print(f"✅ 保存 VAE 配置到: {vae_file}")
    
    # 加载 VAE 配置
    loaded_vae = ConfigLoader.load(vae_file, VAEConfig)
    print(f"✅ 加载 VAE 配置成功: latent_dim={loaded_vae.latent_dim}")
    
    # 测试3: 环境变量替换
    config_with_env = {
        'path': '${HOME}/data',
        'user': '${USER:default_user}'
    }
    replaced = ConfigLoader._replace_env_vars(config_with_env)
    print(f"✅ 环境变量替换测试: {replaced}")
    
    print("\n所有测试完成")
