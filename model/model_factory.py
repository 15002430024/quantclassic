"""
Model Factory - 模型工厂

提供模型注册和动态创建机制，类似 Qlib 的 init_instance_by_config
"""

from typing import Dict, Type, Any, Optional
import importlib
import logging


class ModelRegistry:
    """模型注册表"""
    
    _registry: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type):
        """
        注册模型
        
        Args:
            name: 模型名称
            model_class: 模型类
        """
        cls._registry[name] = model_class
        logging.info(f"✅ 注册模型: {name} -> {model_class.__name__}")
    
    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """
        获取模型类
        
        Args:
            name: 模型名称
            
        Returns:
            模型类，如果不存在返回 None
        """
        return cls._registry.get(name)
    
    @classmethod
    def list_models(cls) -> list:
        """列出所有已注册的模型"""
        return list(cls._registry.keys())
    
    @classmethod
    def clear(cls):
        """清空注册表"""
        cls._registry.clear()


def register_model(name: str):
    """
    模型注册装饰器
    
    Example:
        @register_model('lstm')
        class LSTMModel(Model):
            pass
    
    Args:
        name: 模型名称
    """
    def decorator(model_class: Type):
        ModelRegistry.register(name, model_class)
        return model_class
    return decorator


class ModelFactory:
    """
    模型工厂 - 动态创建模型实例
    
    参照 Qlib 的 init_instance_by_config 设计
    """
    
    @staticmethod
    def create_model(config: Dict[str, Any]):
        """
        根据配置创建模型实例
        
        Args:
            config: 模型配置字典，包含:
                - class: 模型类名
                - module_path: 模型模块路径 (可选)
                - kwargs: 模型参数
        
        Returns:
            模型实例
            
        Example:
            config = {
                'class': 'LSTMModel',
                'module_path': 'quantclassic.model.pytorch_models',
                'kwargs': {
                    'd_feat': 20,
                    'hidden_size': 64,
                    'num_layers': 2
                }
            }
            model = ModelFactory.create_model(config)
        """
        if 'class' not in config:
            raise ValueError("配置中缺少 'class' 字段")
        
        model_name = config['class']
        
        # 1. 尝试从注册表获取
        model_class = ModelRegistry.get(model_name)
        
        # 2. 如果注册表中没有，尝试从模块路径导入
        if model_class is None and 'module_path' in config:
            module_path = config['module_path']
            try:
                module = importlib.import_module(module_path)
                model_class = getattr(module, model_name)
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"无法从 {module_path} 导入 {model_name}: {e}"
                )
        
        # 3. 如果还是找不到，抛出异常
        if model_class is None:
            raise ValueError(
                f"找不到模型 '{model_name}'。"
                f"已注册的模型: {ModelRegistry.list_models()}"
            )
        
        # 4. 创建实例
        kwargs = config.get('kwargs', {})
        
        try:
            model = model_class(**kwargs)
            logging.info(f"✅ 创建模型: {model_name}")
            return model
        except Exception as e:
            raise RuntimeError(
                f"创建模型 {model_name} 失败: {e}"
            )
    
    @staticmethod
    def create_from_yaml(yaml_path: str):
        """
        从 YAML 配置文件创建模型
        
        Args:
            yaml_path: YAML 配置文件路径
            
        Returns:
            模型实例
        """
        import yaml
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'model' not in config:
            raise ValueError("YAML 配置中缺少 'model' 字段")
        
        return ModelFactory.create_model(config['model'])


def init_instance_by_config(config: Dict[str, Any]):
    """
    根据配置初始化实例（兼容 Qlib 接口）
    
    Args:
        config: 配置字典
        
    Returns:
        实例对象
    """
    return ModelFactory.create_model(config)


if __name__ == '__main__':
    print("=" * 80)
    print("Model Factory 测试")
    print("=" * 80)
    
    # 测试注册机制
    from base_model import Model
    
    @register_model('test_model')
    class TestModel(Model):
        def __init__(self, param1, param2):
            super().__init__()
            self.param1 = param1
            self.param2 = param2
        
        def fit(self, train_data, valid_data=None, **kwargs):
            print(f"Training with {self.param1}, {self.param2}")
            self.fitted = True
        
        def predict(self, test_data, **kwargs):
            return "predictions"
    
    # 测试创建模型
    config = {
        'class': 'test_model',
        'kwargs': {
            'param1': 'value1',
            'param2': 'value2'
        }
    }
    
    model = ModelFactory.create_model(config)
    print(f"\n✅ 创建模型成功: {model}")
    print(f"✅ 参数: param1={model.param1}, param2={model.param2}")
    
    # 测试列出模型
    print(f"\n已注册的模型: {ModelRegistry.list_models()}")
    
    print("\n✅ Model Factory 测试完成")
