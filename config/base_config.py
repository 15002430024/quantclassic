"""
BaseConfig - 统一配置基类

提供面向对象的配置管理框架，替代字典配置
核心功能：
- 类型检查和验证
- 序列化/反序列化（YAML, JSON, Dict）
- 配置继承和合并
- 默认值管理
"""

from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Any, Optional, Type, TypeVar, get_type_hints
from pathlib import Path
import yaml
import json
from abc import ABC


T = TypeVar('T', bound='BaseConfig')


@dataclass
class BaseConfig(ABC):
    """
    配置基类 - 所有配置类的基础
    
    特性：
    1. 使用 dataclass 提供类型检查和默认值
    2. 支持 YAML/JSON 序列化
    3. 支持配置验证
    4. 支持嵌套配置对象
    
    Example:
        @dataclass
        class ModelConfig(BaseConfig):
            hidden_dim: int = 128
            learning_rate: float = 0.001
            
        config = ModelConfig()
        config.to_yaml('config.yaml')
        config2 = ModelConfig.from_yaml('config.yaml')
    """
    
    def validate(self) -> bool:
        """
        验证配置的有效性
        
        子类可重写此方法以实现自定义验证逻辑
        
        Returns:
            是否有效
            
        Raises:
            ValueError: 配置无效时
        """
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        递归处理嵌套的 BaseConfig 对象
        
        Returns:
            配置字典
        """
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            
            if isinstance(value, BaseConfig):
                # 递归处理嵌套配置
                result[f.name] = value.to_dict()
            elif isinstance(value, list):
                # 处理列表中的配置对象
                result[f.name] = [
                    item.to_dict() if isinstance(item, BaseConfig) else item
                    for item in value
                ]
            elif isinstance(value, dict):
                # 处理字典中的配置对象
                result[f.name] = {
                    k: v.to_dict() if isinstance(v, BaseConfig) else v
                    for k, v in value.items()
                }
            else:
                result[f.name] = value
        
        return result
    
    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """
        从字典创建配置对象
        
        支持嵌套配置对象的自动实例化
        
        Args:
            config_dict: 配置字典
            
        Returns:
            配置对象实例
        """
        # 获取类的类型注解
        type_hints = get_type_hints(cls)
        
        # 准备初始化参数
        init_kwargs = {}
        
        for key, value in config_dict.items():
            if key not in type_hints:
                # 跳过未定义的字段
                continue
            
            field_type = type_hints[key]
            
            # 检查是否为 BaseConfig 子类
            if isinstance(field_type, type) and issubclass(field_type, BaseConfig):
                # 递归创建嵌套配置对象
                if isinstance(value, dict):
                    init_kwargs[key] = field_type.from_dict(value)
                else:
                    init_kwargs[key] = value
            else:
                init_kwargs[key] = value
        
        # 创建实例
        instance = cls(**init_kwargs)
        
        # 验证配置
        instance.validate()
        
        return instance
    
    def to_yaml(self, yaml_path: str, **kwargs):
        """
        保存配置到 YAML 文件
        
        Args:
            yaml_path: YAML 文件路径
            **kwargs: yaml.dump 的额外参数
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                config_dict, f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                **kwargs
            )
    
    @classmethod
    def from_yaml(cls: Type[T], yaml_path: str) -> T:
        """
        从 YAML 文件加载配置
        
        Args:
            yaml_path: YAML 文件路径
            
        Returns:
            配置对象实例
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    def to_json(self, json_path: str, **kwargs):
        """
        保存配置到 JSON 文件
        
        Args:
            json_path: JSON 文件路径
            **kwargs: json.dump 的额外参数
        """
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False, **kwargs)
    
    @classmethod
    def from_json(cls: Type[T], json_path: str) -> T:
        """
        从 JSON 文件加载配置
        
        Args:
            json_path: JSON 文件路径
            
        Returns:
            配置对象实例
        """
        json_path = Path(json_path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def update(self, **kwargs):
        """
        更新配置参数
        
        Args:
            **kwargs: 要更新的参数
            
        Raises:
            ValueError: 参数不存在时
        """
        valid_fields = {f.name for f in fields(self)}
        
        for key, value in kwargs.items():
            if key not in valid_fields:
                raise ValueError(f"未知配置项: {key}")
            setattr(self, key, value)
        
        # 更新后重新验证
        self.validate()
    
    def merge(self: T, other: T) -> T:
        """
        合并另一个配置对象
        
        other 的非 None 值会覆盖当前配置
        
        Args:
            other: 另一个配置对象
            
        Returns:
            合并后的新配置对象
        """
        if not isinstance(other, self.__class__):
            raise TypeError(f"只能合并相同类型的配置，期望 {self.__class__}，得到 {type(other)}")
        
        # 创建当前配置的副本
        merged_dict = self.to_dict()
        
        # 合并 other 的非 None 值
        for f in fields(other):
            value = getattr(other, f.name)
            if value is not None:
                merged_dict[f.name] = value
        
        return self.__class__.from_dict(merged_dict)
    
    def __repr__(self) -> str:
        """友好的字符串表示"""
        lines = [f"{self.__class__.__name__}("]
        for f in fields(self):
            value = getattr(self, f.name)
            lines.append(f"  {f.name}={repr(value)},")
        lines.append(")")
        return "\n".join(lines)
    
    def __str__(self) -> str:
        """简洁的字符串表示"""
        field_strs = [f"{f.name}={getattr(self, f.name)}" for f in fields(self)]
        return f"{self.__class__.__name__}({', '.join(field_strs)})"


@dataclass
class TaskConfig(BaseConfig):
    """
    任务配置 - 定义模型和数据集
    
    这是 QuantClassic 任务的顶层配置
    
    Args:
        model_class (str): 模型类名，如 'HybridGraphModel'
        model_kwargs (Dict): 模型初始化参数
        dataset_class (str): 数据集类名，如 'DataManager'
        dataset_kwargs (Dict): 数据集初始化参数
        trainer_class (str): 训练器类名，可选值:
            - '' (默认): 使用模型自带的 fit 方法
            - 'RollingDailyTrainer': 滚动窗口训练
            - 'DynamicGraphTrainer': 动态图训练
        trainer_kwargs (Dict): 训练器初始化参数
        use_rolling_loaders (bool): 是否使用滚动窗口加载器
        backtest_enabled (bool): 是否启用回测
        backtest_kwargs (Dict): 回测参数
    """
    # 模型配置
    model_class: str = ""
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # 数据集配置
    dataset_class: str = ""
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # 🆕 训练器配置 - 支持 RollingDailyTrainer / DynamicGraphTrainer
    trainer_class: str = ""  # '' 表示使用默认训练, 'RollingDailyTrainer' / 'DynamicGraphTrainer'
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # 🆕 是否使用滚动窗口日批次加载器
    use_rolling_loaders: bool = False
    
    # 🆕 是否使用日批次加载器（动态图模式）
    use_daily_loaders: bool = False
    
    # 回测配置（可选）
    backtest_enabled: bool = False
    backtest_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """验证任务配置"""
        if not self.model_class:
            raise ValueError("model_class 不能为空")
        
        if not self.dataset_class:
            raise ValueError("dataset_class 不能为空")
        
        # 验证训练器类名
        valid_trainers = ['', 'RollingDailyTrainer', 'DynamicGraphTrainer']
        if self.trainer_class and self.trainer_class not in valid_trainers:
            raise ValueError(f"不支持的训练器: {self.trainer_class}，可选: {valid_trainers}")
        
        return True


if __name__ == '__main__':
    # 测试 BaseConfig
    print("=" * 80)
    print("BaseConfig 测试")
    print("=" * 80)
    
    # 定义测试配置类
    @dataclass
    class TestModelConfig(BaseConfig):
        hidden_dim: int = 128
        learning_rate: float = 0.001
        dropout: float = 0.3
        
        def validate(self) -> bool:
            if self.hidden_dim <= 0:
                raise ValueError("hidden_dim 必须大于 0")
            if not 0 <= self.dropout <= 1:
                raise ValueError("dropout 必须在 [0, 1] 范围内")
            return True
    
    @dataclass
    class TestDataConfig(BaseConfig):
        batch_size: int = 256
        window_size: int = 40
    
    @dataclass
    class TestTaskConfig(BaseConfig):
        name: str = "test_task"
        model: TestModelConfig = field(default_factory=TestModelConfig)
        data: TestDataConfig = field(default_factory=TestDataConfig)
    
    # 测试 1: 创建配置
    print("\n1. 创建配置对象:")
    config = TestTaskConfig()
    print(f"  {config.name}")
    print(f"  模型: hidden_dim={config.model.hidden_dim}")
    print(f"  数据: batch_size={config.data.batch_size}")
    
    # 测试 2: 转换为字典
    print("\n2. 转换为字典:")
    config_dict = config.to_dict()
    print(f"  keys: {list(config_dict.keys())}")
    print(f"  model.hidden_dim: {config_dict['model']['hidden_dim']}")
    
    # 测试 3: 从字典创建
    print("\n3. 从字典创建:")
    config2 = TestTaskConfig.from_dict(config_dict)
    print(f"  name: {config2.name}")
    print(f"  model.learning_rate: {config2.model.learning_rate}")
    
    # 测试 4: 保存和加载 YAML
    print("\n4. YAML 序列化:")
    yaml_path = '/tmp/test_config.yaml'
    config.to_yaml(yaml_path)
    print(f"  已保存到: {yaml_path}")
    
    config3 = TestTaskConfig.from_yaml(yaml_path)
    print(f"  已加载: {config3.name}")
    
    # 测试 5: 更新配置
    print("\n5. 更新配置:")
    config.update(name='updated_task')
    print(f"  新名称: {config.name}")
    
    # 测试 6: 合并配置
    print("\n6. 合并配置:")
    other_config = TestTaskConfig(name='merged_task')
    merged = config.merge(other_config)
    print(f"  合并后名称: {merged.name}")
    
    # 测试 7: 验证
    print("\n7. 配置验证:")
    try:
        invalid_config = TestModelConfig(hidden_dim=-10)
    except ValueError as e:
        print(f"  ✅ 捕获到验证错误: {e}")
    
    print("\n" + "=" * 80)
    print("✅ BaseConfig 测试完成")
    print("=" * 80)
