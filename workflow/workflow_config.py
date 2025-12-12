"""
WorkflowConfig - 工作流配置类

使用面向对象的配置管理实验和记录器
"""

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.base_config import BaseConfig


@dataclass
class RecorderConfig(BaseConfig):
    """
    记录器配置

    管理实验记录器参数，用于记录超参数、指标、工件等实验信息。

    Args:
        experiment_name (str): 实验名称，默认 'default_experiment'。
            用于标识不同的实验任务。
            
        recorder_name (str): 记录器名称，默认 'default_recorder'。
            记录器实例的标识名。
            
        uri (str): 保存 URI 或本地路径，默认 'output/experiments'。
            记录器保存数据的位置（本地路径或 MLflow URI）。
            
        save_dir (str): 保存目录，默认 'output/experiments'。
            实验结果、日志、工件保存的目录。
            
        log_params (bool): 是否记录超参数，默认 True。
            记录所有模型超参数配置。
            
        log_metrics (bool): 是否记录指标，默认 True。
            记录训练过程中的损失、精度等指标。
            
        log_artifacts (bool): 是否记录工件，默认 True。
            保存模型、图表、数据等工件。
            
        tags (Dict[str,str]): 自动标签字典。
            为实验自动添加的标签（如 version、author 等）。
            
        auto_start (bool): 是否自动启动记录器，默认 True。
            初始化后自动开始记录。
    """
    # 实验信息
    experiment_name: str = "default_experiment"
    recorder_name: str = "default_recorder"
    
    # 存储路径
    uri: str = "output/experiments"  # MLflow URI 或本地路径
    save_dir: str = "output/experiments"
    
    # 记录选项
    log_params: bool = True  # 记录超参数
    log_metrics: bool = True  # 记录评估指标
    log_artifacts: bool = True  # 记录模型和图表
    
    # 自动标签
    tags: Dict[str, str] = field(default_factory=dict)
    
    # 自动启动
    auto_start: bool = True
    
    def validate(self) -> bool:
        """验证配置"""
        if not self.experiment_name:
            raise ValueError("experiment_name 不能为空")
        
        return True


@dataclass
class CheckpointConfig(BaseConfig):
    """
    检查点配置

    管理模型检查点的保存、加载和保留策略。

    Args:
        save_frequency (int): 每多少个 epoch 保存一次检查点，默认 10。
            例如: 10 表示每 10 个 epoch 保存一次。
            
        keep_last_n (int): 保留最近多少个检查点，默认 3。
            避免占用过多磁盘空间。
            
        save_best_only (bool): 是否只保存表现最好的模型，默认 False。
            - True: 只保存验证集性能最好的模型
            - False: 定期保存所有检查点
            
        checkpoint_dir (str): 检查点保存目录，默认 'output/checkpoints'。
    """
    # 保存频率
    save_frequency: int = 10  # 每N个epoch保存一次
    
    # 保留策略
    keep_last_n: int = 3  # 保留最近N个checkpoint
    save_best_only: bool = False  # 是否只保存最佳模型
    
    # 保存路径
    checkpoint_dir: str = "output/checkpoints"
    
    def validate(self) -> bool:
        """验证配置"""
        if self.save_frequency <= 0:
            raise ValueError("save_frequency 必须大于 0")
        
        if self.keep_last_n <= 0:
            raise ValueError("keep_last_n 必须大于 0")
        
        return True


@dataclass
class ArtifactConfig(BaseConfig):
    """
    工件配置

    管理要保存的工件（模型、图表、数据等）的相关参数。

    Args:
        artifact_paths (List[str]): 要保存的工件路径列表。
            指定要保存为工件的文件或目录路径。
            
        artifact_types (Dict[str,str]): 工件类型映射。
            将工件路径映射到类型（如 'model'、'plot'、'data'）。
    """
    # 要保存的工件列表
    artifact_paths: List[str] = field(default_factory=list)
    
    # 工件类型
    artifact_types: Dict[str, str] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """验证配置"""
        return True


@dataclass
class WorkflowConfig(BaseConfig):
    """
    工作流总配置

    管理整个实验工作流的所有配置，包括记录、检查点、工件等。

    Args:
        enabled (bool): 是否启用工作流管理，默认 True。
            - True: 启用工作流系统记录所有实验信息
            - False: 禁用工作流管理
            
        recorder (RecorderConfig): 记录器配置对象。
            用于记录超参数、指标、工件等实验信息。
            
        checkpoint (CheckpointConfig): 检查点配置对象。
            用于管理模型检查点的保存和加载。
            
        artifacts (ArtifactConfig): 工件配置对象。
            用于管理要保存的模型、图表、数据等。
            
        results_dir (str): 结果保存目录，默认 'output/results'。
            最终的训练和评估结果保存位置。
    """
    # 是否启用工作流管理
    enabled: bool = True
    
    # 记录器配置
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    
    # 检查点配置
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # 工件配置
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    
    # 结果保存路径
    results_dir: str = "output/results"
    
    def validate(self) -> bool:
        """验证配置"""
        # 递归验证子配置
        if self.enabled:
            self.recorder.validate()
            self.checkpoint.validate()
            self.artifacts.validate()
        
        return True


# 预定义配置模板
class WorkflowTemplates:
    """工作流配置模板"""
    
    @staticmethod
    def default() -> WorkflowConfig:
        """默认配置"""
        return WorkflowConfig()
    
    @staticmethod
    def minimal() -> WorkflowConfig:
        """最小化配置（仅记录基本信息）"""
        return WorkflowConfig(
            recorder=RecorderConfig(
                log_params=True,
                log_metrics=True,
                log_artifacts=False,
            ),
            checkpoint=CheckpointConfig(
                save_best_only=True,
                keep_last_n=1,
            ),
        )
    
    @staticmethod
    def full() -> WorkflowConfig:
        """完整配置（记录所有信息）"""
        return WorkflowConfig(
            recorder=RecorderConfig(
                log_params=True,
                log_metrics=True,
                log_artifacts=True,
                tags={
                    'task_type': 'training',
                    'framework': 'quantclassic',
                },
            ),
            checkpoint=CheckpointConfig(
                save_frequency=5,
                keep_last_n=5,
                save_best_only=False,
            ),
        )
    
    @staticmethod
    def production() -> WorkflowConfig:
        """生产环境配置"""
        return WorkflowConfig(
            recorder=RecorderConfig(
                log_params=True,
                log_metrics=True,
                log_artifacts=True,
                tags={
                    'env': 'production',
                    'framework': 'quantclassic',
                },
            ),
            checkpoint=CheckpointConfig(
                save_frequency=10,
                keep_last_n=3,
                save_best_only=True,
            ),
        )


if __name__ == '__main__':
    # 测试 WorkflowConfig
    print("=" * 80)
    print("WorkflowConfig 测试")
    print("=" * 80)
    
    # 测试 1: 创建默认配置
    print("\n1. 创建默认配置:")
    config = WorkflowConfig()
    print(f"  enabled: {config.enabled}")
    print(f"  experiment_name: {config.recorder.experiment_name}")
    print(f"  save_frequency: {config.checkpoint.save_frequency}")
    
    # 测试 2: 嵌套配置
    print("\n2. 嵌套配置:")
    config = WorkflowConfig(
        recorder=RecorderConfig(
            experiment_name="test_exp",
            tags={'model': 'vae', 'task': 'factor_mining'},
        ),
        checkpoint=CheckpointConfig(
            save_frequency=5,
            keep_last_n=10,
        ),
    )
    print(f"  experiment_name: {config.recorder.experiment_name}")
    print(f"  tags: {config.recorder.tags}")
    print(f"  checkpoint frequency: {config.checkpoint.save_frequency}")
    
    # 测试 3: YAML 序列化
    print("\n3. YAML 序列化:")
    yaml_path = '/tmp/workflow_config.yaml'
    config.to_yaml(yaml_path)
    print(f"  已保存到: {yaml_path}")
    
    config2 = WorkflowConfig.from_yaml(yaml_path)
    print(f"  已加载: {config2.recorder.experiment_name}")
    
    # 测试 4: 字典转换
    print("\n4. 字典转换:")
    config_dict = config.to_dict()
    print(f"  keys: {list(config_dict.keys())}")
    print(f"  recorder keys: {list(config_dict['recorder'].keys())[:3]}...")
    
    # 测试 5: 使用模板
    print("\n5. 使用模板:")
    minimal = WorkflowTemplates.minimal()
    print(f"  最小化配置: log_artifacts={minimal.recorder.log_artifacts}")
    
    full = WorkflowTemplates.full()
    print(f"  完整配置: save_frequency={full.checkpoint.save_frequency}")
    
    production = WorkflowTemplates.production()
    print(f"  生产配置: tags={production.recorder.tags}")
    
    # 测试 6: 配置验证
    print("\n6. 配置验证:")
    try:
        invalid_config = WorkflowConfig(
            checkpoint=CheckpointConfig(save_frequency=-1)
        )
    except ValueError as e:
        print(f"  ✅ 捕获到验证错误: {e}")
    
    print("\n" + "=" * 80)
    print("✅ WorkflowConfig 测试完成")
    print("=" * 80)
