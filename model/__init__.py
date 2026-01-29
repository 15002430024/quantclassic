"""
QuantClassic Model Module - 模型模块

提供标准化的模型接口和实现

图构建架构 (2026-01 重构):
- data_processor/graph_builder.py: 图构建算法 (HOW to build)
- data_set/graph/daily_graph_loader.py: 数据加载 + 调用时机 (WHEN to call)
- model/base_model.py: _parse_batch_data() 自动解析 adj

训练架构 (2026-01 重构):
- model/train/base_trainer.py: 训练基类，统一训练循环
- model/train/simple_trainer.py: 简单训练器
- model/train/rolling_window_trainer.py: 滚动窗口训练器
- model/train/rolling_daily_trainer.py: 日级滚动训练器

核心变更:
- PyTorchModel.fit() 已代理到 SimpleTrainer
- 新增统一的训练引擎 model/train/
- 新增 UnifiedLoss 统一损失函数
"""

from .base_model import BaseModel, Model, PyTorchModel
from .model_factory import (
    ModelFactory, 
    ModelRegistry,
    register_model, 
    init_instance_by_config,
    create_model_from_composite_config,
)
from .pytorch_models import LSTMModel, GRUModel, TransformerModel, VAEModel
from .hybrid_graph_models import HybridGraphModel, HybridNet, TemporalBlock, GraphBlock, FusionBlock
from .loss import (
    get_loss_fn,
    UnifiedLoss,
    MSEWithCorrelationLoss,
    ICLoss,
    ICWithCorrelationLoss,
    CombinedLoss,
    CorrelationRegularizer,
)

# 🆕 训练模块
from .train import (
    BaseTrainer,
    TrainerArtifacts,
    TrainerConfig,
    TrainerCallback,
    EarlyStoppingCallback,
    CheckpointCallback,
    SimpleTrainer,
    RollingWindowTrainer,
    RollingDailyTrainer,
    RollingTrainerConfig,
)

# 🆕 预测助手
from .predict import predict_with_metadata, compute_ic, compute_ic_stats

# 🆕 兼容旧接口（延迟导入，避免 dynamic_graph_trainer 缺失报错）
def create_rolling_trainer(*args, **kwargs):
    """兼容旧接口 - 已废弃，请使用 RollingDailyTrainer"""
    import warnings
    warnings.warn(
        "create_rolling_trainer 已废弃，请使用 model.train.RollingDailyTrainer",
        DeprecationWarning,
        stacklevel=2
    )
    from .train import create_rolling_daily_trainer
    return create_rolling_daily_trainer(*args, **kwargs)

# 兼容类名别名
LegacyRollingDailyTrainer = RollingDailyTrainer
LegacyRollingTrainerConfig = RollingTrainerConfig

__all__ = [
    # 基类
    'BaseModel',
    'Model',
    'PyTorchModel',
    
    # 工厂
    'ModelFactory',
    'ModelRegistry',
    'register_model',
    'init_instance_by_config',
    'create_model_from_composite_config',
    
    # 模型
    'LSTMModel',
    'GRUModel',
    'TransformerModel',
    'VAEModel',
    'HybridGraphModel',
    
    # 纯 nn.Module 组件
    'HybridNet',
    'TemporalBlock',
    'GraphBlock',
    'FusionBlock',
    
    # 损失函数
    'get_loss_fn',
    'UnifiedLoss',
    'MSEWithCorrelationLoss',
    'ICLoss',
    'ICWithCorrelationLoss',
    'CombinedLoss',
    'CorrelationRegularizer',
    
    # 🆕 训练引擎
    'BaseTrainer',
    'TrainerArtifacts',
    'TrainerConfig',
    'TrainerCallback',
    'EarlyStoppingCallback',
    'CheckpointCallback',
    'SimpleTrainer',
    'RollingWindowTrainer',
    'RollingDailyTrainer',
    'RollingTrainerConfig',
    
    # 兼容旧接口
    'create_rolling_trainer',
    
    # 🆕 预测助手
    'predict_with_metadata',
    'compute_ic',
    'compute_ic_stats',
]
