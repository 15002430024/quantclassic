"""
ModularConfig - 模块化配置系统

将混合模型拆分为独立的模块配置，支持灵活组合和扩展变体。

设计思想:
- 每个功能模块 (Temporal/Graph/Fusion) 都有独立的配置类
- 通过组合器 (CompositeConfig) 将多个模块配置组合成完整模型
- 支持插件式扩展：轻松添加新的模块类型或变体

Example:
    # 方式1: 使用独立模块配置
    temporal_cfg = TemporalModuleConfig(rnn_type='lstm', hidden_size=64, use_attention=True)
    graph_cfg = GraphModuleConfig(gat_type='correlation', hidden_dim=32, heads=4)
    fusion_cfg = FusionModuleConfig(hidden_sizes=[64, 32])
    
    model_cfg = CompositeModelConfig(
        temporal=temporal_cfg,
        graph=graph_cfg,
        fusion=fusion_cfg,
        d_feat=20
    )
    
    # 方式2: 使用构建器快速创建
    model_cfg = ModelConfigBuilder() \\
        .add_temporal(rnn_type='gru', hidden_size=128) \\
        .add_graph(gat_type='standard', hidden_dim=64) \\
        .add_fusion(hidden_sizes=[128, 64]) \\
        .build(d_feat=20)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pathlib import Path

# 使用相对导入（相对于 quantclassic 包）
# 移除降级版 BaseConfig，强制依赖正确的基类
try:
    from ..config.base_config import BaseConfig
except ImportError:
    # 直接运行脚本时的后备导入
    from config.base_config import BaseConfig

# 导入 BaseModelConfig（用于模型配置继承）
try:
    from .model_config import BaseModelConfig
except ImportError:
    # 直接运行脚本时的后备导入
    from model.model_config import BaseModelConfig


# ==================== 模块类型枚举 ====================

class ModuleType(str, Enum):
    """模块类型枚举"""
    TEMPORAL = 'temporal'      # 时序特征提取模块
    GRAPH = 'graph'           # 图神经网络模块
    FUSION = 'fusion'         # 特征融合模块
    ATTENTION = 'attention'   # 注意力机制模块


class RNNType(str, Enum):
    """RNN类型"""
    LSTM = 'lstm'
    GRU = 'gru'
    RNN = 'rnn'


class GATType(str, Enum):
    """GAT类型"""
    STANDARD = 'standard'         # 基于行业关系
    CORRELATION = 'correlation'   # 基于收益率相关性
    DYNAMIC = 'dynamic'           # 动态构建图结构


class AttentionType(str, Enum):
    """注意力机制类型"""
    SELF = 'self'                 # Self-Attention
    MULTI_HEAD = 'multi_head'     # Multi-Head Attention
    ADDITIVE = 'additive'         # Additive Attention
    DOT_PRODUCT = 'dot_product'   # Dot-Product Attention


# ==================== 模块配置基类 ====================

@dataclass
class ModuleConfig(BaseConfig):
    """
    模块配置基类
    
    所有功能模块配置的基类。
    
    Args:
        enabled (bool): 是否启用该模块，默认 True。
        name (Optional[str]): 模块名称，用于标识和日志。
    """
    enabled: bool = True              # 是否启用该模块
    name: Optional[str] = None        # 模块名称
    
    def validate(self) -> bool:
        """验证配置"""
        return True


# ==================== 时序模块配置 ====================

@dataclass
class TemporalModuleConfig(ModuleConfig):
    """
    时序特征提取模块配置 (RNN + Attention)
    
    负责从时间序列数据中提取时序特征。
    
    Args:
        rnn_type (str): RNN类型，可选 'lstm', 'gru', 'rnn'。
            - 'lstm': 长短期记忆网络，记忆能力强，适合复杂模式
            - 'gru': 门控循环单元，参数少，训练快
            - 'rnn': 标准RNN，最简单
            
        hidden_size (int): RNN隐藏层大小，默认 64。
            控制时序特征提取能力（64-256 常用）。
            
        num_layers (int): RNN层数，默认 2。
            堆叠RNN层数（1-3层，过深易过拟合）。
            
        bidirectional (bool): 是否使用双向RNN，默认 False。
            双向RNN可捕获未来信息，但参数翻倍。
            
        use_attention (bool): 是否使用注意力机制，默认 True。
            强化关键时间点的权重。
            
        attention_type (str): 注意力类型，默认 'self'。
            可选: 'self', 'multi_head', 'additive', 'dot_product'
            
        attention_heads (int): 多头注意力的头数，默认 4。
            仅在 attention_type='multi_head' 时有效。
            
        dropout (float): Dropout概率，默认 0.3。
    """
    # ==================== RNN配置 ====================
    rnn_type: str = 'lstm'                      # RNN类型: 'lstm', 'gru', 'rnn'
    hidden_size: int = 64                       # 隐藏层大小
    num_layers: int = 2                         # RNN层数
    bidirectional: bool = False                 # 是否双向
    
    # ==================== 注意力配置 ====================
    use_attention: bool = True                  # 是否使用注意力
    attention_type: str = 'self'                # 注意力类型
    attention_heads: int = 4                    # 多头注意力头数
    
    # ==================== 正则化 ====================
    dropout: float = 0.3                        # Dropout概率
    
    # ==================== 🆕 残差连接 ====================
    use_residual: bool = True                   # 是否使用残差连接 (LayerNorm + Skip Connection)
    
    def validate(self) -> bool:
        """验证配置"""
        super().validate()
        
        if self.rnn_type not in ['lstm', 'gru', 'rnn']:
            raise ValueError(f"不支持的RNN类型: {self.rnn_type}")
        
        if self.hidden_size <= 0:
            raise ValueError("hidden_size 必须大于 0")
        
        if self.num_layers <= 0:
            raise ValueError("num_layers 必须大于 0")
        
        if self.use_attention:
            if self.attention_type not in ['self', 'multi_head', 'additive', 'dot_product']:
                raise ValueError(f"不支持的注意力类型: {self.attention_type}")
            
            if self.attention_type == 'multi_head' and self.attention_heads <= 0:
                raise ValueError("attention_heads 必须大于 0")
        
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout 必须在 [0, 1] 范围内")
        
        return True
    
    @property
    def output_dim(self) -> int:
        """计算输出维度"""
        multiplier = 2 if self.bidirectional else 1
        return self.hidden_size * multiplier


# ==================== 图模块配置 ====================

@dataclass
class GraphModuleConfig(ModuleConfig):
    """
    图神经网络模块配置 (GAT)
    
    通过图结构捕捉截面信息（股票间关联）。
    
    Args:
        gat_type (str): GAT类型，默认 'standard'。
            - 'standard': 基于行业分类的静态图
            - 'correlation': 基于收益率相关性的动态图
            - 'dynamic': 完全动态学习图结构
            
        hidden_dim (int): GAT隐藏层维度，默认 32。
            必须能被 heads 整除。
            
        heads (int): 多头注意力头数，默认 4。
            
        concat (bool): 是否拼接多头输出，默认 True。
            False 时取平均。
            
        top_k_neighbors (int): 相关性GAT的邻居数，默认 10。
            仅在 gat_type='correlation' 时有效。
            
        edge_threshold (float): 边权重阈值，默认 0.0。
            低于此值的边会被剪枝（用于稀疏化图）。
            
        use_edge_features (bool): 是否使用边特征，默认 False。
            如果True，可以在边上附加额外信息（如相关系数）。
            
        dropout (float): Dropout概率，默认 0.3。
    """
    # ==================== GAT配置 ====================
    gat_type: str = 'standard'                  # GAT类型
    hidden_dim: int = 32                        # 隐藏层维度
    heads: int = 4                              # 注意力头数
    concat: bool = True                         # 是否拼接多头输出
    
    # ==================== 图结构配置 ====================
    top_k_neighbors: int = 10                   # K近邻数量
    edge_threshold: float = 0.0                 # 边权重阈值
    use_edge_features: bool = False             # 是否使用边特征
    
    # ==================== 正则化 ====================
    dropout: float = 0.3                        # Dropout概率
    
    # ==================== 🆕 残差连接 ====================
    use_residual: bool = True                   # 是否使用残差连接 (LayerNorm + Skip Connection)
    
    def validate(self) -> bool:
        """验证配置"""
        super().validate()
        
        if self.gat_type not in ['standard', 'correlation', 'dynamic']:
            raise ValueError(f"不支持的GAT类型: {self.gat_type}")
        
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim 必须大于 0")
        
        if self.heads <= 0:
            raise ValueError("heads 必须大于 0")
        
        if self.concat and self.hidden_dim % self.heads != 0:
            raise ValueError("concat=True 时，hidden_dim 必须能被 heads 整除")
        
        if self.gat_type == 'correlation' and self.top_k_neighbors <= 0:
            raise ValueError("top_k_neighbors 必须大于 0")
        
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout 必须在 [0, 1] 范围内")
        
        return True
    
    @property
    def output_dim(self) -> int:
        """计算输出维度"""
        if self.concat:
            return self.hidden_dim
        else:
            return self.hidden_dim


# ==================== 融合模块配置 ====================

@dataclass
class FusionModuleConfig(ModuleConfig):
    """
    特征融合模块配置 (MLP)
    
    将多个模块的输出特征融合后进行预测。
    
    Args:
        hidden_sizes (List[int]): MLP隐藏层尺寸列表，默认 [64]。
            例如 [128, 64] 表示两层MLP。
            
        activation (str): 激活函数，默认 'relu'。
            可选: 'relu', 'gelu', 'tanh', 'leaky_relu'
            
        use_batch_norm (bool): 是否使用BatchNorm，默认 False。
            
        use_residual (bool): 是否使用残差连接，默认 False。
            适合深层MLP，防止梯度消失。
            
        dropout (float): Dropout概率，默认 0.3。
            
        output_dim (int): 输出维度，默认 1。
            预测目标数量（单目标回归为1）。
    """
    # ==================== MLP配置 ====================
    hidden_sizes: List[int] = field(default_factory=lambda: [64])  # 隐藏层尺寸
    activation: str = 'relu'                    # 激活函数
    use_batch_norm: bool = False                # 是否使用BatchNorm
    use_residual: bool = False                  # 是否使用残差连接
    
    # ==================== 正则化 ====================
    dropout: float = 0.3                        # Dropout概率
    
    # ==================== 输出 ====================
    output_dim: int = 1                         # 输出维度
    
    def validate(self) -> bool:
        """验证配置"""
        super().validate()
        
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes 不能为空")
        
        if any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes 中的所有值必须大于 0")
        
        if self.activation not in ['relu', 'gelu', 'tanh', 'leaky_relu']:
            raise ValueError(f"不支持的激活函数: {self.activation}")
        
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout 必须在 [0, 1] 范围内")
        
        if self.output_dim <= 0:
            raise ValueError("output_dim 必须大于 0")
        
        return True


# ==================== 组合模型配置 ====================

@dataclass
class CompositeModelConfig(BaseModelConfig):
    """
    组合模型配置 (模块化混合模型)
    
    通过组合多个功能模块构建完整的混合模型。
    
    Args:
        temporal (Optional[TemporalModuleConfig]): 时序模块配置。
            如果为 None，则不使用时序模块。
            
        graph (Optional[GraphModuleConfig]): 图模块配置。
            如果为 None，则不使用图模块。
            
        fusion (FusionModuleConfig): 融合模块配置（必需）。
            
        d_feat (int): 输入特征维度（量价数据），默认 20。
            
        funda_dim (Optional[int]): 基本面数据维度，默认 None。
            如果提供基本面数据，会在适当位置拼接。
            
        adj_matrix_path (Optional[str]): 邻接矩阵路径，默认 None。
            图模块使用的邻接矩阵文件（.pt或.npy）。
            
        feature_fusion_strategy (str): 特征融合策略，默认 'concat'。
            - 'concat': 直接拼接各模块输出
            - 'add': 相加（要求维度相同）
            - 'weighted': 加权求和（可学习权重）
    
    Example:
        # 创建时序+图+融合的完整模型
        config = CompositeModelConfig(
            temporal=TemporalModuleConfig(rnn_type='lstm', hidden_size=64),
            graph=GraphModuleConfig(gat_type='correlation', hidden_dim=32),
            fusion=FusionModuleConfig(hidden_sizes=[64]),
            d_feat=20
        )
        
        # 创建纯时序模型（不使用图）
        config = CompositeModelConfig(
            temporal=TemporalModuleConfig(rnn_type='gru', hidden_size=128),
            graph=None,
            fusion=FusionModuleConfig(hidden_sizes=[64]),
            d_feat=20
        )
    """
    # ==================== 模块配置 ====================
    temporal: Optional[TemporalModuleConfig] = field(default_factory=TemporalModuleConfig)
    graph: Optional[GraphModuleConfig] = None
    fusion: FusionModuleConfig = field(default_factory=FusionModuleConfig)
    
    # ==================== 输入特征 ====================
    d_feat: int = 20                            # 输入特征维度
    funda_dim: Optional[int] = None             # 基本面数据维度
    
    # ==================== 图配置 ====================
    adj_matrix_path: Optional[str] = None       # 邻接矩阵路径
    
    # ==================== 融合策略 ====================
    feature_fusion_strategy: str = 'concat'     # 特征融合策略
    
    # ==================== 🆕 Graph-Aware Inference 配置 ====================
    graph_inference_mode: str = 'batch'         # 图推理模式: 'batch', 'neighbor_sampling', 'cross_sectional'
    max_neighbors: int = 10                     # 邻居采样时的最大邻居数
    cache_size: int = 5000                      # 节点状态缓存大小
    
    # ==================== 🆕 数据格式配置 ====================
    stock_idx_position: Optional[int] = None    # 股票索引在 batch 中的位置
    funda_position: Optional[int] = None        # 基本面数据在 batch 中的位置
    
    # ==================== 🆕 学习率调度器配置 ====================
    use_scheduler: bool = True                  # 是否使用学习率调度器
    scheduler_type: str = 'plateau'             # 调度器类型: 'plateau', 'cosine', 'step'
    scheduler_patience: int = 5                 # ReduceLROnPlateau 的耐心值
    scheduler_factor: float = 0.5               # 学习率衰减因子
    scheduler_min_lr: float = 1e-6              # 最小学习率
    
    # ==================== 🆕 残差连接配置 ====================
    use_residual: bool = True                   # 是否使用残差连接 (LayerNorm + Skip Connection)
    
    # ==================== 🆕 相关性正则化配置 ====================
    lambda_corr: float = 0.0                    # 相关性正则化权重 (0=禁用, 推荐0.001~0.1)
                                                # 作用: 抑制隐藏层特征冗余，鼓励学习更独立的特征表达
                                                # 公式: Loss = BaseLoss + lambda_corr * ||Corr(H) - I||_F
    
    def __post_init__(self):
        """初始化后处理 - 转换字典为模块配置对象"""
        # 如果 temporal 是字典，转换为 TemporalModuleConfig
        if isinstance(self.temporal, dict):
            self.temporal = TemporalModuleConfig(**self.temporal)
        
        # 如果 graph 是字典，转换为 GraphModuleConfig
        if isinstance(self.graph, dict):
            self.graph = GraphModuleConfig(**self.graph)
        
        # 如果 fusion 是字典，转换为 FusionModuleConfig
        if isinstance(self.fusion, dict):
            self.fusion = FusionModuleConfig(**self.fusion)
    
    def validate(self) -> bool:
        """验证配置"""
        super().validate()
        
        # 至少要有一个特征提取模块
        if self.temporal is None and self.graph is None:
            raise ValueError("至少需要启用 temporal 或 graph 模块之一")
        
        # 验证各模块配置
        if self.temporal is not None and self.temporal.enabled:
            self.temporal.validate()
        
        if self.graph is not None and self.graph.enabled:
            self.graph.validate()
            
            # 静态图需要邻接矩阵，动态图（correlation/dynamic）依赖在线构建
            if self.adj_matrix_path is None and self.graph.gat_type == 'standard':
                import warnings
                warnings.warn("graph 模块已启用但未提供 adj_matrix_path")
        
        self.fusion.validate()
        
        # 验证特征维度
        if self.d_feat <= 0:
            raise ValueError("d_feat 必须大于 0")
        
        if self.funda_dim is not None and self.funda_dim <= 0:
            raise ValueError("funda_dim 必须大于 0")
        
        # 验证融合策略
        if self.feature_fusion_strategy not in ['concat', 'add', 'weighted']:
            raise ValueError(f"不支持的融合策略: {self.feature_fusion_strategy}")
        
        return True
    
    def get_fusion_input_dim(self) -> int:
        """
        计算融合模块的输入维度
        
        Returns:
            融合模块输入维度
        """
        total_dim = 0
        
        # 时序模块输出
        if self.temporal is not None and self.temporal.enabled:
            total_dim += self.temporal.output_dim
        
        # 图模块输出
        if self.graph is not None and self.graph.enabled:
            total_dim += self.graph.output_dim
        
        # 基本面数据
        if self.funda_dim is not None:
            total_dim += self.funda_dim
        
        # 如果是加法或加权融合，需要所有维度相同
        if self.feature_fusion_strategy in ['add', 'weighted']:
            temporal_dim = self.temporal.output_dim if self.temporal else 0
            graph_dim = self.graph.output_dim if self.graph else 0
            
            if temporal_dim > 0 and graph_dim > 0 and temporal_dim != graph_dim:
                raise ValueError(
                    f"使用 {self.feature_fusion_strategy} 融合时，"
                    f"temporal 和 graph 的输出维度必须相同 "
                    f"(当前: {temporal_dim} vs {graph_dim})"
                )
            
            # 加法/加权融合后维度不变
            total_dim = max(temporal_dim, graph_dim)
            if self.funda_dim is not None:
                total_dim += self.funda_dim
        
        return total_dim
    
    def summary(self) -> str:
        """生成配置摘要"""
        lines = ["=" * 60, "组合模型配置摘要", "=" * 60, ""]
        
        lines.append(f"输入特征维度: {self.d_feat}")
        if self.funda_dim:
            lines.append(f"基本面维度: {self.funda_dim}")
        lines.append("")
        
        # 时序模块
        if self.temporal and self.temporal.enabled:
            lines.append("【时序模块】")
            lines.append(f"  - RNN类型: {self.temporal.rnn_type}")
            lines.append(f"  - 隐藏层: {self.temporal.hidden_size}")
            lines.append(f"  - 层数: {self.temporal.num_layers}")
            lines.append(f"  - 双向: {self.temporal.bidirectional}")
            lines.append(f"  - 注意力: {self.temporal.use_attention}")
            if self.temporal.use_attention:
                lines.append(f"    类型: {self.temporal.attention_type}")
            lines.append(f"  - 输出维度: {self.temporal.output_dim}")
            lines.append("")
        
        # 图模块
        if self.graph and self.graph.enabled:
            lines.append("【图模块】")
            lines.append(f"  - GAT类型: {self.graph.gat_type}")
            lines.append(f"  - 隐藏维度: {self.graph.hidden_dim}")
            lines.append(f"  - 注意力头数: {self.graph.heads}")
            lines.append(f"  - 拼接多头: {self.graph.concat}")
            if self.graph.gat_type == 'correlation':
                lines.append(f"  - K近邻: {self.graph.top_k_neighbors}")
            lines.append(f"  - 输出维度: {self.graph.output_dim}")
            if self.adj_matrix_path:
                lines.append(f"  - 邻接矩阵: {self.adj_matrix_path}")
            lines.append("")
        
        # 融合模块
        lines.append("【融合模块】")
        lines.append(f"  - 融合策略: {self.feature_fusion_strategy}")
        lines.append(f"  - 输入维度: {self.get_fusion_input_dim()}")
        lines.append(f"  - 隐藏层: {self.fusion.hidden_sizes}")
        lines.append(f"  - 激活函数: {self.fusion.activation}")
        lines.append(f"  - BatchNorm: {self.fusion.use_batch_norm}")
        lines.append(f"  - 残差连接: {self.fusion.use_residual}")
        lines.append(f"  - 输出维度: {self.fusion.output_dim}")
        lines.append("")
        
        # 训练配置
        lines.append("【训练配置】")
        lines.append(f"  - 设备: {self.device}")
        lines.append(f"  - Epochs: {self.n_epochs}")
        lines.append(f"  - Batch Size: {self.batch_size}")
        lines.append(f"  - 学习率: {self.learning_rate}")
        lines.append(f"  - 优化器: {self.optimizer}")
        lines.append(f"  - 损失函数: {self.loss_fn}")
        lines.append(f"  - 早停: {self.early_stop}")
        lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# ==================== 配置构建器 ====================

class ModelConfigBuilder:
    """
    模型配置构建器 (Builder Pattern)
    
    提供流式API快速构建模型配置。
    
    Example:
        config = ModelConfigBuilder() \\
            .set_input(d_feat=20, funda_dim=10) \\
            .add_temporal(rnn_type='lstm', hidden_size=64, use_attention=True) \\
            .add_graph(gat_type='correlation', hidden_dim=32, heads=4) \\
            .add_fusion(hidden_sizes=[64, 32]) \\
            .set_training(n_epochs=100, batch_size=256) \\
            .build()
    """
    
    def __init__(self):
        self._d_feat = 20
        self._funda_dim = None
        self._temporal_config = None
        self._graph_config = None
        self._fusion_config = FusionModuleConfig()
        self._adj_matrix_path = None
        self._training_kwargs = {}
    
    def set_input(self, d_feat: int, funda_dim: Optional[int] = None) -> 'ModelConfigBuilder':
        """设置输入特征维度"""
        self._d_feat = d_feat
        self._funda_dim = funda_dim
        return self
    
    def add_temporal(
        self,
        rnn_type: str = 'lstm',
        hidden_size: int = 64,
        num_layers: int = 2,
        bidirectional: bool = False,
        use_attention: bool = True,
        attention_type: str = 'self',
        dropout: float = 0.3,
        **kwargs
    ) -> 'ModelConfigBuilder':
        """添加时序模块"""
        self._temporal_config = TemporalModuleConfig(
            rnn_type=rnn_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            use_attention=use_attention,
            attention_type=attention_type,
            dropout=dropout,
            **kwargs
        )
        return self
    
    def add_graph(
        self,
        gat_type: str = 'standard',
        hidden_dim: int = 32,
        heads: int = 4,
        concat: bool = True,
        top_k_neighbors: int = 10,
        adj_matrix_path: Optional[str] = None,
        dropout: float = 0.3,
        **kwargs
    ) -> 'ModelConfigBuilder':
        """添加图模块"""
        self._graph_config = GraphModuleConfig(
            gat_type=gat_type,
            hidden_dim=hidden_dim,
            heads=heads,
            concat=concat,
            top_k_neighbors=top_k_neighbors,
            dropout=dropout,
            **kwargs
        )
        if adj_matrix_path:
            self._adj_matrix_path = adj_matrix_path
        return self
    
    def add_fusion(
        self,
        hidden_sizes: List[int] = None,
        activation: str = 'relu',
        use_batch_norm: bool = False,
        use_residual: bool = False,
        dropout: float = 0.3,
        output_dim: int = 1,
        **kwargs
    ) -> 'ModelConfigBuilder':
        """添加融合模块"""
        self._fusion_config = FusionModuleConfig(
            hidden_sizes=hidden_sizes or [64],
            activation=activation,
            use_batch_norm=use_batch_norm,
            use_residual=use_residual,
            dropout=dropout,
            output_dim=output_dim,
            **kwargs
        )
        return self
    
    def set_training(
        self,
        device: str = 'cuda',
        n_epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        optimizer: str = 'adam',
        loss_fn: str = 'mse',
        early_stop: int = 20,
        **kwargs
    ) -> 'ModelConfigBuilder':
        """设置训练参数"""
        self._training_kwargs.update({
            'device': device,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': optimizer,
            'loss_fn': loss_fn,
            'early_stop': early_stop,
            **kwargs
        })
        return self
    
    def build(self, **override_kwargs) -> CompositeModelConfig:
        """
        构建配置对象
        
        Args:
            **override_kwargs: 覆盖参数
            
        Returns:
            CompositeModelConfig 对象
        """
        config = CompositeModelConfig(
            temporal=self._temporal_config,
            graph=self._graph_config,
            fusion=self._fusion_config,
            d_feat=self._d_feat,
            funda_dim=self._funda_dim,
            adj_matrix_path=self._adj_matrix_path,
            **self._training_kwargs,
            **override_kwargs
        )
        
        config.validate()
        return config


# ==================== 预定义模板 ====================

class ConfigTemplates:
    """预定义配置模板"""
    
    @staticmethod
    def pure_temporal(d_feat: int = 20, model_size: str = 'default') -> CompositeModelConfig:
        """
        纯时序模型（不使用图）
        
        Args:
            d_feat: 输入特征维度
            model_size: 模型大小 ('small', 'default', 'large')
        """
        size_map = {
            'small': {'rnn_hidden': 32, 'rnn_layers': 1, 'mlp_hidden': [32]},
            'default': {'rnn_hidden': 64, 'rnn_layers': 2, 'mlp_hidden': [64]},
            'large': {'rnn_hidden': 128, 'rnn_layers': 3, 'mlp_hidden': [128, 64]},
        }
        
        params = size_map.get(model_size, size_map['default'])
        
        return ModelConfigBuilder() \
            .set_input(d_feat=d_feat) \
            .add_temporal(
                rnn_type='lstm',
                hidden_size=params['rnn_hidden'],
                num_layers=params['rnn_layers'],
                use_attention=True
            ) \
            .add_fusion(hidden_sizes=params['mlp_hidden']) \
            .build()
    
    @staticmethod
    def temporal_with_graph(
        d_feat: int = 20,
        gat_type: str = 'standard',
        adj_matrix_path: Optional[str] = None,
        model_size: str = 'default'
    ) -> CompositeModelConfig:
        """
        时序+图混合模型
        
        Args:
            d_feat: 输入特征维度
            gat_type: GAT类型 ('standard' 或 'correlation')
            adj_matrix_path: 邻接矩阵路径
            model_size: 模型大小 ('small', 'default', 'large')
        """
        size_map = {
            'small': {
                'rnn_hidden': 32, 'rnn_layers': 1,
                'gat_hidden': 16, 'gat_heads': 2,
                'mlp_hidden': [32]
            },
            'default': {
                'rnn_hidden': 64, 'rnn_layers': 2,
                'gat_hidden': 32, 'gat_heads': 4,
                'mlp_hidden': [64]
            },
            'large': {
                'rnn_hidden': 128, 'rnn_layers': 3,
                'gat_hidden': 64, 'gat_heads': 8,
                'mlp_hidden': [128, 64]
            },
        }
        
        params = size_map.get(model_size, size_map['default'])
        
        return ModelConfigBuilder() \
            .set_input(d_feat=d_feat) \
            .add_temporal(
                rnn_type='lstm',
                hidden_size=params['rnn_hidden'],
                num_layers=params['rnn_layers'],
                use_attention=True
            ) \
            .add_graph(
                gat_type=gat_type,
                hidden_dim=params['gat_hidden'],
                heads=params['gat_heads'],
                adj_matrix_path=adj_matrix_path
            ) \
            .add_fusion(hidden_sizes=params['mlp_hidden']) \
            .build()
    
    @staticmethod
    def attention_graph_fusion(
        d_feat: int = 20,
        attention_type: str = 'multi_head',
        gat_type: str = 'correlation',
        adj_matrix_path: Optional[str] = None
    ) -> CompositeModelConfig:
        """
        多头注意力 + 相关性图 + 深层融合
        
        Args:
            d_feat: 输入特征维度
            attention_type: 注意力类型
            gat_type: GAT类型
            adj_matrix_path: 邻接矩阵路径
        """
        return ModelConfigBuilder() \
            .set_input(d_feat=d_feat) \
            .add_temporal(
                rnn_type='gru',
                hidden_size=64,
                num_layers=2,
                use_attention=True,
                attention_type=attention_type,
                attention_heads=8
            ) \
            .add_graph(
                gat_type=gat_type,
                hidden_dim=64,
                heads=8,
                top_k_neighbors=15,
                adj_matrix_path=adj_matrix_path
            ) \
            .add_fusion(
                hidden_sizes=[128, 64, 32],
                use_batch_norm=True,
                use_residual=True
            ) \
            .build()


if __name__ == '__main__':
    print("=" * 80)
    print("ModularConfig 测试")
    print("=" * 80)
    
    # 测试 1: 创建独立模块配置
    print("\n1. 创建独立模块配置:")
    temporal_cfg = TemporalModuleConfig(
        rnn_type='lstm',
        hidden_size=64,
        num_layers=2,
        use_attention=True
    )
    print(f"  时序模块: {temporal_cfg.rnn_type}, 输出维度={temporal_cfg.output_dim}")
    
    graph_cfg = GraphModuleConfig(
        gat_type='correlation',
        hidden_dim=32,
        heads=4
    )
    print(f"  图模块: {graph_cfg.gat_type}, 输出维度={graph_cfg.output_dim}")
    
    fusion_cfg = FusionModuleConfig(
        hidden_sizes=[64, 32],
        activation='relu'
    )
    print(f"  融合模块: {fusion_cfg.hidden_sizes}")
    
    # 测试 2: 组合模型配置
    print("\n2. 组合模型配置:")
    composite_cfg = CompositeModelConfig(
        temporal=temporal_cfg,
        graph=graph_cfg,
        fusion=fusion_cfg,
        d_feat=20
    )
    composite_cfg.validate()
    print(f"  融合输入维度: {composite_cfg.get_fusion_input_dim()}")
    
    # 测试 3: 使用构建器
    print("\n3. 使用构建器:")
    builder_cfg = ModelConfigBuilder() \
        .set_input(d_feat=20) \
        .add_temporal(rnn_type='gru', hidden_size=128) \
        .add_graph(gat_type='standard', hidden_dim=64) \
        .add_fusion(hidden_sizes=[128, 64]) \
        .set_training(n_epochs=100, batch_size=256) \
        .build()
    
    print(f"  ✅ Builder 创建成功")
    
    # 测试 4: 纯时序模型
    print("\n4. 纯时序模型模板:")
    pure_temporal = ConfigTemplates.pure_temporal(d_feat=20, model_size='default')
    print(f"  时序模块: {pure_temporal.temporal.rnn_type}")
    print(f"  图模块: {pure_temporal.graph}")
    
    # 测试 5: 时序+图混合模型
    print("\n5. 混合模型模板:")
    hybrid = ConfigTemplates.temporal_with_graph(
        d_feat=20,
        gat_type='correlation',
        model_size='large'
    )
    print(f"  时序隐藏层: {hybrid.temporal.hidden_size}")
    print(f"  图隐藏层: {hybrid.graph.hidden_dim}")
    
    # 测试 6: 配置摘要
    print("\n6. 配置摘要:")
    print(hybrid.summary())
    
    # 测试 7: YAML序列化
    print("\n7. YAML序列化:")
    yaml_path = '/tmp/composite_config.yaml'
    hybrid.to_yaml(yaml_path)
    print(f"  已保存到: {yaml_path}")
    
    loaded_cfg = CompositeModelConfig.from_yaml(yaml_path)
    print(f"  已加载: 时序hidden={loaded_cfg.temporal.hidden_size}")
    
    print("\n" + "=" * 80)
    print("✅ ModularConfig 测试完成")
    print("=" * 80)
