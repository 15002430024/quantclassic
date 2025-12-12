"""
ModelConfig - 模型配置类

使用面向对象的配置替代字典配置
支持所有 QuantClassic 模型的统一配置接口
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.base_config import BaseConfig


@dataclass
class BaseModelConfig(BaseConfig):
    """
    基础模型配置类

    管理训练/优化/保存等模型通用参数，用于所有模型配置的基类。

    Args:
        device (str): 计算设备，例如 'cuda' 或 'cpu'。
            - 'cuda': GPU 加速（推荐用于大规模训练）
            - 'cpu': CPU 计算（支持所有设备，但速度较慢）
            - 'cuda:0'/'cuda:1': 指定特定 GPU 设备
            
        n_epochs (int): 训练轮数，默认 100。
            模型在整个训练集上的完整迭代次数（50-200常用）。
            
        batch_size (int): 训练批次大小，默认 256。
            每个批次包含的样本数（256-1024，越大训练越快但需更多显存）。
            
        learning_rate (float): 初始学习率，默认 0.001。
            优化器的步长（0.0001-0.01，Adam优化器常用0.001）。
            
        early_stop (int): 早停轮数，默认 20。
            验证集性能不提升超过此轮数时停止训练。
            
        optimizer (str): 优化器类型，可选值:
            - 'adam': Adam 优化器（常用，推荐默认）
            - 'adamw': Adam with Weight Decay（带权重衰减）
            - 'sgd': 随机梯度下降（古典方法）
            
        loss_fn (str): 损失函数，可选值:
            - 'mse': 均方误差（默认，用于回归任务）
            - 'mae': 平均绝对误差（对异常值不敏感）
            - 'huber': Huber 损失（结合 MSE 和 MAE 优点）
            
        weight_decay (float): L2 正则化系数，默认 0.0。
            用于防止过拟合（0.0001-0.01，较小值通常有效）。
            
        model_save_path (str): 模型保存路径，默认 'output/best_model.pth'。
            训练完成后保存最佳模型到此路径。
            
        log_dir (str): 日志目录，默认 'logs'。
            训练日志、TensorBoard 事件等保存位置。
            
        verbose (bool): 是否显示详细训练信息，默认 True。
            打印每个 epoch 的损失、指标等。
            
        seed (Optional[int]): 随机种子，默认 None。
            用于复现结果的随机数种子（设置后结果可复现）。
    """
    # ==================== 设备配置 ====================
    device: str = 'cuda'                        # 计算设备：'cuda'(GPU), 'cpu', 'cuda:0'(指定GPU)
    
    # ==================== 训练参数 ====================
    n_epochs: int = 100                         # 训练轮数（50-200常用）
    batch_size: int = 256                       # 批次大小（256-1024，越大训练越快但需更多显存）
    learning_rate: float = 0.001                # 学习率（0.0001-0.01，Adam优化器常用0.001）
    early_stop: int = 20                        # 早停轮数（验证集性能不提升则停止）
    
    # ==================== 优化器和损失函数 ====================
    optimizer: str = 'adam'                     # 优化器：'adam'(常用), 'adamw'(带权重衰减), 'sgd'
    loss_fn: str = 'mse'                        # 损失函数：'mse'(均方误差), 'mae'(平均绝对误差), 'huber'
    weight_decay: float = 0.0                   # L2正则化系数（0.0001-0.01，防止过拟合）
    
    # ==================== 保存路径 ====================
    model_save_path: str = 'output/best_model.pth'  # 模型保存路径
    log_dir: str = 'logs'                       # 日志目录
    
    # ==================== 日志和调试 ====================
    verbose: bool = True                        # 是否显示详细训练信息
    seed: Optional[int] = None                  # 随机种子（用于复现结果）
    
    def validate(self) -> bool:
        """验证配置"""
        if self.n_epochs <= 0:
            raise ValueError("n_epochs 必须大于 0")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay 必须非负")
        if self.optimizer not in ['adam', 'adamw', 'sgd']:
            raise ValueError(f"不支持的优化器: {self.optimizer}")
        
        if self.loss_fn not in ['mse', 'mae', 'huber']:
            raise ValueError(f"不支持的损失函数: {self.loss_fn}")
        
        return True


@dataclass
class LSTMConfig(BaseModelConfig):
    """
    LSTM 模型配置
    
    长短期记忆网络，记忆能力更强，适合复杂时序模式。
    
    Args:
        d_feat (Optional[int]): 输入特征维度，默认 None。
            模型输入特征数量。为 None 时在加载数据后自动推断。
            
        hidden_size (int): LSTM 隐藏层单元数，默认 64。
            隐藏状态的维度（64-256常用范围）。越大模型容量越大但参数越多。
            
        num_layers (int): LSTM 层数，默认 2。
            堆叠 LSTM 单元的数量（1-3层，过深易过拟合）。
            
        dropout (float): Dropout 概率，默认 0.3。
            层间 dropout，防止过拟合（范围 [0, 1]，0.1-0.5 常用）。
            
        bidirectional (bool): 是否使用双向 LSTM，默认 False。
            双向 LSTM 可捕获未来信息，但参数数量翻倍，计算量增加。
            
        output_dim (int): 输出维度，默认 1。
            预测目标的维度（通常为 1，预测单个目标）。
    """
    # ==================== 模型架构参数 ====================
    d_feat: Optional[int] = None        # 输入特征维度（自动推断）
    hidden_size: int = 64               # 隐藏层单元数（64-256常用范围）
    num_layers: int = 2                 # LSTM层数（1-3层，过深易过拟合）
    dropout: float = 0.3                # Dropout概率（层间dropout，防过拟合）
    bidirectional: bool = False         # 双向LSTM（可捕获未来信息，参数x2）
    
    # ==================== 输出参数 ====================
    output_dim: int = 1                 # 输出维度（通常为1，预测单个目标）
    
    def validate(self) -> bool:
        """
        验证 LSTM 配置参数的有效性。
        """
        """验证 LSTM 配置"""
        super().validate()
        
        if self.hidden_size <= 0:
            raise ValueError("hidden_size 必须大于 0")
        
        if self.num_layers <= 0:
            raise ValueError("num_layers 必须大于 0")
        
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout 必须在 [0, 1] 范围内")
        
        return True


@dataclass
class GRUConfig(BaseModelConfig):
    """
    GRU 模型配置
    
    门控循环单元模型，参数更少，训练更快，适合快速实验。
    
    Args:
        d_feat (Optional[int]): 输入特征维度，默认 None。
            模型输入特征数量。为 None 时在加载数据后自动推断。
            
        hidden_size (int): GRU 隐藏层单元数，默认 64。
            隐藏状态的维度（越大模型容量越大）。
            
        num_layers (int): GRU 层数，默认 2。
            堆叠 GRU 单元的数量（2-3层通常效果较好）。
            
        dropout (float): Dropout 概率，默认 0.3。
            层间 dropout，防止过拟合（范围 [0, 1]，0.1-0.5 常用）。
            
        bidirectional (bool): 是否使用双向 GRU，默认 False。
            双向 GRU 可提升性能但参数翻倍，计算量增加。
            
        output_dim (int): 输出维度，默认 1。
            预测目标的维度（通常为 1，预测单个目标）。
    """
    # ==================== 模型架构参数 ====================
    d_feat: Optional[int] = None        # 输入特征维度（自动推断）
    hidden_size: int = 64               # 隐藏层单元数（越大模型容量越大）
    num_layers: int = 2                 # GRU层数（2-3层通常效果较好）
    dropout: float = 0.3                # Dropout概率（0.1-0.5，防止过拟合）
    bidirectional: bool = False         # 是否使用双向GRU（可提升性能但参数翻倍）
    
    # ==================== 输出参数 ====================
    output_dim: int = 1                 # 输出维度（预测目标数量）
    
    def validate(self) -> bool:
        """
        验证 GRU 配置参数的有效性。
        """
        """验证 GRU 配置"""
        super().validate()
        
        if self.hidden_size <= 0:
            raise ValueError("hidden_size 必须大于 0")
        
        if self.num_layers <= 0:
            raise ValueError("num_layers 必须大于 0")
        
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout 必须在 [0, 1] 范围内")
        
        return True


@dataclass
class TransformerConfig(BaseModelConfig):
    """
    Transformer 模型配置
    """
    # 模型架构
    d_feat: Optional[int] = None
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.3
    
    # 位置编码
    use_positional_encoding: bool = True
    max_seq_len: int = 60
    
    # 输出维度
    output_dim: int = 1
    
    def validate(self) -> bool:
        """
        验证 Transformer 配置参数的有效性。
        """
        """验证 Transformer 配置"""
        super().validate()
        
        if self.d_model <= 0:
            raise ValueError("d_model 必须大于 0")
        
        if self.nhead <= 0:
            raise ValueError("nhead 必须大于 0")
        
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model 必须能被 nhead 整除")
        
        if self.num_layers <= 0:
            raise ValueError("num_layers 必须大于 0")
        
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout 必须在 [0, 1] 范围内")
        
        return True


@dataclass
class VAEConfig(BaseModelConfig):
    """
    VAE (Variational Autoencoder) 模型配置
    """
    # 模型架构
    input_dim: Optional[int] = None  # 输入特征维度（自动推断）
    hidden_dim: int = 128
    latent_dim: int = 16
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = False
    
    # 编码器类型
    encoder_type: str = 'gru'  # 'gru', 'lstm', 'mlp'
    decoder_type: str = 'gru'  # 'gru', 'lstm', 'mlp'
    
    # VAE 损失权重
    alpha_recon: float = 0.1  # 重构损失权重
    beta_kl: float = 0.001  # KL 散度损失权重
    gamma_pred: float = 1.0  # 预测损失权重
    
    # 采样策略
    use_reparameterization: bool = True
    
    def validate(self) -> bool:
        """
        验证 VAE 配置参数的有效性。
        """
        """验证 VAE 配置"""
        super().validate()
        
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim 必须大于 0")
        
        if self.latent_dim <= 0:
            raise ValueError("latent_dim 必须大于 0")
        
        if self.num_layers <= 0:
            raise ValueError("num_layers 必须大于 0")
        
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout 必须在 [0, 1] 范围内")
        
        if self.encoder_type not in ['gru', 'lstm', 'mlp']:
            raise ValueError(f"不支持的编码器类型: {self.encoder_type}")
        
        if self.decoder_type not in ['gru', 'lstm', 'mlp']:
            raise ValueError(f"不支持的解码器类型: {self.decoder_type}")
        
        if self.alpha_recon < 0:
            raise ValueError("alpha_recon 必须非负")
        
        if self.beta_kl < 0:
            raise ValueError("beta_kl 必须非负")
        
        if self.gamma_pred < 0:
            raise ValueError("gamma_pred 必须非负")
        
        return True


@dataclass
class MLPConfig(BaseModelConfig):
    """
    MLP (Multi-Layer Perceptron) 模型配置
    """
    # 模型架构
    d_feat: Optional[int] = None
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.3
    batch_norm: bool = True
    activation: str = 'relu'  # 'relu', 'tanh', 'gelu'
    
    # 输出维度
    output_dim: int = 1
    
    def validate(self) -> bool:
        """
        验证 MLP 配置参数的有效性。
        """
        """验证 MLP 配置"""
        super().validate()
        
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes 不能为空")
        
        if any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes 中的所有值必须大于 0")
        
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout 必须在 [0, 1] 范围内")
        
        if self.activation not in ['relu', 'tanh', 'gelu']:
            raise ValueError(f"不支持的激活函数: {self.activation}")
        
        return True


@dataclass
class HybridGraphConfig(BaseModelConfig):
    """
    RNN+Attention+GAT+MLP 混合模型配置 (兼容模式)
    
    ⚠️ 注意: 这是整体配置类，适合快速使用但扩展性较弱。
    
    🆕 推荐使用模块化配置 (modular_config.py)：
        - 独立配置每个模块 (Temporal/Graph/Fusion)
        - 灵活组合不同的模块
        - 支持变体扩展 (如替换不同类型的Attention、GAT等)
        
        Example:
            from model.modular_config import ModelConfigBuilder
            
            config = ModelConfigBuilder() \\
                .add_temporal(rnn_type='lstm', hidden_size=64) \\
                .add_graph(gat_type='correlation', hidden_dim=32) \\
                .add_fusion(hidden_sizes=[64]) \\
                .build(d_feat=20)
    
    结合时序特征提取（RNN+Self-Attention）和截面信息交互（GAT）的混合架构。
    - RNN: 处理单只股票的时间序列特征
    - Self-Attention: 强化关键时间点权重
    - GAT: 捕捉股票间的截面关联（行业联动或相关性）
    - MLP: 融合预测器
    
    Args:
        d_feat (int): 输入特征维度（量价数据维度），默认 20。
        
        rnn_hidden (int): RNN 隐藏层大小，默认 64。
            控制时序特征提取能力（64-256常用）。
            
        rnn_layers (int): RNN 层数，默认 2。
            堆叠 LSTM 层数（2-3层效果较好）。
            
        rnn_type (str): RNN 类型，默认 'lstm'。
            - 'lstm': 长短期记忆网络（记忆能力更强）
            - 'gru': 门控循环单元（参数更少，训练更快）
            
        use_attention (bool): 是否使用 Self-Attention，默认 True。
            强化关键时间点的权重，提升时序建模能力。
            
        use_graph (bool): 是否使用图神经网络，默认 True。
            启用 GAT 进行截面信息交互。
            
        gat_heads (int): GAT 注意力头数，默认 4。
            多头注意力机制（4-8头常用）。
            
        gat_hidden (int): GAT 隐藏层维度，默认 32。
            必须能被 gat_heads 整除。
            
        gat_type (str): GAT 类型，默认 'standard'。
            - 'standard': 基于行业关系的 GAT（使用行业分类）
            - 'correlation': 基于相关性的 GAT（使用收益率相关性）
            
        top_k_neighbors (int): 相关性 GAT 的邻居数，默认 10。
            仅在 gat_type='correlation' 时有效。
            
        funda_dim (Optional[int]): 基本面数据维度，默认 None。
            如果提供基本面数据，在进入 GAT 前拼接。
            
        mlp_hidden_sizes (List[int]): MLP 隐藏层尺寸，默认 [64]。
            融合预测器的隐藏层配置。
            
        dropout (float): Dropout 概率，默认 0.3。
            全局 dropout 率（0.1-0.5）。
            
        adj_matrix_path (Optional[str]): 邻接矩阵路径，默认 None。
            预计算的邻接矩阵文件路径（.pt 或 .npy 格式）。
    """
    # ==================== 输入特征 ====================
    d_feat: int = 20                            # 输入特征维度（量价数据）
    funda_dim: Optional[int] = None             # 基本面数据维度（可选）
    
    # ==================== 时序模块 (RNN + Attention) ====================
    rnn_type: str = 'lstm'                      # RNN类型：'lstm', 'gru'
    rnn_hidden: int = 64                        # RNN隐藏层大小（64-256常用）
    rnn_layers: int = 2                         # RNN层数（2-3层效果较好）
    use_attention: bool = True                  # 是否使用Self-Attention
    
    # ==================== 截面模块 (GAT) ====================
    use_graph: bool = True                      # 是否使用图神经网络
    gat_type: str = 'standard'                  # GAT类型：'standard'(行业), 'correlation'(相关性)
    gat_heads: int = 4                          # GAT注意力头数（4-8头常用）
    gat_hidden: int = 32                        # GAT隐藏层维度（必须能被gat_heads整除）
    top_k_neighbors: int = 10                   # 相关性GAT的邻居数（gat_type='correlation'时使用）
    
    # ==================== 融合模块 (MLP) ====================
    mlp_hidden_sizes: List[int] = field(default_factory=lambda: [64])  # MLP隐藏层尺寸
    
    # ==================== 正则化 ====================
    dropout: float = 0.3                        # 全局Dropout概率（0.1-0.5）
    
    # ==================== 邻接矩阵 ====================
    adj_matrix_path: Optional[str] = None       # 邻接矩阵路径（.pt或.npy格式）
    
    # ==================== 输出参数 ====================
    output_dim: int = 1                         # 输出维度（通常为1，预测单个目标）
    
    def validate(self) -> bool:
        """
        验证 HybridGraph 配置参数的有效性。
        """
        super().validate()
        
        # 验证 RNN 参数
        if self.rnn_hidden <= 0:
            raise ValueError("rnn_hidden 必须大于 0")
        
        if self.rnn_layers <= 0:
            raise ValueError("rnn_layers 必须大于 0")
        
        if self.rnn_type not in ['lstm', 'gru']:
            raise ValueError(f"不支持的 RNN 类型: {self.rnn_type}")
        
        # 验证 GAT 参数
        if self.use_graph:
            if self.gat_hidden <= 0:
                raise ValueError("gat_hidden 必须大于 0")
            
            if self.gat_heads <= 0:
                raise ValueError("gat_heads 必须大于 0")
            
            if self.gat_hidden % self.gat_heads != 0:
                raise ValueError("gat_hidden 必须能被 gat_heads 整除")
            
            if self.gat_type not in ['standard', 'correlation']:
                raise ValueError(f"不支持的 GAT 类型: {self.gat_type}")
            
            if self.gat_type == 'correlation' and self.top_k_neighbors <= 0:
                raise ValueError("top_k_neighbors 必须大于 0")
        
        # 验证 MLP 参数
        if not self.mlp_hidden_sizes:
            raise ValueError("mlp_hidden_sizes 不能为空")
        
        if any(size <= 0 for size in self.mlp_hidden_sizes):
            raise ValueError("mlp_hidden_sizes 中的所有值必须大于 0")
        
        # 验证 dropout
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout 必须在 [0, 1] 范围内")
        
        return True


# 配置工厂
class ModelConfigFactory:
    """
    模型配置工厂 - 根据模型类型创建配置
    """
    
    _config_map = {
        'lstm': LSTMConfig,
        'gru': GRUConfig,
        'transformer': TransformerConfig,
        'vae': VAEConfig,
        'mlp': MLPConfig,
        'hybrid_graph': HybridGraphConfig,
    }
    
    @classmethod
    def create(cls, model_type: str, **kwargs) -> BaseModelConfig:
        """
        创建模型配置
        
        Args:
            model_type: 模型类型（'lstm', 'gru', 'transformer', 'vae', 'mlp'）
            **kwargs: 配置参数
            
        Returns:
            模型配置对象
            
        Example:
            config = ModelConfigFactory.create('vae', hidden_dim=256, latent_dim=32)
        """
        model_type = model_type.lower()
        
        if model_type not in cls._config_map:
            raise ValueError(
                f"不支持的模型类型: {model_type}\n"
                f"支持的类型: {list(cls._config_map.keys())}"
            )
        
        config_class = cls._config_map[model_type]
        return config_class(**kwargs)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> BaseModelConfig:
        """
        从字典创建配置（自动检测模型类型）
        
        Args:
            config_dict: 配置字典，必须包含 'model_type' 字段
            
        Returns:
            模型配置对象
        """
        if 'model_type' not in config_dict:
            raise ValueError("配置字典必须包含 'model_type' 字段")
        
        model_type = config_dict.pop('model_type')
        return cls.create(model_type, **config_dict)
    
    @classmethod
    def get_template(cls, model_type: str, template: str = 'default') -> BaseModelConfig:
        """
        获取预定义模板
        
        Args:
            model_type: 模型类型
            template: 模板名称（'default', 'small', 'large'）
            
        Returns:
            模型配置对象
        """
        templates = {
            'vae': {
                'default': VAEConfig(),
                'small': VAEConfig(
                    hidden_dim=64,
                    latent_dim=8,
                    num_layers=1,
                    n_epochs=50,
                ),
                'large': VAEConfig(
                    hidden_dim=256,
                    latent_dim=32,
                    num_layers=3,
                    n_epochs=200,
                ),
            },
            'lstm': {
                'default': LSTMConfig(),
                'small': LSTMConfig(hidden_size=32, num_layers=1),
                'large': LSTMConfig(hidden_size=128, num_layers=3),
            },
            'gru': {
                'default': GRUConfig(),
                'small': GRUConfig(hidden_size=32, num_layers=1),
                'large': GRUConfig(hidden_size=128, num_layers=3),
            },
            'transformer': {
                'default': TransformerConfig(),
                'small': TransformerConfig(d_model=32, nhead=2, num_layers=1),
                'large': TransformerConfig(d_model=128, nhead=8, num_layers=4),
            },
            'mlp': {
                'default': MLPConfig(),
                'small': MLPConfig(hidden_sizes=[64]),
                'large': MLPConfig(hidden_sizes=[256, 128, 64]),
            },
            'hybrid_graph': {
                'default': HybridGraphConfig(),
                'small': HybridGraphConfig(
                    rnn_hidden=32,
                    rnn_layers=1,
                    gat_hidden=16,
                    gat_heads=2,
                    mlp_hidden_sizes=[32],
                    n_epochs=50,
                ),
                'large': HybridGraphConfig(
                    rnn_hidden=128,
                    rnn_layers=3,
                    gat_hidden=64,
                    gat_heads=8,
                    mlp_hidden_sizes=[128, 64],
                    n_epochs=200,
                ),
            },
        }
        
        model_type = model_type.lower()
        
        if model_type not in templates:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        if template not in templates[model_type]:
            raise ValueError(
                f"不支持的模板: {template}\n"
                f"可用模板: {list(templates[model_type].keys())}"
            )
        
        return templates[model_type][template]


if __name__ == '__main__':
    # 测试模型配置
    print("=" * 80)
    print("ModelConfig 测试")
    print("=" * 80)
    
    # 测试 1: 创建 VAE 配置
    print("\n1. 创建 VAE 配置:")
    vae_config = VAEConfig(
        hidden_dim=128,
        latent_dim=16,
        n_epochs=100,
        learning_rate=0.001,
    )
    print(f"  hidden_dim: {vae_config.hidden_dim}")
    print(f"  latent_dim: {vae_config.latent_dim}")
    print(f"  optimizer: {vae_config.optimizer}")
    
    # 测试 2: 保存和加载 YAML
    print("\n2. YAML 序列化:")
    yaml_path = '/tmp/vae_config.yaml'
    vae_config.to_yaml(yaml_path)
    print(f"  已保存到: {yaml_path}")
    
    vae_config2 = VAEConfig.from_yaml(yaml_path)
    print(f"  已加载: latent_dim={vae_config2.latent_dim}")
    
    # 测试 3: 转换为字典
    print("\n3. 转换为字典:")
    config_dict = vae_config.to_dict()
    print(f"  keys: {list(config_dict.keys())[:5]}...")
    
    # 测试 4: 配置工厂
    print("\n4. 使用配置工厂:")
    lstm_config = ModelConfigFactory.create('lstm', hidden_size=128)
    print(f"  LSTM: {lstm_config.hidden_size}")
    
    # 测试 5: 获取模板
    print("\n5. 使用模板:")
    small_vae = ModelConfigFactory.get_template('vae', 'small')
    print(f"  小型 VAE: hidden_dim={small_vae.hidden_dim}, latent_dim={small_vae.latent_dim}")
    
    large_vae = ModelConfigFactory.get_template('vae', 'large')
    print(f"  大型 VAE: hidden_dim={large_vae.hidden_dim}, latent_dim={large_vae.latent_dim}")
    
    # 测试 6: 配置验证
    print("\n6. 配置验证:")
    try:
        invalid_config = VAEConfig(hidden_dim=-10)
    except ValueError as e:
        print(f"  ✅ 捕获到验证错误: {e}")
    
    # 测试 7: 更新配置
    print("\n7. 更新配置:")
    vae_config.update(learning_rate=0.002, n_epochs=150)
    print(f"  新学习率: {vae_config.learning_rate}")
    print(f"  新训练轮数: {vae_config.n_epochs}")
    
    print("\n" + "=" * 80)
    print("✅ ModelConfig 测试完成")
    print("=" * 80)
