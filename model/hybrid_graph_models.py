"""
Hybrid Graph Models - 混合图神经网络模型

实现 RNN+Attention+GAT 的混合架构，结合时序特征提取和截面信息交互。

模型架构:
1. 时序提取器 (TemporalBlock): RNN + Self-Attention
   - 处理单只股票的时间序列数据
   - 捕捉长期依赖和关键时间点
   
2. 截面交互器 (GraphBlock): GAT
   - 基于行业或相关性构建邻接矩阵
   - 通过注意力机制聚合邻居信息
   
3. 融合预测器 (FusionBlock): MLP
   - 融合时序特征和图特征
   - 输出最终预测结果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union, Dict, List
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict

from .base_model import PyTorchModel
from .model_factory import register_model


# ==================== Graph Inference Mode Constants ====================
GRAPH_MODE_BATCH = 'batch'                    # 原始模式：直接切片邻接矩阵
GRAPH_MODE_CROSS_SECTIONAL = 'cross_sectional'  # 截面模式：按日期分组
GRAPH_MODE_NEIGHBOR_SAMPLING = 'neighbor_sampling'  # 邻居采样模式


# ==================== 子模块 1: 时序特征提取 ====================

class TemporalBlock(nn.Module):
    """
    时序特征提取模块 (RNN + Self-Attention + Residual)
    
    负责从单只股票的时间序列中提取时序特征。
    🆕 支持残差连接，帮助梯度传播，防止深层网络中的梯度消失。
    
    Args:
        d_feat (int): 输入特征维度
        hidden_size (int): RNN隐藏层大小
        num_layers (int): RNN层数
        rnn_type (str): RNN类型 ('lstm' 或 'gru')
        dropout (float): Dropout概率
        use_attention (bool): 是否使用Self-Attention
        use_residual (bool): 🆕 是否使用残差连接
    """
    
    def __init__(
        self,
        d_feat: int,
        hidden_size: int,
        num_layers: int,
        rnn_type: str = 'lstm',
        dropout: float = 0.3,
        use_attention: bool = True,
        use_residual: bool = True  # 🆕 残差连接开关
    ):
        super().__init__()
        
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.hidden_size = hidden_size
        
        # 🆕 输入投影层：将 d_feat 映射到 hidden_size，用于残差连接
        # 只有当 d_feat != hidden_size 时才需要
        if use_residual:
            self.input_proj = nn.Linear(d_feat, hidden_size) if d_feat != hidden_size else nn.Identity()
            self.layer_norm = nn.LayerNorm(hidden_size)
        
        # RNN层
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"不支持的RNN类型: {rnn_type}")
        
        # Self-Attention层
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1)
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, d_feat]
            
        Returns:
            time_feat: [batch_size, hidden_size] - 时序特征向量
        """
        # 🆕 保存输入用于残差连接（取最后一个时间步并投影）
        if self.use_residual:
            # 将最后一个时间步的输入投影到 hidden_size
            residual = self.input_proj(x[:, -1, :])  # [batch, hidden_size]
        
        # RNN编码
        rnn_out, _ = self.rnn(x)  # [batch, seq_len, hidden]
        
        if self.use_attention:
            # Self-Attention加权
            scores = self.attention(rnn_out)  # [batch, seq_len, 1]
            weights = F.softmax(scores, dim=1)  # 归一化注意力权重
            
            # 加权求和，压缩时间维度
            context = torch.sum(rnn_out * weights, dim=1)  # [batch, hidden]
        else:
            # 直接取最后一个时间步
            context = rnn_out[:, -1, :]  # [batch, hidden]
        
        # 🆕 残差连接 + LayerNorm
        if self.use_residual:
            context = self.layer_norm(context + residual)
        
        return self.dropout(context)


# ==================== 子模块 2: 截面特征提取 ====================

class GraphBlock(nn.Module):
    """
    截面特征提取模块 (GAT - Graph Attention Network + Residual)
    
    通过图注意力机制捕捉股票间的截面关联。
    🆕 支持残差连接，增强模型稳定性。
    
    Args:
        in_dim (int): 输入特征维度
        out_dim (int): 输出特征维度
        heads (int): 注意力头数
        dropout (float): Dropout概率
        concat (bool): 是否拼接多头输出（False时取平均）
        use_residual (bool): 🆕 是否使用残差连接
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.3,
        concat: bool = True,
        use_residual: bool = True  # 🆕 残差连接开关
    ):
        super().__init__()
        
        assert out_dim % heads == 0, "out_dim 必须能被 heads 整除"
        
        self.heads = heads
        self.head_dim = out_dim // heads if concat else out_dim
        self.concat = concat
        self.dropout = dropout
        self.use_residual = use_residual
        self.out_dim = out_dim
        
        # 🆕 残差连接：投影层 + LayerNorm
        if use_residual:
            self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
            self.layer_norm = nn.LayerNorm(out_dim)
        
        # 多头注意力参数
        self.W = nn.ModuleList([
            nn.Linear(in_dim, self.head_dim, bias=False)
            for _ in range(heads)
        ])
        
        # 注意力系数计算
        self.a = nn.ModuleList([
            nn.Linear(2 * self.head_dim, 1, bias=False)
            for _ in range(heads)
        ])
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [num_stocks, in_dim] - 节点特征
            adj: [num_stocks, num_stocks] - 邻接矩阵（0/1或权重）
            
        Returns:
            graph_feat: [num_stocks, out_dim] - 图特征
        """
        N = x.size(0)  # 股票数量
        
        # 多头注意力计算
        multi_head_outputs = []
        
        for i in range(self.heads):
            # 线性变换
            h = self.W[i](x)  # [N, head_dim]
            
            # 计算注意力系数
            # 构造 [N, N, 2*head_dim] 的拼接矩阵
            h_repeat = h.repeat(N, 1, 1)  # [N, N, head_dim]
            h_repeat_transpose = h.repeat(1, N).view(N, N, -1)  # [N, N, head_dim]
            h_concat = torch.cat([h_repeat, h_repeat_transpose], dim=-1)  # [N, N, 2*head_dim]
            
            # 计算注意力分数
            e = self.leakyrelu(self.a[i](h_concat).squeeze(-1))  # [N, N]
            
            # 仅保留邻接矩阵中的边（mask掉非邻居）
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            
            # Softmax归一化
            attention = F.softmax(attention, dim=1)
            attention = self.dropout_layer(attention)
            
            # 加权聚合邻居特征
            h_prime = torch.matmul(attention, h)  # [N, head_dim]
            multi_head_outputs.append(h_prime)
        
        # 多头融合
        if self.concat:
            output = torch.cat(multi_head_outputs, dim=-1)  # [N, heads*head_dim=out_dim]
        else:
            output = torch.mean(torch.stack(multi_head_outputs), dim=0)  # [N, out_dim]
        
        # 🆕 残差连接 + LayerNorm
        if self.use_residual:
            residual = self.residual_proj(x)  # [N, out_dim]
            output = self.layer_norm(F.elu(output) + residual)
            return output
        else:
            return F.elu(output)


# ==================== 子模块 3: 融合预测器 ====================

class FusionBlock(nn.Module):
    """
    融合预测模块 (MLP)
    
    将时序特征和图特征融合后进行预测。
    🆕 支持多因子输出：当 output_dim > 1 时，输出 N×F 矩阵（研报 baseline）
    🆕 支持残差连接：增强深层 MLP 的梯度传播
    
    Args:
        input_dim (int): 输入维度
        hidden_sizes (List[int]): 隐藏层尺寸列表
        output_dim (int): 输出维度（因子数量 F），默认 1
        dropout (float): Dropout概率
        use_residual (bool): 🆕 是否使用残差连接
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list,
        output_dim: int = 1,
        dropout: float = 0.3,
        use_residual: bool = False  # 🆕 残差连接开关
    ):
        super().__init__()
        
        self.output_dim = output_dim  # 🆕 保存输出维度
        self.use_residual = use_residual  # 🆕 残差连接开关
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # 🆕 残差连接：投影层 + LayerNorm
        if use_residual:
            # 投影层：将输入维度对齐到输出维度
            self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
            self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, input_dim] - 融合特征
            
        Returns:
            output: 
                - 当 output_dim=1 时: [batch_size] - 单因子预测
                - 当 output_dim>1 时: [batch_size, output_dim] - 多因子矩阵 (N×F)
        """
        out = self.mlp(x)
        
        # 🆕 残差连接 + LayerNorm
        if self.use_residual:
            residual = self.residual_proj(x)
            out = self.layer_norm(out + residual)
        
        # 🆕 只有单因子时才 squeeze，多因子保持 N×F 格式
        if self.output_dim == 1:
            return out.squeeze(-1)
        return out  # [batch_size, F]


# ==================== 主模型: HybridNet ====================

class HybridNet(nn.Module):
    """
    混合图神经网络 (RNN+Attention+GAT+MLP)
    
    组合时序模块、图模块和融合模块。
    🆕 支持残差连接，增强模型训练稳定性。
    
    工作流程:
    1. 时序提取: 对每只股票的时间序列提取时序特征
    2. 特征融合: (可选) 拼接基本面数据
    3. 截面交互: 通过GAT进行跨股票的信息聚合
    4. 融合预测: MLP输出最终预测
    """
    
    def __init__(
        self,
        d_feat: int,
        rnn_hidden: int,
        rnn_layers: int,
        rnn_type: str = 'lstm',
        use_attention: bool = True,
        use_graph: bool = True,
        gat_hidden: int = 32,
        gat_heads: int = 4,
        mlp_hidden_sizes: list = None,
        funda_dim: Optional[int] = None,
        dropout: float = 0.3,
        output_dim: int = 1,
        use_residual: bool = True  # 🆕 残差连接开关
    ):
        super().__init__()
        
        self.use_graph = use_graph
        self.funda_dim = funda_dim
        self.use_residual = use_residual
        
        # 1. 时序模块 (🆕 支持残差连接)
        self.temporal = TemporalBlock(
            d_feat=d_feat,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            rnn_type=rnn_type,
            dropout=dropout,
            use_attention=use_attention,
            use_residual=use_residual  # 🆕 传递残差连接参数
        )
        
        # 计算进入GAT前的特征维度
        graph_input_dim = rnn_hidden
        if funda_dim is not None:
            graph_input_dim += funda_dim
        
        # 2. 图模块 (🆕 支持残差连接)
        if use_graph:
            self.gat = GraphBlock(
                in_dim=graph_input_dim,
                out_dim=gat_hidden,
                heads=gat_heads,
                dropout=dropout,
                concat=True,
                use_residual=use_residual  # 🆕 传递残差连接参数
            )
            mlp_input_dim = rnn_hidden + gat_hidden  # 时序特征 + 图特征
        else:
            mlp_input_dim = rnn_hidden
        
        # 如果有基本面数据，最终预测时也要拼接
        if funda_dim is not None:
            mlp_input_dim += funda_dim
        
        # 3. 融合模块 (🆕 支持残差连接)
        if mlp_hidden_sizes is None:
            mlp_hidden_sizes = [64]
        
        self.fusion = FusionBlock(
            input_dim=mlp_input_dim,
            hidden_sizes=mlp_hidden_sizes,
            output_dim=output_dim,
            dropout=dropout,
            use_residual=use_residual  # 🆕 传递残差连接参数
        )
    
    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        funda: Optional[torch.Tensor] = None,
        return_hidden: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, d_feat] - 时序量价数据
            adj: [batch_size, batch_size] - 邻接矩阵 (可选)
            funda: [batch_size, funda_dim] - 基本面数据 (可选)
            return_hidden: 是否返回隐藏特征（用于相关性正则化）
            
        Returns:
            如果 return_hidden=False:
                pred: [batch_size] - 预测结果
            如果 return_hidden=True:
                (pred, time_feat, combined) - 预测结果、时序特征、融合特征
        """
        # Step 1: 提取时序特征
        time_feat = self.temporal(x)  # [batch, rnn_hidden]
        
        # Step 2: 特征融合 (拼接基本面)
        if self.use_graph:
            # 处理 adj 为 None 的情况
            current_adj = adj
            if current_adj is None:
                # 使用单位矩阵（自环），确保维度匹配
                # 这样即使没有邻接矩阵，模型也能运行（退化为无图交互）
                current_adj = torch.eye(x.size(0), device=x.device)

            if funda is not None:
                # 拼接基本面数据作为图的输入
                graph_input = torch.cat([time_feat, funda], dim=-1)
            else:
                graph_input = time_feat
            
            # Step 3: 图特征提取
            graph_feat = self.gat(graph_input, current_adj)  # [batch, gat_hidden]
            
            # Step 4: 拼接时序特征和图特征
            if funda is not None:
                combined = torch.cat([time_feat, graph_feat, funda], dim=-1)
            else:
                combined = torch.cat([time_feat, graph_feat], dim=-1)
        else:
            # 不使用图网络，直接使用时序特征
            if funda is not None:
                combined = torch.cat([time_feat, funda], dim=-1)
            else:
                combined = time_feat
        
        # Step 5: 融合预测
        pred = self.fusion(combined)
        
        if return_hidden:
            return pred, time_feat, combined
        return pred


# ==================== PyTorch模型封装 ====================

@register_model('hybrid_graph')
@register_model('HybridGraph')
class HybridGraphModel(PyTorchModel):
    """
    混合图神经网络模型 (RNN+Attention+GAT+MLP)
    
    结合时序特征提取和截面信息交互的混合架构，适用于股票收益预测。
    
    特点:
    - RNN+Self-Attention: 捕捉单只股票的时序规律
    - GAT: 学习股票间的截面关联（行业联动/相关性）
    - 灵活的融合机制: 支持拼接基本面数据
    
    Example:
        # 创建模型
        model = HybridGraphModel(
            d_feat=20,
            rnn_hidden=64,
            rnn_layers=2,
            use_graph=True,
            gat_hidden=32,
            gat_heads=4,
            adj_matrix_path='adj_matrix.pt',
            n_epochs=100
        )
        
        # 训练
        model.fit(train_loader, valid_loader)
        
        # 预测
        predictions = model.predict(test_loader)
        
        # 🆕 使用邻居采样模式预测（保持图结构完整性）
        predictions = model.predict(test_loader, graph_inference_mode='neighbor_sampling')
    
    Graph Inference Modes:
        - 'batch' (默认): 直接按 batch 切片邻接矩阵，可能丢失邻居信息
        - 'cross_sectional': 按日期分组，同一天所有股票一起推理
        - 'neighbor_sampling': 邻居采样模式，主动拉取缺失邻居的缓存特征
    """
    
    def __init__(
        self,
        d_feat: int = 20,
        rnn_hidden: int = 64,
        rnn_layers: int = 2,
        rnn_type: str = 'lstm',
        use_attention: bool = True,
        use_graph: bool = True,
        gat_type: str = 'standard',
        gat_hidden: int = 32,
        gat_heads: int = 4,
        top_k_neighbors: int = 10,
        mlp_hidden_sizes: list = None,
        funda_dim: Optional[int] = None,
        dropout: float = 0.3,
        adj_matrix_path: Optional[str] = None,
        use_residual: bool = True,  # 🆕 残差连接开关
        # 🆕 多因子输出参数
        output_dim: int = 1,  # 🆕 输出维度（因子数量 F），默认 1，设置为 F>1 时输出 N×F 矩阵
        # 🆕 Graph-Aware Inference 参数
        graph_inference_mode: str = 'batch',  # 'batch', 'cross_sectional', 'neighbor_sampling'
        max_neighbors: int = 10,  # 邻居采样时的最大邻居数
        cache_size: int = 5000,   # 节点状态缓存大小
        # 🆕 数据格式参数
        stock_idx_position: Optional[int] = None,  # 股票索引在 batch 中的位置（0-indexed）
        funda_position: Optional[int] = None,      # 基本面数据在 batch 中的位置（0-indexed）
        # 🆕 相关性正则化参数
        lambda_corr: float = 0.0,  # 相关性正则化权重，0表示不使用
        **kwargs
    ):
        """
        Args:
            d_feat: 输入特征维度
            rnn_hidden: RNN隐藏层大小
            rnn_layers: RNN层数
            rnn_type: RNN类型 ('lstm' 或 'gru')
            use_attention: 是否使用Self-Attention
            use_graph: 是否使用图神经网络
            gat_type: GAT类型 ('standard' 或 'correlation')
            gat_hidden: GAT隐藏层维度
            gat_heads: GAT注意力头数
            top_k_neighbors: 相关性GAT的邻居数
            mlp_hidden_sizes: MLP隐藏层尺寸列表
            funda_dim: 基本面数据维度
            dropout: Dropout概率
            adj_matrix_path: 邻接矩阵路径
            use_residual: 🆕 是否使用残差连接 (LayerNorm + Skip Connection)
            output_dim: 🆕 输出维度（因子数量 F），默认 1，设置为 F>1 时输出 N×F 矩阵
            lambda_corr: 🆕 相关性正则化权重，0表示不使用
        """
        # 🆕 将 lambda_corr 传递给基类
        super().__init__(lambda_corr=lambda_corr, **kwargs)
        
        # 保存配置
        self.d_feat = d_feat
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        self.use_attention = use_attention
        self.use_graph = use_graph
        self.gat_type = gat_type
        self.gat_hidden = gat_hidden
        self.gat_heads = gat_heads
        self.top_k_neighbors = top_k_neighbors
        self.mlp_hidden_sizes = mlp_hidden_sizes or [64]
        self.funda_dim = funda_dim
        self.dropout_rate = dropout
        self.adj_matrix_path = adj_matrix_path
        self.use_residual = use_residual  # 🆕 保存残差连接配置
        self.output_dim = output_dim  # 🆕 保存输出维度（多因子）
        
        # 🆕 Graph-Aware Inference 配置
        self.graph_inference_mode = graph_inference_mode
        self.max_neighbors = max_neighbors
        self.cache_size = cache_size
        
        # 🆕 数据格式配置（明确指定位置，避免启发式猜测）
        self.stock_idx_position = stock_idx_position
        self.funda_position = funda_position
        
        # 🆕 节点状态缓存 (LRU Cache)
        # Key: stock_idx (int), Value: temporal_feature (Tensor)
        self._node_state_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        
        # 🆕 预计算邻居索引缓存（避免重复计算 Top-K）
        self._neighbor_cache: Optional[Dict[int, torch.Tensor]] = None
        
        # 创建模型 (🆕 支持残差连接 + 多因子输出)
        self.model = HybridNet(
            d_feat=d_feat,
            rnn_hidden=rnn_hidden,
            rnn_layers=rnn_layers,
            rnn_type=rnn_type,
            use_attention=use_attention,
            use_graph=use_graph,
            gat_hidden=gat_hidden,
            gat_heads=gat_heads,
            mlp_hidden_sizes=self.mlp_hidden_sizes,
            funda_dim=funda_dim,
            dropout=dropout,
            output_dim=output_dim,  # 🆕 传递输出维度
            use_residual=use_residual  # 🆕 传递残差连接参数
        ).to(self.device)
        
        # 创建优化器和损失函数
        self.optimizer = self._get_optimizer()
        self.criterion = self._get_loss_fn()
        
        # 加载邻接矩阵；动态图模式下允许为空（按 batch 提供 adj）
        self.adj_matrix = self._load_adj_matrix(adj_matrix_path)

        # 仅在需要静态邻接但未加载时给出警告；
        # 如果用户走日批次动态图（每个 batch 传入 adj），不再误报。
        expects_static_adj = self.use_graph and adj_matrix_path is not None
        using_dynamic_adj = self.use_graph and adj_matrix_path is None and self.graph_inference_mode == GRAPH_MODE_BATCH

        if expects_static_adj and self.adj_matrix is None:
            self.logger.warning("⚠️ 启用了图网络 (use_graph=True) 但未加载邻接矩阵。")
            self.logger.warning("   模型将使用单位矩阵（仅自环），这意味着没有跨股票的信息交互。")
            self.logger.warning("   若需使用图特征，请提供 adj_matrix_path。")
        elif using_dynamic_adj and self.adj_matrix is None:
            self.logger.info("🟢 动态图模式：未加载静态邻接矩阵，训练/推理将使用 batch 内提供的 adj。")
        
        self.logger.info(f"HybridGraph 模型参数:")
        self.logger.info(f"  特征维度: {d_feat}")
        self.logger.info(f"  RNN类型: {rnn_type}, 隐藏层: {rnn_hidden}, 层数: {rnn_layers}")
        self.logger.info(f"  使用Attention: {use_attention}")
        self.logger.info(f"  使用图网络: {use_graph}")
        self.logger.info(f"  🆕 残差连接: {use_residual}")  # 🆕 显示残差连接状态
        if use_graph:
            self.logger.info(f"  GAT类型: {gat_type}, 隐藏层: {gat_hidden}, 头数: {gat_heads}")
        if funda_dim:
            self.logger.info(f"  基本面维度: {funda_dim}")
        self.logger.info(f"  MLP隐藏层: {self.mlp_hidden_sizes}")
        
        # 🆕 显示 Graph-Aware Inference 配置
        self.logger.info(f"  🆕 图推理模式: {graph_inference_mode}")
        if graph_inference_mode == GRAPH_MODE_NEIGHBOR_SAMPLING:
            self.logger.info(f"     最大邻居数: {max_neighbors}")
            self.logger.info(f"     缓存大小: {cache_size}")
        
        # 🆕 显示数据格式配置
        if stock_idx_position is not None:
            self.logger.info(f"  🆕 股票索引位置: batch[{stock_idx_position}]")
        if funda_position is not None:
            self.logger.info(f"  🆕 基本面数据位置: batch[{funda_position}]")

    def to(self, device):
        """将底层 nn.Module 迁移到目标设备并同步更新自身 device."""
        target_device = torch.device(device) if not isinstance(device, torch.device) else device
        self.device = target_device
        self.model = self.model.to(self.device)
        if self.adj_matrix is not None:
            self.adj_matrix = self.adj_matrix.to(self.device)
        return self
    
    def parameters(self):
        """委托给内部 nn.Module 的 parameters 方法，供外部优化器使用"""
        return self.model.parameters()
    
    def named_parameters(self):
        """委托给内部 nn.Module 的 named_parameters 方法"""
        return self.model.named_parameters()
    
    def state_dict(self):
        """委托给内部 nn.Module 的 state_dict 方法"""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict, strict=True):
        """委托给内部 nn.Module 的 load_state_dict 方法"""
        return self.model.load_state_dict(state_dict, strict=strict)
    
    def train(self, mode=True):
        """委托给内部 nn.Module 的 train 方法，设置训练模式"""
        self.model.train(mode)
        return self
    
    def eval(self):
        """委托给内部 nn.Module 的 eval 方法，设置评估模式"""
        self.model.eval()
        return self
    
    def __call__(self, *args, **kwargs):
        """使封装类可以像 nn.Module 一样被调用"""
        return self.forward(*args, **kwargs)
    
    def forward(self, x, adj=None, funda=None, return_hidden=False):
        """委托给内部 nn.Module 的 forward 方法"""
        return self.model(x, adj, funda, return_hidden)
    
    # ==================== 🆕 Node State Cache Methods ====================
    
    def _update_node_cache(self, stock_idx: torch.Tensor, temporal_features: torch.Tensor):
        """
        更新节点状态缓存
        
        在每次 forward 后调用，将计算出的时序特征存入缓存，
        供后续 batch 的邻居采样使用。
        
        Args:
            stock_idx: [batch_size] 股票全局索引
            temporal_features: [batch_size, hidden_dim] 时序特征
        """
        if stock_idx is None or temporal_features is None:
            return
        
        # 转换为 CPU 以节省 GPU 显存
        features_cpu = temporal_features.detach().cpu()
        indices = stock_idx.cpu().tolist()
        
        for idx, feat in zip(indices, features_cpu):
            # 如果已存在，先删除再添加（移到末尾，LRU）
            if idx in self._node_state_cache:
                del self._node_state_cache[idx]
            self._node_state_cache[idx] = feat
            
            # 超出容量时，删除最旧的条目
            if len(self._node_state_cache) > self.cache_size:
                self._node_state_cache.popitem(last=False)
    
    def _fetch_cached_features(
        self, 
        neighbor_idx: torch.Tensor, 
        fallback_feature: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        从缓存中获取邻居特征
        
        Args:
            neighbor_idx: [num_neighbors] 需要获取的邻居索引
            fallback_feature: 缓存未命中时的后备特征（如当前 batch 的均值）
            
        Returns:
            [num_neighbors, hidden_dim] 邻居特征
        """
        features = []
        indices = neighbor_idx.cpu().tolist()
        
        for idx in indices:
            if idx in self._node_state_cache:
                features.append(self._node_state_cache[idx])
            elif fallback_feature is not None:
                # 确保 fallback_feature 在 CPU 上，以便与缓存中的特征（也在 CPU 上）一致
                features.append(fallback_feature.cpu())
            else:
                # 使用零向量作为最终后备
                features.append(torch.zeros(self.rnn_hidden))
        
        return torch.stack(features).to(self.device)
    
    def clear_node_cache(self):
        """清空节点状态缓存"""
        self._node_state_cache.clear()
        self.logger.debug("节点状态缓存已清空")
    
    def _precompute_neighbors(self):
        """
        预计算每个节点的 Top-K 邻居
        
        避免在每次 forward 时重复计算，提升效率。
        """
        if self.adj_matrix is None:
            return
        
        if self._neighbor_cache is not None:
            return  # 已经计算过
        
        self.logger.info(f"预计算 Top-{self.max_neighbors} 邻居索引...")
        self._neighbor_cache = {}
        
        N = self.adj_matrix.size(0)
        for i in range(N):
            # 获取第 i 个节点的所有边权重
            weights = self.adj_matrix[i]
            # 排除自环
            weights[i] = 0
            # Top-K
            k = min(self.max_neighbors, (weights > 0).sum().item())
            if k > 0:
                _, topk_idx = torch.topk(weights, k)
                self._neighbor_cache[i] = topk_idx
            else:
                self._neighbor_cache[i] = torch.tensor([], dtype=torch.long, device=self.device)
        
        self.logger.info(f"邻居索引预计算完成，共 {N} 个节点")
    
    # ==================== 🆕 Unified Forward Step ====================
    
    def _forward_step(
        self,
        batch_x: torch.Tensor,
        stock_idx: Optional[torch.Tensor],
        batch_funda: Optional[torch.Tensor],
        mode: Optional[str] = None,
        update_cache: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        统一的前向传播步骤
        
        将 temporal -> graph -> fusion 的流程封装，供训练、验证、预测共用。
        解决训练与推理计算图不一致的问题。
        
        Args:
            batch_x: [batch_size, seq_len, d_feat] 输入特征
            stock_idx: [batch_size] 股票全局索引（可选）
            batch_funda: [batch_size, funda_dim] 基本面数据（可选）
            mode: 图推理模式，None 时使用默认配置
            update_cache: 是否更新节点缓存
            
        Returns:
            (pred, time_feat)
            - pred: [batch_size] 预测结果
            - time_feat: [batch_size, hidden] 时序特征（用于缓存）
        """
        mode = mode or self.graph_inference_mode
        batch_size = batch_x.size(0)
        
        # Step 1: 提取时序特征
        time_feat = self.model.temporal(batch_x)  # [batch, hidden]
        
        # Step 2: 更新节点缓存（用于后续 batch 的邻居采样）
        if update_cache and stock_idx is not None:
            self._update_node_cache(stock_idx, time_feat.detach())
        
        # Step 3: 准备图上下文并运行 GAT
        if self.use_graph:
            augmented_x, augmented_adj, original_batch_size = self._prepare_graph_context(
                time_feat, stock_idx, mode
            )
            
            # 🆕 处理基本面数据维度问题：对外部邻居填充零
            if batch_funda is not None:
                num_external = augmented_x.size(0) - original_batch_size
                if num_external > 0:
                    # 为外部邻居创建零填充的基本面数据
                    external_funda = torch.zeros(
                        num_external, batch_funda.size(-1),
                        device=batch_funda.device, dtype=batch_funda.dtype
                    )
                    augmented_funda = torch.cat([batch_funda, external_funda], dim=0)
                    graph_input = torch.cat([augmented_x, augmented_funda], dim=-1)
                else:
                    graph_input = torch.cat([augmented_x, batch_funda], dim=-1)
            else:
                graph_input = augmented_x
            
            # 运行 GAT
            graph_feat = self.model.gat(graph_input, augmented_adj)
            
            # 只取原始 batch 的 GAT 输出
            graph_feat = graph_feat[:original_batch_size]
            time_feat_for_fusion = time_feat[:original_batch_size]
            
            # Step 4: 融合特征
            if batch_funda is not None:
                combined = torch.cat([time_feat_for_fusion, graph_feat, batch_funda], dim=-1)
            else:
                combined = torch.cat([time_feat_for_fusion, graph_feat], dim=-1)
        else:
            # 不使用图网络
            if batch_funda is not None:
                combined = torch.cat([time_feat, batch_funda], dim=-1)
            else:
                combined = time_feat
        
        # Step 5: MLP 预测
        pred = self.model.fusion(combined)
        
        # 🆕 处理预测结果维度
        # - 多因子模式 (output_dim > 1): 保持 [batch, F] 形状
        # - 单因子模式 (output_dim == 1): squeeze 为 [batch]
        if pred.dim() == 2 and pred.size(-1) == 1:
            # 单因子，squeeze 最后一维
            pred = pred.squeeze(-1)
        elif pred.dim() == 0:
            pred = pred.unsqueeze(0)
        # 多因子时保持 [batch, F] 不变
        
        # 🆕 返回融合特征用于相关性正则化
        # combined 是进入最终输出层之前的特征向量
        return pred, time_feat, combined
    
    # ==================== 🆕 Graph Context Preparation ====================
    
    def _prepare_graph_context(
        self,
        x_temporal: torch.Tensor,
        stock_idx: Optional[torch.Tensor],
        mode: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        准备图上下文（特征 + 邻接矩阵）
        
        根据不同的推理模式，构建适合 GAT 的输入：
        - 'batch': 直接切片（可能丢失邻居信息）
        - 'neighbor_sampling': 主动采样邻居，从缓存补全特征
        
        Args:
            x_temporal: [batch_size, hidden_dim] 当前 batch 的时序特征
            stock_idx: [batch_size] 股票全局索引
            mode: 推理模式（None 时使用默认配置）
            
        Returns:
            (augmented_x, augmented_adj, batch_size)
            - augmented_x: [batch_size + num_external, hidden_dim]
            - augmented_adj: [batch_size + num_external, batch_size + num_external]
            - batch_size: 原始 batch 大小（用于后续切片）
        """
        mode = mode or self.graph_inference_mode
        batch_size = x_temporal.size(0)
        
        # 如果没有邻接矩阵或没有股票索引，直接返回单位矩阵
        if self.adj_matrix is None or stock_idx is None:
            return x_temporal, torch.eye(batch_size, device=self.device), batch_size
        
        stock_idx = stock_idx.to(self.device)
        
        # ========== Mode: batch (原始模式) ==========
        if mode == GRAPH_MODE_BATCH:
            # 直接切片邻接矩阵
            current_adj = self.adj_matrix[stock_idx][:, stock_idx]
            return x_temporal, current_adj, batch_size
        
        # ========== Mode: neighbor_sampling ==========
        elif mode == GRAPH_MODE_NEIGHBOR_SAMPLING:
            # 确保已预计算邻居
            if self._neighbor_cache is None:
                self._precompute_neighbors()
            
            # 1. 收集所有需要的邻居
            batch_set = set(stock_idx.cpu().tolist())
            external_neighbors = set()
            
            for idx in stock_idx.cpu().tolist():
                if idx in self._neighbor_cache:
                    neighbors = self._neighbor_cache[idx].cpu().tolist()
                    for n in neighbors:
                        if n not in batch_set:
                            external_neighbors.add(n)
            
            # 2. 如果没有外部邻居，退化为 batch 模式
            if len(external_neighbors) == 0:
                current_adj = self.adj_matrix[stock_idx][:, stock_idx]
                return x_temporal, current_adj, batch_size
            
            # 3. 获取外部邻居的特征
            external_idx = torch.tensor(list(external_neighbors), dtype=torch.long, device=self.device)
            
            # 计算 fallback 特征（当前 batch 的均值）
            fallback_feat = x_temporal.mean(dim=0)
            external_features = self._fetch_cached_features(external_idx, fallback_feat)
            
            # 4. 拼接特征: [batch, hidden] + [external, hidden] -> [batch+external, hidden]
            augmented_x = torch.cat([x_temporal, external_features], dim=0)
            
            # 5. 构建扩展邻接矩阵
            # 索引顺序: [batch_nodes..., external_nodes...]
            all_idx = torch.cat([stock_idx, external_idx], dim=0)
            augmented_adj = self.adj_matrix[all_idx][:, all_idx]
            
            return augmented_x, augmented_adj, batch_size
        
        # ========== Mode: cross_sectional ==========
        elif mode == GRAPH_MODE_CROSS_SECTIONAL:
            # 截面模式：假设 batch 已经是同一天的所有股票
            # 如果 batch_size 接近总股票数，直接用全图；否则切片
            if batch_size > self.adj_matrix.size(0) * 0.8:
                # 接近全量，使用全图
                return x_temporal, self.adj_matrix[:batch_size, :batch_size], batch_size
            else:
                # 否则切片
                current_adj = self.adj_matrix[stock_idx][:, stock_idx]
                return x_temporal, current_adj, batch_size
        
        else:
            raise ValueError(f"不支持的图推理模式: {mode}")
    
    def _load_adj_matrix(
        self,
        path: Optional[str]
    ) -> Optional[torch.Tensor]:
        """
        加载邻接矩阵
        
        🆕 支持两种保存格式：
        1. 纯 Tensor：直接加载
        2. 包含元数据的字典：提取 'adj_matrix' 键
        
        Args:
            path: 邻接矩阵文件路径 (.pt, .pth, 或 .npy)
            
        Returns:
            邻接矩阵 Tensor 或 None
        """
        if path is None or not self.use_graph:
            return None
        
        path = Path(path)
        if not path.exists():
            self.logger.warning(f"邻接矩阵文件不存在: {path}")
            return None
        
        try:
            if path.suffix in ['.pt', '.pth']:
                data = torch.load(path, map_location=self.device)
                # 🆕 支持字典格式（build_industry_adj.py 保存的格式）
                if isinstance(data, dict):
                    adj = data['adj_matrix']
                    # 保存元数据供后续使用
                    self._adj_stock_list = data.get('stock_list', None)
                    self._adj_stock_to_idx = data.get('stock_to_idx', None)
                    n_industries = data.get('n_industries', 'N/A')
                    self.logger.info(f"  邻接矩阵元数据: {len(self._adj_stock_list) if self._adj_stock_list else 0} 只股票, {n_industries} 个行业")
                else:
                    # 兼容纯 Tensor 格式
                    adj = data
                # 确保在正确设备上
                adj = adj.to(self.device)
            elif path.suffix == '.npy':
                adj = torch.from_numpy(np.load(path)).float().to(self.device)
            else:
                raise ValueError(f"不支持的文件格式: {path.suffix}")
            
            self.logger.info(f"已加载邻接矩阵: {path}, shape={adj.shape}")
            return adj
        
        except Exception as e:
            self.logger.error(f"加载邻接矩阵失败: {e}")
            return None
    
    def _train_epoch(self, train_loader):
        """
        训练一个 epoch (🆕 使用统一的前向传播流程)
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            平均训练损失
            
        Note:
            🆕 现在使用 _forward_step 统一前向传播，确保训练和推理
            使用相同的计算图（包括邻居采样逻辑），避免分布偏移。
        """
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch_data in tqdm(train_loader, desc="训练", leave=False):
            # 解析 batch 数据
            batch_x, stock_idx, batch_funda = self._parse_batch_data(batch_data)
            batch_y = batch_data[1]
            
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            if batch_funda is not None:
                batch_funda = batch_funda.to(self.device)

            # 🆕 使用统一的前向传播（训练时也使用邻居采样，保持一致性）
            self.optimizer.zero_grad()
            pred, _, hidden_features = self._forward_step(
                batch_x, stock_idx, batch_funda,
                mode=self.graph_inference_mode,
                update_cache=True  # 训练时更新缓存
            )
            
            # 🆕 多因子输出支持：标签广播策略
            # 如果 pred 是 [batch, F]，则将 batch_y 广播为 [batch, F]
            # 让每个因子独立拟合同一个 alpha 标签
            is_multifactor = pred.dim() == 2 and pred.size(1) > 1
            if is_multifactor:
                batch_y_expanded = batch_y.unsqueeze(1).expand_as(pred)
                target_for_loss = batch_y_expanded
            else:
                target_for_loss = batch_y
            
            # 🆕 根据标志位决定是否传递 hidden_features
            # 🔧 多因子模式下对输出因子做正交化，单因子模式下对隐藏特征做正交化
            if self._use_corr_loss:
                corr_target = pred if is_multifactor else hidden_features
                loss = self.criterion(pred, target_for_loss, corr_target)
            else:
                loss = self.criterion(pred, target_for_loss)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0
    
    def _valid_epoch(self, valid_loader):
        """
        验证一个 epoch (🆕 使用统一的前向传播流程)
        
        Args:
            valid_loader: 验证数据加载器
            
        Returns:
            平均验证损失
        """
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch_data in valid_loader:
                # 解析 batch 数据
                batch_x, stock_idx, batch_funda = self._parse_batch_data(batch_data)
                batch_y = batch_data[1]
                
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                if batch_funda is not None:
                    batch_funda = batch_funda.to(self.device)
                
                # 🆕 使用统一的前向传播
                pred, _, hidden_features = self._forward_step(
                    batch_x, stock_idx, batch_funda,
                    mode=self.graph_inference_mode,
                    update_cache=True  # 验证时也更新缓存，为推理预热
                )
                
                # 🆕 多因子输出支持：标签广播策略
                is_multifactor = pred.dim() == 2 and pred.size(1) > 1
                if is_multifactor:
                    batch_y_expanded = batch_y.unsqueeze(1).expand_as(pred)
                    target_for_loss = batch_y_expanded
                else:
                    target_for_loss = batch_y
                
                # 🆕 根据标志位决定是否传递 hidden_features
                # 🔧 多因子模式下对输出因子做正交化，单因子模式下对隐藏特征做正交化
                if self._use_corr_loss:
                    corr_target = pred if is_multifactor else hidden_features
                    loss = self.criterion(pred, target_for_loss, corr_target)
                else:
                    loss = self.criterion(pred, target_for_loss)
                    
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0
    
    def fit(self, train_loader, valid_loader=None, save_path: Optional[str] = None):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器
            save_path: 模型保存路径
        """
        self.logger.info("开始训练 HybridGraph 模型...")
        
        # 🆕 创建学习率调度器
        self.scheduler = self._get_scheduler()
        
        best_valid_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            if valid_loader is not None:
                valid_loss = self._valid_epoch(valid_loader)
                self.valid_losses.append(valid_loss)
                
                # 🆕 获取当前学习率用于显示
                current_lr = self.optimizer.param_groups[0]['lr']
                
                self.logger.info(
                    f"Epoch {epoch+1}/{self.n_epochs} - "
                    f"Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}, "
                    f"LR: {current_lr:.2e}"
                )
                
                # 🆕 更新学习率调度器
                self._step_scheduler(valid_loss)
                
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    self.best_score = -valid_loss
                    self.best_epoch = epoch
                    patience_counter = 0
                    
                    if save_path:
                        self.save_model(save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stop:
                        self.logger.info(f"早停触发 at epoch {epoch+1}")
                        break
            else:
                self.logger.info(f"Epoch {epoch+1}/{self.n_epochs} - Train Loss: {train_loss:.6f}")
                # 🆕 即使没有验证集，也更新调度器（使用训练损失）
                self._step_scheduler(train_loss)
        
        self.fitted = True
        self.logger.info(f"训练完成! 最佳 epoch: {self.best_epoch+1}")
        
        # 🆕 显示学习率变化统计
        if self.lr_history:
            self.logger.info(f"  初始学习率: {self.lr_history[0]:.2e}")
            self.logger.info(f"  最终学习率: {self.lr_history[-1]:.2e}")
            if self.lr_history[0] != self.lr_history[-1]:
                self.logger.info(f"  学习率调整次数: {sum(1 for i in range(1, len(self.lr_history)) if self.lr_history[i] != self.lr_history[i-1])}")
    
    def predict(
        self, 
        test_loader, 
        return_numpy: bool = True,
        graph_inference_mode: Optional[str] = None,
        warm_up_cache: bool = False
    ):
        """
        预测（🆕 支持 Graph-Aware Inference）
        
        Args:
            test_loader: 测试数据加载器
            return_numpy: 是否返回 numpy 数组
            graph_inference_mode: 图推理模式，可覆盖默认配置
                - 'batch': 直接按 batch 切片邻接矩阵（默认）
                - 'cross_sectional': 假设每个 batch 是同一天的所有股票
                - 'neighbor_sampling': 主动采样邻居，从缓存补全特征
            warm_up_cache: 是否在预测前预热缓存（跑一遍 loader 但不产出结果）
            
        Returns:
            预测结果
            
        Note:
            使用 'neighbor_sampling' 模式时，建议在训练后保持缓存，
            或在预测前调用 warm_up_cache=True 进行预热。
        """
        if not self.fitted:
            raise ValueError("模型未训练,请先调用 fit()")
        
        # 确定推理模式
        mode = graph_inference_mode or self.graph_inference_mode
        
        # 🆕 邻居采样模式预热
        if warm_up_cache and mode == GRAPH_MODE_NEIGHBOR_SAMPLING:
            self.logger.info("🔥 预热节点状态缓存...")
            self._warm_up_cache(test_loader)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                # 解析 batch 数据
                batch_x, stock_idx, batch_funda = self._parse_batch_data(batch_data)
                batch_x = batch_x.to(self.device)
                if batch_funda is not None:
                    batch_funda = batch_funda.to(self.device)
                
                # 🆕 使用统一的前向传播
                pred, _, _ = self._forward_step(
                    batch_x, stock_idx, batch_funda,
                    mode=mode,
                    update_cache=True  # 预测时也更新缓存，支持流式推理
                )
                
                predictions.append(pred.cpu())
        
        # 处理空预测列表
        if len(predictions) == 0:
            return np.array([]) if return_numpy else torch.tensor([])
        
        # 🆕 多因子输出支持：保留 [N, F] 结构
        # 检查第一个 batch 的维度来确定输出格式
        first_pred = predictions[0]
        is_multi_factor = first_pred.dim() == 2 and first_pred.size(-1) > 1
        
        if is_multi_factor:
            # 多因子模式：拼接为 [N, F] 矩阵
            predictions = torch.cat(predictions, dim=0)  # [N, F]
            if return_numpy:
                return predictions.numpy()  # 返回 (N, F) 的 2D 数组
            return predictions
        else:
            # 单因子模式：保持原有行为，返回 1D 数组
            predictions = [p.flatten() if p.dim() > 1 else p for p in predictions]
            predictions = torch.cat(predictions, dim=0)
            if return_numpy:
                return predictions.numpy().flatten()
            return predictions
    
    def _parse_batch_data(self, batch_data) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        解析 batch 数据
        
        Args:
            batch_data: DataLoader 返回的 batch（元组或字典）
            
        Returns:
            (batch_x, stock_idx, batch_funda)
            
        Batch 格式支持:
            1. 元组格式（推荐使用 stock_idx_position 明确指定）:
               - (x, y): 无索引
               - (x, y, stock_idx): stock_idx_position=2
               - (x, y, date_idx, stock_idx): stock_idx_position=3
               - (x, y, stock_idx, funda): stock_idx_position=2, funda_position=3
               
            2. 字典格式（自动识别 key）:
               - batch['x'], batch['stock_idx'], batch['funda']
               
        Note:
            🆕 不再使用数值范围启发式猜测，而是通过明确的位置参数
            或字典 key 来识别股票索引，避免 date_idx 被误认为 stock_idx。
        """
        # ========== 字典格式 ==========
        if isinstance(batch_data, dict):
            batch_x = batch_data.get('x') or batch_data.get('features') or batch_data.get('input')
            if batch_x is None:
                raise ValueError("字典格式的 batch 必须包含 'x', 'features' 或 'input' 键")
            
            # 查找股票索引（支持多种命名）
            stock_idx = (
                batch_data.get('stock_idx') or 
                batch_data.get('instrument') or 
                batch_data.get('code_idx') or
                batch_data.get('symbol_idx')
            )
            
            # 查找基本面数据
            batch_funda = (
                batch_data.get('funda') or 
                batch_data.get('fundamental') or
                batch_data.get('static_features')
            )
            
            return batch_x, stock_idx, batch_funda
        
        # ========== 元组格式 ==========
        if not isinstance(batch_data, (tuple, list)):
            raise ValueError(f"不支持的 batch 类型: {type(batch_data)}")
        
        if len(batch_data) < 2:
            raise ValueError(f"batch 元素数量不足: {len(batch_data)}，至少需要 (x, y)")
        
        batch_x = batch_data[0]
        batch_funda = None
        stock_idx = None
        
        # 方式 1: 用户明确指定了位置
        if self.stock_idx_position is not None:
            if self.stock_idx_position < len(batch_data):
                candidate = batch_data[self.stock_idx_position]
                if torch.is_tensor(candidate) and candidate.dtype == torch.long:
                    stock_idx = self._validate_stock_idx_range(candidate)
                else:
                    self.logger.warning(
                        f"batch[{self.stock_idx_position}] 不是 long 类型张量，跳过"
                    )
        
        if self.funda_position is not None:
            if self.funda_position < len(batch_data):
                candidate = batch_data[self.funda_position]
                if torch.is_tensor(candidate) and candidate.dtype in [torch.float, torch.float32, torch.float64]:
                    batch_funda = candidate
        
        # 方式 2: 用户未指定，使用保守的默认规则
        # 只有当 stock_idx_position 未设置时才尝试自动推断
        if self.stock_idx_position is None and stock_idx is None:
            # 保守策略：只在 batch 长度 >= 4 时，假设第 4 个元素是 stock_idx
            # 这是因为常见格式为 (x, y, date_idx, stock_idx)
            if len(batch_data) >= 4:
                candidate = batch_data[3]
                if torch.is_tensor(candidate) and candidate.dtype == torch.long:
                    stock_idx = self._validate_stock_idx_range(candidate)
                    if stock_idx is not None:
                        self.logger.debug(
                            f"自动检测到 stock_idx 在 batch[3]，建议显式设置 stock_idx_position=3"
                        )
        
        return batch_x, stock_idx, batch_funda
    
    def _validate_stock_idx_range(self, idx_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """
        验证股票索引是否在邻接矩阵范围内
        
        这是一个辅助验证，用于在用户明确指定 stock_idx_position 后，
        进一步检查索引值是否合理。
        
        Args:
            idx_tensor: 待验证的索引张量
            
        Returns:
            有效的股票索引，或 None（如果超出范围）
            
        Note:
            如果用户通过 stock_idx_position 明确指定了位置，这里只做范围检查，
            不会因为值小于 num_stocks 就拒绝（那是旧的启发式逻辑的问题）。
        """
        if idx_tensor is None:
            return None
        
        # 如果没有邻接矩阵，无法验证范围，直接返回
        if self.adj_matrix is None:
            return idx_tensor
        
        # 检查索引是否超出邻接矩阵范围
        max_idx = idx_tensor.max().item()
        min_idx = idx_tensor.min().item()
        num_stocks = self.adj_matrix.size(0)
        
        if max_idx >= num_stocks:
            self.logger.warning(
                f"股票索引超出邻接矩阵范围: max_idx={max_idx} >= num_stocks={num_stocks}，"
                f"将忽略该索引。请检查 stock_idx_position 配置是否正确。"
            )
            return None
        
        if min_idx < 0:
            self.logger.warning(
                f"股票索引包含负值: min_idx={min_idx}，将忽略该索引。"
            )
            return None
        
        return idx_tensor
    
    def _warm_up_cache(self, data_loader):
        """
        预热节点状态缓存
        
        在正式预测前跑一遍数据，填充缓存。
        
        Args:
            data_loader: 数据加载器
        """
        self.model.eval()
        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc="预热缓存", leave=False):
                batch_x, stock_idx, _ = self._parse_batch_data(batch_data)
                if stock_idx is None:
                    continue
                
                batch_x = batch_x.to(self.device)
                time_feat = self.model.temporal(batch_x)
                self._update_node_cache(stock_idx, time_feat)
        
        self.logger.info(f"缓存预热完成，当前缓存大小: {len(self._node_state_cache)}")
    
    @classmethod
    def from_config(cls, config, d_feat: int = None):
        """
        从 CompositeModelConfig 创建模型实例
        
        Args:
            config: CompositeModelConfig 对象
            d_feat: 输入特征维度（可选，覆盖配置中的值）
            
        Returns:
            HybridGraphModel 实例
        """
        # 优先使用传入的 d_feat，否则使用配置中的
        input_dim = d_feat if d_feat is not None else config.d_feat
        
        # 提取时序模块配置
        rnn_type = 'lstm'
        rnn_hidden = 64
        rnn_layers = 2
        use_attention = True
        dropout = 0.3
        
        if config.temporal:
            rnn_type = config.temporal.rnn_type
            rnn_hidden = config.temporal.hidden_size
            rnn_layers = config.temporal.num_layers
            use_attention = config.temporal.use_attention
            dropout = config.temporal.dropout
            
        # 提取图模块配置
        use_graph = False
        gat_type = 'standard'
        gat_hidden = 32
        gat_heads = 4
        top_k_neighbors = 10
        
        if config.graph and config.graph.enabled:
            use_graph = True
            gat_type = config.graph.gat_type
            gat_hidden = config.graph.hidden_dim
            gat_heads = config.graph.heads
            top_k_neighbors = config.graph.top_k_neighbors
            
        # 提取融合模块配置
        mlp_hidden_sizes = [64]
        output_dim = 1  # 默认单因子输出
        if config.fusion:
            mlp_hidden_sizes = config.fusion.hidden_sizes
            output_dim = getattr(config.fusion, 'output_dim', 1)  # 🆕 提取多因子输出维度
            
        # 🆕 提取残差连接配置
        use_residual = getattr(config, 'use_residual', True)
        
        # 🆕 提取学习率调度器配置
        use_scheduler = getattr(config, 'use_scheduler', True)
        scheduler_type = getattr(config, 'scheduler_type', 'plateau')
        scheduler_patience = getattr(config, 'scheduler_patience', 5)
        scheduler_factor = getattr(config, 'scheduler_factor', 0.5)
        scheduler_min_lr = getattr(config, 'scheduler_min_lr', 1e-6)
        
        # 🆕 提取 Graph-Aware Inference 配置
        graph_inference_mode = getattr(config, 'graph_inference_mode', 'batch')
        max_neighbors = getattr(config, 'max_neighbors', 10)
        cache_size = getattr(config, 'cache_size', 5000)
        
        # 🆕 提取相关性正则化配置
        lambda_corr = getattr(config, 'lambda_corr', 0.0)
        
        return cls(
            d_feat=input_dim,
            rnn_hidden=rnn_hidden,
            rnn_layers=rnn_layers,
            rnn_type=rnn_type,
            use_attention=use_attention,
            use_graph=use_graph,
            gat_type=gat_type,
            gat_hidden=gat_hidden,
            gat_heads=gat_heads,
            top_k_neighbors=top_k_neighbors,
            mlp_hidden_sizes=mlp_hidden_sizes,
            funda_dim=config.funda_dim,
            dropout=dropout,
            adj_matrix_path=config.adj_matrix_path,
            output_dim=output_dim,  # 🆕 多因子输出维度
            use_residual=use_residual,  # 🆕 残差连接
            # 训练参数
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
            lr=config.learning_rate,  # 🆕 修正参数名
            early_stop=config.early_stop,
            optimizer=config.optimizer,
            device=config.device,
            # 🆕 学习率调度器参数
            use_scheduler=use_scheduler,
            scheduler_type=scheduler_type,
            scheduler_patience=scheduler_patience,
            scheduler_factor=scheduler_factor,
            scheduler_min_lr=scheduler_min_lr,
            # 🆕 Graph-Aware Inference 参数
            graph_inference_mode=graph_inference_mode,
            max_neighbors=max_neighbors,
            cache_size=cache_size,
            # 🆕 数据格式参数
            stock_idx_position=getattr(config, 'stock_idx_position', None),
            funda_position=getattr(config, 'funda_position', None),
            # 🆕 相关性正则化参数
            lambda_corr=lambda_corr
        )


if __name__ == '__main__':
    print("=" * 80)
    print("Hybrid Graph Models 测试")
    print("=" * 80)
    
    # 测试子模块
    print("\n1. 测试 TemporalBlock:")
    temporal = TemporalBlock(d_feat=20, hidden_size=64, num_layers=2)
    x = torch.randn(32, 40, 20)  # [batch=32, seq=40, feat=20]
    time_feat = temporal(x)
    print(f"  输入: {x.shape} -> 输出: {time_feat.shape}")
    
    print("\n2. 测试 GraphBlock:")
    graph = GraphBlock(in_dim=64, out_dim=32, heads=4)
    node_feat = torch.randn(100, 64)  # [stocks=100, feat=64]
    adj = torch.randint(0, 2, (100, 100)).float()  # 随机邻接矩阵
    graph_feat = graph(node_feat, adj)
    print(f"  输入: {node_feat.shape} + adj {adj.shape} -> 输出: {graph_feat.shape}")
    
    print("\n3. 测试 FusionBlock:")
    fusion = FusionBlock(input_dim=96, hidden_sizes=[64], output_dim=1)
    combined = torch.randn(32, 96)
    pred = fusion(combined)
    print(f"  输入: {combined.shape} -> 输出: {pred.shape}")
    
    print("\n4. 测试 HybridNet:")
    model = HybridNet(
        d_feat=20,
        rnn_hidden=64,
        rnn_layers=2,
        use_graph=True,
        gat_hidden=32,
        gat_heads=4,
        mlp_hidden_sizes=[64]
    )
    x = torch.randn(32, 40, 20)
    adj = torch.randint(0, 2, (32, 32)).float()
    pred = model(x, adj)
    print(f"  输入: {x.shape} + adj {adj.shape} -> 输出: {pred.shape}")
    
    print("\n5. 测试 HybridGraphModel 创建:")
    hybrid_model = HybridGraphModel(
        d_feat=20,
        rnn_hidden=64,
        rnn_layers=2,
        use_graph=True,
        gat_hidden=32,
        gat_heads=4,
        n_epochs=10
    )
    print(f"✅ HybridGraphModel 创建成功")
    
    print("\n" + "=" * 80)
    print("✅ Hybrid Graph Models 测试完成")
    print("=" * 80)
