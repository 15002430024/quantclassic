"""
模块化配置使用示例

展示如何使用新的模块化配置系统手动搭建不同的模型变体。
"""

import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.modular_config import (
    TemporalModuleConfig,
    GraphModuleConfig,
    FusionModuleConfig,
    CompositeModelConfig,
    ModelConfigBuilder,
    ConfigTemplates
)


def example_1_basic_usage():
    """
    示例 1: 基础使用 - 手动组合模块
    """
    print("=" * 80)
    print("示例 1: 基础使用 - 手动组合模块")
    print("=" * 80)
    
    # 步骤 1: 独立配置每个模块
    temporal_config = TemporalModuleConfig(
        rnn_type='lstm',
        hidden_size=64,
        num_layers=2,
        use_attention=True,
        attention_type='self',
        dropout=0.3
    )
    
    graph_config = GraphModuleConfig(
        gat_type='correlation',
        hidden_dim=32,
        heads=4,
        top_k_neighbors=10,
        dropout=0.3
    )
    
    fusion_config = FusionModuleConfig(
        hidden_sizes=[64],
        activation='relu',
        dropout=0.3,
        output_dim=1
    )
    
    # 步骤 2: 组合成完整模型配置
    model_config = CompositeModelConfig(
        temporal=temporal_config,
        graph=graph_config,
        fusion=fusion_config,
        d_feat=20,
        n_epochs=100,
        batch_size=256,
        learning_rate=0.001
    )
    
    # 步骤 3: 验证并查看摘要
    model_config.validate()
    print(model_config.summary())
    
    return model_config


def example_2_builder_pattern():
    """
    示例 2: 使用构建器模式 (推荐)
    """
    print("\n" + "=" * 80)
    print("示例 2: 使用构建器模式 (推荐)")
    print("=" * 80)
    
    # 一次性构建完整配置
    config = ModelConfigBuilder() \
        .set_input(d_feat=20, funda_dim=None) \
        .add_temporal(
            rnn_type='gru',
            hidden_size=128,
            num_layers=2,
            use_attention=True,
            attention_type='multi_head',
            attention_heads=8
        ) \
        .add_graph(
            gat_type='standard',
            hidden_dim=64,
            heads=4,
            adj_matrix_path='./adj_matrix.pt'
        ) \
        .add_fusion(
            hidden_sizes=[128, 64],
            activation='gelu',
            use_batch_norm=True
        ) \
        .set_training(
            device='cuda',
            n_epochs=150,
            batch_size=512,
            learning_rate=0.0005,
            optimizer='adamw'
        ) \
        .build()
    
    print(config.summary())
    
    return config


def example_3_pure_temporal():
    """
    示例 3: 纯时序模型 (不使用图)
    """
    print("\n" + "=" * 80)
    print("示例 3: 纯时序模型 (不使用图)")
    print("=" * 80)
    
    config = ModelConfigBuilder() \
        .set_input(d_feat=20) \
        .add_temporal(
            rnn_type='lstm',
            hidden_size=64,
            num_layers=2,
            bidirectional=True,  # 使用双向LSTM
            use_attention=True
        ) \
        .add_fusion(hidden_sizes=[64]) \
        .build()
    
    print(config.summary())
    print(f"\n注意: graph 模块为 None，模型将跳过图神经网络部分")
    
    return config


def example_4_graph_variants():
    """
    示例 4: 不同的图神经网络变体
    """
    print("\n" + "=" * 80)
    print("示例 4: 不同的图神经网络变体")
    print("=" * 80)
    
    # 变体 A: 基于行业关系的标准GAT
    print("\n【变体 A: 行业关系GAT】")
    config_a = ModelConfigBuilder() \
        .add_temporal(rnn_type='lstm', hidden_size=64) \
        .add_graph(
            gat_type='standard',
            hidden_dim=32,
            heads=4
        ) \
        .add_fusion(hidden_sizes=[64]) \
        .build(d_feat=20)
    
    print(f"  GAT类型: {config_a.graph.gat_type}")
    print(f"  输出维度: {config_a.graph.output_dim}")
    
    # 变体 B: 基于相关性的动态GAT
    print("\n【变体 B: 相关性GAT】")
    config_b = ModelConfigBuilder() \
        .add_temporal(rnn_type='gru', hidden_size=64) \
        .add_graph(
            gat_type='correlation',
            hidden_dim=32,
            heads=4,
            top_k_neighbors=15  # 每只股票连接15个最相关的邻居
        ) \
        .add_fusion(hidden_sizes=[64]) \
        .build(d_feat=20)
    
    print(f"  GAT类型: {config_b.graph.gat_type}")
    print(f"  K近邻数: {config_b.graph.top_k_neighbors}")
    
    # 变体 C: 完全动态学习图结构
    print("\n【变体 C: 动态图结构】")
    config_c = ModelConfigBuilder() \
        .add_temporal(rnn_type='lstm', hidden_size=64) \
        .add_graph(
            gat_type='dynamic',
            hidden_dim=64,
            heads=8,
            use_edge_features=True  # 使用边特征
        ) \
        .add_fusion(hidden_sizes=[128, 64]) \
        .build(d_feat=20)
    
    print(f"  GAT类型: {config_c.graph.gat_type}")
    print(f"  边特征: {config_c.graph.use_edge_features}")
    
    return config_a, config_b, config_c


def example_5_attention_variants():
    """
    示例 5: 不同的注意力机制变体
    """
    print("\n" + "=" * 80)
    print("示例 5: 不同的注意力机制变体")
    print("=" * 80)
    
    # 变体 A: Self-Attention
    print("\n【变体 A: Self-Attention】")
    config_a = ModelConfigBuilder() \
        .add_temporal(
            rnn_type='lstm',
            hidden_size=64,
            use_attention=True,
            attention_type='self'
        ) \
        .add_fusion(hidden_sizes=[64]) \
        .build(d_feat=20)
    
    print(f"  注意力类型: {config_a.temporal.attention_type}")
    
    # 变体 B: Multi-Head Attention
    print("\n【变体 B: Multi-Head Attention】")
    config_b = ModelConfigBuilder() \
        .add_temporal(
            rnn_type='gru',
            hidden_size=64,
            use_attention=True,
            attention_type='multi_head',
            attention_heads=8
        ) \
        .add_fusion(hidden_sizes=[64]) \
        .build(d_feat=20)
    
    print(f"  注意力类型: {config_b.temporal.attention_type}")
    print(f"  注意力头数: {config_b.temporal.attention_heads}")
    
    # 变体 C: Additive Attention
    print("\n【变体 C: Additive Attention】")
    config_c = ModelConfigBuilder() \
        .add_temporal(
            rnn_type='lstm',
            hidden_size=64,
            use_attention=True,
            attention_type='additive'
        ) \
        .add_fusion(hidden_sizes=[64]) \
        .build(d_feat=20)
    
    print(f"  注意力类型: {config_c.temporal.attention_type}")
    
    return config_a, config_b, config_c


def example_6_fusion_variants():
    """
    示例 6: 不同的融合策略
    """
    print("\n" + "=" * 80)
    print("示例 6: 不同的融合策略")
    print("=" * 80)
    
    # 变体 A: 简单拼接 (默认)
    print("\n【变体 A: 简单拼接】")
    config_a = ModelConfigBuilder() \
        .add_temporal(rnn_type='lstm', hidden_size=64) \
        .add_graph(gat_type='standard', hidden_dim=32) \
        .add_fusion(hidden_sizes=[64]) \
        .build(d_feat=20)
    
    print(f"  融合策略: {config_a.feature_fusion_strategy}")
    print(f"  融合输入维度: {config_a.get_fusion_input_dim()}")
    
    # 变体 B: 深层MLP + BatchNorm
    print("\n【变体 B: 深层MLP + BatchNorm】")
    config_b = ModelConfigBuilder() \
        .add_temporal(rnn_type='gru', hidden_size=64) \
        .add_graph(gat_type='correlation', hidden_dim=32) \
        .add_fusion(
            hidden_sizes=[128, 64, 32],
            use_batch_norm=True,
            activation='gelu'
        ) \
        .build(d_feat=20)
    
    print(f"  隐藏层: {config_b.fusion.hidden_sizes}")
    print(f"  BatchNorm: {config_b.fusion.use_batch_norm}")
    
    # 变体 C: 残差连接
    print("\n【变体 C: 残差连接】")
    config_c = ModelConfigBuilder() \
        .add_temporal(rnn_type='lstm', hidden_size=64) \
        .add_graph(gat_type='standard', hidden_dim=32) \
        .add_fusion(
            hidden_sizes=[96, 96],  # 与输入维度相同以支持残差
            use_residual=True
        ) \
        .build(d_feat=20)
    
    print(f"  残差连接: {config_c.fusion.use_residual}")
    
    return config_a, config_b, config_c


def example_7_predefined_templates():
    """
    示例 7: 使用预定义模板
    """
    print("\n" + "=" * 80)
    print("示例 7: 使用预定义模板")
    print("=" * 80)
    
    # 模板 A: 小型纯时序模型
    print("\n【模板 A: 小型纯时序模型】")
    config_small = ConfigTemplates.pure_temporal(d_feat=20, model_size='small')
    print(f"  RNN隐藏层: {config_small.temporal.hidden_size}")
    print(f"  RNN层数: {config_small.temporal.num_layers}")
    
    # 模板 B: 默认混合模型
    print("\n【模板 B: 默认混合模型】")
    config_default = ConfigTemplates.temporal_with_graph(
        d_feat=20,
        gat_type='standard',
        model_size='default'
    )
    print(f"  RNN隐藏层: {config_default.temporal.hidden_size}")
    print(f"  GAT隐藏层: {config_default.graph.hidden_dim}")
    
    # 模板 C: 大型高级模型
    print("\n【模板 C: 大型高级模型】")
    config_large = ConfigTemplates.temporal_with_graph(
        d_feat=20,
        gat_type='correlation',
        model_size='large'
    )
    print(f"  RNN隐藏层: {config_large.temporal.hidden_size}")
    print(f"  GAT隐藏层: {config_large.graph.hidden_dim}")
    
    # 模板 D: 多头注意力+相关性图
    print("\n【模板 D: 多头注意力+相关性图】")
    config_advanced = ConfigTemplates.attention_graph_fusion(
        d_feat=20,
        attention_type='multi_head',
        gat_type='correlation'
    )
    print(f"  注意力类型: {config_advanced.temporal.attention_type}")
    print(f"  注意力头数: {config_advanced.temporal.attention_heads}")
    print(f"  GAT类型: {config_advanced.graph.gat_type}")
    print(f"  融合层数: {len(config_advanced.fusion.hidden_sizes)}")
    
    return config_small, config_default, config_large, config_advanced


def example_8_save_and_load():
    """
    示例 8: 保存和加载配置
    """
    print("\n" + "=" * 80)
    print("示例 8: 保存和加载配置")
    print("=" * 80)
    
    # 创建配置
    config = ModelConfigBuilder() \
        .add_temporal(rnn_type='lstm', hidden_size=64) \
        .add_graph(gat_type='correlation', hidden_dim=32) \
        .add_fusion(hidden_sizes=[64]) \
        .build(d_feat=20)
    
    # 保存为 YAML
    yaml_path = '/tmp/my_model_config.yaml'
    config.to_yaml(yaml_path)
    print(f"✅ 配置已保存到: {yaml_path}")
    
    # 加载配置
    loaded_config = CompositeModelConfig.from_yaml(yaml_path)
    print(f"✅ 配置已加载")
    print(f"  时序模块: {loaded_config.temporal.rnn_type}")
    print(f"  图模块: {loaded_config.graph.gat_type}")
    
    # 保存为 JSON
    json_path = '/tmp/my_model_config.json'
    config.to_json(json_path)
    print(f"✅ 配置已保存到: {json_path}")
    
    return config


def example_9_customize_module():
    """
    示例 9: 自定义新的模块配置 (扩展性示例)
    
    展示如何基于现有模块配置创建自己的变体。
    """
    print("\n" + "=" * 80)
    print("示例 9: 自定义新的模块配置 (扩展性示例)")
    print("=" * 80)
    
    # 自定义时序模块: Transformer 风格
    class TransformerTemporalConfig(TemporalModuleConfig):
        """自定义: Transformer风格的时序模块"""
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # 覆盖默认设置
            self.rnn_type = 'gru'  # 基础仍用RNN
            self.use_attention = True
            self.attention_type = 'multi_head'
            self.attention_heads = 8
            self.name = 'TransformerTemporal'
    
    # 使用自定义模块
    custom_temporal = TransformerTemporalConfig(
        hidden_size=128,
        num_layers=3,
        dropout=0.2
    )
    
    config = CompositeModelConfig(
        temporal=custom_temporal,
        graph=None,
        fusion=FusionModuleConfig(hidden_sizes=[128, 64]),
        d_feat=20
    )
    
    print(f"✅ 自定义配置创建成功")
    print(f"  模块名称: {config.temporal.name}")
    print(f"  注意力类型: {config.temporal.attention_type}")
    print(f"  注意力头数: {config.temporal.attention_heads}")
    
    return config


def example_10_comparison():
    """
    示例 10: 新旧配置对比
    
    展示从整体配置迁移到模块化配置的对比。
    """
    print("\n" + "=" * 80)
    print("示例 10: 新旧配置对比")
    print("=" * 80)
    
    # 旧方式: HybridGraphConfig (整体配置)
    print("\n【旧方式: HybridGraphConfig】")
    print("from model.model_config import HybridGraphConfig")
    print("config = HybridGraphConfig(")
    print("    d_feat=20,")
    print("    rnn_hidden=64,")
    print("    rnn_layers=2,")
    print("    rnn_type='lstm',")
    print("    gat_hidden=32,")
    print("    gat_heads=4,")
    print("    mlp_hidden_sizes=[64]")
    print(")")
    print("\n❌ 缺点:")
    print("  - 所有参数混在一起，难以理解")
    print("  - 扩展性差，添加新模块需要修改整个类")
    print("  - 不支持模块复用")
    
    # 新方式: CompositeModelConfig (模块化配置)
    print("\n【新方式: CompositeModelConfig】")
    print("from model.modular_config import ModelConfigBuilder")
    print("config = ModelConfigBuilder() \\")
    print("    .add_temporal(rnn_type='lstm', hidden_size=64, num_layers=2) \\")
    print("    .add_graph(gat_type='standard', hidden_dim=32, heads=4) \\")
    print("    .add_fusion(hidden_sizes=[64]) \\")
    print("    .build(d_feat=20)")
    print("\n✅ 优点:")
    print("  - 模块独立，职责清晰")
    print("  - 高扩展性，轻松添加新模块或变体")
    print("  - 支持模块复用和组合")
    print("  - 流式API，可读性强")
    
    # 创建实际对比
    new_config = ModelConfigBuilder() \
        .add_temporal(rnn_type='lstm', hidden_size=64, num_layers=2) \
        .add_graph(gat_type='standard', hidden_dim=32, heads=4) \
        .add_fusion(hidden_sizes=[64]) \
        .build(d_feat=20)
    
    print(f"\n✅ 新配置创建成功")
    print(f"  融合输入维度: {new_config.get_fusion_input_dim()}")
    
    return new_config


if __name__ == '__main__':
    """运行所有示例"""
    
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "模块化配置系统使用指南" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # 运行所有示例
    try:
        example_1_basic_usage()
        example_2_builder_pattern()
        example_3_pure_temporal()
        example_4_graph_variants()
        example_5_attention_variants()
        example_6_fusion_variants()
        example_7_predefined_templates()
        example_8_save_and_load()
        example_9_customize_module()
        example_10_comparison()
        
        print("\n" + "=" * 80)
        print("✅ 所有示例运行成功!")
        print("=" * 80)
        
        print("\n💡 快速开始:")
        print("  1. 使用构建器快速创建配置 (推荐)")
        print("  2. 使用预定义模板")
        print("  3. 手动组合模块 (最灵活)")
        
        print("\n📚 更多文档:")
        print("  - modular_config.py: 模块化配置源码和详细文档")
        print("  - README_HYBRID_GRAPH.md: 混合模型使用指南")
        
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
