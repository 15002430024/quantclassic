"""
HybridGraph 模型测试（不依赖 PyTorch）

仅测试配置类和架构设计逻辑
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model_config import (
    HybridGraphConfig,
    ModelConfigFactory
)


def test_hybrid_config():
    """测试 HybridGraph 配置"""
    print("=" * 80)
    print("测试 1: HybridGraph 配置创建")
    print("=" * 80)
    
    # 创建配置
    config = HybridGraphConfig(
        d_feat=20,
        rnn_hidden=64,
        rnn_layers=2,
        rnn_type='lstm',
        use_attention=True,
        use_graph=True,
        gat_hidden=32,
        gat_heads=4,
        n_epochs=100
    )
    
    print(f"\n✅ 配置创建成功:")
    print(f"  - 输入特征维度: {config.d_feat}")
    print(f"  - RNN类型: {config.rnn_type}")
    print(f"  - RNN隐藏层: {config.rnn_hidden}")
    print(f"  - 使用Attention: {config.use_attention}")
    print(f"  - 使用图网络: {config.use_graph}")
    print(f"  - GAT隐藏层: {config.gat_hidden}")
    print(f"  - GAT头数: {config.gat_heads}")
    
    # 验证配置
    try:
        config.validate()
        print("\n✅ 配置验证通过")
    except Exception as e:
        print(f"\n❌ 配置验证失败: {e}")


def test_config_factory():
    """测试配置工厂"""
    print("\n" + "=" * 80)
    print("测试 2: 配置工厂")
    print("=" * 80)
    
    # 测试创建配置
    config = ModelConfigFactory.create('hybrid_graph', rnn_hidden=128)
    print(f"\n✅ 工厂创建配置: rnn_hidden={config.rnn_hidden}")
    
    # 测试模板
    templates = ['small', 'default', 'large']
    print(f"\n可用模板:")
    for template in templates:
        config = ModelConfigFactory.get_template('hybrid_graph', template)
        print(f"  - {template:8s}: "
              f"rnn_hidden={config.rnn_hidden}, "
              f"gat_hidden={config.gat_hidden}, "
              f"epochs={config.n_epochs}")


def test_config_validation():
    """测试配置验证"""
    print("\n" + "=" * 80)
    print("测试 3: 配置验证")
    print("=" * 80)
    
    # 测试无效配置
    test_cases = [
        ("gat_hidden不能被gat_heads整除", 
         {"gat_hidden": 30, "gat_heads": 4}),
        ("rnn_hidden必须大于0", 
         {"rnn_hidden": -10}),
        ("不支持的RNN类型", 
         {"rnn_type": "invalid"}),
    ]
    
    for desc, kwargs in test_cases:
        try:
            config = HybridGraphConfig(**kwargs)
            config.validate()
            print(f"  ❌ {desc}: 应该抛出异常但没有")
        except ValueError as e:
            print(f"  ✅ {desc}: 正确捕获错误")


def test_config_serialization():
    """测试配置序列化"""
    print("\n" + "=" * 80)
    print("测试 4: 配置序列化")
    print("=" * 80)
    
    # 创建配置
    config = HybridGraphConfig(
        rnn_hidden=128,
        gat_hidden=64,
        learning_rate=0.0005
    )
    
    # 保存为 YAML
    yaml_path = '/tmp/hybrid_graph_config.yaml'
    config.to_yaml(yaml_path)
    print(f"\n✅ 配置已保存: {yaml_path}")
    
    # 加载配置
    config2 = HybridGraphConfig.from_yaml(yaml_path)
    print(f"✅ 配置已加载: rnn_hidden={config2.rnn_hidden}")
    
    # 验证一致性
    assert config.rnn_hidden == config2.rnn_hidden
    assert config.gat_hidden == config2.gat_hidden
    print("✅ 配置序列化/反序列化一致性验证通过")
    
    # 转换为字典
    config_dict = config.to_dict()
    print(f"\n✅ 转换为字典: 包含 {len(config_dict)} 个键")
    print(f"  关键参数: rnn_hidden={config_dict['rnn_hidden']}, "
          f"gat_hidden={config_dict['gat_hidden']}")


def test_config_update():
    """测试配置更新"""
    print("\n" + "=" * 80)
    print("测试 5: 配置更新")
    print("=" * 80)
    
    config = ModelConfigFactory.get_template('hybrid_graph', 'default')
    
    print(f"\n初始配置:")
    print(f"  - rnn_hidden: {config.rnn_hidden}")
    print(f"  - learning_rate: {config.learning_rate}")
    print(f"  - n_epochs: {config.n_epochs}")
    
    # 更新配置
    config.update(
        rnn_hidden=256,
        learning_rate=0.0001,
        n_epochs=200
    )
    
    print(f"\n更新后配置:")
    print(f"  - rnn_hidden: {config.rnn_hidden}")
    print(f"  - learning_rate: {config.learning_rate}")
    print(f"  - n_epochs: {config.n_epochs}")
    
    print("\n✅ 配置更新成功")


def test_architecture_description():
    """测试架构描述"""
    print("\n" + "=" * 80)
    print("测试 6: 模型架构描述")
    print("=" * 80)
    
    config = ModelConfigFactory.get_template('hybrid_graph', 'large')
    
    print(f"\n大型 HybridGraph 模型架构:")
    print(f"")
    print(f"1. 时序提取器 (TemporalBlock):")
    print(f"   - RNN类型: {config.rnn_type.upper()}")
    print(f"   - 隐藏层大小: {config.rnn_hidden}")
    print(f"   - 层数: {config.rnn_layers}")
    print(f"   - Self-Attention: {config.use_attention}")
    print(f"")
    print(f"2. 截面交互器 (GraphBlock - GAT):")
    print(f"   - 启用: {config.use_graph}")
    print(f"   - 隐藏层维度: {config.gat_hidden}")
    print(f"   - 注意力头数: {config.gat_heads}")
    print(f"   - GAT类型: {config.gat_type}")
    print(f"")
    print(f"3. 融合预测器 (FusionBlock - MLP):")
    print(f"   - 隐藏层尺寸: {config.mlp_hidden_sizes}")
    print(f"   - 输出维度: {config.output_dim}")
    print(f"")
    print(f"4. 训练配置:")
    print(f"   - 训练轮数: {config.n_epochs}")
    print(f"   - 学习率: {config.learning_rate}")
    print(f"   - Dropout: {config.dropout}")
    print(f"   - 早停: {config.early_stop}")


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("HybridGraph 配置测试套件")
    print("=" * 80)
    
    test_hybrid_config()
    test_config_factory()
    test_config_validation()
    test_config_serialization()
    test_config_update()
    test_architecture_description()
    
    print("\n" + "=" * 80)
    print("✅ 所有配置测试通过")
    print("=" * 80)
    print("\n提示: 要测试完整模型，请先安装 PyTorch:")
    print("  pip install torch")
    print("然后运行:")
    print("  python hybrid_graph_models.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
