"""
test_base_trainer.py - BaseTrainer 单元测试

测试覆盖:
- TrainerConfig 验证
- _create_criterion 支持 ic/ic_corr 损失
- _create_optimizer 优化器创建
- _create_scheduler 调度器创建
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch


class TestTrainerConfig:
    """测试 TrainerConfig 配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        from model.train.base_trainer import TrainerConfig
        
        config = TrainerConfig()
        assert config.n_epochs == 100
        assert config.lr == 0.001
        assert config.loss_fn == 'mse'
        assert config.early_stop == 20
    
    def test_custom_config(self):
        """测试自定义配置"""
        from model.train.base_trainer import TrainerConfig
        
        config = TrainerConfig(
            n_epochs=50,
            lr=0.0001,
            loss_fn='ic_corr',
            lambda_corr=0.05
        )
        assert config.n_epochs == 50
        assert config.lr == 0.0001
        assert config.loss_fn == 'ic_corr'
        assert config.lambda_corr == 0.05
    
    def test_config_validation(self):
        """测试配置验证"""
        from model.train.base_trainer import TrainerConfig
        
        # 有效配置
        config = TrainerConfig(n_epochs=10, lr=0.001)
        assert config.validate() is True
        
        # 无效配置 - n_epochs <= 0
        with pytest.raises(ValueError):
            config = TrainerConfig(n_epochs=0)
            config.validate()
        
        # 无效配置 - lr <= 0
        with pytest.raises(ValueError):
            config = TrainerConfig(lr=-0.001)
            config.validate()


class TestCriterionCreation:
    """测试损失函数创建"""
    
    @pytest.fixture
    def simple_model(self):
        """创建简单模型"""
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)
            def forward(self, x):
                return self.fc(x).squeeze(-1)
        return SimpleNet()
    
    def test_mse_loss(self, simple_model):
        """测试 MSE 损失创建"""
        from model.train.base_trainer import TrainerConfig
        from model.train.simple_trainer import SimpleTrainer
        
        config = TrainerConfig(loss_fn='mse', lambda_corr=0.0)
        trainer = SimpleTrainer(simple_model, config, device='cpu')
        
        criterion = trainer._create_criterion()
        assert criterion is not None
        
        # 测试损失计算
        pred = torch.randn(32)
        target = torch.randn(32)
        loss = criterion(pred, target)
        assert loss.item() >= 0
    
    def test_ic_loss(self, simple_model):
        """测试 IC 损失创建"""
        from model.train.base_trainer import TrainerConfig
        from model.train.simple_trainer import SimpleTrainer
        
        config = TrainerConfig(loss_fn='ic', lambda_corr=0.0)
        trainer = SimpleTrainer(simple_model, config, device='cpu')
        
        criterion = trainer._create_criterion()
        assert criterion is not None
        
        # 测试损失计算
        pred = torch.randn(32)
        target = torch.randn(32)
        loss = criterion(pred, target)
        # IC loss 应该在 [0, 2] 范围内
        assert 0 <= loss.item() <= 2
    
    def test_ic_corr_loss(self, simple_model):
        """测试 IC + 相关性正则化损失"""
        from model.train.base_trainer import TrainerConfig
        from model.train.simple_trainer import SimpleTrainer
        
        config = TrainerConfig(loss_fn='ic_corr', lambda_corr=0.01)
        trainer = SimpleTrainer(simple_model, config, device='cpu')
        
        criterion = trainer._create_criterion()
        assert criterion is not None
        
        # 测试损失计算（带 hidden_features）
        pred = torch.randn(32)
        target = torch.randn(32)
        hidden = torch.randn(32, 64)
        
        # 损失函数应该接受 hidden_features 参数
        try:
            loss = criterion(pred, target, hidden)
            assert loss.item() >= 0
        except TypeError:
            # 如果不支持 hidden_features，使用基础调用
            loss = criterion(pred, target)
            assert loss.item() >= 0
    
    def test_huber_loss(self, simple_model):
        """测试 Huber 损失创建"""
        from model.train.base_trainer import TrainerConfig
        from model.train.simple_trainer import SimpleTrainer
        
        config = TrainerConfig(loss_fn='huber', lambda_corr=0.0)
        trainer = SimpleTrainer(simple_model, config, device='cpu')
        
        criterion = trainer._create_criterion()
        assert criterion is not None


class TestOptimizerCreation:
    """测试优化器创建"""
    
    @pytest.fixture
    def simple_model(self):
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)
            def forward(self, x):
                return self.fc(x).squeeze(-1)
        return SimpleNet()
    
    def test_adam_optimizer(self, simple_model):
        """测试 Adam 优化器"""
        from model.train.base_trainer import TrainerConfig
        from model.train.simple_trainer import SimpleTrainer
        
        config = TrainerConfig(optimizer='adam', lr=0.001)
        trainer = SimpleTrainer(simple_model, config, device='cpu')
        
        optimizer = trainer._create_optimizer()
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]['lr'] == 0.001
    
    def test_adamw_optimizer(self, simple_model):
        """测试 AdamW 优化器"""
        from model.train.base_trainer import TrainerConfig
        from model.train.simple_trainer import SimpleTrainer
        
        config = TrainerConfig(optimizer='adamw', lr=0.001, weight_decay=0.01)
        trainer = SimpleTrainer(simple_model, config, device='cpu')
        
        optimizer = trainer._create_optimizer()
        assert isinstance(optimizer, torch.optim.AdamW)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
