"""
test_config_inheritance.py - 配置继承体系单元测试

覆盖内容：
1. BaseConfig 继承链验证
2. TrainerConfigDC 与 TrainerConfig 互转
3. 序列化/反序列化（YAML, JSON, Dict）
4. validate 方法测试
5. isinstance 检测
"""

import pytest
import tempfile
import os
import sys
import warnings

# 确保导入路径正确
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBaseConfigImport:
    """测试 BaseConfig 导入和基本功能"""
    
    def test_import_base_config_from_config(self):
        """验证从 config 模块导入 BaseConfig"""
        from config import BaseConfig, TaskConfig
        assert BaseConfig is not None
        assert TaskConfig is not None
    
    def test_import_base_config_from_base_config(self):
        """验证从 config.base_config 导入 BaseConfig"""
        from config.base_config import BaseConfig
        assert BaseConfig is not None


class TestConfigInheritance:
    """测试配置类继承链"""
    
    def test_data_config_inherits_base_config(self):
        """验证 DataConfig 继承自 BaseConfig"""
        from config.base_config import BaseConfig
        from data_set.config import DataConfig
        
        # 检查继承关系
        assert issubclass(DataConfig, BaseConfig)
        
        # 检查 isinstance
        config = DataConfig()
        assert isinstance(config, BaseConfig)
    
    def test_preprocess_config_inherits_base_config(self):
        """验证 PreprocessConfig 继承自 BaseConfig"""
        from config.base_config import BaseConfig
        from data_processor.preprocess_config import PreprocessConfig
        
        # 检查继承关系
        assert issubclass(PreprocessConfig, BaseConfig)
        
        # 检查 isinstance
        config = PreprocessConfig()
        assert isinstance(config, BaseConfig)
    
    def test_trainer_config_inherits_base_config(self):
        """验证 TrainerConfig 继承自 BaseConfig"""
        from config.base_config import BaseConfig
        from model.train.base_trainer import TrainerConfig
        
        # 检查继承关系
        assert issubclass(TrainerConfig, BaseConfig)
        
        # 检查 isinstance
        config = TrainerConfig()
        assert isinstance(config, BaseConfig)
    
    def test_trainer_config_dc_inherits_base_config(self):
        """验证 TrainerConfigDC 继承自 BaseConfig"""
        from config.base_config import BaseConfig, TrainerConfigDC
        
        # 检查继承关系
        assert issubclass(TrainerConfigDC, BaseConfig)
        
        # 检查 isinstance
        config = TrainerConfigDC()
        assert isinstance(config, BaseConfig)
    
    def test_task_config_inherits_base_config(self):
        """验证 TaskConfig 继承自 BaseConfig"""
        from config.base_config import BaseConfig, TaskConfig
        
        # 检查继承关系
        assert issubclass(TaskConfig, BaseConfig)
        
        # 检查 isinstance
        config = TaskConfig(
            model_class="DummyModel",
            dataset_class="DummyDataset"
        )
        assert isinstance(config, BaseConfig)


class TestTrainerConfigConversion:
    """测试 TrainerConfigDC 与 TrainerConfig 互转"""
    
    def test_trainer_config_dc_to_trainer_config(self):
        """测试 TrainerConfigDC.to_trainer_config() 方法"""
        from config.base_config import TrainerConfigDC
        from model.train.base_trainer import TrainerConfig
        
        # 创建 TrainerConfigDC
        dc_config = TrainerConfigDC(
            n_epochs=50,
            lr=0.01,
            weight_decay=1e-4,
            early_stop=10,
            optimizer='adamw',
            loss_fn='mse',
            use_scheduler=True,
            scheduler_type='cosine',
            verbose=False,
            log_interval=100
        )
        
        # 转换为 TrainerConfig
        trainer_config = dc_config.to_trainer_config()
        
        # 验证类型
        assert isinstance(trainer_config, TrainerConfig)
        
        # 验证字段一致
        assert trainer_config.n_epochs == 50
        assert trainer_config.lr == 0.01
        assert trainer_config.weight_decay == 1e-4
        assert trainer_config.early_stop == 10
        assert trainer_config.optimizer == 'adamw'
        assert trainer_config.loss_fn == 'mse'
        assert trainer_config.use_scheduler is True
        assert trainer_config.scheduler_type == 'cosine'
        assert trainer_config.verbose is False
        assert trainer_config.log_interval == 100
    
    def test_rolling_trainer_config_dc_to_rolling_trainer_config(self):
        """测试 RollingTrainerConfigDC.to_rolling_trainer_config() 方法"""
        from config.base_config import RollingTrainerConfigDC
        from model.train.rolling_window_trainer import RollingTrainerConfig
        
        # 创建 RollingTrainerConfigDC
        dc_config = RollingTrainerConfigDC(
            n_epochs=30,
            lr=0.005,
            weight_inheritance=True,
            save_each_window=False,
            reset_optimizer=False,
            gc_interval=2
        )
        
        # 转换为 RollingTrainerConfig
        rolling_config = dc_config.to_rolling_trainer_config()
        
        # 验证类型
        assert isinstance(rolling_config, RollingTrainerConfig)
        
        # 验证字段
        assert rolling_config.n_epochs == 30
        assert rolling_config.lr == 0.005
        assert rolling_config.weight_inheritance is True
        assert rolling_config.save_each_window is False
        assert rolling_config.reset_optimizer is False
    
    def test_trainer_config_fields_alignment(self):
        """验证 TrainerConfigDC 与 TrainerConfig 字段对齐"""
        from config.base_config import TrainerConfigDC
        from model.train.base_trainer import TrainerConfig
        from dataclasses import fields
        
        dc_fields = {f.name for f in fields(TrainerConfigDC)}
        trainer_fields = {f.name for f in fields(TrainerConfig)}
        
        # TrainerConfigDC 应包含 TrainerConfig 的所有字段
        missing_in_dc = trainer_fields - dc_fields
        assert len(missing_in_dc) == 0, f"TrainerConfigDC 缺少字段: {missing_in_dc}"


class TestConfigSerialization:
    """测试配置序列化/反序列化"""
    
    def test_to_dict_and_from_dict(self):
        """测试 to_dict 和 from_dict"""
        from config.base_config import TrainerConfigDC
        
        config = TrainerConfigDC(
            n_epochs=100,
            lr=0.001,
            loss_fn='huber',
            optimizer='adam'
        )
        
        # 序列化
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['n_epochs'] == 100
        assert config_dict['lr'] == 0.001
        assert config_dict['loss_fn'] == 'huber'
        
        # 反序列化
        restored = TrainerConfigDC.from_dict(config_dict)
        assert restored.n_epochs == 100
        assert restored.lr == 0.001
        assert restored.loss_fn == 'huber'
    
    def test_to_yaml_and_from_yaml(self):
        """测试 YAML 序列化/反序列化"""
        from config.base_config import TrainerConfigDC
        
        config = TrainerConfigDC(
            n_epochs=50,
            lr=0.01,
            loss_fn='ic',
            scheduler_type='plateau'
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name
        
        try:
            # 保存到 YAML
            config.to_yaml(yaml_path)
            
            # 从 YAML 加载
            restored = TrainerConfigDC.from_yaml(yaml_path)
            
            assert restored.n_epochs == 50
            assert restored.lr == 0.01
            assert restored.loss_fn == 'ic'
            assert restored.scheduler_type == 'plateau'
        finally:
            os.unlink(yaml_path)
    
    def test_nested_config_serialization(self):
        """测试嵌套配置序列化"""
        from config.base_config import TaskConfig
        
        config = TaskConfig(
            model_class="MyModel",
            dataset_class="MyDataset",
            model_kwargs={'hidden_dim': 128},
            trainer_class='SimpleTrainer'
        )
        
        # 序列化
        config_dict = config.to_dict()
        assert config_dict['model_class'] == 'MyModel'
        assert config_dict['model_kwargs']['hidden_dim'] == 128
        
        # 反序列化
        restored = TaskConfig.from_dict(config_dict)
        assert restored.model_class == 'MyModel'
        assert restored.model_kwargs['hidden_dim'] == 128


class TestConfigValidation:
    """测试配置验证逻辑"""
    
    def test_trainer_config_validate_epochs(self):
        """测试 n_epochs 验证"""
        from config.base_config import TrainerConfigDC
        
        config = TrainerConfigDC(n_epochs=0)
        with pytest.raises(ValueError, match="n_epochs 必须大于 0"):
            config.validate()
    
    def test_trainer_config_validate_lr(self):
        """测试学习率验证"""
        from config.base_config import TrainerConfigDC
        
        config = TrainerConfigDC(lr=-0.001)
        with pytest.raises(ValueError, match="lr 必须大于 0"):
            config.validate()
    
    def test_trainer_config_validate_optimizer(self):
        """测试优化器验证"""
        from config.base_config import TrainerConfigDC
        
        config = TrainerConfigDC(optimizer='invalid_optimizer')
        with pytest.raises(ValueError, match="不支持的优化器"):
            config.validate()
    
    def test_trainer_config_validate_loss_fn(self):
        """测试损失函数验证"""
        from config.base_config import TrainerConfigDC
        
        config = TrainerConfigDC(loss_fn='invalid_loss')
        with pytest.raises(ValueError, match="不支持的损失函数"):
            config.validate()
    
    def test_trainer_config_validate_supported_losses(self):
        """测试所有支持的损失函数"""
        from config.base_config import TrainerConfigDC
        
        supported_losses = [
            'mse', 'mae', 'huber', 'ic',
            'mse_corr', 'mae_corr', 'huber_corr', 'ic_corr',
            'combined', 'unified'
        ]
        
        for loss_fn in supported_losses:
            config = TrainerConfigDC(loss_fn=loss_fn)
            assert config.validate() is True
    
    def test_trainer_config_validate_scheduler_types(self):
        """测试调度器类型"""
        from model.train.base_trainer import TrainerConfig
        
        for scheduler_type in ['plateau', 'cosine', 'step']:
            config = TrainerConfig(scheduler_type=scheduler_type)
            # 只要不抛出异常就通过
            assert config.validate() is True
    
    def test_task_config_validate_empty_model_class(self):
        """测试空 model_class 验证"""
        from config.base_config import TaskConfig
        
        config = TaskConfig(model_class="", dataset_class="MyDataset")
        with pytest.raises(ValueError, match="model_class 不能为空"):
            config.validate()
    
    def test_task_config_validate_empty_dataset_class(self):
        """测试空 dataset_class 验证"""
        from config.base_config import TaskConfig
        
        config = TaskConfig(model_class="MyModel", dataset_class="")
        with pytest.raises(ValueError, match="dataset_class 不能为空"):
            config.validate()
    
    def test_task_config_validate_invalid_trainer(self):
        """测试无效训练器类型验证"""
        from config.base_config import TaskConfig
        
        config = TaskConfig(
            model_class="MyModel",
            dataset_class="MyDataset",
            trainer_class="InvalidTrainer"
        )
        with pytest.raises(ValueError, match="不支持的训练器"):
            config.validate()


class TestDeprecationWarnings:
    """测试废弃警告"""
    
    def test_trainer_config_dc_deprecation_warning(self):
        """测试 TrainerConfigDC 废弃警告"""
        from config.base_config import TrainerConfigDC
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = TrainerConfigDC()
            
            # 检查是否有废弃警告
            deprecation_warnings = [
                warning for warning in w 
                if issubclass(warning.category, DeprecationWarning)
            ]
            
            # 应该有废弃警告
            assert len(deprecation_warnings) >= 1
            assert "model.train.TrainerConfig" in str(deprecation_warnings[0].message)
    
    def test_rolling_trainer_config_dc_deprecation_warning(self):
        """测试 RollingTrainerConfigDC 废弃警告"""
        from config.base_config import RollingTrainerConfigDC
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = RollingTrainerConfigDC()
            
            # 检查是否有废弃警告
            deprecation_warnings = [
                warning for warning in w 
                if issubclass(warning.category, DeprecationWarning)
            ]
            
            # 应该有废弃警告
            assert len(deprecation_warnings) >= 1


class TestBaseConfigMethods:
    """测试 BaseConfig 基本方法"""
    
    def test_merge_method(self):
        """测试配置合并方法"""
        from config.base_config import TrainerConfigDC
        
        base_config = TrainerConfigDC(n_epochs=100, lr=0.001)
        override = {'n_epochs': 50, 'weight_decay': 1e-4}
        
        merged = base_config.merge(override)
        
        assert merged.n_epochs == 50  # 被覆盖
        assert merged.lr == 0.001  # 保持不变
        assert merged.weight_decay == 1e-4  # 新增
    
    def test_copy_method(self):
        """测试配置复制方法"""
        from config.base_config import TrainerConfigDC
        
        config = TrainerConfigDC(n_epochs=100)
        copied = config.copy()
        
        # 修改原配置不影响副本
        config.n_epochs = 50
        assert copied.n_epochs == 100
    
    def test_update_method(self):
        """测试配置更新方法"""
        from config.base_config import TrainerConfigDC
        
        config = TrainerConfigDC(n_epochs=100, lr=0.001)
        config.update(n_epochs=50, weight_decay=1e-4)
        
        assert config.n_epochs == 50
        assert config.lr == 0.001
        assert config.weight_decay == 1e-4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
