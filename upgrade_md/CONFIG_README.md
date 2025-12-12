# QuantClassic 配置系统 v2.0

## 🎉 重大更新：面向对象配置系统

QuantClassic 配置系统已完成重构，从字典配置升级到**面向对象配置**！

### ✨ 新特性

- ✅ **类型安全** - 编译时类型检查
- ✅ **自动验证** - 参数自动验证，防止错误
- ✅ **统一接口** - 所有配置类继承 BaseConfig
- ✅ **配置工厂** - 快速创建常用配置
- ✅ **预定义模板** - 开箱即用的配置模板
- ✅ **完全兼容** - 向后兼容旧的字典配置

## 🚀 快速开始

### 创建配置

```python
from quantclassic.model.model_config import VAEConfig

# 创建配置（自动验证）
config = VAEConfig(
    hidden_dim=128,
    latent_dim=16,
    n_epochs=100,
    learning_rate=0.001
)

# 保存配置
config.to_yaml('my_config.yaml')

# 加载配置
loaded = VAEConfig.from_yaml('my_config.yaml')
```

### 使用工厂和模板

```python
from quantclassic.model.model_config import ModelConfigFactory

# 使用工厂
config = ModelConfigFactory.create('vae', hidden_dim=256)

# 使用预定义模板
small_vae = ModelConfigFactory.get_template('vae', 'small')
large_vae = ModelConfigFactory.get_template('vae', 'large')
```

### 完整工作流

```python
from quantclassic.config.loader import ConfigLoader
from quantclassic.model.model_config import VAEConfig
from quantclassic.data_manager.config import DataConfig

# 加载配置文件
full_config = ConfigLoader.load('config.yaml', return_dict=True)

# 创建配置对象
model_config = VAEConfig.from_dict(full_config['model'])
data_config = DataConfig.from_dict(full_config['data'])

# 自动验证
model_config.validate()
data_config.validate()
```

## 📚 文档

- **[迁移指南](MIGRATION_GUIDE.md)** - 从旧配置迁移到新配置
- **[快速参考](CONFIG_QUICKREF.md)** - 常用代码片段和API
- **[重构总结](REFACTORING_SUMMARY.md)** - 重构内容和特性
- **[完成报告](REFACTORING_COMPLETE.md)** - 测试结果和交付清单

## 📦 配置类

### 模型配置
- `VAEConfig` - VAE 模型
- `LSTMConfig` - LSTM 模型
- `GRUConfig` - GRU 模型
- `TransformerConfig` - Transformer 模型
- `MLPConfig` - MLP 模型

### 数据配置
- `DataConfig` - 数据管理配置
- `PreprocessConfig` - 数据预处理配置
- `WorkflowConfig` - 工作流配置

## 🔄 向后兼容

旧代码无需修改即可运行：

```python
# 旧方式（仍然支持）
config = ConfigLoader.load('config.yaml')  # 返回字典
```

## 📝 配置模板

查看 `config/templates/vae_oop.yaml` 获取完整的配置示例。

## 🧪 测试

运行测试验证配置系统：

```bash
python test_config_standalone.py
```

测试结果：✅ 6/6 通过

## 💡 最佳实践

1. ✅ 新项目使用面向对象配置
2. ✅ 利用工厂和模板快速开始
3. ✅ 配置文件加入版本控制
4. ✅ 先验证后使用配置

## 📊 改进对比

| 特性 | 旧方式（字典） | 新方式（OOP） |
|------|---------------|--------------|
| 类型检查 | ❌ | ✅ 编译时 |
| 参数验证 | ❌ 手动 | ✅ 自动 |
| IDE 支持 | ❌ | ✅ 自动完成 |
| 可维护性 | ❌ | ✅ 高 |

## 🎯 示例代码

### 创建并保存配置

```python
config = VAEConfig(hidden_dim=128, latent_dim=16)
config.to_yaml('config.yaml')
```

### 加载并修改配置

```python
config = VAEConfig.from_yaml('config.yaml')
config.update(hidden_dim=256, learning_rate=0.002)
```

### 合并配置

```python
base_config = VAEConfig()
custom_config = VAEConfig(latent_dim=32)
merged = base_config.merge(custom_config)
```

## 🔗 相关链接

- [BaseConfig API](config/base_config.py)
- [模型配置](model/model_config.py)
- [数据配置](data_manager/config.py)
- [预处理配置](data_processor/preprocess_config.py)
- [工作流配置](workflow/workflow_config.py)

## 📅 版本历史

### v2.0.0 (2025-11-20)
- ✅ 全新的面向对象配置系统
- ✅ 配置工厂和模板
- ✅ 完整的文档和测试
- ✅ 向后兼容旧配置

---

**需要帮助？** 查看 [迁移指南](MIGRATION_GUIDE.md) 或 [快速参考](CONFIG_QUICKREF.md)
