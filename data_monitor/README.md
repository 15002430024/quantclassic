# Data Monitor - 数据泄漏检测模块

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.7+-green.svg)]()

## 📖 概述

`data_monitor` 是 quantclassic 项目的数据泄漏检测模块，提供了一套完整的工具来检测机器学习模型训练和推理过程中可能存在的时间泄漏问题。

### 主要特性

- ✅ **静态检测**: 代码分析，无需运行模型
- 🔄 **动态监控**: 运行时数据访问追踪
- 📊 **自动报告**: 详细的测试报告生成
- ⚙️ **灵活配置**: 多种检测模式和参数
- 🎯 **高封装度**: 按照 quantclassic 风格设计

## 🚀 快速开始

### 安装

```bash
# quantclassic 项目中已包含此模块
cd /path/to/quantclassic
```

### 基础使用

```python
from quantclassic.data_monitor import LeakageDetector

# 快速检查
detector = LeakageDetector.quick_check()
results = detector.detect(model, data)

# 查看结果
if detector.is_passed():
    print("✅ 通过检测")
else:
    print("❌ 发现泄漏:", detector.get_failed_tests())
```

## 📚 模块结构

```
data_monitor/
├── __init__.py                      # 模块入口
├── leakage_detection_config.py      # 配置类
├── static_leakage_detector.py       # 静态检测器
├── dynamic_leakage_detector.py      # 动态检测器
├── leakage_detector.py              # 主检测器
├── example_leakage_detection.py     # 使用示例
├── LEAKAGE_DETECTION_GUIDE.md       # 详细文档
└── README.md                        # 本文件
```

## 🔍 检测内容

### 静态检测

| 检测项 | 说明 |
|--------|------|
| **特征窗口检测** | 验证特征窗口是否包含当前月数据 |
| **因子输入检测** | 检查因子输入是否使用当期数据 |
| **calFactor检测** | 验证因子计算是否使用历史数据 |
| **源代码分析** | 分析代码中的可疑模式 |

### 动态监控

| 监控项 | 说明 |
|--------|------|
| **数据访问监控** | 追踪运行时的数据访问行为 |
| **时间边界检查** | 强制执行时间边界限制 |
| **缓存增长监控** | 检测异常的缓存增长 |

## 💡 使用示例

### 示例1: 快速检查（静态）

```python
from quantclassic.data_monitor import LeakageDetector

detector = LeakageDetector.quick_check(verbose=True)
results = detector.detect(model, data)
```

### 示例2: 完整验证（静态+动态）

```python
detector = LeakageDetector.full_validation(
    verbose=True,
    generate_report=True
)

results = detector.detect(
    model=your_model,
    data=your_data,
    train_months=[200701, 200702, 200703],
    test_start_month=201901
)
```

### 示例3: 自定义配置

```python
from quantclassic.data_monitor import LeakageDetectionConfig, LeakageTestMode

config = LeakageDetectionConfig(
    test_mode=LeakageTestMode.FULL,
    verbose=True,
    check_feature_window=True,
    monitor_data_access=True,
    generate_report=True,
    report_path='./my_report.txt'
)

detector = LeakageDetector(config)
results = detector.detect(model, data)
```

## ⚙️ 配置参数

### 检测模式

```python
LeakageTestMode.STATIC_ONLY   # 仅静态（快速）
LeakageTestMode.DYNAMIC_ONLY  # 仅动态
LeakageTestMode.FULL          # 完整检测（推荐）
```

### 主要参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `test_mode` | LeakageTestMode | FULL | 检测模式 |
| `verbose` | bool | True | 详细输出 |
| `time_column` | str | 'year_month' | 时间列名 |
| `stock_column` | str | 'ts_code' | 股票列名 |
| `check_feature_window` | bool | True | 特征窗口检查 |
| `check_factor_input` | bool | True | 因子输入检查 |
| `monitor_data_access` | bool | True | 数据访问监控 |
| `enforce_time_boundary` | bool | True | 时间边界强制 |
| `generate_report` | bool | True | 生成报告 |
| `report_path` | str | './leakage_detection_report.txt' | 报告路径 |

完整参数列表请参考 [配置文档](./leakage_detection_config.py)

## 📋 模型要求

被检测的模型需要满足：

### 必需方法

```python
def _get_item(self, month: int):
    """
    获取指定月份的数据
    返回: (stock_index, features, factor_inputs, labels)
    """
    pass
```

### 可选方法/属性

```python
def calFactor(self, month: int):
    """计算因子"""
    pass

self.window_len = 12        # 窗口长度
self._data_cache = {}       # 数据缓存
```

## 📊 数据要求

数据必须是 pandas DataFrame，包含：

- **必需列**: `time_column`（如 'year_month'）、`stock_column`（如 'ts_code'）
- **推荐列**: `return_column`（如 'rm_rf'）、`label_column`（如 'target'）

## 📈 使用场景

| 场景 | 推荐配置 | 说明 |
|------|----------|------|
| **开发调试** | `quick_check()` | 静态检测，速度快 |
| **上线前验证** | `full_validation()` | 完整检测，最可靠 |
| **CI/CD** | `verbose=False` | 自动化检测 |
| **运行时监控** | `runtime_monitor()` | 轻量级监控 |

## 🔧 高级功能

### 从配置文件加载

```yaml
# config.yaml
test_mode: full
verbose: true
check_feature_window: true
monitor_data_access: true
```

```python
detector = LeakageDetector('config.yaml')
```

### 批量检测

```python
models = [model1, model2, model3]
detector = LeakageDetector.full_validation(verbose=False)

for model in models:
    results = detector.detect(model, data)
    if not detector.is_passed():
        print(f"失败: {detector.get_failed_tests()}")
```

### CI/CD 集成

```python
config = LeakageDetectionConfig(
    test_mode=LeakageTestMode.FULL,
    verbose=False,
    generate_report=True
)
detector = LeakageDetector(config)
results = detector.detect(model, data)

if not detector.is_passed():
    sys.exit(1)  # 失败退出
```

## 📝 完整文档

- [详细使用指南](./LEAKAGE_DETECTION_GUIDE.md) - 完整的使用文档
- [配置参数文档](./leakage_detection_config.py) - 所有配置参数说明
- [使用示例](./example_leakage_detection.py) - 可运行的示例代码

## 🧪 运行示例

```bash
cd /home/u2025210237/jupyterlab/quantclassic/data_monitor
python example_leakage_detection.py
```

## ⚠️ 常见问题

### Q: 如何处理"模型缺少 _get_item 方法"错误？

**A**: 确保模型实现了 `_get_item(month)` 方法。

### Q: 检测太慢怎么办？

**A**: 
1. 使用 `LeakageTestMode.STATIC_ONLY`
2. 减少 `test_stocks_limit`
3. 指定较少的测试月份

### Q: 如何自定义列名？

**A**: 在配置中指定：
```python
config = LeakageDetectionConfig(
    time_column='my_time',
    stock_column='my_stock'
)
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本模块是 quantclassic 项目的一部分，遵循项目许可证。

## 🔗 相关链接

- quantclassic 项目: `/home/u2025210237/jupyterlab/quantclassic`
- 数据处理模块: `../data_processor`
- 模型模块: `../model`

---

**Version**: 0.1.0  
**Author**: quantclassic team  
**Last Updated**: 2025-11-24
