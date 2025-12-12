# 数据泄漏检测工具使用指南

## 📖 简介

`quantclassic.data_monitor` 模块提供了一套完整的数据泄漏检测工具，用于在模型训练前验证数据处理流程是否存在时间泄漏问题。

### 主要功能

- **静态检测**: 通过代码分析检测明显的时间泄漏
- **动态监控**: 在运行时监控数据访问模式
- **自动化报告**: 生成详细的检测报告
- **灵活配置**: 支持多种检测模式和参数

## 🚀 快速开始

### 1. 基础使用

```python
from quantclassic.data_monitor import LeakageDetector
import pandas as pd

# 加载数据
data = pd.read_csv('your_data.csv')

# 创建检测器（快速检查模式）
detector = LeakageDetector.quick_check()

# 执行检测
results = detector.detect(model, data)

# 查看结果
if detector.is_passed():
    print("✅ 所有测试通过")
else:
    print("❌ 发现数据泄漏问题")
    print("失败的测试:", detector.get_failed_tests())
```

### 2. 完整验证

```python
from quantclassic.data_monitor import LeakageDetector

# 完整验证模式（静态 + 动态）
detector = LeakageDetector.full_validation(
    verbose=True,
    generate_report=True
)

# 执行检测
results = detector.detect(
    model=your_model,
    data=your_data,
    train_months=[200701, 200702, 200703],  # 可选：指定训练月份
    test_start_month=201901  # 可选：指定测试期开始
)

# 报告自动保存到 ./leakage_detection_report.txt
```

### 3. 自定义配置

```python
from quantclassic.data_monitor import LeakageDetector, LeakageDetectionConfig, LeakageTestMode

# 创建自定义配置
config = LeakageDetectionConfig(
    test_mode=LeakageTestMode.FULL,
    verbose=True,
    
    # 列名配置
    time_column='year_month',
    stock_column='ts_code',
    return_column='rm_rf',
    label_column='target',
    
    # 静态检测开关
    check_feature_window=True,
    check_factor_input=True,
    check_calFactor=True,
    
    # 动态检测开关
    monitor_data_access=True,
    monitor_cache_growth=True,
    enforce_time_boundary=True,
    
    # 报告配置
    generate_report=True,
    report_path='./my_leakage_report.txt',
    show_summary=True
)

# 创建检测器
detector = LeakageDetector(config)

# 执行检测
results = detector.detect(model, data)
```

## 📋 配置说明

### 检测模式

```python
from quantclassic.data_monitor import LeakageTestMode

# 三种检测模式
LeakageTestMode.STATIC_ONLY   # 仅静态检测（快速）
LeakageTestMode.DYNAMIC_ONLY  # 仅动态监控
LeakageTestMode.FULL          # 完整检测（推荐）
```

### 静态检测项

| 检测项 | 配置参数 | 说明 |
|--------|----------|------|
| 特征窗口检测 | `check_feature_window` | 检测特征窗口是否包含当前月 |
| 因子输入检测 | `check_factor_input` | 检测因子输入是否使用当期数据 |
| calFactor检测 | `check_calFactor` | 检测calFactor是否使用历史数据 |
| 源代码分析 | `check_source_code` | 分析源代码中的可疑模式 |

### 动态监控项

| 监控项 | 配置参数 | 说明 |
|--------|----------|------|
| 数据访问监控 | `monitor_data_access` | 监控运行时的数据访问 |
| 缓存增长监控 | `monitor_cache_growth` | 检测异常的缓存增长 |
| 时间边界检查 | `enforce_time_boundary` | 强制执行时间边界限制 |

## 🔍 模型要求

被检测的模型需要满足以下要求：

### 必需方法

```python
class YourModel:
    def _get_item(self, month: int):
        """
        获取指定月份的数据
        
        Args:
            month: 月份，格式如 202101
        
        Returns:
            (stock_index, features, factor_inputs, labels) 或类似结构
        """
        pass
```

### 可选方法

```python
class YourModel:
    def calFactor(self, month: int):
        """
        计算指定月份的因子
        
        Args:
            month: 月份，格式如 202101
        
        Returns:
            因子值
        """
        pass
```

### 可选属性

```python
class YourModel:
    def __init__(self):
        self.window_len = 12  # 特征窗口长度
        self._data_cache = {}  # 数据缓存
```

## 📊 数据要求

数据必须是 pandas DataFrame，包含以下列：

| 列名 | 默认名称 | 说明 | 必需 |
|------|----------|------|------|
| 时间列 | `year_month` | 格式: YYYYMM | ✅ |
| 股票列 | `ts_code` | 股票代码 | ✅ |
| 收益率列 | `rm_rf` | 用于因子输入检测 | 推荐 |
| 标签列 | `target` | 标签 | 推荐 |

可以通过配置自定义列名：

```python
config = LeakageDetectionConfig(
    time_column='my_time_col',
    stock_column='my_stock_col',
    return_column='my_return_col',
    label_column='my_label_col'
)
```

## 📈 使用场景

### 场景1: 开发阶段快速检查

```python
# 静态检测，速度快
detector = LeakageDetector.quick_check(verbose=True)
results = detector.detect(model, data)
```

### 场景2: 模型上线前完整验证

```python
# 完整检测（静态+动态）
detector = LeakageDetector.full_validation(
    verbose=True,
    generate_report=True
)
results = detector.detect(model, data)
```

### 场景3: CI/CD自动化检测

```python
# 非详细模式，生成报告
config = LeakageDetectionConfig(
    test_mode=LeakageTestMode.FULL,
    verbose=False,
    generate_report=True,
    show_summary=False
)
detector = LeakageDetector(config)
results = detector.detect(model, data)

# 检查结果
if not detector.is_passed():
    raise ValueError("数据泄漏检测失败！")
```

### 场景4: 训练中实时监控

```python
# 运行时监控
detector = LeakageDetector.runtime_monitor()
results = detector.detect(model, data)
```

## 🎯 结果解读

### 测试结果

```python
# 获取测试结果
results = detector.get_test_results()
# 返回: {'feature_window': True, 'factor_input': False, ...}

# 获取详细结果
detailed = detector.get_detailed_results()
# 返回: {'feature_window': {'passed': True, 'message': '...', 'details': {...}}, ...}

# 判断是否全部通过
passed = detector.is_passed()

# 获取失败的测试
failed = detector.get_failed_tests()
```

### 报告内容

生成的报告包含：

1. **基本信息**: 检测时间、模式、配置
2. **测试结果**: 每项测试的通过/失败状态
3. **详细信息**: 失败原因、相关数据
4. **修复建议**: 针对性的修复方案

## 🔧 高级用法

### 从YAML配置文件加载

```yaml
# leakage_config.yaml
test_mode: full
verbose: true
time_column: year_month
stock_column: ts_code

check_feature_window: true
check_factor_input: true
monitor_data_access: true

generate_report: true
report_path: ./reports/leakage_report.txt
```

```python
# 加载配置
detector = LeakageDetector('leakage_config.yaml')
results = detector.detect(model, data)
```

### 使用模板配置

```python
from quantclassic.data_monitor import LeakageDetectionTemplates

# 快速检查模板
config = LeakageDetectionTemplates.quick_check()

# 完整验证模板
config = LeakageDetectionTemplates.full_validation()

# 运行时监控模板
config = LeakageDetectionTemplates.runtime_monitor()

# 自定义模板
config = LeakageDetectionTemplates.custom(
    test_mode='full',
    verbose=True,
    check_feature_window=True
)
```

### 批量检测多个模型

```python
models = [model1, model2, model3]
detector = LeakageDetector.full_validation(verbose=False)

for i, model in enumerate(models):
    print(f"\n检测模型 {i+1}")
    results = detector.detect(model, data)
    
    if not detector.is_passed():
        print(f"  ⚠️ 模型 {i+1} 存在数据泄漏")
        print(f"  失败测试: {detector.get_failed_tests()}")
```

## ⚠️ 常见问题

### Q1: 如何处理"模型缺少 _get_item 方法"错误？

**A**: 确保模型实现了 `_get_item(month)` 方法，该方法应返回指定月份的数据。

### Q2: 检测速度太慢怎么办？

**A**: 
1. 使用静态检测模式（`LeakageTestMode.STATIC_ONLY`）
2. 减少 `test_stocks_limit` 参数
3. 指定较少的 `test_months`

### Q3: 如何自定义列名？

**A**: 在配置中指定列名：

```python
config = LeakageDetectionConfig(
    time_column='your_time_col',
    stock_column='your_stock_col'
)
```

### Q4: 为什么某些测试被跳过？

**A**: 
1. 检查模型是否有相应的方法（如 `calFactor`）
2. 确认配置中相应的检查开关是否开启

## 📝 最佳实践

1. **开发阶段**: 使用快速检查模式，频繁验证
2. **测试阶段**: 使用完整验证，生成报告
3. **生产环境**: 在训练前自动运行检测
4. **持续集成**: 将检测集成到 CI/CD 流程

## 🔗 相关文档

- [配置参数完整列表](./leakage_detection_config.py)
- [静态检测器文档](./static_leakage_detector.py)
- [动态检测器文档](./dynamic_leakage_detector.py)

## 💡 示例代码

完整示例请参考: [example_leakage_detection.py](./example_leakage_detection.py)
