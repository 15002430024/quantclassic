# FactorHub - 量化因子计算框架

<div align="center">

**从数据到因子的端到端解决方案**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Type Hints](https://img.shields.io/badge/Type%20Hints-100%25-brightgreen.svg)](https://www.python.org/dev/peps/pep-0484/)

</div>

---

## 🎯 核心特性

| 特性 | 说明 |
|------|------|
| 📊 **标准化数据协议** | 定义统一的数据格式，解耦数据源和因子逻辑 |
| 🔌 **适配器模式** | 轻松对接任何外部数据源（Wind、Tushare、datafetch...）|
| 🏭 **因子工厂** | 装饰器自动注册，动态加载因子 |
| ⚙️ **计算引擎** | 智能调度、异常容错、结果聚合 |
| 💾 **多格式输出** | 支持 CSV、Parquet、Pickle 等格式 |
| 🔍 **Type Hints** | 100% 类型注解，IDE 友好 |

---

## 🚀 快速开始

### 安装依赖

```bash
cd /Users/shiyunshuo/Desktop/pythonproject
pip install pandas numpy pyarrow
```

### 5分钟上手

```python
import sys
sys.path.insert(0, '/Users/shiyunshuo/Desktop/pythonproject')

from quantclassic.factor_hub import FactorEngine, MockDataProvider
from quantclassic.factor_hub.factors import demo_factors  # 导入Demo因子

# 1. 创建数据提供者
provider = MockDataProvider(seed=2024)

# 2. 初始化因子引擎
engine = FactorEngine(provider)

# 3. 运行因子计算
result = engine.run(
    symbols=["000001.SZ", "600000.SH"],
    factor_names=["return_1d", "volatility", "price_range"],
    start="2024-01-01",
    end="2024-03-31"
)

# 4. 查看结果
print(result.factors_data.head())
print(f"成功因子: {result.successful_factors}")
```

---

## 📖 核心概念

### 1. 标准化数据协议

```python
from quantclassic.factor_hub.protocols import StandardDataProtocol

std_data = StandardDataProtocol(raw_df)
std_data.close        # 获取收盘价
std_data.symbols      # 股票列表
std_data.start_date   # 起始日期
```

### 2. 因子开发

```python
from quantclassic.factor_hub.factors import BaseFactor, factor_registry

@factor_registry.register
class MyFactor(BaseFactor):
    @property
    def meta(self):
        return FactorMeta(name="my_factor", category="custom")
    
    def compute(self, data):
        return data.close.pct_change()
```

### 3. 数据源对接

```python
from quantclassic.factor_hub.providers import BaseDataAdapter

class MyAdapter(BaseDataAdapter):
    def get_history(self, symbols, start, end):
        # 调用外部 API
        raw = my_api.fetch(...)
        # 转换为标准格式
        return self._to_standard(raw)
```

---

## 🧪 验证测试

```bash
# 测试各个模块
python quantclassic/factor_hub/tests/test_step1_protocol.py
python quantclassic/factor_hub/tests/test_step2_factors.py
python quantclassic/factor_hub/tests/test_step3_engine.py

# 端到端测试
python quantclassic/factor_hub/main.py
```

---

## 📊 内置因子

| 因子名 | 类别 | 说明 |
|--------|------|------|
| `return_1d` | momentum | 1日收益率 |
| `return_5d` | momentum | 5日收益率 |
| `volatility` | risk | N日波动率 |
| `turnover_ratio` | liquidity | 换手率 |
| `price_range` | volatility | 日内振幅 |

---

## 📚 文档

- [快速开始](快速开始.md)
- [架构设计](ARCHITECTURE.md)
- [系统架构文档](系统架构文档.md)

---

## 🤝 贡献

欢迎贡献新因子或改进代码！

---

<div align="center">

**FactorHub v1.0.0** - 让因子计算变得简单而优雅

</div>
