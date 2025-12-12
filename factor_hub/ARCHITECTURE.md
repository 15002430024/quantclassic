# FactorHub 系统架构文档

## 📐 设计哲学

FactorHub 是一个专业的量化因子计算框架，遵循以下设计原则：

### SOLID 原则

1. **单一职责原则 (SRP)**: 每个模块只负责一个功能
   - `protocols/`: 只定义数据协议
   - `providers/`: 只负责数据获取
   - `factors/`: 只负责因子计算
   - `engine/`: 只负责流程调度
   - `io/`: 只负责结果输出

2. **开闭原则 (OCP)**: 对扩展开放，对修改关闭
   - 添加新因子：只需继承 `BaseFactor` 并注册
   - 添加新数据源：只需实现 `IDataProvider` 接口
   - 添加新输出格式：只需实现 `IFactorWriter` 接口

3. **里氏替换原则 (LSP)**: 子类可以替换父类
   - 所有 Provider 都可以互换使用
   - 所有 Writer 都可以互换使用

4. **接口隔离原则 (ISP)**: 接口最小化
   - `IDataProvider` 只定义必要的数据获取方法
   - `IFactorWriter` 只定义必要的写入方法

5. **依赖倒置原则 (DIP)**: 依赖抽象而非具体
   - `FactorEngine` 依赖 `IDataProvider` 接口，而非具体实现
   - 使用依赖注入传递 Provider

### 设计模式

| 模式 | 应用位置 | 目的 |
|------|---------|------|
| **适配器模式** | `DataFetchAdapter` | 对接外部数据源 |
| **注册表模式** | `FactorRegistry` | 管理因子类 |
| **模板方法模式** | `BaseFactor` | 定义因子计算框架 |
| **策略模式** | `IFactorWriter` | 支持多种输出策略 |
| **工厂模式** | `FactorWriterFactory` | 创建写入器实例 |
| **装饰器模式** | `@factor_registry.register` | 自动注册因子 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     FactorHub System                         │
│                                                              │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐        │
│  │   Main     │───▶│   Engine   │───▶│   Writer   │        │
│  │  (Entry)   │    │ (Scheduler)│    │  (Output)  │        │
│  └────────────┘    └─────┬──────┘    └────────────┘        │
│                          │                                   │
│              ┌───────────┼───────────┐                      │
│              │           │           │                       │
│         ┌────▼─────┐ ┌──▼────┐ ┌───▼──────┐               │
│         │ Provider │ │Factor │ │ Protocol │               │
│         │(Adapter) │ │(Algo) │ │  (Std)   │               │
│         └──────────┘ └───────┘ └──────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### 数据流向

```
Raw Data → Adapter → StandardProtocol → FactorEngine → Factors → Writer → Files
   ↑          ↑           ↑                  ↑            ↑          ↑        ↑
External  Interface   Validation          Scheduler   Compute    Format   Output
```

---

## 📦 模块详解

### 1. protocols/ - 数据协议层

**职责**: 定义系统内部数据交换的"标准语言"

**核心类**:
- `StandardDataProtocol`: 标准化数据容器
  - Index: `MultiIndex(datetime, symbol)`
  - Columns: `open, high, low, close, volume, amount, vwap`
  - Validation: 数据完整性校验
  - Access: 便捷的数据访问接口

**设计要点**:
```python
# 数据必须符合的格式约定
REQUIRED_COLUMNS = {
    "symbol",    # 股票代码
    "datetime",  # 日期时间
    "open",      # 开盘价
    "high",      # 最高价
    "low",       # 最低价
    "close",     # 收盘价
    "volume",    # 成交量
}

# 自动标准化：列 → MultiIndex
data = StandardDataProtocol(raw_df)  # 自动转换
```

**优势**:
- ✅ 解耦：因子无需关心原始数据格式
- ✅ 验证：自动检查数据完整性
- ✅ 便利：`data.close` 直接访问收盘价

---

### 2. providers/ - 数据提供层

**职责**: 从外部数据源获取数据并转换为标准格式

**接口设计**:
```python
class IDataProvider(ABC):
    @abstractmethod
    def get_history(
        self, 
        symbols: List[str], 
        start: str, 
        end: str
    ) -> pd.DataFrame:
        """返回符合 StandardDataProtocol 的数据"""
        pass
```

**实现类**:

| 类名 | 用途 | 状态 |
|------|------|------|
| `MockDataProvider` | 测试数据生成器 | ✅ 已实现 |
| `DataFetchAdapter` | 外部 datafetch 包适配器 | 🔧 接口预留 |

**适配器模式示例**:
```python
# 步骤1: 实现适配器
class MyDataAdapter(BaseDataAdapter):
    def get_history(self, symbols, start, end):
        # 调用外部 API
        raw = external_api.fetch(...)
        # 转换为标准格式
        return self._to_standard_format(raw)

# 步骤2: 注入到 Engine
provider = MyDataAdapter()
engine = FactorEngine(provider)  # 依赖注入
```

**优势**:
- ✅ 扩展性：添加新数据源无需修改引擎
- ✅ 可测试：可以用 Mock 替换真实数据
- ✅ 灵活性：同一套因子可用于不同数据源

---

### 3. factors/ - 因子计算层

**职责**: 定义因子计算逻辑和注册机制

**核心组件**:

#### BaseFactor - 因子基类
```python
class BaseFactor(ABC):
    @property
    @abstractmethod
    def meta(self) -> FactorMeta:
        """因子元数据"""
        pass
    
    @abstractmethod
    def compute(self, data: StandardDataProtocol) -> pd.Series:
        """计算因子值"""
        pass
```

#### FactorRegistry - 注册中心
```python
# 装饰器自动注册
@factor_registry.register
class Return1DFactor(BaseFactor):
    @property
    def meta(self):
        return FactorMeta(
            name="return_1d",
            description="1日收益率",
            category="momentum"
        )
    
    def compute(self, data):
        return data.close.groupby(level="symbol").pct_change(1)
```

**注册机制流程**:
1. 装饰器在类定义时自动触发
2. 提取 `meta.name` 作为注册键
3. 存入全局 `factor_registry` 字典
4. Engine 通过名称查找并实例化

**已实现的Demo因子**:

| 因子名 | 类别 | 说明 |
|--------|------|------|
| `return_1d` | momentum | 1日收益率 |
| `return_5d` | momentum | 5日收益率 |
| `volatility` | risk | N日波动率 |
| `turnover_ratio` | liquidity | 换手率 |
| `price_range` | volatility | 日内振幅 |

**扩展新因子**:
```python
@factor_registry.register
class MyCustomFactor(BaseFactor):
    @property
    def meta(self):
        return FactorMeta(name="my_factor", ...)
    
    @property
    def default_params(self):
        return {"window": 20}  # 默认参数
    
    def compute(self, data):
        # 自定义逻辑
        return ...
```

---

### 4. engine/ - 调度引擎层

**职责**: 协调数据获取、因子计算和异常处理

**核心流程** (`FactorEngine.run`):

```python
def run(symbols, factor_names, start, end):
    # Step A: 获取原始数据
    raw_data = self._provider.get_history(symbols, start, end)
    
    # Step B: 校验数据格式
    std_data = StandardDataProtocol(raw_data)
    
    # Step C & D: 遍历因子计算 (带异常捕获)
    for factor_name in factor_names:
        try:
            factor = factor_registry.get(factor_name)()
            result = factor.compute(std_data)
            results.append(result)
        except Exception as e:
            if continue_on_error:
                log_error(e)
                continue
            else:
                raise
    
    # Step E: 拼接结果
    df = pd.concat(results, axis=1)
    return df
```

**容错机制**:
- `continue_on_error=True`: 单个因子失败不影响其他因子
- 记录详细错误信息: `FactorComputeResult.error`
- 返回成功/失败因子列表

**优势**:
- ✅ 健壮性：单点故障不影响全局
- ✅ 可观测性：详细的执行日志
- ✅ 灵活性：支持部分成功

---

### 5. io/ - 输出层

**职责**: 将因子结果写入不同格式

**接口设计**:
```python
class IFactorWriter(ABC):
    @abstractmethod
    def write(
        self, 
        data: pd.DataFrame, 
        path: str
    ) -> str:
        """写入文件，返回实际路径"""
        pass
```

**已实现的Writer**:

| Writer | 格式 | 特点 |
|--------|------|------|
| `CSVWriter` | CSV | 文本格式，易读 |
| `ParquetWriter` | Parquet | 列式存储，高效 |
| `PickleWriter` | Pickle | Python 原生序列化 |

**工厂模式**:
```python
# 根据格式自动创建
writer = FactorWriterFactory.create("parquet", compression="snappy")
writer.write(df, "output/factors.parquet")

# 根据路径自动识别
writer = FactorWriterFactory.from_path("output/factors.csv")
```

---

## 🔧 关键技术决策

### 为什么使用 MultiIndex？

**原因**:
1. **性能**: GroupBy 操作更快
2. **对齐**: 因子自动按 (datetime, symbol) 对齐
3. **兼容**: 符合 Pandas 的最佳实践

**示例**:
```python
# MultiIndex 让分组操作更简洁
returns = close.groupby(level="symbol").pct_change()

# 而不是
returns = close.groupby("symbol").pct_change()
```

### 为什么使用依赖注入？

**原因**:
1. **可测试性**: 方便 Mock
2. **灵活性**: 运行时切换数据源
3. **解耦**: Engine 不依赖具体 Provider

**示例**:
```python
# 测试时用 Mock
engine = FactorEngine(MockDataProvider())

# 生产环境用真实数据
engine = FactorEngine(DataFetchAdapter(api_key="xxx"))
```

### 为什么使用装饰器注册？

**原因**:
1. **自动化**: 无需手动调用注册
2. **可见性**: 类定义时明确标记
3. **简洁性**: 减少样板代码

**对比**:
```python
# ❌ 手动注册（繁琐）
class MyFactor(BaseFactor):
    pass
factor_registry.add("my_factor", MyFactor)

# ✅ 装饰器注册（优雅）
@factor_registry.register
class MyFactor(BaseFactor):
    pass
```

---

## 📈 性能优化

### 数据处理优化

1. **向量化计算**: 使用 Pandas 矢量操作
```python
# ✅ 向量化
returns = close.pct_change()

# ❌ 循环
returns = [close[i] / close[i-1] - 1 for i in range(1, len(close))]
```

2. **分组优化**: 利用 MultiIndex
```python
# ✅ 高效
volatility = returns.groupby(level="symbol").rolling(20).std()

# ❌ 低效
for symbol in symbols:
    vol = returns[returns.symbol == symbol].rolling(20).std()
```

3. **内存优化**: 使用合适的数据类型
```python
# 优化前：默认 float64 (8 bytes)
# 优化后：float32 (4 bytes) 或 category
```

### 并行计算（未来扩展）

```python
# 因子间并行（未来版本）
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor() as executor:
    futures = [
        executor.submit(compute_factor, name, data)
        for name in factor_names
    ]
    results = [f.result() for f in futures]
```

---

## 🛡️ 错误处理策略

### 分层异常

```python
# 数据层
class DataFetchError(Exception):
    """数据获取失败"""
    pass

# 协议层
class DataValidationError(Exception):
    """数据校验失败"""
    pass

# 因子层
class FactorComputeError(Exception):
    """因子计算失败"""
    pass
```

### 容错策略

| 层级 | 策略 | 行为 |
|------|------|------|
| Engine | `continue_on_error` | 单个因子失败继续 |
| Provider | 重试机制 | 网络错误自动重试 |
| Protocol | 严格校验 | 数据格式错误立即失败 |

---

## 🔮 扩展指南

### 添加新因子

```python
# 1. 继承 BaseFactor
@factor_registry.register
class MyFactor(BaseFactor):
    # 2. 定义元数据
    @property
    def meta(self):
        return FactorMeta(
            name="my_factor",
            description="我的自定义因子",
            category="custom"
        )
    
    # 3. 实现计算逻辑
    def compute(self, data):
        # 使用标准化数据
        close = data.close
        # 返回 Series
        return close.rolling(10).mean()
```

### 添加新数据源

```python
# 1. 实现 IDataProvider 接口
class MyDataProvider(BaseDataAdapter):
    @property
    def name(self):
        return "MyProvider"
    
    def get_history(self, symbols, start, end, fields):
        # 调用外部 API
        raw = my_api.fetch(...)
        # 转换为标准格式
        return self._to_standard_format(raw)
    
    def _to_standard_format(self, raw):
        # 确保包含必要字段
        df = pd.DataFrame(raw)
        df = df.rename(columns={...})
        return df
```

### 添加新输出格式

```python
# 1. 实现 IFactorWriter 接口
class HDF5Writer(IFactorWriter):
    @property
    def format_name(self):
        return "HDF5"
    
    def write(self, data, path, **kwargs):
        data.to_hdf(path, key="factors", mode="w")
        return path

# 2. 注册到工厂
FactorWriterFactory.register("hdf5", HDF5Writer)
```

---

## 📚 最佳实践

### 因子开发

1. **命名规范**: 使用小写蛇形命名 `return_1d`, `volatility_20d`
2. **参数化**: 将窗口期等参数化 `default_params = {"window": 20}`
3. **文档化**: 清晰的 docstring 和 meta 信息
4. **单元测试**: 为每个因子编写测试

### 数据处理

1. **验证优先**: 始终使用 `StandardDataProtocol` 包装原始数据
2. **缺失值处理**: 明确处理 NaN 的策略
3. **数据对齐**: 利用 Pandas 的自动对齐特性

### 性能优化

1. **批量计算**: 一次计算多个因子，而非多次调用
2. **缓存结果**: 对中间结果进行缓存
3. **监控性能**: 使用 `FactorComputeResult.compute_time`

---

## 🎯 总结

FactorHub 通过以下设计实现了专业的因子计算框架：

| 特性 | 实现方式 | 收益 |
|------|---------|------|
| 模块化 | SOLID 原则 | 易维护、易测试 |
| 可扩展 | 接口 + 注册表 | 添加功能无需改动核心代码 |
| 健壮性 | 分层异常 + 容错 | 单点故障不影响全局 |
| 灵活性 | 依赖注入 + 适配器 | 支持多种数据源和输出格式 |
| 高性能 | 向量化 + MultiIndex | 充分利用 Pandas 优化 |

**核心理念**: "Define Once, Use Anywhere"
- 定义一次标准协议
- 因子在任何数据源上都能运行
- 结果可以输出到任何格式
