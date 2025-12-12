# 预处理配置 - 快速参考卡片

## 一句话总结

🎯 **核心配置参数 - 记住这些就够了！**

### LabelGeneratorConfig（标签生成）

```python
config.label_config.enabled = True                    # 启用标签生成
config.label_config.base_price_col = 'close'          # 研报标准（T+1）
config.label_config.return_periods = [1, 5, 10]       # 生成周期
config.label_config.label_prefix = 'y_ret'            # 标签前缀
```

### NeutralizeConfig（中性化）

```python
config.neutralize_config.similarity_threshold = 0.7   # 相似度（0.6-0.8）
config.neutralize_config.lookback_window = 252        # 历史窗口（252=1年）
config.neutralize_config.correlation_method = 'pearson'  # 相关性方法
```

### ProcessMethod（处理方法）

```python
# 推荐顺序
ProcessMethod.GENERATE_LABELS              # 1️⃣ 生成标签
ProcessMethod.WINSORIZE                    # 2️⃣ 去极值
ProcessMethod.Z_SCORE                      # 3️⃣ 标准化
ProcessMethod.OLS_NEUTRALIZE               # 4️⃣ 特征中性化
ProcessMethod.SIMSTOCK_LABEL_NEUTRALIZE    # 5️⃣ 标签中性化
ProcessMethod.FILLNA_MEDIAN                # 6️⃣ 填充缺失
```

---

## 最常用的参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `base_price_col` | 'close' | **必选** - 研报标准 |
| `return_periods` | [1, 5, 10] | **常用** - 三个周期 |
| `label_prefix` | 'y_ret' | **推荐** - 区分标签 |
| `similarity_threshold` | 0.7 | **标准** - 平衡选择 |
| `lookback_window` | 252 | **标准** - 一年数据 |
| `normalize_mode` | 'cross_section' | **推荐** - 截面标准化 |

---

## 参数取值范围参考

### similarity_threshold（相似度）
```
0.5 ──────── 0.6 ──────── 0.7 ──────── 0.8 ──────── 0.9
宽松         较宽松      标准(推荐)   严格       非常严格
多数据       充足数据    平衡        严谨       极端
```

### lookback_window（历史窗口）
```
60天      120天     252天(推荐)   504天
3个月     6个月     1年          2年
短期      中期      标准         长期
```

### correlation_method（相关性）
```
pearson          spearman
线性相关          等级相关
对异常值敏感      对异常值不敏感
标准选择         鲁棒选择
```

---

## 常见场景配置

### 场景 1: 快速开始（最小配置）
```python
config = PreprocessConfig()
config.label_config.base_price_col = 'close'

config.add_step('生成标签', ProcessMethod.GENERATE_LABELS)
config.add_step('标准化', ProcessMethod.Z_SCORE)
config.add_step('填充缺失', ProcessMethod.FILLNA_MEDIAN)
```

### 场景 2: 标准配置（推荐）
```python
config = PreprocessConfig()

# 标签配置
config.label_config.base_price_col = 'close'
config.label_config.return_periods = [1, 5, 10]
config.label_config.label_prefix = 'y_ret'

# 中性化配置
config.neutralize_config.similarity_threshold = 0.7
config.neutralize_config.lookback_window = 252

# 处理步骤
config.add_step('生成标签', ProcessMethod.GENERATE_LABELS)
config.add_step('去极值', ProcessMethod.WINSORIZE, params={'limits': [0.025, 0.025]})
config.add_step('标准化', ProcessMethod.Z_SCORE, params={'normalize_mode': 'cross_section'})
config.add_step('中性化', ProcessMethod.OLS_NEUTRALIZE)
config.add_step('标签中性化', ProcessMethod.SIMSTOCK_LABEL_NEUTRALIZE)
config.add_step('填充缺失', ProcessMethod.FILLNA_MEDIAN)
```

### 场景 3: 严格配置（高要求）
```python
config = PreprocessConfig()

config.label_config.base_price_col = 'close'
config.neutralize_config.similarity_threshold = 0.8
config.neutralize_config.min_similar_stocks = 10
config.neutralize_config.correlation_method = 'spearman'

# ... 添加所有处理步骤 ...
```

### 场景 4: VWAP 配置（成交价）
```python
config = PreprocessConfig()

config.label_config.price_col = 'vwap'
config.label_config.base_price_col = 'vwap'
config.label_config.label_prefix = 'y_vwap_ret'

# ... 添加处理步骤 ...
```

---

## 参数速查表

### LabelGeneratorConfig

| 参数 | 默认值 | 可选值 | 推荐值 |
|------|--------|--------|--------|
| enabled | True | True/False | True |
| stock_col | 'order_book_id' | str | 保持不变 |
| time_col | 'trade_date' | str | 保持不变 |
| price_col | 'close' | 'close'/'vwap'/'open' | 'close' |
| **base_price_col** | None | None/'close'/'vwap' | **'close'** |
| label_type | 'return' | 'return'/'class' | 'return' |
| return_periods | [1, 5, 10] | 任意列表 | [1, 5, 10] |
| return_method | 'simple' | 'simple'/'log' | 'simple' |
| **label_prefix** | 'y_ret' | 任意字符串 | **'y_ret'** |

### NeutralizeConfig

| 参数 | 默认值 | 范围/可选值 | 推荐值 |
|------|--------|-----------|--------|
| industry_column | 'industry_name' | str | 保持不变 |
| market_cap_column | 'total_mv' | str | 保持不变 |
| min_samples | 10 | int | 5-20 |
| label_column | 'y_ret_1d' | str | 'y_ret_1d' |
| **similarity_threshold** | 0.7 | 0.0-1.0 | **0.7-0.8** |
| **lookback_window** | 252 | int | **252** |
| min_similar_stocks | 5 | int | 5-10 |
| correlation_method | 'pearson' | 'pearson'/'spearman' | 'pearson' |

### PreprocessConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| pipeline_steps | [] | 通过 add_step() 添加 |
| column_mapping | {} | 需要时配置 |
| groupby_columns | ['trade_date'] | 通常保持不变 |
| id_columns | ['order_book_id', 'trade_date'] | 通常保持不变 |
| label_config | LabelGeneratorConfig() | 配置标签生成 |
| neutralize_config | NeutralizeConfig() | 配置中性化 |
| save_intermediate | False | 调试时设为 True |
| verbose | True | 保持 True 查看日志 |

---

## 一分钟快速启动

```python
from quantclassic.data_processor.preprocess_config import PreprocessConfig, ProcessMethod
from quantclassic.data_processor.data_preprocessor import DataPreprocessor

# 创建配置
config = PreprocessConfig()

# 关键：配置标签生成
config.label_config.base_price_col = 'close'          # ⭐ 最重要！
config.label_config.return_periods = [1, 5, 10]       
config.label_config.label_prefix = 'y_ret'

# 添加处理步骤（按顺序）
config.add_step('生成标签', ProcessMethod.GENERATE_LABELS)
config.add_step('去极值', ProcessMethod.WINSORIZE, params={'limits': [0.025, 0.025]})
config.add_step('标准化', ProcessMethod.Z_SCORE, params={'normalize_mode': 'cross_section'})
config.add_step('填充缺失', ProcessMethod.FILLNA_MEDIAN)

# 执行预处理
processor = DataPreprocessor(config)
df_processed = processor.fit_transform(df_raw)

# 验证结果
print(df_processed.columns)  # 应包含 y_ret_1d, y_ret_5d, y_ret_10d
```

---

## ⚠️ 常见错误

### ❌ 错误 1: base_price_col 设为 None
```python
# 错误 - 使用传统标准（无法交易）
config.label_config.base_price_col = None

# ✅ 正确 - 使用研报标准（真实交易）
config.label_config.base_price_col = 'close'
```

### ❌ 错误 2: 标签生成不在第一步
```python
# 错误
config.add_step('去极值', ProcessMethod.WINSORIZE)
config.add_step('生成标签', ProcessMethod.GENERATE_LABELS)

# ✅ 正确
config.add_step('生成标签', ProcessMethod.GENERATE_LABELS)
config.add_step('去极值', ProcessMethod.WINSORIZE)
```

### ❌ 错误 3: 标签列名冲突
```python
# 错误 - 与特征名冲突
config.label_config.label_prefix = 'ret'  # 生成 ret_1d 与特征冲突！

# ✅ 正确 - 使用 y_ 前缀区分
config.label_config.label_prefix = 'y_ret'  # 生成 y_ret_1d
```

### ❌ 错误 4: 相似度阈值过高
```python
# 错误 - 找不到足够相似的股票
config.neutralize_config.similarity_threshold = 0.95

# ✅ 正确 - 平衡选择
config.neutralize_config.similarity_threshold = 0.7
```

---

## 🎓 学习资源

更详细的信息，请查看：

1. **完整文档**: `PREPROCESS_CONFIG_ARGS_GUIDE.md`
2. **标签生成指南**: `LABEL_GENERATION_CONFIG_GUIDE.md`
3. **研报标准**: `RESEARCH_STANDARD_LABEL.md`
4. **集成总结**: `LABEL_GENERATION_INTEGRATION_SUMMARY.md`
5. **源代码注释**: `data_processor/preprocess_config.py`

---

## 📞 快速问题解答

**Q: 该用哪些周期？**
A: [1, 5, 10] 最常用，也可根据需要调整

**Q: 相似度选多少？**
A: 0.7 是标准，0.8 更严格

**Q: 需要所有步骤都做吗？**
A: 最少需要：生成标签 → 标准化 → 填充缺失

**Q: 配置能保存吗？**
A: 能，用 `config.to_yaml('file.yaml')`

**Q: 如何禁用某个步骤？**
A: `enabled=False` 参数

---

**最后提醒：** 记住这个最关键的一个参数！
```python
config.label_config.base_price_col = 'close'  # ⭐ 使用研报标准（T+1 基准）
```
