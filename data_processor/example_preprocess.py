"""
数据预处理模块使用示例
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

from data_processor import (
    DataPreprocessor,
    PreprocessConfig,
    PreprocessTemplates,
    ProcessingStep,
    ProcessMethod
)


def example_1_basic_preprocessing():
    """示例1: 基础预处理流程"""
    print("\n" + "="*60)
    print("示例1: 基础预处理流程")
    print("="*60)
    
    # 生成测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    stocks = [f'stock_{i:03d}' for i in range(50)]
    
    data = []
    for date in dates:
        for stock in stocks:
            data.append({
                'trade_date': date,
                'order_book_id': stock,
                'industry_name': f'industry_{np.random.randint(0, 5)}',
                'total_mv': np.random.lognormal(20, 2),
                'feature_1': np.random.randn() * 10 + 50,
                'feature_2': np.random.randn() * 5 + 20,
                'feature_3': np.random.randn() * 15 + 100
            })
    
    df = pd.DataFrame(data)
    print(f"原始数据: {df.shape}")
    print(df.head())
    
    # 使用基础模板
    config = PreprocessTemplates.basic_pipeline()
    preprocessor = DataPreprocessor(config)
    
    # 查看管道步骤
    print("\n管道步骤:")
    print(preprocessor.get_pipeline_summary())
    
    # 拟合并转换
    df_processed = preprocessor.fit_transform(
        df,
        feature_columns=['feature_1', 'feature_2', 'feature_3']
    )
    
    print(f"\n处理后数据: {df_processed.shape}")
    print(df_processed[['order_book_id', 'feature_1', 'feature_2', 'feature_3']].describe())
    
    # 保存预处理器
    preprocessor.save('output/basic_preprocessor.pkl')
    print("\n预处理器已保存")


def example_2_custom_pipeline():
    """示例2: 自定义处理流程"""
    print("\n" + "="*60)
    print("示例2: 自定义处理流程")
    print("="*60)
    
    # 自定义步骤
    custom_steps = [
        ProcessingStep(
            name="极值处理",
            method=ProcessMethod.WINSORIZE,
            features=[],  # 空列表表示所有特征
            params={'limits': (0.01, 0.01)}
        ),
        ProcessingStep(
            name="Z-score标准化",
            method=ProcessMethod.Z_SCORE,
            features=['feature_1', 'feature_2'],
            params={'clip_sigma': 3.0}
        ),
        ProcessingStep(
            name="秩归一化",
            method=ProcessMethod.RANK,
            features=['feature_3'],
            params={'output_range': (-1, 1)}
        ),
        ProcessingStep(
            name="填充缺失值",
            method=ProcessMethod.FILLNA_MEDIAN,
            features=[]
        )
    ]
    
    config = PreprocessConfig(pipeline_steps=custom_steps)
    preprocessor = DataPreprocessor(config)
    
    # 生成测试数据(带缺失值)
    np.random.seed(42)
    df = pd.DataFrame({
        'trade_date': pd.date_range('2023-01-01', periods=1000, freq='D').repeat(50),
        'order_book_id': ([f'stock_{i:03d}' for i in range(50)] * 1000),
        'feature_1': np.random.randn(50000) * 10 + 50,
        'feature_2': np.random.randn(50000) * 5 + 20,
        'feature_3': np.random.randn(50000) * 15 + 100
    })
    
    # 添加缺失值
    df.loc[df.sample(frac=0.05).index, 'feature_1'] = np.nan
    
    print(f"原始数据缺失率: {df['feature_1'].isnull().mean():.2%}")
    
    # 处理
    df_processed = preprocessor.fit_transform(df)
    
    print(f"处理后缺失率: {df_processed['feature_1'].isnull().mean():.2%}")
    print("\n处理后统计:")
    print(df_processed[['feature_1', 'feature_2', 'feature_3']].describe())


def example_3_industry_neutralize():
    """示例3: 行业市值中性化"""
    print("\n" + "="*60)
    print("示例3: 行业市值中性化")
    print("="*60)
    
    # 生成测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    data = []
    for date in dates:
        for i in range(100):
            industry = f'industry_{i % 5}'
            # 让特征与行业和市值相关
            mv = np.random.lognormal(20, 2)
            industry_effect = (i % 5) * 10
            
            data.append({
                'trade_date': date,
                'order_book_id': f'stock_{i:03d}',
                'industry_name': industry,
                'total_mv': mv,
                'feature_1': np.log(mv) * 5 + industry_effect + np.random.randn() * 3
            })
    
    df = pd.DataFrame(data)
    
    print("原始数据 - 特征与行业的相关性:")
    print(df.groupby('industry_name')['feature_1'].mean())
    
    # OLS中性化
    steps_ols = [
        ProcessingStep(
            name="OLS中性化",
            method=ProcessMethod.OLS_NEUTRALIZE,
            features=['feature_1'],
            params={
                'industry_column': 'industry_name',
                'market_cap_column': 'total_mv'
            }
        )
    ]
    
    config_ols = PreprocessConfig(pipeline_steps=steps_ols)
    preprocessor_ols = DataPreprocessor(config_ols)
    
    df_ols = preprocessor_ols.fit_transform(df, feature_columns=['feature_1'])
    
    print("\nOLS中性化后 - 特征与行业的相关性:")
    print(df_ols.groupby('industry_name')['feature_1'].mean())
    
    # 减均值版中性化
    steps_mean = [
        ProcessingStep(
            name="均值中性化",
            method=ProcessMethod.MEAN_NEUTRALIZE,
            features=['feature_1'],
            params={
                'industry_column': 'industry_name',
                'market_cap_column': 'total_mv',
                'n_quantiles': 3
            }
        )
    ]
    
    config_mean = PreprocessConfig(pipeline_steps=steps_mean)
    preprocessor_mean = DataPreprocessor(config_mean)
    
    df_mean = preprocessor_mean.fit_transform(df, feature_columns=['feature_1'])
    
    print("\n均值中性化后 - 特征与行业的相关性:")
    print(df_mean.groupby('industry_name')['feature_1'].mean())


def example_4_simstock_neutralize():
    """示例4: SimStock中性化"""
    print("\n" + "="*60)
    print("示例4: SimStock中性化")
    print("="*60)
    
    # 生成相关的收益率数据
    np.random.seed(42)
    n_stocks = 20
    n_days = 500
    
    # 创建相关的收益率
    base_returns = np.random.randn(n_days) * 0.02  # 市场收益
    
    data = []
    dates = pd.date_range('2022-01-01', periods=n_days, freq='D')
    
    for i in range(n_stocks):
        # 部分股票高度相关
        if i < 10:
            beta = 0.8 + np.random.rand() * 0.4
            returns = base_returns * beta + np.random.randn(n_days) * 0.01
        else:
            returns = np.random.randn(n_days) * 0.02
        
        for j, date in enumerate(dates):
            data.append({
                'trade_date': date,
                'order_book_id': f'stock_{i:03d}',
                'ret_1d': returns[j],
                'feature_1': returns[j] * 100 + np.random.randn() * 5
            })
    
    df = pd.DataFrame(data)
    
    print("原始特征标准差:")
    print(df.groupby('order_book_id')['feature_1'].std().describe())
    
    # SimStock中性化
    steps = [
        ProcessingStep(
            name="SimStock中性化",
            method=ProcessMethod.SIMSTOCK_NEUTRALIZE,
            features=['feature_1'],
            params={
                'similarity_threshold': 0.6,
                'lookback_window': 252,
                'min_similar_stocks': 3
            }
        )
    ]
    
    config = PreprocessConfig(pipeline_steps=steps)
    preprocessor = DataPreprocessor(config)
    
    df_processed = preprocessor.fit_transform(
        df,
        feature_columns=['feature_1'],
        target_column='ret_1d'
    )
    
    print("\nSimStock中性化后标准差:")
    print(df_processed.groupby('order_book_id')['feature_1'].std().describe())


def example_5_transform_mode():
    """示例5: fit_transform vs transform"""
    print("\n" + "="*60)
    print("示例5: 训练/推理模式")
    print("="*60)
    
    # 训练数据
    np.random.seed(42)
    df_train = pd.DataFrame({
        'trade_date': pd.date_range('2023-01-01', periods=5000, freq='D').repeat(50),
        'order_book_id': ([f'stock_{i:03d}' for i in range(50)] * 5000),
        'feature_1': np.random.randn(250000) * 10 + 50,
        'feature_2': np.random.randn(250000) * 5 + 20
    })
    
    # 测试数据(不同分布)
    df_test = pd.DataFrame({
        'trade_date': pd.date_range('2023-01-01', periods=1000, freq='D').repeat(50),
        'order_book_id': ([f'stock_{i:03d}' for i in range(50)] * 1000),
        'feature_1': np.random.randn(50000) * 12 + 55,
        'feature_2': np.random.randn(50000) * 6 + 22
    })
    
    print("训练数据统计:")
    print(df_train[['feature_1', 'feature_2']].describe())
    
    print("\n测试数据统计(原始):")
    print(df_test[['feature_1', 'feature_2']].describe())
    
    # 使用Z-score标准化
    config = PreprocessConfig(pipeline_steps=[
        ProcessingStep(
            name="标准化",
            method=ProcessMethod.Z_SCORE,
            features=[]
        )
    ])
    
    preprocessor = DataPreprocessor(config)
    
    # Fit on training data
    df_train_processed = preprocessor.fit_transform(df_train)
    
    print("\n训练数据处理后:")
    print(df_train_processed[['feature_1', 'feature_2']].describe())
    
    # Transform test data (使用训练集的均值和标准差)
    df_test_processed = preprocessor.transform(df_test)
    
    print("\n测试数据处理后(使用训练集参数):")
    print(df_test_processed[['feature_1', 'feature_2']].describe())
    
    # 保存和加载
    preprocessor.save('output/zscore_preprocessor.pkl')
    
    loaded_preprocessor = DataPreprocessor.load('output/zscore_preprocessor.pkl')
    print(f"\n加载的预处理器: {loaded_preprocessor}")


def example_6_advanced_pipeline():
    """示例6: 高级完整流程"""
    print("\n" + "="*60)
    print("示例6: 高级完整流程")
    print("="*60)
    
    # 使用高级模板
    config = PreprocessTemplates.advanced_pipeline()
    preprocessor = DataPreprocessor(config)
    
    # 生成完整测试数据
    np.random.seed(42)
    n_days = 252
    n_stocks = 100
    
    data = []
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    
    for date in dates:
        for i in range(n_stocks):
            data.append({
                'trade_date': date,
                'order_book_id': f'stock_{i:03d}',
                'industry_name': f'industry_{i % 10}',
                'total_mv': np.random.lognormal(20, 2),
                'pe_ratio': np.random.lognormal(2, 1),
                'pb_ratio': np.random.lognormal(0, 0.8),
                'ps_ratio': np.random.lognormal(1, 0.5),
                'momentum_20d': np.random.randn() * 0.1
            })
    
    df = pd.DataFrame(data)
    
    # 添加一些极端值
    df.loc[df.sample(frac=0.01).index, 'pe_ratio'] = df['pe_ratio'].max() * 10
    
    print("原始数据:")
    print(df[['pe_ratio', 'pb_ratio', 'momentum_20d']].describe())
    
    print("\n管道步骤:")
    print(preprocessor.get_pipeline_summary())
    
    # 处理
    df_processed = preprocessor.fit_transform(df)
    
    print("\n处理后数据:")
    print(df_processed[['pe_ratio', 'pb_ratio', 'momentum_20d']].describe())
    
    # 数据质量报告
    print("\n数据质量报告:")
    report = preprocessor.validate_data(df_processed)
    print(f"数据形状: {report['shape']}")
    print(f"零方差特征: {report['zero_std_features']}")


def example_7_alpha_research():
    """示例7: Alpha因子研究流程"""
    print("\n" + "="*60)
    print("示例7: Alpha因子研究流程")
    print("="*60)
    
    # 使用Alpha模板
    config = PreprocessTemplates.alpha_pipeline()
    preprocessor = DataPreprocessor(config)
    
    # 生成因子数据
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=500, freq='D')
    stocks = [f'stock_{i:03d}' for i in range(200)]
    
    data = []
    for date in dates:
        for stock in stocks:
            data.append({
                'trade_date': date,
                'order_book_id': stock,
                'industry_name': f'industry_{np.random.randint(0, 20)}',
                'total_mv': np.random.lognormal(20, 2),
                'ret_1d': np.random.randn() * 0.02,
                'alpha_factor_1': np.random.randn() * 5,
                'alpha_factor_2': np.random.randn() * 3,
                'alpha_factor_3': np.random.randn() * 8
            })
    
    df = pd.DataFrame(data)
    
    print(f"原始因子数据: {df.shape}")
    
    # 处理
    df_processed = preprocessor.fit_transform(
        df,
        feature_columns=['alpha_factor_1', 'alpha_factor_2', 'alpha_factor_3'],
        target_column='ret_1d'
    )
    
    print("\n处理后因子统计:")
    print(df_processed[['alpha_factor_1', 'alpha_factor_2', 'alpha_factor_3']].describe())
    
    # 查看各行业因子均值(应该接近0)
    print("\n各行业因子均值(中性化后):")
    print(df_processed.groupby('industry_name')[['alpha_factor_1', 'alpha_factor_2']].mean().head())


def example_8_incremental_update():
    """示例8: 增量更新"""
    print("\n" + "="*60)
    print("示例8: 增量更新场景")
    print("="*60)
    
    # 历史数据
    np.random.seed(42)
    df_historical = pd.DataFrame({
        'trade_date': pd.date_range('2023-01-01', periods=10000, freq='D').repeat(50),
        'order_book_id': ([f'stock_{i:03d}' for i in range(50)] * 10000),
        'feature_1': np.random.randn(500000) * 10 + 50
    })
    
    print("历史数据:", df_historical.shape)
    
    # 训练预处理器
    config = PreprocessConfig(pipeline_steps=[
        ProcessingStep(name="标准化", method=ProcessMethod.Z_SCORE, features=[])
    ])
    
    preprocessor = DataPreprocessor(config)
    df_historical_processed = preprocessor.fit_transform(df_historical)
    
    # 保存
    preprocessor.save('output/incremental_preprocessor.pkl')
    
    # --- 模拟新数据到达 ---
    print("\n新数据到达...")
    
    df_new = pd.DataFrame({
        'trade_date': pd.date_range('2023-01-01', periods=100, freq='D').repeat(50),
        'order_book_id': ([f'stock_{i:03d}' for i in range(50)] * 100),
        'feature_1': np.random.randn(5000) * 10 + 50
    })
    
    print("新数据:", df_new.shape)
    
    # 加载预处理器
    loaded_preprocessor = DataPreprocessor.load('output/incremental_preprocessor.pkl')
    
    # 使用transform处理新数据
    df_new_processed = loaded_preprocessor.transform(df_new)
    
    print("新数据处理完成:", df_new_processed.shape)
    print("\n新数据统计(使用历史参数):")
    print(df_new_processed['feature_1'].describe())


if __name__ == '__main__':
    # 创建输出目录
    Path('output').mkdir(exist_ok=True)
    
    # 运行示例
    example_1_basic_preprocessing()
    example_2_custom_pipeline()
    example_3_industry_neutralize()
    example_4_simstock_neutralize()
    example_5_transform_mode()
    example_6_advanced_pipeline()
    example_7_alpha_research()
    example_8_incremental_update()
    
    print("\n" + "="*60)
    print("所有示例运行完成!")
    print("="*60)
