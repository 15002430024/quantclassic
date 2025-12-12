#!/usr/bin/env python3
"""
FactorHub - 端到端演示脚本

该脚本模拟用户调用过程：
1. 用户指定股票池和时间
2. 用户选择因子列表
3. 系统自动拉取数据 -> 计算 -> 保存文件

Usage:
    python main.py
    
    或者作为模块运行:
    python -m quantclassic.factor_hub.main
"""

import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from datetime import datetime
from typing import List

import pandas as pd

# 导入 FactorHub 组件
from quantclassic.factor_hub.factors import demo_factors  # noqa: 触发因子注册
from quantclassic.factor_hub.providers.mock_provider import MockDataProvider
from quantclassic.factor_hub.engine.factor_engine import FactorEngine
from quantclassic.factor_hub.io.writers import CSVWriter, ParquetWriter, FactorWriterFactory


def run_factor_pipeline(
    symbols: List[str],
    factor_names: List[str],
    start: str,
    end: str,
    output_dir: str = "./output",
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    运行因子计算流水线
    
    Args:
        symbols: 股票代码列表
        factor_names: 因子名称列表
        start: 起始日期 (YYYY-MM-DD)
        end: 结束日期 (YYYY-MM-DD)
        output_dir: 输出目录
        output_format: 输出格式 (csv/parquet)
        
    Returns:
        pd.DataFrame: 因子计算结果
    """
    print("\n" + "=" * 70)
    print("         FactorHub - 因子计算框架 v1.0.0")
    print("=" * 70)
    
    # 1. 配置参数显示
    print("\n📋 配置参数:")
    print(f"    股票池: {symbols}")
    print(f"    因子列表: {factor_names}")
    print(f"    时间范围: {start} ~ {end}")
    print(f"    输出目录: {output_dir}")
    print(f"    输出格式: {output_format}")
    
    # 2. 初始化数据提供者
    print("\n📊 初始化数据提供者...")
    provider = MockDataProvider(seed=2024)  # 使用固定种子保证可复现
    print(f"    ✓ 使用 {provider.name}")
    
    # 3. 初始化因子计算引擎
    print("\n⚙️ 初始化因子计算引擎...")
    engine = FactorEngine(provider, continue_on_error=True, verbose=False)
    print(f"    ✓ 可用因子: {engine.list_available_factors()}")
    
    # 4. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 5. 运行因子计算
    print("\n🚀 开始因子计算...")
    print("-" * 60)
    
    result = engine.run(
        symbols=symbols,
        factor_names=factor_names,
        start=start,
        end=end,
        factor_params={
            "volatility": {"window": 20},
            "return_5d": {"period": 5},
            "turnover_ratio": {"window": 20},
        }
    )
    
    print("-" * 60)
    
    # 6. 显示计算结果
    print("\n📈 计算结果:")
    print(f"    成功因子: {result.successful_factors}")
    print(f"    失败因子: {result.failed_factors}")
    print(f"    结果形状: {result.factors_data.shape}")
    print(f"    总耗时: {result.total_time:.2f}s")
    
    if not result.factors_data.empty:
        print("\n    因子数据预览 (前10行):")
        print(result.factors_data.head(10))
        
        # 因子统计
        print("\n    因子统计信息:")
        print(result.factors_data.describe())
    
    # 7. 保存结果
    print("\n💾 保存结果...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存 CSV
    csv_path = os.path.join(output_dir, f"factors_{timestamp}.csv")
    csv_writer = CSVWriter()
    csv_writer.write(result.factors_data, csv_path)
    print(f"    ✓ CSV 文件已保存: {csv_path}")
    
    # 保存 Parquet
    parquet_path = os.path.join(output_dir, f"factors_{timestamp}.parquet")
    parquet_writer = ParquetWriter(compression="snappy")
    parquet_writer.write(result.factors_data, parquet_path)
    print(f"    ✓ Parquet 文件已保存: {parquet_path}")
    
    # 8. 验证保存的文件
    print("\n✅ 验证保存的文件...")
    
    # 读取并显示 CSV 文件信息
    df_csv = pd.read_csv(csv_path, index_col=[0, 1], parse_dates=True)
    print(f"    CSV 文件:")
    print(f"        - 大小: {os.path.getsize(csv_path) / 1024:.2f} KB")
    print(f"        - 行数: {len(df_csv)}")
    print(f"        - 列数: {len(df_csv.columns)}")
    
    # 读取并显示 Parquet 文件信息
    df_parquet = pd.read_parquet(parquet_path)
    print(f"    Parquet 文件:")
    print(f"        - 大小: {os.path.getsize(parquet_path) / 1024:.2f} KB")
    print(f"        - 行数: {len(df_parquet)}")
    print(f"        - 列数: {len(df_parquet.columns)}")
    
    print("\n" + "=" * 70)
    print("         ✓ 端到端流程完成!")
    print("=" * 70)
    
    return result.factors_data


def main():
    """主函数 - 演示完整的因子计算流程"""
    
    # 用户配置
    SYMBOLS = [
        "000001.SZ",  # 平安银行
        "000002.SZ",  # 万科A
        "600000.SH",  # 浦发银行
        "600519.SH",  # 贵州茅台
        "000858.SZ",  # 五粮液
    ]
    
    FACTOR_NAMES = [
        "return_1d",      # 1日收益率
        "return_5d",      # 5日收益率
        "volatility",     # 波动率
        "turnover_ratio", # 换手率
        "price_range",    # 价格振幅
    ]
    
    START_DATE = "2024-01-01"
    END_DATE = "2024-03-31"
    OUTPUT_DIR = "./quantclassic/factor_hub/output"
    
    # 运行流水线
    factors_df = run_factor_pipeline(
        symbols=SYMBOLS,
        factor_names=FACTOR_NAMES,
        start=START_DATE,
        end=END_DATE,
        output_dir=OUTPUT_DIR,
    )
    
    return factors_df


if __name__ == "__main__":
    main()
