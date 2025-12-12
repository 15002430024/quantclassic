"""
Step 1 验证脚本 - 验证标准化协议与数据接入层

运行此脚本验证：
1. MockDataProvider 能正确生成数据
2. 生成的数据符合 StandardDataProtocol
3. 数据格式正确（columns、dtypes、index 结构）
"""

import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from quantclassic.factor_hub.providers.mock_provider import MockDataProvider
from quantclassic.factor_hub.protocols import StandardDataProtocol, DataValidationError


def main():
    print("=" * 70)
    print("Step 1 验证: 标准化协议与数据接入层")
    print("=" * 70)
    
    # 1. 实例化 MockDataProvider
    print("\n[1] 创建 MockDataProvider 实例...")
    provider = MockDataProvider(seed=42)  # 使用固定种子保证可复现
    print(f"    ✓ Provider: {provider}")
    print(f"    ✓ Name: {provider.name}")
    print(f"    ✓ Description: {provider.description}")
    print(f"    ✓ Is Available: {provider.is_available()}")
    
    # 2. 获取历史数据
    print("\n[2] 获取历史数据...")
    symbols = ["000001.SZ", "600000.SH", "600519.SH"]
    start = "2024-01-01"
    end = "2024-01-31"
    
    raw_data = provider.get_history(symbols, start, end)
    print(f"    ✓ 获取 {len(symbols)} 只股票的数据")
    print(f"    ✓ 时间范围: {start} ~ {end}")
    print(f"    ✓ 原始数据行数: {len(raw_data)}")
    
    # 3. 打印原始数据信息
    print("\n[3] 原始 DataFrame 信息:")
    print("-" * 50)
    print(raw_data.head(10))
    print("-" * 50)
    raw_data.info()
    
    # 4. 使用 StandardDataProtocol 包装数据
    print("\n[4] 使用 StandardDataProtocol 包装数据...")
    try:
        std_data = StandardDataProtocol(raw_data)
        print("    ✓ 数据校验通过!")
    except DataValidationError as e:
        print(f"    ✗ 数据校验失败: {e}")
        return False
    
    # 5. 打印标准化后的数据信息
    print("\n[5] StandardDataProtocol 数据信息:")
    std_data.info()
    
    # 6. 验证数据属性
    print("\n[6] 验证数据属性...")
    print(f"    ✓ 股票列表: {std_data.symbols}")
    print(f"    ✓ 起始日期: {std_data.start_date}")
    print(f"    ✓ 结束日期: {std_data.end_date}")
    print(f"    ✓ 日期数量: {len(std_data.datetimes)}")
    
    # 7. 验证数据访问
    print("\n[7] 验证数据访问接口...")
    print(f"    ✓ 收盘价 (close) 前5行:\n{std_data.close.head()}")
    print(f"\n    ✓ 获取单个股票数据 (000001.SZ):")
    print(std_data.get_symbol_data("000001.SZ").head(3))
    
    # 8. 验证透视表功能
    print("\n[8] 验证透视表功能...")
    close_pivot = std_data.to_pivot("close")
    print(f"    ✓ Close 透视表 shape: {close_pivot.shape}")
    print(close_pivot.head())
    
    print("\n" + "=" * 70)
    print("✓ Step 1 验证完成! 数据接入层工作正常。")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
