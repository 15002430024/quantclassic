#!/usr/bin/env python
"""
QuantClassic Run - CLI入口

类似Qlib的qrun命令，一键运行配置文件

使用方式:
    python -m quantclassic.config.cli config.yaml
    或
    qcrun config.yaml
"""

import sys
import os
from pathlib import Path

# 添加父目录到Python路径，确保可以导入quantclassic
current_dir = Path(__file__).resolve().parent
quantclassic_root = current_dir.parent.parent  # 向上两级到jupyterlab目录
if str(quantclassic_root) not in sys.path:
    sys.path.insert(0, str(quantclassic_root))

from quantclassic.config.loader import ConfigLoader
from quantclassic.config.runner import TaskRunner


def main():
    """CLI主函数"""
    if len(sys.argv) < 2:
        print("使用方式: qcrun <config.yaml>")
        print("\n示例:")
        print("  qcrun config/templates/lstm_basic.yaml")
        print("  qcrun my_experiment.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # 检查配置文件是否存在
    if not Path(config_path).exists():
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)
    
    print(f"=== QuantClassic Run ===")
    print(f"配置文件: {config_path}\n")
    
    # 加载配置
    try:
        config = ConfigLoader.load(config_path)
        print(f"✅ 配置加载成功")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        sys.exit(1)
    
    # 验证配置
    try:
        ConfigLoader.validate(config)
        print(f"✅ 配置验证通过\n")
    except Exception as e:
        print(f"❌ 配置验证失败: {e}")
        sys.exit(1)
    
    # 获取实验名称
    experiment_name = config.get('experiment_name', Path(config_path).stem)
    
    # 运行任务
    runner = TaskRunner(
        log_level=config.get('quantclassic_init', {}).get('log_level', 'INFO')
    )
    
    try:
        results = runner.run(config, experiment_name=experiment_name)
        print(f"\n✅ 任务执行成功!")
        print(f"实验名称: {experiment_name}")
        
        if 'train_results' in results and results['train_results']:
            metrics = results['train_results'].get('metrics', {})
            if metrics:
                print(f"\n训练指标:")
                for k, v in metrics.items():
                    print(f"  {k}: {v}")
        
    except Exception as e:
        print(f"\n❌ 任务执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
