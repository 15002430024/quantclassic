"""
QCRecorder - QuantClassic全局Recorder接口

提供类似Qlib的R接口，用于简化实验管理：
    with R.start(experiment_name="my_exp"):
        R.log_params(learning_rate=0.001)
        R.log_metrics(loss=0.5)
        R.save_objects(model=my_model)
"""

from contextlib import contextmanager
from typing import Any, Dict, Optional
from .exp_manager import ExpManager
from .recorder import Recorder


class QCRecorder:
    """
    全局Recorder接口
    
    提供简化的API来管理实验和记录数据
    """
    
    def __init__(self, exp_dir: str = "output/experiments"):
        """
        初始化全局Recorder
        
        Args:
            exp_dir: 实验保存目录
        """
        self.exp_manager = ExpManager(exp_dir)
        self._current_recorder: Optional[Recorder] = None
    
    @contextmanager
    def start(self, 
              experiment_name: str,
              recorder_name: Optional[str] = None,
              resume: bool = False,
              **params):
        """
        启动一个recorder的上下文管理器
        
        Args:
            experiment_name: 实验名称
            recorder_name: recorder名称，如果为None则自动生成
            resume: 是否恢复已有的recorder
            **params: 初始参数
            
        Example:
            with R.start(experiment_name="test"):
                R.log_params(lr=0.001)
                R.log_metrics(loss=0.5)
        """
        # 启动recorder
        recorder_id = self.exp_manager.start_recorder(
            experiment_name=experiment_name,
            recorder_name=recorder_name,
            resume=resume
        )
        
        # 获取recorder实例
        self._current_recorder = self.exp_manager.get_recorder(
            experiment_name=experiment_name,
            recorder_id=recorder_id
        )
        
        # 记录初始参数
        if params:
            self._current_recorder.log_params(**params)
        
        try:
            yield self._current_recorder
        finally:
            # 自动结束recorder
            self.exp_manager.end_recorder(
                experiment_name=experiment_name,
                recorder_id=recorder_id,
                status="FINISHED"
            )
            self._current_recorder = None
    
    def log_params(self, **params) -> None:
        """
        记录参数
        
        Args:
            **params: 参数字典
        """
        if self._current_recorder is None:
            raise RuntimeError("请先使用 R.start() 启动recorder")
        self._current_recorder.log_params(**params)
    
    def log_metrics(self, step: Optional[int] = None, **metrics) -> None:
        """
        记录指标
        
        Args:
            step: 步数/epoch，如果为None则自动递增
            **metrics: 指标字典
        """
        if self._current_recorder is None:
            raise RuntimeError("请先使用 R.start() 启动recorder")
        self._current_recorder.log_metrics(step=step, **metrics)
    
    def save_objects(self, **objects) -> None:
        """
        保存对象
        
        Args:
            **objects: 对象字典，key为对象名，value为要保存的对象
        """
        if self._current_recorder is None:
            raise RuntimeError("请先使用 R.start() 启动recorder")
        self._current_recorder.save_objects(**objects)
    
    def load_object(self, 
                    experiment_name: str,
                    recorder_id: str,
                    object_name: str) -> Any:
        """
        加载对象
        
        Args:
            experiment_name: 实验名称
            recorder_id: recorder ID
            object_name: 对象名称
            
        Returns:
            加载的对象
        """
        recorder = self.exp_manager.get_recorder(
            experiment_name=experiment_name,
            recorder_id=recorder_id
        )
        return recorder.load_object(object_name)
    
    def list_experiments(self) -> Dict[str, Dict]:
        """
        列出所有实验
        
        Returns:
            实验信息字典
        """
        return self.exp_manager.list_experiments()
    
    def list_recorders(self, experiment_name: str) -> Dict[str, Dict]:
        """
        列出实验的所有recorders
        
        Args:
            experiment_name: 实验名称
            
        Returns:
            recorder信息字典
        """
        return self.exp_manager.list_recorders(experiment_name)
    
    def get_recorder(self, 
                     experiment_name: str,
                     recorder_id: str) -> Recorder:
        """
        获取recorder实例
        
        Args:
            experiment_name: 实验名称
            recorder_id: recorder ID
            
        Returns:
            Recorder实例
        """
        return self.exp_manager.get_recorder(
            experiment_name=experiment_name,
            recorder_id=recorder_id
        )
    
    def search_recorders(self,
                        experiment_name: Optional[str] = None,
                        status: Optional[str] = None,
                        **params) -> list:
        """
        搜索符合条件的recorders
        
        Args:
            experiment_name: 实验名称过滤
            status: 状态过滤
            **params: 参数过滤
            
        Returns:
            符合条件的recorder列表
        """
        return self.exp_manager.search_recorders(
            experiment_name=experiment_name,
            status=status,
            **params
        )
    
    @property
    def current_recorder(self) -> Optional[Recorder]:
        """获取当前活跃的recorder"""
        return self._current_recorder
    
    def generate_report(self, 
                       experiment_name: str,
                       recorder_id: str,
                       report_type: str = "summary",
                       save_path: Optional[str] = None) -> str:
        """
        生成实验报告
        
        Args:
            experiment_name: 实验名称
            recorder_id: recorder ID
            report_type: 报告类型 ("summary", "detailed", "comparison")
            save_path: 保存路径，如果为None则保存到recorder目录
            
        Returns:
            报告内容（字符串）
        """
        from datetime import datetime
        
        # 明确使用关键字参数，避免位置参数混淆
        recorder = self.exp_manager.get_recorder(
            experiment_id=None,
            experiment_name=experiment_name,
            recorder_id=recorder_id,
            recorder_name=None
        )
        
        if recorder is None:
            raise ValueError(f"无法找到 recorder: experiment_name={experiment_name}, recorder_id={recorder_id}")
        
        params = recorder.get_params()
        metrics = recorder.get_metrics()
        
        # 提取配置信息
        data_config = params.get('data_config', {})
        model_config = params.get('lstm_config', params.get('gru_config', {}))
        backtest_config = params.get('backtest_config', {})
        
        # 提取指标
        training_summary = {k: v for k, v in metrics.items() if 'train' in k or 'val' in k or 'window' in k}
        ic_stats = {k: v for k, v in metrics.items() if 'ic' in k.lower()}
        backtest_metrics = {k: v for k, v in metrics.items() 
                          if any(x in k for x in ['return', 'sharpe', 'drawdown', 'volatility', 'calmar', 'win_rate'])}
        
        # 生成报告
        if report_type == "summary":
            report = self._generate_summary_report(
                recorder, params, metrics, 
                data_config, model_config, backtest_config,
                training_summary, ic_stats, backtest_metrics
            )
        elif report_type == "detailed":
            report = self._generate_detailed_report(
                recorder, params, metrics,
                data_config, model_config, backtest_config,
                training_summary, ic_stats, backtest_metrics
            )
        else:
            report = f"Unsupported report type: {report_type}"
        
        # 保存报告
        if save_path is None:
            save_path = f"{recorder.recorder_dir}/EXPERIMENT_REPORT.txt"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report
    
    def _generate_summary_report(self, recorder, params, metrics,
                                data_config, model_config, backtest_config,
                                training_summary, ic_stats, backtest_metrics) -> str:
        """生成摘要报告"""
        from datetime import datetime
        
        # 提取关键指标
        model_type = model_config.get('model_type', 'LSTM/GRU')
        n_windows = training_summary.get('n_windows', 'N/A')
        avg_train_loss = training_summary.get('avg_train_loss', 0)
        avg_val_loss = training_summary.get('avg_val_loss', 0)
        
        ic_mean = ic_stats.get('ic_mean', 0)
        icir = ic_stats.get('icir', 0)
        ic_win_rate = ic_stats.get('ic_win_rate', 0)
        t_stat = ic_stats.get('t_stat', 0)
        
        annual_return = backtest_metrics.get('annual_return', 0)
        sharpe_ratio = backtest_metrics.get('sharpe_ratio', 0)
        max_drawdown = backtest_metrics.get('max_drawdown', 0)
        calmar_ratio = backtest_metrics.get('calmar_ratio', 0)
        
        report = f"""
{'=' * 80}
实验摘要报告
{'=' * 80}

【实验信息】
  实验名称: {recorder.experiment_id}
  Recorder: {recorder.name} ({recorder.id})
  创建时间: {recorder.start_time}
  状态: {recorder.status}

【模型配置】
  模型类型: {model_type}
  特征维度: {model_config.get('d_feat', 'N/A')}
  隐藏层: {model_config.get('hidden_size', 'N/A')} x {model_config.get('num_layers', 'N/A')}层
  训练轮数: {model_config.get('n_epochs', 'N/A')} (学习率: {model_config.get('learning_rate', 'N/A')})

【训练结果】
  窗口数量: {n_windows}
  平均训练损失: {avg_train_loss:.6f}
  平均验证损失: {avg_val_loss:.6f}

【因子效果】
  IC均值: {ic_mean:.4f}
  ICIR: {icir:.4f}
  IC胜率: {ic_win_rate:.2%}
  显著性: {'显著' if abs(t_stat) > 2 else '不显著'} (t={t_stat:.2f})

【回测表现】
  年化收益: {annual_return:.2%}
  夏普比率: {sharpe_ratio:.4f}
  最大回撤: {max_drawdown:.2%}
  卡玛比率: {calmar_ratio:.4f}

【保存路径】
  {recorder.recorder_dir}/

{'=' * 80}
"""
        return report
    
    def _generate_detailed_report(self, recorder, params, metrics,
                                 data_config, model_config, backtest_config,
                                 training_summary, ic_stats, backtest_metrics) -> str:
        """生成详细报告"""
        
        model_type = model_config.get('model_type', 'LSTM/GRU')
        
        report = f"""
{'=' * 80}
实验详细报告
{'=' * 80}

【实验信息】
  实验ID: {recorder.experiment_id}
  Recorder ID: {recorder.id}
  Recorder名称: {recorder.name}
  创建时间: {recorder.start_time}
  结束时间: {recorder.end_time}
  运行状态: {recorder.status}
  保存路径: {recorder.recorder_dir}

【数据配置】
  数据文件: {data_config.get('data_file', 'N/A')}
  滚动窗口: {data_config.get('rolling_window_size', 'N/A')}天
  滚动步长: {data_config.get('rolling_step', 'N/A')}天
  序列窗口: {data_config.get('window_size', 'N/A')}天
  批次大小: {data_config.get('batch_size', 'N/A')}
  数据划分: {data_config.get('split_strategy', 'N/A')}

【模型配置】
  模型类型: {model_type}
  特征维度: {model_config.get('d_feat', 'N/A')}
  隐藏单元: {model_config.get('hidden_size', 'N/A')}
  网络层数: {model_config.get('num_layers', 'N/A')}
  Dropout: {model_config.get('dropout', 'N/A')}
  训练轮数: {model_config.get('n_epochs', 'N/A')}
  学习率: {model_config.get('learning_rate', 'N/A')}
  早停轮数: {model_config.get('early_stop', 'N/A')}

【训练结果】
  窗口数量: {training_summary.get('n_windows', 'N/A')}
  平均训练损失: {training_summary.get('avg_train_loss', 0):.6f} ± {training_summary.get('std_train_loss', 0):.6f}
  平均验证损失: {training_summary.get('avg_val_loss', 0):.6f} ± {training_summary.get('std_val_loss', 0):.6f}
  平均最佳Epoch: {training_summary.get('avg_best_epoch', 0):.1f}

【因子效果分析】
  IC均值: {ic_stats.get('ic_mean', 0):.4f}
  IC标准差: {ic_stats.get('ic_std', 0):.4f}
  ICIR: {ic_stats.get('icir', 0):.4f}
  IC胜率: {ic_stats.get('ic_win_rate', 0):.2%}
  t统计量: {ic_stats.get('t_stat', 0):.4f}
  p值: {ic_stats.get('p_value', 0):.6f}
  显著性: {'显著 (p<0.05)' if ic_stats.get('p_value', 1) < 0.05 else '不显著'}

【回测配置】
  调仓频率: {backtest_config.get('rebalance_freq', 'N/A')}
  分组数量: {backtest_config.get('n_groups', 'N/A')}
  多空比例: 多{backtest_config.get('long_ratio', 0):.0%} / 空{backtest_config.get('short_ratio', 0):.0%}
  交易成本: 佣金{backtest_config.get('commission_rate', 0):.4f} + 印花税{backtest_config.get('stamp_tax_rate', 0):.4f}

【回测表现】
  年化收益: {backtest_metrics.get('annual_return', 0):.2%}
  年化波动: {backtest_metrics.get('annual_volatility', 0):.2%}
  夏普比率: {backtest_metrics.get('sharpe_ratio', 0):.4f}
  最大回撤: {backtest_metrics.get('max_drawdown', 0):.2%}
  卡玛比率: {backtest_metrics.get('calmar_ratio', 0):.4f}
  胜率: {backtest_metrics.get('win_rate', 0):.2%}
  年化超额: {backtest_metrics.get('annual_return', 0) - 0.08:.2%} (vs 8%基准)

【保存的对象】
"""
        # 列出保存的对象
        artifacts = recorder.list_artifacts()
        if artifacts:
            for artifact in artifacts:
                report += f"  ✓ {artifact}\n"
        else:
            report += "  (无保存对象)\n"
        
        report += f"""
【文件结构】
  {recorder.recorder_dir}/
  ├── meta.json              # 元数据
  ├── recorder.log           # 运行日志
  ├── artifacts/             # 保存的对象
  └── EXPERIMENT_REPORT.txt  # 本报告

{'=' * 80}
"""
        return report


# 创建全局单例
R = QCRecorder()


if __name__ == "__main__":
    # 示例1: 基本使用
    print("=== 示例1: 基本使用 ===")
    with R.start(experiment_name="demo1", lr=0.001, batch_size=32):
        R.log_metrics(step=1, loss=0.5, acc=0.8)
        R.log_metrics(step=2, loss=0.3, acc=0.85)
        R.save_objects(config={"model": "resnet"})
    
    # 示例2: 多次运行对比
    print("\n=== 示例2: 多次运行对比 ===")
    for lr in [0.001, 0.01, 0.1]:
        with R.start(experiment_name="lr_tuning", learning_rate=lr):
            for epoch in range(3):
                # 模拟训练
                loss = 1.0 / (epoch + 1) * (1 + lr * 10)
                R.log_metrics(epoch=epoch, loss=loss)
    
    # 查看所有实验
    print("\n=== 所有实验 ===")
    experiments = R.list_experiments()
    for exp_name, exp_info in experiments.items():
        print(f"实验: {exp_name}")
        print(f"  - 创建时间: {exp_info['create_time']}")
        print(f"  - Recorder数: {exp_info['recorder_count']}")
    
    # 查看lr_tuning的所有runs
    print("\n=== lr_tuning的所有runs ===")
    recorders = R.list_recorders("lr_tuning")
    for rec_id, rec_info in recorders.items():
        print(f"Recorder: {rec_id}")
        print(f"  - 参数: {rec_info.get('params', {})}")
        print(f"  - 状态: {rec_info.get('status')}")
    
    # 搜索最佳learning rate
    print("\n=== 搜索实验 ===")
    finished_runs = R.search_recorders(
        experiment_name="lr_tuning",
        status="FINISHED"
    )
    print(f"找到 {len(finished_runs)} 个已完成的runs")
    
    # 示例3: 加载保存的对象
    print("\n=== 示例3: 加载对象 ===")
    demo1_recorders = R.list_recorders("demo1")
    if demo1_recorders:
        first_recorder_id = list(demo1_recorders.keys())[0]
        config = R.load_object(
            experiment_name="demo1",
            recorder_id=first_recorder_id,
            object_name="config"
        )
        print(f"加载的config: {config}")
