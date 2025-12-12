"""
因子回测系统主控制器
协调所有组件，执行完整回测流程
"""

import os
import logging
import pandas as pd
import torch
from typing import Dict, Any, Optional
from datetime import datetime

from .backtest_config import BacktestConfig
from .factor_generator import FactorGenerator
from .factor_processor import FactorProcessor
from .portfolio_builder import PortfolioBuilder
from .ic_analyzer import ICAnalyzer
from .performance_evaluator import PerformanceEvaluator
from .result_visualizer import ResultVisualizer


class FactorBacktestSystem:
    """因子回测系统主控制器"""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        初始化回测系统
        
        Args:
            config: 回测配置（None时使用默认配置）
        """
        self.config = config if config is not None else BacktestConfig()
        
        # 设置日志
        self._setup_logging()
        
        # 初始化各组件
        self.factor_generator = None  # 需要模型才能初始化
        self.factor_processor = FactorProcessor(self.config)
        self.portfolio_builder = PortfolioBuilder(self.config)
        self.ic_analyzer = ICAnalyzer(self.config)
        self.performance_evaluator = PerformanceEvaluator(self.config)
        self.visualizer = ResultVisualizer(self.config)
        
        self.logger.info("因子回测系统初始化完成")
        self.logger.info(f"配置: {self.config.to_dict()}")
    
    def _setup_logging(self):
        """设置日志系统"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # 清除已有的处理器
        self.logger.handlers = []
        
        # 控制台处理器
        if self.config.console_log:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.config.log_level))
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 文件处理器
        if self.config.log_file:
            os.makedirs(os.path.dirname(self.config.log_file), exist_ok=True)
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setLevel(getattr(logging, self.config.log_level))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def load_model(self, model_path: str) -> torch.nn.Module:
        """
        加载训练好的模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            加载的模型
        """
        self.logger.info(f"加载模型: {model_path}")
        
        # 加载模型（需要根据实际模型结构调整）
        model = torch.load(model_path, map_location=self.config.device)
        model.eval()
        
        # 初始化因子生成器
        self.factor_generator = FactorGenerator(model, self.config)
        
        self.logger.info("模型加载完成")
        return model
    
    def run_backtest(self,
                    data_df: pd.DataFrame,
                    model: Optional[torch.nn.Module] = None,
                    factor_col: str = 'factor_raw_std',
                    return_col: str = 'y_true') -> Dict[str, Any]:
        """
        运行完整回测流程
        
        Args:
            data_df: 原始数据DataFrame
            model: 训练好的模型（None时使用已加载的模型）
            factor_col: 使用的因子列名
            return_col: 收益列名
            
        Returns:
            回测结果字典
        """
        self.logger.info("="*80)
        self.logger.info("开始因子回测")
        self.logger.info("="*80)
        
        results = {}
        
        # 1. 生成因子
        if model is not None:
            self.factor_generator = FactorGenerator(model, self.config)
        
        if self.factor_generator is None:
            raise ValueError("请先加载模型或提供模型参数")
        
        self.logger.info("\n步骤1: 生成因子")
        factor_df = self.factor_generator.generate_factors(
            data_df, 
            feature_cols=self.config.feature_cols
        )
        results['raw_factors'] = factor_df
        
        # 2. 处理因子
        self.logger.info("\n步骤2: 处理因子")
        processed_df = self.factor_processor.process(factor_df)
        
        # 添加收益列（如果原始数据中有）
        if return_col in data_df.columns:
            # 合并收益数据
            processed_df = self._merge_returns(processed_df, data_df, return_col)
        
        results['processed_factors'] = processed_df
        
        # 3. IC分析
        self.logger.info("\n步骤3: IC分析")
        ic_df = self.ic_analyzer.calculate_ic(processed_df, factor_col, return_col)
        ic_stats = self.ic_analyzer.analyze_ic_statistics(ic_df)
        
        results['ic_df'] = ic_df
        results['ic_stats'] = ic_stats
        
        self._print_ic_stats(ic_stats)
        
        # 4. 构建组合
        self.logger.info("\n步骤4: 构建投资组合")
        portfolios = self.portfolio_builder.build_portfolios(
            processed_df, factor_col, return_col
        )
        results['portfolios'] = portfolios
        
        # 5. 绩效评估
        self.logger.info("\n步骤5: 绩效评估")
        performance_metrics = {}
        
        for portfolio_name, portfolio_df in portfolios.items():
            if portfolio_name == 'groups':
                continue
            
            self.logger.info(f"\n评估 {portfolio_name} 组合:")
            metrics = self.performance_evaluator.evaluate_portfolio(portfolio_df)
            performance_metrics[portfolio_name] = metrics
            self._print_performance_metrics(metrics)
        
        results['performance_metrics'] = performance_metrics
        
        # 6. 生成图表
        self.logger.info("\n步骤6: 生成可视化图表")
        if self.config.save_plots:
            output_dir = os.path.join(self.config.output_dir, 'plots')
            self.visualizer.create_comprehensive_report(
                portfolios, ic_df, performance_metrics, output_dir
            )
        
        # 7. 保存结果
        self.logger.info("\n步骤7: 保存结果")
        self._save_results(results)
        
        self.logger.info("\n"+"="*80)
        self.logger.info("回测完成!")
        self.logger.info("="*80)
        
        return results
    
    def _merge_returns(self, 
                      factor_df: pd.DataFrame,
                      data_df: pd.DataFrame,
                      return_col: str) -> pd.DataFrame:
        """合并收益数据到因子DataFrame"""
        # 确保两个DataFrame都有必要的列
        if 'ts_code' not in factor_df.columns or 'trade_date' not in factor_df.columns:
            self.logger.warning("因子DataFrame缺少ts_code或trade_date列")
            return factor_df
        
        # 提取收益数据
        return_data = data_df[['ts_code', 'trade_date', return_col]].copy()
        
        # 合并
        merged_df = pd.merge(
            factor_df,
            return_data,
            on=['ts_code', 'trade_date'],
            how='left'
        )
        
        return merged_df
    
    def _print_ic_stats(self, ic_stats: Dict[str, float]):
        """打印IC统计信息"""
        self.logger.info("\nIC统计指标:")
        self.logger.info(f"  IC均值: {ic_stats['ic_mean']:.4f}")
        self.logger.info(f"  Rank IC均值: {ic_stats['rank_ic_mean']:.4f}")
        self.logger.info(f"  IC标准差: {ic_stats['ic_std']:.4f}")
        self.logger.info(f"  ICIR: {ic_stats['icir']:.4f}")
        self.logger.info(f"  Rank ICIR: {ic_stats['rank_icir']:.4f}")
        self.logger.info(f"  IC胜率: {ic_stats['ic_win_rate']:.2%}")
        self.logger.info(f"  绝对IC均值: {ic_stats['abs_ic_mean']:.4f}")
        self.logger.info(f"  t统计量: {ic_stats['t_stat']:.4f}")
    
    def _print_performance_metrics(self, metrics: Dict[str, float]):
        """打印绩效指标"""
        self.logger.info(f"  累计收益: {metrics['total_return']:.2%}")
        self.logger.info(f"  年化收益: {metrics['annual_return']:.2%}")
        self.logger.info(f"  年化波动率: {metrics['annual_volatility']:.2%}")
        self.logger.info(f"  夏普比率: {metrics['sharpe_ratio']:.4f}")
        self.logger.info(f"  最大回撤: {metrics['max_drawdown']:.2%}")
        self.logger.info(f"  卡玛比率: {metrics['calmar_ratio']:.4f}")
        self.logger.info(f"  胜率: {metrics['win_rate']:.2%}")
    
    def _save_results(self, results: Dict[str, Any]):
        """保存回测结果"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 保存因子
        if 'processed_factors' in results:
            factor_path = os.path.join(self.config.output_dir, 'factors.csv')
            results['processed_factors'].to_csv(factor_path, index=False)
            self.logger.info(f"  因子已保存: {factor_path}")
        
        # 保存IC
        if 'ic_df' in results:
            ic_path = os.path.join(self.config.output_dir, 'ic_analysis.csv')
            results['ic_df'].to_csv(ic_path, index=False)
            self.logger.info(f"  IC分析已保存: {ic_path}")
        
        # 保存组合收益
        if 'portfolios' in results:
            for name, portfolio_df in results['portfolios'].items():
                if name != 'groups':
                    portfolio_path = os.path.join(
                        self.config.output_dir, f'portfolio_{name}.csv'
                    )
                    portfolio_df.to_csv(portfolio_path, index=False)
                    self.logger.info(f"  {name}组合已保存: {portfolio_path}")
        
        # 保存绩效指标
        if 'performance_metrics' in results and self.config.generate_excel:
            metrics_df = pd.DataFrame(results['performance_metrics']).T
            metrics_path = os.path.join(self.config.output_dir, 'performance_metrics.xlsx')
            metrics_df.to_excel(metrics_path)
            self.logger.info(f"  绩效指标已保存: {metrics_path}")
    
    def quick_test(self, 
                  data_df: pd.DataFrame,
                  model: torch.nn.Module) -> Dict[str, Any]:
        """
        快速测试（使用简化配置）
        
        Args:
            data_df: 数据DataFrame
            model: 模型
            
        Returns:
            回测结果
        """
        # 临时修改配置
        original_config = self.config
        from .backtest_config import ConfigTemplates
        self.config = ConfigTemplates.fast_test()
        
        # 重新初始化组件
        self.__init__(self.config)
        
        # 运行回测
        results = self.run_backtest(data_df, model)
        
        # 恢复配置
        self.config = original_config
        
        return results
