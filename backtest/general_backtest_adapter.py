"""
general_backtest_adapter.py - GeneralBacktest 适配层

将 quantclassic 的因子/预测输出转换为 GeneralBacktest 所需的 weights_data 与 price_data，
并封装 GeneralBacktest 的调用、参数映射与结果绘图。

GeneralBacktest 已内嵌于 quantclassic.backtest.general_backtest 子模块中，无需外部安装。

Usage:
    from quantclassic.backtest.general_backtest_adapter import GeneralBacktestAdapter

    adapter = GeneralBacktestAdapter(config)
    results = adapter.run(
        factor_df=processed_df,
        price_df=price_df,
        factor_col='factor_raw_std',
    )
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest_config import BacktestConfig
from .portfolio_builder import PortfolioBuilder
from .general_backtest import GeneralBacktest

logger = logging.getLogger(__name__)


def is_general_backtest_available() -> bool:
    """检查 GeneralBacktest 是否可用（内嵌版始终可用）"""
    return True


# ---------------------------------------------------------------------------
# 数据适配工具函数
# ---------------------------------------------------------------------------

def prepare_price_data(
    price_df: pd.DataFrame,
    date_col: str = 'trade_date',
    code_col: str = 'order_book_id',
    adj_factor_col: str = 'adj_factor',
    open_col: str = 'open',
    close_col: str = 'close',
) -> pd.DataFrame:
    """
    将 quantclassic 格式的价格数据转换为 GeneralBacktest 所需格式。

    GeneralBacktest 需要:
        date (datetime64), code (str), open (float), close (float), adj_factor (float)

    Args:
        price_df: 价格数据 DataFrame
        date_col: 日期列名
        code_col: 股票代码列名
        adj_factor_col: 复权因子列名
        open_col: 开盘价列名
        close_col: 收盘价列名

    Returns:
        标准化后的价格 DataFrame
    """
    required_for_mapping = {date_col: 'date', code_col: 'code'}
    for col, target in required_for_mapping.items():
        if col not in price_df.columns:
            raise ValueError(f"价格数据缺少必要列 '{col}' (目标映射: '{target}')")

    out = pd.DataFrame()
    out['date'] = pd.to_datetime(price_df[date_col])
    out['code'] = price_df[code_col].astype(str)

    # close 必须存在
    if close_col not in price_df.columns:
        raise ValueError(f"价格数据缺少收盘价列 '{close_col}'")
    out['close'] = price_df[close_col].astype(float)

    # open: 缺失时用 close 填充
    if open_col in price_df.columns:
        out['open'] = price_df[open_col].astype(float)
    else:
        logger.warning(f"价格数据缺少开盘价列 '{open_col}'，使用收盘价填充")
        out['open'] = out['close']

    # adj_factor: 缺失时默认 1.0
    if adj_factor_col in price_df.columns:
        out['adj_factor'] = price_df[adj_factor_col].astype(float)
    else:
        logger.warning(f"价格数据缺少复权因子列 '{adj_factor_col}'，默认填充 1.0")
        out['adj_factor'] = 1.0

    return out


# ---------------------------------------------------------------------------
# GeneralBacktestAdapter
# ---------------------------------------------------------------------------

class GeneralBacktestAdapter:
    """
    GeneralBacktest 适配器

    封装 quantclassic → GeneralBacktest 的全部转换与调用逻辑，包括：
    - 权重生成（通过 PortfolioBuilder.generate_weights）
    - 价格数据格式转换
    - GeneralBacktest 实例化与运行
    - 结果收集与可视化保存
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.logger = logging.getLogger(__name__)
        self.portfolio_builder = PortfolioBuilder(self.config)
        self._bt_instance = None  # 保存最近一次 GeneralBacktest 实例

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def run(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_col: str = 'factor_raw_std',
        weight_mode: str = 'long_short',
        weights_df: Optional[pd.DataFrame] = None,
        benchmark_weights: Optional[pd.DataFrame] = None,
        benchmark_name: str = 'Benchmark',
        save_plots: bool = True,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        运行 GeneralBacktest 回测。

        Args:
            factor_df: 含因子的 DataFrame（需含 trade_date, stock_col, factor_col）
            price_df: 价格数据 DataFrame（需含 trade_date, stock_col, open, close, adj_factor）
            factor_col: 因子列名
            weight_mode: 权重模式 ('long_only', 'short_only', 'long_short', 'group')
            weights_df: 可选，直接提供权重表 [date, code, weight]，跳过权重生成
            benchmark_weights: 可选基准权重 [date, code, weight]
            benchmark_name: 基准名称
            save_plots: 是否保存图表
            output_dir: 输出目录（None 时使用 config.output_dir）

        Returns:
            回测结果字典：
                nav_series: 净值序列
                positions: 持仓明细
                trade_records: 交易记录
                metrics: 绩效指标
                bt_instance: GeneralBacktest 实例（可调用 plot_all 等）
        """
        self.logger.info("=" * 60)
        self.logger.info("🚀 开始 GeneralBacktest 回测")
        self.logger.info("=" * 60)

        # 1. 生成或使用已有权重
        if weights_df is not None:
            self.logger.info("使用外部提供的权重表")
            w_df = weights_df.copy()
            w_df['date'] = pd.to_datetime(w_df['date'])
        else:
            self.logger.info(f"通过 PortfolioBuilder 生成权重 (mode={weight_mode})")
            w_df = self.portfolio_builder.generate_weights(
                factor_df, factor_col=factor_col, mode=weight_mode
            )

        # 2. 输入校验
        self._validate_inputs(w_df, price_df)

        # 3. 准备价格数据
        code_col = 'order_book_id' if 'order_book_id' in price_df.columns else 'ts_code'
        gb_options = self.config.general_backtest_options
        p_df = prepare_price_data(
            price_df,
            date_col='trade_date',
            code_col=code_col,
            adj_factor_col=gb_options.get('adj_factor_col', 'adj_factor'),
            open_col='open',
            close_col='close',
        )

        # 4. 确定回测时间范围
        all_dates = sorted(set(w_df['date'].tolist() + p_df['date'].tolist()))
        start_date = str(min(all_dates).date())
        end_date = str(max(all_dates).date())

        # 5. 实例化并运行
        bt = GeneralBacktest(start_date=start_date, end_date=end_date)

        # 参数映射
        buy_cost = self.config.commission_rate if self.config.consider_cost else 0.0
        sell_cost = (self.config.commission_rate + self.config.stamp_tax_rate) if self.config.consider_cost else 0.0
        slippage = self.config.slippage_rate if self.config.consider_cost else 0.0

        run_kwargs = dict(
            weights_data=w_df,
            price_data=p_df,
            buy_price=self.config.buy_price,
            sell_price=self.config.sell_price,
            adj_factor_col='adj_factor',
            close_price_col='close',
            date_col='date',
            asset_col='code',
            weight_col='weight',
            rebalance_threshold=gb_options.get('rebalance_threshold', 0.005),
            transaction_cost=[buy_cost, sell_cost],
            initial_capital=gb_options.get('initial_capital', 1.0),
            slippage=slippage,
        )

        if benchmark_weights is not None:
            run_kwargs['benchmark_weights'] = benchmark_weights
            run_kwargs['benchmark_name'] = benchmark_name

        self.logger.info(f"  回测区间: {start_date} ~ {end_date}")
        self.logger.info(f"  权重行数: {len(w_df)}, 价格行数: {len(p_df)}")
        self.logger.info(f"  买入价: {self.config.buy_price}, 卖出价: {self.config.sell_price}")
        self.logger.info(f"  交易成本: buy={buy_cost:.4f}, sell={sell_cost:.4f}, slippage={slippage:.4f}")

        results = bt.run_backtest(**run_kwargs)
        self._bt_instance = bt

        # 6. 打印指标
        bt.print_metrics()

        # 7. 可视化
        out_dir = output_dir or os.path.join(self.config.output_dir, 'plots', 'general_backtest')
        if save_plots:
            self._save_plots(bt, out_dir)

        # 8. 组装返回
        output = {
            'nav_series': bt.daily_nav,
            'positions': bt.daily_positions,
            'trade_records': bt.trade_records,
            'metrics': bt.metrics,
            'weights_data': w_df,
            'bt_instance': bt,
        }

        self.logger.info("GeneralBacktest 回测完成 ✅")
        return output

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _validate_inputs(self, weights_df: pd.DataFrame, price_df: pd.DataFrame):
        """校验权重与价格数据的基本合规性"""
        if weights_df.empty:
            raise ValueError("输入权重表为空")

        price_dates = set(pd.to_datetime(price_df['trade_date']).dt.date) if 'trade_date' in price_df.columns \
            else set(pd.to_datetime(price_df['date']).dt.date) if 'date' in price_df.columns else set()
        weight_dates = set(weights_df['date'].dt.date)

        overlap = price_dates & weight_dates
        if len(overlap) == 0:
            raise ValueError(
                "价格数据日期与权重日期无交集。\n"
                f"  权重日期范围: {min(weight_dates)} ~ {max(weight_dates)}\n"
                f"  价格日期范围: {min(price_dates)} ~ {max(price_dates)}"
            )

    def _save_plots(self, bt, output_dir: str):
        """保存 GeneralBacktest 生成的图表"""
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"保存 GeneralBacktest 图表至 {output_dir}")

        plot_methods = [
            ('plot_all', 'dashboard.png'),
            ('plot_nav_curve', 'nav_curve.png'),
            ('plot_monthly_returns_heatmap', 'monthly_returns.png'),  # 修正方法名
        ]

        import matplotlib.pyplot as plt
        for method_name, filename in plot_methods:
            method = getattr(bt, method_name, None)
            if method is None:
                continue
            try:
                save_path = os.path.join(output_dir, filename)
                # 🔴 修复：直接传 save_path，让内部方法在 plt.show() 之前保存
                # 避免 plt.show() 清空 figure 后 savefig 保存空白图片
                try:
                    method(save_path=save_path)
                except TypeError:
                    # 如果方法不接受 save_path 参数，退回到先 savefig 再 show
                    plt.close('all')
                    method()
                    fig = plt.gcf()
                    fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
                plt.close('all')
                self.logger.info(f"  ✅ 已保存: {filename}")
            except Exception as e:
                self.logger.warning(f"  ⚠️ 保存 {method_name} 失败: {e}")
                plt.close('all')
