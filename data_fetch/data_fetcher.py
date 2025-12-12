"""
数据获取器模块
负责从米筐API获取原始数据
"""
import time
import logging
from typing import List, Dict, Optional, Tuple
from functools import wraps
import pandas as pd
from tqdm import tqdm

try:
    import rqdatac
except ImportError:
    rqdatac = None

from .config_manager import ConfigManager
from .resume_manager import ResumeManager


logger = logging.getLogger(__name__)


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"{func.__name__} 失败(已重试{max_retries}次): {e}")
                        raise
                    logger.warning(f"{func.__name__} 失败, 第{attempt + 1}次重试... 错误: {e}")
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator


class DataFetcher:
    """数据获取器"""
    
    def __init__(self, config: ConfigManager, resume_manager: Optional[ResumeManager] = None):
        """
        初始化数据获取器
        
        Args:
            config: 配置管理器实例
        """
        self.config = config
        self.resume_manager = resume_manager
        self._init_api()
    
    def _init_api(self):
        """初始化API连接"""
        if self.config.data_source.source == 'rq':
            if rqdatac is None:
                raise ImportError("未安装rqdatac, 请运行: pip install rqdatac")
            
            try:
                auth_cfg = self.config.data_source
                if auth_cfg.auth_method == 'account':
                    rqdatac.init(auth_cfg.username, auth_cfg.password)
                elif auth_cfg.auth_method == 'token':
                    rqdatac.init(token=auth_cfg.token)
                elif auth_cfg.auth_method == 'config_file':
                    if auth_cfg.config_path:
                        rqdatac.init(config_file=auth_cfg.config_path)
                    else:
                        rqdatac.init()
                else:
                    raise ValueError(
                        "auth_method 仅支持 account/token/config_file"
                    )
                logger.info("米筐 API 初始化成功!")
                logger.info(f"API信息: {rqdatac.info()}")
            except Exception as e:
                logger.error(f"米筐 API 初始化失败: {e}")
                raise
        else:
            raise ValueError(f"暂不支持数据源: {self.config.data_source.source}")
    
    @retry_on_failure(max_retries=3)
    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表
        
        Returns:
            包含股票基本信息的DataFrame
        """
        logger.info(f"=== 获取股票列表: {self.config.universe.universe_type} ===")
        
        universe_type = self.config.universe.universe_type
        end_date = self.config.time.end_date
        
        # 根据股票池类型获取股票列表
        if universe_type == 'custom':
            # 自定义股票池
            df_stocks = self._get_custom_stocks()
        elif universe_type in ['csi800', 'csi300', 'csi500']:
            # 指数成分股
            df_stocks = self._get_index_components(universe_type, end_date)
        elif universe_type == 'all_a':
            # 全部A股
            df_stocks = self._get_all_a_stocks(end_date)
        else:
            raise ValueError(f"不支持的股票池类型: {universe_type}")
        
        # 应用筛选规则
        df_stocks = self._apply_filters(df_stocks)
        
        logger.info(f"最终股票池: {len(df_stocks)} 只股票")
        logger.info(f"股票市场分布:\n{df_stocks['order_book_id'].str[-5:].value_counts()}")
        
        return df_stocks
    
    def _get_index_components(self, index_type: str, date: str) -> pd.DataFrame:
        """获取指数成分股"""
        index_code_map = {
            'csi800': '000906.XSHG',  # 中证800
            'csi300': '000300.XSHG',  # 沪深300
            'csi500': '000905.XSHG',  # 中证500
        }
        
        index_code = index_code_map.get(index_type)
        if not index_code:
            raise ValueError(f"未知的指数类型: {index_type}")
        
        try:
            # 获取指数成分股
            component_stocks = rqdatac.index_components(index_code, date=date)
            
            if component_stocks is None or len(component_stocks) == 0:
                logger.warning(f"未能从{index_type}获取成分股, 使用备选方案")
                return self._get_all_a_stocks(date)
            
            logger.info(f"从{index_type}获取到 {len(component_stocks)} 只成分股")
            
            # 获取股票详细信息
            all_instruments = rqdatac.all_instruments(type='CS', date=date)
            df_stocks = all_instruments[all_instruments['order_book_id'].isin(component_stocks)].copy()
            
            return df_stocks
        
        except Exception as e:
            logger.error(f"获取指数成分股失败: {e}")
            logger.info("使用备选方案: 获取全部A股")
            return self._get_all_a_stocks(date)
    
    def _get_all_a_stocks(self, date: str) -> pd.DataFrame:
        """获取全部A股"""
        all_instruments = rqdatac.all_instruments(type='CS', date=date)
        logger.info(f"获取到 {len(all_instruments)} 只A股股票")
        
        # 筛选沪深股票
        df_stocks = all_instruments[
            (all_instruments['order_book_id'].str.endswith('.XSHG')) | 
            (all_instruments['order_book_id'].str.endswith('.XSHE'))
        ].copy()
        
        return df_stocks
    
    def _get_custom_stocks(self) -> pd.DataFrame:
        """获取自定义股票池"""
        custom_stocks = self.config.universe.custom_stocks
        if not custom_stocks:
            raise ValueError("自定义股票池为空")
        
        all_instruments = rqdatac.all_instruments(type='CS', date=self.config.time.end_date)
        df_stocks = all_instruments[all_instruments['order_book_id'].isin(custom_stocks)].copy()
        
        logger.info(f"自定义股票池: {len(df_stocks)} 只股票")
        return df_stocks
    
    def _apply_filters(self, df_stocks: pd.DataFrame) -> pd.DataFrame:
        """应用筛选规则"""
        original_count = len(df_stocks)
        
        # 排除ST股票
        if self.config.universe.exclude_st:
            df_stocks = df_stocks[~df_stocks['symbol'].str.contains('ST', na=False)]
            logger.info(f"排除ST股票: {original_count} -> {len(df_stocks)}")
        
        # 筛选上市时间
        df_stocks['listed_date'] = pd.to_datetime(df_stocks['listed_date'])
        df_stocks = df_stocks[df_stocks['listed_date'] <= pd.to_datetime(self.config.time.end_date)]
        
        return df_stocks
    
    @retry_on_failure(max_retries=3)
    def get_trading_calendar(self) -> pd.DataFrame:
        """
        获取交易日历
        
        Returns:
            交易日历DataFrame
        """
        logger.info("=== 获取交易日历 ===")
        
        trading_dates = rqdatac.get_trading_dates(
            self.config.time.start_date,
            self.config.time.end_date,
            market=self.config.data_source.market
        )
        
        df_calendar = pd.DataFrame({
            'trade_date': trading_dates,
            'is_trading_day': 1
        })
        
        logger.info(f"获取到 {len(df_calendar)} 个交易日")
        return df_calendar
    
    @retry_on_failure(max_retries=3)
    def get_industry_data(self, order_book_ids: List[str]) -> pd.DataFrame:
        """
        获取行业分类数据
        
        Args:
            order_book_ids: 股票代码列表
            
        Returns:
            行业分类DataFrame
        """
        logger.info("=== 获取行业分类 ===")
        
        try:
            industry_data = rqdatac.get_instrument_industry(
                order_book_ids=order_book_ids,
                source=self.config.fields.industry_source,
                level=self.config.fields.industry_level,
                date=None
            )
            
            if industry_data is not None and not industry_data.empty:
                df_industry = industry_data.reset_index()
                df_industry.columns = ['order_book_id', 'industry_code', 'industry_name']
                
                logger.info(f"成功获取 {len(df_industry)} 只股票的行业分类")
                logger.info(f"\n行业分布:\n{df_industry['industry_name'].value_counts()}")
                
                return df_industry
            else:
                logger.warning("未获取到行业分类数据")
                return pd.DataFrame(columns=['order_book_id', 'industry_code', 'industry_name'])
        
        except Exception as e:
            logger.error(f"获取行业分类失败: {e}")
            return pd.DataFrame(columns=['order_book_id', 'industry_code', 'industry_name'])
    
    def _merge_with_existing(
        self,
        existing_df: Optional[pd.DataFrame],
        new_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        frames = []
        if existing_df is not None and not existing_df.empty:
            frames.append(existing_df)
        if new_df is not None and not new_df.empty:
            frames.append(new_df)
        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, ignore_index=True)
        if all(col in merged.columns for col in ['order_book_id', 'trade_date']):
            merged['trade_date'] = pd.to_datetime(merged['trade_date'], errors='coerce')
            if merged['trade_date'].isna().any():
                logger.warning(
                    "合并数据时发现 %d 条 trade_date 无法解析，已忽略",
                    merged['trade_date'].isna().sum()
                )
                merged = merged.dropna(subset=['trade_date'])
            merged = merged.drop_duplicates(['order_book_id', 'trade_date'], keep='last')
            merged = merged.sort_values(['order_book_id', 'trade_date']).reset_index(drop=True)
        else:
            merged = merged.drop_duplicates().reset_index(drop=True)
        return merged

    def _register_existing(self, entity: str, df: Optional[pd.DataFrame]):
        if df is None or df.empty or self.resume_manager is None:
            return
        if 'order_book_id' not in df.columns:
            return
        start_value = None
        end_value = None
        if 'trade_date' in df.columns:
            trade_dates = pd.to_datetime(df['trade_date'], errors='coerce')
            valid_dates = trade_dates.dropna()
            if not valid_dates.empty:
                start_value = str(valid_dates.min().date())
                end_value = str(valid_dates.max().date())
        self.resume_manager.record_existing(
            entity,
            df['order_book_id'].unique(),
            start_date=start_value,
            end_date=end_value
        )

    def get_price_data(
        self,
        order_book_ids: List[str],
        existing_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        获取价格行情数据(分批处理)
        
        Args:
            order_book_ids: 股票代码列表
            
        Returns:
            价格数据DataFrame
        """
        logger.info("=== 获取日频行情数据 ===")
        self._register_existing('price', existing_df)
        pending_ids = order_book_ids
        if self.resume_manager is not None:
            pending_ids = self.resume_manager.filter_pending('price', pending_ids)
            if not pending_ids:
                logger.info("断点续传: 行情数据已覆盖所有股票, 跳过API调用")
                return existing_df.copy() if existing_df is not None else pd.DataFrame()
        logger.info(f"字段: {self.config.fields.price_fields}")
        resolved_fields, include_vwap = self._resolve_price_fields()
        logger.info(f"API字段映射: {resolved_fields}，VWAP: {'是' if include_vwap else '否'}")
        self._resolved_price_fields = resolved_fields
        self._need_vwap = include_vwap
        
        df_new = self._batch_fetch_data(
            order_book_ids=pending_ids,
            fetch_func=self._fetch_price_batch,
            desc="获取行情数据",
            entity_name='price'
        )
        return self._merge_with_existing(existing_df, df_new)
    
    def _normalize_index_columns(
        self,
        df: Optional[pd.DataFrame],
        date_column: str = 'trade_date'
    ) -> Optional[pd.DataFrame]:
        """标准化 reset_index 后的索引列."""
        if df is None or df.empty:
            return df
        df_reset = df.reset_index()
        rename_map = {}
        for col in df_reset.columns[:2]:
            col_lower = str(col).lower()
            if any(key in col_lower for key in ['order_book_id', 'orderbookid', 'instrument']):
                rename_map[col] = 'order_book_id'
            elif col_lower in {'date', 'datetime', 'tradingdate', 'trade_date', 'trading_date'}:
                rename_map[col] = date_column
        df_reset = df_reset.rename(columns=rename_map)

        expected_order = [date_column, 'order_book_id']
        ordered_cols = [col for col in expected_order if col in df_reset.columns]
        remaining_cols = [col for col in df_reset.columns if col not in ordered_cols]
        if ordered_cols:
            df_reset = df_reset[ordered_cols + remaining_cols]

        if date_column not in df_reset.columns or 'order_book_id' not in df_reset.columns:
            logger.warning(
                "索引列标准化后缺少 trade_date 或 order_book_id, 请检查数据源返回格式"
            )

        return df_reset

    def _fetch_price_batch(self, batch_stocks: List[str]) -> Optional[pd.DataFrame]:
        """获取单批次价格数据"""
        price_fields = getattr(self, '_resolved_price_fields', None)
        include_vwap = getattr(self, '_need_vwap', False)
        if not price_fields:
            price_fields, include_vwap = self._resolve_price_fields()
            self._resolved_price_fields = price_fields
            self._need_vwap = include_vwap
        
        df_price = rqdatac.get_price(
            order_book_ids=batch_stocks,
            start_date=self.config.time.start_date,
            end_date=self.config.time.end_date,
            frequency=self.config.time.frequency,
            fields=price_fields,
            adjust_type='post',  # 后复权
            expect_df=True
        )
        
        if df_price is not None and not df_price.empty:
            df_price = self._normalize_index_columns(df_price)
            
            # 重命名列
            rename_dict = {
                'volume': 'vol',
                'total_turnover': 'amount'
            }
            df_price = df_price.rename(columns=rename_dict)
            
            # 获取 VWAP 数据（如果配置启用）
            if include_vwap:
                try:
                    df_vwap = rqdatac.get_vwap(
                        order_book_ids=batch_stocks,
                        start_date=self.config.time.start_date,
                        end_date=self.config.time.end_date,
                        frequency=self.config.time.frequency
                    )
                    
                    if df_vwap is not None and not df_vwap.empty:
                        df_vwap = self._normalize_index_columns(df_vwap)
                        if 'vwap' not in df_vwap.columns:
                            # reset_index 会把值列放在末尾, 保留 vwap
                            value_cols = [col for col in df_vwap.columns if col not in ['trade_date', 'order_book_id']]
                            if value_cols:
                                df_vwap = df_vwap.rename(columns={value_cols[0]: 'vwap'})
                        
                        # 合并 VWAP 数据
                        df_price = pd.merge(
                            df_price,
                            df_vwap,
                            on=['order_book_id', 'trade_date'],
                            how='left'
                        )
                        logger.info("  ✓ VWAP 数据已添加")
                except Exception as e:
                    logger.warning(f"  ⚠️ VWAP 数据获取失败: {e}")
            
            return df_price
        
        return None

    def _resolve_price_fields(self) -> Tuple[List[str], bool]:
        """解析配置中请求的价格字段，转换为API可用字段并判断是否需要VWAP"""
        requested_fields = self.config.fields.price_fields or []
        alias_map = {
            'vol': 'volume',
            'volume': 'volume',
            'amount': 'total_turnover',
            'total_turnover': 'total_turnover'
        }
        supported_fields = {
            'open', 'high', 'low', 'close', 'volume', 'total_turnover',
            'num_trades', 'prev_close', 'limit_up', 'limit_down'
        }
        resolved: List[str] = []
        include_vwap = getattr(self.config.fields, 'include_vwap', False)
        for field in requested_fields:
            normalized = field.lower()
            if normalized == 'vwap':
                include_vwap = True
                continue
            actual = alias_map.get(normalized, normalized)
            if actual not in supported_fields:
                logger.warning(f"字段 {field} 不被米筐API支持，自动跳过")
                continue
            if actual not in resolved:
                resolved.append(actual)
        if not resolved:
            raise ValueError("price_fields 解析后为空，请确认配置是否正确")
        return resolved, include_vwap
    
    def get_valuation_data(
        self,
        order_book_ids: List[str],
        existing_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        获取估值数据(分批处理)
        
        Args:
            order_book_ids: 股票代码列表
            
        Returns:
            估值数据DataFrame
        """
        if not self.config.fields.valuation_fields:
            logger.info("估值字段列表为空，跳过估值数据获取")
            return existing_df if existing_df is not None else None
        logger.info("=== 获取日频估值数据 ===")
        logger.info(f"字段: {self.config.fields.valuation_fields}")
        self._register_existing('valuation', existing_df)
        pending_ids = order_book_ids
        if self.resume_manager is not None:
            pending_ids = self.resume_manager.filter_pending('valuation', pending_ids)
            if not pending_ids:
                logger.info("断点续传: 估值数据已覆盖所有股票, 跳过API调用")
                return existing_df.copy() if existing_df is not None else pd.DataFrame()
        
        df_val = self._batch_fetch_data(
            order_book_ids=pending_ids,
            fetch_func=self._fetch_valuation_batch,
            desc="获取估值数据",
            entity_name='valuation'
        )
        return self._merge_with_existing(existing_df, df_val)
    
    def _fetch_valuation_batch(self, batch_stocks: List[str]) -> Optional[pd.DataFrame]:
        """获取单批次估值数据"""
        df_val = rqdatac.get_factor(
            order_book_ids=batch_stocks,
            factor=self.config.fields.valuation_fields,
            start_date=self.config.time.start_date,
            end_date=self.config.time.end_date,
            expect_df=True
        )
        
        if df_val is not None and not df_val.empty:
            df_val = self._normalize_index_columns(df_val)
            
            # 重命名列
            rename_dict = {
                'pe_ratio': 'pe',
                'pe_ratio_ttm': 'pe_ttm',
                'pb_ratio': 'pb',
                'ps_ratio': 'ps',
                'ps_ratio_ttm': 'ps_ttm',
                'market_cap': 'total_mv',
                'a_share_market_val_in_circulation': 'circ_mv'
            }
            df_val = df_val.rename(columns=rename_dict)
            
            return df_val
        
        return None
    
    def get_share_data(
        self,
        order_book_ids: List[str],
        existing_df: Optional[pd.DataFrame] = None
    ) -> Optional[pd.DataFrame]:
        """
        获取股本数据(分批处理)
        
        Args:
            order_book_ids: 股票代码列表
            
        Returns:
            股本数据DataFrame或None
        """
        if not self.config.fields.share_fields:
            logger.info("股本字段列表为空，跳过股本数据获取")
            return existing_df if existing_df is not None else None
        logger.info("=== 获取股本数据 ===")
        self._register_existing('share', existing_df)
        
        # 查找可用的股本因子
        try:
            all_factors = rqdatac.get_all_factor_names()
            actual_factors = [f for f in self.config.fields.share_fields if f in all_factors]
            
            if not actual_factors:
                # 尝试搜索
                float_factors = [f for f in all_factors if 'float' in f.lower() and 'share' in f.lower()]
                total_factors = [f for f in all_factors if 'total' in f.lower() and 'share' in f.lower()]
                
                if float_factors:
                    actual_factors.append(float_factors[0])
                if total_factors:
                    actual_factors.append(total_factors[0])
            
            if not actual_factors:
                logger.warning("未找到股本因子, 跳过股本数据获取")
                return None
            
            logger.info(f"将获取以下股本字段: {actual_factors}")
            
            # 临时保存因子列表
            original_fields = self.config.fields.share_fields
            self.config.fields.share_fields = actual_factors
            try:
                pending_ids = order_book_ids
                if self.resume_manager is not None:
                    pending_ids = self.resume_manager.filter_pending('share', pending_ids)
                    if not pending_ids:
                        logger.info("断点续传: 股本数据已覆盖所有股票, 跳过API调用")
                        return existing_df.copy() if existing_df is not None else pd.DataFrame()

                result = self._batch_fetch_data(
                    order_book_ids=pending_ids,
                    fetch_func=self._fetch_share_batch,
                    desc="获取股本数据",
                    entity_name='share'
                )
                return self._merge_with_existing(existing_df, result)
            finally:
                self.config.fields.share_fields = original_fields
        
        except Exception as e:
            logger.error(f"获取股本数据失败: {e}")
            return None
    
    def _fetch_share_batch(self, batch_stocks: List[str]) -> Optional[pd.DataFrame]:
        """获取单批次股本数据"""
        df_share = rqdatac.get_factor(
            order_book_ids=batch_stocks,
            factor=self.config.fields.share_fields,
            start_date=self.config.time.start_date,
            end_date=self.config.time.end_date,
            expect_df=True
        )
        
        if df_share is not None and not df_share.empty:
            df_share = self._normalize_index_columns(df_share)
            
            # 智能重命名
            rename_map = {}
            for col in df_share.columns:
                if 'total' in col.lower() and 'share' in col.lower():
                    rename_map[col] = 'total_share'
                elif 'float' in col.lower() and 'share' in col.lower():
                    rename_map[col] = 'float_share'
            
            if rename_map:
                df_share = df_share.rename(columns=rename_map)
            
            return df_share
        
        return None
    
    def _batch_fetch_data(
        self, 
        order_book_ids: List[str], 
        fetch_func,
        desc: str = "获取数据",
        entity_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        批量获取数据的通用方法
        
        Args:
            order_book_ids: 股票代码列表
            fetch_func: 数据获取函数
            desc: 进度条描述
            
        Returns:
            合并后的DataFrame
        """
        batch_size = self.config.process.batch_size
        sleep_interval = self.config.process.sleep_interval
        all_data = []
        
        for i in tqdm(range(0, len(order_book_ids), batch_size), desc=desc):
            batch_stocks = order_book_ids[i:i+batch_size]
            
            try:
                df_batch = fetch_func(batch_stocks)
                if df_batch is not None:
                    all_data.append(df_batch)
                    if self.resume_manager is not None and entity_name:
                        if 'order_book_id' in df_batch.columns:
                            completed_ids = df_batch['order_book_id'].dropna().unique().tolist()
                        else:
                            completed_ids = batch_stocks
                        if completed_ids:
                            self.resume_manager.mark_completed(
                                entity_name,
                                completed_ids,
                                self.config.time.start_date,
                                self.config.time.end_date
                            )
            except Exception as e:
                logger.error(f"批次 {i//batch_size + 1} 获取失败: {e}")
                continue
            
            # 避免请求过快
            time.sleep(sleep_interval)
        
        # 合并所有数据
        if all_data:
            df_merged = pd.concat(all_data, ignore_index=True)
            logger.info(f"获取到 {len(df_merged)} 条数据")
            return df_merged
        else:
            logger.warning("未获取到任何数据")
            return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, subdir: str, filename: str):
        """
        保存数据
        
        Args:
            df: 待保存的DataFrame
            subdir: 子目录(basic_data/daily_data)
            filename: 文件名(不含扩展名)
        """
        if df is None:
            logger.info(f"跳过保存 {filename}: 数据为空")
            return
        file_format = self.config.storage.file_format
        full_filename = f"{filename}.{file_format}"
        filepath = self.config.storage.get_full_path(subdir, full_filename)
        
        if file_format == 'parquet':
            df.to_parquet(filepath, index=False, compression=self.config.storage.compression)
        elif file_format == 'csv':
            df.to_csv(filepath, index=False)
        elif file_format == 'hdf5':
            df.to_hdf(filepath, key='data', mode='w')
        else:
            raise ValueError(f"不支持的文件格式: {file_format}")
        
        logger.info(f"数据已保存: {filepath}")
