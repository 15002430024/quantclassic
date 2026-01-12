"""
TaskRunner - 任务运行器

支持两种配置模式:
1. 字典配置 (向后兼容): {'task': {'model': {...}, 'dataset': {...}}}
2. Dataclass 配置 (新模式): TaskConfig 对象

支持多种训练模式:
1. 默认训练: 使用模型的 fit 方法
2. 滚动窗口训练: 使用 RollingDailyTrainer
3. 动态图训练: 使用 DynamicGraphTrainer
"""

import logging
import copy
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import fields

from .utils import init_instance_by_config
from .base_config import BaseConfig, TaskConfig


def _is_dataclass_config(obj) -> bool:
    """检查对象是否为 BaseConfig 子类实例"""
    return isinstance(obj, BaseConfig)


def _config_to_dict(config) -> Dict[str, Any]:
    """将配置对象转换为字典（支持 BaseConfig 和普通字典）"""
    if _is_dataclass_config(config):
        return config.to_dict()
    elif isinstance(config, dict):
        return config
    else:
        raise TypeError(f"不支持的配置类型: {type(config)}")


def _adapt_task_config_to_legacy(task_config: TaskConfig) -> Dict[str, Any]:
    """
    🆕 TaskConfig 适配器
    
    将新版 TaskConfig（扁平结构）转换为旧版 Runner 期望的嵌套字典结构
    
    TaskConfig 结构:
        model_class, model_kwargs, dataset_class, dataset_kwargs, ...
        
    旧版结构:
        {'task': {'model': {'class': ..., 'kwargs': ...}, 'dataset': {...}}}
    
    Args:
        task_config: TaskConfig 对象
        
    Returns:
        旧版格式的配置字典
    """
    legacy_config = {'task': {}}
    
    # 转换模型配置
    if task_config.model_class:
        legacy_config['task']['model'] = {
            'class': task_config.model_class,
            'module_path': 'quantclassic.model',  # 默认模块路径
            'kwargs': task_config.model_kwargs or {}
        }
    
    # 转换数据集配置
    if task_config.dataset_class:
        legacy_config['task']['dataset'] = {
            'class': task_config.dataset_class,
            'module_path': 'quantclassic.data_set',  # 默认模块路径
            'kwargs': task_config.dataset_kwargs or {}
        }
    
    # 转换回测配置
    if task_config.backtest_enabled:
        legacy_config['task']['backtest'] = task_config.backtest_kwargs or {}
    
    return legacy_config


class TaskRunner:
    """任务运行器 - 执行配置定义的完整工作流"""
    
    def __init__(self, log_level: str = 'INFO'):
        self.logger = self._setup_logger(log_level)
        # 🆕 保存模型配置用于重建模型工厂
        self._model_config: Optional[Dict[str, Any]] = None
    
    def _setup_logger(self, log_level: str) -> logging.Logger:
        logger = logging.getLogger('TaskRunner')
        logger.setLevel(getattr(logging, log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run(self, 
            config: Union[Dict[str, Any], TaskConfig],
            experiment_name: str = 'default_experiment',
            recorder_name: Optional[str] = None) -> Dict[str, Any]:
        """
        运行完整的训练任务
        
        Args:
            config: 配置字典或 TaskConfig 对象
            experiment_name: 实验名称
            recorder_name: 记录器名称
            
        Returns:
            训练结果字典
        """
        self.logger.info(f"开始运行任务: {experiment_name}")
        
        # 🆕 统一处理配置格式 - 使用适配器
        task_config: Optional[TaskConfig] = None
        
        if isinstance(config, TaskConfig):
            # 新版 TaskConfig 对象 → 转换为旧版字典
            self.logger.info("检测到 TaskConfig 对象，使用适配器转换...")
            task_config = config
            config_dict = _adapt_task_config_to_legacy(config)
            self.logger.debug(f"转换后的配置: {config_dict}")
        elif _is_dataclass_config(config):
            # 其他 BaseConfig 子类
            self.logger.info("检测到 BaseConfig 对象，转换为字典...")
            config_dict = {'task': _config_to_dict(config)}
        elif isinstance(config, dict):
            # 旧版字典配置
            config_dict = config
            # 尝试从字典构建 TaskConfig（用于检测高级功能）
            if 'task' in config_dict:
                try:
                    task_config = TaskConfig.from_dict(config_dict['task'])
                except Exception:
                    pass  # 老格式配置，不强制转换
        else:
            raise TypeError(f"不支持的配置类型: {type(config)}")
        
        # 验证配置结构
        if 'task' not in config_dict:
            raise ValueError("配置必须包含 'task' 键")
        
        try:
            from ..workflow import R
            use_recorder = True
            self.logger.info("使用 workflow.R 记录实验")
        except ImportError:
            use_recorder = False
            self.logger.warning("workflow模块不可用")
        
        if use_recorder:
            ctx = R.start(experiment_name=experiment_name, recorder_name=recorder_name)
            ctx.__enter__()
            R.log_params(**self._flatten_config(config_dict.get('task', {})))
        
        try:
            dataset = None
            data_manager = None
            rolling_loaders = None
            daily_loaders = None
            
            # ==================== 步骤 1: 初始化数据集 ====================
            if 'dataset' in config_dict['task']:
                self.logger.info("步骤 1/4: 初始化数据集...")
                dataset, data_manager = self._init_dataset(config_dict['task']['dataset'])
                self.logger.info(f"数据集初始化完成: {type(dataset).__name__}")
                
                # 🆕 检测是否需要创建滚动/日批次加载器
                # 直接从 data_manager.config 获取 graph_builder_config，不从嵌套字典抠
                if task_config and data_manager is not None:
                    graph_config = getattr(data_manager.config, 'graph_builder_config', None)
                    
                    if task_config.use_rolling_loaders:
                        self.logger.info("创建滚动窗口日批次加载器...")
                        rolling_loaders = data_manager.create_rolling_daily_loaders(
                            graph_builder_config=graph_config
                        )
                        self.logger.info(f"滚动窗口数量: {len(rolling_loaders)}")
                    elif task_config.use_daily_loaders:
                        self.logger.info("创建日批次加载器...")
                        daily_loaders = data_manager.create_daily_loaders(
                            graph_builder_config=graph_config
                        )
                        self.logger.info("日批次加载器创建完成")
            
            # ==================== 步骤 2: 初始化模型 ====================
            model = None
            if 'model' in config_dict['task']:
                self.logger.info("步骤 2/4: 初始化模型...")
                # 🆕 保存模型配置用于后续重建
                self._model_config = config_dict['task']['model']
                model = self._init_model(config_dict['task']['model'])
                self.logger.info(f"模型初始化完成: {type(model).__name__}")
            
            # ==================== 步骤 3: 训练模型 ====================
            train_results = {}
            if model is not None and dataset is not None:
                self.logger.info("步骤 3/4: 训练模型...")
                
                # 🆕 根据配置选择训练方式 (支持新训练架构)
                trainer_class = task_config.trainer_class if task_config else ''
                trainer_kwargs = task_config.trainer_kwargs if task_config else {}
                
                if trainer_class == 'RollingDailyTrainer' and rolling_loaders:
                    train_results = self._train_rolling(
                        model, rolling_loaders, trainer_kwargs or {}
                    )
                elif trainer_class == 'RollingWindowTrainer' and rolling_loaders:
                    # 🆕 新增: 支持 RollingWindowTrainer
                    train_results = self._train_rolling_window(
                        model, rolling_loaders, trainer_kwargs or {}
                    )
                elif trainer_class == 'SimpleTrainer':
                    # 🆕 新增: 支持 SimpleTrainer
                    train_results = self._train_simple(
                        model, dataset, trainer_kwargs or {}
                    )
                elif trainer_class == 'DynamicGraphTrainer' and daily_loaders:
                    # ⚠️ DynamicGraphTrainer 已废弃，内部改用 SimpleTrainer
                    self.logger.warning(
                        "⚠️ trainer_class='DynamicGraphTrainer' 已废弃！\n"
                        "   实际使用 SimpleTrainer 执行。建议改用 trainer_class='SimpleTrainer'。"
                    )
                    train_results = self._train_dynamic_graph(
                        model, daily_loaders, trainer_kwargs or {}
                    )
                else:
                    train_results = self._train_model(model, dataset, config_dict['task'])
                
                self.logger.info("模型训练完成")
                
                if use_recorder and train_results:
                    R.log_metrics(**train_results.get('metrics', {}))
                    if 'model' in train_results:
                        R.save_objects(model=train_results['model'])
            
            # ==================== 步骤 4: 回测 ====================
            backtest_results = {}
            if 'backtest' in config_dict['task']:
                self.logger.info("步骤 4/4: 执行回测...")
                backtest_results = self._run_backtest(model, dataset, config_dict['task']['backtest'])
                self.logger.info("回测完成")
                
                if use_recorder and backtest_results:
                    R.log_metrics(**backtest_results.get('metrics', {}))
            
            results = {
                'model': model,
                'dataset': dataset,
                'data_manager': data_manager,
                'rolling_loaders': rolling_loaders,
                'daily_loaders': daily_loaders,
                'train_results': train_results,
                'backtest_results': backtest_results,
                'experiment_name': experiment_name
            }
            
            if use_recorder:
                R.save_objects(config=config_dict, results=results)
            
            self.logger.info(f"任务完成: {experiment_name}")
            return results
            
        except Exception as e:
            self.logger.error(f"任务执行失败: {e}", exc_info=True)
            raise
        
        finally:
            if use_recorder:
                ctx.__exit__(None, None, None)
    
    def _init_dataset(self, dataset_config: Dict[str, Any]):
        """
        初始化数据集
        
        🆕 修复: 正确处理 DataConfig，不再将其传给 run_full_pipeline 的 file_path 参数
        
        Args:
            dataset_config: 数据集配置字典
            
        Returns:
            (loaders, data_manager): 数据加载器和管理器
        """
        # 支持 BaseConfig 对象
        if _is_dataclass_config(dataset_config):
            dataset_config = _config_to_dict(dataset_config)
        
        data_manager = None
        
        if dataset_config.get('class') == 'DataManager':
            # 🆕 修复: 检查 kwargs 中是否有 config
            kwargs = dataset_config.get('kwargs', {})
            config_dict = kwargs.get('config', {})
            
            if config_dict:
                from ..data_set.config import DataConfig
                from ..data_set.manager import DataManager
                
                # 构建 DataConfig 对象
                if isinstance(config_dict, dict):
                    data_config = DataConfig(**config_dict)
                else:
                    data_config = config_dict
                
                # 🆕 检查并警告图构建器配置缺失
                if data_config.graph_builder_config is None:
                    self.logger.warning(
                        "⚠️ graph_builder_config 未配置！\n"
                        "   模型将使用单位矩阵（无图交互），可能导致性能下降。\n"
                        "   建议在 DataConfig 中配置 graph_builder_config。"
                    )
                
                # 🆕 修复: 用 DataConfig 初始化 DataManager，而不是传给 run_full_pipeline
                manager = DataManager(config=data_config)
                data_manager = manager
                
                # 🆕 修复: run_full_pipeline 不传参数（或仅传可选参数）
                loaders = manager.run_full_pipeline()
                return loaders, data_manager
            else:
                # 没有 config，使用旧版初始化方式
                manager = init_instance_by_config(dataset_config)
                data_manager = manager
                return manager, data_manager
        
        return init_instance_by_config(dataset_config), None
    
    def _init_model(self, model_config: Dict[str, Any]):
        """初始化模型，支持 BaseConfig 对象"""
        if _is_dataclass_config(model_config):
            model_config = _config_to_dict(model_config)
        
        return init_instance_by_config(model_config)
    
    def _train_model(self, model, dataset, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """默认训练方式：调用模型的 fit 方法"""
        results = {}
        
        if not hasattr(model, 'fit'):
            self.logger.warning("模型没有fit方法，跳过训练")
            return results
        
        if hasattr(dataset, 'train') and hasattr(dataset, 'val'):
            train_loader = dataset.train
            val_loader = dataset.val
            test_loader = dataset.test if hasattr(dataset, 'test') else None
            
            model.fit(train_loader, val_loader)
            
            if hasattr(model, 'best_metrics'):
                results['metrics'] = model.best_metrics
            
            if test_loader is not None:
                predictions = model.predict(test_loader)
                results['predictions'] = predictions
        else:
            model.fit(dataset)
        
        results['model'] = model
        return results
    
    # ==================== 🆕 SimpleTrainer 训练方法 ====================
    
    def _train_simple(self, model, dataset, trainer_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用 SimpleTrainer 进行训练
        
        Args:
            model: 模型
            dataset: 数据集（需要有 train, val 属性）
            trainer_kwargs: 训练器参数
            
        Returns:
            训练结果字典
        """
        from ..model.train import SimpleTrainer, TrainerConfig
        
        self.logger.info("使用 SimpleTrainer 进行训练")
        
        # 获取底层 nn.Module
        if hasattr(model, 'model'):
            nn_model = model.model
        else:
            nn_model = model
        
        # 创建配置
        config = TrainerConfig(**trainer_kwargs) if trainer_kwargs else TrainerConfig()
        
        # 创建训练器
        trainer = SimpleTrainer(nn_model, config)
        
        # 获取数据加载器
        train_loader = dataset.train if hasattr(dataset, 'train') else dataset
        val_loader = dataset.val if hasattr(dataset, 'val') else None
        test_loader = dataset.test if hasattr(dataset, 'test') else None
        
        # 训练
        result = trainer.train(train_loader, val_loader)
        
        # 预测
        if test_loader is not None:
            predictions = trainer.predict(test_loader)
            result['predictions'] = predictions
        
        result['model'] = model
        result['trainer'] = trainer
        
        return result
    
    # ==================== 🆕 RollingWindowTrainer 训练方法 ====================
    
    def _train_rolling_window(self, model, rolling_loaders, trainer_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用 RollingWindowTrainer 进行滚动窗口训练
        
        Args:
            model: 模型
            rolling_loaders: 滚动窗口数据加载器
            trainer_kwargs: 训练器参数
            
        Returns:
            训练结果字典
        """
        from ..model.train import RollingWindowTrainer, RollingTrainerConfig
        
        self.logger.info("使用 RollingWindowTrainer 进行滚动窗口训练")
        
        # 获取底层 nn.Module
        if hasattr(model, 'model'):
            nn_model = model.model
        else:
            nn_model = model
        
        # 创建模型工厂
        initial_model_copy = copy.deepcopy(nn_model)
        
        def model_factory():
            return copy.deepcopy(initial_model_copy)
        
        # 🆕 修复: init_params 中的参数同时透传给 config
        init_params = {'weight_inheritance', 'save_each_window', 'device'}
        fit_params = {'save_dir', 'n_epochs'}
        
        init_kwargs = {}
        fit_kwargs = {}
        config_kwargs = {}
        
        for key, value in trainer_kwargs.items():
            if key in init_params:
                init_kwargs[key] = value
                # 🆕 weight_inheritance 和 save_each_window 同时传入 config
                if key in {'weight_inheritance', 'save_each_window'}:
                    config_kwargs[key] = value
            elif key in fit_params:
                fit_kwargs[key] = value
            else:
                config_kwargs[key] = value
        
        # 创建配置
        config = RollingTrainerConfig(**config_kwargs) if config_kwargs else RollingTrainerConfig()
        
        # 创建训练器
        trainer = RollingWindowTrainer(
            model_factory=model_factory,
            config=config,
            device=init_kwargs.get('device', 'cuda')
        )
        
        # 训练
        save_dir = fit_kwargs.get('save_dir', 'output/rolling_models')
        n_epochs = fit_kwargs.get('n_epochs')
        
        results = trainer.train(rolling_loaders, n_epochs=n_epochs, save_dir=save_dir)
        
        # 获取预测
        all_predictions = trainer.get_all_predictions()
        results['predictions'] = all_predictions
        results['model'] = model
        results['trainer'] = trainer
        
        return results
    
    def _run_backtest(self, model, dataset, backtest_config: Dict[str, Any]) -> Dict[str, Any]:
        """执行回测"""
        from ..backtest import BacktestSystem
        
        if hasattr(dataset, 'test'):
            test_loader = dataset.test
            predictions = model.predict(test_loader)
        else:
            self.logger.warning("数据集没有test部分，跳过回测")
            return {}
        
        backtest_system = BacktestSystem(**backtest_config)
        backtest_results = backtest_system.run_backtest(predictions=predictions, **backtest_config)
        
        return {'metrics': backtest_results, 'predictions': predictions}
    
    # ==================== 🆕 滚动训练方法（重构） ====================
    
    def _train_rolling(self, model, rolling_loaders, trainer_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用 RollingDailyTrainer 进行滚动窗口训练
        
        🆕 重构: 使用新的 model/train/ 模块
        
        Args:
            model: 模型（需要是 nn.Module 或有 .model 属性）
            rolling_loaders: RollingDailyLoaderCollection
            trainer_kwargs: 训练器参数
            
        Returns:
            训练结果字典
        """
        # 🆕 优先使用新的训练架构
        try:
            from ..model.train import RollingDailyTrainer, RollingTrainerConfig
            use_new_trainer = True
        except ImportError:
            from ..model.rolling_daily_trainer import RollingDailyTrainer, RollingTrainerConfig
            use_new_trainer = False
        
        self.logger.info(f"使用 {'新' if use_new_trainer else '旧'} RollingDailyTrainer 进行滚动窗口训练")
        
        # 获取底层 nn.Module
        if hasattr(model, 'model'):
            nn_model = model.model
        else:
            nn_model = model
        
        # 使用 copy.deepcopy 创建模型工厂
        initial_model_copy = copy.deepcopy(nn_model)
        
        def model_factory():
            """模型工厂：返回初始状态模型的深拷贝"""
            return copy.deepcopy(initial_model_copy)
        
        # 🆕 拆分 trainer_kwargs
        from dataclasses import fields as dc_fields
        
        try:
            config_field_names = {f.name for f in dc_fields(RollingTrainerConfig)}
        except Exception:
            config_field_names = set()
        
        # RollingDailyTrainer 构造函数接受的参数
        trainer_init_params = {'warm_start', 'save_each_window', 'device'}
        
        # trainer.fit() 接受的参数
        fit_params = {'save_dir', 'n_epochs'}
        
        # 分离参数
        config_kwargs = {}
        init_kwargs = {}
        fit_kwargs = {}
        
        for key, value in trainer_kwargs.items():
            if key in trainer_init_params:
                init_kwargs[key] = value
            elif key in fit_params:
                fit_kwargs[key] = value
            elif key in config_field_names:
                config_kwargs[key] = value
            else:
                config_kwargs[key] = value
        
        # 创建训练配置
        if use_new_trainer:
            from ..model.train.rolling_daily_trainer import DailyRollingConfig
            try:
                config = DailyRollingConfig(**config_kwargs) if config_kwargs else DailyRollingConfig()
            except TypeError as e:
                self.logger.warning(f"创建 DailyRollingConfig 失败: {e}，使用默认配置")
                config = DailyRollingConfig()
        else:
            config = RollingTrainerConfig(**config_kwargs) if config_kwargs else RollingTrainerConfig()
        
        # 创建训练器
        trainer = RollingDailyTrainer(
            model_factory=model_factory,
            config=config,
            warm_start=init_kwargs.get('warm_start', True),
            save_each_window=init_kwargs.get('save_each_window', True),
            device=init_kwargs.get('device', 'cuda')
        )
        
        # 训练
        save_dir = fit_kwargs.get('save_dir', 'output/rolling_models')
        n_epochs = fit_kwargs.get('n_epochs', config.n_epochs)
        
        results = trainer.fit(rolling_loaders, n_epochs=n_epochs, save_dir=save_dir)
        
        # 获取所有预测
        all_predictions = trainer.get_all_predictions()
        results['predictions'] = all_predictions
        results['trainer'] = trainer
        
        return results
    
    def _train_dynamic_graph(self, model, daily_loaders, trainer_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用 SimpleTrainer 进行动态图训练
        
        🆕 重构: DynamicGraphTrainer 已废弃，改用 SimpleTrainer
        
        Args:
            model: 模型
            daily_loaders: DailyLoaderCollection (train, val, test)
            trainer_kwargs: 训练器参数
            
        Returns:
            训练结果字典
        """
        # 🆕 使用新的 SimpleTrainer 替代已废弃的 DynamicGraphTrainer
        from ..model.train import SimpleTrainer, TrainerConfig
        
        self.logger.info("使用 SimpleTrainer 进行动态图训练 (DynamicGraphTrainer 已废弃)")
        
        # 获取底层 nn.Module
        if hasattr(model, 'model'):
            nn_model = model.model
        else:
            nn_model = model
        
        # 拆分参数
        fit_params = {'save_path', 'n_epochs'}
        
        config_kwargs = {}
        fit_kwargs = {}
        
        for key, value in trainer_kwargs.items():
            if key in fit_params:
                fit_kwargs[key] = value
            else:
                config_kwargs[key] = value
        
        # 创建训练配置
        config = TrainerConfig(**config_kwargs) if config_kwargs else TrainerConfig()
        
        # 创建训练器
        trainer = SimpleTrainer(
            model=nn_model,
            config=config,
            device=trainer_kwargs.get('device', 'cuda')
        )
        
        # 训练
        save_path = fit_kwargs.get('save_path', 'output/best_model.pth')
        n_epochs = fit_kwargs.get('n_epochs', config.n_epochs)
        
        # 准备 DataLoader
        train_loader = daily_loaders.train if hasattr(daily_loaders, 'train') else daily_loaders
        val_loader = daily_loaders.val if hasattr(daily_loaders, 'val') else None
        
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=n_epochs,
            save_path=save_path
        )
        
        # 预测
        if hasattr(daily_loaders, 'test') and daily_loaders.test:
            predictions = trainer.predict(daily_loaders.test)
            results['predictions'] = predictions
        
        results['trainer'] = trainer
        
        return results
    
    def _flatten_config(self, config: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """将嵌套配置展平为单层字典"""
        items = []
        for k, v in config.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict) and k not in ['kwargs']:
                items.extend(self._flatten_config(v, new_key, sep=sep).items())
            elif isinstance(v, (str, int, float, bool)) or v is None:
                items.append((new_key, v))
        
        return dict(items)
