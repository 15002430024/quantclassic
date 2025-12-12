"""
Recorder - 实验记录器

参照 qlib.workflow.recorder.Recorder 设计
负责记录单次实验的所有信息：参数、指标、对象等
"""

import os
import json
import pickle
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import logging


class Recorder:
    """
    实验记录器 - 记录单次实验的所有信息
    
    参照 Qlib 的 Recorder 设计，状态包括:
    - SCHEDULED: 已调度
    - RUNNING: 运行中
    - FINISHED: 已完成
    - FAILED: 失败
    
    Example:
        >>> recorder = Recorder(experiment_id='exp_001', name='lstm_run_1')
        >>> recorder.start()
        >>> recorder.log_params(lr=0.001, batch_size=256)
        >>> recorder.log_metrics(train_loss=0.05)
        >>> recorder.save_objects(model=my_model)
        >>> recorder.end(status='FINISHED')
    """
    
    # 状态常量
    STATUS_S = "SCHEDULED"
    STATUS_R = "RUNNING"
    STATUS_FI = "FINISHED"
    STATUS_FA = "FAILED"
    
    def __init__(self, experiment_id: str, name: str, save_dir: str = 'output/experiments'):
        """
        Args:
            experiment_id: 所属实验的 ID
            name: 记录器名称
            save_dir: 保存目录
        """
        self.experiment_id = experiment_id
        self.name = name
        self.id = self._generate_id()
        self.save_dir = save_dir
        
        # 创建记录器专属目录
        self.recorder_dir = os.path.join(save_dir, experiment_id, self.id)
        Path(self.recorder_dir).mkdir(parents=True, exist_ok=True)
        
        # 时间戳
        self.start_time = None
        self.end_time = None
        
        # 状态
        self.status = Recorder.STATUS_S
        
        # 存储
        self.params = {}
        self.metrics = {}
        self.artifacts = {}  # 保存的对象路径
        
        # 日志
        self.logger = self._setup_logger()
        
        self.logger.info(f"Recorder 创建: {self.id} ({self.name})")
    
    def _generate_id(self) -> str:
        """生成唯一 ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return f"rec_{timestamp}"
    
    def _setup_logger(self) -> logging.Logger:
        """配置日志"""
        logger = logging.getLogger(f'Recorder.{self.id}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 文件处理器
            log_file = os.path.join(self.recorder_dir, 'recorder.log')
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # 格式
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger
    
    @property
    def info(self) -> Dict[str, Any]:
        """获取记录器信息"""
        return {
            'id': self.id,
            'name': self.name,
            'experiment_id': self.experiment_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'status': self.status,
            'params': self.params,
            'metrics': self.metrics,
            'artifacts': list(self.artifacts.keys()),
        }
    
    def start(self):
        """启动记录器"""
        self.start_time = datetime.now().isoformat()
        self.status = Recorder.STATUS_R
        self.logger.info(f"Recorder 启动: {self.name}")
        self._save_meta()
    
    def end(self, status: str = STATUS_FI):
        """
        结束记录器
        
        Args:
            status: 结束状态 (FINISHED 或 FAILED)
        """
        self.end_time = datetime.now().isoformat()
        self.status = status
        self.logger.info(f"Recorder 结束: {self.name}, 状态: {status}")
        self._save_meta()
    
    def log_params(self, **kwargs):
        """
        记录参数
        
        Example:
            >>> recorder.log_params(lr=0.001, batch_size=256, epochs=100)
        """
        self.params.update(kwargs)
        self.logger.info(f"记录参数: {kwargs}")
        self._save_meta()
    
    def log_metrics(self, step: Optional[int] = None, **kwargs):
        """
        记录指标
        
        Args:
            step: 步骤编号（可选，用于记录训练过程中的指标）
            **kwargs: 指标键值对
        
        Example:
            >>> recorder.log_metrics(train_loss=0.05, valid_loss=0.06)
            >>> recorder.log_metrics(step=10, train_loss=0.04)
        """
        if step is not None:
            # 时序指标
            if 'history' not in self.metrics:
                self.metrics['history'] = {}
            
            for key, value in kwargs.items():
                if key not in self.metrics['history']:
                    self.metrics['history'][key] = []
                self.metrics['history'][key].append({'step': step, 'value': value})
        else:
            # 最终指标
            self.metrics.update(kwargs)
        
        self.logger.info(f"记录指标 (step={step}): {kwargs}")
        self._save_meta()
    
    def save_objects(self, local_path: Optional[str] = None, 
                    artifact_path: Optional[str] = None, **kwargs):
        """
        保存对象（模型、预测结果等）
        
        Args:
            local_path: 如果提供，则复制该文件/目录到 artifact 目录
            artifact_path: artifact 的相对路径
            **kwargs: 对象名称和对象的键值对
        
        Example:
            >>> recorder.save_objects(model=my_model, predictions=pred)
            >>> recorder.save_objects(local_path='model.pth', artifact_path='models/')
        """
        # 创建 artifacts 目录
        artifacts_dir = os.path.join(self.recorder_dir, 'artifacts')
        Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
        
        # 保存本地文件/目录
        if local_path is not None:
            dest_path = os.path.join(artifacts_dir, artifact_path or '')
            Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
            
            if os.path.isdir(local_path):
                shutil.copytree(local_path, dest_path, dirs_exist_ok=True)
            else:
                shutil.copy2(local_path, dest_path)
            
            self.logger.info(f"保存文件: {local_path} -> {dest_path}")
        
        # 保存 Python 对象
        for name, obj in kwargs.items():
            obj_path = os.path.join(artifacts_dir, f'{name}.pkl')
            
            try:
                with open(obj_path, 'wb') as f:
                    pickle.dump(obj, f)
                
                self.artifacts[name] = obj_path
                self.logger.info(f"保存对象: {name} -> {obj_path}")
            except Exception as e:
                self.logger.error(f"保存对象失败 {name}: {e}")
                raise
        
        self._save_meta()
    
    def load_object(self, name: str) -> Any:
        """
        加载已保存的对象
        
        Args:
            name: 对象名称
        
        Returns:
            加载的对象
        
        Example:
            >>> model = recorder.load_object('model')
        """
        if name not in self.artifacts:
            raise KeyError(f"对象 '{name}' 不存在。可用对象: {list(self.artifacts.keys())}")
        
        obj_path = self.artifacts[name]
        
        try:
            with open(obj_path, 'rb') as f:
                obj = pickle.load(f)
            
            self.logger.info(f"加载对象: {name} <- {obj_path}")
            return obj
        except Exception as e:
            self.logger.error(f"加载对象失败 {name}: {e}")
            raise
    
    def list_artifacts(self) -> list:
        """列出所有已保存的对象"""
        return list(self.artifacts.keys())
    
    def get_params(self) -> Dict[str, Any]:
        """
        获取所有参数
        
        Returns:
            参数字典
        """
        return self.params.copy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取所有指标
        
        Returns:
            指标字典。如果有时序指标（step-wise），包含在 'history' 键中
        """
        return self.metrics.copy()
    
    def _save_meta(self):
        """保存元数据到 JSON 文件"""
        meta_path = os.path.join(self.recorder_dir, 'meta.json')
        
        try:
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(self.info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存元数据失败: {e}")
    
    @classmethod
    def load_from_dir(cls, recorder_dir: str) -> 'Recorder':
        """
        从目录加载记录器
        
        Args:
            recorder_dir: 记录器目录
        
        Returns:
            Recorder 实例
        """
        meta_path = os.path.join(recorder_dir, 'meta.json')
        
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"元数据文件不存在: {meta_path}")
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        # 重建 Recorder
        recorder = cls.__new__(cls)
        recorder.id = meta['id']
        recorder.name = meta['name']
        recorder.experiment_id = meta['experiment_id']
        recorder.recorder_dir = recorder_dir
        recorder.save_dir = str(Path(recorder_dir).parent.parent)
        
        recorder.start_time = meta.get('start_time')
        recorder.end_time = meta.get('end_time')
        recorder.status = meta.get('status', cls.STATUS_S)
        recorder.params = meta.get('params', {})
        recorder.metrics = meta.get('metrics', {})
        
        # 重建 artifacts 路径
        recorder.artifacts = {}
        artifacts_dir = os.path.join(recorder_dir, 'artifacts')
        for name in meta.get('artifacts', []):
            obj_path = os.path.join(artifacts_dir, f'{name}.pkl')
            if os.path.exists(obj_path):
                recorder.artifacts[name] = obj_path
        
        recorder.logger = recorder._setup_logger()
        
        return recorder
    
    def __repr__(self) -> str:
        return f"Recorder(id={self.id}, name={self.name}, status={self.status})"
    
    def __str__(self) -> str:
        return f"Recorder '{self.name}' ({self.status})"


if __name__ == '__main__':
    # 测试 Recorder
    print("=" * 80)
    print("Recorder 测试")
    print("=" * 80)
    
    # 创建记录器
    recorder = Recorder(experiment_id='test_exp', name='test_run')
    print(f"\n✅ 创建记录器: {recorder}")
    
    # 启动
    recorder.start()
    print(f"✅ 启动记录器")
    
    # 记录参数
    recorder.log_params(lr=0.001, batch_size=256, epochs=100)
    print(f"✅ 记录参数")
    
    # 记录指标
    recorder.log_metrics(train_loss=0.05, valid_loss=0.06)
    recorder.log_metrics(step=1, train_loss=0.04)
    recorder.log_metrics(step=2, train_loss=0.03)
    print(f"✅ 记录指标")
    
    # 保存对象
    test_obj = {'data': [1, 2, 3], 'info': 'test'}
    recorder.save_objects(test_model=test_obj)
    print(f"✅ 保存对象")
    
    # 结束
    recorder.end(Recorder.STATUS_FI)
    print(f"✅ 结束记录器")
    
    # 查看信息
    print(f"\n记录器信息:")
    print(f"  ID: {recorder.id}")
    print(f"  参数: {recorder.params}")
    print(f"  指标: {recorder.metrics}")
    print(f"  对象: {recorder.list_artifacts()}")
    
    # 加载对象
    loaded_obj = recorder.load_object('test_model')
    print(f"\n✅ 加载对象: {loaded_obj}")
    
    # 重新加载记录器
    print(f"\n从目录重新加载记录器...")
    recorder2 = Recorder.load_from_dir(recorder.recorder_dir)
    print(f"✅ 加载成功: {recorder2}")
    print(f"  参数: {recorder2.params}")
    
    print("\n" + "=" * 80)
    print("✅ Recorder 测试完成")
    print("=" * 80)
