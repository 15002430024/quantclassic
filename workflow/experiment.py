"""
Experiment - 实验管理

参照 qlib.workflow.exp.Experiment 设计
负责管理单个实验及其下的多个记录器
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging


class Experiment:
    """
    实验管理类 - 管理单个实验及其多个记录器
    
    一个实验可以包含多个记录器（runs），每个记录器代表一次训练/测试运行
    
    Example:
        >>> exp = Experiment(name='lstm_experiments')
        >>> recorder1 = exp.create_recorder('run_1')
        >>> recorder2 = exp.create_recorder('run_2')
        >>> exp.list_recorders()
    """
    
    def __init__(self, name: str, save_dir: str = 'output/experiments'):
        """
        Args:
            name: 实验名称
            save_dir: 保存目录
        """
        self.name = name
        self.id = self._generate_id()
        self.save_dir = save_dir
        
        # 创建实验目录
        self.exp_dir = os.path.join(save_dir, self.id)
        Path(self.exp_dir).mkdir(parents=True, exist_ok=True)
        
        # 时间戳
        self.create_time = datetime.now().isoformat()
        
        # 记录器列表
        self.recorders: List = []
        
        # 日志
        self.logger = self._setup_logger()
        
        # 保存元数据
        self._save_meta()
        
        self.logger.info(f"实验创建: {self.id} ({self.name})")
    
    def _generate_id(self) -> str:
        """生成唯一 ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"exp_{self.name}_{timestamp}"
    
    def _setup_logger(self) -> logging.Logger:
        """配置日志"""
        logger = logging.getLogger(f'Experiment.{self.id}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @property
    def info(self) -> Dict[str, Any]:
        """获取实验信息"""
        return {
            'id': self.id,
            'name': self.name,
            'create_time': self.create_time,
            'num_recorders': len(self.recorders),
            'recorders': [r.id for r in self.recorders],
        }
    
    def create_recorder(self, name: str, auto_start: bool = True):
        """
        创建新的记录器
        
        Args:
            name: 记录器名称
            auto_start: 是否自动启动
        
        Returns:
            Recorder 实例
        """
        from .recorder import Recorder
        
        recorder = Recorder(
            experiment_id=self.id,
            name=name,
            save_dir=self.save_dir
        )
        
        if auto_start:
            recorder.start()
        
        self.recorders.append(recorder)
        self._save_meta()
        
        self.logger.info(f"创建记录器: {recorder.id} ({name})")
        
        return recorder
    
    def get_recorder(self, recorder_id: Optional[str] = None, 
                    recorder_name: Optional[str] = None):
        """
        获取记录器
        
        Args:
            recorder_id: 记录器 ID
            recorder_name: 记录器名称
        
        Returns:
            Recorder 实例或 None
        """
        for recorder in self.recorders:
            if recorder_id and recorder.id == recorder_id:
                return recorder
            if recorder_name and recorder.name == recorder_name:
                return recorder
        
        # 尝试从磁盘加载
        return self._load_recorder_from_disk(recorder_id, recorder_name)
    
    def _load_recorder_from_disk(self, recorder_id: Optional[str] = None,
                                 recorder_name: Optional[str] = None):
        """从磁盘加载记录器"""
        from .recorder import Recorder
        
        # 搜索实验目录下的所有记录器
        for item in os.listdir(self.exp_dir):
            item_path = os.path.join(self.exp_dir, item)
            if not os.path.isdir(item_path):
                continue
            
            meta_path = os.path.join(item_path, 'meta.json')
            if not os.path.exists(meta_path):
                continue
            
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                if recorder_id and meta.get('id') == recorder_id:
                    return Recorder.load_from_dir(item_path)
                if recorder_name and meta.get('name') == recorder_name:
                    return Recorder.load_from_dir(item_path)
            except Exception as e:
                self.logger.warning(f"加载记录器失败 {item_path}: {e}")
        
        return None
    
    def list_recorders(self) -> List[Dict[str, Any]]:
        """
        列出所有记录器
        
        Returns:
            记录器信息列表
        """
        # 从磁盘加载所有记录器
        from .recorder import Recorder
        
        recorder_infos = []
        
        for item in os.listdir(self.exp_dir):
            item_path = os.path.join(self.exp_dir, item)
            if not os.path.isdir(item_path):
                continue
            
            meta_path = os.path.join(item_path, 'meta.json')
            if not os.path.exists(meta_path):
                continue
            
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                recorder_infos.append(meta)
            except Exception as e:
                self.logger.warning(f"读取记录器元数据失败 {item_path}: {e}")
        
        return recorder_infos
    
    def _save_meta(self):
        """保存实验元数据"""
        meta_path = os.path.join(self.exp_dir, 'experiment_meta.json')
        
        try:
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(self.info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存实验元数据失败: {e}")
    
    @classmethod
    def load(cls, exp_id: str, save_dir: str = 'output/experiments') -> 'Experiment':
        """
        从磁盘加载实验
        
        Args:
            exp_id: 实验 ID
            save_dir: 保存目录
        
        Returns:
            Experiment 实例
        """
        exp_dir = os.path.join(save_dir, exp_id)
        meta_path = os.path.join(exp_dir, 'experiment_meta.json')
        
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"实验元数据不存在: {meta_path}")
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        # 重建 Experiment
        exp = cls.__new__(cls)
        exp.id = meta['id']
        exp.name = meta['name']
        exp.save_dir = save_dir
        exp.exp_dir = exp_dir
        exp.create_time = meta.get('create_time')
        exp.recorders = []
        exp.logger = exp._setup_logger()
        
        return exp
    
    def __repr__(self) -> str:
        return f"Experiment(id={self.id}, name={self.name}, recorders={len(self.recorders)})"
    
    def __str__(self) -> str:
        return f"Experiment '{self.name}' ({len(self.recorders)} recorders)"


if __name__ == '__main__':
    # 测试 Experiment
    print("=" * 80)
    print("Experiment 测试")
    print("=" * 80)
    
    # 创建实验
    exp = Experiment(name='test_experiment')
    print(f"\n✅ 创建实验: {exp}")
    
    # 创建记录器
    recorder1 = exp.create_recorder('run_1')
    recorder1.log_params(lr=0.001, batch_size=256)
    recorder1.log_metrics(loss=0.05)
    recorder1.end()
    print(f"✅ 创建记录器 1: {recorder1}")
    
    recorder2 = exp.create_recorder('run_2')
    recorder2.log_params(lr=0.002, batch_size=512)
    recorder2.log_metrics(loss=0.04)
    recorder2.end()
    print(f"✅ 创建记录器 2: {recorder2}")
    
    # 列出所有记录器
    print(f"\n所有记录器:")
    for info in exp.list_recorders():
        print(f"  - {info['name']}: {info['status']}, params={info.get('params')}")
    
    # 获取特定记录器
    rec = exp.get_recorder(recorder_name='run_1')
    print(f"\n✅ 获取记录器: {rec}")
    print(f"  参数: {rec.params}")
    
    # 重新加载实验
    print(f"\n从磁盘重新加载实验...")
    exp2 = Experiment.load(exp.id)
    print(f"✅ 加载成功: {exp2}")
    print(f"  记录器数量: {len(exp2.list_recorders())}")
    
    print("\n" + "=" * 80)
    print("✅ Experiment 测试完成")
    print("=" * 80)
