"""
ExpManager - 实验管理器

参照 qlib.workflow.expm.ExpManager 设计
负责管理所有实验，提供实验的创建、查询、删除等功能
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging


class ExpManager:
    """
    实验管理器 - 管理所有实验
    
    负责:
    - 创建和管理实验
    - 查询实验和记录器
    - 维护实验索引
    
    Example:
        >>> manager = ExpManager()
        >>> exp = manager.create_experiment('lstm_test')
        >>> recorder = manager.start_recorder(experiment_name='lstm_test', recorder_name='run_1')
    """
    
    def __init__(self, save_dir: str = 'output/experiments'):
        """
        Args:
            save_dir: 保存目录
        """
        self.save_dir = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # 索引文件
        self.index_file = os.path.join(save_dir, 'experiments_index.json')
        
        # 加载索引
        self.experiments_index = self._load_index()
        
        # 当前活动的实验和记录器
        self.current_experiment = None
        self.current_recorder = None
        
        # 日志
        self.logger = logging.getLogger('ExpManager')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _load_index(self) -> Dict[str, Any]:
        """加载实验索引"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载索引失败: {e}，创建新索引")
        
        return {'experiments': {}}
    
    def _save_index(self):
        """保存实验索引"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.experiments_index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存索引失败: {e}")
    
    def create_experiment(self, name: str) -> 'Experiment':
        """
        创建新实验
        
        Args:
            name: 实验名称
        
        Returns:
            Experiment 实例
        """
        from .experiment import Experiment
        
        exp = Experiment(name=name, save_dir=self.save_dir)
        
        # 更新索引
        self.experiments_index['experiments'][exp.id] = {
            'id': exp.id,
            'name': name,
            'create_time': exp.create_time,
        }
        self._save_index()
        
        self.logger.info(f"创建实验: {exp.id} ({name})")
        
        return exp
    
    def get_experiment(self, experiment_id: Optional[str] = None,
                      experiment_name: Optional[str] = None) -> Optional['Experiment']:
        """
        获取实验
        
        Args:
            experiment_id: 实验 ID
            experiment_name: 实验名称（如果有多个同名实验，返回最新的）
        
        Returns:
            Experiment 实例或 None
        """
        from .experiment import Experiment
        
        # 按 ID 查找
        if experiment_id:
            if experiment_id in self.experiments_index['experiments']:
                return Experiment.load(experiment_id, self.save_dir)
        
        # 按名称查找（返回最新的）
        if experiment_name:
            matching_exps = []
            for exp_id, exp_info in self.experiments_index['experiments'].items():
                if exp_info['name'] == experiment_name:
                    matching_exps.append((exp_id, exp_info.get('create_time', '')))
            
            if matching_exps:
                # 按创建时间排序，返回最新的
                matching_exps.sort(key=lambda x: x[1], reverse=True)
                latest_id = matching_exps[0][0]
                return Experiment.load(latest_id, self.save_dir)
        
        return None
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        列出所有实验
        
        Returns:
            实验信息列表
        """
        return list(self.experiments_index['experiments'].values())
    
    def delete_experiment(self, experiment_id: str):
        """
        删除实验
        
        Args:
            experiment_id: 实验 ID
        """
        import shutil
        
        if experiment_id in self.experiments_index['experiments']:
            # 删除目录
            exp_dir = os.path.join(self.save_dir, experiment_id)
            if os.path.exists(exp_dir):
                shutil.rmtree(exp_dir)
            
            # 更新索引
            del self.experiments_index['experiments'][experiment_id]
            self._save_index()
            
            self.logger.info(f"删除实验: {experiment_id}")
        else:
            self.logger.warning(f"实验不存在: {experiment_id}")
    
    def start_recorder(self, experiment_id: Optional[str] = None,
                      experiment_name: Optional[str] = None,
                      recorder_name: Optional[str] = None,
                      auto_create_exp: bool = True,
                      resume: bool = False):
        """
        启动记录器
        
        Args:
            experiment_id: 实验 ID
            experiment_name: 实验名称
            recorder_name: 记录器名称
            auto_create_exp: 如果实验不存在，是否自动创建
            resume: 是否恢复已有的recorder
        
        Returns:
            Recorder 实例
        """
        # 获取或创建实验
        exp = self.get_experiment(experiment_id, experiment_name)
        
        if exp is None:
            if auto_create_exp and experiment_name:
                exp = self.create_experiment(experiment_name)
            else:
                raise ValueError(f"实验不存在: {experiment_id or experiment_name}")
        
        # 创建记录器
        recorder_name = recorder_name or f"run_{len(exp.list_recorders()) + 1}"
        recorder = exp.create_recorder(recorder_name)

        # 设置为当前活动
        self.current_experiment = exp
        self.current_recorder = recorder

        # 更新索引中的 recorder_count（方便快速查看）
        try:
            exp_info = self.experiments_index['experiments'].get(exp.id, {})
            exp_info['recorder_count'] = len(exp.list_recorders())
            exp_info['name'] = exp.name
            exp_info['create_time'] = exp.create_time
            self.experiments_index['experiments'][exp.id] = exp_info
            self._save_index()
        except Exception:
            # 索引更新不是关键路径，失败时记录警告
            self.logger.warning(f"更新索引失败: {exp.id}")

        # 返回 recorder id 以便外层（如 QCRecorder）使用
        return recorder.id
    
    def end_recorder(self,
                     experiment_id: Optional[str] = None,
                     experiment_name: Optional[str] = None,
                     recorder_id: Optional[str] = None,
                     status: str = 'FINISHED'):
        """
        结束指定的记录器或当前活动记录器

        Args:
            experiment_id: 实验 ID（可选）
            experiment_name: 实验名称（可选）
            recorder_id: 记录器 ID（可选）。如果未提供且存在当前活动记录器，则结束当前记录器。
            status: 结束状态
        """
        recorder = None

        # 优先使用指定的 experiment + recorder_id
        if recorder_id:
            exp = self.get_experiment(experiment_id, experiment_name)
            if exp:
                recorder = exp.get_recorder(recorder_id)

        # 如果未找到，尝试使用当前活动记录器
        if recorder is None and self.current_recorder:
            recorder = self.current_recorder

        if recorder:
            try:
                recorder.end(status)
                self.logger.info(f"结束记录器: {recorder.id} (status={status})")
            except Exception as e:
                self.logger.error(f"结束记录器失败: {e}")

            # 如果结束的是当前活动记录器，清空引用
            if self.current_recorder and recorder.id == getattr(self.current_recorder, 'id', None):
                self.current_recorder = None
        else:
            self.logger.warning("没有找到要结束的记录器")
    
    def get_recorder(self, experiment_id: Optional[str] = None,
                    experiment_name: Optional[str] = None,
                    recorder_id: Optional[str] = None,
                    recorder_name: Optional[str] = None):
        """
        获取记录器
        
        Args:
            experiment_id: 实验 ID
            experiment_name: 实验名称
            recorder_id: 记录器 ID
            recorder_name: 记录器名称
        
        Returns:
            Recorder 实例或 None
        """
        exp = self.get_experiment(experiment_id, experiment_name)
        
        if exp is None:
            return None
        
        return exp.get_recorder(recorder_id, recorder_name)
    
    def list_recorders(self, experiment_name: str) -> Dict[str, Dict]:
        """
        列出指定实验的所有 recorders
        
        Args:
            experiment_name: 实验名称
            
        Returns:
            Dict[recorder_id, recorder_info] - recorder信息字典
        """
        exp = self.get_experiment(experiment_name=experiment_name)
        
        if exp is None:
            self.logger.warning(f"实验不存在: {experiment_name}")
            return {}
        
        # 获取所有 recorder 信息列表
        recorder_list = exp.list_recorders()
        
        # 转换为字典格式 {recorder_id: info}
        recorder_dict = {}
        for rec_info in recorder_list:
            rec_id = rec_info.get('id')
            if rec_id:
                recorder_dict[rec_id] = rec_info
        
        return recorder_dict
    
    def search_recorders(self,
                        experiment_name: Optional[str] = None,
                        status: Optional[str] = None,
                        **params) -> List[Dict]:
        """
        搜索符合条件的 recorders
        
        Args:
            experiment_name: 实验名称过滤（可选）
            status: 状态过滤（可选）
            **params: 参数过滤
            
        Returns:
            符合条件的 recorder 信息列表
        """
        results = []
        
        # 确定要搜索的实验范围
        if experiment_name:
            exp = self.get_experiment(experiment_name=experiment_name)
            experiments = [exp] if exp else []
        else:
            # 搜索所有实验
            experiments = []
            for exp_info in self.experiments_index['experiments'].values():
                exp = self.get_experiment(experiment_id=exp_info['id'])
                if exp:
                    experiments.append(exp)
        
        # 在每个实验中搜索
        for exp in experiments:
            for rec_info in exp.list_recorders():
                # 状态过滤
                if status and rec_info.get('status') != status:
                    continue
                
                # 参数过滤
                if params:
                    rec_params = rec_info.get('params', {})
                    match = all(rec_params.get(k) == v for k, v in params.items())
                    if not match:
                        continue
                
                # 添加实验信息
                rec_info['experiment_name'] = exp.name
                rec_info['experiment_id'] = exp.id
                results.append(rec_info)
        
        return results


if __name__ == '__main__':
    # 测试 ExpManager
    print("=" * 80)
    print("ExpManager 测试")
    print("=" * 80)
    
    # 创建管理器
    manager = ExpManager(save_dir='output/test_experiments')
    print(f"\n✅ 创建管理器")
    
    # 创建实验
    exp1 = manager.create_experiment('test_exp_1')
    print(f"✅ 创建实验 1: {exp1}")
    
    exp2 = manager.create_experiment('test_exp_2')
    print(f"✅ 创建实验 2: {exp2}")
    
    # 列出所有实验
    print(f"\n所有实验:")
    for exp_info in manager.list_experiments():
        print(f"  - {exp_info['name']}: {exp_info['id']}")
    
    # 启动记录器
    recorder1 = manager.start_recorder(experiment_name='test_exp_1', recorder_name='run_1')
    recorder1.log_params(lr=0.001)
    recorder1.log_metrics(loss=0.05)
    manager.end_recorder('FINISHED')
    print(f"\n✅ 记录器 1 完成")
    
    recorder2 = manager.start_recorder(experiment_name='test_exp_1', recorder_name='run_2')
    recorder2.log_params(lr=0.002)
    recorder2.log_metrics(loss=0.04)
    manager.end_recorder('FINISHED')
    print(f"✅ 记录器 2 完成")
    
    # 获取实验
    exp = manager.get_experiment(experiment_name='test_exp_1')
    print(f"\n✅ 获取实验: {exp}")
    print(f"  记录器数量: {len(exp.list_recorders())}")
    
    # 获取记录器
    rec = manager.get_recorder(experiment_name='test_exp_1', recorder_name='run_1')
    print(f"\n✅ 获取记录器: {rec}")
    print(f"  参数: {rec.params}")
    print(f"  指标: {rec.metrics}")
    
    print("\n" + "=" * 80)
    print("✅ ExpManager 测试完成")
    print("=" * 80)
