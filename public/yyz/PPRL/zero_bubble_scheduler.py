"""
Zero Bubble Pipeline Parallelism Scheduler
实现零气泡流水线并行调度算法

核心思想：通过精心安排micro-batch的执行顺序，最小化设备空闲时间（bubble time）
"""

import numpy as np
from typing import List, Dict, Tuple
from pipeline_simulator import PipelineSimulator, Operation, OperationType


class ZeroBubbleScheduler:
    """
    Zero Bubble Pipeline Parallelism调度器
    
    算法原理：
    1. 优先选择能最早开始执行的操作
    2. 在相同开始时间下，选择持续时间最短的操作
    3. 确保满足所有依赖关系约束
    4. 最小化设备空闲时间
    """
    
    def __init__(self, simulator: PipelineSimulator):
        """
        初始化调度器
        
        Args:
            simulator: 流水线并行仿真器实例
        """
        self.simulator = simulator
    
    def schedule(self) -> Dict:
        """
        执行Zero Bubble调度算法
        
        Returns:
            Dict: 包含调度结果的字典，包括总时间、吞吐量、设备利用率等
        """
        # 创建所有需要调度的操作
        operations = self.simulator.create_operations()
        
        # 初始化调度状态
        self.simulator.schedule = []  # 清空之前的调度结果
        self.simulator.current_time = 0.0
        completed_ops = set()  # 已完成的操作集合
        
        # 按设备分组操作，便于管理
        device_queues = {i: [] for i in range(self.simulator.num_devices)}
        for op in operations:
            device_queues[op.device_id].append(op)
        
        # 为每个设备的操作按micro-batch ID排序，确保顺序执行
        for device_id in range(self.simulator.num_devices):
            device_ops = device_queues[device_id]
            device_ops.sort(key=lambda x: x.micro_batch_id)
        
        # 主调度循环：持续调度直到所有操作完成
        while len(completed_ops) < len(operations):
            # 找到当前可以执行的操作（依赖已满足且未调度）
            ready_ops = self.simulator.get_ready_operations(operations, completed_ops)
            
            if not ready_ops:
                # 如果没有可执行的操作，等待下一个操作完成
                if self.simulator.schedule:
                    next_completion = min(op.end_time for op in self.simulator.schedule 
                                        if op not in [op for op in self.simulator.schedule 
                                                    if (op.micro_batch_id, op.device_id, op.op_type) in completed_ops])
                    self.simulator.current_time = next_completion
                continue
            
            # Zero Bubble策略：选择最优的操作组合
            best_ops = self._select_best_operations(ready_ops, completed_ops)
            
            # 调度选中的操作
            for op in best_ops:
                self.simulator.schedule_operation(op, self.simulator.current_time)
                completed_ops.add((op.micro_batch_id, op.device_id, op.op_type))
        
        # 计算最终统计信息
        self.simulator.calculate_statistics()
        return self.simulator.get_schedule_summary()
    
    def _select_best_operations(self, ready_ops: List[Operation], 
                              completed_ops: set) -> List[Operation]:
        """
        选择最优的操作组合
        
        Zero Bubble核心策略：
        1. 选择能最早开始执行的操作
        2. 在相同开始时间下，选择持续时间最短的操作
        3. 避免设备冲突（同一设备同时只能执行一个操作）
        
        Args:
            ready_ops: 当前可执行的操作列表
            completed_ops: 已完成的操作集合
            
        Returns:
            List[Operation]: 选中的最优操作列表
        """
        if not ready_ops:
            return []
        
        # 按设备分组，避免同一设备同时执行多个操作
        device_ops = {}
        for op in ready_ops:
            if op.device_id not in device_ops:
                device_ops[op.device_id] = []
            device_ops[op.device_id].append(op)
        
        selected_ops = []
        
        # 为每个设备选择最优操作
        for device_id, ops in device_ops.items():
            if not ops:
                continue
            
            # 选择能最早开始且持续时间最短的操作
            best_op = min(ops, key=lambda x: (
                self._calculate_earliest_start_time(x, completed_ops),  # 最早开始时间
                self.simulator.calculate_operation_duration(x)          # 持续时间
            ))
            selected_ops.append(best_op)
        
        return selected_ops
    
    def _calculate_earliest_start_time(self, operation: Operation, 
                                     completed_ops: set) -> float:
        """
        计算操作的最早开始时间
        
        考虑因素：
        1. 设备是否被其他操作占用
        2. 依赖操作是否已完成
        
        Args:
            operation: 要计算的操作
            completed_ops: 已完成的操作集合
            
        Returns:
            float: 操作的最早开始时间
        """
        earliest_start = self.simulator.current_time
        
        # 检查设备是否被占用
        device_busy_until = 0.0
        for scheduled_op in self.simulator.schedule:
            if (scheduled_op.device_id == operation.device_id and 
                scheduled_op.end_time > device_busy_until):
                device_busy_until = scheduled_op.end_time
        
        earliest_start = max(earliest_start, device_busy_until)
        
        # 检查依赖关系
        for dep_m, dep_n, dep_type in operation.dependencies:
            dep_key = (dep_m, dep_n, dep_type)
            if dep_key in completed_ops:
                # 找到依赖操作的完成时间
                for scheduled_op in self.simulator.schedule:
                    if ((scheduled_op.micro_batch_id, scheduled_op.device_id, 
                         scheduled_op.op_type) == dep_key):
                        earliest_start = max(earliest_start, scheduled_op.end_time)
                        break
        
        return earliest_start