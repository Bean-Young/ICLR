"""
Pipeline Parallelism Simulator
流水线并行调度的核心仿真框架

实现功能：
1. 流水线并行调度的基本建模
2. 操作依赖关系管理
3. 时间波动模拟
4. 性能统计计算
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time


class OperationType(Enum):
    """操作类型枚举"""
    FORWARD = "F"        # 前向传播
    BACKWARD = "B"       # 反向传播
    WEIGHT_UPDATE = "W"  # 权重更新


@dataclass
class Operation:
    """
    表示一个操作的数据结构
    
    包含操作的基本信息和调度状态
    """
    op_type: OperationType                    # 操作类型
    micro_batch_id: int                      # micro-batch ID
    device_id: int                           # 设备ID
    start_time: float = 0.0                  # 开始时间
    end_time: float = 0.0                    # 结束时间
    duration: float = 0.0                    # 持续时间
    dependencies: List[Tuple[int, int, OperationType]] = None  # 依赖关系
    
    def __post_init__(self):
        """初始化后处理"""
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class DeviceConfig:
    """
    设备配置信息
    
    包含设备的计算能力和通信带宽
    """
    device_id: int
    compute_capability: float = 1.0      # 计算能力系数
    communication_bandwidth: float = 1.0  # 通信带宽系数


class TimeModel:
    """
    时间模型，包含计算和通信时间建模
    
    支持时间波动模拟，用于测试不同调度算法在不确定性环境下的表现
    """
    
    def __init__(self, base_compute_time: float = 1.0, base_comm_time: float = 0.5, 
                 noise_level: float = 0.2, seed: int = 42):
        """
        初始化时间模型
        
        Args:
            base_compute_time: 基础计算时间
            base_comm_time: 基础通信时间
            noise_level: 波动水平 (0.1-0.3 对应10%-30%波动)
            seed: 随机种子
        """
        self.base_compute_time = base_compute_time
        self.base_comm_time = base_comm_time
        self.noise_level = noise_level
        self.rng = np.random.RandomState(seed)
        
        # 不同操作类型的计算时间比例
        # 根据实际经验：Backward > Forward > Weight Update
        self.compute_ratios = {
            OperationType.FORWARD: 1.0,
            OperationType.BACKWARD: 1.5,      # Backward比Forward慢50%
            OperationType.WEIGHT_UPDATE: 0.3  # Weight Update最快
        }
    
    def get_compute_time(self, op_type: OperationType, micro_batch_size: int, 
                        device_config: DeviceConfig) -> float:
        """
        计算操作的计算时间
        
        Args:
            op_type: 操作类型
            micro_batch_size: micro-batch大小
            device_config: 设备配置
            
        Returns:
            float: 计算时间（包含波动）
        """
        # 基础计算时间 = 基础时间 × 操作类型系数 × micro-batch大小 / 设备计算能力
        base_time = (self.base_compute_time * 
                    self.compute_ratios[op_type] * 
                    micro_batch_size / 
                    device_config.compute_capability)
        
        # 添加随机波动（正态分布）
        noise = self.rng.normal(0, self.noise_level)
        return max(0.1, base_time * (1 + noise))
    
    def get_communication_time(self, data_size: int, device_config: DeviceConfig) -> float:
        """
        计算通信时间
        
        Args:
            data_size: 数据大小
            device_config: 设备配置
            
        Returns:
            float: 通信时间（包含波动）
        """
        # 基础通信时间 = 基础时间 × 数据大小 / 设备通信带宽
        base_time = self.base_comm_time * data_size / device_config.communication_bandwidth
        
        # 添加随机波动
        noise = self.rng.normal(0, self.noise_level)
        return max(0.05, base_time * (1 + noise))


class PipelineSimulator:
    """
    流水线并行仿真器
    
    核心功能：
    1. 创建和管理所有操作
    2. 维护操作依赖关系
    3. 计算操作持续时间
    4. 统计性能指标
    """
    
    def __init__(self, num_devices: int, num_micro_batches: int, 
                 micro_batch_size: int = 32, time_model: TimeModel = None):
        """
        初始化仿真器
        
        Args:
            num_devices: 设备数量
            num_micro_batches: micro-batch数量
            micro_batch_size: micro-batch大小
            time_model: 时间模型
        """
        self.num_devices = num_devices
        self.num_micro_batches = num_micro_batches
        self.micro_batch_size = micro_batch_size
        self.time_model = time_model or TimeModel()
        
        # 初始化设备配置
        self.devices = [DeviceConfig(i) for i in range(num_devices)]
        
        # 操作调度表
        self.schedule: List[Operation] = []
        self.current_time = 0.0
        
        # 统计信息
        self.stats = {
            'total_time': 0.0,
            'device_utilization': [0.0] * num_devices,
            'bubble_time': 0.0,
            'throughput': 0.0
        }
    
    def create_operations(self) -> List[Operation]:
        """
        创建所有需要的操作
        
        根据流水线并行的依赖关系创建操作：
        1. Forward操作：F(m,n) 依赖 F(m, n-1)
        2. Backward操作：B(m,n) 依赖 B(m, n+1) 和 F(m,N)
        3. Weight Update操作：W(m,n) 依赖 B(m,n)
        
        Returns:
            List[Operation]: 所有操作的列表
        """
        operations = []
        
        for m in range(self.num_micro_batches):
            for n in range(self.num_devices):
                # 1. Forward操作
                forward_deps = []
                if n > 0:  # F(m,n) 依赖 F(m, n-1)
                    forward_deps.append((m, n-1, OperationType.FORWARD))
                
                forward_op = Operation(
                    op_type=OperationType.FORWARD,
                    micro_batch_id=m,
                    device_id=n,
                    dependencies=forward_deps
                )
                operations.append(forward_op)
                
                # 2. Backward操作
                backward_deps = []
                if n < self.num_devices - 1:  # B(m,n) 依赖 B(m, n+1)
                    backward_deps.append((m, n+1, OperationType.BACKWARD))
                # B(m,n) 依赖所有设备上该micro-batch的Forward操作都完成
                for device_id in range(self.num_devices):
                    backward_deps.append((m, device_id, OperationType.FORWARD))
                
                backward_op = Operation(
                    op_type=OperationType.BACKWARD,
                    micro_batch_id=m,
                    device_id=n,
                    dependencies=backward_deps
                )
                operations.append(backward_op)
                
                # 3. Weight Update操作
                weight_update_deps = [(m, n, OperationType.BACKWARD)]  # W(m,n) 依赖 B(m,n)
                
                weight_update_op = Operation(
                    op_type=OperationType.WEIGHT_UPDATE,
                    micro_batch_id=m,
                    device_id=n,
                    dependencies=weight_update_deps
                )
                operations.append(weight_update_op)
        
        return operations
    
    def calculate_operation_duration(self, operation: Operation) -> float:
        """
        计算操作持续时间
        
        Args:
            operation: 要计算的操作
            
        Returns:
            float: 操作持续时间
        """
        device = self.devices[operation.device_id]
        
        if operation.op_type == OperationType.WEIGHT_UPDATE:
            # Weight update只涉及计算，不涉及通信
            return self.time_model.get_compute_time(
                operation.op_type, self.micro_batch_size, device
            )
        else:
            # Forward和Backward涉及计算和通信
            compute_time = self.time_model.get_compute_time(
                operation.op_type, self.micro_batch_size, device
            )
            comm_time = self.time_model.get_communication_time(
                self.micro_batch_size, device
            )
            return compute_time + comm_time
    
    def is_ready_to_execute(self, operation: Operation, completed_ops: set) -> bool:
        """
        检查操作是否可以执行（依赖是否满足）
        
        Args:
            operation: 要检查的操作
            completed_ops: 已完成的操作集合
            
        Returns:
            bool: 是否可以执行
        """
        for dep_m, dep_n, dep_type in operation.dependencies:
            dep_key = (dep_m, dep_n, dep_type)
            if dep_key not in completed_ops:
                return False
        return True
    
    def get_ready_operations(self, operations: List[Operation], 
                           completed_ops: set) -> List[Operation]:
        """
        获取当前可以执行的操作
        
        Args:
            operations: 所有操作列表
            completed_ops: 已完成的操作集合
            
        Returns:
            List[Operation]: 可执行的操作列表
        """
        ready_ops = []
        for op in operations:
            if (op not in self.schedule and 
                self.is_ready_to_execute(op, completed_ops)):
                ready_ops.append(op)
        return ready_ops
    
    def schedule_operation(self, operation: Operation, start_time: float):
        """
        调度操作执行
        
        Args:
            operation: 要调度的操作
            start_time: 开始时间
        """
        operation.start_time = start_time
        operation.duration = self.calculate_operation_duration(operation)
        operation.end_time = start_time + operation.duration
        self.schedule.append(operation)
        self.current_time = max(self.current_time, operation.end_time)
    
    def calculate_statistics(self):
        """
        计算统计信息
        
        包括：
        1. 总执行时间
        2. 设备利用率
        3. 吞吐量
        4. Bubble time
        """
        if not self.schedule:
            return
        
        # 1. 总时间
        self.stats['total_time'] = max(op.end_time for op in self.schedule)
        
        # 2. 设备利用率
        for device_id in range(self.num_devices):
            device_ops = [op for op in self.schedule if op.device_id == device_id]
            if device_ops:
                total_work_time = sum(op.duration for op in device_ops)
                self.stats['device_utilization'][device_id] = total_work_time / self.stats['total_time']
        
        # 3. 吞吐量 (micro-batches per second)
        if self.stats['total_time'] > 0:
            self.stats['throughput'] = self.num_micro_batches / self.stats['total_time']
        else:
            self.stats['throughput'] = 0.0
        
        # 4. 计算bubble time (设备空闲时间)
        self.calculate_bubble_time()
    
    def calculate_bubble_time(self):
        """
        计算bubble time
        
        Bubble time = 总可能时间 - 实际工作时间
        总可能时间 = 设备数量 × 总执行时间
        实际工作时间 = 所有操作的总持续时间
        """
        total_possible_time = self.num_devices * self.stats['total_time']
        total_work_time = sum(op.duration for op in self.schedule)
        self.stats['bubble_time'] = total_possible_time - total_work_time
    
    def get_schedule_summary(self) -> Dict:
        """
        获取调度摘要
        
        Returns:
            Dict: 包含各种性能指标的字典
        """
        return {
            'total_operations': len(self.schedule),
            'total_time': self.stats['total_time'],
            'device_utilization': self.stats['device_utilization'],
            'average_utilization': np.mean(self.stats['device_utilization']),
            'bubble_time': self.stats['bubble_time'],
            'bubble_ratio': self.stats['bubble_time'] / (self.num_devices * self.stats['total_time']),
            'throughput': self.stats['throughput']
        }