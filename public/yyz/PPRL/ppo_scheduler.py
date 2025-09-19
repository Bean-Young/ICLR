"""
PPO-based Pipeline Parallelism Scheduler
基于PPO强化学习的流水线并行调度系统

核心思想：使用强化学习训练智能体，学习最优的调度策略
能够适应不同的时间波动情况，在动态环境中表现更好
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Dict, Tuple, Optional
from pipeline_simulator import PipelineSimulator, Operation, OperationType
import gym
from gym import spaces


class PipelineEnvironment(gym.Env):
    """
    流水线调度强化学习环境
    
    状态空间：当前调度状态的特征向量
    动作空间：选择要执行的操作
    奖励函数：基于调度效率设计的奖励机制
    """
    
    def __init__(self, simulator: PipelineSimulator):
        """
        初始化环境
        
        Args:
            simulator: 流水线并行仿真器实例
        """
        super().__init__()
        self.simulator = simulator
        self.operations = simulator.create_operations()
        self.num_operations = len(self.operations)
        
        # 动作空间：选择要执行的操作（操作索引）
        self.action_space = spaces.Discrete(self.num_operations)
        
        # 状态空间：当前调度状态的特征向量
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self._get_state_dim(),), dtype=np.float32
        )
        
        # 环境状态变量
        self.current_step = 0
        self.completed_ops = set()  # 已完成的操作
        self.scheduled_ops = []     # 已调度的操作
        self.current_time = 0.0
        self.episode_reward = 0.0
    
    def _get_state_dim(self) -> int:
        """
        计算状态空间维度
        
        状态包括：
        1. 每个操作的状态（是否完成、是否可执行、是否已调度）
        2. 每个设备的负载状态
        3. 当前时间
        4. 操作类型分布
        
        Returns:
            int: 状态空间维度
        """
        return (self.num_operations * 3 +  # 操作状态：完成、可执行、已调度
                self.simulator.num_devices +  # 设备负载
                1 +  # 当前时间
                3)  # 操作类型计数（Forward、Backward、Weight Update）
    
    def _get_state(self) -> np.ndarray:
        """
        获取当前环境状态
        
        Returns:
            np.ndarray: 状态向量
        """
        state = np.zeros(self._get_state_dim(), dtype=np.float32)
        idx = 0
        
        # 1. 操作状态编码
        for op in self.operations:
            op_key = (op.micro_batch_id, op.device_id, op.op_type)
            
            # 是否完成
            state[idx] = 1.0 if op_key in self.completed_ops else 0.0
            idx += 1
            
            # 是否可执行
            state[idx] = 1.0 if self._is_ready_to_execute(op) else 0.0
            idx += 1
            
            # 是否已调度
            state[idx] = 1.0 if op in self.scheduled_ops else 0.0
            idx += 1
        
        # 2. 设备负载状态
        for device_id in range(self.simulator.num_devices):
            device_ops = [op for op in self.scheduled_ops if op.device_id == device_id]
            total_duration = sum(op.duration for op in device_ops)
            state[idx] = min(1.0, total_duration / 100.0)  # 归一化到[0,1]
            idx += 1
        
        # 3. 当前时间（归一化）
        state[idx] = min(1.0, self.current_time / 1000.0)
        idx += 1
        
        # 4. 操作类型分布
        type_counts = {op_type: 0 for op_type in OperationType}
        for op in self.scheduled_ops:
            type_counts[op.op_type] += 1
        
        for op_type in OperationType:
            state[idx] = type_counts[op_type] / self.num_operations
            idx += 1
        
        return state
    
    def _is_ready_to_execute(self, operation: Operation) -> bool:
        """
        检查操作是否可以执行（依赖是否满足）
        
        Args:
            operation: 要检查的操作
            
        Returns:
            bool: 是否可以执行
        """
        for dep_m, dep_n, dep_type in operation.dependencies:
            dep_key = (dep_m, dep_n, dep_type)
            if dep_key not in self.completed_ops:
                return False
        return True
    
    def _get_valid_actions(self) -> List[int]:
        """
        获取有效的动作（可执行且未调度的操作）
        
        Returns:
            List[int]: 有效动作索引列表
        """
        valid_actions = []
        for i, op in enumerate(self.operations):
            if (op not in self.scheduled_ops and 
                self._is_ready_to_execute(op)):
                valid_actions.append(i)
        return valid_actions
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作，返回新状态、奖励、是否结束、额外信息
        
        Args:
            action: 要执行的动作（操作索引）
            
        Returns:
            Tuple: (新状态, 奖励, 是否结束, 额外信息)
        """
        if self.current_step >= self.num_operations:
            return self._get_state(), 0.0, True, {}
        
        # 检查动作是否有效
        valid_actions = self._get_valid_actions()
        if not valid_actions:
            # 没有有效动作，等待
            self.current_time += 0.1
            return self._get_state(), -0.1, False, {}
        
        if action not in valid_actions:
            # 无效动作，给予惩罚
            return self._get_state(), -1.0, False, {}
        
        # 执行动作：调度操作
        operation = self.operations[action]
        self._schedule_operation(operation)
        
        # 计算奖励
        reward = self._calculate_reward(operation)
        self.episode_reward += reward
        
        # 检查是否完成
        done = len(self.completed_ops) >= self.num_operations
        
        self.current_step += 1
        
        return self._get_state(), reward, done, {}
    
    def _schedule_operation(self, operation: Operation):
        """
        调度操作执行
        
        Args:
            operation: 要调度的操作
        """
        # 计算开始时间
        start_time = self._calculate_earliest_start_time(operation)
        
        # 计算持续时间
        duration = self.simulator.calculate_operation_duration(operation)
        
        # 更新操作信息
        operation.start_time = start_time
        operation.duration = duration
        operation.end_time = start_time + duration
        
        # 添加到调度表
        self.scheduled_ops.append(operation)
        self.completed_ops.add((operation.micro_batch_id, operation.device_id, 
                               operation.op_type))
        
        # 更新当前时间
        self.current_time = max(self.current_time, operation.end_time)
    
    def _calculate_earliest_start_time(self, operation: Operation) -> float:
        """
        计算操作的最早开始时间
        
        Args:
            operation: 要计算的操作
            
        Returns:
            float: 最早开始时间
        """
        earliest_start = self.current_time
        
        # 检查设备是否被占用
        device_busy_until = 0.0
        for scheduled_op in self.scheduled_ops:
            if (scheduled_op.device_id == operation.device_id and 
                scheduled_op.end_time > device_busy_until):
                device_busy_until = scheduled_op.end_time
        
        earliest_start = max(earliest_start, device_busy_until)
        
        # 检查依赖关系
        for dep_m, dep_n, dep_type in operation.dependencies:
            dep_key = (dep_m, dep_n, dep_type)
            if dep_key in self.completed_ops:
                # 找到依赖操作的完成时间
                for scheduled_op in self.scheduled_ops:
                    if ((scheduled_op.micro_batch_id, scheduled_op.device_id, 
                         scheduled_op.op_type) == dep_key):
                        earliest_start = max(earliest_start, scheduled_op.end_time)
                        break
        
        return earliest_start
    
    def _calculate_reward(self, operation: Operation) -> float:
        """
        计算奖励函数
        
        奖励设计原则：
        1. 基础奖励：成功调度操作
        2. 时间效率：选择能减少总执行时间的操作
        3. 负载均衡：选择负载较轻的设备
        4. 依赖链：选择能解锁更多操作的操作
        
        Args:
            operation: 刚调度的操作
            
        Returns:
            float: 奖励值
        """
        reward = 0.0
        
        # 1. 基础奖励：成功调度操作
        reward += 1.0
        
        # 2. 时间效率奖励：选择能减少总执行时间的操作
        time_efficiency = self._calculate_time_efficiency(operation)
        reward += time_efficiency * 0.5
        
        # 3. 负载均衡奖励：选择负载较轻的设备
        load_balance = self._calculate_load_balance(operation)
        reward += load_balance * 0.3
        
        # 4. 依赖链奖励：选择能解锁更多操作的操作
        dependency_chain = self._calculate_dependency_chain(operation)
        reward += dependency_chain * 0.2
        
        return reward
    
    def _calculate_time_efficiency(self, operation: Operation) -> float:
        """
        计算时间效率奖励
        
        Args:
            operation: 操作
            
        Returns:
            float: 时间效率分数
        """
        # 简化的时间效率计算：持续时间越短，效率越高
        duration = operation.duration
        return 1.0 / (duration + 0.1)
    
    def _calculate_load_balance(self, operation: Operation) -> float:
        """
        计算负载均衡奖励
        
        Args:
            operation: 操作
            
        Returns:
            float: 负载均衡分数
        """
        device_id = operation.device_id
        device_load = sum(op.duration for op in self.scheduled_ops 
                         if op.device_id == device_id)
        avg_load = np.mean([sum(op.duration for op in self.scheduled_ops 
                              if op.device_id == d) 
                           for d in range(self.simulator.num_devices)])
        
        if avg_load > 0:
            return 1.0 / (abs(device_load - avg_load) + 0.1)
        return 1.0
    
    def _calculate_dependency_chain(self, operation: Operation) -> float:
        """
        计算依赖链解锁效果
        
        Args:
            operation: 操作
            
        Returns:
            float: 依赖链分数
        """
        # 计算这个操作能解锁多少个后续操作
        unlocked_count = 0
        for op in self.operations:
            if (op not in self.scheduled_ops and 
                op != operation and
                self._is_ready_to_execute(op)):
                unlocked_count += 1
        
        return min(1.0, unlocked_count / 10.0)
    
    def reset(self) -> np.ndarray:
        """
        重置环境到初始状态
        
        Returns:
            np.ndarray: 初始状态
        """
        self.current_step = 0
        self.completed_ops = set()
        self.scheduled_ops = []
        self.current_time = 0.0
        self.episode_reward = 0.0
        return self._get_state()
    
    def render(self, mode='human'):
        """
        渲染环境状态（用于调试）
        
        Args:
            mode: 渲染模式
        """
        if mode == 'human':
            print(f"Step: {self.current_step}, Time: {self.current_time:.2f}, "
                  f"Completed: {len(self.completed_ops)}/{self.num_operations}")


class PPOAgent(nn.Module):
    """
    PPO智能体网络
    
    包含策略网络和价值网络：
    - 策略网络：输出动作概率分布
    - 价值网络：估计状态价值
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        初始化PPO智能体
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        # 策略网络：输出动作概率分布
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # 价值网络：估计状态价值
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: 输入状态
            
        Returns:
            Tuple: (动作概率分布, 状态价值)
        """
        action_probs = self.policy_net(state)
        value = self.value_net(state)
        return action_probs, value
    
    def get_action(self, state, valid_actions=None):
        """
        根据状态获取动作
        
        Args:
            state: 当前状态
            valid_actions: 有效动作列表
            
        Returns:
            Tuple: (动作, log概率, 状态价值)
        """
        with torch.no_grad():
            action_probs, value = self.forward(state)
            
            if valid_actions is not None:
                # 只考虑有效动作
                valid_probs = action_probs[valid_actions]
                valid_probs = valid_probs / valid_probs.sum()
                dist = Categorical(valid_probs)
                action_idx = dist.sample()
                action = valid_actions[action_idx.item()]
                # 计算原始概率分布中的log概率
                log_prob = torch.log(action_probs[action] + 1e-8)
            else:
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            return action, log_prob, value


class PPOScheduler:
    """
    基于PPO的调度器
    
    使用强化学习训练智能体，学习最优的调度策略
    能够适应不同的时间波动情况
    """
    
    def __init__(self, simulator: PipelineSimulator, 
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 eps_clip: float = 0.2,
                 k_epochs: int = 4):
        """
        初始化PPO调度器
        
        Args:
            simulator: 流水线并行仿真器
            learning_rate: 学习率
            gamma: 折扣因子
            eps_clip: PPO裁剪参数
            k_epochs: 每次更新的训练轮数
        """
        self.simulator = simulator
        self.env = PipelineEnvironment(simulator)
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        self.agent = PPOAgent(state_dim, action_dim)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # 经验缓冲区
        self.memory = []
    
    def train(self, num_episodes: int = 1000, update_frequency: int = 10):
        """
        训练PPO智能体
        
        Args:
            num_episodes: 训练轮数
            update_frequency: 更新频率
        """
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # 获取有效动作
                valid_actions = self.env._get_valid_actions()
                
                if not valid_actions:
                    # 没有有效动作，等待
                    state, reward, done, _ = self.env.step(0)
                    episode_reward += reward
                    continue
                
                # 获取动作
                action, log_prob, value = self.agent.get_action(
                    torch.FloatTensor(state), valid_actions
                )
                
                # 执行动作
                next_state, reward, done, _ = self.env.step(action)
                
                # 存储经验
                self.memory.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'log_prob': log_prob,
                    'value': value,
                    'done': done
                })
                
                state = next_state
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            
            # 定期更新网络
            if episode % update_frequency == 0 and len(self.memory) > 0:
                self._update_network()
                self.memory = []
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
        
        return episode_rewards
    
    def _update_network(self):
        """
        更新网络参数（PPO算法）
        """
        if len(self.memory) < 10:
            return
        
        # 计算折扣奖励
        rewards = []
        discounted_reward = 0
        for exp in reversed(self.memory):
            if exp['done']:
                discounted_reward = 0
            discounted_reward = exp['reward'] + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        
        # 标准化奖励
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # 准备数据
        states = torch.FloatTensor([exp['state'] for exp in self.memory])
        actions = torch.LongTensor([exp['action'] for exp in self.memory])
        old_log_probs = torch.FloatTensor([exp['log_prob'] for exp in self.memory])
        old_values = torch.FloatTensor([exp['value'] for exp in self.memory])
        
        # PPO更新
        for _ in range(self.k_epochs):
            # 前向传播
            action_probs, values = self.agent(states)
            
            # 计算优势
            advantages = rewards - values.squeeze()
            
            # 计算新的log概率
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            # 计算概率比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 计算PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = nn.MSELoss()(values.squeeze(), rewards)
            
            # 总损失
            total_loss = policy_loss + 0.5 * value_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
    
    def schedule(self) -> Dict:
        """
        使用训练好的智能体进行调度
        
        Returns:
            Dict: 调度结果
        """
        state = self.env.reset()
        done = False
        
        while not done:
            valid_actions = self.env._get_valid_actions()
            
            if not valid_actions:
                state, _, done, _ = self.env.step(0)
                continue
            
            with torch.no_grad():
                action, _, _ = self.agent.get_action(
                    torch.FloatTensor(state), valid_actions
                )
            
            state, _, done, _ = self.env.step(action)
        
        # 更新simulator的调度结果
        self.simulator.schedule = self.env.scheduled_ops
        self.simulator.current_time = self.env.current_time
        self.simulator.calculate_statistics()
        
        return self.simulator.get_schedule_summary()