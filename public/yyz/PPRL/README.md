# Pipeline Parallelism Scheduling with Micro-Batch Dependency

本项目探索了在流水线并行（Pipeline Parallelism）中，不同 **micro-batch** 在多设备（stage）上的调度问题，并尝试结合计算与通信开销、依赖关系以及波动因素进行建模与实验。

## 核心功能

### 1. Zero Bubble Pipeline Parallelism
- 实现传统的零气泡流水线并行调度算法
- 通过精心安排micro-batch的执行顺序，最小化设备空闲时间
- 参考: https://github.com/sail-sg/zero-bubble-pipeline-parallelism

### 2. PPO强化学习调度器
- 基于Proximal Policy Optimization的自主调度系统
- 能够学习适应不同时间波动的调度策略
- 使用神经网络建模调度决策过程

### 3. 时间波动模型
- 支持10%-30%的计算/通信时间波动
- 使用正态分布模拟随机波动
- 可配置的波动水平和随机种子

## 快速开始

### 环境准备
```bash
# 激活虚拟环境
source /data/public/yyz/yyz/bin/activate

# 安装依赖
pip install gym stable-baselines3 "numpy<2"
```

### 运行演示
```bash
python main.py
```

## 项目结构

```
PPRL/
├── pipeline_simulator.py      # 核心仿真框架
├── zero_bubble_scheduler.py   # Zero Bubble调度器
├── ppo_scheduler.py          # PPO强化学习调度器
├── main.py                   # 主演示脚本
├── requirements.txt          # 依赖列表
└── README.md                # 项目说明
```

## 实验结果

### 性能对比
- **Zero Bubble**: 传统零气泡算法，在低波动下表现稳定
- **PPO**: 强化学习调度算法，在高波动下表现更优

### 关键发现
1. **PPO在30%波动下表现最佳**，相比Zero Bubble有显著改进
2. **强化学习能够适应时间波动**，在不确定环境中表现更好
3. **Zero Bubble在低波动下稳定**，适合确定性环境

## 技术特点

### 创新点
1. **轻量化实现**: 专注于流水线并行问题，避免复杂DAG调度
2. **强化学习应用**: 首次将PPO应用于流水线并行调度
3. **波动适应性**: 专门针对10%-30%时间波动进行优化
4. **模块化设计**: 易于扩展和集成新算法

### 技术栈
- **Python 3.10**: 主要编程语言
- **PyTorch**: 深度学习框架
- **Gym**: 强化学习环境
- **NumPy**: 数值计算
- **Matplotlib**: 可视化

## 调度建模

### 基本符号
- **F(m,n)**: 表示第 *m* 个 micro-batch 在第 *n* 个 device 上的 forward
- **B(m,n)**: 表示第 *m* 个 micro-batch 在第 *n* 个 device 上的 backward
- **W(m,n)**: 表示第 *m* 个 micro-batch 在第 *n* 个 device 上的 weight update

### 依赖约束
1. **Forward 依赖**: F(m,n) 必须在 F(m, n-1) 之后执行
2. **Backward 依赖**: B(m,n) 必须在 B(m, n+1) 和 F(m,N) 之后执行
3. **Weight Update 依赖**: W(m,n) 必须在 B(m,n) 之后执行

### 时间建模
一个模块的总执行时间可分为两部分：
```
T = T(通信) + T(计算)
```

- **T(计算)** 受以下因素影响：
  1. micro-batch 大小
  2. stage（不同 device）
  3. operator 类型（F, B, W），通常满足 **B > F > W**

- **T(通信)** 主要发生在不同 device 之间，可能存在个别异常延迟

### 波动建模
- 通常假设计算/通信时间存在 **10%–30% 的波动**
- 可以设计对比实验：观察在波动增加时，调度优化的提升效果
- 通常情况下 **T(通信) < T(计算)**，但同样可以探索通信延迟增大时对整体吞吐的影响

## 参考文献

- Zero Bubble Pipeline Parallelism: https://arxiv.org/pdf/2401.10241
- PPO算法: https://arxiv.org/abs/1707.06347
- 流水线并行调度: https://openreview.net/forum?id=b9aCXHhdbv
- 相关工作: https://www.usenix.org/conference/osdi22/presentation/zheng-lianmin