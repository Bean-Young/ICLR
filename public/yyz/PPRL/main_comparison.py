#!/usr/bin/env python3
"""
Pipeline Parallelism Scheduling Comparison Demo
流水线并行调度对比演示脚本

对比两种训练模式：
1. 每次重新训练PPO（原始方法）
2. 使用预训练PPO模型（不重新训练）

在10%-30%时间波动下的性能表现
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from pipeline_simulator import PipelineSimulator, TimeModel
from zero_bubble_scheduler import ZeroBubbleScheduler
from ppo_scheduler import PPOScheduler


def train_ppo_model(simulator, model_path="ppo_model.pth", num_episodes=200):
    """
    训练PPO模型并保存
    
    Args:
        simulator: 流水线仿真器
        model_path: 模型保存路径
        num_episodes: 训练轮数
    """
    print(f"Training PPO model for {num_episodes} episodes...")
    
    ppo_scheduler = PPOScheduler(simulator)
    episode_rewards = ppo_scheduler.train(num_episodes=num_episodes, update_frequency=20)
    
    # 保存模型
    torch.save(ppo_scheduler.agent.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return ppo_scheduler, episode_rewards


def load_ppo_model(simulator, model_path="ppo_model.pth"):
    """
    加载预训练的PPO模型
    
    Args:
        simulator: 流水线仿真器
        model_path: 模型路径
        
    Returns:
        PPOScheduler: 加载了预训练模型的调度器
    """
    ppo_scheduler = PPOScheduler(simulator)
    
    if os.path.exists(model_path):
        ppo_scheduler.agent.load_state_dict(torch.load(model_path))
        print(f"Loaded pre-trained model from {model_path}")
    else:
        print(f"Model file {model_path} not found, using untrained model")
    
    return ppo_scheduler


def run_comparison_with_training_modes():
    """
    运行对比实验：5种方法对比
    
    测试不同噪声水平下五种方法的性能：
    1. Zero Bubble（基准）
    2. 预训练100轮
    3. 重新训练50轮
    4. 预训练200轮
    5. 重新训练100轮
    """
    print("Pipeline Parallelism Scheduling Comparison")
    print("=" * 100)
    print("Algorithms: Zero Bubble vs Pre-train(100) vs Retrain(50) vs Pre-train(200) vs Retrain(100)")
    print("=" * 100)
    
    # 实验配置
    num_devices = 4
    num_micro_batches = 6
    micro_batch_size = 32
    noise_levels = [0.0, 0.1, 0.2, 0.3]  # 0%, 10%, 20%, 30%波动
    num_trials = 5  # 减少试验次数以加快速度
    
    # 存储结果
    results = {
        'noise_levels': noise_levels,
        'zero_bubble_times': [],
        'pretrain_100_times': [],
        'retrain_50_times': [],
        'pretrain_200_times': [],
        'retrain_100_times': [],
        'zero_bubble_throughputs': [],
        'pretrain_100_throughputs': [],
        'retrain_50_throughputs': [],
        'pretrain_200_throughputs': [],
        'retrain_100_throughputs': []
    }
    
    # 为每个噪声水平训练预训练模型（100轮和200轮）
    print("\n" + "="*50)
    print("STEP 1: Training pre-trained models for each noise level")
    print("="*50)
    
    pretrained_models_100 = {}
    pretrained_models_200 = {}
    
    for noise_level in noise_levels:
        print(f"\nTraining pre-trained models for {noise_level*100:.0f}% noise...")
        
        time_model = TimeModel(
            base_compute_time=1.0,
            base_comm_time=0.5,
            noise_level=noise_level,
            seed=42
        )
        
        simulator = PipelineSimulator(
            num_devices=num_devices,
            num_micro_batches=num_micro_batches,
            micro_batch_size=micro_batch_size,
            time_model=time_model
        )
        
        # 训练100轮预训练模型
        model_path_100 = f"/data/public/yyz/PPRL/ppo_model_{int(noise_level*100)}_100.pth"
        ppo_scheduler_100, training_rewards_100 = train_ppo_model(
            simulator, 
            model_path=model_path_100,
            num_episodes=100
        )
        pretrained_models_100[noise_level] = model_path_100
        print(f"100-episode model saved to {model_path_100}")
        print(f"100-episode training completed. Final average reward: {np.mean(training_rewards_100[-10:]):.2f}")
        
        # 训练200轮预训练模型
        model_path_200 = f"/data/public/yyz/PPRL/ppo_model_{int(noise_level*100)}_200.pth"
        ppo_scheduler_200, training_rewards_200 = train_ppo_model(
            simulator, 
            model_path=model_path_200,
            num_episodes=200
        )
        pretrained_models_200[noise_level] = model_path_200
        print(f"200-episode model saved to {model_path_200}")
        print(f"200-episode training completed. Final average reward: {np.mean(training_rewards_200[-10:]):.2f}")
    
    # 对每个噪声水平进行测试
    for noise_level in noise_levels:
        print(f"\n" + "="*50)
        print(f"Testing with {noise_level*100:.0f}% noise level...")
        print(f"Running {num_trials} trials for each algorithm...")
        print("="*50)
        
        # 存储多次试验的结果
        zb_times = []
        pretrain_100_times = []
        retrain_50_times = []
        pretrain_200_times = []
        retrain_100_times = []
        zb_throughputs = []
        pretrain_100_throughputs = []
        retrain_50_throughputs = []
        pretrain_200_throughputs = []
        retrain_100_throughputs = []
        
        for trial in range(num_trials):
            print(f"  Trial {trial + 1}/{num_trials}:")
            
            # 创建时间模型（使用不同的种子）
            time_model = TimeModel(
                base_compute_time=1.0,
                base_comm_time=0.5,
                noise_level=noise_level,
                seed=42 + trial
            )
            
            # 1. Zero Bubble调度（基准）
            print("    Running Zero Bubble scheduler...")
            zb_sim = PipelineSimulator(
                num_devices=num_devices,
                num_micro_batches=num_micro_batches,
                micro_batch_size=micro_batch_size,
                time_model=time_model
            )
            zb_scheduler = ZeroBubbleScheduler(zb_sim)
            zb_result = zb_scheduler.schedule()
            
            zb_times.append(zb_result['total_time'])
            zb_throughputs.append(zb_result['throughput'])
            
            # 2. 预训练100轮
            print("    Running PPO scheduler (pre-trained 100)...")
            pretrain_100_sim = PipelineSimulator(
                num_devices=num_devices,
                num_micro_batches=num_micro_batches,
                micro_batch_size=micro_batch_size,
                time_model=time_model
            )
            pretrain_100_scheduler = load_ppo_model(pretrain_100_sim, pretrained_models_100[noise_level])
            pretrain_100_result = pretrain_100_scheduler.schedule()
            
            pretrain_100_times.append(pretrain_100_result['total_time'])
            pretrain_100_throughputs.append(pretrain_100_result['throughput'])
            
            # 3. 重新训练50轮
            print("    Running PPO scheduler (retrain 50)...")
            retrain_50_sim = PipelineSimulator(
                num_devices=num_devices,
                num_micro_batches=num_micro_batches,
                micro_batch_size=micro_batch_size,
                time_model=time_model
            )
            retrain_50_scheduler = PPOScheduler(retrain_50_sim)
            print("      Training PPO agent (50 episodes)...")
            retrain_50_scheduler.train(num_episodes=50, update_frequency=10)
            retrain_50_result = retrain_50_scheduler.schedule()
            
            retrain_50_times.append(retrain_50_result['total_time'])
            retrain_50_throughputs.append(retrain_50_result['throughput'])
            
            # 4. 预训练200轮
            print("    Running PPO scheduler (pre-trained 200)...")
            pretrain_200_sim = PipelineSimulator(
                num_devices=num_devices,
                num_micro_batches=num_micro_batches,
                micro_batch_size=micro_batch_size,
                time_model=time_model
            )
            pretrain_200_scheduler = load_ppo_model(pretrain_200_sim, pretrained_models_200[noise_level])
            pretrain_200_result = pretrain_200_scheduler.schedule()
            
            pretrain_200_times.append(pretrain_200_result['total_time'])
            pretrain_200_throughputs.append(pretrain_200_result['throughput'])
            
            # 5. 重新训练100轮
            print("    Running PPO scheduler (retrain 100)...")
            retrain_100_sim = PipelineSimulator(
                num_devices=num_devices,
                num_micro_batches=num_micro_batches,
                micro_batch_size=micro_batch_size,
                time_model=time_model
            )
            retrain_100_scheduler = PPOScheduler(retrain_100_sim)
            print("      Training PPO agent (100 episodes)...")
            retrain_100_scheduler.train(num_episodes=100, update_frequency=20)
            retrain_100_result = retrain_100_scheduler.schedule()
            
            retrain_100_times.append(retrain_100_result['total_time'])
            retrain_100_throughputs.append(retrain_100_result['throughput'])
        
        # 计算平均值
        avg_zb_time = np.mean(zb_times)
        avg_pretrain_100_time = np.mean(pretrain_100_times)
        avg_retrain_50_time = np.mean(retrain_50_times)
        avg_pretrain_200_time = np.mean(pretrain_200_times)
        avg_retrain_100_time = np.mean(retrain_100_times)
        avg_zb_throughput = np.mean(zb_throughputs)
        avg_pretrain_100_throughput = np.mean(pretrain_100_throughputs)
        avg_retrain_50_throughput = np.mean(retrain_50_throughputs)
        avg_pretrain_200_throughput = np.mean(pretrain_200_throughputs)
        avg_retrain_100_throughput = np.mean(retrain_100_throughputs)
        
        # 存储平均结果
        results['zero_bubble_times'].append(avg_zb_time)
        results['pretrain_100_times'].append(avg_pretrain_100_time)
        results['retrain_50_times'].append(avg_retrain_50_time)
        results['pretrain_200_times'].append(avg_pretrain_200_time)
        results['retrain_100_times'].append(avg_retrain_100_time)
        results['zero_bubble_throughputs'].append(avg_zb_throughput)
        results['pretrain_100_throughputs'].append(avg_pretrain_100_throughput)
        results['retrain_50_throughputs'].append(avg_retrain_50_throughput)
        results['pretrain_200_throughputs'].append(avg_pretrain_200_throughput)
        results['retrain_100_throughputs'].append(avg_retrain_100_throughput)
        
        # 打印当前结果
        print(f"    Average Results:")
        print(f"      Zero Bubble:      {avg_zb_time:.2f}s, Throughput: {avg_zb_throughput:.4f}")
        print(f"      Pre-train(100):   {avg_pretrain_100_time:.2f}s, Throughput: {avg_pretrain_100_throughput:.4f}")
        print(f"      Retrain(50):      {avg_retrain_50_time:.2f}s, Throughput: {avg_retrain_50_throughput:.4f}")
        print(f"      Pre-train(200):   {avg_pretrain_200_time:.2f}s, Throughput: {avg_pretrain_200_throughput:.4f}")
        print(f"      Retrain(100):     {avg_retrain_100_time:.2f}s, Throughput: {avg_retrain_100_throughput:.4f}")
        
        # 计算改进百分比
        pretrain_100_improvement = (avg_zb_time - avg_pretrain_100_time) / avg_zb_time * 100
        retrain_50_improvement = (avg_zb_time - avg_retrain_50_time) / avg_zb_time * 100
        pretrain_200_improvement = (avg_zb_time - avg_pretrain_200_time) / avg_zb_time * 100
        retrain_100_improvement = (avg_zb_time - avg_retrain_100_time) / avg_zb_time * 100
        print(f"      Improvement: Pre-train(100) {pretrain_100_improvement:.1f}%, Retrain(50) {retrain_50_improvement:.1f}%, Pre-train(200) {pretrain_200_improvement:.1f}%, Retrain(100) {retrain_100_improvement:.1f}%")
    
    # 绘制结果图表
    plot_comparison_results(results)
    
    # 打印详细分析
    print_detailed_analysis(results)
    
    # 清理临时文件
    print("\nCleaning up temporary model files...")
    for noise_level in noise_levels:
        model_path_100 = f"/data/public/yyz/PPRL/ppo_model_{int(noise_level*100)}_100.pth"
        model_path_200 = f"/data/public/yyz/PPRL/ppo_model_{int(noise_level*100)}_200.pth"
        if os.path.exists(model_path_100):
            os.remove(model_path_100)
            print(f"Removed {model_path_100}")
        if os.path.exists(model_path_200):
            os.remove(model_path_200)
            print(f"Removed {model_path_200}")
    print("Cleanup completed.")


def plot_comparison_results(results):
    """
    绘制对比结果图表
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    noise_levels = results['noise_levels']
    
    # 1. 总执行时间对比
    ax1.plot(noise_levels, results['zero_bubble_times'], 'o-', 
             label='Zero Bubble', linewidth=2, markersize=8)
    ax1.plot(noise_levels, results['pretrain_100_times'], '^-', 
             label='Pre-train(100)', linewidth=2, markersize=8)
    ax1.plot(noise_levels, results['retrain_50_times'], 's-', 
             label='Retrain(50)', linewidth=2, markersize=8)
    ax1.plot(noise_levels, results['pretrain_200_times'], 'd-', 
             label='Pre-train(200)', linewidth=2, markersize=8)
    ax1.plot(noise_levels, results['retrain_100_times'], 'v-', 
             label='Retrain(100)', linewidth=2, markersize=8)
    ax1.set_xlabel('Noise Level')
    ax1.set_ylabel('Total Execution Time (s)')
    ax1.set_title('Execution Time vs Noise Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 吞吐量对比
    ax2.plot(noise_levels, results['zero_bubble_throughputs'], 'o-', 
             label='Zero Bubble', linewidth=2, markersize=8)
    ax2.plot(noise_levels, results['pretrain_100_throughputs'], '^-', 
             label='Pre-train(100)', linewidth=2, markersize=8)
    ax2.plot(noise_levels, results['retrain_50_throughputs'], 's-', 
             label='Retrain(50)', linewidth=2, markersize=8)
    ax2.plot(noise_levels, results['pretrain_200_throughputs'], 'd-', 
             label='Pre-train(200)', linewidth=2, markersize=8)
    ax2.plot(noise_levels, results['retrain_100_throughputs'], 'v-', 
             label='Retrain(100)', linewidth=2, markersize=8)
    ax2.set_xlabel('Noise Level')
    ax2.set_ylabel('Throughput (micro-batches/s)')
    ax2.set_title('Throughput vs Noise Level')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_results_5methods.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_detailed_analysis(results):
    """
    打印详细分析结果
    """
    print("\n" + "=" * 120)
    print("DETAILED ANALYSIS: 5 Methods Comparison")
    print("=" * 120)
    
    noise_levels = results['noise_levels']
    
    # 创建结果表格
    print(f"\n{'Noise':<8} {'Zero Bubble':<12} {'Pre-train(100)':<12} {'Retrain(50)':<12} {'Pre-train(200)':<12} {'Retrain(100)':<12}")
    print(f"{'Level':<8} {'Time (s)':<12} {'Time (s)':<12} {'Time (s)':<12} {'Time (s)':<12} {'Time (s)':<12}")
    print("-" * 120)
    
    for i, noise_level in enumerate(noise_levels):
        zb_time = results['zero_bubble_times'][i]
        pretrain_100_time = results['pretrain_100_times'][i]
        retrain_50_time = results['retrain_50_times'][i]
        pretrain_200_time = results['pretrain_200_times'][i]
        retrain_100_time = results['retrain_100_times'][i]
        
        print(f"{noise_level*100:>6.0f}% {zb_time:>10.2f} {pretrain_100_time:>10.2f} {retrain_50_time:>10.2f} "
              f"{pretrain_200_time:>10.2f} {retrain_100_time:>10.2f}")
    
    # 计算改进百分比
    print(f"\n{'Noise':<8} {'Pre-train(100)':<12} {'Retrain(50)':<12} {'Pre-train(200)':<12} {'Retrain(100)':<12}")
    print(f"{'Level':<8} {'Improvement':<12} {'Improvement':<12} {'Improvement':<12} {'Improvement':<12}")
    print("-" * 120)
    
    for i, noise_level in enumerate(noise_levels):
        zb_time = results['zero_bubble_times'][i]
        pretrain_100_improvement = (zb_time - results['pretrain_100_times'][i]) / zb_time * 100
        retrain_50_improvement = (zb_time - results['retrain_50_times'][i]) / zb_time * 100
        pretrain_200_improvement = (zb_time - results['pretrain_200_times'][i]) / zb_time * 100
        retrain_100_improvement = (zb_time - results['retrain_100_times'][i]) / zb_time * 100
        
        print(f"{noise_level*100:>6.0f}% {pretrain_100_improvement:>10.1f}% {retrain_50_improvement:>10.1f}% "
              f"{pretrain_200_improvement:>10.1f}% {retrain_100_improvement:>10.1f}%")
    
    # 统计分析
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*80}")
    
    # 计算平均改进
    avg_pretrain_100_improvement = np.mean([
        (results['zero_bubble_times'][i] - results['pretrain_100_times'][i]) / 
        results['zero_bubble_times'][i] * 100
        for i in range(len(noise_levels))
    ])
    
    avg_retrain_50_improvement = np.mean([
        (results['zero_bubble_times'][i] - results['retrain_50_times'][i]) / 
        results['zero_bubble_times'][i] * 100
        for i in range(len(noise_levels))
    ])
    
    avg_pretrain_200_improvement = np.mean([
        (results['zero_bubble_times'][i] - results['pretrain_200_times'][i]) / 
        results['zero_bubble_times'][i] * 100
        for i in range(len(noise_levels))
    ])
    
    avg_retrain_100_improvement = np.mean([
        (results['zero_bubble_times'][i] - results['retrain_100_times'][i]) / 
        results['zero_bubble_times'][i] * 100
        for i in range(len(noise_levels))
    ])
    
    print(f"Average improvement of Pre-train(100) over Zero Bubble: {avg_pretrain_100_improvement:.2f}%")
    print(f"Average improvement of Retrain(50) over Zero Bubble: {avg_retrain_50_improvement:.2f}%")
    print(f"Average improvement of Pre-train(200) over Zero Bubble: {avg_pretrain_200_improvement:.2f}%")
    print(f"Average improvement of Retrain(100) over Zero Bubble: {avg_retrain_100_improvement:.2f}%")
    
    # 排序
    methods = [
        ("Pre-train(100)", avg_pretrain_100_improvement),
        ("Retrain(50)", avg_retrain_50_improvement),
        ("Pre-train(200)", avg_pretrain_200_improvement),
        ("Retrain(100)", avg_retrain_100_improvement)
    ]
    methods.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*60}")
    print("RANKING BY PERFORMANCE")
    print(f"{'='*60}")
    for i, (method, improvement) in enumerate(methods, 1):
        print(f"{i}. {method}: {improvement:.2f}% improvement")
    
    print(f"\n📊 Results saved to 'comparison_results_5methods.png'")


if __name__ == "__main__":
    run_comparison_with_training_modes()
