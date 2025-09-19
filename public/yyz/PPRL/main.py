#!/usr/bin/env python3
"""
Pipeline Parallelism Scheduling Main Demo
流水线并行调度主演示脚本

对比两种调度算法：
1. Zero Bubble Pipeline Parallelism
2. PPO强化学习调度

在10%-30%时间波动下的性能表现
"""

import numpy as np
import matplotlib.pyplot as plt
from pipeline_simulator import PipelineSimulator, TimeModel
from zero_bubble_scheduler import ZeroBubbleScheduler
from ppo_scheduler import PPOScheduler


def run_comparison():
    """
    运行对比实验
    
    测试不同噪声水平下两种算法的性能
    使用多次试验取平均来获得更稳定的结果
    """
    print("Pipeline Parallelism Scheduling Comparison")
    print("=" * 60)
    print("Algorithms: Zero Bubble vs PPO")
    print("=" * 60)
    
    # 实验配置
    num_devices = 4
    num_micro_batches = 6
    micro_batch_size = 32
    noise_levels = [0.0, 0.1, 0.2, 0.3]  # 0%, 10%, 20%, 30%波动
    num_trials = 10  # 每个配置运行10次取平均
    
    # 存储结果
    results = {
        'noise_levels': noise_levels,
        'zero_bubble_times': [],
        'ppo_times': [],
        'zero_bubble_throughputs': [],
        'ppo_throughputs': []
    }
    
    # 对每个噪声水平进行测试
    for noise_level in noise_levels:
        print(f"\nTesting with {noise_level*100:.0f}% noise level...")
        print(f"Running {num_trials} trials for each algorithm...")
        
        # 存储多次试验的结果
        zb_times = []
        ppo_times = []
        zb_throughputs = []
        ppo_throughputs = []
        
        for trial in range(num_trials):
            print(f"  Trial {trial + 1}/{num_trials}:")
            
            # 创建时间模型（使用不同的种子）
            time_model = TimeModel(
                base_compute_time=1.0,
                base_comm_time=0.5,
                noise_level=noise_level,
                seed=42 + trial  # 不同试验使用不同种子
            )
            
            # 1. Zero Bubble调度
            print("    Running Zero Bubble scheduler...")
            zero_bubble_sim = PipelineSimulator(
                num_devices=num_devices,
                num_micro_batches=num_micro_batches,
                micro_batch_size=micro_batch_size,
                time_model=time_model
            )
            zero_bubble_scheduler = ZeroBubbleScheduler(zero_bubble_sim)
            zero_bubble_result = zero_bubble_scheduler.schedule()
            
            zb_times.append(zero_bubble_result['total_time'])
            zb_throughputs.append(zero_bubble_result['throughput'])
            
            # 2. PPO调度
            print("    Running PPO scheduler...")
            ppo_sim = PipelineSimulator(
                num_devices=num_devices,
                num_micro_batches=num_micro_batches,
                micro_batch_size=micro_batch_size,
                time_model=time_model
            )
            ppo_scheduler = PPOScheduler(ppo_sim)
            
            # 训练PPO智能体（减少训练轮数以加快速度）
            print("      Training PPO agent (50 episodes)...")
            ppo_scheduler.train(num_episodes=50, update_frequency=10)
            
            # 使用训练好的智能体进行调度
            ppo_result = ppo_scheduler.schedule()
            
            ppo_times.append(ppo_result['total_time'])
            ppo_throughputs.append(ppo_result['throughput'])
        
        # 计算平均值
        avg_zb_time = np.mean(zb_times)
        avg_ppo_time = np.mean(ppo_times)
        avg_zb_throughput = np.mean(zb_throughputs)
        avg_ppo_throughput = np.mean(ppo_throughputs)
        
        # 存储平均结果
        results['zero_bubble_times'].append(avg_zb_time)
        results['ppo_times'].append(avg_ppo_time)
        results['zero_bubble_throughputs'].append(avg_zb_throughput)
        results['ppo_throughputs'].append(avg_ppo_throughput)
        
        # 打印当前结果
        print(f"    Average Results:")
        print(f"      Zero Bubble: {avg_zb_time:.2f}s, Throughput: {avg_zb_throughput:.4f}")
        print(f"      PPO:         {avg_ppo_time:.2f}s, Throughput: {avg_ppo_throughput:.4f}")
        print(f"      Improvement: {((avg_zb_time - avg_ppo_time) / avg_zb_time * 100):.1f}%")
    
    # 绘制结果图表
    plot_results(results)
    
    # 打印详细分析
    print_analysis(results)


def plot_results(results):
    """
    绘制对比结果图表
    
    Args:
        results: 实验结果字典
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    noise_levels = results['noise_levels']
    
    # 1. 总执行时间对比
    ax1.plot(noise_levels, results['zero_bubble_times'], 'o-', 
             label='Zero Bubble', linewidth=2, markersize=8)
    ax1.plot(noise_levels, results['ppo_times'], '^-', 
             label='PPO', linewidth=2, markersize=8)
    ax1.set_xlabel('Noise Level')
    ax1.set_ylabel('Total Execution Time (s)')
    ax1.set_title('Execution Time vs Noise Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 吞吐量对比
    ax2.plot(noise_levels, results['zero_bubble_throughputs'], 'o-', 
             label='Zero Bubble', linewidth=2, markersize=8)
    ax2.plot(noise_levels, results['ppo_throughputs'], '^-', 
             label='PPO', linewidth=2, markersize=8)
    ax2.set_xlabel('Noise Level')
    ax2.set_ylabel('Throughput (micro-batches/s)')
    ax2.set_title('Throughput vs Noise Level')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_analysis(results):
    """
    打印详细分析结果
    
    Args:
        results: 实验结果字典
    """
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    
    noise_levels = results['noise_levels']
    
    # 创建结果表格
    print(f"\n{'Noise':<8} {'Zero Bubble':<15} {'PPO':<15} {'Improvement':<12} {'Winner':<10}")
    print(f"{'Level':<8} {'Time (s)':<15} {'Time (s)':<15} {'(%)':<12} {'':<10}")
    print("-" * 80)
    
    for i, noise_level in enumerate(noise_levels):
        zb_time = results['zero_bubble_times'][i]
        ppo_time = results['ppo_times'][i]
        
        improvement = (zb_time - ppo_time) / zb_time * 100
        winner = "PPO" if ppo_time < zb_time else "Zero Bubble"
        
        print(f"{noise_level*100:>6.0f}% {zb_time:>13.2f} {ppo_time:>13.2f} {improvement:>10.1f}% {winner:<10}")
    
    # 统计分析
    print(f"\n{'='*50}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*50}")
    
    # 计算平均改进
    avg_improvement = np.mean([
        (results['zero_bubble_times'][i] - results['ppo_times'][i]) / 
        results['zero_bubble_times'][i] * 100
        for i in range(len(noise_levels))
    ])
    
    print(f"Average improvement of PPO over Zero Bubble: {avg_improvement:.2f}%")
    
    # 波动影响分析
    print(f"\n{'='*50}")
    print("NOISE IMPACT ANALYSIS")
    print(f"{'='*50}")
    
    baseline_zb = results['zero_bubble_times'][0]
    baseline_ppo = results['ppo_times'][0]
    
    for i, noise_level in enumerate(noise_levels[1:], 1):
        zb_impact = (results['zero_bubble_times'][i] - baseline_zb) / baseline_zb * 100
        ppo_impact = (results['ppo_times'][i] - baseline_ppo) / baseline_ppo * 100
        
        print(f"{noise_level*100:.0f}% noise:")
        print(f"  Zero Bubble performance change: {zb_impact:+.2f}%")
        print(f"  PPO performance change: {ppo_impact:+.2f}%")
    
    # 最佳性能分析
    print(f"\n{'='*50}")
    print("BEST PERFORMANCE ANALYSIS")
    print(f"{'='*50}")
    
    ppo_wins = 0
    for i, noise_level in enumerate(noise_levels):
        zb_time = results['zero_bubble_times'][i]
        ppo_time = results['ppo_times'][i]
        
        if ppo_time < zb_time:
            ppo_wins += 1
            improvement = (zb_time - ppo_time) / zb_time * 100
            print(f"{noise_level*100:.0f}% noise: PPO wins ({improvement:.1f}% better)")
        else:
            improvement = (ppo_time - zb_time) / ppo_time * 100
            print(f"{noise_level*100:.0f}% noise: Zero Bubble wins ({improvement:.1f}% better)")
    
    print(f"\nPPO wins in {ppo_wins}/{len(noise_levels)} scenarios")
    
    # 结论
    print(f"\n{'='*50}")
    print("CONCLUSIONS")
    print(f"{'='*50}")
    
    if avg_improvement > 0:
        print(f"✅ PPO algorithm shows better overall performance")
        print(f"   Average improvement: {avg_improvement:.2f}%")
    else:
        print(f"✅ Zero Bubble algorithm shows better overall performance")
        print(f"   Average improvement: {-avg_improvement:.2f}%")
    
    if ppo_wins > len(noise_levels) // 2:
        print(f"✅ PPO algorithm wins in most scenarios ({ppo_wins}/{len(noise_levels)})")
    else:
        print(f"✅ Zero Bubble algorithm wins in most scenarios ({len(noise_levels)-ppo_wins}/{len(noise_levels)})")
    
    print(f"\n📊 Results saved to 'comparison_results.png'")


if __name__ == "__main__":
    run_comparison()
