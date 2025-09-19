#!/usr/bin/env python3
"""
Training Time Comparison: Pre-trained 200 episodes vs Zero Bubble
训练时间对比：预训练200轮 vs Zero Bubble

测试大规模配置：
- 20个设备
- 30个micro-batch
- 对比训练时间和性能
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import time
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


def run_time_comparison():
    """
    运行训练时间对比实验
    
    对比两种方法：
    1. Zero Bubble（无训练时间）
    2. 预训练200轮（有训练时间）
    """
    print("Training Time Comparison: Pre-trained 200 episodes vs Zero Bubble")
    print("=" * 80)
    print("Large Scale Configuration: 20 devices, 30 micro-batches")
    print("=" * 80)
    
    # 大规模实验配置
    num_devices = 20
    num_micro_batches = 30
    micro_batch_size = 32
    noise_levels = [0.0, 0.1, 0.2, 0.3]  # 0%, 10%, 20%, 30%波动
    num_trials = 3  # 减少试验次数以加快速度
    
    # 存储结果
    results = {
        'noise_levels': noise_levels,
        'zero_bubble_times': [],
        'pretrain_200_times': [],
        'zero_bubble_throughputs': [],
        'pretrain_200_throughputs': [],
        'training_times': [],
        'total_operations': num_devices * num_micro_batches * 3
    }
    
    print(f"Problem Scale: {num_devices} devices × {num_micro_batches} micro-batches = {results['total_operations']} operations")
    
    # 为每个噪声水平训练预训练模型
    print("\n" + "="*60)
    print("STEP 1: Training pre-trained models for each noise level")
    print("="*60)
    
    pretrained_models = {}
    training_times = {}
    
    for noise_level in noise_levels:
        print(f"\nTraining pre-trained model for {noise_level*100:.0f}% noise...")
        
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
        
        # 训练200轮预训练模型并记录时间
        model_path = f"/data/public/yyz/PPRL/ppo_model_large_{int(noise_level*100)}_200.pth"
        
        start_time = time.time()
        ppo_scheduler, training_rewards = train_ppo_model(
            simulator, 
            model_path=model_path,
            num_episodes=200
        )
        training_time = time.time() - start_time
        
        pretrained_models[noise_level] = model_path
        training_times[noise_level] = training_time
        
        print(f"200-episode model saved to {model_path}")
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Final average reward: {np.mean(training_rewards[-10:]):.2f}")
    
    # 对每个噪声水平进行测试
    for noise_level in noise_levels:
        print(f"\n" + "="*60)
        print(f"Testing with {noise_level*100:.0f}% noise level...")
        print(f"Running {num_trials} trials for each algorithm...")
        print("="*60)
        
        # 存储多次试验的结果
        zb_times = []
        pretrain_200_times = []
        zb_throughputs = []
        pretrain_200_throughputs = []
        
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
            start_time = time.time()
            zb_sim = PipelineSimulator(
                num_devices=num_devices,
                num_micro_batches=num_micro_batches,
                micro_batch_size=micro_batch_size,
                time_model=time_model
            )
            zb_scheduler = ZeroBubbleScheduler(zb_sim)
            zb_result = zb_scheduler.schedule()
            zb_execution_time = time.time() - start_time
            
            zb_times.append(zb_result['total_time'])
            zb_throughputs.append(zb_result['throughput'])
            print(f"      Zero Bubble execution time: {zb_execution_time:.2f}s")
            
            # 2. 预训练200轮
            print("    Running PPO scheduler (pre-trained 200)...")
            start_time = time.time()
            pretrain_200_sim = PipelineSimulator(
                num_devices=num_devices,
                num_micro_batches=num_micro_batches,
                micro_batch_size=micro_batch_size,
                time_model=time_model
            )
            pretrain_200_scheduler = load_ppo_model(pretrain_200_sim, pretrained_models[noise_level])
            pretrain_200_result = pretrain_200_scheduler.schedule()
            pretrain_execution_time = time.time() - start_time
            
            pretrain_200_times.append(pretrain_200_result['total_time'])
            pretrain_200_throughputs.append(pretrain_200_result['throughput'])
            print(f"      Pre-trained execution time: {pretrain_execution_time:.2f}s")
        
        # 计算平均值
        avg_zb_time = np.mean(zb_times)
        avg_pretrain_200_time = np.mean(pretrain_200_times)
        avg_zb_throughput = np.mean(zb_throughputs)
        avg_pretrain_200_throughput = np.mean(pretrain_200_throughputs)
        
        # 存储平均结果
        results['zero_bubble_times'].append(avg_zb_time)
        results['pretrain_200_times'].append(avg_pretrain_200_time)
        results['zero_bubble_throughputs'].append(avg_zb_throughput)
        results['pretrain_200_throughputs'].append(avg_pretrain_200_throughput)
        results['training_times'].append(training_times[noise_level])
        
        # 打印当前结果
        print(f"    Average Results:")
        print(f"      Zero Bubble:      {avg_zb_time:.2f}s, Throughput: {avg_zb_throughput:.4f}")
        print(f"      Pre-train(200):   {avg_pretrain_200_time:.2f}s, Throughput: {avg_pretrain_200_throughput:.4f}")
        
        # 计算改进百分比
        pretrain_200_improvement = (avg_zb_time - avg_pretrain_200_time) / avg_zb_time * 100
        print(f"      Improvement: Pre-train(200) {pretrain_200_improvement:.1f}%")
        print(f"      Training time: {training_times[noise_level]:.2f}s")
    
    # 绘制结果图表
    plot_time_comparison_results(results)
    
    # 打印详细分析
    print_time_analysis(results)
    
    # 清理临时文件
    print("\nCleaning up temporary model files...")
    for noise_level in noise_levels:
        model_path = f"/data/public/yyz/PPRL/ppo_model_large_{int(noise_level*100)}_200.pth"
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Removed {model_path}")
    print("Cleanup completed.")


def plot_time_comparison_results(results):
    """
    绘制训练时间对比结果图表
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    noise_levels = results['noise_levels']
    
    # 1. 总执行时间对比
    ax1.plot(noise_levels, results['zero_bubble_times'], 'o-', 
             label='Zero Bubble', linewidth=2, markersize=8)
    ax1.plot(noise_levels, results['pretrain_200_times'], '^-', 
             label='Pre-train(200)', linewidth=2, markersize=8)
    ax1.set_xlabel('Noise Level')
    ax1.set_ylabel('Total Execution Time (s)')
    ax1.set_title('Execution Time vs Noise Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 吞吐量对比
    ax2.plot(noise_levels, results['zero_bubble_throughputs'], 'o-', 
             label='Zero Bubble', linewidth=2, markersize=8)
    ax2.plot(noise_levels, results['pretrain_200_throughputs'], '^-', 
             label='Pre-train(200)', linewidth=2, markersize=8)
    ax2.set_xlabel('Noise Level')
    ax2.set_ylabel('Throughput (micro-batches/s)')
    ax2.set_title('Throughput vs Noise Level')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 训练时间
    ax3.bar([f"{int(nl*100)}%" for nl in noise_levels], results['training_times'], 
            color='skyblue', alpha=0.7)
    ax3.set_xlabel('Noise Level')
    ax3.set_ylabel('Training Time (s)')
    ax3.set_title('PPO Training Time by Noise Level')
    ax3.grid(True, alpha=0.3)
    
    # 4. 性能改进 vs 训练时间
    improvements = []
    for i, noise_level in enumerate(noise_levels):
        zb_time = results['zero_bubble_times'][i]
        pretrain_time = results['pretrain_200_times'][i]
        improvement = (zb_time - pretrain_time) / zb_time * 100
        improvements.append(improvement)
    
    ax4.scatter(results['training_times'], improvements, s=100, alpha=0.7)
    for i, noise_level in enumerate(noise_levels):
        ax4.annotate(f"{int(noise_level*100)}%", 
                    (results['training_times'][i], improvements[i]),
                    xytext=(5, 5), textcoords='offset points')
    ax4.set_xlabel('Training Time (s)')
    ax4.set_ylabel('Performance Improvement (%)')
    ax4.set_title('Training Time vs Performance Improvement')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('time_comparison_large_scale.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_time_analysis(results):
    """
    打印训练时间分析结果
    """
    print("\n" + "=" * 100)
    print("TRAINING TIME ANALYSIS: Large Scale (20 devices, 30 micro-batches)")
    print("=" * 100)
    
    noise_levels = results['noise_levels']
    
    # 创建结果表格
    print(f"\n{'Noise':<8} {'Zero Bubble':<12} {'Pre-train(200)':<12} {'Improvement':<12} {'Training Time':<12}")
    print(f"{'Level':<8} {'Time (s)':<12} {'Time (s)':<12} {'(%)':<12} {'(s)':<12}")
    print("-" * 100)
    
    for i, noise_level in enumerate(noise_levels):
        zb_time = results['zero_bubble_times'][i]
        pretrain_time = results['pretrain_200_times'][i]
        training_time = results['training_times'][i]
        
        improvement = (zb_time - pretrain_time) / zb_time * 100
        
        print(f"{noise_level*100:>6.0f}% {zb_time:>10.2f} {pretrain_time:>10.2f} "
              f"{improvement:>10.1f}% {training_time:>10.1f}")
    
    # 统计分析
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*80}")
    
    # 计算平均改进
    avg_improvement = np.mean([
        (results['zero_bubble_times'][i] - results['pretrain_200_times'][i]) / 
        results['zero_bubble_times'][i] * 100
        for i in range(len(noise_levels))
    ])
    
    # 计算平均训练时间
    avg_training_time = np.mean(results['training_times'])
    
    # 计算总训练时间
    total_training_time = sum(results['training_times'])
    
    print(f"Average improvement of Pre-train(200) over Zero Bubble: {avg_improvement:.2f}%")
    print(f"Average training time per noise level: {avg_training_time:.2f}s")
    print(f"Total training time for all noise levels: {total_training_time:.2f}s")
    print(f"Problem scale: {results['total_operations']} operations")
    
    # 效率分析
    print(f"\n{'='*80}")
    print("EFFICIENCY ANALYSIS")
    print(f"{'='*80}")
    
    print(f"Training time per operation: {total_training_time / results['total_operations']:.6f}s")
    print(f"Training time per device: {total_training_time / 20:.2f}s")
    print(f"Training time per micro-batch: {total_training_time / 30:.2f}s")
    
    # 成本效益分析
    print(f"\n{'='*80}")
    print("COST-BENEFIT ANALYSIS")
    print(f"{'='*80}")
    
    if avg_improvement > 0:
        print(f"✅ Pre-train(200) provides {avg_improvement:.2f}% performance improvement")
        print(f"   Training cost: {total_training_time:.2f}s")
        print(f"   Performance gain: {avg_improvement:.2f}%")
        print(f"   Cost per 1% improvement: {total_training_time / avg_improvement:.2f}s")
    else:
        print(f"❌ Pre-train(200) does not provide performance improvement")
        print(f"   Training cost: {total_training_time:.2f}s (wasted)")
    
    print(f"\n📊 Results saved to 'time_comparison_large_scale.png'")


if __name__ == "__main__":
    run_time_comparison()
