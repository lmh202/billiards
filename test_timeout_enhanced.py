"""
增强版超时测试 - 检测卡死和慢步骤

测试策略:
1. 单环境测试 - 检测单个环境是否会卡死
2. 多环境测试 - 检测并行环境的交互
3. 长时间测试 - 运行足够多的步骤来触发边缘情况
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train.pool_rl_env import PoolRLEnv, VectorPoolEnv
from train.config import DEVICE
import numpy as np
import time
import threading

def timeout_monitor(timeout_seconds, test_name):
    """监控线程 - 如果主线程卡死,打印警告"""
    time.sleep(timeout_seconds)
    print(f"\n{'='*60}")
    print(f"⚠️⚠️⚠️ 警告: {test_name} 运行超过 {timeout_seconds}秒!")
    print(f"可能已经卡死,请检查主线程状态")
    print(f"{'='*60}\n")

def test_single_env(num_steps=100):
    """测试单个环境"""
    print("\n" + "="*60)
    print("测试 1: 单个环境稳定性测试")
    print("="*60)
    
    # 启动监控线程
    monitor = threading.Thread(target=timeout_monitor, args=(30, "单环境测试"), daemon=True)
    monitor.start()
    
    env = PoolRLEnv(
        opponent_type='random',
        enable_noise=True,
        device=DEVICE,
        two_ball_mode=True
    )
    
    print(f"开始运行 {num_steps} 步...")
    obs = env.reset()
    
    slow_steps = []
    total_start = time.time()
    
    for step in range(num_steps):
        step_start = time.time()
        
        # 随机动作
        action = np.random.uniform(-1, 1, 5).astype(np.float32)
        
        try:
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
                
            step_elapsed = time.time() - step_start
            
            if step_elapsed > 1.0:
                slow_steps.append((step, step_elapsed))
                print(f"  ⚠️ 慢步骤 {step+1}/{num_steps}: {step_elapsed:.2f}秒")
            
            if (step + 1) % 20 == 0:
                avg_time = (time.time() - total_start) / (step + 1)
                print(f"进度: {step+1}/{num_steps} | 平均耗时: {avg_time:.3f}s/步")
                
        except Exception as e:
            print(f"❌ 步骤 {step+1} 失败: {type(e).__name__}: {e}")
            break
    
    total_elapsed = time.time() - total_start
    print(f"\n✅ 单环境测试完成!")
    print(f"  总耗时: {total_elapsed:.2f}s")
    print(f"  平均每步: {total_elapsed/num_steps:.3f}s")
    print(f"  慢步骤数: {len(slow_steps)}/{num_steps}")
    
    if slow_steps:
        print(f"\n  最慢的5步:")
        for step, elapsed in sorted(slow_steps, key=lambda x: x[1], reverse=True)[:5]:
            print(f"    步骤 {step+1}: {elapsed:.2f}s")
    
    return len(slow_steps) == 0

def test_vector_env(num_envs=4, num_steps=50):
    """测试向量化环境"""
    print("\n" + "="*60)
    print(f"测试 2: 向量化环境测试 ({num_envs}个并行环境)")
    print("="*60)
    
    # 启动监控线程
    monitor = threading.Thread(target=timeout_monitor, args=(60, "向量化环境测试"), daemon=True)
    monitor.start()
    
    envs = VectorPoolEnv(
        num_envs=num_envs,
        device=DEVICE,
        opponent_type='random',
        enable_noise=True,
        two_ball_mode=True
    )
    
    print(f"开始运行 {num_steps} 步...")
    obs = envs.reset()
    
    slow_steps = []
    total_start = time.time()
    
    for step in range(num_steps):
        step_start = time.time()
        
        # 随机动作
        actions = np.random.uniform(-1, 1, (num_envs, 5)).astype(np.float32)
        
        try:
            next_obs, rewards, dones, infos = envs.step(actions)
            
            step_elapsed = time.time() - step_start
            
            if step_elapsed > 1.5:
                slow_steps.append((step, step_elapsed))
                print(f"  ⚠️ 慢步骤 {step+1}/{num_steps}: {step_elapsed:.2f}秒")
            
            if (step + 1) % 10 == 0:
                avg_time = (time.time() - total_start) / (step + 1)
                avg_reward = np.mean(rewards)
                print(f"进度: {step+1}/{num_steps} | 平均耗时: {avg_time:.3f}s/步 | 平均奖励: {avg_reward:.2f}")
                
        except Exception as e:
            print(f"❌ 步骤 {step+1} 失败: {type(e).__name__}: {e}")
            break
    
    total_elapsed = time.time() - total_start
    print(f"\n✅ 向量化环境测试完成!")
    print(f"  总耗时: {total_elapsed:.2f}s")
    print(f"  平均每步: {total_elapsed/num_steps:.3f}s")
    print(f"  慢步骤数: {len(slow_steps)}/{num_steps}")
    
    if slow_steps:
        print(f"\n  最慢的5步:")
        for step, elapsed in sorted(slow_steps, key=lambda x: x[1], reverse=True)[:5]:
            print(f"    步骤 {step+1}: {elapsed:.2f}s")
    
    return len(slow_steps) == 0

def main():
    print("\n" + "#"*60)
    print("# 台球训练环境 - 卡死检测测试")
    print("#"*60)
    
    # 测试1: 单环境
    test1_pass = test_single_env(num_steps=100)
    
    # 测试2: 向量化环境
    test2_pass = test_vector_env(num_envs=4, num_steps=50)
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"  单环境测试: {'✅ 通过' if test1_pass else '⚠️ 有慢步骤'}")
    print(f"  向量化环境测试: {'✅ 通过' if test2_pass else '⚠️ 有慢步骤'}")
    
    if test1_pass and test2_pass:
        print("\n🎉 所有测试通过! 可以开始训练了")
        print("运行: python -m train.train_phase1")
    else:
        print("\n⚠️ 存在慢步骤,但如果没有卡死就可以继续")
        print("如果训练时卡死,请检查日志中的慢步骤模式")

if __name__ == '__main__':
    main()
