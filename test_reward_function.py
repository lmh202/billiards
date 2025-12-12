"""
测试奖励函数 - 验证奖励机制是否合理

检查:
1. 进球奖励是否正确
2. 距离改善奖励是否合理
3. 角度奖励不会产生过度惩罚
4. 随机动作的奖励分布
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train.pool_rl_env import PoolRLEnv
from train.config import DEVICE
import numpy as np
import time

def test_reward_distribution():
    """测试随机动作的奖励分布"""
    print("\n" + "="*60)
    print("测试: 随机动作的奖励分布")
    print("="*60)
    
    env = PoolRLEnv(
        opponent_type='random',
        enable_noise=True,
        device=DEVICE,
        two_ball_mode=True
    )
    
    rewards = []
    pockets = 0
    hits = 0
    cue_pockets = 0
    
    num_episodes = 20
    steps_per_episode = 50
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_pockets = 0
        episode_hits = 0
        
        for step in range(steps_per_episode):
            # 随机动作
            action = np.random.uniform(-1, 1, 5).astype(np.float32)
            obs, reward, done, info = env.step(action)
            
            rewards.append(reward)
            episode_reward += reward
            
            # 统计
            if info['step_info'].get('ME_INTO_POCKET'):
                pockets += 1
                episode_pockets += 1
            if info['step_info'].get('TARGET_HIT'):
                hits += 1
                episode_hits += 1
            if info['step_info'].get('WHITE_BALL_INTO_POCKET'):
                cue_pockets += 1
            
            if done:
                break
        
        print(f"回合 {episode+1:02d}: 总奖励={episode_reward:+7.2f} | "
              f"进球={episode_pockets} | 命中={episode_hits}")
    
    # 统计
    rewards = np.array(rewards)
    print(f"\n" + "="*60)
    print(f"统计结果 (总共 {len(rewards)} 步):")
    print(f"="*60)
    print(f"奖励均值: {rewards.mean():+.2f}")
    print(f"奖励标准差: {rewards.std():.2f}")
    print(f"奖励最小值: {rewards.min():+.2f}")
    print(f"奖励最大值: {rewards.max():+.2f}")
    print(f"奖励中位数: {np.median(rewards):+.2f}")
    print(f"\n进球次数: {pockets} ({pockets/num_episodes:.1f}/回合)")
    print(f"命中次数: {hits} ({hits/len(rewards)*100:.1f}%)")
    print(f"白球进洞: {cue_pockets}")
    
    # 奖励分布
    print(f"\n奖励分布:")
    print(f"  < -5:  {(rewards < -5).sum()} ({(rewards < -5).sum()/len(rewards)*100:.1f}%)")
    print(f"  -5~0:  {((rewards >= -5) & (rewards < 0)).sum()} ({((rewards >= -5) & (rewards < 0)).sum()/len(rewards)*100:.1f}%)")
    print(f"  0~5:   {((rewards >= 0) & (rewards < 5)).sum()} ({((rewards >= 0) & (rewards < 5)).sum()/len(rewards)*100:.1f}%)")
    print(f"  5~10:  {((rewards >= 5) & (rewards < 10)).sum()} ({((rewards >= 5) & (rewards < 10)).sum()/len(rewards)*100:.1f}%)")
    print(f"  > 10:  {(rewards >= 10).sum()} ({(rewards >= 10).sum()/len(rewards)*100:.1f}%)")
    
    # 判断奖励是否合理
    print(f"\n" + "="*60)
    if rewards.mean() < -10:
        print("⚠️ 警告: 平均奖励过低,可能惩罚过度!")
    elif rewards.mean() > 5:
        print("⚠️ 警告: 平均奖励过高,可能奖励过度!")
    else:
        print("✅ 奖励范围合理")
    
    if (rewards < -5).sum() > len(rewards) * 0.3:
        print("⚠️ 警告: 超过30%的步骤奖励<-5,惩罚可能过度!")
    else:
        print("✅ 负奖励比例合理")
    
    if hits / len(rewards) < 0.05:
        print("⚠️ 警告: 命中率<5%,任务可能太难!")
    else:
        print(f"✅ 命中率正常 ({hits/len(rewards)*100:.1f}%)")

def test_specific_scenarios():
    """测试特定场景的奖励"""
    print("\n" + "="*60)
    print("测试: 特定场景奖励")
    print("="*60)
    
    env = PoolRLEnv(
        opponent_type='random',
        enable_noise=False,  # 关闭噪声,便于测试
        device=DEVICE,
        two_ball_mode=True
    )
    
    # 测试1: 正常击球
    print("\n场景1: 正常击球 (中等力度,随机方向)")
    obs = env.reset()
    action = np.array([2.0, 45.0, 30.0, 0.0, 0.0], dtype=np.float32)
    obs, reward, done, info = env.step(action)
    print(f"  奖励: {reward:+.2f}")
    print(f"  命中: {info['step_info'].get('TARGET_HIT')}")
    print(f"  进球: {len(info['step_info'].get('ME_INTO_POCKET', []))} ")
    
    # 测试2: 极小力度
    print("\n场景2: 极小力度 (几乎不动)")
    obs = env.reset()
    action = np.array([0.5, 0.0, 30.0, 0.0, 0.0], dtype=np.float32)
    obs, reward, done, info = env.step(action)
    print(f"  奖励: {reward:+.2f} (应该有站桩惩罚)")
    
    # 测试3: 过大力度
    print("\n场景3: 过大力度")
    obs = env.reset()
    action = np.array([8.0, 0.0, 10.0, 0.0, 0.0], dtype=np.float32)
    obs, reward, done, info = env.step(action)
    print(f"  奖励: {reward:+.2f} (可能有力度惩罚)")

if __name__ == '__main__':
    print("\n" + "#"*60)
    print("# Phase 1 奖励函数测试")
    print("#"*60)
    
    # 测试1: 奖励分布
    test_reward_distribution()
    
    # 测试2: 特定场景
    test_specific_scenarios()
    
    print("\n" + "#"*60)
    print("# 测试完成")
    print("#"*60)
