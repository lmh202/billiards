#!/usr/bin/env python
"""
train_ppo.py - PPO训练主脚本

使用方法:
    python train/train_ppo.py [OPTIONS]

选项:
    --timesteps INT     总训练步数 (默认: 2000000)
    --num_envs INT      并行环境数 (默认: 8)
    --lr FLOAT          学习率 (默认: 3e-4)
    --gamma FLOAT       折扣因子 (默认: 0.99)
    --checkpoint PATH   从检查点恢复训练
    --eval_only         仅评估模式
"""

import argparse
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from train.config import PPO_CONFIG, TRAIN_CONFIG, NETWORK_CONFIG, DEVICE
from train.ppo_trainer import PPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='PPO台球智能体训练')
    
    # 训练参数
    parser.add_argument('--timesteps', type=int, default=TRAIN_CONFIG['total_timesteps'],
                       help='总训练步数')
    parser.add_argument('--num_envs', type=int, default=TRAIN_CONFIG['num_envs'],
                       help='并行环境数量')
    
    # PPO超参数
    parser.add_argument('--lr', type=float, default=PPO_CONFIG['lr_actor'],
                       help='Actor学习率')
    parser.add_argument('--lr_critic', type=float, default=PPO_CONFIG['lr_critic'],
                       help='Critic学习率')
    parser.add_argument('--gamma', type=float, default=PPO_CONFIG['gamma'],
                       help='折扣因子')
    parser.add_argument('--clip', type=float, default=PPO_CONFIG['clip_epsilon'],
                       help='PPO裁剪系数')
    parser.add_argument('--entropy', type=float, default=PPO_CONFIG['entropy_coef'],
                       help='熵正则化系数')
    
    # 检查点
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='从检查点恢复训练')
    parser.add_argument('--save_freq', type=int, default=TRAIN_CONFIG['save_freq'],
                       help='保存频率')
    
    # 设备
    parser.add_argument('--device', type=str, default=None,
                       help='训练设备 (cuda/cpu)')
    
    # 模式
    parser.add_argument('--eval_only', action='store_true',
                       help='仅评估模式')
    parser.add_argument('--eval_episodes', type=int, default=TRAIN_CONFIG['eval_episodes'],
                       help='评估局数')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("PPO 台球智能体训练")
    print("=" * 60)
    
    # 更新配置
    ppo_config = PPO_CONFIG.copy()
    ppo_config['lr_actor'] = args.lr
    ppo_config['lr_critic'] = args.lr_critic
    ppo_config['gamma'] = args.gamma
    ppo_config['clip_epsilon'] = args.clip
    ppo_config['entropy_coef'] = args.entropy
    
    train_config = TRAIN_CONFIG.copy()
    train_config['total_timesteps'] = args.timesteps
    train_config['num_envs'] = args.num_envs
    train_config['save_freq'] = args.save_freq
    train_config['eval_episodes'] = args.eval_episodes
    
    # 设备
    device = args.device if args.device else DEVICE
    
    print(f"\n配置信息:")
    print(f"  设备: {device}")
    print(f"  总步数: {args.timesteps:,}")
    print(f"  并行环境: {args.num_envs}")
    print(f"  学习率 (Actor): {args.lr}")
    print(f"  学习率 (Critic): {args.lr_critic}")
    print(f"  折扣因子: {args.gamma}")
    print(f"  PPO裁剪: {args.clip}")
    print(f"  熵系数: {args.entropy}")
    print()
    
    # 创建训练器
    trainer = PPOTrainer(
        config=ppo_config,
        train_config=train_config,
        network_config=NETWORK_CONFIG,
        device=device
    )
    
    # 加载检查点
    if args.checkpoint:
        print(f"从检查点加载: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    if args.eval_only:
        # 仅评估模式
        print("\n开始评估...")
        results = trainer.evaluate(n_episodes=args.eval_episodes)
        print(f"\n评估结果 ({args.eval_episodes}局):")
        print(f"  胜率: {results['win_rate']:.2%}")
        print(f"  平局率: {results['draw_rate']:.2%}")
        print(f"  负率: {results['loss_rate']:.2%}")
        print(f"  平均奖励: {results['avg_reward']:.2f}")
        print(f"  胜/平/负: {results['wins']}/{results['draws']}/{results['losses']}")
    else:
        # 训练模式
        print("\n开始训练...")
        trainer.train(total_timesteps=args.timesteps)


if __name__ == '__main__':
    main()
