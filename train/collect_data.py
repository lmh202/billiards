"""
collect_data.py - 数据收集脚本

功能:
- 使用BasicAgentPro进行自我对弈，收集(state, action)对
- 用于BC(行为克隆)训练
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm
import argparse

from poolenv import PoolEnv
from agents import BasicAgentPro
from state_utils import extract_state_features, action_to_network_target, BCDataset


def collect_expert_data(n_games: int = 100, save_path: str = 'expert_data.npz', 
                        verbose: bool = True) -> BCDataset:
    """
    收集专家数据
    
    Args:
        n_games: 收集的局数
        save_path: 数据保存路径
        verbose: 是否打印详细信息
        
    Returns:
        BCDataset: 收集的数据集
    """
    env = PoolEnv()
    expert_agent = BasicAgentPro(n_simulations=50)  # 使用较多模拟次数确保动作质量
    
    dataset = BCDataset()
    target_ball_choices = ['solid', 'solid', 'stripe', 'stripe']
    
    total_shots = 0
    
    print(f"[数据收集] 开始收集 {n_games} 局专家数据...")
    
    for game_idx in tqdm(range(n_games), desc="收集数据"):
        env.reset(target_ball=target_ball_choices[game_idx % 4])
        
        while True:
            player = env.get_curr_player()
            balls, my_targets, table = env.get_observation(player)
            
            # 提取状态特征
            state_features = extract_state_features(balls, my_targets, table)
            
            # 让专家做出决策
            action = expert_agent.decision(balls, my_targets, table)
            
            # 转换动作为训练目标格式
            action_target = action_to_network_target(action)
            
            # 添加到数据集
            dataset.add(state_features, action_target)
            total_shots += 1
            
            # 执行动作
            step_info = env.take_shot(action)
            
            done, info = env.get_done()
            if done:
                if verbose and game_idx % 10 == 0:
                    print(f"[Game {game_idx}] 结束，累计收集 {total_shots} 条数据")
                break
    
    print(f"[数据收集] 完成！共收集 {len(dataset)} 条数据")
    
    # 保存数据
    if save_path:
        dataset.save(save_path)
    
    return dataset


def collect_self_play_data(bc_agent, n_games: int = 50, 
                           save_path: str = 'dagger_data.npz') -> BCDataset:
    """
    DAgger: 用当前策略打球，让专家标注
    
    Args:
        bc_agent: 当前训练的BC智能体
        n_games: 收集的局数
        save_path: 数据保存路径
        
    Returns:
        BCDataset: 收集的数据集
    """
    env = PoolEnv()
    expert_agent = BasicAgentPro(n_simulations=50)
    
    dataset = BCDataset()
    target_ball_choices = ['solid', 'solid', 'stripe', 'stripe']
    
    print(f"[DAgger] 开始收集 {n_games} 局自博弈数据...")
    
    for game_idx in tqdm(range(n_games), desc="DAgger数据收集"):
        env.reset(target_ball=target_ball_choices[game_idx % 4])
        
        while True:
            player = env.get_curr_player()
            balls, my_targets, table = env.get_observation(player)
            
            # 提取状态特征
            state_features = extract_state_features(balls, my_targets, table)
            
            # 使用当前策略决策（用于执行）
            bc_action = bc_agent.decision(balls, my_targets, table)
            
            # 让专家给出标注（用于训练）
            expert_action = expert_agent.decision(balls, my_targets, table)
            expert_target = action_to_network_target(expert_action)
            
            # 添加专家标注的数据
            dataset.add(state_features, expert_target)
            
            # 执行BC智能体的动作（这是DAgger的关键：用自己的动作产生分布）
            step_info = env.take_shot(bc_action)
            
            done, info = env.get_done()
            if done:
                break
    
    print(f"[DAgger] 完成！共收集 {len(dataset)} 条新数据")
    
    if save_path:
        dataset.save(save_path)
    
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='收集专家数据')
    parser.add_argument('--n_games', type=int, default=100, help='收集的局数')
    parser.add_argument('--save_path', type=str, default='train/expert_data.npz', help='保存路径')
    parser.add_argument('--verbose', action='store_true', help='是否打印详细信息')
    
    args = parser.parse_args()
    
    collect_expert_data(
        n_games=args.n_games,
        save_path=args.save_path,
        verbose=args.verbose
    )
