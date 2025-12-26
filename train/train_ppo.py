"""
train_ppo.py - PPO自博弈微调训练脚本

功能:
- 加载BC预训练模型
- 使用PPO进行自博弈微调
- 支持对手混合策略（BasicAgent/BasicAgentPro/历史自己）
- 使用稀疏胜负奖励 + 小幅shaping
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
import copy
import random
from tqdm import tqdm
import argparse

from poolenv import PoolEnv
from agents import BasicAgent, BasicAgentPro
from networks import ActorCritic
from state_utils import (
    extract_state_features, action_to_network_target, network_output_to_action,
    state_dict_to_tensor, batch_states
)


class RolloutBuffer:
    """经验收集缓冲区"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
    
    def compute_returns_and_advantages(self, last_value: float, gamma: float = 0.99, 
                                        gae_lambda: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """计算GAE优势估计和回报"""
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values + [last_value])
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values[:-1]
        
        return returns, advantages
    
    def get_batches(self, batch_size: int, returns: np.ndarray, advantages: np.ndarray, device):
        """生成训练批次"""
        n_samples = len(self.states)
        indices = np.random.permutation(n_samples)
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]
            
            batch_states = [self.states[i] for i in batch_indices]
            batched_states = {
                'ball_features': torch.from_numpy(np.stack([s['ball_features'] for s in batch_states])).to(device),
                'pocket_positions': torch.from_numpy(np.stack([s['pocket_positions'] for s in batch_states])).to(device),
                'extra_features': torch.from_numpy(np.stack([s['extra_features'] for s in batch_states])).to(device)
            }
            
            yield (
                batched_states,
                torch.from_numpy(np.stack([self.actions[i] for i in batch_indices])).to(device),
                torch.from_numpy(np.array([self.log_probs[i] for i in batch_indices])).to(device),
                torch.from_numpy(returns[batch_indices]).float().to(device),
                torch.from_numpy(advantages[batch_indices]).float().to(device)
            )


class PPOTrainer:
    """PPO训练器"""
    
    def __init__(self, 
                 hidden_dim: int = 256,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 device: str = None):
        
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"[PPOTrainer] 使用设备: {self.device}")
        
        self.model = ActorCritic(hidden_dim=hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.buffer = RolloutBuffer()
    
    def load_bc_weights(self, bc_model_path: str):
        """加载BC预训练权重"""
        if not os.path.exists(bc_model_path):
            print(f"[PPOTrainer] BC模型文件 {bc_model_path} 不存在")
            return
        
        checkpoint = torch.load(bc_model_path, map_location=self.device)
        bc_state_dict = checkpoint['model_state_dict']
        
        # 尝试加载匹配的权重
        model_state_dict = self.model.state_dict()
        loaded_keys = []
        
        for key in bc_state_dict:
            # BC模型的state_encoder对应ActorCritic的state_encoder
            if key.startswith('state_encoder'):
                if key in model_state_dict:
                    model_state_dict[key] = bc_state_dict[key]
                    loaded_keys.append(key)
            # BC模型的action_head对应policy_head
            elif key.startswith('action_head'):
                new_key = key.replace('action_head', 'policy_head')
                if new_key in model_state_dict:
                    model_state_dict[new_key] = bc_state_dict[key]
                    loaded_keys.append(key)
        
        self.model.load_state_dict(model_state_dict)
        print(f"[PPOTrainer] 从BC模型加载了 {len(loaded_keys)} 个权重")
    
    def get_action(self, state_features: Dict, deterministic: bool = False) -> Tuple[Dict, float, float]:
        """获取动作"""
        self.model.eval()
        
        state_tensor = state_dict_to_tensor(state_features, self.device)
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.model.get_action_and_value(
                state_tensor, deterministic=deterministic
            )
        
        action_np = action.squeeze(0).cpu().numpy()
        action_dict = network_output_to_action(action_np)
        
        return action_dict, log_prob.item(), value.item(), action_np
    
    def compute_reward(self, step_info: Dict, done: bool, winner: str, 
                       is_my_turn: bool, my_player: str) -> float:
        """
        计算奖励
        
        Args:
            step_info: 击球结果信息
            done: 游戏是否结束
            winner: 赢家
            is_my_turn: 是否是我方回合
            my_player: 我方玩家标识 ('A' 或 'B')
        
        Returns:
            reward: 奖励值
        """
        reward = 0.0
        
        if done:
            # 稀疏的胜负奖励
            if winner == my_player:
                reward += 10.0  # 胜利
            elif winner == 'SAME':
                reward += 0.0   # 平局
            else:
                reward -= 10.0  # 失败
        
        if is_my_turn:
            # 小幅shaping奖励
            me_pocketed = step_info.get('ME_INTO_POCKET', [])
            enemy_pocketed = step_info.get('ENEMY_INTO_POCKET', [])
            
            # 进自己的球
            reward += len(me_pocketed) * 0.5
            # 帮对手进球
            reward -= len(enemy_pocketed) * 0.3
            
            # 犯规惩罚
            if step_info.get('WHITE_BALL_INTO_POCKET'):
                reward -= 1.0
            if step_info.get('FOUL_FIRST_HIT'):
                reward -= 0.5
            if step_info.get('NO_POCKET_NO_RAIL'):
                reward -= 0.5
            if step_info.get('NO_HIT'):
                reward -= 0.5
            
            # 黑八进袋
            if step_info.get('BLACK_BALL_INTO_POCKET'):
                if done and winner == my_player:
                    reward += 2.0  # 合法打进黑八
                else:
                    reward -= 5.0  # 非法黑八
        
        return reward
    
    def update(self, batch_size: int = 64, n_epochs: int = 10) -> Dict:
        """PPO更新"""
        self.model.train()
        
        # 计算最后一个状态的价值
        if len(self.buffer.states) == 0:
            return {}
        
        last_state = self.buffer.states[-1]
        last_state_tensor = state_dict_to_tensor(last_state, self.device)
        with torch.no_grad():
            _, _, _, last_value = self.model.get_action_and_value(last_state_tensor)
        
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value.item(), self.gamma, self.gae_lambda
        )
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        entropy_sum = 0
        n_updates = 0
        
        for epoch in range(n_epochs):
            for batch in self.buffer.get_batches(batch_size, returns, advantages, self.device):
                states, actions, old_log_probs, batch_returns, batch_advantages = batch
                
                # 获取当前策略的动作概率和价值
                _, new_log_probs, entropy, values = self.model.get_action_and_value(states, actions)
                
                # 计算比率
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # PPO裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(values, batch_returns)
                
                # 熵损失
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += entropy.mean().item()
                n_updates += 1
        
        self.buffer.clear()
        
        return {
            'total_loss': total_loss / max(1, n_updates),
            'policy_loss': policy_loss_sum / max(1, n_updates),
            'value_loss': value_loss_sum / max(1, n_updates),
            'entropy': entropy_sum / max(1, n_updates)
        }
    
    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"[PPOTrainer] 模型已保存到 {path}")
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[PPOTrainer] 模型已从 {path} 加载")


class PPOAgent:
    """用于对战的PPO智能体"""
    
    def __init__(self, trainer: PPOTrainer = None, model_path: str = None, 
                 hidden_dim: int = 256, device: str = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        if trainer is not None:
            self.model = trainer.model
        else:
            self.model = ActorCritic(hidden_dim=hidden_dim).to(self.device)
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
    
    def decision(self, balls, my_targets, table) -> Dict:
        state_features = extract_state_features(balls, my_targets, table)
        state_tensor = state_dict_to_tensor(state_features, self.device)
        
        with torch.no_grad():
            action, _, _, _ = self.model.get_action_and_value(state_tensor, deterministic=True)
        
        action_np = action.squeeze(0).cpu().numpy()
        return network_output_to_action(action_np)


class OpponentPool:
    """对手池：管理多种对手"""
    
    def __init__(self, include_basic: bool = True, include_pro: bool = True):
        self.opponents = []
        self.opponent_names = []
        self.weights = []
        
        if include_basic:
            self.opponents.append(BasicAgent())
            self.opponent_names.append("BasicAgent")
            self.weights.append(0.3)
        
        if include_pro:
            self.opponents.append(BasicAgentPro(n_simulations=30))  # 减少模拟次数加快训练
            self.opponent_names.append("BasicAgentPro")
            self.weights.append(0.5)
        
        # 历史自己 (将在训练中动态添加)
        self.self_history = []
        self.max_history = 5
    
    def add_self(self, model_state_dict, hidden_dim: int = 256, device = None):
        """添加历史版本的自己到对手池"""
        if len(self.self_history) >= self.max_history:
            self.self_history.pop(0)
        
        # 创建一个新的模型副本
        self_model = ActorCritic(hidden_dim=hidden_dim)
        if device:
            self_model = self_model.to(device)
        self_model.load_state_dict(copy.deepcopy(model_state_dict))
        self_model.eval()
        
        self.self_history.append(self_model)
    
    def sample_opponent(self, device=None) -> Tuple[object, str]:
        """采样一个对手"""
        # 计算权重
        total_weight = sum(self.weights)
        history_weight = 0.2 if self.self_history else 0
        
        if random.random() < history_weight and self.self_history:
            # 选择历史自己
            model = random.choice(self.self_history)
            return PPOAgent(model_path=None, device=device), "SelfHistory"
        else:
            # 从预定义对手中选择
            idx = random.choices(range(len(self.opponents)), weights=self.weights)[0]
            return self.opponents[idx], self.opponent_names[idx]


def train_ppo(
    bc_model_path: str = None,
    n_games: int = 1000,
    update_freq: int = 10,
    batch_size: int = 64,
    n_epochs: int = 10,
    save_freq: int = 100,
    save_path: str = 'train/ppo_model.pt',
    hidden_dim: int = 256,
    lr: float = 3e-4
):
    """
    PPO自博弈训练
    
    Args:
        bc_model_path: BC预训练模型路径
        n_games: 训练局数
        update_freq: 更新频率（每多少局更新一次）
        batch_size: 批次大小
        n_epochs: 每次更新的epoch数
        save_freq: 保存频率
        save_path: 模型保存路径
        hidden_dim: 隐藏层维度
        lr: 学习率
    """
    # 创建训练器
    trainer = PPOTrainer(hidden_dim=hidden_dim, lr=lr)
    
    # 加载BC预训练权重
    if bc_model_path:
        trainer.load_bc_weights(bc_model_path)
    
    # 创建对手池
    opponent_pool = OpponentPool(include_basic=True, include_pro=True)
    
    # 创建环境
    env = PoolEnv()
    target_ball_choices = ['solid', 'solid', 'stripe', 'stripe']
    
    # 统计
    win_count = 0
    lose_count = 0
    draw_count = 0
    recent_wins = deque(maxlen=100)
    
    print(f"[PPO训练] 开始训练，共 {n_games} 局")
    
    for game_idx in tqdm(range(n_games), desc="PPO训练"):
        # 采样对手
        opponent, opponent_name = opponent_pool.sample_opponent(trainer.device)
        
        # 随机决定我方是A还是B
        my_is_A = game_idx % 2 == 0
        my_player = 'A' if my_is_A else 'B'
        
        # 重置环境
        env.reset(target_ball=target_ball_choices[game_idx % 4])
        
        game_rewards = []
        
        while True:
            current_player = env.get_curr_player()
            is_my_turn = (current_player == my_player)
            
            balls, my_targets, table = env.get_observation(current_player)
            
            if is_my_turn:
                # 我方决策
                state_features = extract_state_features(balls, my_targets, table)
                action_dict, log_prob, value, action_np = trainer.get_action(state_features)
                
                # 执行动作
                step_info = env.take_shot(action_dict)
                
                done, info = env.get_done()
                winner = info.get('winner', None)
                
                # 计算奖励
                reward = trainer.compute_reward(step_info, done, winner, True, my_player)
                game_rewards.append(reward)
                
                # 存储经验
                trainer.buffer.add(state_features, action_np, reward, done, log_prob, value)
            else:
                # 对手决策
                step_info = env.take_shot(opponent.decision(balls, my_targets, table))
                
                done, info = env.get_done()
                winner = info.get('winner', None)
                
                # 对手回合也可以给我方一点信号（可选）
                if done and len(trainer.buffer.states) > 0:
                    # 更新最后一个状态的奖励
                    final_reward = trainer.compute_reward({}, done, winner, False, my_player)
                    trainer.buffer.rewards[-1] += final_reward
            
            if done:
                # 统计胜负
                if winner == my_player:
                    win_count += 1
                    recent_wins.append(1)
                elif winner == 'SAME':
                    draw_count += 1
                    recent_wins.append(0.5)
                else:
                    lose_count += 1
                    recent_wins.append(0)
                break
        
        # 定期更新
        if (game_idx + 1) % update_freq == 0 and len(trainer.buffer.states) > 0:
            losses = trainer.update(batch_size=batch_size, n_epochs=n_epochs)
            
            if (game_idx + 1) % (update_freq * 5) == 0:
                recent_winrate = sum(recent_wins) / len(recent_wins) if recent_wins else 0
                print(f"\n[Game {game_idx+1}] 对手: {opponent_name} | "
                      f"胜率: {recent_winrate:.2%} | "
                      f"损失: {losses.get('total_loss', 0):.4f}")
        
        # 定期添加历史自己
        if (game_idx + 1) % 200 == 0:
            opponent_pool.add_self(trainer.model.state_dict(), hidden_dim, trainer.device)
            print(f"[Game {game_idx+1}] 添加历史自己到对手池")
        
        # 定期保存
        if (game_idx + 1) % save_freq == 0:
            trainer.save(save_path.replace('.pt', f'_game{game_idx+1}.pt'))
    
    # 最终保存
    trainer.save(save_path)
    
    # 打印最终统计
    total = win_count + lose_count + draw_count
    print(f"\n[PPO训练完成]")
    print(f"总局数: {total}")
    print(f"胜: {win_count} ({win_count/total:.2%})")
    print(f"负: {lose_count} ({lose_count/total:.2%})")
    print(f"平: {draw_count} ({draw_count/total:.2%})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO自博弈训练')
    parser.add_argument('--bc_model', type=str, default='train/bc_model.pt', help='BC模型路径')
    parser.add_argument('--n_games', type=int, default=1000, help='训练局数')
    parser.add_argument('--update_freq', type=int, default=10, help='更新频率')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--n_epochs', type=int, default=10, help='PPO epochs')
    parser.add_argument('--save_freq', type=int, default=100, help='保存频率')
    parser.add_argument('--save_path', type=str, default='train/ppo_model.pt', help='保存路径')
    parser.add_argument('--hidden_dim', type=int, default=768, help='隐藏层维度')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    
    args = parser.parse_args()
    
    train_ppo(
        bc_model_path=args.bc_model,
        n_games=args.n_games,
        update_freq=args.update_freq,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        save_freq=args.save_freq,
        save_path=args.save_path,
        hidden_dim=args.hidden_dim,
        lr=args.lr
    )
