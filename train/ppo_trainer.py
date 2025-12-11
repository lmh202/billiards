"""
ppo_trainer.py - PPO训练器

实现PPO算法的完整训练流程，包括:
- 经验收集
- GAE优势估计
- PPO策略更新
- 模型保存和评估
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import time
from collections import deque
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.config import PPO_CONFIG, TRAIN_CONFIG, NETWORK_CONFIG, DEVICE
from train.networks import PPOActorCritic
from train.pool_rl_env import PoolRLEnv, VectorPoolEnv


class RolloutBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, buffer_size: int, num_envs: int, device: str = 'cpu'):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
        self.ptr = 0
        
    def add(self, obs: Dict[str, torch.Tensor], action: torch.Tensor, 
            log_prob: torch.Tensor, reward: np.ndarray, 
            done: np.ndarray, value: torch.Tensor):
        """添加一步经验"""
        self.observations.append({k: v.cpu() for k, v in obs.items()})
        self.actions.append(action.cpu())
        self.log_probs.append(log_prob.cpu())
        self.rewards.append(torch.from_numpy(reward).float())
        self.dones.append(torch.from_numpy(done).float())
        self.values.append(value.cpu())
        self.ptr += 1
        
    def compute_returns_and_advantages(self, last_values: torch.Tensor, 
                                        gamma: float, gae_lambda: float):
        """
        计算GAE优势估计和回报
        """
        last_values = last_values.cpu()
        
        advantages = torch.zeros(self.ptr, self.num_envs)
        returns = torch.zeros(self.ptr, self.num_envs)
        
        last_gae_lam = 0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = self.values[t + 1]
                
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            
        returns = advantages + torch.stack(self.values)
        
        self.advantages = advantages.view(-1)
        self.returns = returns.view(-1)
        
        # Flatten其他数据
        self.flat_observations = {
            k: torch.cat([obs[k] for obs in self.observations], dim=0)
            for k in self.observations[0].keys()
        }
        self.flat_actions = torch.cat(self.actions, dim=0)
        self.flat_log_probs = torch.cat(self.log_probs, dim=0)
        self.flat_values = torch.cat(self.values, dim=0)
        
    def get_batches(self, batch_size: int):
        """生成随机mini-batch"""
        total_size = self.ptr * self.num_envs
        indices = np.random.permutation(total_size)
        
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_indices = indices[start:end]
            
            yield {
                'observations': {k: v[batch_indices].to(self.device) 
                                for k, v in self.flat_observations.items()},
                'actions': self.flat_actions[batch_indices].to(self.device),
                'old_log_probs': self.flat_log_probs[batch_indices].to(self.device),
                'advantages': self.advantages[batch_indices].to(self.device),
                'returns': self.returns[batch_indices].to(self.device),
                'old_values': self.flat_values[batch_indices].to(self.device)
            }
            
    def clear(self):
        """清空缓冲区"""
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.ptr = 0


class PPOTrainer:
    """PPO训练器"""
    
    def __init__(self, 
                 config: dict = PPO_CONFIG,
                 train_config: dict = TRAIN_CONFIG,
                 network_config: dict = NETWORK_CONFIG,
                 device: str = None):
        
        self.config = config
        self.train_config = train_config
        self.network_config = network_config
        self.device = device if device else DEVICE
        
        # 创建网络
        self.policy = PPOActorCritic(network_config).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.state_encoder.parameters(), 'lr': config['lr_actor']},
            {'params': self.policy.actor.parameters(), 'lr': config['lr_actor']},
            {'params': self.policy.critic.parameters(), 'lr': config['lr_critic']}
        ])
        
        # 创建环境
        self.num_envs = train_config['num_envs']
        self.envs = VectorPoolEnv(
            num_envs=self.num_envs,
            device=self.device,
            opponent_type='self',
            enable_noise=True
        )
        
        # 设置自我对弈
        self.envs.set_opponent_policy(self.policy)
        
        # 创建缓冲区
        self.buffer = RolloutBuffer(
            buffer_size=config['update_freq'] // self.num_envs,
            num_envs=self.num_envs,
            device=self.device
        )
        
        # 统计
        self.total_timesteps = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.wins = deque(maxlen=100)
        
        # 创建目录
        os.makedirs(train_config['checkpoint_dir'], exist_ok=True)
        os.makedirs(train_config['log_dir'], exist_ok=True)
        
        # 日志
        self.log_file = open(
            os.path.join(train_config['log_dir'], 
                        f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            'w'
        )
        
    def collect_rollouts(self, n_steps: int):
        """收集经验"""
        self.policy.eval()
        
        obs = self.envs.reset()
        episode_rewards_sum = np.zeros(self.num_envs)
        episode_lengths_count = np.zeros(self.num_envs)
        
        for step in range(n_steps):
            with torch.no_grad():
                obs_device = {k: v.to(self.device) for k, v in obs.items()}
                action, log_prob, value, _ = self.policy.get_action(obs_device)
                
            action_np = action.cpu().numpy()
            next_obs, rewards, dones, infos = self.envs.step(action_np)
            
            # 添加到缓冲区
            self.buffer.add(obs, action, log_prob, rewards, dones, value)
            
            # 统计
            episode_rewards_sum += rewards
            episode_lengths_count += 1
            
            for i, done in enumerate(dones):
                if done:
                    self.episode_rewards.append(episode_rewards_sum[i])
                    self.episode_lengths.append(episode_lengths_count[i])
                    
                    # 检查胜负
                    info = infos[i]
                    if 'game_info' in info and info['game_info'].get('winner'):
                        winner = info['game_info']['winner']
                        agent_player = info.get('agent_player', 'A')
                        if winner == agent_player:
                            self.wins.append(1)
                        elif winner == 'SAME':
                            self.wins.append(0.5)
                        else:
                            self.wins.append(0)
                            
                    episode_rewards_sum[i] = 0
                    episode_lengths_count[i] = 0
                    
            obs = next_obs
            self.total_timesteps += self.num_envs
            
        # 计算最后一步的价值用于GAE
        with torch.no_grad():
            obs_device = {k: v.to(self.device) for k, v in obs.items()}
            last_values = self.policy.get_value(obs_device)
            
        self.buffer.compute_returns_and_advantages(
            last_values,
            self.config['gamma'],
            self.config['gae_lambda']
        )
        
    def update(self):
        """PPO策略更新"""
        self.policy.train()
        
        # 标准化优势
        advantages = self.buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages
        
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        entropy_sum = 0
        n_updates = 0
        
        for epoch in range(self.config['ppo_epochs']):
            for batch in self.buffer.get_batches(self.config['mini_batch_size']):
                # 评估当前策略
                new_log_probs, values, entropy = self.policy.evaluate(
                    batch['observations'],
                    batch['actions']
                )
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch['old_log_probs'])
                
                # 裁剪的策略损失
                advantages = batch['advantages']
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 
                                   1 - self.config['clip_epsilon'],
                                   1 + self.config['clip_epsilon']) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失 (可选裁剪)
                value_loss = F.mse_loss(values, batch['returns'])
                
                # 熵损失
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = (policy_loss + 
                       self.config['value_loss_coef'] * value_loss +
                       self.config['entropy_coef'] * entropy_loss)
                
                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 
                                        self.config['max_grad_norm'])
                self.optimizer.step()
                
                total_loss += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += entropy.mean().item()
                n_updates += 1
                
        self.buffer.clear()
        
        return {
            'total_loss': total_loss / n_updates,
            'policy_loss': policy_loss_sum / n_updates,
            'value_loss': value_loss_sum / n_updates,
            'entropy': entropy_sum / n_updates
        }
    
    def train(self, total_timesteps: int = None):
        """主训练循环"""
        if total_timesteps is None:
            total_timesteps = self.train_config['total_timesteps']
            
        n_steps = self.config['update_freq'] // self.num_envs
        n_iterations = total_timesteps // self.config['update_freq']
        
        print(f"开始训练 - 设备: {self.device}")
        print(f"总步数: {total_timesteps}, 迭代次数: {n_iterations}")
        print(f"并行环境数: {self.num_envs}, 每次更新步数: {n_steps}")
        
        start_time = time.time()
        
        for iteration in range(n_iterations):
            # 收集经验
            self.collect_rollouts(n_steps)
            
            # 更新策略
            losses = self.update()
            
            # 日志
            if (iteration + 1) % (self.train_config['log_freq'] // self.config['update_freq']) == 0:
                elapsed = time.time() - start_time
                fps = self.total_timesteps / elapsed
                
                avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                win_rate = np.mean(self.wins) if self.wins else 0
                
                log_msg = (
                    f"[Iter {iteration+1}/{n_iterations}] "
                    f"步数: {self.total_timesteps}, FPS: {fps:.0f}, "
                    f"奖励: {avg_reward:.2f}, 回合长度: {avg_length:.1f}, "
                    f"胜率: {win_rate:.2%}, "
                    f"损失: {losses['total_loss']:.4f}, "
                    f"策略损失: {losses['policy_loss']:.4f}, "
                    f"价值损失: {losses['value_loss']:.4f}, "
                    f"熵: {losses['entropy']:.4f}"
                )
                print(log_msg)
                self.log_file.write(log_msg + '\n')
                self.log_file.flush()
                
            # 保存检查点
            if (iteration + 1) % (self.train_config['save_freq'] // self.config['update_freq']) == 0:
                self.save_checkpoint(f'checkpoint_{self.total_timesteps}.pt')
                
            # 评估
            if (iteration + 1) % (self.train_config['eval_freq'] // self.config['update_freq']) == 0:
                eval_results = self.evaluate()
                eval_msg = f"[评估] 胜率: {eval_results['win_rate']:.2%}, 平均奖励: {eval_results['avg_reward']:.2f}"
                print(eval_msg)
                self.log_file.write(eval_msg + '\n')
                self.log_file.flush()
                
        # 训练结束，保存最终模型
        self.save_checkpoint('final_model.pt')
        self.log_file.close()
        print("训练完成!")
        
    def evaluate(self, n_episodes: int = None) -> Dict:
        """评估当前策略"""
        if n_episodes is None:
            n_episodes = self.train_config['eval_episodes']
            
        self.policy.eval()
        
        eval_env = PoolRLEnv(opponent_type='random', enable_noise=True)
        
        total_rewards = []
        wins = 0
        losses = 0
        draws = 0
        
        for ep in range(n_episodes):
            obs = eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    obs_device = {k: v.to(self.device) for k, v in obs.items()}
                    action, _, _, _ = self.policy.get_action(obs_device, deterministic=True)
                    
                action_np = action.cpu().numpy().squeeze()
                obs, reward, done, info = eval_env.step(action_np)
                episode_reward += reward
                
            total_rewards.append(episode_reward)
            
            if 'game_info' in info and info['game_info'].get('winner'):
                winner = info['game_info']['winner']
                agent_player = info.get('agent_player', 'A')
                if winner == agent_player:
                    wins += 1
                elif winner == 'SAME':
                    draws += 1
                else:
                    losses += 1
                    
        return {
            'avg_reward': np.mean(total_rewards),
            'win_rate': wins / n_episodes,
            'draw_rate': draws / n_episodes,
            'loss_rate': losses / n_episodes,
            'wins': wins,
            'draws': draws,
            'losses': losses
        }
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        path = os.path.join(self.train_config['checkpoint_dir'], filename)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_timesteps': self.total_timesteps,
            'config': self.config,
            'network_config': self.network_config
        }, path)
        print(f"检查点已保存: {path}")
        
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_timesteps = checkpoint['total_timesteps']
        print(f"检查点已加载: {path}")


if __name__ == '__main__':
    # 简单测试
    print(f"使用设备: {DEVICE}")
    
    trainer = PPOTrainer(device=DEVICE)
    
    # 测试收集一轮经验
    print("测试经验收集...")
    trainer.collect_rollouts(10)
    print(f"缓冲区大小: {trainer.buffer.ptr}")
    
    # 测试更新
    print("测试策略更新...")
    losses = trainer.update()
    print(f"损失: {losses}")
    
    print("测试通过!")
