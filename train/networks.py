"""
networks.py - PPO神经网络架构

包含:
- BallEncoder: 球特征编码器
- StateEncoder: 整体状态编码器
- ActorNetwork: 策略网络
- CriticNetwork: 价值网络
- PPOActorCritic: 完整的Actor-Critic网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Dict

from train.config import NETWORK_CONFIG, ACTION_BOUNDS, DEVICE


class BallEncoder(nn.Module):
    """球特征编码器 - 使用注意力机制处理不同数量的球"""
    
    def __init__(self, feature_dim: int = 7, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 球特征嵌入
        self.ball_embed = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
        # 自注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=4, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, ball_features: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            ball_features: [batch, num_balls, feature_dim]
            mask: [batch, num_balls], True表示该位置有效
        
        Returns:
            encoded: [batch, embed_dim]
        """
        # 嵌入每个球的特征
        embedded = self.ball_embed(ball_features)  # [batch, num_balls, embed_dim]
        
        # 创建注意力mask (True表示忽略)
        if mask is not None:
            attn_mask = ~mask  # 反转mask
        else:
            attn_mask = None
            
        # 自注意力
        attn_out, _ = self.attention(
            embedded, embedded, embedded, 
            key_padding_mask=attn_mask
        )
        attn_out = self.norm(attn_out + embedded)
        
        # 全局平均池化 (考虑mask)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (attn_out * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            pooled = attn_out.mean(dim=1)
            
        return pooled


class StateEncoder(nn.Module):
    """状态编码器 - 整合所有信息"""
    
    def __init__(self, config: dict = NETWORK_CONFIG):
        super().__init__()
        self.config = config
        
        # 球特征编码器
        self.ball_encoder = BallEncoder(
            feature_dim=config['ball_feature_dim'],
            embed_dim=config['ball_embed_dim']
        )
        
        # 球袋特征编码
        self.pocket_embed = nn.Sequential(
            nn.Linear(config['pocket_feature_dim'] * config['num_pockets'], 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, config['pocket_embed_dim']),
            nn.ReLU()
        )
        
        # 目标球信息编码
        self.target_embed = nn.Sequential(
            nn.Linear(config['max_balls'], 32),  # one-hot编码目标球
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # 游戏状态编码 (剩余球数等)
        self.game_state_embed = nn.Sequential(
            nn.Linear(4, 16),  # [own_remaining, enemy_remaining, is_targeting_8, hit_count_normalized]
            nn.ReLU()
        )
        
        # 整合所有特征
        total_dim = (config['ball_embed_dim'] + 
                     config['pocket_embed_dim'] + 
                     32 +  # target embed
                     16)   # game state embed
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, config['state_embed_dim']),
            nn.LayerNorm(config['state_embed_dim']),
            nn.ReLU(),
            nn.Linear(config['state_embed_dim'], config['state_embed_dim']),
            nn.LayerNorm(config['state_embed_dim']),
            nn.ReLU()
        )
        
        self.output_dim = config['state_embed_dim']
        
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            obs: 包含以下key的字典:
                - ball_features: [batch, max_balls, 7]
                - ball_mask: [batch, max_balls]
                - pocket_features: [batch, 6*3]
                - target_mask: [batch, max_balls]
                - game_state: [batch, 4]
        
        Returns:
            state_embed: [batch, state_embed_dim]
        """
        # 编码球特征
        ball_embed = self.ball_encoder(obs['ball_features'], obs['ball_mask'])
        
        # 编码球袋
        pocket_embed = self.pocket_embed(obs['pocket_features'])
        
        # 编码目标球
        target_embed = self.target_embed(obs['target_mask'].float())
        
        # 编码游戏状态
        game_embed = self.game_state_embed(obs['game_state'])
        
        # 融合
        concat = torch.cat([ball_embed, pocket_embed, target_embed, game_embed], dim=-1)
        state_embed = self.fusion(concat)
        
        return state_embed


class ActorNetwork(nn.Module):
    """策略网络 - 输出连续动作的均值和标准差"""
    
    def __init__(self, state_dim: int, action_dim: int = 5, hidden_dim: int = 512):
        super().__init__()
        
        self.action_dim = action_dim
        
        # 共享隐藏层
        self.hidden = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 均值头
        self.mean_head = nn.Linear(hidden_dim // 2, action_dim)
        
        # 对数标准差 (可学习参数)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # 动作边界
        self.register_buffer('action_low', torch.tensor([
            ACTION_BOUNDS['V0'][0],
            ACTION_BOUNDS['phi'][0], 
            ACTION_BOUNDS['theta'][0],
            ACTION_BOUNDS['a'][0],
            ACTION_BOUNDS['b'][0]
        ], dtype=torch.float32))
        
        self.register_buffer('action_high', torch.tensor([
            ACTION_BOUNDS['V0'][1],
            ACTION_BOUNDS['phi'][1],
            ACTION_BOUNDS['theta'][1],
            ACTION_BOUNDS['a'][1],
            ACTION_BOUNDS['b'][1]
        ], dtype=torch.float32))
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean: [batch, action_dim] 在 [-1, 1] 范围内
            std: [batch, action_dim]
        """
        hidden = self.hidden(state)
        mean = torch.tanh(self.mean_head(hidden))  # [-1, 1]
        std = torch.exp(torch.clamp(self.log_std, -20, 2))
        return mean, std.expand_as(mean)
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作、对数概率和熵
        
        Returns:
            action: 原始动作空间的动作
            log_prob: 对数概率
            entropy: 熵
        """
        mean, std = self.forward(state)
        
        if deterministic:
            action_normalized = mean
        else:
            dist = Normal(mean, std)
            action_normalized = dist.rsample()  # 重参数化采样
            
        # 限制在[-1, 1]范围内
        action_normalized = torch.clamp(action_normalized, -1, 1)
        
        # 计算对数概率
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action_normalized).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        # 转换到原始动作空间
        action = self.scale_action(action_normalized)
        
        return action, log_prob, entropy
    
    def scale_action(self, action_normalized: torch.Tensor) -> torch.Tensor:
        """将归一化动作 [-1, 1] 转换到原始动作空间"""
        return (action_normalized + 1) / 2 * (self.action_high - self.action_low) + self.action_low
    
    def unscale_action(self, action: torch.Tensor) -> torch.Tensor:
        """将原始动作转换到归一化空间 [-1, 1]"""
        return 2 * (action - self.action_low) / (self.action_high - self.action_low) - 1
    
    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估给定动作的对数概率和熵"""
        mean, std = self.forward(state)
        
        # 将动作转换回归一化空间
        action_normalized = self.unscale_action(action)
        action_normalized = torch.clamp(action_normalized, -1 + 1e-6, 1 - 1e-6)
        
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action_normalized).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """价值网络"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)


class PPOActorCritic(nn.Module):
    """完整的PPO Actor-Critic 网络"""
    
    def __init__(self, config: dict = NETWORK_CONFIG):
        super().__init__()
        
        self.config = config
        
        # 状态编码器
        self.state_encoder = StateEncoder(config)
        
        # Actor和Critic
        self.actor = ActorNetwork(
            state_dim=self.state_encoder.output_dim,
            action_dim=config['action_dim'],
            hidden_dim=config['hidden_dim']
        )
        
        self.critic = CriticNetwork(
            state_dim=self.state_encoder.output_dim,
            hidden_dim=config['hidden_dim']
        )
        
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action_mean: [batch, action_dim]
            value: [batch]
        """
        state = self.state_encoder(obs)
        mean, std = self.actor(state)
        value = self.critic(state)
        return mean, value
    
    def get_action(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        """获取动作用于环境交互"""
        state = self.state_encoder(obs)
        action, log_prob, entropy = self.actor.get_action(state, deterministic)
        value = self.critic(state)
        return action, log_prob, value, entropy
    
    def evaluate(self, obs: Dict[str, torch.Tensor], action: torch.Tensor):
        """评估给定状态和动作"""
        state = self.state_encoder(obs)
        log_prob, entropy = self.actor.evaluate_action(state, action)
        value = self.critic(state)
        return log_prob, value, entropy
    
    def get_value(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """仅获取价值估计"""
        state = self.state_encoder(obs)
        return self.critic(state)
