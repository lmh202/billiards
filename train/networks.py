"""
networks.py - 神经网络模块

包含：
- PolicyNetwork: 策略网络 (Actor)，输出击球参数
- ValueNetwork: 价值网络 (Critic)，估计状态价值
- ActorCritic: 联合网络，用于PPO训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


class StateEncoder(nn.Module):
    """状态编码器：将球局状态编码为特征向量"""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 球状态嵌入：每个球有位置(x, y)、是否进袋、是否是目标球等信息
        # 最多17个球(cue + 1-15)，每个球编码维度为8
        self.ball_embed_dim = 8
        self.ball_encoder = nn.Sequential(
            nn.Linear(6, 32),  # 位置(2) + 是否进袋(1) + 是否目标球(1) + 是否黑八(1) + 是否白球(1)
            nn.ReLU(),
            nn.Linear(32, self.ball_embed_dim)
        )
        
        # 球袋位置编码
        self.pocket_encoder = nn.Sequential(
            nn.Linear(2, 16),  # 每个袋的位置
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # 全局特征编码
        # 17球 * 8 + 6袋 * 8 + 额外特征
        self.global_encoder = nn.Sequential(
            nn.Linear(17 * self.ball_embed_dim + 6 * 8 + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            state: 字典，包含:
                - ball_features: [batch, 17, 6] 球特征
                - pocket_positions: [batch, 6, 2] 袋位置
                - extra_features: [batch, 4] 额外特征(剩余球数等)
        
        Returns:
            encoded: [batch, hidden_dim] 编码后的状态
        """
        batch_size = state['ball_features'].shape[0]
        
        # 编码每个球
        ball_features = state['ball_features']  # [batch, 17, 6]
        ball_encoded = self.ball_encoder(ball_features)  # [batch, 17, 8]
        ball_flat = ball_encoded.view(batch_size, -1)  # [batch, 17*8]
        
        # 编码袋位置
        pocket_positions = state['pocket_positions']  # [batch, 6, 2]
        pocket_encoded = self.pocket_encoder(pocket_positions)  # [batch, 6, 8]
        pocket_flat = pocket_encoded.view(batch_size, -1)  # [batch, 6*8]
        
        # 额外特征
        extra = state['extra_features']  # [batch, 4]
        
        # 拼接所有特征
        combined = torch.cat([ball_flat, pocket_flat, extra], dim=-1)
        
        # 全局编码
        encoded = self.global_encoder(combined)
        
        return encoded


class PolicyNetwork(nn.Module):
    """策略网络：输出击球动作参数的分布"""
    
    def __init__(self, hidden_dim: int = 256, state_encoder: StateEncoder = None):
        super().__init__()
        
        self.state_encoder = state_encoder if state_encoder else StateEncoder(hidden_dim)
        self.hidden_dim = hidden_dim
        
        # 动作头
        # 输出: V0, sin(phi), cos(phi), theta, a, b
        # 使用sin(phi), cos(phi)来处理角度的周期性
        self.action_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # V0, sin_phi, cos_phi, theta, a, b
        )
        
        # 可学习的log标准差
        self.log_std = nn.Parameter(torch.zeros(6))
        
        # 初始化最后一层接近0
        nn.init.xavier_uniform_(self.action_mean[-1].weight, gain=0.01)
        nn.init.zeros_(self.action_mean[-1].bias)
    
    def forward(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 状态字典
            
        Returns:
            mean: [batch, 6] 动作均值
            std: [batch, 6] 动作标准差
        """
        encoded = self.state_encoder(state)
        mean = self.action_mean(encoded)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std
    
    def get_action(self, state: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作
        
        Args:
            state: 状态字典
            deterministic: 是否使用确定性策略
            
        Returns:
            action: [batch, 6] 动作
            log_prob: [batch] 对数概率
        """
        mean, std = self.forward(state)
        
        if deterministic:
            action = mean
            log_prob = torch.zeros(mean.shape[0], device=mean.device)
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate_actions(self, state: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估动作的对数概率和熵
        
        Args:
            state: 状态字典
            action: [batch, 6] 动作
            
        Returns:
            log_prob: [batch] 对数概率
            entropy: [batch] 熵
        """
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy
    
    def convert_network_output_to_action(self, output: torch.Tensor) -> Dict[str, float]:
        """
        将网络输出转换为实际的击球参数
        
        Args:
            output: [6] 网络输出 (V0_norm, sin_phi, cos_phi, theta_norm, a_norm, b_norm)
            
        Returns:
            dict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        output = output.detach().cpu().numpy()
        
        # V0: tanh激活后映射到[0.5, 8.0]
        V0 = float(np.clip(output[0] * 3.75 + 4.25, 0.5, 8.0))
        
        # phi: 从sin和cos恢复角度
        sin_phi, cos_phi = float(output[1]), float(output[2])
        phi = float(np.degrees(np.arctan2(sin_phi, cos_phi)) % 360)
        
        # theta: tanh激活后映射到[0, 45] (通常不需要太大角度)
        theta = float(np.clip(output[3] * 22.5 + 22.5, 0, 45))
        
        # a, b: tanh激活后映射到[-0.5, 0.5]
        a = float(np.clip(output[4] * 0.5, -0.5, 0.5))
        b = float(np.clip(output[5] * 0.5, -0.5, 0.5))
        
        return {'V0': V0, 'phi': phi, 'theta': theta, 'a': a, 'b': b}


class ValueNetwork(nn.Module):
    """价值网络：估计状态价值"""
    
    def __init__(self, hidden_dim: int = 256, state_encoder: StateEncoder = None):
        super().__init__()
        
        self.state_encoder = state_encoder if state_encoder else StateEncoder(hidden_dim)
        self.hidden_dim = hidden_dim
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态字典
            
        Returns:
            value: [batch, 1] 状态价值
        """
        encoded = self.state_encoder(state)
        value = self.value_head(encoded)
        return value


class ActorCritic(nn.Module):
    """Actor-Critic联合网络，共享状态编码器"""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        # 共享的状态编码器
        self.state_encoder = StateEncoder(hidden_dim)
        self.hidden_dim = hidden_dim
        
        # 策略头 (Actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # V0, sin_phi, cos_phi, theta, a, b
        )
        
        # 可学习的log标准差
        self.log_std = nn.Parameter(torch.zeros(6))
        
        # 价值头 (Critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """正交初始化"""
        for module in [self.policy_head, self.value_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)
        
        # 策略头最后一层使用较小的增益
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
    
    def forward(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            mean: [batch, 6] 策略均值
            std: [batch, 6] 策略标准差
            value: [batch, 1] 状态价值
        """
        encoded = self.state_encoder(state)
        
        mean = self.policy_head(encoded)
        std = torch.exp(self.log_std).expand_as(mean)
        value = self.value_head(encoded)
        
        return mean, std, value
    
    def get_action_and_value(self, state: Dict[str, torch.Tensor], 
                              action: torch.Tensor = None,
                              deterministic: bool = False):
        """
        获取动作和价值
        
        Args:
            state: 状态字典
            action: 如果提供，则计算给定动作的对数概率
            deterministic: 是否确定性策略
            
        Returns:
            action: [batch, 6] 动作
            log_prob: [batch] 对数概率
            entropy: [batch] 熵
            value: [batch] 价值
        """
        mean, std, value = self.forward(state)
        value = value.squeeze(-1)
        
        dist = torch.distributions.Normal(mean, std)
        
        if action is None:
            if deterministic:
                action = mean
            else:
                action = dist.rsample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value
    
    def convert_output_to_action(self, output: torch.Tensor) -> Dict[str, float]:
        """将网络输出转换为实际的击球参数"""
        output = output.detach().cpu().numpy()
        
        # V0: 使用tanh将输出映射到合理范围，然后缩放到[0.5, 8.0]
        V0_raw = np.tanh(output[0])  # [-1, 1]
        V0 = float(np.clip(V0_raw * 3.75 + 4.25, 0.5, 8.0))
        
        # phi: 从sin和cos恢复角度
        sin_phi = np.tanh(output[1])
        cos_phi = np.tanh(output[2])
        phi = float(np.degrees(np.arctan2(sin_phi, cos_phi)) % 360)
        
        # theta: 映射到[0, 45]度
        theta_raw = np.tanh(output[3])
        theta = float(np.clip(theta_raw * 22.5 + 22.5, 0, 45))
        
        # a, b: 映射到[-0.5, 0.5]
        a = float(np.clip(np.tanh(output[4]) * 0.5, -0.5, 0.5))
        b = float(np.clip(np.tanh(output[5]) * 0.5, -0.5, 0.5))
        
        return {'V0': V0, 'phi': phi, 'theta': theta, 'a': a, 'b': b}


class BCPolicyNetwork(nn.Module):
    """
    行为克隆专用策略网络
    直接预测动作参数，不输出分布
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        self.state_encoder = StateEncoder(hidden_dim)
        self.hidden_dim = hidden_dim
        
        # 动作预测头
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 6)  # V0_norm, sin_phi, cos_phi, theta_norm, a_norm, b_norm
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.action_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Returns:
            action: [batch, 6] 动作 (归一化值)
        """
        encoded = self.state_encoder(state)
        action = self.action_head(encoded)
        return action
    
    def convert_output_to_action(self, output: torch.Tensor) -> Dict[str, float]:
        """将网络输出转换为实际的击球参数"""
        output = output.detach().cpu().numpy()
        
        # V0: sigmoid映射到[0.5, 8.0]
        V0 = float(np.clip(output[0] * 3.75 + 4.25, 0.5, 8.0))
        
        # phi: 从sin和cos恢复角度 (输出范围约-1到1，但不强制)
        sin_phi = float(output[1])
        cos_phi = float(output[2])
        # 归一化到单位圆
        norm = np.sqrt(sin_phi**2 + cos_phi**2) + 1e-8
        sin_phi, cos_phi = sin_phi / norm, cos_phi / norm
        phi = float(np.degrees(np.arctan2(sin_phi, cos_phi)) % 360)
        
        # theta: 映射到[0, 45]
        theta = float(np.clip(output[3] * 22.5 + 10, 0, 45))
        
        # a, b: 映射到[-0.5, 0.5]
        a = float(np.clip(output[4] * 0.25, -0.5, 0.5))
        b = float(np.clip(output[5] * 0.25, -0.5, 0.5))
        
        return {'V0': V0, 'phi': phi, 'theta': theta, 'a': a, 'b': b}
