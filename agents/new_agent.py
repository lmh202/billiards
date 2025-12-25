"""
new_agent.py - 基于神经网络的智能体

用于evaluate.py测试，加载训练好的模型进行推理
决策速度比BasicAgentPro快得多
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from .agent import Agent

# 项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)


# ========== 内联网络定义 ==========

class StateEncoder(nn.Module):
    """状态编码器"""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ball_embed_dim = 8
        
        self.ball_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, self.ball_embed_dim)
        )
        
        self.pocket_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        self.global_encoder = nn.Sequential(
            nn.Linear(17 * self.ball_embed_dim + 6 * 8 + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = state['ball_features'].shape[0]
        
        ball_features = state['ball_features']
        ball_encoded = self.ball_encoder(ball_features)
        ball_flat = ball_encoded.view(batch_size, -1)
        
        pocket_positions = state['pocket_positions']
        pocket_encoded = self.pocket_encoder(pocket_positions)
        pocket_flat = pocket_encoded.view(batch_size, -1)
        
        extra = state['extra_features']
        combined = torch.cat([ball_flat, pocket_flat, extra], dim=-1)
        encoded = self.global_encoder(combined)
        
        return encoded


class ActorCritic(nn.Module):
    """Actor-Critic网络（用于PPO模型）"""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        self.state_encoder = StateEncoder(hidden_dim)
        self.hidden_dim = hidden_dim
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        
        self.log_std = nn.Parameter(torch.zeros(6))
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        encoded = self.state_encoder(state)
        mean = self.policy_head(encoded)
        return mean
    
    def get_action_and_value(self, state, action=None, deterministic=False):
        encoded = self.state_encoder(state)
        mean = self.policy_head(encoded)
        std = torch.exp(self.log_std).expand_as(mean)
        value = self.value_head(encoded).squeeze(-1)
        
        dist = torch.distributions.Normal(mean, std)
        
        if action is None:
            if deterministic:
                action = mean
            else:
                action = dist.rsample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value


class BCPolicyNetwork(nn.Module):
    """BC策略网络"""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        self.state_encoder = StateEncoder(hidden_dim)
        self.hidden_dim = hidden_dim
        
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 6)
        )
    
    def forward(self, state):
        encoded = self.state_encoder(state)
        action = self.action_head(encoded)
        return action


# ========== 状态处理工具 ==========

BALL_ID_TO_IDX = {
    'cue': 0,
    '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8,
    '9': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15
}

POCKET_ID_TO_IDX = {
    'lb': 0, 'lc': 1, 'lt': 2,
    'rb': 3, 'rc': 4, 'rt': 5
}


def extract_state_features(balls: Dict, my_targets: List[str], table) -> Dict[str, np.ndarray]:
    """提取状态特征"""
    table_w = table.w if hasattr(table, 'w') else 0.99
    table_l = table.l if hasattr(table, 'l') else 1.98
    
    ball_features = np.zeros((17, 6), dtype=np.float32)
    
    # 白球
    if 'cue' in balls:
        cue_ball = balls['cue']
        pos = cue_ball.state.rvw[0]
        is_pocketed = 1.0 if cue_ball.state.s == 4 else 0.0
        ball_features[0] = [
            pos[0] / table_l, pos[1] / table_w, is_pocketed, 0.0, 0.0, 1.0
        ]
    
    # 其他球
    for ball_id in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']:
        if ball_id not in balls:
            continue
        
        ball = balls[ball_id]
        idx = BALL_ID_TO_IDX[ball_id]
        pos = ball.state.rvw[0]
        is_pocketed = 1.0 if ball.state.s == 4 else 0.0
        is_my_target = 1.0 if ball_id in my_targets else 0.0
        is_eight = 1.0 if ball_id == '8' else 0.0
        
        ball_features[idx] = [
            pos[0] / table_l, pos[1] / table_w, is_pocketed, is_my_target, is_eight, 0.0
        ]
    
    # 袋口位置
    pocket_positions = np.zeros((6, 2), dtype=np.float32)
    for pocket_id, pocket in table.pockets.items():
        idx = POCKET_ID_TO_IDX.get(pocket_id)
        if idx is not None:
            center = pocket.center
            pocket_positions[idx] = [center[0] / table_l, center[1] / table_w]
    
    # 额外特征
    remaining_my = sum(1 for bid in my_targets if bid in balls and balls[bid].state.s != 4)
    remaining_enemy = sum(1 for bid in balls if bid not in my_targets and bid not in ['cue', '8'] 
                         and balls[bid].state.s != 4)
    is_clearing = 1.0 if remaining_my == 0 or my_targets == ['8'] else 0.0
    cue_x = balls['cue'].state.rvw[0][0] / table_l if 'cue' in balls else 0.5
    
    extra_features = np.array([
        remaining_my / 7.0, remaining_enemy / 7.0, is_clearing, cue_x
    ], dtype=np.float32)
    
    return {
        'ball_features': ball_features,
        'pocket_positions': pocket_positions,
        'extra_features': extra_features
    }


def network_output_to_action(output: np.ndarray) -> Dict[str, float]:
    """将网络输出转换为动作参数"""
    # V0
    V0 = float(np.clip(output[0] * 3.75 + 4.25, 0.5, 8.0))
    
    # phi: 从sin和cos恢复角度
    sin_phi = float(output[1])
    cos_phi = float(output[2])
    norm = np.sqrt(sin_phi**2 + cos_phi**2) + 1e-8
    sin_phi, cos_phi = sin_phi / norm, cos_phi / norm
    phi = float(np.degrees(np.arctan2(sin_phi, cos_phi)) % 360)
    
    # theta
    theta = float(np.clip(output[3] * 22.5 + 10, 0, 45))
    
    # a, b
    a = float(np.clip(output[4] * 0.25, -0.5, 0.5))
    b = float(np.clip(output[5] * 0.25, -0.5, 0.5))
    
    return {'V0': V0, 'phi': phi, 'theta': theta, 'a': a, 'b': b}


# ========== NewAgent 实现 ==========

class NewAgent(Agent):
    """基于神经网络的智能体
    
    支持加载PPO模型或BC模型进行推理
    推理速度远快于BasicAgentPro的MCTS搜索
    """
    
    def __init__(self, 
                 model_path: str = None, 
                 model_type: str = 'ppo',  # 'ppo' or 'bc'
                 hidden_dim: int = 256,
                 device: str = None):
        """
        初始化NewAgent
        
        Args:
            model_path: 模型文件路径，如果为None则尝试从默认位置加载
            model_type: 模型类型，'ppo'或'bc'
            hidden_dim: 隐藏层维度
            device: 运算设备
        """
        super().__init__()
        
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        
        # 创建模型
        if model_type == 'ppo':
            self.model = ActorCritic(hidden_dim=hidden_dim).to(self.device)
        else:
            self.model = BCPolicyNetwork(hidden_dim=hidden_dim).to(self.device)
        
        # 尝试加载模型
        self._load_model(model_path)
        
        self.model.eval()
        print(f"[NewAgent] 初始化完成，设备: {self.device}, 模型类型: {model_type}")
    
    def _load_model(self, model_path: str = None):
        """加载模型权重"""
        paths_to_try = []
        
        if model_path:
            paths_to_try.append(model_path)
        
        # 默认路径
        eval_dir = os.path.join(PROJECT_ROOT, 'eval')
        train_dir = os.path.join(PROJECT_ROOT, 'train', 'checkpoints')
        
        if self.model_type == 'ppo':
            paths_to_try.extend([
                os.path.join(eval_dir, 'model.pt'),
                os.path.join(eval_dir, 'ppo_model.pt'),
                os.path.join(train_dir, 'ppo_model.pt'),
            ])
        else:
            paths_to_try.extend([
                os.path.join(eval_dir, 'bc_model.pt'),
                os.path.join(train_dir, 'bc_model_best.pt'),
                os.path.join(train_dir, 'bc_model.pt'),
            ])
        
        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"[NewAgent] 成功加载模型: {path}")
                    return True
                except Exception as e:
                    print(f"[NewAgent] 加载模型 {path} 失败: {e}")
                    continue
        
        print("[NewAgent] 警告: 未找到模型文件，将使用随机初始化的权重")
        return False
    
    def decision(self, balls=None, my_targets=None, table=None) -> Dict[str, float]:
        """
        决策方法
        
        Args:
            balls: 球状态字典
            my_targets: 我方目标球列表
            table: 球桌对象
            
        Returns:
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
        """
        if balls is None:
            print("[NewAgent] 未收到有效的球局信息，使用随机动作")
            return self._random_action()
        
        try:
            # 处理目标球
            remaining_own = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ['8']
            
            # 提取状态特征
            state_features = extract_state_features(balls, my_targets, table)
            
            # 转换为张量
            state_tensor = {
                'ball_features': torch.from_numpy(state_features['ball_features']).unsqueeze(0).to(self.device),
                'pocket_positions': torch.from_numpy(state_features['pocket_positions']).unsqueeze(0).to(self.device),
                'extra_features': torch.from_numpy(state_features['extra_features']).unsqueeze(0).to(self.device)
            }
            
            # 推理
            with torch.no_grad():
                if self.model_type == 'ppo':
                    action, _, _, _ = self.model.get_action_and_value(state_tensor, deterministic=True)
                else:
                    action = self.model(state_tensor)
                
                output = action.squeeze(0).cpu().numpy()
            
            # 转换为动作字典
            action_dict = network_output_to_action(output)
            
            return action_dict
            
        except Exception as e:
            print(f"[NewAgent] 决策出错: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()