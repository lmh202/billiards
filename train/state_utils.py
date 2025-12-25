"""
state_utils.py - 状态处理工具

功能:
- 将台球环境的原始状态转换为神经网络输入格式
- 处理球的位置、状态等信息
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any
import copy


# 球的ID到索引的映射
BALL_ID_TO_IDX = {
    'cue': 0,
    '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8,
    '9': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15
}

# 袋口ID到索引的映射
POCKET_ID_TO_IDX = {
    'lb': 0, 'lc': 1, 'lt': 2,
    'rb': 3, 'rc': 4, 'rt': 5
}


def extract_state_features(balls: Dict, my_targets: List[str], table) -> Dict[str, np.ndarray]:
    """
    从环境状态中提取特征
    
    Args:
        balls: 球状态字典 {ball_id: Ball对象}
        my_targets: 我方目标球ID列表
        table: 球桌对象
        
    Returns:
        dict: 包含以下键:
            - ball_features: [17, 6] 球特征 (每个球: x, y, is_pocketed, is_my_target, is_eight, is_cue)
            - pocket_positions: [6, 2] 袋口位置
            - extra_features: [4] 额外特征 (剩余目标球数, 剩余对手球数, 清台状态, 白球x)
    """
    # 获取桌面尺寸用于归一化
    table_w = table.w if hasattr(table, 'w') else 0.99
    table_l = table.l if hasattr(table, 'l') else 1.98
    
    # 初始化球特征 [17, 6]
    # 特征: x_norm, y_norm, is_pocketed, is_my_target, is_eight, is_cue
    ball_features = np.zeros((17, 6), dtype=np.float32)
    
    # 白球
    if 'cue' in balls:
        cue_ball = balls['cue']
        pos = cue_ball.state.rvw[0]
        is_pocketed = 1.0 if cue_ball.state.s == 4 else 0.0
        ball_features[0] = [
            pos[0] / table_l,  # x归一化
            pos[1] / table_w,  # y归一化
            is_pocketed,
            0.0,  # 白球不是目标球
            0.0,  # 不是黑八
            1.0   # 是白球
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
            pos[0] / table_l,
            pos[1] / table_w,
            is_pocketed,
            is_my_target,
            is_eight,
            0.0  # 不是白球
        ]
    
    # 袋口位置 [6, 2]
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
        remaining_my / 7.0,  # 归一化
        remaining_enemy / 7.0,
        is_clearing,
        cue_x
    ], dtype=np.float32)
    
    return {
        'ball_features': ball_features,
        'pocket_positions': pocket_positions,
        'extra_features': extra_features
    }


def action_to_network_target(action: Dict[str, float]) -> np.ndarray:
    """
    将动作参数转换为网络训练目标
    
    Args:
        action: {'V0', 'phi', 'theta', 'a', 'b'}
        
    Returns:
        np.ndarray: [6] (V0_norm, sin_phi, cos_phi, theta_norm, a_norm, b_norm)
    """
    V0 = action['V0']
    phi = action['phi']
    theta = action['theta']
    a = action['a']
    b = action['b']
    
    # V0: [0.5, 8.0] -> 归一化 (线性)
    V0_norm = (V0 - 4.25) / 3.75
    
    # phi: 转换为sin和cos
    phi_rad = np.radians(phi)
    sin_phi = np.sin(phi_rad)
    cos_phi = np.cos(phi_rad)
    
    # theta: [0, 45] -> 归一化
    theta_norm = (theta - 10) / 22.5
    
    # a, b: [-0.5, 0.5] -> 归一化
    a_norm = a / 0.25
    b_norm = b / 0.25
    
    return np.array([V0_norm, sin_phi, cos_phi, theta_norm, a_norm, b_norm], dtype=np.float32)


def network_output_to_action(output: np.ndarray) -> Dict[str, float]:
    """
    将网络输出转换为动作参数
    
    Args:
        output: [6] (V0_norm, sin_phi, cos_phi, theta_norm, a_norm, b_norm)
        
    Returns:
        dict: {'V0', 'phi', 'theta', 'a', 'b'}
    """
    # V0
    V0 = float(np.clip(output[0] * 3.75 + 4.25, 0.5, 8.0))
    
    # phi: 从sin和cos恢复角度
    sin_phi = float(output[1])
    cos_phi = float(output[2])
    # 归一化确保在单位圆上
    norm = np.sqrt(sin_phi**2 + cos_phi**2) + 1e-8
    sin_phi, cos_phi = sin_phi / norm, cos_phi / norm
    phi = float(np.degrees(np.arctan2(sin_phi, cos_phi)) % 360)
    
    # theta
    theta = float(np.clip(output[3] * 22.5 + 10, 0, 45))
    
    # a, b
    a = float(np.clip(output[4] * 0.25, -0.5, 0.5))
    b = float(np.clip(output[5] * 0.25, -0.5, 0.5))
    
    return {'V0': V0, 'phi': phi, 'theta': theta, 'a': a, 'b': b}


def state_dict_to_tensor(state_dict: Dict[str, np.ndarray], device: torch.device = None) -> Dict[str, torch.Tensor]:
    """
    将numpy状态字典转换为PyTorch张量
    
    Args:
        state_dict: numpy格式的状态字典
        device: 目标设备
        
    Returns:
        dict: PyTorch张量格式的状态字典
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return {
        'ball_features': torch.from_numpy(state_dict['ball_features']).unsqueeze(0).to(device),
        'pocket_positions': torch.from_numpy(state_dict['pocket_positions']).unsqueeze(0).to(device),
        'extra_features': torch.from_numpy(state_dict['extra_features']).unsqueeze(0).to(device)
    }


def batch_states(states: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    将多个状态打包成批次
    
    Args:
        states: 状态字典列表
        
    Returns:
        dict: 批次状态字典
    """
    return {
        'ball_features': np.stack([s['ball_features'] for s in states]),
        'pocket_positions': np.stack([s['pocket_positions'] for s in states]),
        'extra_features': np.stack([s['extra_features'] for s in states])
    }


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state: Dict[str, np.ndarray], action: np.ndarray, 
             reward: float = 0.0, next_state: Dict[str, np.ndarray] = None, 
             done: bool = False):
        """添加一条经验"""
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """随机采样一个批次"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = [self.buffer[i][0] for i in indices]
        actions = np.array([self.buffer[i][1] for i in indices])
        rewards = np.array([self.buffer[i][2] for i in indices])
        next_states = [self.buffer[i][3] for i in indices]
        dones = np.array([self.buffer[i][4] for i in indices])
        
        return batch_states(states), actions, rewards, next_states, dones
    
    def sample_bc(self, batch_size: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """采样用于行为克隆的批次 (只需要state和action)"""
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        
        states = [self.buffer[i][0] for i in indices]
        actions = np.array([self.buffer[i][1] for i in indices])
        
        return batch_states(states), actions
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, path: str):
        """保存缓冲区到文件"""
        np.savez_compressed(
            path,
            buffer=np.array(self.buffer, dtype=object),
            position=self.position
        )
    
    def load(self, path: str):
        """从文件加载缓冲区"""
        data = np.load(path, allow_pickle=True)
        self.buffer = list(data['buffer'])
        self.position = int(data['position'])


class BCDataset:
    """行为克隆数据集"""
    
    def __init__(self):
        self.states = []
        self.actions = []
    
    def add(self, state: Dict[str, np.ndarray], action: np.ndarray):
        """添加一条数据"""
        self.states.append(state)
        self.actions.append(action)
    
    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """采样一个批次"""
        indices = np.random.choice(len(self.states), min(batch_size, len(self.states)), replace=False)
        
        batch_states_list = [self.states[i] for i in indices]
        batch_actions = np.array([self.actions[i] for i in indices])
        
        # 转换为批次格式
        batched = {
            'ball_features': np.stack([s['ball_features'] for s in batch_states_list]),
            'pocket_positions': np.stack([s['pocket_positions'] for s in batch_states_list]),
            'extra_features': np.stack([s['extra_features'] for s in batch_states_list])
        }
        
        return batched, batch_actions
    
    def __len__(self):
        return len(self.states)
    
    def save(self, path: str):
        """保存数据集"""
        np.savez_compressed(
            path,
            states=np.array(self.states, dtype=object),
            actions=np.array(self.actions)
        )
        print(f"[BCDataset] 保存 {len(self.states)} 条数据到 {path}")
    
    def load(self, path: str):
        """加载数据集"""
        data = np.load(path, allow_pickle=True)
        self.states = list(data['states'])
        self.actions = list(data['actions'])
        print(f"[BCDataset] 加载 {len(self.states)} 条数据从 {path}")
    
    def merge(self, other: 'BCDataset'):
        """合并另一个数据集"""
        self.states.extend(other.states)
        self.actions.extend(other.actions)
