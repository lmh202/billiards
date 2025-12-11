"""
MuZero训练脚本 - 用于台球环境

实现MuZero算法的完整训练流程，包括：
1. 神经网络组件（表示、动力学、预测）
2. MCTS搜索算法
3. 经验回放缓冲区
4. 训练循环
"""

import os
import sys
import math
import copy
import random
import numpy as np
from collections import deque
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 添加父目录到路径以导入poolenv和agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poolenv import PoolEnv, save_balls_state
from agent import analyze_shot_for_reward


# ========================= 配置参数 =========================
class MuZeroConfig:
    """MuZero配置参数"""
    
    def __init__(self):
        # 环境参数
        self.v0_bins = 10
        self.phi_bins = 10
        self.action_space_size = self.v0_bins * self.phi_bins  # 离散化动作空间大小
        self.state_channels = 32  # 状态表示的通道数
        self.hidden_state_size = 256  # 隐藏状态向量维度

        # 网络参数
        self.encoding_size = 64  # 编码层大小
        self.fc_reward_layers = [64]  # 奖励预测层
        self.fc_value_layers = [64]  # 价值预测层
        self.fc_policy_layers = [64]  # 策略预测层

        # MCTS参数
        self.num_simulations = 50  # MCTS模拟次数
        self.num_simulations_eval = 30  # 评估时的模拟次数
        self.max_moves = 60  # 最大步数
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # 训练参数
        self.training_steps = 4000  # 总训练步数
        self.batch_size = 16  # 批大小
        self.num_unroll_steps = 5  # 展开步数
        self.td_steps = 10  # TD(n)步数
        self.lr_init = 0.001  # 初始学习率
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 5000
        self.weight_decay = 1e-4

        # 价值和奖励的缩放（根据图片评分标准）
        self.value_support_min = -400
        self.value_support_max = 400
        self.reward_support_min = -20
        self.reward_support_max = 20

        # 经验回放
        self.replay_buffer_size = 8000
        self.priority_alpha = 0.6  # 优先级指数
        self.priority_beta = 0.4  # 重要性采样指数

        # 自我对弈
        self.num_actors = 1  # 并发actor数量
        self.self_play_games = 40  # 每次迭代的自我对弈局数

        # 保存和日志
        self.checkpoint_interval = 1  # 保存间隔
        self.save_dir = "checkpoints"

        # 策略平滑，避免早期动作概率塌缩
        self.policy_smoothing = 1e-3
        self.policy_uniform_mix = 0.2  # 与均匀分布混合的权重
        self.v0_uniform_mix = 0.3
        self.phi_uniform_mix = 0.1

        # 物理模拟辅助（混合策略）
        self.use_physics_assist = True  # 是否在 MCTS 中启用物理模拟辅助
        self.physics_assist_samples = 15  # 对多少个候选动作进行物理模拟
        self.physics_assist_weight = 0.5  # 物理得分在先验修正中的权重

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================= 工具函数 =========================

def discretize_action(action_dict, config):
    """将连续动作离散化为索引"""
    # 简化的离散化：将V0和phi组合
    v0_bins = getattr(config, 'v0_bins', 10)
    phi_bins = getattr(config, 'phi_bins', 10)
    
    v0_idx = int((action_dict['V0'] - 0.5) / 7.5 * v0_bins)
    phi_idx = int(action_dict['phi'] / 360 * phi_bins)
    
    v0_idx = np.clip(v0_idx, 0, v0_bins - 1)
    phi_idx = np.clip(phi_idx, 0, phi_bins - 1)
    
    action_idx = v0_idx * phi_bins + phi_idx
    return action_idx


def action_index_to_dict(action_idx, config):
    """将动作索引转换回动作字典"""
    v0_min, v0_max = 0.5, 8.0
    v0_bins = getattr(config, 'v0_bins', 10)
    phi_bins = getattr(config, 'phi_bins', 10)

    v0_idx = action_idx // phi_bins
    phi_idx = action_idx % phi_bins

    v0 = v0_min + (v0_idx + 0.5) * (v0_max - v0_min) / v0_bins
    phi = (phi_idx + 0.5) * 360 / phi_bins

    action = {
        'V0': float(v0),
        'phi': float(phi),
        'theta': 45.0,
        'a': 0.0,
        'b': 0.0
    }
    return action


def sample_action_index(action_probs, config):
    """在训练阶段按照分离的V0/phi分布采样动作，以保证多样性"""
    probs = np.array(action_probs, dtype=np.float64)
    if probs.ndim != 1 or len(probs) != config.action_space_size:
        probs = np.ones(config.action_space_size, dtype=np.float64) / config.action_space_size
    probs = np.maximum(probs, 1e-12)
    probs /= probs.sum()
    v0_bins = getattr(config, 'v0_bins', 10)
    phi_bins = getattr(config, 'phi_bins', 10)
    policy_2d = probs.reshape(v0_bins, phi_bins)
    v0_marginal = policy_2d.sum(axis=1)
    v0_mix = getattr(config, 'v0_uniform_mix', 0.0)
    if v0_mix > 0:
        v0_marginal = (1 - v0_mix) * v0_marginal + v0_mix / v0_bins
    if v0_marginal.sum() == 0:
        v0_marginal = np.ones(v0_bins) / v0_bins
    else:
        v0_marginal /= v0_marginal.sum()
    v0_idx = np.random.choice(v0_bins, p=v0_marginal)
    phi_probs = policy_2d[v0_idx]
    phi_mix = getattr(config, 'phi_uniform_mix', 0.0)
    if phi_mix > 0:
        phi_probs = (1 - phi_mix) * phi_probs + phi_mix / phi_bins
    if phi_probs.sum() == 0:
        phi_probs = np.ones(phi_bins) / phi_bins
    else:
        phi_probs /= phi_probs.sum()
    phi_idx = np.random.choice(phi_bins, p=phi_probs)
    return v0_idx * phi_bins + phi_idx


def encode_state(balls, my_targets, table):
    """将游戏状态编码为神经网络输入
    
    返回: numpy array, shape (features,)
    """
    features = []
    
    # 白球位置和速度
    cue_ball = balls['cue']
    cue_pos = cue_ball.state.rvw[0][:2]  # [x, y]
    cue_vel = cue_ball.state.rvw[1][:2]  # [vx, vy]
    features.extend(cue_pos)
    features.extend(cue_vel)
    
    # 目标球位置和状态
    for i in range(1, 16):
        ball_id = str(i)
        if ball_id in balls:
            ball = balls[ball_id]
            pos = ball.state.rvw[0][:2]
            is_pocketed = 1.0 if ball.state.s == 4 else 0.0
            is_target = 1.0 if ball_id in my_targets else 0.0
            features.extend(pos)
            features.append(is_pocketed)
            features.append(is_target)
        else:
            features.extend([0.0, 0.0, 1.0, 0.0])  # 不存在的球
    
    # 球桌尺寸
    features.append(table.w)
    features.append(table.l)
    
    # 目标球数量
    features.append(len(my_targets))
    
    return np.array(features, dtype=np.float32)


def evaluate_action_with_physics(balls, my_targets, table, action_dict):
    """使用 pooltool 物理模拟评估单个动作的得分
    
    参数:
        balls: 球状态字典
        my_targets: 目标球ID列表
        table: 球桌对象
        action_dict: 动作字典 {'V0', 'phi', 'theta', 'a', 'b'}
    
    返回:
        float: 物理模拟得分（归一化到 [-1, 1]）
    """
    try:
        import pooltool as pt
        
        # 深拷贝当前状态用于模拟
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        shot.cue.set_state(
            V0=action_dict['V0'],
            phi=action_dict['phi'],
            theta=action_dict['theta'],
            a=action_dict['a'],
            b=action_dict['b']
        )
        
        # 执行物理模拟
        pt.simulate(shot, inplace=True)
        
        # 分析结果并计算奖励（复用 analyze_shot_for_reward）
        from agent import analyze_shot_for_reward
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        score = analyze_shot_for_reward(
            shot=shot,
            last_state=last_state_snapshot,
            player_targets=my_targets
        )
        
        # 归一化得分到 [-1, 1]（假设得分范围约 -150 到 +150）
        normalized_score = np.clip(score / 150.0, -1.0, 1.0)
        return normalized_score
        
    except Exception as e:
        # 模拟失败返回较低分
        return -0.5


def calculate_muzero_reward(result, env, player, my_targets=None):
    """根据图片评分标准计算MuZero奖励
    
    评分标准：
    1. 致命犯规（缺省-1分）：
       - 只要被扣中扣以下任何一种情况，分数会被扣减-4000分
       - evaluate_action_robustness函数中设置
    
    2. 胜利奖励：
       - 完胜对手：+100,000分
       - 在洗空自己球后，合法将黑8打进：完胜
    
    3. 进攻与推进得分：
       - 打进瞄准的目标球：+1500分（每颗）
       - 打进己方球但不是目标球：+200分（每颗）
       - 这是AI进攻的主要驱动力
       - 破坏（打进对方的球）：-500分（每颗）
       - 防止AI为了连续球权去破坏对手的球阵
    
    4. 击球犯规（扣分，但不一定会输）：
       - 这些扣分会被限制，降低其权重，但如果在存在弧性规则的情况下，可能会被忽略
       - 首次犯规（没打到自己球或对手球上）：-500分
       - 本届犯规：-200分
    
    5. 死底防守机制：
       - 安全球奖励：+50分
    """
    reward = 0.0
    
    # 检查游戏是否结束
    done, info = env.get_done()
    
    if done:
        # 2. 胜利奖励
        if info.get('winner') == player:
            reward += 15.0  # 完胜对手
            return reward
        elif info['winner'] == 'SAME':
            # 平局，根据剩余球数判断
            return 0.0
        else:
            # 失败
            reward -= 15.0  # 致命犯规导致失败
            return reward
    
    # 3. 进攻与推进得分（区分瞄准球和非瞄准球）
    if 'ME_INTO_POCKET' in result:
        me_pocketed = result['ME_INTO_POCKET']
        if my_targets is not None:
            # 区分瞄准球和非瞄准球
            for ball_id in me_pocketed:
                if ball_id in my_targets:
                    reward += 1.5  # 打进瞄准的目标球：+1.5/颗
                else:
                    reward += 0.2  # 打进己方球但不是目标球：+0.2/颗
        else:
            # 如果没有传入目标球信息，使用默认奖励
            reward += len(me_pocketed) * 1.5
    
    if 'ENEMY_INTO_POCKET' in result:
        enemy_pocketed = result['ENEMY_INTO_POCKET']
        reward -= len(enemy_pocketed) * 0.5  # 打进对方球：-0.5/颗
    
    # 4. 击球犯规
    if result.get('FOUL_FIRST_HIT'):
        reward -= 2.5  # 首次犯规（首球打错）
    
    if result.get('NO_POCKET_NO_RAIL'):
        reward -= 1.5  # 本届犯规（无进球无碰库）
    
    if result.get('WHITE_BALL_INTO_POCKET'):
        reward -= 12.0  # 白球进袋是致命犯规
    
    if result.get('BLACK_BALL_INTO_POCKET') and not done:
        # 非法打进黑8（不是完胜情况）
        reward -= 12.0

    # 5. 安全球奖励
    # 如果没有进球，也没有犯规，给予小奖励
    has_pocket = ('ME_INTO_POCKET' in result and len(result['ME_INTO_POCKET']) > 0)
    has_foul = (result.get('FOUL_FIRST_HIT') or 
                result.get('NO_POCKET_NO_RAIL') or 
                result.get('WHITE_BALL_INTO_POCKET'))
    
    if not has_pocket and not has_foul:
        reward += 0.3  # 安全球奖励
    
    return reward


def scalar_to_support(x, min_value, max_value, num_bins=601):
    """将标量转换为分类分布支持（用于价值和奖励）
    
    使用固定的bins数量，而不是根据范围计算
    """
    x = np.clip(x, min_value, max_value)
    # 归一化到[0, num_bins-1]
    x_normalized = (x - min_value) / (max_value - min_value) * (num_bins - 1)
    
    # 创建one-hot编码的概率分布
    support = np.zeros(num_bins, dtype=np.float32)
    
    # 线性插值
    lower = int(np.floor(x_normalized))
    upper = int(np.ceil(x_normalized))
    
    if lower == upper:
        support[lower] = 1.0
    else:
        support[lower] = upper - x_normalized
        support[upper] = x_normalized - lower
    
    return support


def support_to_scalar(support, min_value, max_value):
    """将分类分布支持转换回标量"""
    num_bins = len(support)
    bins = np.linspace(min_value, max_value, num_bins)
    return np.sum(support * bins)


# ========================= 神经网络组件 =========================

class RepresentationNetwork(nn.Module):
    """表示网络：将原始观测编码为隐藏状态"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 计算输入特征维度
        # 白球(4) + 15个球*(4) + 球桌(2) + 目标数(1) = 67
        input_size = 67
        
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, config.hidden_state_size)
        
    def forward(self, observation):
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        hidden_state = F.tanh(self.fc3(x))
        return hidden_state


class DynamicsNetwork(nn.Module):
    """动力学网络：预测下一个隐藏状态和即时奖励"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 隐藏状态 + 动作one-hot编码
        input_size = config.hidden_state_size + config.action_space_size
        
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_state = nn.Linear(256, config.hidden_state_size)
        
        # 奖励预测头（使用固定的bins数量）
        self.reward_bins = 601  # 固定bins数量
        self.fc_reward1 = nn.Linear(256, 64)
        self.fc_reward2 = nn.Linear(64, self.reward_bins)
        
    def forward(self, hidden_state, action):
        # action是one-hot编码
        x = torch.cat([hidden_state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # 下一个隐藏状态
        next_hidden_state = F.tanh(self.fc_state(x))
        
        # 奖励预测
        reward_x = F.relu(self.fc_reward1(x))
        reward_logits = self.fc_reward2(reward_x)
        
        return next_hidden_state, reward_logits


class PredictionNetwork(nn.Module):
    """预测网络：从隐藏状态预测策略和价值"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 策略头
        self.fc_policy1 = nn.Linear(config.hidden_state_size, 128)
        self.fc_policy2 = nn.Linear(128, config.action_space_size)
        
        # 价值头（使用固定的bins数量）
        self.value_bins = 601  # 固定bins数量
        self.fc_value1 = nn.Linear(config.hidden_state_size, 128)
        self.fc_value2 = nn.Linear(128, self.value_bins)
        
    def forward(self, hidden_state):
        # 策略
        policy_x = F.relu(self.fc_policy1(hidden_state))
        policy_logits = self.fc_policy2(policy_x)
        
        # 价值
        value_x = F.relu(self.fc_value1(hidden_state))
        value_logits = self.fc_value2(value_x)
        
        return policy_logits, value_logits


class MuZeroNetwork(nn.Module):
    """MuZero完整网络"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.representation = RepresentationNetwork(config)
        self.dynamics = DynamicsNetwork(config)
        self.prediction = PredictionNetwork(config)
        
    def initial_inference(self, observation):
        """初始推理：从观测得到初始隐藏状态、策略和价值"""
        hidden_state = self.representation(observation)
        policy_logits, value_logits = self.prediction(hidden_state)
        return hidden_state, policy_logits, value_logits
    
    def recurrent_inference(self, hidden_state, action):
        """循环推理：从隐藏状态和动作预测下一状态、奖励、策略和价值"""
        # action需要是one-hot编码
        next_hidden_state, reward_logits = self.dynamics(hidden_state, action)
        policy_logits, value_logits = self.prediction(next_hidden_state)
        return next_hidden_state, reward_logits, policy_logits, value_logits


# ========================= MCTS =========================

class Node:
    """MCTS节点"""
    
    def __init__(self, prior):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0
        
    def expanded(self):
        return len(self.children) > 0
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class MCTS:
    """蒙特卡洛树搜索"""
    
    def __init__(self, config):
        self.config = config
        
    class MinMaxStats:
        """用于将价值归一化到[0,1]，防止极端奖励压制探索"""
        def __init__(self):
            self.minimum = float('inf')
            self.maximum = -float('inf')

        def update(self, value):
            self.minimum = min(self.minimum, value)
            self.maximum = max(self.maximum, value)

        def normalize(self, value):
            if self.maximum > self.minimum:
                return (value - self.minimum) / (self.maximum - self.minimum)
            return 0.0
        
    def run(self, network, observation, add_exploration_noise=False, 
            physics_context=None):
        """运行MCTS搜索
        
        参数:
            network: MuZero网络
            observation: 状态观测
            add_exploration_noise: 是否添加探索噪声
            physics_context: 物理模拟上下文 (balls, my_targets, table)，用于辅助评估
        
        返回: 策略分布 (action_probs)
        """
        min_max_stats = self.MinMaxStats()
        root = Node(0)
        
        # 初始推理
        with torch.no_grad():
            observation_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.config.device)
            hidden_state, policy_logits, value_logits = network.initial_inference(observation_tensor)
            
            root.hidden_state = hidden_state
            
            # 扩展根节点
            policy = torch.softmax(policy_logits, dim=-1).squeeze().cpu().numpy()
            
            # 物理模拟辅助：对部分候选动作进行物理评估
            if self.config.use_physics_assist and physics_context is not None:
                balls, my_targets, table = physics_context
                
                # 选择概率最高的N个动作进行物理模拟
                n_samples = min(self.config.physics_assist_samples, self.config.action_space_size)
                top_actions = np.argsort(policy)[-n_samples:]
                
                physics_scores = np.zeros(self.config.action_space_size)
                for action_idx in top_actions:
                    action_dict = action_index_to_dict(action_idx, self.config)
                    score = evaluate_action_with_physics(balls, my_targets, table, action_dict)
                    physics_scores[action_idx] = score
                
                # 用物理得分修正先验概率
                # 将物理得分归一化并与原策略混合
                physics_scores_normalized = physics_scores - physics_scores.min()
                if physics_scores_normalized.max() > 0:
                    physics_scores_normalized /= physics_scores_normalized.max()
                
                weight = self.config.physics_assist_weight
                policy = (1 - weight) * policy + weight * physics_scores_normalized
                policy = policy / policy.sum()
            
            # 添加Dirichlet噪声用于探索
            if add_exploration_noise:
                noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_space_size)
                policy = policy * (1 - self.config.root_exploration_fraction) + \
                         noise * self.config.root_exploration_fraction
            
            for action_idx in range(self.config.action_space_size):
                root.children[action_idx] = Node(policy[action_idx])

        
        # 运行模拟
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            
            # 选择
            while node.expanded():
                action, node = self._select_child(node, min_max_stats)
                search_path.append(node)
            
            # 扩展和评估
            parent = search_path[-2]
            action = None
            for a, n in parent.children.items():
                if n == node:
                    action = a
                    break
            
            # 使用网络进行递归推理
            with torch.no_grad():
                action_tensor = F.one_hot(
                    torch.tensor([action]), 
                    num_classes=self.config.action_space_size
                ).float().to(self.config.device)
                
                next_hidden, reward_logits, policy_logits, value_logits = \
                    network.recurrent_inference(parent.hidden_state, action_tensor)
                
                node.hidden_state = next_hidden
                
                # 解码奖励和价值
                reward_probs = torch.softmax(reward_logits, dim=-1).squeeze().cpu().numpy()
                reward = support_to_scalar(reward_probs, 
                                          self.config.reward_support_min,
                                          self.config.reward_support_max)
                node.reward = reward
                
                value_probs = torch.softmax(value_logits, dim=-1).squeeze().cpu().numpy()
                value = support_to_scalar(value_probs,
                                         self.config.value_support_min,
                                         self.config.value_support_max)
                
                # 扩展节点
                policy = torch.softmax(policy_logits, dim=-1).squeeze().cpu().numpy()
                for action_idx in range(self.config.action_space_size):
                    node.children[action_idx] = Node(policy[action_idx])
            
            # 回溯更新
            self._backpropagate(search_path, value, min_max_stats)
        
        # 返回访问计数作为策略
        action_probs = np.zeros(self.config.action_space_size)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count
        
        total_visits = np.sum(action_probs)
        if total_visits > 0:
            action_probs = action_probs / total_visits
        else:
            action_probs = np.ones(self.config.action_space_size) / self.config.action_space_size

        if self.config.policy_smoothing > 0:
            action_probs = action_probs + self.config.policy_smoothing
            action_probs = action_probs / np.sum(action_probs)

        if self.config.policy_uniform_mix > 0:
            uniform = np.ones_like(action_probs) / self.config.action_space_size
            action_probs = (1 - self.config.policy_uniform_mix) * action_probs + \
                           self.config.policy_uniform_mix * uniform
            action_probs = action_probs / np.sum(action_probs)
            
        return action_probs
    
    def _select_child(self, node, min_max_stats):
        """使用PUCT公式选择子节点（支持随机打破并列）"""
        best_action = None
        best_child = None
        best_score = -float('inf')
        
        for action, child in node.children.items():
            ucb = self._ucb_score(node, child, min_max_stats)
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child
            elif math.isclose(ucb, best_score, rel_tol=1e-9, abs_tol=1e-12):
                # 当得分相等时随机选择，避免始终落在同一动作
                if random.random() < 0.5:
                    best_score = ucb
                    best_action = action
                    best_child = child
        
        return best_action, best_child
    
    def _ucb_score(self, parent, child, min_max_stats):
        """计算UCB分数"""
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) / 
                       self.config.pb_c_base) + self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        
        prior_score = pb_c * child.prior
        value_score = min_max_stats.normalize(child.value())
        
        return prior_score + value_score
    
    def _backpropagate(self, search_path, value, min_max_stats):
        """回溯更新节点值"""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.value())
            value = node.reward + 0.99 * value  # 使用折扣因子


# ========================= 经验回放 =========================

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, config):
        self.config = config
        self.buffer = deque(maxlen=config.replay_buffer_size)
        self.priorities = deque(maxlen=config.replay_buffer_size)
        
    def save_game(self, game_history):
        """保存一局游戏"""
        self.buffer.append(game_history)
        self.priorities.append(1.0)  # 初始优先级
        
    def sample_batch(self):
        """采样一个批次"""
        if len(self.buffer) < self.config.batch_size:
            return None
        
        # 使用优先级采样
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.config.priority_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), 
                                  size=self.config.batch_size, 
                                  p=probs, 
                                  replace=False)
        
        games = [self.buffer[idx] for idx in indices]
        
        # 为每个游戏采样一个位置
        batch = []
        for game in games:
            if len(game['observations']) == 0:
                continue
            pos = np.random.randint(0, len(game['observations']))
            batch.append((game, pos))
        
        return batch


class GameHistory:
    """游戏历史记录"""
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.policies = []
        self.values = []
        
    def store_transition(self, observation, action, reward, policy, value):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.values.append(value)


# ========================= 训练 =========================

def play_game(config, network, env, train_mode=True):
    """进行一局游戏（自我对弈）
    
    返回: GameHistory
    """
    game_history = GameHistory()
    
    # 重置环境
    env.reset(target_ball='solid')
    
    mcts = MCTS(config)
    done = False
    step = 0
    
    while not done and step < config.max_moves:
        player = env.get_curr_player()
        balls, my_targets, table = env.get_observation(player)
        
        # 编码状态
        observation = encode_state(balls, my_targets, table)
        
        # 运行MCTS（传递物理上下文用于辅助）
        physics_context = (balls, my_targets, table) if train_mode else None
        action_probs = mcts.run(network, observation, 
                               add_exploration_noise=train_mode,
                               physics_context=physics_context)
        
        # 选择动作
        if train_mode:
            # 训练时在V0/phi两个维度都保持探索
            action_idx = sample_action_index(action_probs, config)
        else:
            # 评估时选择最优
            action_idx = int(np.argmax(action_probs))
        
        # 转换为实际动作
        action = action_index_to_dict(action_idx, config)
        
        # 执行动作
        last_state = save_balls_state(balls)
        result = env.take_shot(action)
        
        # 计算奖励（根据图片评分标准，传入目标球信息）
        reward = calculate_muzero_reward(result, env, player, my_targets)
        
        # 存储转换
        game_history.store_transition(observation, action_idx, reward, action_probs, 0.0)
        
        done, info = env.get_done()
        step += 1
    
    # 计算目标价值（回溯计算）
    returns = []
    G = 0
    for reward in reversed(game_history.rewards):
        G = reward + 0.99 * G
        returns.insert(0, G)
    game_history.values = returns
    
    return game_history


def train_network(config, network, optimizer, batch, training_step):
    """训练网络"""
    network.train()
    
    total_loss = 0
    value_losses = []
    reward_losses = []
    policy_losses = []
    
    for game, pos in batch:
        # 获取初始观测
        observation = game['observations'][pos]
        observation_tensor = torch.FloatTensor(observation).unsqueeze(0).to(config.device)
        
        # 初始推理
        hidden_state, policy_logits, value_logits = network.initial_inference(observation_tensor)
        
        # 计算初始价值损失
        target_value = game['values'][pos]
        target_value_support = scalar_to_support(target_value, 
                                                 config.value_support_min,
                                                 config.value_support_max)
        target_value_tensor = torch.FloatTensor(target_value_support).unsqueeze(0).to(config.device)
        value_loss = F.cross_entropy(value_logits, target_value_tensor)
        value_losses.append(value_loss)
        
        # 计算初始策略损失
        target_policy = game['policies'][pos]
        target_policy_tensor = torch.FloatTensor(target_policy).unsqueeze(0).to(config.device)
        policy_loss = F.cross_entropy(policy_logits, target_policy_tensor)
        policy_losses.append(policy_loss)
        
        # 展开训练
        for k in range(config.num_unroll_steps):
            if pos + k + 1 >= len(game['actions']):
                break
            
            action = game['actions'][pos + k]
            action_tensor = F.one_hot(torch.tensor([action]), 
                                     num_classes=config.action_space_size).float().to(config.device)
            
            # 递归推理
            hidden_state, reward_logits, policy_logits, value_logits = \
                network.recurrent_inference(hidden_state, action_tensor)
            
            # 奖励损失
            target_reward = game['rewards'][pos + k + 1] if pos + k + 1 < len(game['rewards']) else 0
            target_reward_support = scalar_to_support(target_reward,
                                                     config.reward_support_min,
                                                     config.reward_support_max)
            target_reward_tensor = torch.FloatTensor(target_reward_support).unsqueeze(0).to(config.device)
            reward_loss = F.cross_entropy(reward_logits, target_reward_tensor)
            reward_losses.append(reward_loss)
            
            # 价值损失
            if pos + k + 1 < len(game['values']):
                target_value = game['values'][pos + k + 1]
                target_value_support = scalar_to_support(target_value,
                                                        config.value_support_min,
                                                        config.value_support_max)
                target_value_tensor = torch.FloatTensor(target_value_support).unsqueeze(0).to(config.device)
                value_loss = F.cross_entropy(value_logits, target_value_tensor)
                value_losses.append(value_loss)
            
            # 策略损失
            if pos + k + 1 < len(game['policies']):
                target_policy = game['policies'][pos + k + 1]
                target_policy_tensor = torch.FloatTensor(target_policy).unsqueeze(0).to(config.device)
                policy_loss = F.cross_entropy(policy_logits, target_policy_tensor)
                policy_losses.append(policy_loss)
    
    # 计算总损失
    loss = sum(value_losses) + sum(reward_losses) + sum(policy_losses)
    
    # 优化
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 10.0)
    optimizer.step()
    
    return loss.item()


def main():
    """主训练函数"""
    print("初始化MuZero训练...")
    
    # 配置
    config = MuZeroConfig()
    print(f"使用设备: {config.device}")
    
    # 创建网络
    network = MuZeroNetwork(config).to(config.device)
    optimizer = optim.Adam(network.parameters(), 
                          lr=config.lr_init, 
                          weight_decay=config.weight_decay)
    
    # 创建环境和缓冲区
    env = PoolEnv()
    replay_buffer = ReplayBuffer(config)
    
    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)
    
    print("开始训练...")
    training_step = 0
    
    for iteration in range(config.training_steps // config.self_play_games):
        print(f"\n=== 迭代 {iteration + 1} ===")
        
        # 自我对弈阶段
        print("自我对弈中，局数：", config.self_play_games)
        for game_idx in range(config.self_play_games):
            print(f"  开始第 {game_idx + 1} 局游戏")
            game_history = play_game(config, network, env, train_mode=True)
            
            # 转换为字典格式
            game_dict = {
                'observations': game_history.observations,
                'actions': game_history.actions,
                'rewards': game_history.rewards,
                'policies': game_history.policies,
                'values': game_history.values
            }
            replay_buffer.save_game(game_dict)
            
        
        # 训练阶段
        print("训练网络中...")
        for _ in range(config.batch_size):
            batch = replay_buffer.sample_batch()
            if batch is None:
                print("None")
                continue
            
            loss = train_network(config, network, optimizer, batch, training_step)
            training_step += 1
            
            if training_step % 10 == 0:
                print(f"  训练步数: {training_step}, 损失: {loss:.4f}")
                with open("train.txt", "a", encoding="utf-8") as f:
                    f.write(f"训练步数: {training_step}, 损失: {loss:.4f}\n")
        
        # 保存检查点
        if (iteration + 1) % config.checkpoint_interval == 0:
            checkpoint_path = os.path.join(config.save_dir, f"muzero_checkpoint_{training_step}.pth")
            torch.save({
                'network_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_step': training_step,
                'config': config
            }, checkpoint_path)
            print(f"保存检查点: {checkpoint_path}")
    
    # 保存最终模型
    final_path = os.path.join(config.save_dir, "muzero_final.pth")
    torch.save({
        'network_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_step': training_step,
        'config': config
    }, final_path)
    print(f"\n训练完成！最终模型保存至: {final_path}")


if __name__ == "__main__":
    main()
