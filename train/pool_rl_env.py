"""
pool_rl_env.py - 台球强化学习环境封装

将PoolEnv封装为标准的单智能体RL环境，包含:
- 状态空间处理
- 动作空间处理
- 精细的奖励函数设计
"""

import numpy as np
import torch
import copy
import pooltool as pt
from typing import Dict, Tuple, List, Optional, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poolenv import PoolEnv, save_balls_state
from train.config import REWARD_CONFIG, NETWORK_CONFIG, ACTION_BOUNDS, DEVICE


class PoolRLEnv:
    """
    台球强化学习环境
    
    单智能体环境，智能体扮演一方玩家，对手使用简单策略或自我对弈
    """
    
    def __init__(self, 
                 opponent_type: str = 'self',  # 'self', 'random', 'basic'
                 enable_noise: bool = True,
                 reward_config: dict = REWARD_CONFIG,
                 device: str = 'cpu'):
        """
        Args:
            opponent_type: 对手类型
                - 'self': 自我对弈(使用相同策略)
                - 'random': 随机对手
                - 'basic': 使用BasicAgent作为对手
            enable_noise: 是否启用物理噪声
            reward_config: 奖励配置
            device: 设备
        """
        self.env = PoolEnv()
        self.env.enable_noise = enable_noise
        
        self.opponent_type = opponent_type
        self.reward_config = reward_config
        self.device = device
        self.basic_agent = None
        
        # 球ID映射
        self.ball_ids = ['cue'] + [str(i) for i in range(1, 16)]  # 16个球
        self.ball_id_to_idx = {bid: i for i, bid in enumerate(self.ball_ids)}
        
        # 球袋ID
        self.pocket_ids = ['lb', 'lc', 'lt', 'rb', 'rc', 'rt']
        
        # 当前玩家身份 ('A' 或 'B')
        self.agent_player = None
        
        # 用于追踪目标球
        self.current_target_ball = None
        
        # 对手策略 (用于自我对弈)
        self.opponent_policy = None
        
    def set_opponent_policy(self, policy):
        """设置对手策略 (用于自我对弈)"""
        self.opponent_policy = policy
        
    def reset(self, target_ball: str = None) -> Dict[str, torch.Tensor]:
        """
        重置环境
        
        Args:
            target_ball: 'solid' 或 'stripe', None则随机选择
            
        Returns:
            observation: 观测字典
        """
        if target_ball is None:
            target_ball = np.random.choice(['solid', 'stripe'])
            
        self.env.reset(target_ball=target_ball)
        
        # 随机选择智能体扮演哪一方
        self.agent_player = np.random.choice(['A', 'B'])
        
        # 如果智能体是后手，让对手先行动直到轮到智能体
        while self.env.get_curr_player() != self.agent_player:
            if self._check_done():
                # 对手已经结束游戏
                break
            self._opponent_step()
            
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict]:
        """
        执行一步
        
        Args:
            action: [V0, phi, theta, a, b] 原始动作空间
            
        Returns:
            observation: 下一状态观测
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 保存击球前状态用于奖励计算
        balls_before = copy.deepcopy(self.env.balls)
        my_targets_before = self.env.player_targets[self.agent_player].copy()
        
        # 计算当前目标球 (用于额外奖励)
        self.current_target_ball = self._get_best_target_ball(balls_before, my_targets_before)
        
        # 转换动作格式
        action_dict = {
            'V0': float(action[0]),
            'phi': float(action[1]),
            'theta': float(action[2]),
            'a': float(action[3]),
            'b': float(action[4])
        }
        
        # 执行击球
        step_info = self.env.take_shot(action_dict)
        
        # 计算奖励
        reward = self._compute_reward(balls_before, my_targets_before, step_info)
        
        # 检查游戏是否结束
        done, game_info = self.env.get_done()
        
        if not done:
            # 如果轮到对手,让对手行动
            while self.env.get_curr_player() != self.agent_player:
                if self._check_done():
                    done = True
                    break
                self._opponent_step()
                
        # 再次检查
        done, game_info = self.env.get_done()
        
        # 游戏结束的额外奖励
        if done:
            if game_info.get('winner') == self.agent_player:
                reward += self.reward_config['win_game']
            elif game_info.get('winner') == 'SAME':
                pass  # 平局无额外奖励
            else:
                reward += self.reward_config['lose_game']
                
        obs = self._get_observation()
        
        info = {
            'step_info': step_info,
            'game_info': game_info,
            'agent_player': self.agent_player,
            'hit_count': self.env.hit_count
        }
        
        return obs, reward, done, info
    
    def _get_observation(self) -> Dict[str, torch.Tensor]:
        """
        获取观测
        
        Returns:
            obs: 包含以下key的字典
                - ball_features: [max_balls, 7] 球特征
                - ball_mask: [max_balls] 有效球mask
                - pocket_features: [6*3] 球袋特征
                - target_mask: [max_balls] 目标球mask
                - game_state: [4] 游戏状态
        """
        balls, my_targets, table = self.env.get_observation(self.agent_player)
        
        # 1. 球特征 [max_balls, 7]
        max_balls = NETWORK_CONFIG['max_balls']
        ball_features = np.zeros((max_balls, 7), dtype=np.float32)
        ball_mask = np.zeros(max_balls, dtype=np.bool_)
        
        for bid, ball in balls.items():
            if bid not in self.ball_id_to_idx:
                continue
            idx = self.ball_id_to_idx[bid]
            
            # 位置 (归一化到 [0, 1])
            pos = ball.state.rvw[0]
            ball_features[idx, 0] = pos[0] / table.l  # x
            ball_features[idx, 1] = pos[1] / table.w  # y
            ball_features[idx, 2] = pos[2]  # z (通常为0)
            
            # 速度 (归一化)
            vel = ball.state.rvw[1]
            ball_features[idx, 3] = vel[0] / 10.0
            ball_features[idx, 4] = vel[1] / 10.0
            ball_features[idx, 5] = vel[2] / 10.0
            
            # 是否进袋
            ball_features[idx, 6] = 1.0 if ball.state.s == 4 else 0.0
            
            # 只有未进袋的球才有效
            ball_mask[idx] = (ball.state.s != 4)
            
        # 白球始终有效
        ball_mask[0] = True
        
        # 2. 球袋特征 [6*3]
        pocket_features = np.zeros(6 * 3, dtype=np.float32)
        for i, pid in enumerate(self.pocket_ids):
            if pid in table.pockets:
                center = table.pockets[pid].center
                pocket_features[i*3] = center[0] / table.l
                pocket_features[i*3 + 1] = center[1] / table.w
                pocket_features[i*3 + 2] = center[2]
                
        # 3. 目标球mask [max_balls]
        target_mask = np.zeros(max_balls, dtype=np.float32)
        remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        
        if len(remaining_targets) == 0:
            # 目标球清空，瞄准8号球
            target_mask[self.ball_id_to_idx['8']] = 1.0
        else:
            for bid in remaining_targets:
                target_mask[self.ball_id_to_idx[bid]] = 1.0
                
        # 4. 游戏状态 [4]
        own_remaining = len([bid for bid in my_targets if balls[bid].state.s != 4])
        opponent_player = 'B' if self.agent_player == 'A' else 'A'
        enemy_targets = self.env.player_targets[opponent_player]
        enemy_remaining = len([bid for bid in enemy_targets if balls[bid].state.s != 4])
        is_targeting_8 = 1.0 if own_remaining == 0 else 0.0
        hit_count_normalized = self.env.hit_count / self.env.MAX_HIT_COUNT
        
        game_state = np.array([
            own_remaining / 7.0,
            enemy_remaining / 7.0,
            is_targeting_8,
            hit_count_normalized
        ], dtype=np.float32)
        
        device = torch.device(self.device)
        return {
            'ball_features': torch.from_numpy(ball_features).unsqueeze(0).to(device),
            'ball_mask': torch.from_numpy(ball_mask).unsqueeze(0).to(device),
            'pocket_features': torch.from_numpy(pocket_features).unsqueeze(0).to(device),
            'target_mask': torch.from_numpy(target_mask).unsqueeze(0).to(device),
            'game_state': torch.from_numpy(game_state).unsqueeze(0).to(device)
        }
    
    def _get_best_target_ball(self, balls: dict, my_targets: list) -> Optional[str]:
        """
        获取最佳目标球 (用于奖励塑形)
        简单策略: 选择与白球最近的目标球
        """
        cue_pos = balls['cue'].state.rvw[0][:2]
        
        remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining) == 0:
            return '8' if balls['8'].state.s != 4 else None
            
        best_ball = None
        min_dist = float('inf')
        
        for bid in remaining:
            ball_pos = balls[bid].state.rvw[0][:2]
            dist = np.linalg.norm(cue_pos - ball_pos)
            if dist < min_dist:
                min_dist = dist
                best_ball = bid
                
        return best_ball
    
    def _compute_reward(self, balls_before: dict, my_targets_before: list, 
                        step_info: dict) -> float:
        """
        计算精细的奖励
        
        奖励设计原则:
        1. 鼓励打进当前瞄准的目标球
        2. 不过分鼓励打进非瞄准的目标球
        3. 严厉惩罚提前打进黑球
        4. 惩罚各种犯规行为
        """
        reward = self.reward_config['step_penalty']  # 基础时间惩罚
        
        # 获取进袋信息
        own_pocketed = step_info.get('ME_INTO_POCKET', [])
        enemy_pocketed = step_info.get('ENEMY_INTO_POCKET', [])
        cue_pocketed = step_info.get('WHITE_BALL_INTO_POCKET', False)
        eight_pocketed = step_info.get('BLACK_BALL_INTO_POCKET', False)
        balls_after = step_info.get('BALLS', {})
        
        # 检查是否应该瞄准8号球
        remaining_before = [bid for bid in my_targets_before if balls_before[bid].state.s != 4]
        is_targeting_eight = len(remaining_before) == 0
        
        # === 严厉惩罚: 白球+黑8同时进袋 ===
        if cue_pocketed and eight_pocketed:
            reward += self.reward_config['cue_and_eight_pocketed']
            return reward  # 直接返回,这是最严重的犯规
        
        # === 白球进袋 ===
        if cue_pocketed:
            reward += self.reward_config['cue_pocketed']
            return reward
        
        # === 黑8进袋 ===
        if eight_pocketed:
            if is_targeting_eight:
                # 合法打进8号球 - 巨大奖励
                reward += self.reward_config['legal_eight_pocketed']
            else:
                # 非法打进8号球 - 严厉惩罚!!!
                reward += self.reward_config['illegal_eight_pocketed']
            return reward
        
        # === 犯规检查 ===
        if step_info.get('FOUL_FIRST_HIT', False):
            reward += self.reward_config['foul_first_hit']
            
        if step_info.get('NO_POCKET_NO_RAIL', False):
            reward += self.reward_config['no_rail_foul']
            
        if step_info.get('NO_HIT', False):
            reward += self.reward_config['no_hit_foul']
            
        # === 打进己方目标球 ===
        for bid in own_pocketed:
            reward += self.reward_config['own_ball_pocketed']
            
            # 如果打进的是当前瞄准的球,额外奖励
            if bid == self.current_target_ball:
                reward += self.reward_config['own_ball_pocketed_bonus_targeted']
                
        # === 打进对方球 (轻微惩罚) ===
        for bid in enemy_pocketed:
            reward += self.reward_config['enemy_ball_pocketed']
            
        # === 连续击球权 ===
        if len(own_pocketed) > 0 and not cue_pocketed:
            reward += self.reward_config['continue_shot']
        elif len(own_pocketed) == 0:
            reward += self.reward_config['lose_turn']
            
        # === 走位奖励 (如果有连续击球权) ===
        if len(own_pocketed) > 0 and not cue_pocketed and balls_after:
            position_reward = self._compute_position_reward(balls_after, my_targets_before)
            reward += position_reward

        # === 稠密引导: 目标球靠近袋口 + 白球靠近目标球 ===
        target_ball = self.current_target_ball or self._get_best_target_ball(balls_before, my_targets_before)
        if target_ball and balls_after:
            if (target_ball in balls_before and target_ball in balls_after and
                    balls_before[target_ball].state.s != 4 and balls_after[target_ball].state.s != 4):
                table = self.env.table
                before_pos = balls_before[target_ball].state.rvw[0][:2]
                after_pos = balls_after[target_ball].state.rvw[0][:2]

                before_pocket = self._nearest_pocket_dist(before_pos, table)
                after_pocket = self._nearest_pocket_dist(after_pos, table)
                progress = np.clip(before_pocket - after_pocket, -1.0, 1.0)
                reward += progress * self.reward_config.get('pocket_progress_weight', 0.0)

                cue_before = balls_before['cue'].state.rvw[0][:2]
                cue_after = balls_after['cue'].state.rvw[0][:2]
                cue_target_before = np.linalg.norm(cue_before - before_pos)
                cue_target_after = np.linalg.norm(cue_after - after_pos)
                cue_progress = np.clip(cue_target_before - cue_target_after, -1.0, 1.0)
                reward += cue_progress * self.reward_config.get('cue_target_align_weight', 0.0)
                
        return reward
    
    def _compute_position_reward(self, balls_after: dict, my_targets: list) -> float:
        """
        计算走位奖励
        基于白球与下一个目标球的距离和角度
        """
        cue_pos = balls_after['cue'].state.rvw[0][:2]
        
        remaining = [bid for bid in my_targets if balls_after[bid].state.s != 4]
        if len(remaining) == 0:
            # 检查8号球位置
            if balls_after['8'].state.s != 4:
                target_pos = balls_after['8'].state.rvw[0][:2]
                dist = np.linalg.norm(cue_pos - target_pos)
                # 距离越近越好
                if dist < 0.5:
                    return self.reward_config['good_position']
            return 0.0
            
        # 找最近的目标球
        min_dist = float('inf')
        for bid in remaining:
            ball_pos = balls_after[bid].state.rvw[0][:2]
            dist = np.linalg.norm(cue_pos - ball_pos)
            min_dist = min(min_dist, dist)
            
        # 距离越近奖励越高
        if min_dist < 0.3:
            return self.reward_config['good_position']
        elif min_dist < 0.5:
            return self.reward_config['good_position'] * 0.5
            
        return 0.0

    def _nearest_pocket_dist(self, pos: np.ndarray, table) -> float:
        """计算某球到最近袋口的距离"""
        dists = []
        for pid in self.pocket_ids:
            if pid in table.pockets:
                pocket_pos = np.array(table.pockets[pid].center[:2])
                dists.append(np.linalg.norm(pos - pocket_pos))
        return min(dists) if dists else 0.0
    
    def _opponent_step(self):
        """对手执行一步"""
        if self.opponent_type == 'random':
            action = self._random_action()
        elif self.opponent_type == 'self' and self.opponent_policy is not None:
            # 自我对弈
            obs = self._get_opponent_observation()
            with torch.no_grad():
                action, _, _, _ = self.opponent_policy.get_action(obs, deterministic=False)
                action = action.cpu().numpy().squeeze()
        elif self.opponent_type == 'basic':
            if self.basic_agent is None:
                from agent import BasicAgent
                self.basic_agent = BasicAgent()
            opponent_player = 'B' if self.agent_player == 'A' else 'A'
            action = self.basic_agent.decision(
                balls=self.env.balls,
                my_targets=self.env.player_targets[opponent_player],
                table=self.env.table
            )
        else:
            # 默认随机
            action = self._random_action()
        
        if action is None:
            action = self._random_action()
            
        if isinstance(action, dict):
            action_dict = {
                'V0': float(action['V0']),
                'phi': float(action['phi']),
                'theta': float(action['theta']),
                'a': float(action['a']),
                'b': float(action['b'])
            }
        else:
            action_dict = {
                'V0': float(action[0]),
                'phi': float(action[1]),
                'theta': float(action[2]),
                'a': float(action[3]),
                'b': float(action[4])
            }
        self.env.take_shot(action_dict)
        
    def _get_opponent_observation(self) -> Dict[str, torch.Tensor]:
        """获取对手视角的观测"""
        opponent_player = 'B' if self.agent_player == 'A' else 'A'
        
        balls, my_targets, table = self.env.get_observation(opponent_player)
        
        # 与_get_observation相同的处理，但使用对手的目标
        max_balls = NETWORK_CONFIG['max_balls']
        ball_features = np.zeros((max_balls, 7), dtype=np.float32)
        ball_mask = np.zeros(max_balls, dtype=np.bool_)
        
        for bid, ball in balls.items():
            if bid not in self.ball_id_to_idx:
                continue
            idx = self.ball_id_to_idx[bid]
            
            pos = ball.state.rvw[0]
            ball_features[idx, 0] = pos[0] / table.l
            ball_features[idx, 1] = pos[1] / table.w
            ball_features[idx, 2] = pos[2]
            
            vel = ball.state.rvw[1]
            ball_features[idx, 3] = vel[0] / 10.0
            ball_features[idx, 4] = vel[1] / 10.0
            ball_features[idx, 5] = vel[2] / 10.0
            
            ball_features[idx, 6] = 1.0 if ball.state.s == 4 else 0.0
            ball_mask[idx] = (ball.state.s != 4)
            
        ball_mask[0] = True
        
        pocket_features = np.zeros(6 * 3, dtype=np.float32)
        for i, pid in enumerate(self.pocket_ids):
            if pid in table.pockets:
                center = table.pockets[pid].center
                pocket_features[i*3] = center[0] / table.l
                pocket_features[i*3 + 1] = center[1] / table.w
                pocket_features[i*3 + 2] = center[2]
                
        target_mask = np.zeros(max_balls, dtype=np.float32)
        remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        
        if len(remaining_targets) == 0:
            target_mask[self.ball_id_to_idx['8']] = 1.0
        else:
            for bid in remaining_targets:
                target_mask[self.ball_id_to_idx[bid]] = 1.0
                
        own_remaining = len(remaining_targets)
        agent_targets = self.env.player_targets[self.agent_player]
        enemy_remaining = len([bid for bid in agent_targets if balls[bid].state.s != 4])
        is_targeting_8 = 1.0 if own_remaining == 0 else 0.0
        hit_count_normalized = self.env.hit_count / self.env.MAX_HIT_COUNT
        
        game_state = np.array([
            own_remaining / 7.0,
            enemy_remaining / 7.0,
            is_targeting_8,
            hit_count_normalized
        ], dtype=np.float32)
        
        device = torch.device(self.device)
        return {
            'ball_features': torch.from_numpy(ball_features).unsqueeze(0).to(device),
            'ball_mask': torch.from_numpy(ball_mask).unsqueeze(0).to(device),
            'pocket_features': torch.from_numpy(pocket_features).unsqueeze(0).to(device),
            'target_mask': torch.from_numpy(target_mask).unsqueeze(0).to(device),
            'game_state': torch.from_numpy(game_state).unsqueeze(0).to(device)
        }
    
    def _random_action(self) -> np.ndarray:
        """生成随机动作"""
        return np.array([
            np.random.uniform(ACTION_BOUNDS['V0'][0], ACTION_BOUNDS['V0'][1]),
            np.random.uniform(ACTION_BOUNDS['phi'][0], ACTION_BOUNDS['phi'][1]),
            np.random.uniform(ACTION_BOUNDS['theta'][0], ACTION_BOUNDS['theta'][1]),
            np.random.uniform(ACTION_BOUNDS['a'][0], ACTION_BOUNDS['a'][1]),
            np.random.uniform(ACTION_BOUNDS['b'][0], ACTION_BOUNDS['b'][1])
        ], dtype=np.float32)
    
    def _check_done(self) -> bool:
        """检查游戏是否结束"""
        done, _ = self.env.get_done()
        return done


class VectorPoolEnv:
    """向量化环境 - 支持多个并行环境"""
    
    def __init__(self, num_envs: int, device: str = 'cpu', **kwargs):
        self.num_envs = num_envs
        self.device = device
        self.envs = [PoolRLEnv(device=device, **kwargs) for _ in range(num_envs)]
        
    def set_opponent_policy(self, policy):
        """设置所有环境的对手策略"""
        for env in self.envs:
            env.set_opponent_policy(policy)
            
    def reset(self) -> Dict[str, torch.Tensor]:
        """重置所有环境"""
        obs_list = [env.reset() for env in self.envs]
        return self._stack_obs(obs_list)
    
    def step(self, actions: np.ndarray) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray, List[Dict]]:
        """
        执行一步
        
        Args:
            actions: [num_envs, action_dim]
            
        Returns:
            observations, rewards, dones, infos
        """
        obs_list = []
        rewards = []
        dones = []
        infos = []
        
        for i, env in enumerate(self.envs):
            obs, reward, done, info = env.step(actions[i])
            
            if done:
                # 自动重置
                obs = env.reset()
                
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            
        return (
            self._stack_obs(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            infos
        )
    
    def _stack_obs(self, obs_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """堆叠观测"""
        return {
            key: torch.cat([obs[key] for obs in obs_list], dim=0)
            for key in obs_list[0].keys()
        }
