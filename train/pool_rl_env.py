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
import concurrent.futures
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
                 device: str = 'cpu',
                 two_ball_mode: bool = False):
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
        self.two_ball_mode = two_ball_mode
        if not self.two_ball_mode:
            self.env = PoolEnv()
            self.env.enable_noise = enable_noise
        else:
            # two-ball simplified environment (used for phase-1 pretraining)
            self.env = None
            self.enable_noise = enable_noise
        
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
        if not self.two_ball_mode:
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

        # === two-ball mode ===
        # Create a fresh simple table and two balls (cue + target)
        table = pt.Table.default()
        R = pt.objects.BallParams().R

        # helper to sample a valid position (away from pockets and edges)
        def sample_pos():
            margin_x = 0.2 + R
            margin_y = 0.2 + R
            for _ in range(100):
                x = np.random.uniform(-table.l/2 + margin_x, table.l/2 - margin_x)
                y = np.random.uniform(-table.w/2 + margin_y, table.w/2 - margin_y)
                pos = np.array([x, y])
                # ensure not too close to any pocket
                dists = [np.linalg.norm(pos - np.array(table.pockets[pid].center[:2])) for pid in table.pockets]
                if min(dists) > 0.2:
                    return [x, y]
            return [0.0, 0.0]

        cue_xy = sample_pos()
        target_xy = sample_pos()
        # ensure separation
        if np.linalg.norm(np.array(cue_xy) - np.array(target_xy)) < 2 * R + 0.05:
            target_xy[0] += (2 * R + 0.1)

        cue_ball = pt.objects.Ball.create('cue', xy=cue_xy)
        # 随机选择一个非黑8的普通球 (1-7 或 9-15)
        valid_target_ids = [str(i) for i in range(1, 8)] + [str(i) for i in range(9, 16)]
        target_ball_id = np.random.choice(valid_target_ids)
        target_ball = pt.objects.Ball.create(target_ball_id, xy=target_xy)

        # build balls dict and system
        balls = {'cue': cue_ball, target_ball_id: target_ball}

        self.simple_table = table
        self.simple_balls = balls
        self.simple_cue = pt.Cue(cue_ball_id='cue')
        self.agent_player = 'A'
        self.hit_count = 0
        self.MAX_HIT_COUNT = 50
        self.current_target_ball = target_ball_id
        self.shot_record = pt.MultiSystem()

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
        if not self.two_ball_mode:
            balls_before = copy.deepcopy(self.env.balls)
            my_targets_before = self.env.player_targets[self.agent_player].copy()
        else:
            balls_before = {k: v.copy() for k, v in self.simple_balls.items()}
            my_targets_before = [self.current_target_ball]
        
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
        if not self.two_ball_mode:
            step_info = self.env.take_shot(action_dict)
        else:
            # simulate simple two-ball shot
            shot = pt.System(table=self.simple_table, balls=self.simple_balls, cue=self.simple_cue)
            self.simple_cue.set_state(V0=action_dict['V0'], phi=action_dict['phi'], theta=action_dict['theta'], a=action_dict['a'], b=action_dict['b'])
            sim_result = self._simulate_two_ball_with_timeout(shot, timeout_s=3.0)
            if isinstance(sim_result, Exception):
                # 仿真异常时直接结束本回合，给予惩罚，避免卡死
                step_info = {
                    'ME_INTO_POCKET': [],
                    'ENEMY_INTO_POCKET': [],
                    'WHITE_BALL_INTO_POCKET': False,
                    'BLACK_BALL_INTO_POCKET': False,
                    'FOUL_FIRST_HIT': False,
                    'NO_POCKET_NO_RAIL': False,
                    'NO_HIT': True,
                    'BALLS': {k: v.copy() for k, v in self.simple_balls.items()},
                    'error': str(sim_result)
                }
                return self._get_observation(), -25.0, True, {
                    'step_info': step_info,
                    'game_info': {'winner': 'B'},
                    'agent_player': self.agent_player,
                    'hit_count': self.hit_count
                }
            if sim_result is None:
                # 超时，直接结束本回合并惩罚
                step_info = {
                    'ME_INTO_POCKET': [],
                    'ENEMY_INTO_POCKET': [],
                    'WHITE_BALL_INTO_POCKET': False,
                    'BLACK_BALL_INTO_POCKET': False,
                    'FOUL_FIRST_HIT': False,
                    'NO_POCKET_NO_RAIL': False,
                    'NO_HIT': True,
                    'BALLS': {k: v.copy() for k, v in self.simple_balls.items()},
                    'timeout': True
                }
                return self._get_observation(), -30.0, True, {
                    'step_info': step_info,
                    'game_info': {'winner': 'B'},
                    'agent_player': self.agent_player,
                    'hit_count': self.hit_count
                }
            shot = sim_result

            self.shot_record.append(copy.deepcopy(shot))
            self.simple_balls = shot.balls

            new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and balls_before[bid].state.s != 4]
            own_pocketed = [bid for bid in new_pocketed if bid == self.current_target_ball]
            cue_pocketed = 'cue' in new_pocketed

            # 检测是否命中目标球：检查事件历史中是否有白球与目标球的碰撞
            target_hit = False
            if hasattr(shot, 'events') and shot.events is not None:
                for event in shot.events:
                    # 检查球球碰撞事件 (event_type='ball_ball')
                    if hasattr(event, 'event_type') and event.event_type == 'ball_ball':
                        if hasattr(event, 'agents') and len(event.agents) == 2:
                            # agents 是 Agent 对象的元组，需要提取 id
                            ids = [agent.id for agent in event.agents if hasattr(agent, 'id')]
                            if 'cue' in ids and self.current_target_ball in ids:
                                target_hit = True
                                break

            step_info = {
                'ME_INTO_POCKET': own_pocketed,
                'ENEMY_INTO_POCKET': [],
                'WHITE_BALL_INTO_POCKET': cue_pocketed,
                'BLACK_BALL_INTO_POCKET': False,
                'FOUL_FIRST_HIT': False,
                'NO_POCKET_NO_RAIL': False,
                'NO_HIT': False,
                'TARGET_HIT': target_hit,
                'BALLS': {k: v.copy() for k, v in self.simple_balls.items()}
            }
        
        # 计算奖励
        reward = self._compute_reward(balls_before, my_targets_before, step_info)
        
        # 检查游戏是否结束
        if not self.two_ball_mode:
            done, game_info = self.env.get_done()
        else:
            done = False
            game_info = {}
            # end when target pocketed or white pocketed or max hits
            if len(step_info.get('ME_INTO_POCKET', [])) > 0:
                done = True
                game_info['winner'] = self.agent_player
            elif step_info.get('WHITE_BALL_INTO_POCKET', False):
                done = True
                game_info['winner'] = 'B'  # treat as loss
            else:
                self.hit_count += 1
                if self.hit_count >= self.MAX_HIT_COUNT:
                    done = True
                    game_info['winner'] = 'SAME'
        
        if not done and not self.two_ball_mode:
            # 如果轮到对手,让对手行动
            while self.env.get_curr_player() != self.agent_player:
                if self._check_done():
                    done = True
                    break
                self._opponent_step()

            # 再次检查
            done, game_info = self.env.get_done()
        
        # 游戏结束的额外奖励
        if done and not self.two_ball_mode:
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
            'hit_count': self.hit_count if self.two_ball_mode else self.env.hit_count
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
        if not self.two_ball_mode:
            balls, my_targets, table = self.env.get_observation(self.agent_player)
        else:
            balls = self.simple_balls
            my_targets = [self.current_target_ball]
            table = self.simple_table
        
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
        if not self.two_ball_mode:
            enemy_targets = self.env.player_targets[opponent_player]
            enemy_remaining = len([bid for bid in enemy_targets if balls[bid].state.s != 4])
            hit_count_normalized = self.env.hit_count / self.env.MAX_HIT_COUNT
        else:
            enemy_remaining = 0
            is_targeting_8 = 1.0 if own_remaining == 0 else 0.0
            hit_count_normalized = float(self.hit_count) / float(self.MAX_HIT_COUNT)
        is_targeting_8 = 1.0 if own_remaining == 0 else 0.0
        
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

    def _simulate_two_ball_with_timeout(self, shot, timeout_s: float = 3.0):
        """
        在单杆两球模式下，给物理仿真加超时保护。
        
        关键修复: 使用 max_events 限制物理引擎的最大迭代次数,防止无限循环
        - 两球模式下,正常情况只需要几十个事件
        - 限制为1000个事件足够处理所有正常情况
        - 防止极端情况下物理引擎永远不返回
        
        返回值:
            - 成功: 返回 shot 对象
            - 超时: 返回 None
            - 异常: 返回 Exception
        """
        def run_sim():
            # 关键: 添加 max_events 限制,防止物理引擎无限循环
            pt.simulate(shot, inplace=True, max_events=1000)
            return shot

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(run_sim)
            try:
                return future.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError:
                print(f"[Simulate][TIMEOUT] 物理仿真超过{timeout_s}秒")
                future.cancel()
                return None
            except Exception as e:
                print(f"[Simulate][ERROR] 物理仿真异常: {e}")
                future.cancel()
                return e
    
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
    
    def _compute_reward_two_ball_mode(self, balls_before: dict, step_info: dict) -> float:
        """
        Phase 1 专用奖励函数 (two_ball_mode) - 优化版
        
        设计原则:
        1. 稀疏奖励为主: 进洞+20, 白球进洞-10, 命中+5
        2. 稠密引导为辅: 距离改善给予适度奖励,避免过度惩罚
        3. 鼓励探索: 不要对角度偏差过度惩罚,只奖励好的角度
        """
        reward = -0.01  # 小的时间惩罚,鼓励快速完成
        
        cue_pocketed = step_info.get('WHITE_BALL_INTO_POCKET', False)
        own_pocketed = step_info.get('ME_INTO_POCKET', [])
        target_hit = step_info.get('TARGET_HIT', False)
        balls_after = step_info.get('BALLS', {})
        
        # === 白球进洞惩罚 ===
        if cue_pocketed:
            reward += -10.0
            return reward
        
        # === 目标球进洞奖励 (最重要!) ===
        if len(own_pocketed) > 0:
            reward += 50.0  # 增加到50,让进球成为主要目标
            return reward
        
        # === 命中目标球奖励 ===
        if target_hit:
            reward += 5.0
        
        # === 1. 距离改善奖励 (稠密引导) ===
        target_ball_id = self.current_target_ball
        if target_ball_id and balls_after and target_ball_id in balls_before and target_ball_id in balls_after:
            target_before = balls_before[target_ball_id]
            target_after = balls_after[target_ball_id]
            
            # 只有目标球未进袋时才计算
            if target_before.state.s != 4 and target_after.state.s != 4:
                table = self.simple_table
                
                # 白球位置
                cue_before_pos = balls_before['cue'].state.rvw[0][:2]
                cue_after_pos = balls_after['cue'].state.rvw[0][:2]
                
                # 目标球位置
                target_before_pos = target_before.state.rvw[0][:2]
                target_after_pos = target_after.state.rvw[0][:2]
                
                # 1.1 白球→目标球距离改善 (只有未命中时才鼓励靠近)
                if not target_hit:
                    cue_target_dist_before = np.linalg.norm(cue_before_pos - target_before_pos)
                    cue_target_dist_after = np.linalg.norm(cue_after_pos - target_after_pos)
                    delta_cue_target = cue_target_dist_before - cue_target_dist_after
                    # 放宽clip限制,让距离改善更明显
                    delta_cue_target = np.clip(delta_cue_target, -0.5, 0.5)
                    reward += delta_cue_target * 5.0  # 降低权重
                
                # 1.2 目标球→最近袋口距离改善
                pocket_dist_before = self._nearest_pocket_dist(target_before_pos, table)
                pocket_dist_after = self._nearest_pocket_dist(target_after_pos, table)
                delta_pocket = pocket_dist_before - pocket_dist_after
                delta_pocket = np.clip(delta_pocket, -0.5, 0.5)
                reward += delta_pocket * 8.0  # 适度降低权重
                
                # === 2. 视线/角度奖励 (只奖励不惩罚!) ===
                # 找到最近的袋口
                nearest_pocket_pos = None
                min_pocket_dist = float('inf')
                for pid in self.pocket_ids:
                    if pid in table.pockets:
                        pocket_pos = np.array(table.pockets[pid].center[:2])
                        dist = np.linalg.norm(target_after_pos - pocket_pos)
                        if dist < min_pocket_dist:
                            min_pocket_dist = dist
                            nearest_pocket_pos = pocket_pos
                
                if nearest_pocket_pos is not None:
                    # 计算 白球→目标球 和 目标球→袋口 的夹角
                    vec_cue_to_target = target_after_pos - cue_after_pos
                    vec_target_to_pocket = nearest_pocket_pos - target_after_pos
                    
                    # 归一化
                    norm_ct = np.linalg.norm(vec_cue_to_target)
                    norm_tp = np.linalg.norm(vec_target_to_pocket)
                    
                    if norm_ct > 1e-6 and norm_tp > 1e-6:
                        vec_cue_to_target /= norm_ct
                        vec_target_to_pocket /= norm_tp
                        
                        # 计算夹角的余弦值
                        cos_theta = np.dot(vec_cue_to_target, vec_target_to_pocket)
                        
                        # 关键修改: 只在角度很好时给奖励,不好也不惩罚!
                        if cos_theta > 0.95:  # 18度以内
                            angle_reward = 1.0 + (cos_theta - 0.95) * 20.0  # +1 到 +2
                            reward += angle_reward
                        elif cos_theta > 0.85:  # 18-31度, 给一点奖励
                            reward += 0.5
                        # 角度不好就不给奖励,但也不惩罚!
        
        # === 3. 动作幅度/速度约束 ===
        if 'cue' in balls_after:
            cue_vel = balls_after['cue'].state.rvw[1][:2]
            cue_speed = np.linalg.norm(cue_vel)
            
            # 极小力度惩罚 (几乎不动)
            if cue_speed < 0.1:
                reward += -1.0  # 增加惩罚,防止"站桩"
            
            # 过大力度轻微惩罚
            elif cue_speed > 6.0:  # 提高阈值到6
                reward += -0.5
        
        return reward
    
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
        # Phase 1 两球模式使用专用奖励函数
        if self.two_ball_mode:
            return self._compute_reward_two_ball_mode(balls_before, step_info)
        
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
                table = self.simple_table if self.two_ball_mode else self.env.table
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
        
        remaining = [bid for bid in my_targets if bid in balls_after and balls_after[bid].state.s != 4]
        if len(remaining) == 0:
            # 检查8号球位置 (only if exists)
            if '8' in balls_after and balls_after['8'].state.s != 4:
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
