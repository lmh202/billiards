"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板
- analyze_shot_for_reward: 击球结果评分函数
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...]
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8）, -30（首球/碰库犯规）
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞
    first_contact_ball_id = None
    foul_first_hit = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue']
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    if first_contact_ball_id is None:
        if len(last_state) > 2:  # 只有白球和8号球时不算犯规
             foul_first_hit = True
    else:
        remaining_own_before = [bid for bid in player_targets if last_state[bid].state.s != 4]
        opponent_plus_eight = [bid for bid in last_state.keys() if bid not in player_targets and bid not in ['cue']]
        if ('8' not in opponent_plus_eight):
            opponent_plus_eight.append('8')
            
        if len(remaining_own_before) > 0 and first_contact_ball_id in opponent_plus_eight:
            foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # 计算奖励分数
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 150
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 100 if is_targeting_eight_ball_legally else -150
            
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score

class Agent():
    """Agent 基类"""
    def __init__(self):
        pass
    
    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）
        
        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass
    
    def _random_action(self,):
        """生成随机击球动作
        
        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # 初速度 0.5~8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # 水平角度 (0°~360°)
            'theta': round(random.uniform(0, 90), 2),   # 垂直角度
            'a': round(random.uniform(-0.5, 0.5), 3),   # 杆头横向偏移（单位：球半径比例）
            'b': round(random.uniform(-0.5, 0.5), 3)    # 杆头纵向偏移
        }
        return action



class BasicAgent(Agent):
    """基于贝叶斯优化的智能 Agent"""
    
    def __init__(self, target_balls=None):
        """初始化 Agent
        
        参数：
            target_balls: 保留参数，暂未使用
        """
        super().__init__()
        
        # 搜索空间
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90), 
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        
        # 优化参数
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.ALPHA = 1e-2
        
        # 模拟噪声（可调整以改变训练难度）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.enable_noise = False
        
        print("BasicAgent (Smart, pooltool-native) 已初始化。")

    
    def _create_optimizer(self, reward_function, seed):
        """创建贝叶斯优化器
        
        参数：
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子
        
        返回：
            BayesianOptimization对象
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )
        
        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        
        return optimizer


    def decision(self, balls=None, my_targets=None, table=None):
        """使用贝叶斯优化搜索最佳击球参数
        
        参数：
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象
        
        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
                失败时返回随机动作
        """
        if balls is None:
            print(f"[BasicAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:
            
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[BasicAgent] 我的目标球已全部清空，自动切换目标为：8号球")

            # 1.动态创建“奖励函数” (Wrapper)
            # 贝叶斯优化器会调用此函数，并传入参数
            def reward_fn_wrapper(V0, phi, theta, a, b):
                # 创建一个用于模拟的沙盒系统
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")

                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                try:
                    if self.enable_noise:
                        V0_noisy = V0 + np.random.normal(0, self.noise_std['V0'])
                        phi_noisy = phi + np.random.normal(0, self.noise_std['phi'])
                        theta_noisy = theta + np.random.normal(0, self.noise_std['theta'])
                        a_noisy = a + np.random.normal(0, self.noise_std['a'])
                        b_noisy = b + np.random.normal(0, self.noise_std['b'])
                        
                        V0_noisy = np.clip(V0_noisy, 0.5, 8.0)
                        phi_noisy = phi_noisy % 360
                        theta_noisy = np.clip(theta_noisy, 0, 90)
                        a_noisy = np.clip(a_noisy, -0.5, 0.5)
                        b_noisy = np.clip(b_noisy, -0.5, 0.5)
                        
                        shot.cue.set_state(V0=V0_noisy, phi=phi_noisy, theta=theta_noisy, a=a_noisy, b=b_noisy)
                    else:
                        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    
                    # 关键：使用 pooltool 物理引擎 (世界A)
                    pt.simulate(shot, inplace=True)
                except Exception as e:
                    # 模拟失败，给予极大惩罚
                    return -500
                
                # 使用我们的“裁判”来打分
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )


                return score

            print(f"[BasicAgent] 正在为 Player (targets: {my_targets}) 搜索最佳击球...")
            
            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            optimizer.maximize(
                init_points=self.INITIAL_SEARCH,
                n_iter=self.OPT_SEARCH
            )
            
            best_result = optimizer.max
            best_params = best_result['params']
            best_score = best_result['target']

            if best_score < 10:
                print(f"[BasicAgent] 未找到好的方案 (最高分: {best_score:.2f})。使用随机动作。")
                return self._random_action()
            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b']),
            }

            print(f"[BasicAgent] 决策 (得分: {best_score:.2f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action

        except Exception as e:
            print(f"[BasicAgent] 决策时发生严重错误，使用随机动作。原因: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()

class NewAgent(Agent):
    """基于PPO强化学习的智能 Agent"""
    
    def __init__(self, checkpoint_path: str = None):
        """
        初始化Agent
        
        参数:
            checkpoint_path: 训练好的模型检查点路径
                            默认为 './train/checkpoints/final_model.pt'
        """
        super().__init__()
        
        import torch
        import numpy as np
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 默认检查点路径
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                os.path.dirname(__file__), 
                'train', 'checkpoints', 'final_model.pt'
            )
        
        # 球ID映射
        self.ball_ids = ['cue'] + [str(i) for i in range(1, 16)]
        self.ball_id_to_idx = {bid: i for i, bid in enumerate(self.ball_ids)}
        self.pocket_ids = ['lb', 'lc', 'lt', 'rb', 'rc', 'rt']
        
        # 加载模型
        self.policy = None
        if os.path.exists(checkpoint_path):
            try:
                from train.networks import PPOActorCritic
                from train.config import NETWORK_CONFIG
                
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                network_config = checkpoint.get('network_config', NETWORK_CONFIG)
                
                self.policy = PPOActorCritic(network_config).to(self.device)
                self.policy.load_state_dict(checkpoint['policy_state_dict'])
                self.policy.eval()
                
                print(f"[NewAgent] PPO模型已加载: {checkpoint_path}")
            except Exception as e:
                print(f"[NewAgent] 加载模型失败: {e}, 将使用随机策略")
                self.policy = None
        else:
            print(f"[NewAgent] 未找到模型文件: {checkpoint_path}, 将使用随机策略")
            
    def _get_observation(self, balls, my_targets, table):
        """
        将环境观测转换为网络输入格式
        """
        import torch
        import numpy as np
        
        max_balls = 16
        
        # 1. 球特征 [max_balls, 7]
        ball_features = np.zeros((max_balls, 7), dtype=np.float32)
        ball_mask = np.zeros(max_balls, dtype=np.bool_)
        
        for bid, ball in balls.items():
            if bid not in self.ball_id_to_idx:
                continue
            idx = self.ball_id_to_idx[bid]
            
            # 位置 (归一化)
            pos = ball.state.rvw[0]
            ball_features[idx, 0] = pos[0] / table.l
            ball_features[idx, 1] = pos[1] / table.w
            ball_features[idx, 2] = pos[2]
            
            # 速度 (归一化)
            vel = ball.state.rvw[1]
            ball_features[idx, 3] = vel[0] / 10.0
            ball_features[idx, 4] = vel[1] / 10.0
            ball_features[idx, 5] = vel[2] / 10.0
            
            # 是否进袋
            ball_features[idx, 6] = 1.0 if ball.state.s == 4 else 0.0
            ball_mask[idx] = (ball.state.s != 4)
            
        ball_mask[0] = True  # 白球始终有效
        
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
        own_remaining = len(remaining_targets)
        is_targeting_8 = 1.0 if own_remaining == 0 else 0.0
        
        game_state = np.array([
            own_remaining / 7.0,
            0.5,  # 对手信息在评估时未知，使用默认值
            is_targeting_8,
            0.5   # hit_count未知
        ], dtype=np.float32)
        
        return {
            'ball_features': torch.from_numpy(ball_features).unsqueeze(0).to(self.device),
            'ball_mask': torch.from_numpy(ball_mask).unsqueeze(0).to(self.device),
            'pocket_features': torch.from_numpy(pocket_features).unsqueeze(0).to(self.device),
            'target_mask': torch.from_numpy(target_mask).unsqueeze(0).to(self.device),
            'game_state': torch.from_numpy(game_state).unsqueeze(0).to(self.device)
        }
    
    def decision(self, balls=None, my_targets=None, table=None):
        """
        决策方法 - 使用PPO策略网络进行决策
        
        参数：
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象
        
        返回：
            dict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        import torch
        
        if balls is None:
            print("[NewAgent] 未收到balls信息，使用随机动作")
            return self._random_action()
            
        # 检查目标球是否清空
        remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining_targets) == 0:
            my_targets = ['8']
            print("[NewAgent] 目标球已清空，切换到8号球")
            
        # 如果模型未加载，使用随机策略
        if self.policy is None:
            print("[NewAgent] 模型未加载，使用随机动作")
            return self._random_action()
            
        try:
            # 获取观测
            obs = self._get_observation(balls, my_targets, table)
            
            # 使用策略网络决策
            with torch.no_grad():
                action, _, _, _ = self.policy.get_action(obs, deterministic=True)
                action = action.cpu().numpy().squeeze()
                
            action_dict = {
                'V0': float(action[0]),
                'phi': float(action[1]),
                'theta': float(action[2]),
                'a': float(action[3]),
                'b': float(action[4])
            }
            
            print(f"[NewAgent] 决策: V0={action_dict['V0']:.2f}, phi={action_dict['phi']:.2f}, "
                  f"theta={action_dict['theta']:.2f}, a={action_dict['a']:.3f}, b={action_dict['b']:.3f}")
            
            return action_dict
            
        except Exception as e:
            print(f"[NewAgent] 决策失败: {e}, 使用随机动作")
            import traceback
            traceback.print_exc()
            return self._random_action()