"""
value_agent.py
集成 Value Network 的 Agent：在评分中加入 γ·V(s') 项。

评分公式: Score(a) = r_me(a) - λ·r_opp(a) + γ·V̂(s')
其中:
    - s' 是模拟我方一杆后的新局面
    - V̂(s') = V(s') 如果仍轮到我; 否则 V̂(s') = 1 - V(s') (对手视角)
    - γ 一开始设小一点 (0.2~0.5)，避免 V 没训好就"喧宾夺主"
"""
import math
import os
import sys
import numpy as np
import pooltool as pt
import copy
import random
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent import Agent
from agents.basic_agent import analyze_shot_for_reward
from train.state_utils import extract_state_features, get_feature_dim
from train.networks import load_value_network


def safe_simulate(shot):
    """安全执行模拟，捕获异常"""
    try:
        pt.simulate(shot, inplace=True)
        return True
    except Exception:
        return False


class ValueAgent(Agent):
    """
    集成 Value Network 的 Agent。
    基于 NewAgent 的搜索框架，在评分中加入 V(s') 项。
    """
    
    def __init__(self, value_net_path=None, device='cpu'):
        """
        Args:
            value_net_path: Value Network 检查点路径。若为 None，则不使用 V(s)。
            device: 推理设备 ('cpu' 或 'cuda')
        """
        super().__init__()
        
        # 搜索参数
        self.SEARCH_SAMPLES = 48
        self.TOP_N_2PLY = 6
        self.ROBUST_SAMPLES = 3
        self.LAMBDA_OPP = 0.7
        self.POSITION_WEIGHT = 0.5
        self.SAFETY_THRESHOLD = 25.0
        self.BALL_RADIUS = 0.028575
        
        # Value Network 参数
        self.GAMMA_VALUE = 0.3  # V(s') 的权重，一开始设小一点
        self.device = device
        self.value_net = None
        
        # 加载 Value Network
        if value_net_path and os.path.exists(value_net_path):
            try:
                self.value_net = load_value_network(
                    value_net_path,
                    feature_dim=get_feature_dim(),
                    device=device
                )
                print(f"[ValueAgent] 已加载 Value Network: {value_net_path}")
            except Exception as e:
                print(f"[ValueAgent] 加载 Value Network 失败: {e}")
                self.value_net = None
        else:
            print(f"[ValueAgent] 未指定或找不到 Value Network，将不使用 V(s)")
    
    def is_path_blocked(self, start_pos, end_pos, all_balls, ignore_ids):
        """检查从 start_pos 到 end_pos 的路径是否被阻挡"""
        vec = end_pos - start_pos
        dist = np.linalg.norm(vec)
        if dist < 1e-5:
            return False
        unit_vec = vec / dist
        for bid, ball in all_balls.items():
            if bid in ignore_ids or ball.state.s == 4:
                continue
            ball_pos = ball.state.rvw[0]
            rel_pos = ball_pos - start_pos
            proj_dist = np.dot(rel_pos, unit_vec)
            if 0 < proj_dist < dist:
                perp_dist = np.linalg.norm(rel_pos - proj_dist * unit_vec)
                if perp_dist < self.BALL_RADIUS * 2.1:
                    return True
        return False

    def is_fatal_error(self, shot, my_targets, balls_before):
        """检查是否发生致命错误（白球入袋、提前打黑8等）"""
        new_pocketed = [bid for bid, b in shot.balls.items()
                       if b.state.s == 4 and balls_before[bid].state.s != 4]
        remaining_own = [bid for bid in my_targets
                        if bid != '8' and balls_before[bid].state.s != 4]
        if "cue" in new_pocketed and "8" in new_pocketed:
            return True, -10000.0
        if "8" in new_pocketed and len(remaining_own) > 0:
            return True, -8000.0
        if "cue" in new_pocketed:
            return True, -3000.0
        return False, 0.0

    def get_ghost_ball_candidates(self, balls, my_targets, table):
        """生成 ghost-ball 候选动作"""
        candidates = []
        cue_pos = balls['cue'].state.rvw[0]
        for bid in my_targets:
            if balls[bid].state.s == 4:
                continue
            target_pos = balls[bid].state.rvw[0]
            for pocket in table.pockets.values():
                pocket_pos = pocket.center
                tp_vec = pocket_pos - target_pos
                tp_dist = np.linalg.norm(tp_vec)
                if tp_dist < 1e-5:
                    continue
                unit_tp = tp_vec / tp_dist
                ghost_pos = target_pos - unit_tp * (self.BALL_RADIUS * 2)
                if self.is_path_blocked(cue_pos, ghost_pos, balls, ['cue', bid]):
                    continue
                vec_cg = ghost_pos - cue_pos
                cg_dist = np.linalg.norm(vec_cg)
                if cg_dist < 1e-5:
                    continue
                phi = math.degrees(math.atan2(vec_cg[1], vec_cg[0])) % 360
                v0 = np.clip(cg_dist * 1.8 + tp_dist * 1.2, 1.2, 5.5)
                candidates.append({'V0': float(v0), 'phi': float(phi), 'theta': 0.0, 'a': 0.0, 'b': 0.0})
        return candidates

    def generate_perturbation_candidates(self, base_action, count=8):
        """生成扰动候选动作"""
        candidates = []
        for _ in range(count):
            action = {
                'V0': float(np.clip(base_action['V0'] + np.random.uniform(-0.3, 0.3), 0.8, 6.0)),
                'phi': float((base_action['phi'] + np.random.uniform(-2.0, 2.0)) % 360),
                'theta': 0.0,
                'a': float(np.clip(np.random.uniform(-0.2, 0.2), -0.4, 0.4)),
                'b': float(np.clip(np.random.uniform(-0.2, 0.2), -0.4, 0.4))
            }
            candidates.append(action)
        return candidates

    def predict_value(self, balls, my_targets, player_id='A'):
        """
        使用 Value Network 预测当前局面的胜率。
        
        Args:
            balls: 球局状态
            my_targets: 当前击球方的目标球
            player_id: 当前击球方标识
        
        Returns:
            float: V(s) ∈ (0, 1)，如果没有 Value Network 则返回 0.5
        """
        if self.value_net is None:
            return 0.5
        
        try:
            features = extract_state_features(balls, my_targets, player_id)
            features_tensor = torch.FloatTensor(features).to(self.device)
            value = self.value_net.predict(features_tensor).item()
            return value
        except Exception as e:
            return 0.5

    def decision(self, balls=None, my_targets=None, table=None):
        """做出击球决策"""
        if balls is None:
            return self._random_action()
        
        last_state = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        current_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not current_targets:
            current_targets = ["8"]
        
        # 生成候选动作
        candidates = self.get_ghost_ball_candidates(balls, current_targets, table)
        if candidates:
            base_candidates = candidates[:min(4, len(candidates))]
            for base in base_candidates:
                candidates.extend(self.generate_perturbation_candidates(base, count=4))
        while len(candidates) < self.SEARCH_SAMPLES:
            candidates.append(self._random_action())
        
        # Phase 1: 快速筛选
        phase1_results = []
        for action in candidates:
            score = self.evaluate_action_phase1(action, balls, table, current_targets, last_state)
            if score > -5000:
                phase1_results.append((action, score))
        
        if not phase1_results:
            return self._random_action()
        
        phase1_results.sort(key=lambda x: x[1], reverse=True)
        top_candidates = phase1_results[:self.TOP_N_2PLY]
        
        # Phase 2: 2-ply 评估（加入 V(s')）
        best_action = None
        best_score = -float('inf')
        
        for action, phase1_score in top_candidates:
            if phase1_score > self.SAFETY_THRESHOLD:
                # 鲁棒性测试
                robust_scores = []
                for _ in range(self.ROBUST_SAMPLES):
                    noisy_act = {
                        'V0': action['V0'] + np.random.normal(0, 0.08),
                        'phi': action['phi'] + np.random.normal(0, 0.08),
                        'theta': action.get('theta', 0.0),
                        'a': action.get('a', 0.0),
                        'b': action.get('b', 0.0)
                    }
                    robust_scores.append(self.evaluate_action_2ply(
                        noisy_act, balls, table, current_targets, last_state, my_targets
                    ))
                final_score = np.mean(robust_scores)
            else:
                final_score = self.evaluate_action_2ply(
                    action, balls, table, current_targets, last_state, my_targets
                )
            
            if final_score > best_score:
                best_score = final_score
                best_action = action
        
        # 低分时提高力度
        if best_score < 10.0 and best_action:
            best_action['V0'] = float(np.clip(best_action.get('V0', 2.0) + 0.5, 2.0, 4.0))
        
        print(f"[ValueAgent] Score={best_score:.2f}")
        return best_action if best_action else self._random_action()

    def evaluate_action_phase1(self, action, balls, table, targets, last_state):
        """Phase 1 快速评估（不含 V(s')）"""
        try:
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            shot = pt.System(table=copy.deepcopy(table), balls=sim_balls, cue=pt.Cue("cue"))
            action_clean = {k: v for k, v in action.items() if k in ['V0', 'phi', 'theta', 'a', 'b']}
            shot.cue.set_state(**action_clean)
            
            if not safe_simulate(shot):
                return -500.0
            
            is_fatal, penalty = self.is_fatal_error(shot, targets, last_state)
            if is_fatal:
                return penalty
            
            base_reward = analyze_shot_for_reward(shot, last_state, targets)
            pos_reward = self.evaluate_positioning(shot, targets)
            return base_reward + self.POSITION_WEIGHT * pos_reward
        except Exception:
            return -1000.0

    def evaluate_action_2ply(self, action, balls, table, targets, last_state, original_my_targets):
        """
        Phase 2 完整评估，包含 V(s') 项。
        
        Score(a) = r_me(a) - λ·r_opp(a) + γ·V̂(s')
        """
        try:
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            shot = pt.System(table=copy.deepcopy(table), balls=sim_balls, cue=pt.Cue("cue"))
            action_clean = {k: v for k, v in action.items() if k in ['V0', 'phi', 'theta', 'a', 'b']}
            shot.cue.set_state(**action_clean)
            
            if not safe_simulate(shot):
                return -500.0
            
            is_fatal, penalty = self.is_fatal_error(shot, targets, last_state)
            if is_fatal:
                return penalty
            
            # r_me: 我方得分
            r_me = analyze_shot_for_reward(shot, last_state, targets)
            pos_reward = self.evaluate_positioning(shot, targets)
            r_me += self.POSITION_WEIGHT * pos_reward
            
            # 检查是否进了自己的球
            new_pocketed = [bid for bid, b in shot.balls.items()
                          if b.state.s == 4 and last_state[bid].state.s != 4]
            own_pocketed = [bid for bid in new_pocketed if bid in targets]
            cue_pocketed = "cue" in new_pocketed
            
            # 如果进了自己的球且没犯规，继续击球
            if own_pocketed and not cue_pocketed:
                # 仍轮到我，V̂(s') = V(s')
                v_value = self.predict_value(shot.balls, original_my_targets, 'A')
                return r_me + self.GAMMA_VALUE * v_value * 100  # 缩放 V 值
            
            # 否则对手击球
            r_opp = self.simulate_opponent_response(shot, original_my_targets)
            
            # 轮到对手，V̂(s') = 1 - V(s') (对手视角)
            v_value = self.predict_value(shot.balls, original_my_targets, 'A')
            v_hat = 1.0 - v_value  # 对手的胜率是我的败率
            
            return r_me - self.LAMBDA_OPP * r_opp + self.GAMMA_VALUE * v_hat * 100
        except Exception:
            return -1000.0

    def simulate_opponent_response(self, shot_after_my_turn, my_targets):
        """轻量级对手模拟 - 使用 ghost-ball 直接采样"""
        try:
            all_solids = ['1', '2', '3', '4', '5', '6', '7']
            all_stripes = ['9', '10', '11', '12', '13', '14', '15']
            
            if set(my_targets) & set(all_solids):
                opp_targets = [bid for bid in all_stripes if shot_after_my_turn.balls[bid].state.s != 4]
            else:
                opp_targets = [bid for bid in all_solids if shot_after_my_turn.balls[bid].state.s != 4]
            
            if not opp_targets:
                opp_targets = ['8']
            
            opp_candidates = self.get_ghost_ball_candidates(
                shot_after_my_turn.balls, opp_targets, shot_after_my_turn.table
            )
            if not opp_candidates:
                return 0.0
            
            best_opp_reward = 0.0
            opp_last_state = {bid: copy.deepcopy(ball) for bid, ball in shot_after_my_turn.balls.items()}
            
            for opp_action in opp_candidates[:3]:
                try:
                    opp_balls = {bid: copy.deepcopy(ball) for bid, ball in shot_after_my_turn.balls.items()}
                    opp_shot = pt.System(
                        table=copy.deepcopy(shot_after_my_turn.table),
                        balls=opp_balls,
                        cue=pt.Cue("cue")
                    )
                    action_clean = {k: v for k, v in opp_action.items() if k in ['V0', 'phi', 'theta', 'a', 'b']}
                    opp_shot.cue.set_state(**action_clean)
                    
                    if not safe_simulate(opp_shot):
                        continue
                    
                    r_opp = analyze_shot_for_reward(opp_shot, opp_last_state, opp_targets)
                    if r_opp > best_opp_reward:
                        best_opp_reward = r_opp
                except Exception:
                    continue
            
            return max(best_opp_reward, 0.0)
        except Exception:
            return 0.0

    def evaluate_positioning(self, shot, my_targets):
        """评估击球后的站位质量"""
        cue_pos = shot.balls['cue'].state.rvw[0]
        
        if shot.balls['cue'].state.s == 4:
            return -200.0
        
        # 检查母球是否离袋口太近
        for pocket in shot.table.pockets.values():
            dist_to_pocket = np.linalg.norm(cue_pos - pocket.center)
            if dist_to_pocket < 0.10:
                return -150.0
            elif dist_to_pocket < 0.18:
                return -50.0
        
        # 评估与目标球的距离
        remaining = [bid for bid in my_targets if shot.balls[bid].state.s != 4]
        if not remaining:
            return 50.0
        
        dists = [np.linalg.norm(cue_pos - shot.balls[bid].state.rvw[0]) for bid in remaining]
        min_dist = min(dists)
        
        if min_dist < 0.15:
            return 20.0
        elif min_dist < 0.6:
            return 40.0 * math.exp(-((min_dist - 0.4) ** 2) / 0.05)
        else:
            return 10.0 * math.exp(-((min_dist - 0.6) ** 2) / 0.2)

    def _random_action(self):
        """生成随机动作"""
        return {
            'V0': round(random.uniform(1.0, 5.0), 2),
            'phi': round(random.uniform(0, 360), 2),
            'theta': 0.0,
            'a': round(random.uniform(-0.3, 0.3), 3),
            'b': round(random.uniform(-0.3, 0.3), 3)
        }


if __name__ == "__main__":
    # 简单测试
    agent = ValueAgent(value_net_path=None)
    print("ValueAgent 初始化成功（无 Value Network）")
    
    # 测试随机动作
    action = agent._random_action()
    print(f"随机动作: {action}")
