import math
import numpy as np
import pooltool as pt
import copy
from .agent import Agent
from .basic_agent import analyze_shot_for_reward, simulate_with_timeout

class NewAgent(Agent):
    def __init__(self):
        super().__init__()
        self.SEARCH_SAMPLES = 45        # 减少基础采样，为鲁棒性测试留出时间
        self.ROBUST_SAMPLES = 3         # 鲁棒性测试次数
        self.POSITION_WEIGHT = 0.6
        self.SAFETY_THRESHOLD = 25.0
        self.BALL_RADIUS = 0.028575

    def is_path_blocked(self, start_pos, end_pos, all_balls, ignore_ids):
        """几何路径遮挡检测：判断白球路径是否被非法球阻挡"""
        vec = end_pos - start_pos
        dist = np.linalg.norm(vec)
        if dist < 1e-5: return False
        unit_vec = vec / dist
        
        for bid, ball in all_balls.items():
            if bid in ignore_ids or ball.state.s == 4: continue
            # 计算球心到路径的垂直距离
            ball_pos = ball.state.rvw[0]
            rel_pos = ball_pos - start_pos
            proj_dist = np.dot(rel_pos, unit_vec)
            if 0 < proj_dist < dist:
                perp_dist = np.linalg.norm(rel_pos - proj_dist * unit_vec)
                if perp_dist < self.BALL_RADIUS * 2.1: # 稍微放宽余量
                    return True
        return False

    def is_fatal_error(self, shot, my_targets, balls_before):
        """基于对局日志的即时判负严格过滤"""
        new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and balls_before[bid].state.s != 4]
        
        # 1. 致命：白球+黑8同时落袋 (Image 3.png)
        if "cue" in new_pocketed and "8" in new_pocketed:
            return True, -10000.0
        
        # 2. 致命：清台前误打黑8 (Image 1.png, 4.png)
        remaining_own = [bid for bid in my_targets if bid != '8' and balls_before[bid].state.s != 4]
        if "8" in new_pocketed and len(remaining_own) > 0:
            return True, -8000.0
            
        # 3. 严重：洗袋 (Image 2.jpg)
        if "cue" in new_pocketed:
            return True, -3000.0
            
        return False, 0

    def get_ghost_ball_candidates(self, balls, my_targets, table):
        """几何启发采样 + 路径预检"""
        candidates = []
        cue_pos = balls['cue'].state.rvw[0]
        
        for bid in my_targets:
            if balls[bid].state.s == 4: continue
            target_pos = balls[bid].state.rvw[0]
            
            for pocket in table.pockets.values():
                pocket_pos = pocket.center
                unit_tp = (pocket_pos - target_pos) / np.linalg.norm(pocket_pos - target_pos)
                ghost_pos = target_pos - unit_tp * (self.BALL_RADIUS * 2)
                
                # 预检：如果白球到虚球路径被挡（特别是黑8或对手球），直接跳过
                if self.is_path_blocked(cue_pos, ghost_pos, balls, ['cue', bid]): continue
                
                vec_cg = ghost_pos - cue_pos
                phi = math.degrees(math.atan2(vec_cg[1], vec_cg[0])) % 360
                # 动态力量计算
                v0 = np.clip(np.linalg.norm(vec_cg) * 1.8 + np.linalg.norm(pocket_pos - target_pos) * 1.2, 1.2, 5.5)
                
                candidates.append({'V0': float(v0), 'phi': float(phi), 'theta': 0.0, 'a': 0.0, 'b': 0.0})
        return candidates

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None: return self._random_action()
        last_state = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        current_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not current_targets: current_targets = ["8"]

        candidates = self.get_ghost_ball_candidates(balls, current_targets, table)
        while len(candidates) < self.SEARCH_SAMPLES:
            candidates.append(self._random_action())

        best_action = None
        max_robust_score = -float('inf')

        for action in candidates:
            # 第一轮：基础评分
            base_sim_score = self.evaluate_action(action, balls, table, current_targets, last_state)
            
            # 第二轮：高分动作进行鲁棒性测试 (抗噪声测试)
            if base_sim_score > self.SAFETY_THRESHOLD:
                robust_scores = []
                for _ in range(self.ROBUST_SAMPLES):
                    # 加入环境噪声偏差进行模拟
                    noisy_act = copy.deepcopy(action)
                    noisy_act['phi'] += np.random.normal(0, 0.08) # 模拟角度误差
                    noisy_act['V0'] += np.random.normal(0, 0.08)  # 模拟速度误差
                    robust_scores.append(self.evaluate_action(noisy_act, balls, table, current_targets, last_state))
                
                final_score = np.mean(robust_scores)
            else:
                final_score = base_sim_score

            if final_score > max_robust_score:
                max_robust_score = final_score
                best_action = action

        # 僵局处理
        if max_robust_score < 10.0 and best_action:
            best_action['V0'] = 2.5 # 适度加力破局
            
        return best_action if best_action else self._random_action()

    def evaluate_action(self, action, balls, table, targets, last_state):
        """执行单次模拟评分"""
        try:
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            shot = pt.System(table=copy.deepcopy(table), balls=sim_balls, cue=pt.Cue("cue"))
            shot.cue.set_state(**action)
            if not simulate_with_timeout(shot, timeout=1): return -500
            
            is_fatal, penalty = self.is_fatal_error(shot, targets, last_state)
            if is_fatal: return penalty
            
            base_reward = analyze_shot_for_reward(shot, last_state, targets)
            pos_reward = self.evaluate_positioning(shot, targets)
            
            return base_reward + (self.POSITION_WEIGHT * pos_reward)
        except:
            return -1000

    def evaluate_positioning(self, shot, my_targets):
        """走位评估：远离袋口，靠近剩余球心"""
        cue_pos = shot.balls['cue'].state.rvw[0]
        # 强制袋口避让 (Image 9.jpg)
        for pocket in shot.table.pockets.values():
            if np.linalg.norm(cue_pos - pocket.center) < 0.15:
                return -150.0
        
        remaining = [bid for bid in my_targets if shot.balls[bid].state.s != 4]
        if not remaining: return 50.0
        
        dists = [np.linalg.norm(cue_pos - shot.balls[bid].state.rvw[0]) for bid in remaining]
        return 40.0 * math.exp(-(min(dists) - 0.6)**2 / 0.1)