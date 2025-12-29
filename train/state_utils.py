"""
state_utils.py
状态特征提取模块：将球局状态转换为固定长度的特征向量，用于 Value Network 输入。
"""
import numpy as np


# 球桌尺寸（标准 8 球桌：2.84m x 1.42m），用于归一化坐标
TABLE_LENGTH = 2.84
TABLE_WIDTH = 1.42

# 球的 ID 列表
ALL_BALL_IDS = ['cue'] + [str(i) for i in range(1, 16)]  # cue, 1-15
SOLIDS = [str(i) for i in range(1, 8)]      # 1-7
STRIPES = [str(i) for i in range(9, 16)]    # 9-15
BLACK8 = '8'


def extract_state_features(balls, my_targets, player_id='A'):
    """
    从球局状态中提取特征向量。
    
    参数:
        balls: dict[str, Ball] - 所有球的状态
        my_targets: list[str] - 当前击球方的目标球 ID 列表
        player_id: str - 当前击球方标识 ('A' 或 'B')
    
    返回:
        np.ndarray - 特征向量 (shape: (feature_dim,))
    
    特征设计 (共 ~70 维):
        - 母球位置 (2): x, y 归一化
        - 母球是否在袋 (1): 0/1
        - 各目标球位置 + 是否在袋 (最多 8 球 × 3 = 24)
        - 对手球位置 + 是否在袋 (最多 7 球 × 3 = 21)
        - 黑 8 位置 + 是否在袋 (3)
        - 我方剩余球数 (1)
        - 对手剩余球数 (1)
        - 当前是否可打黑 8 (1): 我方其他球全进则为 1
        - 母球到各袋口最小距离 (1)
        - 母球到最近目标球距离 (1)
        - 当前击球方 one-hot (2): [is_A, is_B]
    """
    features = []
    
    # ========== 母球特征 ==========
    cue_ball = balls.get('cue')
    if cue_ball is not None:
        cue_pos = cue_ball.state.rvw[0]
        cue_pocketed = 1.0 if cue_ball.state.s == 4 else 0.0
        features.extend([
            cue_pos[0] / TABLE_LENGTH,
            cue_pos[1] / TABLE_WIDTH,
            cue_pocketed
        ])
    else:
        features.extend([0.0, 0.0, 1.0])
    
    # ========== 确定我方和对方目标球 ==========
    if set(my_targets) & set(SOLIDS):
        own_balls = SOLIDS
        opp_balls = STRIPES
    else:
        own_balls = STRIPES
        opp_balls = SOLIDS
    
    # ========== 我方目标球特征 (最多 7 球) ==========
    own_features = []
    own_remaining = 0
    for bid in own_balls:
        ball = balls.get(bid)
        if ball is not None:
            pos = ball.state.rvw[0]
            pocketed = 1.0 if ball.state.s == 4 else 0.0
            own_features.extend([
                pos[0] / TABLE_LENGTH,
                pos[1] / TABLE_WIDTH,
                pocketed
            ])
            if pocketed == 0.0:
                own_remaining += 1
        else:
            own_features.extend([0.0, 0.0, 1.0])
    features.extend(own_features)
    
    # ========== 对方目标球特征 (最多 7 球) ==========
    opp_features = []
    opp_remaining = 0
    for bid in opp_balls:
        ball = balls.get(bid)
        if ball is not None:
            pos = ball.state.rvw[0]
            pocketed = 1.0 if ball.state.s == 4 else 0.0
            opp_features.extend([
                pos[0] / TABLE_LENGTH,
                pos[1] / TABLE_WIDTH,
                pocketed
            ])
            if pocketed == 0.0:
                opp_remaining += 1
        else:
            opp_features.extend([0.0, 0.0, 1.0])
    features.extend(opp_features)
    
    # ========== 黑 8 特征 ==========
    ball8 = balls.get(BLACK8)
    if ball8 is not None:
        pos8 = ball8.state.rvw[0]
        pocketed8 = 1.0 if ball8.state.s == 4 else 0.0
        features.extend([
            pos8[0] / TABLE_LENGTH,
            pos8[1] / TABLE_WIDTH,
            pocketed8
        ])
    else:
        features.extend([0.0, 0.0, 1.0])
    
    # ========== 统计特征 ==========
    features.append(own_remaining / 7.0)  # 我方剩余球数归一化
    features.append(opp_remaining / 7.0)  # 对方剩余球数归一化
    
    # 是否可打黑 8
    can_shoot_8 = 1.0 if own_remaining == 0 else 0.0
    features.append(can_shoot_8)
    
    # ========== 母球到袋口距离 (使用 6 个袋口的近似位置) ==========
    pocket_positions = [
        np.array([0.0, 0.0]),
        np.array([TABLE_LENGTH / 2, 0.0]),
        np.array([TABLE_LENGTH, 0.0]),
        np.array([0.0, TABLE_WIDTH]),
        np.array([TABLE_LENGTH / 2, TABLE_WIDTH]),
        np.array([TABLE_LENGTH, TABLE_WIDTH])
    ]
    if cue_ball is not None and cue_ball.state.s != 4:
        cue_pos_2d = cue_ball.state.rvw[0][:2]
        min_pocket_dist = min(np.linalg.norm(cue_pos_2d - p) for p in pocket_positions)
        features.append(min_pocket_dist / TABLE_LENGTH)  # 归一化
    else:
        features.append(0.0)
    
    # ========== 母球到最近目标球距离 ==========
    if cue_ball is not None and cue_ball.state.s != 4:
        cue_pos_2d = cue_ball.state.rvw[0][:2]
        target_dists = []
        for bid in my_targets:
            ball = balls.get(bid)
            if ball is not None and ball.state.s != 4:
                target_dists.append(np.linalg.norm(cue_pos_2d - ball.state.rvw[0][:2]))
        if target_dists:
            features.append(min(target_dists) / TABLE_LENGTH)
        else:
            features.append(1.0)  # 没有剩余目标球
    else:
        features.append(1.0)
    
    # ========== 当前击球方 one-hot ==========
    features.append(1.0 if player_id == 'A' else 0.0)
    features.append(1.0 if player_id == 'B' else 0.0)
    
    return np.array(features, dtype=np.float32)


def get_feature_dim():
    """返回特征向量的维度"""
    # 母球: 3
    # 我方球 (7): 21
    # 对方球 (7): 21
    # 黑8: 3
    # 统计: 5 (own_remaining, opp_remaining, can_shoot_8, min_pocket_dist, min_target_dist)
    # 击球方: 2
    return 3 + 21 + 21 + 3 + 5 + 2  # = 55


if __name__ == "__main__":
    # 简单测试
    print(f"Feature dimension: {get_feature_dim()}")
