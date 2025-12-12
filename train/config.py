"""
config.py - PPO训练配置

包含所有训练超参数和环境配置
"""

import torch

# ============ 设备配置 ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ 环境配置 ============
ENV_CONFIG = {
    'enable_noise': True,           # 是否启用物理噪声
    'max_steps_per_episode': 60,    # 每局最大击球数
}

# ============ 网络架构配置 ============
NETWORK_CONFIG = {
    # 状态编码器
    'ball_feature_dim': 7,          # 每个球的特征维度: [x, y, z, vx, vy, vz, pocketed]
    'max_balls': 16,                # 最大球数量
    'pocket_feature_dim': 3,        # 每个球袋特征维度: [x, y, z]
    'num_pockets': 6,               # 球袋数量
    
    # 编码器维度
    'ball_embed_dim': 64,           # 球特征嵌入维度
    'pocket_embed_dim': 32,         # 球袋特征嵌入维度
    'state_embed_dim': 256,         # 状态嵌入维度
    
    # Actor-Critic 网络
    'hidden_dim': 512,              # 隐藏层维度
    'num_hidden_layers': 3,         # 隐藏层数量
    
    # 动作空间
    'action_dim': 5,                # 动作维度: V0, phi, theta, a, b
}

# ============ PPO超参数 ============
PPO_CONFIG = {
    'lr_actor': 3e-4,               # Actor学习率
    'lr_critic': 1e-3,              # Critic学习率
    'gamma': 0.99,                  # 折扣因子
    'gae_lambda': 0.95,             # GAE参数
    'clip_epsilon': 0.2,            # PPO裁剪系数
    'entropy_coef': 0.01,           # 熵正则化系数
    'value_loss_coef': 0.5,         # 价值损失系数
    'max_grad_norm': 0.5,           # 梯度裁剪
    'ppo_epochs': 10,               # 每次更新的PPO迭代次数
    'mini_batch_size': 64,          # 小批量大小
    'update_freq': 2048,            # 更新频率(收集多少步后更新)
}

# ============ 训练配置 ============
TRAIN_CONFIG = {
    'total_timesteps': 2_000_000,     # 总训练步数
    'num_envs': 8,                    # 并行环境数量
    # 下列频率均为update_freq(2048)的整数倍，避免训练循环取整后为0
    'log_freq': 10_240,               # 5个迭代记录一次（约5k步）
    'eval_freq': 51_200,             # 25个迭代评估一次
    'save_freq': 51_200,             # 25个迭代保存一次
    'eval_episodes': 20,              # 评估局数
    'checkpoint_dir': './train/checkpoints',  # 检查点目录
    'log_dir': './train/logs',        # 日志目录
}

# ============ 奖励配置 ============
REWARD_CONFIG = {
    # 基础奖励
    'own_ball_pocketed': 120.0,             # 打进己方目标球
    'own_ball_pocketed_bonus_targeted': 60.0,  # 打进当前瞄准的球额外奖励
    'legal_eight_pocketed': 520.0,          # 合法打进8号球(获胜)
    
    # 惩罚
    'cue_pocketed': -150.0,                 # 白球进袋
    'illegal_eight_pocketed': -1000.0,      # 非法打进8号球(直接判负)
    'cue_and_eight_pocketed': -1000.0,      # 白球和8号球同时进袋
    'enemy_ball_pocketed': -15.0,           # 打进对方球(轻微惩罚)
    'foul_first_hit': -40.0,                # 首球犯规
    'no_rail_foul': -40.0,                  # 未碰库犯规
    'no_hit_foul': -70.0,                   # 未击中任何球
    
    # 形势奖励/惩罚
    'good_position': 12.0,                  # 好的走位(白球与下一目标球距离近)
    'continue_shot': 25.0,                  # 连续击球权奖励
    'lose_turn': -1.0,                      # 失去击球权
    'pocket_progress_weight': 30.0,         # 目标球最紧口袋距离的改善提示
    'cue_target_align_weight': 6.0,         # 白球距目标球的改善提示
    
    # 游戏结束奖励
    'win_game': 200.0,                      # 赢得比赛额外奖励
    'lose_game': -200.0,                    # 输掉比赛额外惩罚
    
    # 时间惩罚(鼓励快速结束)
    'step_penalty': -0.2,                   # 每步小惩罚
}

# ============ 动作空间边界 ============
ACTION_BOUNDS = {
    'V0': (0.5, 8.0),       # 初速度 m/s
    'phi': (0.0, 360.0),    # 水平角度 度
    'theta': (0.0, 90.0),   # 垂直角度 度
    'a': (-0.5, 0.5),       # 横向偏移
    'b': (-0.5, 0.5),       # 纵向偏移
}
