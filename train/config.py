"""
MuZero配置文件 - 可以通过修改此文件来调整训练参数
"""

class TrainingConfig:
    """训练配置 - 根据需要修改这些参数"""
    
    # ==================== 环境参数 ====================
    # 动作空间离散化
    V0_BINS = 10  # 速度离散化数量 (0.5-8.0 m/s)
    PHI_BINS = 10  # 角度离散化数量 (0-360度)
    ACTION_SPACE_SIZE = V0_BINS * PHI_BINS  # 总动作数 = 100
    
    # ==================== 网络参数 ====================
    HIDDEN_STATE_SIZE = 256  # 隐藏状态维度
    ENCODING_SIZE = 64  # 编码层大小
    
    # ==================== MCTS参数 ====================
    # 训练时的模拟次数（越大越准确但越慢）
    NUM_SIMULATIONS_TRAIN = 50
    # 评估时的模拟次数
    NUM_SIMULATIONS_EVAL = 30
    
    # UCB公式参数
    PB_C_BASE = 19652
    PB_C_INIT = 1.25
    
    # 探索参数
    ROOT_DIRICHLET_ALPHA = 0.3
    ROOT_EXPLORATION_FRACTION = 0.25
    
    # ==================== 训练参数 ====================
    # 总训练步数（建议：快速测试1000，正式训练10000+）
    TRAINING_STEPS = 10000
    
    # 批大小（内存不足时可减小）
    BATCH_SIZE = 64
    
    # 每次迭代的自我对弈局数
    SELF_PLAY_GAMES = 100
    
    # 展开步数（用于训练）
    NUM_UNROLL_STEPS = 5
    
    # TD(n)步数
    TD_STEPS = 10
    
    # 学习率
    LEARNING_RATE = 0.001
    LR_DECAY_RATE = 0.1
    LR_DECAY_STEPS = 5000
    
    # 权重衰减（L2正则化）
    WEIGHT_DECAY = 1e-4
    
    # ==================== 经验回放参数 ====================
    # 回放缓冲区大小（内存不足时可减小）
    REPLAY_BUFFER_SIZE = 10000
    
    # 优先级采样参数
    PRIORITY_ALPHA = 0.6  # 优先级指数
    PRIORITY_BETA = 0.4  # 重要性采样指数
    
    # ==================== 奖励和价值范围 ====================
    # 根据图片评分标准设定：
    # 最大奖励：完胜 +100,000 或 打进所有球 7*1000 = 7,000
    # 最小奖励：致命犯规 -4,000
    # 价值范围需要考虑累积奖励
    VALUE_SUPPORT_MIN = -10000
    VALUE_SUPPORT_MAX = 110000
    
    REWARD_SUPPORT_MIN = -5000
    REWARD_SUPPORT_MAX = 105000
    
    # ==================== 游戏参数 ====================
    MAX_MOVES = 60  # 每局最大步数
    
    # ==================== 保存和日志 ====================
    # 检查点保存间隔（迭代次数）
    CHECKPOINT_INTERVAL = 100
    
    # 保存目录
    SAVE_DIR = "checkpoints"
    
    # 日志间隔（训练步数）
    LOG_INTERVAL = 10
    
    # ==================== 设备设置 ====================
    # 'cuda' 或 'cpu'，留空自动检测
    DEVICE = None  # None = 自动检测


# ==================== 预设配置 ====================

class QuickTestConfig(TrainingConfig):
    """快速测试配置 - 用于验证代码能否运行"""
    TRAINING_STEPS = 100
    SELF_PLAY_GAMES = 5
    NUM_SIMULATIONS_TRAIN = 10
    NUM_SIMULATIONS_EVAL = 5
    BATCH_SIZE = 16
    REPLAY_BUFFER_SIZE = 500
    MAX_MOVES = 20
    CHECKPOINT_INTERVAL = 50


class SmallScaleConfig(TrainingConfig):
    """小规模配置 - 用于快速训练和调试"""
    TRAINING_STEPS = 1000
    SELF_PLAY_GAMES = 20
    NUM_SIMULATIONS_TRAIN = 30
    NUM_SIMULATIONS_EVAL = 20
    BATCH_SIZE = 32
    REPLAY_BUFFER_SIZE = 2000
    CHECKPOINT_INTERVAL = 200


class ProductionConfig(TrainingConfig):
    """生产配置 - 用于正式训练"""
    TRAINING_STEPS = 20000
    SELF_PLAY_GAMES = 100
    NUM_SIMULATIONS_TRAIN = 80
    NUM_SIMULATIONS_EVAL = 50
    BATCH_SIZE = 128
    REPLAY_BUFFER_SIZE = 20000
    CHECKPOINT_INTERVAL = 500


# ==================== 使用说明 ====================
"""
使用方法：

1. 快速测试（验证代码）：
   config = QuickTestConfig()

2. 小规模训练（1-2小时）：
   config = SmallScaleConfig()

3. 正式训练（4-8小时）：
   config = ProductionConfig()

4. 自定义配置：
   class MyConfig(TrainingConfig):
       TRAINING_STEPS = 5000
       NUM_SIMULATIONS_TRAIN = 60
       # ... 其他参数
   
   config = MyConfig()

然后在muzero_train.py中使用：
   from config import QuickTestConfig  # 或其他配置
   config = QuickTestConfig()
"""
