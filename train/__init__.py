"""
train - 训练模块

包含:
- networks: 神经网络定义
- state_utils: 状态处理工具
- collect_data: 数据收集
- train_bc: 行为克隆训练
- train_ppo: PPO训练
- train: 完整训练流程
"""

from .networks import PolicyNetwork, ValueNetwork, ActorCritic, BCPolicyNetwork
from .state_utils import (
    extract_state_features, 
    action_to_network_target, 
    network_output_to_action,
    state_dict_to_tensor,
    BCDataset,
    ReplayBuffer
)
