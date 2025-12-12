"""
训练脚本：PPO (NewAgent) 对战 BasicAgent

用途：
- 使用 BasicAgent 作为固定对手，为 NewAgent 提供更密集的反馈信号
- 复用现有 PPOTrainer，只需将 opponent_type 设置为 'basic'
"""

from train.config import DEVICE, PPO_CONFIG, TRAIN_CONFIG, NETWORK_CONFIG
from train.ppo_trainer import PPOTrainer


def main():
    trainer = PPOTrainer(
        config=PPO_CONFIG,
        train_config=TRAIN_CONFIG,
        network_config=NETWORK_CONFIG,
        device=DEVICE,
        opponent_type='basic',
        enable_noise=False,  # 固定对手时先关噪声，便于收敛
    )
    trainer.train()


if __name__ == "__main__":
    main()
