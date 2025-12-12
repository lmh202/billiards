"""Phase-2 training: train an agent against BasicAgent using the full rules.

This script loads phase-1 pretrained weights (if available) and then trains
the agent in the complete 8-ball environment against BasicAgent. This phase
adds rule understanding and strategic play on top of basic potting skills.
"""

import os
from train.ppo_trainer import PPOTrainer
from train.config import DEVICE, TRAIN_CONFIG, PPO_CONFIG


if __name__ == '__main__':
    # create trainer with standard 8-ball env playing against BasicAgent
    trainer = PPOTrainer(
        config=PPO_CONFIG,
        train_config=TRAIN_CONFIG,
        device=DEVICE,
        opponent_type='basic',
        enable_noise=True
    )

    # try to load phase-1 pretrained weights
    phase1_path = os.path.join(TRAIN_CONFIG['checkpoint_dir'], 'phase1_pretrained.pt')
    if os.path.exists(phase1_path):
        print(f"Loading phase-1 weights from: {phase1_path}")
        trainer.load_checkpoint(phase1_path)
    else:
        print("No phase-1 weights found, starting from scratch")

    # train for the full duration configured
    total_timesteps = TRAIN_CONFIG['total_timesteps']
    print(f"Starting phase-2 training for {total_timesteps} timesteps against BasicAgent")
    trainer.train(total_timesteps=total_timesteps)

    # save final model
    trainer.save_checkpoint('phase2_vs_basic.pt')
    print("Phase-2 training complete, weights saved as phase2_vs_basic.pt")
