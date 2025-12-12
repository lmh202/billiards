"""Phase-1 training: train an agent to pot a single target ball using a simplified two-ball environment.

This script runs a short PPO training on the simplified two-ball environment created
inside `train.pool_rl_env.PoolRLEnv(two_ball_mode=True)`. It saves the final model to
`checkpoints/phase1_pretrained.pt` which can be loaded later by `NewAgent` for further
rule-based training against `BasicAgent`.
"""

from train.ppo_trainer import PPOTrainer
from train.pool_rl_env import VectorPoolEnv
from train.config import DEVICE, TRAIN_CONFIG, PPO_CONFIG
import os

if __name__ == '__main__':
    # create trainer with two-ball env
    trainer = PPOTrainer(
        config=PPO_CONFIG,
        train_config=TRAIN_CONFIG,
        device=DEVICE,
        opponent_type='random',
        enable_noise=True
    )

    # override envs to use two-ball mode
    trainer.envs = VectorPoolEnv(
        num_envs=trainer.num_envs,
        device=trainer.device,
        opponent_type='random',
        enable_noise=True,
        two_ball_mode=True
    )

    # adjust training length for phase-1 (shorter by default)
    total_timesteps = 5_000_000
    print(f"Starting phase-1 pretraining for {total_timesteps} timesteps on two-ball env")
    trainer.train(total_timesteps=total_timesteps)

    # save with clear name for later reuse
    ckdir = TRAIN_CONFIG['checkpoint_dir']
    os.makedirs(ckdir, exist_ok=True)
    outpath = os.path.join(ckdir, 'phase1_pretrained.pt')
    trainer.save_checkpoint('phase1_pretrained.pt')
    print(f"Phase-1 pretrained model saved to: {outpath}")
