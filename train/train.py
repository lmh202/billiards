"""
train.py - 完整训练流程主脚本

训练流程:
1. 阶段1: 数据收集 - 使用BasicAgentPro收集专家数据
2. 阶段2: BC训练 - 行为克隆，学习专家的下限
3. 阶段3: DAgger - 迭代数据收集，解决分布偏移
4. 阶段4: PPO微调 - 自博弈训练，学习超越专家
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime

from collect_data import collect_expert_data
from train_bc import BCTrainer, BCAgent, run_dagger
from train_ppo import train_ppo
from state_utils import BCDataset


def full_training_pipeline(
    # 数据收集参数
    n_expert_games: int = 200,
    # BC训练参数
    bc_epochs: int = 200,
    bc_lr: float = 1e-3,
    bc_batch_size: int = 64,
    # DAgger参数
    use_dagger: bool = True,
    dagger_iterations: int = 3,
    dagger_games_per_iter: int = 30,
    # PPO参数
    ppo_games: int = 2000,
    ppo_update_freq: int = 10,
    ppo_batch_size: int = 64,
    ppo_epochs: int = 10,
    ppo_lr: float = 3e-4,
    # 通用参数
    hidden_dim: int = 256,
    output_dir: str = 'train/checkpoints'
):
    """
    完整训练流程
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("=" * 60)
    print("台球智能体训练流程")
    print("算法: BC/DAgger + PPO自博弈微调")
    print("=" * 60)
    
    # ========== 阶段1: 专家数据收集 ==========
    print("\n" + "=" * 60)
    print("阶段1: 专家数据收集")
    print("=" * 60)
    
    expert_data_path = os.path.join(output_dir, 'expert_data.npz')
    
    if os.path.exists(expert_data_path):
        print(f"发现已有专家数据 {expert_data_path}，跳过收集")
        dataset = BCDataset()
        dataset.load(expert_data_path)
    else:
        dataset = collect_expert_data(
            n_games=n_expert_games,
            save_path=expert_data_path,
            verbose=True
        )
    
    print(f"专家数据量: {len(dataset)}")
    
    # ========== 阶段2: BC训练 ==========
    print("\n" + "=" * 60)
    print("阶段2: 行为克隆 (BC) 训练")
    print("=" * 60)
    
    bc_model_path = os.path.join(output_dir, 'bc_model.pt')
    bc_best_path = os.path.join(output_dir, 'bc_model_best.pt')
    
    bc_trainer = BCTrainer(hidden_dim=hidden_dim, lr=bc_lr)
    bc_trainer.train(
        dataset=dataset,
        n_epochs=bc_epochs,
        batch_size=bc_batch_size,
        save_path=bc_model_path
    )
    
    # ========== 阶段3: DAgger ==========
    if use_dagger:
        print("\n" + "=" * 60)
        print("阶段3: DAgger 迭代优化")
        print("=" * 60)
        
        run_dagger(
            bc_trainer=bc_trainer,
            dataset=dataset,
            n_iterations=dagger_iterations,
            games_per_iter=dagger_games_per_iter
        )
        
        # 更新模型路径
        bc_best_path = os.path.join(output_dir, f'bc_model_dagger{dagger_iterations}.pt')
    
    # ========== 阶段4: PPO自博弈微调 ==========
    print("\n" + "=" * 60)
    print("阶段4: PPO自博弈微调")
    print("=" * 60)
    
    ppo_model_path = os.path.join(output_dir, 'ppo_model.pt')
    
    train_ppo(
        bc_model_path=bc_best_path,
        n_games=ppo_games,
        update_freq=ppo_update_freq,
        batch_size=ppo_batch_size,
        n_epochs=ppo_epochs,
        save_freq=200,
        save_path=ppo_model_path,
        hidden_dim=hidden_dim,
        lr=ppo_lr
    )
    
    # ========== 完成 ==========
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"BC模型: {bc_model_path}")
    print(f"PPO模型: {ppo_model_path}")
    print(f"\n将 {ppo_model_path} 复制到 eval/ 目录，并更新 agents/new_agent.py 即可使用")


def quick_bc_only(
    n_games: int = 100,
    epochs: int = 100,
    output_dir: str = 'train/checkpoints'
):
    """快速BC训练（用于测试）"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("快速BC训练模式")
    
    # 数据收集
    expert_data_path = os.path.join(output_dir, 'expert_data.npz')
    if os.path.exists(expert_data_path):
        dataset = BCDataset()
        dataset.load(expert_data_path)
    else:
        dataset = collect_expert_data(n_games=n_games, save_path=expert_data_path)
    
    # BC训练
    bc_model_path = os.path.join(output_dir, 'bc_model.pt')
    trainer = BCTrainer(hidden_dim=256, lr=1e-3)
    trainer.train(dataset, n_epochs=epochs, save_path=bc_model_path)
    
    print(f"BC模型已保存到 {bc_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='台球智能体训练')
    parser.add_argument('--mode', type=str, default='full', 
                        choices=['full', 'bc_only', 'ppo_only'],
                        help='训练模式: full(完整), bc_only(仅BC), ppo_only(仅PPO)')
    
    # 数据收集参数
    parser.add_argument('--n_expert_games', type=int, default=200, help='专家数据收集局数')
    
    # BC参数
    parser.add_argument('--bc_epochs', type=int, default=200, help='BC训练轮数')
    parser.add_argument('--bc_lr', type=float, default=1e-3, help='BC学习率')
    
    # DAgger参数
    parser.add_argument('--no_dagger', action='store_true', help='不使用DAgger')
    parser.add_argument('--dagger_iters', type=int, default=3, help='DAgger迭代次数')
    
    # PPO参数
    parser.add_argument('--ppo_games', type=int, default=2000, help='PPO训练局数')
    parser.add_argument('--ppo_lr', type=float, default=3e-4, help='PPO学习率')
    
    # 通用参数
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--output_dir', type=str, default='train/checkpoints', help='输出目录')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        full_training_pipeline(
            n_expert_games=args.n_expert_games,
            bc_epochs=args.bc_epochs,
            bc_lr=args.bc_lr,
            use_dagger=not args.no_dagger,
            dagger_iterations=args.dagger_iters,
            ppo_games=args.ppo_games,
            ppo_lr=args.ppo_lr,
            hidden_dim=args.hidden_dim,
            output_dir=args.output_dir
        )
    elif args.mode == 'bc_only':
        quick_bc_only(
            n_games=args.n_expert_games,
            epochs=args.bc_epochs,
            output_dir=args.output_dir
        )
    elif args.mode == 'ppo_only':
        train_ppo(
            bc_model_path=os.path.join(args.output_dir, 'bc_model_best.pt'),
            n_games=args.ppo_games,
            save_path=os.path.join(args.output_dir, 'ppo_model.pt'),
            hidden_dim=args.hidden_dim,
            lr=args.ppo_lr
        )
