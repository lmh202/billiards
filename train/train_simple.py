"""
简化的训练启动脚本
用法: python train/train_simple.py [config_name]
配置选项: quick, small, production, default
"""

import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_config(config_name='default'):
    """根据名称获取配置"""
    if config_name == 'quick':
        from train.config import QuickTestConfig
        print("使用快速测试配置（约10分钟）")
        return QuickTestConfig()
    elif config_name == 'small':
        from train.config import SmallScaleConfig
        print("使用小规模配置（约1-2小时）")
        return SmallScaleConfig()
    elif config_name == 'production':
        from train.config import ProductionConfig
        print("使用生产配置（约4-8小时）")
        return ProductionConfig()
    else:
        from train.config import TrainingConfig
        print("使用默认配置（约2-4小时）")
        return TrainingConfig()


def print_config_info(config):
    """打印配置信息"""
    print("\n" + "="*60)
    print("训练配置信息")
    print("="*60)
    print(f"训练步数: {config.TRAINING_STEPS}")
    print(f"每次迭代游戏数: {config.SELF_PLAY_GAMES}")
    print(f"MCTS模拟次数: {config.NUM_SIMULATIONS_TRAIN}")
    print(f"批大小: {config.BATCH_SIZE}")
    print(f"回放缓冲区大小: {config.REPLAY_BUFFER_SIZE}")
    print(f"检查点保存间隔: {config.CHECKPOINT_INTERVAL}")
    print(f"保存目录: {config.SAVE_DIR}")
    print("="*60 + "\n")


def main():
    """主函数"""
    # 解析命令行参数
    config_name = 'default'
    if len(sys.argv) > 1:
        config_name = sys.argv[1].lower()
    
    # 获取配置
    user_config = get_config(config_name)
    print_config_info(user_config)
    
    # 导入训练模块
    print("加载训练模块...")
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from train.muzero_train import (
            MuZeroConfig, MuZeroNetwork, MCTS, ReplayBuffer,
            play_game, train_network
        )
        from poolenv import PoolEnv
        print("✓ 模块加载成功\n")
    except ImportError as e:
        print(f"✗ 模块加载失败: {e}")
        print("\n请确保：")
        print("1. 已激活conda环境: conda activate billiards")
        print("2. 已安装PyTorch: pip install torch")
        print("3. 已安装其他依赖: pip install numpy")
        return
    
    # 创建MuZero配置（使用用户配置）
    config = MuZeroConfig()
    
    # 应用用户配置
    config.training_steps = user_config.TRAINING_STEPS
    config.self_play_games = user_config.SELF_PLAY_GAMES
    config.num_simulations = user_config.NUM_SIMULATIONS_TRAIN
    config.num_simulations_eval = user_config.NUM_SIMULATIONS_EVAL
    config.batch_size = user_config.BATCH_SIZE
    config.replay_buffer_size = user_config.REPLAY_BUFFER_SIZE
    config.checkpoint_interval = user_config.CHECKPOINT_INTERVAL
    config.save_dir = user_config.SAVE_DIR
    config.max_moves = user_config.MAX_MOVES
    config.lr_init = user_config.LEARNING_RATE
    config.weight_decay = user_config.WEIGHT_DECAY
    
    # 设置设备
    if user_config.DEVICE:
        config.device = torch.device(user_config.DEVICE)
    else:
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {config.device}")
    if config.device.type == 'cpu':
        print("⚠️ 警告: 使用CPU训练会很慢，建议使用GPU")
    
    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 创建网络
    print("\n初始化网络...")
    network = MuZeroNetwork(config).to(config.device)
    optimizer = optim.Adam(network.parameters(), 
                          lr=config.lr_init, 
                          weight_decay=config.weight_decay)
    print("✓ 网络初始化完成")
    
    # 创建环境和缓冲区
    print("初始化环境和缓冲区...")
    env = PoolEnv()
    replay_buffer = ReplayBuffer(config)
    print("✓ 环境初始化完成\n")
    
    # 开始训练
    print("="*60)
    print("开始训练")
    print("="*60)
    
    training_step = 0
    num_iterations = config.training_steps // config.self_play_games
    
    try:
        for iteration in range(num_iterations):
            print(f"\n{'='*60}")
            print(f"迭代 {iteration + 1}/{num_iterations}")
            print(f"{'='*60}")
            
            # 自我对弈阶段
            print(f"自我对弈 ({config.self_play_games}局)...")
            for game_idx in range(config.self_play_games):
                game_history = play_game(config, network, env, train_mode=True)
                
                # 转换为字典格式
                game_dict = {
                    'observations': game_history.observations,
                    'actions': game_history.actions,
                    'rewards': game_history.rewards,
                    'policies': game_history.policies,
                    'values': game_history.values
                }
                replay_buffer.save_game(game_dict)
                
                if (game_idx + 1) % 10 == 0:
                    avg_reward = sum(game_history.rewards) / len(game_history.rewards) if game_history.rewards else 0
                    print(f"  [{game_idx + 1}/{config.self_play_games}] "
                          f"步数: {len(game_history.observations)}, "
                          f"平均奖励: {avg_reward:.2f}")
            
            # 训练阶段
            print(f"训练网络...")
            total_loss = 0
            num_batches = 0
            
            for _ in range(config.batch_size):
                batch = replay_buffer.sample_batch()
                if batch is None:
                    continue
                
                loss = train_network(config, network, optimizer, batch, training_step)
                total_loss += loss
                num_batches += 1
                training_step += 1
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"  平均损失: {avg_loss:.4f}")
            
            # 保存检查点
            if (iteration + 1) % config.checkpoint_interval == 0:
                checkpoint_path = os.path.join(config.save_dir, f"muzero_step_{training_step}.pth")
                torch.save({
                    'network_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training_step': training_step,
                    'config': config
                }, checkpoint_path)
                print(f"✓ 保存检查点: {checkpoint_path}")
        
        # 保存最终模型
        final_path = os.path.join(config.save_dir, "muzero_final.pth")
        torch.save({
            'network_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_step': training_step,
            'config': config
        }, final_path)
        
        print("\n" + "="*60)
        print("训练完成！")
        print("="*60)
        print(f"最终模型: {final_path}")
        print(f"总训练步数: {training_step}")
        print("\n使用方法:")
        print(f"  agent = NewAgent(checkpoint_path='{final_path}')")
        
    except KeyboardInterrupt:
        print("\n\n训练被中断！")
        print("保存当前模型...")
        interrupted_path = os.path.join(config.save_dir, "muzero_interrupted.pth")
        torch.save({
            'network_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_step': training_step,
            'config': config
        }, interrupted_path)
        print(f"✓ 模型已保存: {interrupted_path}")
    except Exception as e:
        print(f"\n✗ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("""
MuZero台球训练脚本
==================

用法:
  python train/train_simple.py [config]

配置选项:
  quick       - 快速测试配置（约10分钟）
  small       - 小规模配置（约1-2小时）
  production  - 生产配置（约4-8小时）
  default     - 默认配置（约2-4小时）[默认]

示例:
  python train/train_simple.py quick
  python train/train_simple.py small
  python train/train_simple.py

按Ctrl+C可随时中断训练并保存当前模型。
""")
    
    input("按Enter键开始训练...")
    print()
    
    main()
