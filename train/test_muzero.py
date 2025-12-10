"""
快速测试脚本 - 用于验证MuZero实现
"""

import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """测试所有必要的导入"""
    print("测试导入...")
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ PyTorch导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy版本: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy导入失败: {e}")
        return False
    
    try:
        import pooltool as pt
        print(f"✓ Pooltool导入成功")
    except ImportError as e:
        print(f"✗ Pooltool导入失败: {e}")
        return False
    
    return True


def test_muzero_components():
    """测试MuZero组件"""
    print("\n测试MuZero组件...")
    try:
        from train.muzero_train import (
            MuZeroConfig, MuZeroNetwork, MCTS, 
            encode_state, action_index_to_dict,
            RepresentationNetwork, DynamicsNetwork, PredictionNetwork
        )
        print("✓ MuZero组件导入成功")
        
        # 创建配置
        config = MuZeroConfig()
        print(f"✓ 配置创建成功 (设备: {config.device})")
        
        # 创建网络
        network = MuZeroNetwork(config)
        print(f"✓ 网络创建成功")
        
        # 测试网络前向传播
        import torch
        dummy_obs = torch.randn(1, 67)
        hidden, policy, value = network.initial_inference(dummy_obs)
        print(f"✓ 初始推理测试通过 (hidden: {hidden.shape}, policy: {policy.shape}, value: {value.shape})")
        
        # 测试动作转换
        action_dict = action_index_to_dict(0, config)
        print(f"✓ 动作转换测试通过: {action_dict}")
        
        return True
    except Exception as e:
        print(f"✗ MuZero组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent():
    """测试Agent"""
    print("\n测试NewAgent...")
    try:
        from agent import NewAgent
        
        # 创建Agent（不加载模型）
        agent = NewAgent(checkpoint_path=None)
        print("✓ NewAgent创建成功（无模型）")
        
        # 测试随机决策
        action = agent._random_action()
        print(f"✓ 随机动作生成成功: V0={action['V0']:.2f}, phi={action['phi']:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Agent测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment():
    """测试环境"""
    print("\n测试环境...")
    try:
        from poolenv import PoolEnv
        
        env = PoolEnv()
        env.reset(target_ball='solid')
        print("✓ 环境重置成功")
        
        balls, my_targets, table = env.get_observation('A')
        print(f"✓ 观测获取成功 (目标球: {my_targets})")
        
        # 测试状态编码
        from train.muzero_train import encode_state
        obs = encode_state(balls, my_targets, table)
        print(f"✓ 状态编码成功 (形状: {obs.shape})")
        
        return True
    except Exception as e:
        print(f"✗ 环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_loop():
    """测试训练循环（单次迭代）"""
    print("\n测试训练循环（小规模）...")
    try:
        import torch
        from train.muzero_train import MuZeroConfig, MuZeroNetwork, play_game
        from poolenv import PoolEnv
        
        # 创建小规模配置
        config = MuZeroConfig()
        config.num_simulations = 5  # 减少模拟次数
        config.max_moves = 10  # 减少最大步数
        
        network = MuZeroNetwork(config).to(config.device)
        env = PoolEnv()
        
        print("开始单局游戏测试...")
        game_history = play_game(config, network, env, train_mode=False)
        
        print(f"✓ 游戏完成: {len(game_history.observations)}步")
        print(f"  总奖励: {sum(game_history.rewards):.2f}")
        
        return True
    except Exception as e:
        print(f"✗ 训练循环测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("MuZero实现验证测试")
    print("=" * 60)
    
    results = []
    
    # 测试1: 导入
    results.append(("导入测试", test_imports()))
    
    # 测试2: MuZero组件
    results.append(("MuZero组件测试", test_muzero_components()))
    
    # 测试3: Agent
    results.append(("Agent测试", test_agent()))
    
    # 测试4: 环境
    results.append(("环境测试", test_environment()))
    
    # 测试5: 训练循环（可选，较慢）
    print("\n是否运行训练循环测试？这可能需要几分钟...")
    try:
        user_input = input("输入 'y' 继续，或按Enter跳过: ").strip().lower()
        if user_input == 'y':
            results.append(("训练循环测试", test_training_loop()))
    except:
        print("跳过训练循环测试")
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！可以开始训练了。")
        print("\n运行训练命令:")
        print("  conda activate billiards")
        print("  python train/muzero_train.py")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息并修复。")


if __name__ == "__main__":
    main()
