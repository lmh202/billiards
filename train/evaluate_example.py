"""
使用MuZero模型进行评估的示例

这个文件展示如何修改evaluate.py来使用训练好的MuZero模型
"""

# ============ 方法1: 修改evaluate.py ============
"""
在evaluate.py中，找到这一行：

    agent_a, agent_b = BasicAgent(), NewAgent()

替换为：

    # 使用训练好的MuZero模型
    agent_a = BasicAgent()
    agent_b = NewAgent(checkpoint_path="checkpoints/muzero_final.pth")

或者让两个MuZero模型对战：

    agent_a = NewAgent(checkpoint_path="checkpoints/muzero_final.pth")
    agent_b = NewAgent(checkpoint_path="checkpoints/muzero_final.pth")
"""

# ============ 方法2: 创建自定义评估脚本 ============

from utils import set_random_seed
from poolenv import PoolEnv
from agent import BasicAgent, NewAgent

# 设置随机种子
set_random_seed(enable=False, seed=42)

# 创建环境
env = PoolEnv()
results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
n_games = 120  # 对战局数

# ===== 在这里指定检查点路径 =====
CHECKPOINT_PATH = "checkpoints/muzero_final.pth"  # 修改为你的模型路径

# 创建Agent
print("加载模型...")
agent_a = BasicAgent()  # 对手使用BasicAgent
agent_b = NewAgent(checkpoint_path=CHECKPOINT_PATH)  # 我们的MuZero模型
print(f"Agent A: BasicAgent")
print(f"Agent B: NewAgent with model {CHECKPOINT_PATH}\n")

players = [agent_a, agent_b]
target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']

for i in range(n_games): 
    print()
    print(f"------- 第 {i+1}/{n_games} 局比赛 -------")
    env.reset(target_ball=target_ball_choice[i % 4])
    player_class = players[i % 2].__class__.__name__
    ball_type = target_ball_choice[i % 4]
    print(f"本局 Player A: {player_class}, 目标球型: {ball_type}")
    
    while True:
        player = env.get_curr_player()
        print(f"[第{env.hit_count}次击球] player: {player}")
        obs = env.get_observation(player)
        
        if player == 'A':
            action = players[i % 2].decision(*obs)
        else:
            action = players[(i + 1) % 2].decision(*obs)
        
        step_info = env.take_shot(action)
        
        done, info = env.get_done()
        if not done and step_info.get('ENEMY_INTO_POCKET'):
            print(f"对方球入袋：{step_info['ENEMY_INTO_POCKET']}")
        
        if done:
            # 统计结果
            if info['winner'] == 'SAME':
                results['SAME'] += 1
                print(f"平局！")
            elif info['winner'] == 'A':
                winner_agent = ['AGENT_A_WIN', 'AGENT_B_WIN'][i % 2]
                results[winner_agent] += 1
                winner_name = "BasicAgent" if winner_agent == 'AGENT_A_WIN' else "NewAgent(MuZero)"
                print(f"🏆 {winner_name} 获胜！")
            else:
                winner_agent = ['AGENT_A_WIN', 'AGENT_B_WIN'][(i+1) % 2]
                results[winner_agent] += 1
                winner_name = "BasicAgent" if winner_agent == 'AGENT_A_WIN' else "NewAgent(MuZero)"
                print(f"🏆 {winner_name} 获胜！")
            break

# 计算分数
results['AGENT_A_SCORE'] = results['AGENT_A_WIN'] * 1 + results['SAME'] * 0.5
results['AGENT_B_SCORE'] = results['AGENT_B_WIN'] * 1 + results['SAME'] * 0.5

print("\n" + "="*60)
print("最终统计结果")
print("="*60)
print(f"Agent A (BasicAgent) 获胜: {results['AGENT_A_WIN']} 局")
print(f"Agent B (MuZero) 获胜: {results['AGENT_B_WIN']} 局")
print(f"平局: {results['SAME']} 局")
print(f"\nAgent A 得分: {results['AGENT_A_SCORE']:.1f}")
print(f"Agent B 得分: {results['AGENT_B_SCORE']:.1f}")
print(f"\nMuZero胜率: {results['AGENT_B_WIN']/n_games*100:.1f}%")
print("="*60)


# ============ 方法3: 快速测试单局 ============
def quick_test(checkpoint_path):
    """快速测试单局游戏"""
    print("\n快速测试模式...")
    
    env = PoolEnv()
    agent_a = BasicAgent()
    agent_b = NewAgent(checkpoint_path=checkpoint_path)
    
    env.reset(target_ball='solid')
    print("Player A: BasicAgent (solid)")
    print("Player B: MuZero (stripe)\n")
    
    move_count = 0
    while True:
        player = env.get_curr_player()
        obs = env.get_observation(player)
        
        if player == 'A':
            action = agent_a.decision(*obs)
        else:
            action = agent_b.decision(*obs)
        
        env.take_shot(action)
        move_count += 1
        
        done, info = env.get_done()
        if done:
            print(f"\n游戏结束！")
            print(f"胜者: {info['winner']}")
            print(f"总步数: {move_count}")
            break
    
    return info['winner']


if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # 快速测试模式
        checkpoint = "checkpoints/muzero_final.pth"
        if len(sys.argv) > 2:
            checkpoint = sys.argv[2]
        quick_test(checkpoint)
    else:
        # 完整评估模式
        print("""
MuZero模型评估脚本
==================

用法:
  python train/evaluate_example.py           # 运行120局完整评估
  python train/evaluate_example.py quick     # 快速测试单局
  python train/evaluate_example.py quick <checkpoint_path>  # 指定模型路径

注意:
  - 请先修改CHECKPOINT_PATH变量指向你的模型文件
  - 完整评估需要较长时间（约30-60分钟）
  - 可以按Ctrl+C中断评估
        """)
        input("按Enter开始评估...")
        # 这里会执行上面的完整评估代码
