"""
测试MuZero奖励计算函数
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.muzero_train import calculate_muzero_reward


def test_reward_calculations():
    """测试各种情况下的奖励计算"""
    
    print("="*60)
    print("MuZero奖励机制测试")
    print("="*60)
    
    # 模拟环境和结果
    class MockEnv:
        def __init__(self, done=False, winner=None):
            self.done = done
            self.winner = winner
        
        def get_done(self):
            if self.done:
                return True, {'winner': self.winner}
            return False, {}
    
    player = 'A'
    
    # 测试1: 完胜
    print("\n测试1: 完胜对手")
    env = MockEnv(done=True, winner='A')
    result = {}
    reward = calculate_muzero_reward(result, env, player)
    print(f"  奖励: {reward:,}")
    assert reward == 100000, f"期望100000，实际{reward}"
    print("  ✓ 通过")
    
    # 测试2: 失败
    print("\n测试2: 游戏失败")
    env = MockEnv(done=True, winner='B')
    result = {}
    reward = calculate_muzero_reward(result, env, player)
    print(f"  奖励: {reward:,}")
    assert reward == -4000, f"期望-4000，实际{reward}"
    print("  ✓ 通过")
    
    # 测试3: 打进己方球
    print("\n测试3: 打进3个己方球")
    env = MockEnv(done=False)
    result = {
        'ME_INTO_POCKET': ['1', '2', '3'],
        'ENEMY_INTO_POCKET': []
    }
    reward = calculate_muzero_reward(result, env, player)
    print(f"  奖励: {reward:,}")
    assert reward == 3000, f"期望3000，实际{reward}"
    print("  ✓ 通过")
    
    # 测试4: 打进对方球
    print("\n测试4: 打进2个对方球")
    env = MockEnv(done=False)
    result = {
        'ME_INTO_POCKET': [],
        'ENEMY_INTO_POCKET': ['9', '10']
    }
    reward = calculate_muzero_reward(result, env, player)
    print(f"  奖励: {reward:,}")
    assert reward == -1000, f"期望-1000，实际{reward}"
    print("  ✓ 通过")
    
    # 测试5: 白球进袋
    print("\n测试5: 白球进袋（致命犯规）")
    env = MockEnv(done=False)
    result = {
        'ME_INTO_POCKET': [],
        'ENEMY_INTO_POCKET': [],
        'WHITE_BALL_INTO_POCKET': True
    }
    reward = calculate_muzero_reward(result, env, player)
    print(f"  奖励: {reward:,}")
    assert reward == -4000, f"期望-4000，实际{reward}"
    print("  ✓ 通过")
    
    # 测试6: 首球犯规
    print("\n测试6: 首球犯规")
    env = MockEnv(done=False)
    result = {
        'ME_INTO_POCKET': [],
        'ENEMY_INTO_POCKET': [],
        'FOUL_FIRST_HIT': True
    }
    reward = calculate_muzero_reward(result, env, player)
    print(f"  奖励: {reward:,}")
    assert reward == -500, f"期望-500，实际{reward}"
    print("  ✓ 通过")
    
    # 测试7: 无进球无碰库
    print("\n测试7: 无进球无碰库")
    env = MockEnv(done=False)
    result = {
        'ME_INTO_POCKET': [],
        'ENEMY_INTO_POCKET': [],
        'NO_POCKET_NO_RAIL': True
    }
    reward = calculate_muzero_reward(result, env, player)
    print(f"  奖励: {reward:,}")
    assert reward == -200, f"期望-200，实际{reward}"
    print("  ✓ 通过")
    
    # 测试8: 安全球
    print("\n测试8: 安全球（无犯规无进球）")
    env = MockEnv(done=False)
    result = {
        'ME_INTO_POCKET': [],
        'ENEMY_INTO_POCKET': []
    }
    reward = calculate_muzero_reward(result, env, player)
    print(f"  奖励: {reward:,}")
    assert reward == 50, f"期望50，实际{reward}"
    print("  ✓ 通过")
    
    # 测试9: 复合情况 - 打进己方球同时误进对方球
    print("\n测试9: 打进2个己方球+1个对方球")
    env = MockEnv(done=False)
    result = {
        'ME_INTO_POCKET': ['1', '2'],
        'ENEMY_INTO_POCKET': ['9']
    }
    reward = calculate_muzero_reward(result, env, player)
    print(f"  奖励: {reward:,}")
    expected = 2000 - 500  # 2*1000 - 1*500
    assert reward == expected, f"期望{expected}，实际{reward}"
    print("  ✓ 通过")
    
    # 测试10: 复合情况 - 打进己方球但首球犯规
    print("\n测试10: 打进1个己方球但首球犯规")
    env = MockEnv(done=False)
    result = {
        'ME_INTO_POCKET': ['1'],
        'ENEMY_INTO_POCKET': [],
        'FOUL_FIRST_HIT': True
    }
    reward = calculate_muzero_reward(result, env, player)
    print(f"  奖励: {reward:,}")
    expected = 1000 - 500  # 1*1000 - 500
    assert reward == expected, f"期望{expected}，实际{reward}"
    print("  ✓ 通过")
    
    print("\n" + "="*60)
    print("所有测试通过！✓")
    print("="*60)
    
    # 打印奖励总结
    print("\n奖励机制总结：")
    print("  胜利: +100,000")
    print("  失败: -4,000")
    print("  己方进球: +1,000/球")
    print("  对方进球: -500/球")
    print("  白球进袋: -4,000")
    print("  首球犯规: -500")
    print("  无进球无碰库: -200")
    print("  安全球: +50")


if __name__ == "__main__":
    test_reward_calculations()
