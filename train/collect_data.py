"""
collect_data.py
数据收集脚本：运行对局并记录每一步的状态、当前击球方、最终胜负标签。
用于训练 Value Network。

采样策略（参考方案 C）:
1. NewAgent vs BasicAgentPro (最贴近目标分布)
2. NewAgent vs BasicAgent (快速产生多样局面)
3. NewAgent self-play (产生僵持/安全球局面)

记录方式：每次调用 decision() 前记录一个样本 (s_t, player_t, winner)
"""
import os
import sys
import json
import random
import argparse
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poolenv import PoolEnv
from agents.new_agent import NewAgent
from agents.basic_agent import BasicAgent
from agents.basic_agent_pro import BasicAgentPro
from train.state_utils import extract_state_features, get_feature_dim


class DataCollector:
    """对局数据收集器"""
    
    def __init__(self, output_dir='train/data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.samples = []  # [(features, player_id, winner)]
        self.game_count = 0
    
    def collect_game(self, env, agent_a, agent_b, agent_a_name='A', agent_b_name='B'):
        """
        运行一局游戏并收集数据。
        
        Args:
            env: PoolEnv 实例
            agent_a, agent_b: 两个 Agent
            agent_a_name, agent_b_name: Agent 标识
        
        Returns:
            winner: 'A' 或 'B'
            game_samples: 本局收集的样本列表
        """
        game_samples = []
        obs = env.reset()
        done = False
        
        current_player = 'A'
        agents = {'A': agent_a, 'B': agent_b}
        
        step_count = 0
        max_steps = 200  # 防止死循环
        
        while not done and step_count < max_steps:
            step_count += 1
            
            # 获取当前状态
            balls = env.system.balls
            my_targets = obs.get('target_balls', [])
            table = env.system.table
            
            # 记录状态特征
            features = extract_state_features(balls, my_targets, current_player)
            game_samples.append({
                'features': features,
                'player': current_player,
                'winner': None  # 稍后填充
            })
            
            # 获取动作
            agent = agents[current_player]
            try:
                action = agent.decision(balls, my_targets, table)
            except Exception as e:
                print(f"[DataCollector] Agent {current_player} decision error: {e}")
                action = {'V0': 2.0, 'phi': random.uniform(0, 360), 'theta': 0, 'a': 0, 'b': 0}
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            
            # 切换玩家（如果需要）
            if info.get('switch_player', False):
                current_player = 'B' if current_player == 'A' else 'A'
        
        # 确定获胜者
        winner = info.get('winner', None)
        if winner is None:
            # 如果没有明确胜者，根据剩余球数判断
            winner = 'A'  # 默认
        
        # 填充胜负标签（从当前击球方视角）
        for sample in game_samples:
            player = sample['player']
            # y_t = 1 如果当前击球方最终赢了
            sample['winner'] = 1.0 if sample['player'] == winner else 0.0
        
        self.samples.extend(game_samples)
        self.game_count += 1
        
        return winner, game_samples
    
    def collect_games(self, num_games, agent_configs, verbose=True):
        """
        收集多局游戏数据。
        
        Args:
            num_games: 要收集的游戏数量
            agent_configs: list of (agent_a_class, agent_b_class, weight)
                例如: [(NewAgent, BasicAgentPro, 0.4), (NewAgent, BasicAgent, 0.4), (NewAgent, NewAgent, 0.2)]
            verbose: 是否打印进度
        """
        env = PoolEnv()
        
        # 根据权重计算每种配置的游戏数
        total_weight = sum(cfg[2] for cfg in agent_configs)
        games_per_config = []
        for cfg in agent_configs:
            n = int(num_games * cfg[2] / total_weight)
            games_per_config.append(n)
        # 确保总数正确
        games_per_config[-1] += num_games - sum(games_per_config)
        
        game_idx = 0
        for (agent_a_cls, agent_b_cls, _), n_games in zip(agent_configs, games_per_config):
            if verbose:
                print(f"\n收集 {n_games} 局: {agent_a_cls.__name__} vs {agent_b_cls.__name__}")
            
            for i in range(n_games):
                agent_a = agent_a_cls()
                agent_b = agent_b_cls()
                
                try:
                    winner, samples = self.collect_game(env, agent_a, agent_b)
                    game_idx += 1
                    
                    if verbose and game_idx % 10 == 0:
                        print(f"  已完成 {game_idx}/{num_games} 局，当前样本数: {len(self.samples)}")
                except Exception as e:
                    print(f"  游戏 {game_idx} 出错: {e}")
                    continue
        
        if verbose:
            print(f"\n数据收集完成: {self.game_count} 局, {len(self.samples)} 个样本")
    
    def save_data(self, filename=None):
        """保存收集的数据"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"value_data_{timestamp}.npz"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 转换为 numpy 数组
        features = np.array([s['features'] for s in self.samples], dtype=np.float32)
        labels = np.array([s['winner'] for s in self.samples], dtype=np.float32)
        
        np.savez(filepath, features=features, labels=labels)
        print(f"数据已保存到 {filepath}")
        print(f"  特征形状: {features.shape}")
        print(f"  标签形状: {labels.shape}")
        print(f"  正样本比例: {labels.mean():.2%}")
        
        return filepath
    
    def load_data(self, filepath):
        """加载已保存的数据"""
        data = np.load(filepath)
        features = data['features']
        labels = data['labels']
        print(f"已加载数据: {filepath}")
        print(f"  特征形状: {features.shape}")
        print(f"  标签形状: {labels.shape}")
        return features, labels


def main():
    parser = argparse.ArgumentParser(description='收集 Value Network 训练数据')
    parser.add_argument('--num_games', type=int, default=100, help='收集的游戏局数')
    parser.add_argument('--output_dir', type=str, default='train/data', help='输出目录')
    parser.add_argument('--pro_ratio', type=float, default=0.3, help='vs BasicAgentPro 的比例')
    parser.add_argument('--basic_ratio', type=float, default=0.5, help='vs BasicAgent 的比例')
    parser.add_argument('--self_ratio', type=float, default=0.2, help='self-play 的比例')
    args = parser.parse_args()
    
    collector = DataCollector(output_dir=args.output_dir)
    
    # 配置采样策略
    agent_configs = [
        (NewAgent, BasicAgentPro, args.pro_ratio),
        (NewAgent, BasicAgent, args.basic_ratio),
        (NewAgent, NewAgent, args.self_ratio),
    ]
    
    print("=" * 50)
    print("Value Network 数据收集")
    print("=" * 50)
    print(f"目标局数: {args.num_games}")
    print(f"采样配置:")
    for cls_a, cls_b, ratio in agent_configs:
        print(f"  {cls_a.__name__} vs {cls_b.__name__}: {ratio:.0%}")
    print("=" * 50)
    
    collector.collect_games(args.num_games, agent_configs)
    collector.save_data()


if __name__ == "__main__":
    main()
