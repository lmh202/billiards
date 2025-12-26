"""
train_bc.py - 行为克隆训练脚本

功能:
- 加载专家数据
- 训练BC策略网络
- 支持DAgger迭代
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import argparse
from typing import Dict, Tuple

from networks import BCPolicyNetwork
from state_utils import (
    BCDataset, extract_state_features, action_to_network_target,
    network_output_to_action, state_dict_to_tensor
)


class BCTrainer:
    """行为克隆训练器"""
    
    def __init__(self, hidden_dim: int = 384, lr: float = 3e-4, device: str = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"[BCTrainer] 使用设备: {self.device}")
        
        self.model = BCPolicyNetwork(hidden_dim=hidden_dim).to(self.device)
        # 去掉过强的正则、降低学习率，提升拟合能力
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        # 角度损失使用余弦相似度
        self.cos_loss = nn.CosineEmbeddingLoss()
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        计算损失
        
        Args:
            pred: [batch, 6] 预测值
            target: [batch, 6] 目标值
            
        Returns:
            loss: 总损失
            loss_dict: 各项损失的字典
        """
        # V0损失
        v0_loss = self.mse_loss(pred[:, 0], target[:, 0])
        
        # 角度损失 (sin, cos)
        # 使用余弦相似度确保角度的周期性被正确处理
        pred_angle = pred[:, 1:3]  # sin_phi, cos_phi
        target_angle = target[:, 1:3]
        # 归一化到单位向量
        pred_angle_norm = pred_angle / (pred_angle.norm(dim=1, keepdim=True) + 1e-8)
        target_angle_norm = target_angle / (target_angle.norm(dim=1, keepdim=True) + 1e-8)
        # 角度损失 = 1 - cos_similarity
        angle_loss = 1 - (pred_angle_norm * target_angle_norm).sum(dim=1).mean()
        # 同时加入L2以稳定收敛
        angle_l2 = self.mse_loss(pred_angle_norm, target_angle_norm)
        
        # theta损失
        theta_loss = self.mse_loss(pred[:, 3], target[:, 3])
        
        # a, b损失
        ab_loss = self.mse_loss(pred[:, 4:], target[:, 4:])
        
        # 总损失 (加权)
        total_loss = (
            v0_loss * 1.0
            + angle_loss * 3.0
            + angle_l2 * 1.0
            + theta_loss * 1.0
            + ab_loss * 0.5
        )
        
        loss_dict = {
            'v0_loss': v0_loss.item(),
            'angle_loss': angle_loss.item(),
            'theta_loss': theta_loss.item(),
            'ab_loss': ab_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def train_epoch(self, dataset: BCDataset, batch_size: int = 64) -> Dict:
        """训练一个epoch"""
        self.model.train()
        
        n_samples = len(dataset)
        n_batches = max(1, n_samples // batch_size)
        
        epoch_losses = {
            'v0_loss': 0, 'angle_loss': 0, 'theta_loss': 0, 
            'ab_loss': 0, 'total_loss': 0
        }
        
        for _ in range(n_batches):
            # 采样批次
            batch_states, batch_actions = dataset.sample(batch_size)
            
            # 转换为张量
            states = {
                'ball_features': torch.from_numpy(batch_states['ball_features']).to(self.device),
                'pocket_positions': torch.from_numpy(batch_states['pocket_positions']).to(self.device),
                'extra_features': torch.from_numpy(batch_states['extra_features']).to(self.device)
            }
            actions = torch.from_numpy(batch_actions).to(self.device)
            
            # 前向传播
            pred_actions = self.model(states)
            
            # 计算损失
            loss, loss_dict = self.compute_loss(pred_actions, actions)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 累计损失
            for k, v in loss_dict.items():
                epoch_losses[k] += v / n_batches
        
        return epoch_losses
    
    def train(self, dataset: BCDataset, n_epochs: int = 300, batch_size: int = 128,
              save_path: str = 'bc_model.pt', save_freq: int = 20):
        """训练模型"""
        print(f"[BCTrainer] 开始训练，数据量: {len(dataset)}, epochs: {n_epochs}")
        
        best_loss = float('inf')
        
        for epoch in range(n_epochs):
            losses = self.train_epoch(dataset, batch_size)
            
            # 更新学习率
            self.scheduler.step(losses['total_loss'])
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | "
                      f"Total: {losses['total_loss']:.4f} | "
                      f"V0: {losses['v0_loss']:.4f} | "
                      f"Angle: {losses['angle_loss']:.4f} | "
                      f"θ: {losses['theta_loss']:.4f} | "
                      f"a/b: {losses['ab_loss']:.4f}")
            
            # 保存最佳模型
            if losses['total_loss'] < best_loss:
                best_loss = losses['total_loss']
                self.save(save_path.replace('.pt', '_best.pt'))
            
            # 定期保存
            if (epoch + 1) % save_freq == 0:
                self.save(save_path.replace('.pt', f'_epoch{epoch+1}.pt'))
        
        # 最终保存
        self.save(save_path)
        print(f"[BCTrainer] 训练完成，最佳损失: {best_loss:.4f}")
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"[BCTrainer] 模型已保存到 {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[BCTrainer] 模型已从 {path} 加载")


class BCAgent:
    """基于BC训练的智能体，用于推理和DAgger"""
    
    def __init__(self, model_path: str = None, hidden_dim: int = 256, device: str = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = BCPolicyNetwork(hidden_dim=hidden_dim).to(self.device)
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[BCAgent] 加载模型 {model_path}")
        
        self.model.eval()
    
    def decision(self, balls, my_targets, table) -> Dict[str, float]:
        """做出决策"""
        # 提取状态特征
        state_features = extract_state_features(balls, my_targets, table)
        
        # 转换为张量
        state_tensor = state_dict_to_tensor(state_features, self.device)
        
        # 推理
        with torch.no_grad():
            output = self.model(state_tensor)
            output = output.squeeze(0).cpu().numpy()
        
        # 转换为动作
        action = network_output_to_action(output)
        
        return action


def run_dagger(bc_trainer: BCTrainer, dataset: BCDataset, 
               n_iterations: int = 5, games_per_iter: int = 20):
    """
    运行DAgger迭代
    
    Args:
        bc_trainer: BC训练器
        dataset: 初始数据集
        n_iterations: DAgger迭代次数
        games_per_iter: 每次迭代收集的局数
    """
    from collect_data import collect_self_play_data
    
    print(f"[DAgger] 开始DAgger迭代，共 {n_iterations} 轮")
    
    for iteration in range(n_iterations):
        print(f"\n=== DAgger 迭代 {iteration + 1}/{n_iterations} ===")
        
        # 创建当前策略的Agent
        bc_agent = BCAgent()
        bc_agent.model = bc_trainer.model
        bc_agent.model.eval()
        
        # 用当前策略打球，收集新数据
        new_data = collect_self_play_data(bc_agent, n_games=games_per_iter)
        
        # 合并数据
        dataset.merge(new_data)
        print(f"[DAgger] 合并后数据量: {len(dataset)}")
        
        # 继续训练
        bc_trainer.train(dataset, n_epochs=50, batch_size=64,
                         save_path=f'train/bc_model_dagger{iteration+1}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BC训练')
    parser.add_argument('--data_path', type=str, default='train/checkpoints/expert_data.npz', help='数据路径')
    parser.add_argument('--n_epochs', type=int, default=1600, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=512, help='批次大小')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=768, help='隐藏层维度')
    parser.add_argument('--save_path', type=str, default='train/bc_model.pt', help='模型保存路径')
    parser.add_argument('--dagger', action='store_true', help='是否使用DAgger')
    parser.add_argument('--dagger_iters', type=int, default=5, help='DAgger迭代次数')
    
    args = parser.parse_args()
    
    # 加载数据
    dataset = BCDataset()
    if os.path.exists(args.data_path):
        dataset.load(args.data_path)
    else:
        print(f"数据文件 {args.data_path} 不存在，请先运行 collect_data.py")
        exit(1)
    
    # 创建训练器
    trainer = BCTrainer(hidden_dim=args.hidden_dim, lr=args.lr)
    
    # 训练
    trainer.train(
        dataset=dataset,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        save_path=args.save_path
    )
    
    # DAgger迭代
    if args.dagger:
        run_dagger(trainer, dataset, n_iterations=args.dagger_iters)
