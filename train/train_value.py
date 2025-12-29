"""
train_value.py
Value Network 训练脚本：使用 BCE 损失训练 V(s) 预测当前击球方的胜率。
支持迭代式数据聚合（多轮收集 + 训练）。
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.networks import create_value_network, load_value_network
from train.state_utils import get_feature_dim


class ValueDataset(Dataset):
    """Value Network 训练数据集"""
    
    def __init__(self, features, labels, weights=None):
        """
        Args:
            features: (N, feature_dim) 状态特征
            labels: (N,) 胜负标签 (0 或 1)
            weights: (N,) 样本权重（可选，用于残局加权）
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)
        if weights is not None:
            self.weights = torch.FloatTensor(weights)
        else:
            self.weights = torch.ones(len(labels))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.weights[idx]


def compute_sample_weights(features, labels, remaining_ball_idx=48):
    """
    计算样本权重：让残局样本（剩余球少）权重更大。
    
    Args:
        features: (N, feature_dim) 状态特征
        labels: (N,) 胜负标签
        remaining_ball_idx: 特征中 "我方剩余球数" 的索引（归一化后的值）
    
    Returns:
        weights: (N,) 样本权重
    """
    # 我方剩余球数在特征中的位置（归一化为 0-1）
    own_remaining = features[:, remaining_ball_idx]  # 归一化值
    
    # 权重公式：剩余球越少，权重越大
    # 例如：剩余 7 球 (own_remaining=1.0) -> 权重 1.0
    #       剩余 1 球 (own_remaining≈0.14) -> 权重 3.0
    #       剩余 0 球 (own_remaining=0.0) -> 权重 5.0 (打黑 8)
    weights = 1.0 + 4.0 * (1.0 - own_remaining)
    
    return weights


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for features, labels, weights in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        
        # 加权 BCE 损失
        loss = criterion(outputs, labels)
        loss = (loss * weights.unsqueeze(1)).mean()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        total_samples += len(labels)
    
    return total_loss / total_samples


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels, weights in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels).mean()
            
            total_loss += loss.item() * len(labels)
            
            preds = (outputs > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)
            
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    accuracy = total_correct / total_samples
    avg_loss = total_loss / total_samples
    
    # 计算 AUC（可选）
    try:
        from sklearn.metrics import roc_auc_score, brier_score_loss
        auc = roc_auc_score(all_labels, all_preds)
        brier = brier_score_loss(all_labels, all_preds)
    except:
        auc = 0.0
        brier = 0.0
    
    return avg_loss, accuracy, auc, brier


def train_value_network(
    data_files,
    output_dir='train/checkpoints',
    epochs=50,
    batch_size=64,
    learning_rate=1e-3,
    hidden_dims=[128, 64, 32],
    use_sample_weights=True,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    resume_from=None
):
    """
    训练 Value Network。
    
    Args:
        data_files: list of str, 数据文件路径列表（.npz）
        output_dir: 模型保存目录
        epochs: 训练轮数
        batch_size: 批大小
        learning_rate: 学习率
        hidden_dims: 隐藏层维度
        use_sample_weights: 是否使用样本权重
        device: 训练设备
        resume_from: 继续训练的检查点路径
    
    Returns:
        model: 训练好的模型
        best_checkpoint: 最佳检查点路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载所有数据
    all_features = []
    all_labels = []
    for filepath in data_files:
        data = np.load(filepath)
        all_features.append(data['features'])
        all_labels.append(data['labels'])
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(f"总样本数: {len(labels)}")
    print(f"正样本比例: {labels.mean():.2%}")
    print(f"特征维度: {features.shape[1]}")
    
    # 计算样本权重
    if use_sample_weights:
        weights = compute_sample_weights(features, labels)
        print(f"样本权重范围: [{weights.min():.2f}, {weights.max():.2f}]")
    else:
        weights = None
    
    # 创建数据集
    dataset = ValueDataset(features, labels, weights)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    feature_dim = get_feature_dim()
    if resume_from:
        model = load_value_network(resume_from, feature_dim, hidden_dims, device)
        print(f"从 {resume_from} 恢复模型")
    else:
        model = create_value_network(feature_dim, hidden_dims, device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"训练设备: {device}")
    
    # 损失函数和优化器
    criterion = nn.BCELoss(reduction='none')  # 使用 none 以便加权
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 训练循环
    best_val_loss = float('inf')
    best_checkpoint = None
    
    print("\n开始训练...")
    print("-" * 60)
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, val_brier = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2%} | "
              f"Val AUC: {val_auc:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_checkpoint = os.path.join(output_dir, f"value_net_best_{timestamp}.pt")
            torch.save(model.state_dict(), best_checkpoint)
            print(f"  -> 保存最佳模型到 {best_checkpoint}")
    
    print("-" * 60)
    print(f"训练完成！最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳模型: {best_checkpoint}")
    
    return model, best_checkpoint


def main():
    parser = argparse.ArgumentParser(description='训练 Value Network')
    parser.add_argument('--data_files', type=str, nargs='+', required=True, help='训练数据文件路径')
    parser.add_argument('--output_dir', type=str, default='train/checkpoints', help='模型保存目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64, 32], help='隐藏层维度')
    parser.add_argument('--no_sample_weights', action='store_true', help='不使用样本权重')
    parser.add_argument('--resume', type=str, default=None, help='继续训练的检查点')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Value Network 训练")
    print("=" * 60)
    
    train_value_network(
        data_files=args.data_files,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dims=args.hidden_dims,
        use_sample_weights=not args.no_sample_weights,
        device=args.device,
        resume_from=args.resume
    )


if __name__ == "__main__":
    main()
