"""
networks.py
Value Network 定义：一个简单的 MLP，输入状态特征，输出 V(s) ∈ (0, 1) 表示当前击球方的胜率。
"""
import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """
    Value Network: 预测当前局面下击球方的胜率。
    
    输入: 状态特征向量 (batch_size, feature_dim)
    输出: V(s) ∈ (0, 1) (batch_size, 1)
    """
    
    def __init__(self, feature_dim=55, hidden_dims=[128, 64, 32]):
        """
        Args:
            feature_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
        """
        super().__init__()
        
        layers = []
        in_dim = feature_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())  # 输出 (0, 1)
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: (batch_size, feature_dim) 状态特征
        Returns:
            (batch_size, 1) 胜率预测
        """
        return self.network(x)
    
    def predict(self, x):
        """
        推理时使用，不计算梯度
        Args:
            x: (batch_size, feature_dim) 或 (feature_dim,) 状态特征
        Returns:
            float 或 (batch_size,) 胜率预测
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return self.forward(x).squeeze(-1)


def create_value_network(feature_dim=55, hidden_dims=[128, 64, 32], device='cpu'):
    """工厂函数：创建并返回 Value Network"""
    model = ValueNetwork(feature_dim, hidden_dims)
    model.to(device)
    return model


def load_value_network(checkpoint_path, feature_dim=55, hidden_dims=[128, 64, 32], device='cpu'):
    """从检查点加载 Value Network"""
    model = create_value_network(feature_dim, hidden_dims, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


if __name__ == "__main__":
    # 简单测试
    import numpy as np
    model = create_value_network()
    
    # 随机输入测试
    dummy_input = torch.randn(4, 55)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.squeeze().tolist()}")
    
    # 单样本预测
    single_input = torch.randn(55)
    pred = model.predict(single_input)
    print(f"Single prediction: {pred.item():.4f}")
