# PPO 台球智能体训练

本项目使用 **PPO (Proximal Policy Optimization)** 算法训练台球智能体。

## 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

对于其他CUDA版本，请参考 [PyTorch官网](https://pytorch.org/get-started/locally/)。

### 依赖列表

- Python >= 3.10
- PyTorch >= 2.0 (支持CUDA)
- pooltool-billiards
- numpy
- bayesian-optimization

## 训练

### 基础训练

```bash
cd billiards
python train/train_ppo.py
```

### 自定义训练参数

```bash
python train/train_ppo.py \
    --timesteps 5000000 \
    --num_envs 16 \
    --lr 1e-4 \
    --gamma 0.99 \
    --device cuda
```

### 从检查点恢复训练

```bash
python train/train_ppo.py --checkpoint train/checkpoints/checkpoint_1000000.pt
```

### 主要超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--timesteps` | 2,000,000 | 总训练步数 |
| `--num_envs` | 8 | 并行环境数量 |
| `--lr` | 3e-4 | Actor学习率 |
| `--lr_critic` | 1e-3 | Critic学习率 |
| `--gamma` | 0.99 | 折扣因子 |
| `--clip` | 0.2 | PPO裁剪系数 |
| `--entropy` | 0.01 | 熵正则化系数 |
| `--log_freq` | 20,480 | 记录一次训练日志（10个迭代） |
| `--eval_freq` | 102,400 | 触发一次评估（50个迭代） |
| `--save_freq` | 204,800 | 保存检查点（100个迭代） |

> 以上频率均为 `update_freq=2048` 的整数倍，避免整除为0导致日志/评估/保存被跳过。如果调整 `update_freq`，请同步修改这些频率。

## 评估

### 评估训练好的模型

```bash
python train/train_ppo.py --eval_only --checkpoint train/checkpoints/final_model.pt --eval_episodes 40
```

### 与BasicAgent对战评估

```bash
python evaluate.py
```

## 文件结构

```
train/
├── config.py          # 训练配置和超参数
├── networks.py        # 神经网络架构 (Actor-Critic)
├── pool_rl_env.py     # 强化学习环境封装
├── ppo_trainer.py     # PPO训练器
├── train_ppo.py       # 训练主脚本
├── README.md          # 本文档
├── checkpoints/       # 模型检查点
└── logs/              # 训练日志
```

## 网络架构

### 状态编码器
- **球特征编码**: 使用自注意力机制处理16个球的特征
- **球袋特征编码**: 6个球袋位置的MLP编码
- **目标球编码**: 目标球的one-hot编码
- **游戏状态编码**: 剩余球数、是否瞄准8号球等

### Actor网络
- 输出连续动作的均值和标准差
- 动作维度: 5 (V0, phi, theta, a, b)
- 使用tanh激活函数限制动作范围

### Critic网络
- 输出状态价值估计
- 用于GAE优势估计

## 奖励设计

### 正向奖励
| 事件 | 奖励 |
|------|------|
| 打进己方目标球 | +100 |
| 打进当前瞄准的球 | +50 (额外) |
| 合法打进8号球 | +500 |
| 赢得比赛 | +200 |
| 连续击球权 | +20 |
| 好的走位 | +10 |

### 负向奖励
| 事件 | 惩罚 |
|------|------|
| 非法打进8号球 | -1000 |
| 白球和8号球同时进袋 | -1000 |
| 输掉比赛 | -200 |
| 白球进袋 | -150 |
| 未击中任何球 | -80 |
| 首球犯规 | -50 |
| 未碰库犯规 | -50 |
| 打进对方球 | -30 |
| 每步时间惩罚 | -1 |

## 训练技巧

1. **自我对弈**: 智能体与自己对弈，逐步提升水平
2. **课程学习**: 可以先禁用噪声训练，再启用噪声微调
3. **奖励塑形**: 通过精细的奖励设计引导智能体学习正确策略
4. **多环境并行**: 使用多个并行环境加速经验收集

## 训练监控

训练日志保存在 `train/logs/` 目录下，包含:
- 每次更新的损失值
- 平均奖励和胜率
- 回合长度统计

## 常见问题

### CUDA内存不足
减少 `--num_envs` 或 `mini_batch_size`

### 训练不收敛
- 降低学习率
- 增加熵系数以鼓励探索
- 检查奖励设计是否合理

### 模型表现不稳定
- 增加训练步数
- 使用更大的网络
- 调整GAE参数
