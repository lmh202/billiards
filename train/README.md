# Value Network 训练指南

本目录包含用于训练 Value Network V(s) 的完整流程，实现方案 C：只训练 Value（不学动作），用它指导搜索。

## 文件结构

```
train/
├── state_utils.py      # 状态特征提取模块
├── networks.py         # Value Network 定义 (MLP)
├── collect_data.py     # 对局数据收集脚本
├── train_value.py      # Value Network 训练脚本
├── data/               # 收集的数据存放目录
└── checkpoints/        # 模型检查点存放目录

agents/
└── value_agent.py      # 集成 V(s) 的 Agent
```

## 评分公式

```
Score(a) = r_me(a) - λ·r_opp(a) + γ·V̂(s')
```

其中：
- `s'` 是模拟我方一杆后的新局面
- `V̂(s') = V(s')` 如果仍轮到我击球
- `V̂(s') = 1 - V(s')` 如果轮到对手击球（对手视角）
- `γ` 一开始设小一点 (默认 0.3)，避免 V 没训好就"喧宾夺主"

## 使用流程

### 1. 收集数据

```powershell
conda activate billiards
python train/collect_data.py --num_games 100 --output_dir train/data
```

参数说明：
- `--num_games`: 收集的游戏局数
- `--pro_ratio`: vs BasicAgentPro 的比例 (默认 0.3)
- `--basic_ratio`: vs BasicAgent 的比例 (默认 0.5)
- `--self_ratio`: self-play 的比例 (默认 0.2)

采样策略（参考方案 C）:
1. NewAgent vs BasicAgentPro (最贴近目标分布)
2. NewAgent vs BasicAgent (快速产生多样局面)
3. NewAgent self-play (产生僵持/安全球局面)

### 2. 训练 Value Network

```powershell
python train/train_value.py --data_files train/data/value_data_*.npz --epochs 50
```

参数说明：
- `--data_files`: 训练数据文件路径（支持通配符）
- `--epochs`: 训练轮数 (默认 50)
- `--batch_size`: 批大小 (默认 64)
- `--lr`: 学习率 (默认 1e-3)
- `--hidden_dims`: 隐藏层维度 (默认 [128, 64, 32])
- `--no_sample_weights`: 不使用样本权重（默认使用，让残局样本权重更大）
- `--resume`: 继续训练的检查点路径

训练损失：Binary Cross Entropy (BCE)
评估指标：Accuracy, AUC, Brier Score

### 3. 使用训练好的 Value Network

```python
from agents.value_agent import ValueAgent

# 加载训练好的模型
agent = ValueAgent(value_net_path="train/checkpoints/value_net_best_xxx.pt")

# 使用 agent 进行决策
action = agent.decision(balls, my_targets, table)
```

### 4. 迭代式数据聚合（推荐）

最推荐的训练方式是 2~3 轮迭代：

1. 先收集一批数据（比如 5k~20k 个 state 样本）训练出 V0
2. 把 V0 接入 ValueAgent 搜索（Score 里加 γV(s')）跑更多对局
3. 把新对局产生的 state 再加进数据集训练 V1
4. 重复一次

这样 V(s) 会快速贴近"你实际会遇到的局面分布"，不会出现"离线训得很好，线上一用就飘"。

```powershell
# 第一轮：收集初始数据
python train/collect_data.py --num_games 200

# 第一轮：训练 V0
python train/train_value.py --data_files train/data/value_data_round1.npz --epochs 50

# 第二轮：使用 V0 收集更多数据（需要修改 collect_data.py 使用 ValueAgent）
# ...

# 第二轮：训练 V1
python train/train_value.py --data_files train/data/value_data_round1.npz train/data/value_data_round2.npz --epochs 50
```

## 特征设计

状态特征共 55 维：
- 母球位置 (2) + 是否在袋 (1)
- 我方目标球位置 + 是否在袋 (7 × 3 = 21)
- 对方目标球位置 + 是否在袋 (7 × 3 = 21)
- 黑 8 位置 + 是否在袋 (3)
- 我方剩余球数 (1)
- 对方剩余球数 (1)
- 是否可打黑 8 (1)
- 母球到袋口最小距离 (1)
- 母球到最近目标球距离 (1)
- 当前击球方 one-hot (2)

## 样本权重

为了让训练更关注关键局面（残局），使用样本权重：
- 剩余球越少，权重越大
- 例如：剩余 7 球 → 权重 1.0；剩余 0 球（打黑 8）→ 权重 5.0

## 调参建议

- `GAMMA_VALUE`: 初始设为 0.2~0.3，随着 V(s) 准确度提升可以逐渐增大到 0.5
- `hidden_dims`: 默认 [128, 64, 32]，可以尝试更大的网络 [256, 128, 64]
- `epochs`: 建议至少 50 轮，观察验证损失收敛情况

## 注意事项

1. 确保在 billiards conda 环境中运行
2. 数据收集较慢（每局约 10-30 秒），建议一次收集较多局数
3. V(s) 需要足够多样的数据才能泛化，建议至少收集 100 局以上
