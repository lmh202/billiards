# 台球智能体训练指南

## 算法概述

本项目采用 **模仿学习（BC/DAgger） + 自博弈 PPO 微调（带对手混合）** 的训练流程：

1. **行为克隆 (BC)**: 让 BasicAgentPro 自己打多局，收集 `(obs -> action)` 数据，训练策略网络拟合它的输出
2. **DAgger**: 用当前策略打球，遇到的局面让 BasicAgentPro 标注"它会怎么打"，解决分布偏移问题
3. **PPO 自博弈微调**: 初期对手混合 BasicAgent / BasicAgentPro / 历史自己，逐步提高对 BasicAgentPro 的对战比例

## 环境配置

```bash
# 激活环境
conda activate poolenv

# 确保已安装以下依赖
pip install torch numpy tqdm
pip install bayesian-optimization scikit-learn  # BasicAgent 需要
```

## 训练命令

### 完整训练流程（推荐）

```bash
cd billiard
python train/train.py --mode full
```

### 参数说明

```bash
python train/train.py --mode full \
    --n_expert_games 200 \    # 专家数据收集局数
    --bc_epochs 200 \         # BC训练轮数
    --dagger_iters 3 \        # DAgger迭代次数
    --ppo_games 2000 \        # PPO训练局数
    --hidden_dim 256 \        # 网络隐藏层维度
    --output_dir train/checkpoints
```

### 分阶段训练

1. **仅收集数据和BC训练**:
```bash
python train/train.py --mode bc_only --n_expert_games 100 --bc_epochs 100
```

2. **仅PPO微调**（需要先有BC模型）:
```bash
python train/train.py --mode ppo_only --ppo_games 1000
```

### 单独运行各阶段

```bash
# 1. 收集专家数据
python train/collect_data.py --n_games 200 --save_path train/expert_data.npz

# 2. BC训练
python train/train_bc.py --data_path train/expert_data.npz --n_epochs 200

# 3. BC + DAgger
python train/train_bc.py --data_path train/expert_data.npz --dagger --dagger_iters 5

# 4. PPO微调
python train/train_ppo.py --bc_model train/bc_model_best.pt --n_games 2000
```

## 超参数设置

### BC阶段
- 学习率: 1e-3
- 批次大小: 64
- 训练轮数: 200
- 角度表示: 使用 sin(phi), cos(phi) 处理周期性

### DAgger阶段
- 迭代次数: 3-5
- 每次迭代收集: 20-30 局

### PPO阶段
- 学习率: 3e-4
- 批次大小: 64
- Gamma: 0.99
- GAE Lambda: 0.95
- Clip Epsilon: 0.2
- 更新频率: 每10局更新一次
- 对手混合比例:
  - BasicAgent: 30%
  - BasicAgentPro: 50%
  - 历史自己: 20%

### 奖励设置
- 胜利: +10
- 失败: -10
- 平局: 0
- 进自己的球: +0.5
- 帮对手进球: -0.3
- 白球进袋: -1.0
- 其他犯规: -0.5

## 输出文件

训练完成后，在 `train/checkpoints/` 目录下会生成：

- `expert_data.npz`: 专家数据
- `bc_model.pt`: BC模型
- `bc_model_best.pt`: 最佳BC模型
- `bc_model_dagger{N}.pt`: DAgger迭代后的模型
- `ppo_model.pt`: 最终PPO模型
- `ppo_model_game{N}.pt`: 训练过程中的检查点

## 部署到eval

训练完成后：

1. 将最终模型复制到 `eval/` 目录:
```bash
cp train/checkpoints/ppo_model.pt eval/model.pt
```

2. `agents/new_agent.py` 已配置好加载模型，直接运行评估即可:
```bash
python evaluate.py
```

## 预期效果

- BC训练后：对 BasicAgent 胜率约 60-70%
- DAgger后：对 BasicAgent 胜率约 75-85%
- PPO微调后：对 BasicAgent 胜率 > 88%

## 注意事项

1. **推理效率**: NewAgent 使用神经网络直接推理，每一杆决策时间远小于 BasicAgentPro 的 MCTS 搜索
2. **角度周期性**: 使用 sin(phi), cos(phi) 表示角度，避免在 0/360 附近学崩
3. **对手混合**: PPO阶段逐步增加强对手比例，避免过早过拟合弱对手
