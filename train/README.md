# MuZero for Billiards - 训练说明

## 环境配置

### 1. 基础环境
```bash
# 激活conda环境
conda activate b### 价值范围（根据新奖励机制调整）
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `value_support_min` | -10,000 | 价值函数最小值 |
| `value_support_max` | 110,000 | 价值函数最大值 |
| `reward_support_min` | -5,000 | 奖励最小值 |
| `reward_support_max` | 105,000 | 奖励最大值 |

**注意**：网络使用固定的601个bins来表示奖励和价值，避免网络过大。s

# 安装PyTorch（如果尚未安装）
pip install torch torchvision torchaudio

# 确保已安装项目依赖
pip install numpy bayesian-optimization
```

### 2. 验证环境
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import pooltool; print('Pooltool imported successfully')"
```

## 训练命令

### 方法1: 简化训练脚本（推荐）
```bash
# 在billiards目录下运行
cd d:\Downloads\billiards\billiards
conda activate billiards

# 快速测试（约10分钟）
python train/train_simple.py quick

# 小规模训练（约1-2小时）
python train/train_simple.py small

# 正式训练（约4-8小时）
python train/train_simple.py production

# 使用默认配置
python train/train_simple.py
```

### 方法2: 直接运行训练脚本
```bash
cd d:\Downloads\billiards\billiards
conda activate billiards
python train/muzero_train.py
```

### 训练输出
- 检查点将保存在 `checkpoints/` 目录
- 定期保存检查点：`muzero_step_<step>.pth`
- 训练完成后保存最终模型：`muzero_final.pth`
- 如果中断训练：保存为 `muzero_interrupted.pth`

## 超参数设置

所有超参数在 `train/muzero_train.py` 的 `MuZeroConfig` 类中定义：

### 核心超参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `action_space_size` | 100 | 离散化动作空间大小 (10×10: V0×phi) |
| `hidden_state_size` | 256 | 隐藏状态向量维度 |
| `num_simulations` | 50 | MCTS训练时的模拟次数 |
| `num_simulations_eval` | 30 | MCTS评估时的模拟次数 |
| `max_moves` | 60 | 每局游戏最大步数 |

### MCTS参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pb_c_base` | 19652 | UCB公式中的基础常数 |
| `pb_c_init` | 1.25 | UCB公式中的初始化常数 |
| `root_dirichlet_alpha` | 0.3 | 根节点Dirichlet噪声参数 |
| `root_exploration_fraction` | 0.25 | 根节点探索比例 |

### 训练参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `training_steps` | 10000 | 总训练步数 |
| `batch_size` | 64 | 批大小 |
| `num_unroll_steps` | 5 | 展开步数 |
| `td_steps` | 10 | TD(n)步数 |
| `lr_init` | 0.001 | 初始学习率 |
| `weight_decay` | 1e-4 | 权重衰减 |

### 经验回放参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `replay_buffer_size` | 10000 | 缓冲区大小 |
| `self_play_games` | 100 | 每次迭代的自我对弈局数 |
| `priority_alpha` | 0.6 | 优先级采样指数 |
| `priority_beta` | 0.4 | 重要性采样指数 |

### 奖励/价值范围
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `value_support_min` | -300 | 价值函数最小值 |
| `value_support_max` | 300 | 价值函数最大值 |
| `reward_support_min` | -200 | 奖励最小值 |
| `reward_support_max` | 200 | 奖励最大值 |

## 奖励机制说明

MuZero的奖励机制按照图片中的评分标准设计：

### 1. 致命犯规（缺省-1分）
- ❌ **-4,000分**：白球进袋
- ❌ **-4,000分**：非法打进黑8
- ❌ **-4,000分**：导致游戏失败的犯规

### 2. 胜利奖励
- ✅ **+100,000分**：完胜对手（清空己方球后合法打进黑8）

### 3. 进攻与推进得分
- ✅ **+1,000分/球**：打进己方目标球
- ❌ **-500分/球**：打进对方球（破坏）

### 4. 击球犯规（扣分）
- ❌ **-500分**：首球犯规（没打到己方球）
- ❌ **-200分**：本届犯规（无进球无碰库）

### 5. 安全球奖励
- ✅ **+50分**：合法无进球（既未犯规也未进球）

## 模型架构

### 网络组件
1. **RepresentationNetwork**: 将原始观测编码为隐藏状态
   - 输入：67维特征向量（白球4维 + 15球×4维 + 球桌2维 + 其他1维）
   - 输出：256维隐藏状态

2. **DynamicsNetwork**: 预测状态转移和奖励
   - 输入：隐藏状态(256) + 动作one-hot(100)
   - 输出：下一个隐藏状态(256) + 奖励分布(401)

3. **PredictionNetwork**: 预测策略和价值
   - 输入：隐藏状态(256)
   - 输出：策略logits(100) + 价值分布(601)

### 状态编码
观测状态包含以下信息：
- 白球位置 (x, y) 和速度 (vx, vy)
- 15个球的位置、是否进袋、是否为目标球
- 球桌尺寸
- 当前目标球数量

## 使用训练好的模型

在 `agent.py` 中使用NewAgent：

```python
# 初始化Agent（提供检查点路径）
agent = NewAgent(checkpoint_path="checkpoints/muzero_final.pth")

# 进行决策
action = agent.decision(balls, my_targets, table)
```

## 调试建议

### 1. 减少训练规模（快速测试）
修改 `MuZeroConfig`：
```python
self.training_steps = 1000  # 减少训练步数
self.self_play_games = 10   # 减少每次迭代的游戏数
self.num_simulations = 20   # 减少MCTS模拟次数
```

### 2. 监控训练进度
训练过程会输出：
- 每局游戏的完成情况
- 训练损失
- 检查点保存信息

### 3. 测试模型
使用 `evaluate.py` 测试训练好的模型：
```python
# 在evaluate.py中修改agent初始化
agent_a = NewAgent(checkpoint_path="checkpoints/muzero_final.pth")
agent_b = BasicAgent()  # 或其他对手
```

## 训练时间估计

- CPU训练：约2-4小时（10000步，取决于CPU性能）
- GPU训练：约30分钟-1小时（推荐使用GPU）

## 常见问题

### 1. CUDA不可用
如果没有GPU，代码会自动使用CPU训练，但速度较慢。

### 2. 内存不足
减少以下参数：
- `replay_buffer_size`
- `batch_size`
- `self_play_games`

### 3. 导入错误
确保在项目根目录运行，并且Python能找到相关模块。

## 文件结构

```
billiards/
├── train/
│   ├── muzero_train.py    # MuZero训练脚本（主要文件）
│   └── README.md          # 本文件
├── checkpoints/           # 训练检查点保存目录（自动创建）
├── agent.py               # Agent定义（包含NewAgent）
├── poolenv.py            # 台球环境
├── evaluate.py           # 评估脚本
└── utils.py              # 工具函数
```

## 参考资料

- MuZero论文: "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"
- 项目文档: PROJECT_GUIDE.md
- 游戏规则: GAME_RULES.md
