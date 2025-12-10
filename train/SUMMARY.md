# MuZero实现总结

## 实现概述

本项目为台球环境实现了完整的MuZero算法，包括：
1. ✅ 三个核心神经网络（表示、动力学、预测）
2. ✅ 蒙特卡洛树搜索（MCTS）
3. ✅ 经验回放缓冲区
4. ✅ 完整的训练循环
5. ✅ 集成到NewAgent中进行决策

## 文件结构

```
billiards/
├── agent.py                    # Agent定义（已修改NewAgent）
├── poolenv.py                  # 台球环境（不可修改）
├── evaluate.py                 # 评估脚本
├── utils.py                    # 工具函数
├── train/
│   ├── muzero_train.py        # MuZero完整实现（主要文件）
│   ├── train_simple.py        # 简化训练脚本
│   ├── config.py              # 配置文件
│   ├── test_muzero.py         # 测试脚本
│   ├── evaluate_example.py    # 评估示例
│   ├── README.md              # 训练说明
│   └── SUMMARY.md             # 本文件
└── checkpoints/               # 模型保存目录（训练时创建）
```

## 核心实现细节

### 1. 状态表示
- **输入特征**: 67维向量
  - 白球：位置(2) + 速度(2) = 4维
  - 15个球：每球4维（位置2 + 是否进袋1 + 是否目标球1）= 60维
  - 球桌信息：宽度 + 长度 = 2维
  - 其他：目标球数量 = 1维

### 2. 动作空间
- **离散化**: 将连续动作空间离散化为100个动作
  - V0（速度）: 10个档位 (0.5-8.0 m/s)
  - phi（角度）: 10个档位 (0-360度)
  - 简化: theta=45°, a=0, b=0（固定）

### 3. 奖励机制
根据图片评分标准设计的MuZero奖励：
- ✅ **+100,000分**：完胜对手
- ✅ **+1,000分/球**：打进己方球
- ✅ **+50分**：安全球
- ❌ **-4,000分**：致命犯规（白球进袋、非法黑8、失败）
- ❌ **-500分/球**：打进对方球
- ❌ **-500分**：首球犯规
- ❌ **-200分**：无进球无碰库

### 4. 网络架构

#### RepresentationNetwork (表示网络)
```
输入(67) -> FC(256) -> FC(256) -> FC(256) -> 隐藏状态
```

#### DynamicsNetwork (动力学网络)
```
[隐藏状态(256) + 动作(100)] -> FC(256) -> FC(256) 
    ├─> 下一状态(256)
    └─> 奖励预测(401维分类分布)
```

#### PredictionNetwork (预测网络)
```
隐藏状态(256) 
    ├─> FC(128) -> 策略(100)
    └─> FC(128) -> 价值(601维分类分布)
```

### 5. MCTS算法
- **UCB公式**: 平衡探索与利用
- **虚拟推演**: 使用网络进行状态转移模拟
- **根节点探索**: Dirichlet噪声增加探索
- **模拟次数**: 训练50次，评估30次

### 6. 训练流程
1. **自我对弈**: 每次迭代进行100局游戏
2. **经验存储**: 将游戏历史存入回放缓冲区
3. **批量训练**: 从缓冲区采样进行网络训练
4. **定期保存**: 每100次迭代保存检查点

## 使用方法

### 快速开始

#### 1. 环境配置
```bash
conda activate billiards
pip install torch numpy
```

#### 2. 验证安装
```bash
python train/test_muzero.py
```

#### 3. 开始训练
```bash
# 快速测试（10分钟）
python train/train_simple.py quick

# 小规模训练（1-2小时）
python train/train_simple.py small

# 正式训练（4-8小时）
python train/train_simple.py production
```

#### 4. 评估模型
修改`evaluate.py`中的agent初始化：
```python
agent_b = NewAgent(checkpoint_path="checkpoints/muzero_final.pth")
```

然后运行：
```bash
python evaluate.py
```

### 调试建议

#### 调试命令
```bash
# 1. 测试所有组件
python train/test_muzero.py

# 2. 快速训练测试
python train/train_simple.py quick

# 3. 单局游戏测试
python train/evaluate_example.py quick
```

#### 常见问题

**问题1: 导入错误**
```
解决: 确保在billiards目录下运行命令
cd d:\Downloads\billiards\billiards
```

**问题2: CUDA不可用**
```
解决: 代码会自动使用CPU，但训练较慢
建议: 使用quick配置进行测试
```

**问题3: 内存不足**
```
解决: 减小配置参数
- BATCH_SIZE: 64 -> 32
- REPLAY_BUFFER_SIZE: 10000 -> 2000
- NUM_SIMULATIONS: 50 -> 30
```

## 算法特点

### 优势
1. **无需环境模型**: MuZero学习隐式环境模型
2. **前瞻规划**: MCTS提供多步前瞻能力
3. **端到端训练**: 从原始观测直接学习策略
4. **自我提升**: 通过自我对弈不断改进

### 局限性
1. **训练时间长**: 需要大量自我对弈数据
2. **计算开销大**: MCTS模拟增加推理时间
3. **动作空间简化**: 当前仅离散化V0和phi
4. **超参数敏感**: 需要仔细调整配置

## 可能的改进方向

### 1. 动作空间扩展
- 增加theta、a、b的离散化
- 使用连续动作空间（Continuous MuZero）
- 分层动作选择

### 2. 网络架构优化
- 使用卷积网络处理空间信息
- 增加注意力机制
- 使用更深的网络

### 3. 训练策略改进
- 课程学习（从简单到复杂）
- 对抗训练（与强化对手）
- 迁移学习（预训练模型）

### 4. MCTS优化
- 并行化MCTS搜索
- 自适应模拟次数
- 动态探索参数

### 5. 奖励塑形
- 添加中间奖励（如击中目标球）
- 潜在价值引导
- 好奇心驱动探索

## 评估指标

### 训练监控
- 平均奖励
- 训练损失（价值、策略、奖励）
- 游戏步数

### 对战评估
- 对BasicAgent的胜率
- 平均击球数
- 进球效率

### 期望结果
- 快速测试: 随机表现
- 小规模训练: 40-50%胜率
- 正式训练: 60-70%胜率

## 技术栈

- **深度学习**: PyTorch
- **物理模拟**: pooltool
- **优化**: MCTS + SGD
- **语言**: Python 3.13

## 参考文献

1. Schrittwieser, J., et al. (2020). "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model." Nature.
2. Silver, D., et al. (2017). "Mastering the game of Go without human knowledge." Nature.
3. Browne, C., et al. (2012). "A Survey of Monte Carlo Tree Search Methods." IEEE Transactions.

## 项目贡献

本实现提供：
- ✅ 完整的MuZero算法实现
- ✅ 易于使用的训练脚本
- ✅ 详细的文档和注释
- ✅ 灵活的配置系统
- ✅ 完整的测试和评估工具

## 许可和致谢

- 基于AI3603课程台球大作业
- 使用pooltool物理引擎
- 参考DeepMind的MuZero论文
