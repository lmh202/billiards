# MuZero实现 - 变更日志

## 2025-12-10 - 奖励机制重大更新

### 背景
根据提供的评分标准图片，重新设计了MuZero的奖励机制，使其更符合游戏目标。

### 主要变更

#### 1. 奖励函数重写
**文件**: `train/muzero_train.py`

**新增函数**: `calculate_muzero_reward(result, env, player)`

**奖励标准**:
- ✅ **+100,000**: 完胜对手
- ✅ **+1,000/球**: 打进己方球
- ✅ **+50**: 安全球
- ❌ **-4,000**: 致命犯规（白球进袋、非法黑8、失败）
- ❌ **-500/球**: 打进对方球
- ❌ **-500**: 首球犯规
- ❌ **-200**: 无进球无碰库

**替换了**:
- 原来使用 `analyze_shot_for_reward()` 函数
- 现在使用 `calculate_muzero_reward()` 函数

#### 2. 配置更新
**文件**: `train/config.py`, `train/muzero_train.py`

**奖励/价值范围**:
```python
# 旧值
VALUE_SUPPORT_MIN = -300
VALUE_SUPPORT_MAX = 300
REWARD_SUPPORT_MIN = -200
REWARD_SUPPORT_MAX = 200

# 新值
VALUE_SUPPORT_MIN = -10000
VALUE_SUPPORT_MAX = 110000
REWARD_SUPPORT_MIN = -5000
REWARD_SUPPORT_MAX = 105000
```

#### 3. 网络优化
**文件**: `train/muzero_train.py`

**Bins数量固定化**:
```python
# DynamicsNetwork
self.reward_bins = 601  # 固定，不是 110001

# PredictionNetwork
self.value_bins = 601   # 固定，不是 120001
```

**原因**: 避免网络输出层过大，保持训练效率。

#### 4. 文档更新
更新了以下文档以反映新的奖励机制：
- ✅ `train/README.md` - 训练说明
- ✅ `train/SUMMARY.md` - 实现总结
- ✅ `MUZERO_QUICKSTART.md` - 快速指南
- ✅ `train/REWARD_UPDATE.md` - 奖励更新详细说明（新增）

#### 5. 测试工具
**文件**: `train/test_reward.py`（新增）

测试各种情况下的奖励计算，包括：
- 完胜/失败
- 进球（己方/对方）
- 各类犯规
- 安全球
- 复合情况

### 设计理念对比

#### 旧设计（analyze_shot_for_reward）
- 聚焦单次击球表现
- 奖励范围较小（-200 ~ +200）
- 主要用于评估击球质量

#### 新设计（calculate_muzero_reward）
- 聚焦游戏整体目标
- 奖励范围较大（-5000 ~ +105000）
- 明确区分胜利、进攻、防守的重要性

### 关键差异

| 情况 | 旧奖励 | 新奖励 | 倍数 |
|------|--------|--------|------|
| 完胜 | +100 | +100,000 | x1000 |
| 己方进球 | +50 | +1,000 | x20 |
| 对方进球 | -20 | -500 | x25 |
| 白球进袋 | -100 | -4,000 | x40 |
| 首球犯规 | -30 | -500 | x16.7 |
| 安全球 | +10 | +50 | x5 |

### 预期影响

#### 积极影响
1. **更强的目标导向**: AI明确知道赢得游戏是首要目标
2. **更积极的进攻**: 进球奖励大幅提升
3. **更少的犯规**: 犯规惩罚显著增加
4. **更合理的策略**: 区分进攻和防守时机

#### 潜在挑战
1. **训练时间**: 可能需要更多训练迭代
2. **奖励稀疏性**: 胜利奖励很大但不频繁
3. **价值估计**: 价值网络需要适应新范围
4. **超参数调整**: 学习率等可能需要调整

### 兼容性

#### 向后兼容
- ❌ 旧模型无法直接使用（奖励范围不同）
- ❌ 需要重新训练

#### 环境兼容
- ✅ 完全兼容现有poolenv
- ✅ 不需要修改环境代码
- ✅ analyze_shot_for_reward仍可用于其他agent

### 使用建议

#### 训练建议
1. 从小规模配置开始测试
2. 监控训练曲线，特别是价值预测
3. 可能需要调整学习率
4. 考虑使用学习率衰减

#### 评估建议
1. 对比新旧奖励下的表现
2. 记录犯规率、进球率等指标
3. 观察策略变化（进攻vs防守）

### 测试方法

#### 单元测试
```bash
python train/test_reward.py
```

#### 集成测试
```bash
python train/test_muzero.py
```

#### 快速训练测试
```bash
python train/train_simple.py quick
```

### 回滚方案

如果新奖励机制效果不佳，可以回滚：

1. 恢复 `calculate_muzero_reward` 使用旧逻辑
2. 恢复配置文件中的范围
3. 恢复网络的bins计算方式

保留了原来的 `analyze_shot_for_reward` 函数作为参考。

### 后续优化方向

1. **奖励平滑**: 考虑缩放所有奖励值
2. **奖励塑形**: 添加更多中间奖励
3. **课程学习**: 逐步增加难度
4. **对抗训练**: 与不同水平对手训练

### 相关文件

- `train/muzero_train.py` - 核心实现
- `train/config.py` - 配置更新
- `train/REWARD_UPDATE.md` - 详细说明
- `train/test_reward.py` - 测试脚本
- `train/README.md` - 使用文档

### 贡献者
- AI Assistant - 实现和文档

### 参考
- 评分标准图片
- MuZero论文奖励设计章节
- 台球规则标准
