# MuZero实现快速开始指南

## ⚡ 快速开始

### 1️⃣ 激活环境
```bash
conda activate billiards
```

### 2️⃣ 安装依赖
```bash
pip install torch numpy
```

### 3️⃣ 验证安装（可选）
```bash
python train/test_muzero.py
```

### 4️⃣ 开始训练
```bash
# 快速测试（约10分钟）
python train/train_simple.py quick

# 小规模训练（约1-2小时）
python train/train_simple.py small

# 正式训练（约4-8小时）
python train/train_simple.py production
```

### 5️⃣ 评估模型
修改 `evaluate.py` 第33行：
```python
agent_a, agent_b = BasicAgent(), NewAgent(checkpoint_path="checkpoints/muzero_final.pth")
```

运行评估：
```bash
python evaluate.py
```

## 📁 重要文件

| 文件 | 说明 |
|------|------|
| `train/muzero_train.py` | MuZero完整实现 |
| `train/train_simple.py` | 简化训练脚本（推荐使用）|
| `train/README.md` | 详细训练文档 |
| `train/SUMMARY.md` | 实现总结 |
| `agent.py` | Agent定义（已实现NewAgent）|

## 🎯 实现特点

- ✅ 完整的MuZero算法（表示、动力学、预测网络）
- ✅ MCTS搜索算法
- ✅ 经验回放缓冲区
- ✅ 按照图示奖励机制（不修改analyze_shot_for_reward）
- ✅ 易于使用的训练脚本
- ✅ 灵活的配置系统

## 📊 奖励机制

根据图片评分标准设计：

**胜利奖励**
- ✅ **+100,000分** - 完胜对手

**进攻得分**
- ✅ **+1,000分/球** - 打进己方目标球
- ✅ **+50分** - 安全球（合法无进球）

**惩罚**
- ❌ **-4,000分** - 致命犯规（白球进袋、非法黑8、失败）
- ❌ **-500分/球** - 打进对方球
- ❌ **-500分** - 首球犯规
- ❌ **-200分** - 无进球无碰库

## 🔧 调试建议

### 如果训练很慢
```bash
# 使用快速配置
python train/train_simple.py quick
```

### 如果内存不足
修改 `train/config.py` 中的配置：
- 减小 `BATCH_SIZE`
- 减小 `REPLAY_BUFFER_SIZE`
- 减小 `NUM_SIMULATIONS_TRAIN`

### 测试单局游戏
```bash
python train/evaluate_example.py quick
```

## 📖 详细文档

- **训练说明**: `train/README.md`
- **实现总结**: `train/SUMMARY.md`
- **项目指南**: `PROJECT_GUIDE.md`
- **游戏规则**: `GAME_RULES.md`

## ⚠️ 注意事项

1. **不要修改** `analyze_shot_for_reward` 函数
2. **确保在正确目录**: `d:\Downloads\billiards\billiards`
3. **训练前先测试**: 运行 `test_muzero.py` 验证环境
4. **保存检查点**: 训练会自动保存到 `checkpoints/` 目录

## 🆘 遇到问题？

1. 查看 `train/README.md` 的常见问题部分
2. 运行 `python train/test_muzero.py` 检查环境
3. 检查是否在正确的conda环境中

## 🎓 算法说明

MuZero是DeepMind开发的强化学习算法，特点：
- 学习环境的隐式模型
- 使用MCTS进行前瞻规划
- 通过自我对弈不断改进
- 无需了解游戏规则

更多细节请参考 `train/SUMMARY.md`

---

**祝训练顺利！🚀**
