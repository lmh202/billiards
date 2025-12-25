# 评估说明

## 文件结构

```
eval/
├── README.md        # 本文件
├── model.pt         # 训练好的模型权重（需要从train/checkpoints复制）
```

## 使用方法

### 1. 复制模型文件

训练完成后，将模型文件复制到此目录：

```bash
cp train/checkpoints/ppo_model.pt eval/model.pt
```

或者使用BC模型（如果PPO未收敛）：
```bash
cp train/checkpoints/bc_model_best.pt eval/model.pt
```

### 2. 运行评估

```bash
cd billiard
python evaluate.py
```

### 3. 预期结果

- 对BasicAgent胜率 > 88%
- 对BasicAgentPro胜率 > 50%（可选）

## 配置说明

NewAgent会自动从以下位置按顺序查找模型：
1. `eval/model.pt`
2. `eval/ppo_model.pt`
3. `train/checkpoints/ppo_model.pt`

## 注意事项

1. 确保PyTorch已安装
2. 模型推理使用GPU（如果可用），否则使用CPU
3. NewAgent的决策速度远快于BasicAgentPro
