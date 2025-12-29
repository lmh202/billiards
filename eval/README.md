# 2-Ply 对手前瞻规划代理 (NewAgent) - 测试说明

## 环境配置

```bash
# 激活 conda 环境
conda activate billiards

# 确保已按照 PROJECT_GUIDE.md 安装所有依赖
pip install bayesian-optimization numpy
```

## 测试命令

```bash
# 在项目根目录下运行
cd d:\Downloads\billiards\billiard
python evaluate.py
```

## 方法说明

本方法采用 **2-Ply 对手前瞻规划**，核心思想是在选择动作时不仅考虑自己的即时收益，还要预测对手的响应并最小化对手的收益。

### 算法流程

1. **候选动作生成** (K=48 个候选):
   - 50% 来自启发式几何瞄准（Ghost-Ball 方法）
   - 50% 来自最佳动作的局部扰动

2. **两阶段评估**:
   - **第一阶段（快速筛选）**: 对所有 K 个候选动作进行物理仿真，只计算 R_me 和 Safety
   - **第二阶段（深度评估）**: 对 top-8 候选动作调用 BasicAgentPro 模拟对手响应，计算 R_opp

3. **综合评分选择最优动作**:
   
   $$Score = R_{me} - \lambda R_{opp} + \beta \cdot Safety(s')$$

### 超参数设置

| 参数 | 值 | 说明 |
|------|-----|------|
| K (K_CANDIDATES) | 48 | 候选动作数量 |
| N (TOP_N_FOR_OPPONENT_SIM) | 8 | 进入第二阶段的候选数 |
| λ (LAMBDA_OPP) | 0.8 | 对手得分惩罚系数 (0.7~1.2，越大越偏防守) |
| β (BETA_SAFETY) | 0.3 | 安全性评估权重 |
| V0_PERTURB | ±0.5 m/s | 速度扰动范围 |
| PHI_PERTURB | ±5° | 角度扰动范围 |
| SPIN_PERTURB | ±0.15 | 旋转参数扰动范围 |

### Ghost-Ball 几何瞄准原理

Ghost-Ball 是一种经典的台球瞄准方法：

1. 计算目标球到袋口的方向向量
2. 在目标球后方（沿反方向）放置一个"幽灵球"，距离为两球直径
3. 白球瞄准幽灵球位置击打，即可将目标球送入袋口

```
[白球] ---> [幽灵球位置] ---> [目标球] ---> [袋口]
```

### 安全性指标 Safety(s')

安全性评估考虑以下因素:
1. **白球远离袋口的程度** (最高 1.0 分): 防止自己送球
2. **白球在球台中央的程度** (最高 0.5 分): 中央位置有更多选择
3. **白球到我方目标球的可达性** (最高 1.0 分): 保持进攻选择
4. **对手目标球的不利程度** (最高 0.5 分): 增加防守性

## 文件结构

```
eval/
├── README.md          # 本文件
agents/
├── __init__.py        # 模块导出
├── agent.py           # Agent 基类
├── new_agent.py       # NewAgent 实现 (2-Ply 对手前瞻)
├── basic_agent.py     # BasicAgent 对手
└── basic_agent_pro.py # BasicAgentPro 对手
```

## 注意事项

1. NewAgent 会在首次调用对手响应时延迟加载 BasicAgentPro
2. 物理仿真会添加小量噪声以模拟实际执行误差
3. 两阶段评估策略显著提升了性能（只对 top-8 调用对手模拟）

## 预期效果

- 对 BasicAgent 胜率: > 88%
- 对 BasicAgentPro 胜率: 有竞争力（具体取决于参数调优）

## 调参建议

如果希望提高胜率，可以尝试调整以下参数:

| 目标 | 调整方法 |
|------|----------|
| 提高进球率 | 增大 `K_CANDIDATES` (如 64) |
| 更注重防守 | 增大 `LAMBDA_OPP` (如 1.0~1.2) |
| 更安全的走位 | 增大 `BETA_SAFETY` (如 0.4~0.5) |
| 更精确的搜索 | 增大 `TOP_N_FOR_OPPONENT_SIM` (如 12) |
| 减少扰动 | 减小 `V0_PERTURB`, `PHI_PERTURB` |
