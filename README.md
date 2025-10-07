# 🧠 MAS-RL-Tutorial | 多智能体强化学习实战教程

> 🚀 一步步动手实践，从 Gymnasium 到 PettingZoo，带你走进多智能体强化学习的世界。

---

## 🌍 项目简介

这个仓库是我学习 **强化学习（Reinforcement Learning, RL）** 和 **多智能体强化学习（Multi-Agent Reinforcement Learning, MARL）** 的一些案例。

内容以 **[Gymnasium](https://gymnasium.farama.org/)** 为起点，逐步深入到 **[PettingZoo](https://pettingzoo.farama.org/)** 多智能体环境。
目标是从「能跑起来」到「能理解」，再到「能扩展」🎯。

📚 本项目适合你如果你想：

* 了解 RL 和 MARL 的基础环境搭建；
* 实践经典环境（如 Blackjack, GridWorld, CarRacing）；
* 学习如何构建多智能体环境；
* 探索训练日志、可视化、策略评估等环节。

---

## 🧩 仓库结构与内容导航

```
MAS-RL-Tutorial/
├── env_programe/                     # 环境脚本与配置文件
├── logs/                             # 训练结果与实验日志
├── videos/gymnasium_env/GridWorld/   # 可视化视频与测试片段
│
├── blackjack.py                      # Gymnasium Blackjack 实战案例
├── carRacing_agent_train.py          # CarRacing 智能体训练脚本
├── carRacing_agent_test.py           # CarRacing 智能体测试脚本
├── gridworld_env.py                  # 自定义 GridWorld 环境构建
├── gridworld_agent.py                # GridWorld 智能体训练逻辑
│
├── gymnasium_begin.py                # Gymnasium 框架入门示例
├── pettingzoo_begin.py               # PettingZoo 多智能体入门案例（即将更新）
├── test.py                           # 测试与实验验证脚本
└── ...
```

---

## ⚙️ 环境配置

推荐使用 Python 3.10+ 并创建虚拟环境👇：

```bash
conda create -n marl python=3.10
conda activate marl
```

安装依赖（视情况添加 gymnasium 的扩展包）：

```bash
pip install gymnasium
pip install pettingzoo
pip install numpy
pip install matplotlib
pip install stable-baselines3 
```


## 🧪 快速上手

### 1️⃣ 运行第一个 Gymnasium 环境

暂空


### 2️⃣ 自定义环境：GridWorld

暂空

### 3️⃣ 玩转 CarRacing 环境

暂空



### 4️⃣ Blackjack：经典案例入门

暂空

## 🧠 进阶计划（Roadmap）

| 阶段 | 内容                         | 状态  |
| -- | -------------------------- | --- |
| ✅  | Gymnasium 框架入门与基础案例        | 已完成 |
| ✅  | GridWorld 自定义环境 + 智能体实现    | 已完成 |
| 🔄 | CarRacing 连续动作控制           | 进行中 |
| ⏳  | PettingZoo 多智能体环境搭建        | 计划中 |
| ⏳  | MARL 算法实现（MADDPG / QMIX 等） | 计划中 |
| ⏳  | 训练可视化 + 性能对比分析             | 计划中 |

---

## 📊 结果与日志

训练过程中生成的日志与可视化结果将保存至：

* `logs/`：智能体 reward、loss、episode 数据
* `videos/`：环境运行视频（需启用渲染）

后续将加入一键可视化脚本，用于绘制 reward 曲线与策略变化。

---

## 🧩 未来扩展方向

* [ ] 实现 PettingZoo 环境的多智能体协作与竞争案例

---

## 🤝 参与交流

欢迎任何形式的交流与改进建议：

* 提个 [Issue](https://github.com/yourusername/MAS-RL-Tutorial/issues)

---

## 📚 学习资源推荐

| 主题                    | 链接                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------ |
| Gymnasium 官方文档        | [https://gymnasium.farama.org](https://gymnasium.farama.org)                         |
| PettingZoo 多智能体环境     | [https://pettingzoo.farama.org](https://pettingzoo.farama.org)                       |
| Stable-Baselines3 算法库 | [https://stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io) |
| Multi-Agent RL 综述论文   | [ArXiv:1908.03963](https://arxiv.org/abs/1908.03963)                                 |

---

💬 *强化学习的乐趣，不止在于训练智能体，更在于让自己“成为那个能训练智能体的人”。*

---
