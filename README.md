# 杀戮尖塔 AI 决策助手

基于历史对局数据训练的杀戮尖塔决策辅助系统。通过分析大量高阶（A20）对局数据，为玩家在关键决策点提供实时建议。

**支持角色：铁甲战士（Ironclad）、静默猎手（Silent）、机器人（Defect）、观者（Watcher）**

## 功能概览

### AI 决策顾问

在以下场景提供建议：

- **卡牌奖励** — 战斗胜利后，推荐从奖励中选择哪张卡牌（或跳过）
- **篝火决策** — 根据当前血量和卡组，建议休息还是升级
- **Boss 遗物** — 击败 Boss 后，推荐选择哪个遗物奖励
- **商店购买** — 根据金币和卡组状态，建议在商店购买什么

推理时所有可用模型同时运行，通过 **Borda Count** 投票综合排名给出最终建议。

### 数据分析

对原始对局数据进行统计分析，生成：

- 胜率、通关层数等总体统计
- 卡牌选取率与胜率关联分析
- 遗物效果评估
- Boss 伤害统计
- 篝火决策分布
- CSV 表格、PNG 图表、Markdown 报告

## 模型与数据规模

### 训练数据

| 指标 | 铁甲战士 | 静默猎手 | 机器人 | 观者 |
|------|----------|----------|--------|------|
| 对局数 | 14,586 局 | 12,328 局 | — | 5,166 局 |
| 卡牌决策样本 | 250,260 条 | 200,171 条 | — | 92,929 条 |
| 篝火决策样本 | 80,759 条 | 66,884 条 | — | 29,938 条 |
| Boss 遗物决策样本 | 16,185 条 | 12,870 条 | — | 6,423 条 |
| 商店决策样本 | 706 条 | 602 条 | — | 543 条 |

### 模型架构（三代演进）

#### V1 — 二分类（XGBoost / LightGBM）

每个候选选项独立打分：`f(游戏状态, 选项)` → 预测「选择该选项后的胜率」。

- 5-fold StratifiedKFold 交叉验证，OOF 预测评估 AUC
- GPU 加速（XGBoost `device=cuda`，LightGBM `device=gpu`）
- 铁甲战士专有（其他角色使用 V2 作为基础模型）

#### V2 — 排序 + 统计（LambdaMART / LogReg / CWR-Delta）

三种模型并行，解决 V1 区分度不足的问题：

| 模型 | 类型 | 训练目标 |
|------|------|---------|
| **LambdaMART** (LGBMRanker) | listwise 排序 | 直接优化选项排序，标签 2=选了且赢 / 1=选了且输 / 0=没选 |
| **Logistic Regression** | 选择概率 | 学习「高手倾向选什么」，Pipeline(SimpleImputer → StandardScaler → LogReg) |
| **CWR-Delta** | 纯统计 | 条件胜率差异 + 贝叶斯平滑（按 act × deck_size_bucket × context 分桶，strength=5） |

V2 在 V1 基础上新增特征（观者为例）：
- 卡组攻击 / 技能 / 能力占比（3 维）
- 机制关键词组计数（13 维）：Wrath 生成器、Calm 收益、神性、占卜、格挡缩放等
- 流派匹配得分（5 维）：wrath、calm_control、divinity、scry、thin_deck
- 协同效应得分（11 维）：候选卡与卡组中各 synergy pair 的匹配
- 时序特征（8 维）：act 内进度、距 boss 楼层数、精英区标记、HP 分桶
- 候选卡在卡组中已有数量（防止推荐重复）

#### V3 — Transformer 集合排序（PyTorch）

**核心改进**：V1/V2 对每个候选选项独立打分，无法感知同一决策中其他候选项的存在。V3 通过自注意力机制，让每个选项的评分受整个候选集影响，从而捕捉**选项间的机会成本与相对价值**。

架构（`transformer_core.py`）：

```
输入 (N, F)
  → Linear(F, d_model=128)           # 投影到隐空间
  → 2 × TransformerBlock(Pre-Norm)   # 自注意力 + FFN
      LN → MultiHeadAttention(4头)→ 残差
      LN → FFN(d_ff=256, ReLU)    → 残差
  → Linear(d_model, 1)               # 每个选项输出一个分数
  → softmax                          # 归一化为概率 (N,)
```

其中 N = 本次决策的候选选项数（卡牌奖励 N≈3，Boss 遗物 N=3，篝火 N=2，商店 N=5~10）。

**训练细节**：
- 损失函数：**ListNet loss**（KL 散度形式）—— 标签经 softmax 软化后与预测分布做交叉熵，对「选了但没赢」的情况梯度更平滑
- 标签来源：复用 V2 排序标签（2 / 1 / 0）
- 变长候选集处理：padding + `key_padding_mask` 屏蔽 padding 位置的注意力
- Batch 内按候选数排序后局部打乱，减少 padding 浪费
- 优化器：Adam，CosineAnnealingLR（60 epochs，lr 从 5e-4 → 5e-6）
- 梯度裁剪：`clip_grad_norm_(max_norm=1.0)`
- 训练用 CUDA（若可用），推理时模型移回 CPU 序列化

**后处理**：对卡组中已有的能力牌（Power）候选项施加 ×0.5 重复惩罚，避免推荐无意义的重复能力牌。

### 已训练模型状态

| 角色 | V1 (XGB/LGB) | V2 (LambdaMART/LogReg/CWR) | V3 (Transformer) |
|------|:---:|:---:|:---:|
| 铁甲战士 | ✓ | ✓ | ✓ |
| 静默猎手 | — | ✓ | ✓ |
| 机器人 | — | — | ✓ |
| 观者 | ✓ | ✓ | 未训练 |

### 特征工程（基础特征，所有模型共享）

| 特征 | 维度 | 说明 |
|------|------|------|
| 数值特征 | 6 | floor、hp_pct、deck_size、num_relics、num_upgrades、upgrade_ratio |
| Act one-hot | 4 | Act 1~4 |
| 卡组计数向量 | ~400 | 每张卡在卡组中的数量 |
| 遗物 0/1 向量 | ~200 | 是否持有该遗物 |
| 卡组升级向量 | ~400 | 每张卡被升级的次数 |

## 快速开始

### 环境要求

- Python 3.10+
- [Poetry](https://python-poetry.org/) 包管理器

### 安装

```shell
git clone <repo-url>
cd STS1

pip install poetry
poetry install

# V3 Transformer 需要 PyTorch（pyproject.toml 未包含，需单独安装）
pip install torch
```

### 构建决策数据库

在使用 ML 模型前，需先从对局数据中构建决策数据库：

```shell
python -m ironclad_advisor.build_db
python -m silent_advisor.build_db
python -m defect_advisor.build_db
python -m watcher_advisor.build_db
```

### 训练模型

```shell
# 铁甲战士（全部三代）
python -m ironclad_advisor.ml_advisor train       # V1: XGBoost + LightGBM
python -m ironclad_advisor.ml_advisor_v2 train    # V2: LambdaMART + LogReg + CWR
python -m ironclad_advisor.ml_advisor_v3 train    # V3: Transformer

# 静默猎手
python -m silent_advisor.ml_advisor train
python -m silent_advisor.ml_advisor_v2 train
python -m silent_advisor.ml_advisor_v3 train

# 机器人
python -m defect_advisor.ml_advisor train
python -m defect_advisor.ml_advisor_v2 train
python -m defect_advisor.ml_advisor_v3 train

# 观者
python -m watcher_advisor.ml_advisor train
python -m watcher_advisor.ml_advisor_v2 train
python -m watcher_advisor.ml_advisor_v3 train    # V3 尚未训练，运行此命令生成
```

### CLI 推理

```shell
# 卡牌奖励建议 — V1（铁甲战士）
python -m ironclad_advisor.ml_advisor card \
  --floor 8 --act 1 --hp 45 --max-hp 80 \
  --relics "Burning Blood,Bag of Marbles" \
  --deck "Strike_R x4,Defend_R x4,Bash,Iron Wave,Shrug It Off" \
  --options "Barricade,Feel No Pain,Pommel Strike"

# 卡牌奖励建议 — V2 综合（LambdaMART + LogReg + CWR-Delta）
python -m ironclad_advisor.ml_advisor_v2 card \
  --floor 8 --act 1 --hp 45 --max-hp 80 \
  --relics "Burning Blood,Bag of Marbles" \
  --deck "Strike_R x4,Defend_R x4,Bash,Iron Wave,Shrug It Off" \
  --options "Barricade,Feel No Pain,Pommel Strike"

# 卡牌奖励建议 — V3 Transformer
python -m ironclad_advisor.ml_advisor_v3 card \
  --floor 8 --act 1 --hp 45 --max-hp 80 \
  --relics "Burning Blood,Bag of Marbles" \
  --deck "Strike_R x4,Defend_R x4,Bash,Iron Wave,Shrug It Off" \
  --options "Barricade,Feel No Pain,Pommel Strike"

# 篝火决策
python -m ironclad_advisor.ml_advisor_v2 campfire \
  --floor 11 --act 1 --hp 35 --max-hp 80 \
  --relics "Burning Blood" \
  --deck "Strike_R x4,Defend_R x4,Bash,Clothesline,Iron Wave"

# Boss 遗物选择
python -m ironclad_advisor.ml_advisor_v2 boss-relic \
  --act 1 --hp 55 --max-hp 80 \
  --relics "Burning Blood,Bag of Marbles" \
  --deck "Strike_R x4,Defend_R x4,Bash,Barricade,Feel No Pain,Iron Wave" \
  --options "Snecko Eye,Cursed Key,Coffee Dripper"

# 商店购买建议
python -m ironclad_advisor.ml_advisor_v2 shop \
  --floor 27 --act 2 --hp 60 --max-hp 80 --gold 300 \
  --relics "Burning Blood,Snecko Eye,Bag of Marbles" \
  --deck "Strike_R x3,Defend_R x4,Bash,Barricade,Feel No Pain x2,Reaper" \
  --cards "Demon Form,Reaper,Impervious" \
  --shop-relics "Mark of Pain,Du-Vu Doll" \
  --potions "Strength Potion,BloodPotion"
```

### 实时游戏建议（CommunicationMod 集成）

配合 [CommunicationMod](https://github.com/ForgottenArbiter/CommunicationMod) 在游戏中自动给出建议：

```shell
# 各角色独立入口
python -m ironclad_advisor.communicate
python -m silent_advisor.communicate
python -m defect_advisor.communicate
python -m watcher_advisor.communicate

# 统一入口（自动识别当前角色，推荐）
python communicate.py
```

## 项目结构

```
STS1/
├── transformer_core.py      # V3 共享核心：PyTorch Transformer 排序模型（四角色共用）
├── communicate.py           # 统一 CommunicationMod 入口（自动识别角色）
├── ironclad_advisor/        # 铁甲战士 ML 顾问
│   ├── build_db.py          # 决策数据库构建（解析 .run 文件）
│   ├── ml_advisor.py        # V1：XGBoost + LightGBM 二分类（5-fold CV）
│   ├── ml_advisor_v2.py     # V2：LambdaMART + LogReg + CWR-Delta
│   ├── ml_advisor_v3.py     # V3：Transformer 自注意力排序
│   ├── communicate.py       # CommunicationMod 实时集成
│   ├── db/                  # 生成的决策数据库（JSON）
│   └── models/              # 训练好的模型（.pkl / .pt）
├── silent_advisor/          # 静默猎手 ML 顾问（结构同上）
│   ├── ml_advisor_v2.py     # 毒 / 弃牌 / 小刀 / 格挡 synergy
│   └── ...
├── defect_advisor/          # 机器人 ML 顾问（结构同上）
│   ├── ml_advisor_v2.py     # 法球 / Focus / Claw / 能量 synergy
│   └── ...
├── watcher_advisor/         # 观者 ML 顾问（结构同上）
│   ├── ml_advisor_v2.py     # 愤怒 / 冷静 / 神性 / 占卜 synergy
│   └── ...
├── runs/                    # 原始对局数据（.run 文件）
└── pyproject.toml           # 项目依赖
```

## 数据集

| 数据集 | 说明 |
|--------|------|
| 200-rotating-sample | 200 局轮换角色样本 |
| bad-silent | 50 局弱牌组 Silent |
| chegs | 主播 Cheg 的对局数据 |
| lose-all-gold-max-hp-sample | 特殊策略对局 |
| panacea-ironclad-sample | 主播 panacea108 的铁甲战士数据 |
| robit | 50 局 Defect 对局 |

## 技术栈

- **Python 3.10+** — 主语言
- **PyTorch** — V3 Transformer 模型（Pre-Norm 编码器、ListNet loss、Adam + CosineAnnealingLR）
- **XGBoost** — V1 梯度提升分类器（GPU 加速）
- **LightGBM** — V1 分类 + V2 LambdaMART 排序（GPU 加速）
- **scikit-learn** — Logistic Regression、交叉验证、评估指标（AUC、NDCG）
- **NumPy** — 特征编码与矩阵运算
- **Poetry** — 依赖管理

## 开发路线

- [x] 对局数据统计分析
- [x] 铁甲战士 AI 顾问（V1 + V2 + V3 全部训练完成）
- [x] 静默猎手 AI 顾问（V2 + V3 训练完成）
- [x] 机器人 AI 顾问（V3 训练完成）
- [x] 观者 AI 顾问（V1 + V2 训练完成）
- [x] 重复能力牌惩罚机制
- [x] V3 PyTorch Transformer（自注意力，候选选项互相感知，ListNet loss）
- [ ] 观者 V3 Transformer 训练
- [ ] 静默猎手 / 机器人 V1 模型训练

## 许可证

Apache 2.0
