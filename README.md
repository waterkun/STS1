# 杀戮尖塔 AI 决策助手

基于历史对局数据和 Claude AI 构建的杀戮尖塔决策辅助工具。通过分析大量高阶（A20）对局数据，为玩家在关键决策点提供实时建议。

**当前支持角色：铁甲战士（Ironclad）、静默猎手（The Silent）、观者（Watcher）**

## 功能概览

### AI 决策顾问

在以下场景提供 AI 建议：

- **卡牌奖励** — 战斗胜利后，推荐从奖励中选择哪张卡牌（或跳过）
- **篝火决策** — 根据当前血量和卡组，建议休息还是升级（以及升级哪张）
- **Boss 遗物** — 击败 Boss 后，推荐选择哪个遗物奖励
- **商店购买** — 根据金币和卡组状态，建议在商店购买什么

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

| 指标 | 铁甲战士 | 静默猎手 | 观者 |
|------|----------|----------|------|
| 对局数 | 14,586 局 | 12,328 局 | 5,166 局 |
| 卡牌决策样本 | 250,260 条 | 200,171 条 | 92,929 条 |
| 篝火决策样本 | 80,759 条 | 66,884 条 | 29,938 条 |
| Boss 遗物决策样本 | 16,185 条 | 12,870 条 | 6,423 条 |
| 商店决策样本 | 706 条 | 602 条 | 543 条 |

### 模型架构

项目包含三代模型（V1 + V2 + V3），推理时综合所有模型的 Borda 排名给出最终建议。

| 模型 | 类型 | 说明 |
|------|------|------|
| **XGBoost** (V1) | 二分类 | 5-fold CV，预测「选择该选项后的胜率」，GPU 加速 |
| **LightGBM** (V1) | 二分类 | 同上，作为对照引擎 |
| **LambdaMART** (V2) | 排序模型 (LGBMRanker) | 直接优化选项排序，标签 2=选了且赢/1=选了且输/0=没选 |
| **Logistic Regression** (V2) | 选择模型 | 学习「高手倾向选什么」，Pipeline(Imputer+Scaler+LogReg) |
| **CWR-Delta** (V2) | 纯统计 | 条件胜率差异 + 贝叶斯平滑，按 act × deck_size × context 分桶 |
| **Transformer** (V3) | 集合排序模型 | 纯 NumPy 实现的自注意力编码器，候选选项互相感知，ListNet 损失 |

**V3 与 V1/V2 的核心区别**：V1/V2 对每个候选选项独立打分，无法感知同一决策中其他候选项的存在。V3 通过自注意力机制让每个选项的评分受整个候选集影响，从而捕捉选项间的协同与机会成本（例如：当 Barricade 也在奖励中时，Feel No Pain 的价值更高）。

### 特征工程

基础特征向量（所有模型共享）：
- 数值特征：floor、hp_pct、deck_size、num_relics、**num_upgrades**、**upgrade_ratio**
- Act one-hot (4维)
- 卡组计数向量 (~400维)
- 遗物 0/1 向量 (~200维)
- **卡组升级向量** (~400维) — 记录每张卡被升级的次数

V2 额外特征：卡组攻击/技能/能力占比、候选卡在卡组中已有数量、机制关键词组计数、流派匹配得分、synergy 协同得分。

V3 使用与 V2 完全相同的特征向量，但通过自注意力机制在选项之间共享信息，而非独立评分。

推理后处理：对卡组中已有的能力牌（Power）施加重复惩罚（×0.5），避免推荐无意义的重复能力牌。

## 快速开始

### 环境要求

- Python 3.10+
- [Poetry](https://python-poetry.org/) 包管理器

### 安装

```shell
# 克隆项目
git clone <repo-url>
cd STS1

# 安装依赖
pip install poetry
poetry install
```

### 构建决策数据库

在使用 ML 模型前，需要先从对局数据中构建决策数据库：

```shell
python -m ironclad_advisor.build_db
python -m silent_advisor.build_db
python -m watcher_advisor.build_db
```

### 训练模型

```shell
# 铁甲战士
python -m ironclad_advisor.ml_advisor train
python -m ironclad_advisor.ml_advisor_v2 train
python -m ironclad_advisor.ml_advisor_v3 train

# 静默猎手
python -m silent_advisor.ml_advisor train
python -m silent_advisor.ml_advisor_v2 train
python -m silent_advisor.ml_advisor_v3 train

# 观者
python -m watcher_advisor.ml_advisor train
python -m watcher_advisor.ml_advisor_v2 train
python -m watcher_advisor.ml_advisor_v3 train
```

### CLI 推理

```shell
# 卡牌奖励建议 (V1)
python -m ironclad_advisor.ml_advisor card \
  --floor 8 --act 1 --hp 45 --max-hp 80 \
  --relics "Burning Blood,Bag of Marbles" \
  --deck "Strike_R x4,Defend_R x4,Bash,Iron Wave,Shrug It Off" \
  --options "Barricade,Feel No Pain,Pommel Strike"

# 卡牌奖励建议 (V2 综合)
python -m ironclad_advisor.ml_advisor_v2 card \
  --floor 8 --act 1 --hp 45 --max-hp 80 \
  --relics "Burning Blood,Bag of Marbles" \
  --deck "Strike_R x4,Defend_R x4,Bash,Iron Wave,Shrug It Off" \
  --options "Barricade,Feel No Pain,Pommel Strike"

# 卡牌奖励建议 (V3 Transformer)
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

配合 [CommunicationMod](https://github.com/ForgottenArbiter/CommunicationMod) 使用，在游戏中自动给出建议：

```shell
# 根据角色选择对应的 advisor
python -m ironclad_advisor.communicate
python -m silent_advisor.communicate
python -m watcher_advisor.communicate
```

## 项目结构

```
STS1/
├── transformer_core.py     # V3 共享核心：纯 NumPy Transformer（三角色共用）
├── ironclad_advisor/       # 铁甲战士 ML 顾问
│   ├── build_db.py         # 决策数据库构建（解析 .run 文件）
│   ├── ml_advisor.py       # V1 模型：XGBoost + LightGBM 二分类
│   ├── ml_advisor_v2.py    # V2 模型：LambdaMART + LogReg + CWR
│   ├── ml_advisor_v3.py    # V3 模型：Transformer 自注意力排序
│   ├── communicate.py      # CommunicationMod 实时集成
│   ├── db/                 # 生成的决策数据库 (JSON)
│   └── models/             # 训练好的模型 (pickle)
├── silent_advisor/         # 静默猎手 ML 顾问（结构同上）
│   ├── build_db.py         # 过滤 THE_SILENT，基础牌组 Strike_G/Defend_G
│   ├── ml_advisor.py
│   ├── ml_advisor_v2.py    # 毒/弃牌/小刀/格挡 synergy
│   ├── ml_advisor_v3.py    # V3 Transformer
│   ├── communicate.py
│   ├── db/
│   └── models/
├── watcher_advisor/        # 观者 ML 顾问（结构同上）
│   ├── build_db.py         # 过滤 WATCHER，基础牌组 Strike_P/Defend_P
│   ├── ml_advisor.py
│   ├── ml_advisor_v2.py    # 愤怒/冷静/神性/占卜 synergy
│   ├── ml_advisor_v3.py    # V3 Transformer
│   ├── communicate.py
│   ├── db/
│   └── models/
├── runs/                   # 原始对局数据 (.run 文件)
└── pyproject.toml          # 项目依赖
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
- **XGBoost** — V1 梯度提升分类器 (GPU)
- **LightGBM** — V1 分类 + V2 LambdaMART 排序 (GPU)
- **scikit-learn** — Logistic Regression、交叉验证、评估指标
- **NumPy** — 特征编码与矩阵运算；V3 Transformer 的完整实现（前向/反向传播、Adam 优化器均纯 NumPy 手写，无框架依赖）
- **Poetry** — 依赖管理

## 开发路线

- [x] 对局数据统计分析
- [x] 铁甲战士 AI 顾问
- [x] 静默猎手（Silent）AI 顾问
- [ ] 机器人（Defect）AI 顾问
- [x] 观者（Watcher）AI 顾问
- [x] 重复能力牌惩罚机制
- [x] V3 Transformer 模型（纯 NumPy 自注意力，候选选项互相感知）

## 许可证

Apache 2.0
