# 从 V1 到 V2 的学习指南

## 第一步：理解 V1 的基础架构

**先读这个文件：** 任意一个角色的 `ml_advisor.py`（建议从 Ironclad 开始）

```
ironclad_advisor/ml_advisor.py
```

**V1 核心概念：**
- 二分类模型（XGBoost / LightGBM）
- 每个候选选项**独立评分**：`f(游戏状态, 选项) → 预测胜率`
- 5-fold StratifiedKFold 交叉验证
- 特征工程基础：数值特征(6维) + Act编码(4维) + 卡牌向量(~400维) + 遗物向量(~200维)

**学习重点：**
1. 数据如何从 `.run` 文件解析（先看 `build_db.py`）
2. 特征是如何构建的
3. 模型训练和评估流程

---

## 第二步：理解 V1 的不足

V1 的核心问题：**区分度不足** — 独立评分无法很好地区分相似选项。这是 V2 要解决的问题。

---

## 第三步：学习 V2 的三模型并行架构

**读这个文件：**
```
ironclad_advisor/ml_advisor_v2.py
```

**V2 引入了三个并行模型：**

| 模型 | 作用 | 关键点 |
|------|------|--------|
| **LambdaMART (LGBMRanker)** | 列表排序优化 | 标签 2/1/0，直接优化排序 |
| **Logistic Regression** | 选择概率预测 | 学习"高手玩家倾向选什么" |
| **CWR-Delta** | 条件胜率统计 | 贝叶斯平滑，按act×牌组大小分桶 |

---

## 第四步：建议的阅读顺序

1. **`README.md`** — 项目整体文档（中文，约16KB）
2. **`ironclad_advisor/build_db.py`** — 理解数据来源和格式
3. **`ironclad_advisor/ml_advisor.py`** — V1 完整实现
4. **`ironclad_advisor/ml_advisor_v2.py`** — V2 完整实现，对比 V1 的变化
5. **`IRONCLAD_SYNERGY.md`** — V2 新增的协同特征文档
6. **`train_all.py`** — 训练流程控制器

---

## 第五步：关键差异对比

从 V1 到 V2 的核心变化：

| 方面 | V1 | V2 |
|------|----|----|
| 模型数量 | 1个 | 3个并行 |
| 标签体系 | 0/1 二分类 | 0/1/2 排序标签 |
| 评分方式 | 独立评分 | 独立评分 + 统计修正 |
| 特征 | 基础特征 | 新增协同特征、原型匹配、序列上下文 |
| 融合方式 | 单模型输出 | 多模型综合 |

---

## 实践建议

1. **先跑通 V1**：用 `python train_all.py --chars ironclad --steps db v1` 训练一次
2. **对比代码**：逐函数对比 `ml_advisor.py` 和 `ml_advisor_v2.py`
3. **关注新增特征**：V2 的协同关键词(synergy keywords)、原型匹配(archetype matching) 是重要创新
4. **理解 CWR-Delta**：这是纯统计方法，不需要ML背景就能理解，是很好的入门点
