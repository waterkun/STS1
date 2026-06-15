# Ironclad 套路协同手册

本文档描述铁甲战士的四大核心流派、14 组协同信号、及其相互关系。
所有内容与 `ml_advisor_v2.py` 中的常量保持一致。

> **关于升级牌（"+" 牌）：** 模型在所有协同信号计算中自动去掉 `+` 后缀——`"Pommel Strike+"` 与 `"Pommel Strike"` 触发完全相同的协同集合。
>
> 但**升级状态会影响整体评分权重**，模型通过 `upgrade_context_features` 捕捉以下三个维度：
>
> | 特征 | 含义 |
> |---|---|
> | `is_upgraded` | 候选牌本身是否已升级（有"+"） |
> | `unupgraded_ratio` | 牌组中非基础牌（除打击/防御外）未升级的比例 |
> | `is_power_card` | 是否是能力牌（Corruption、Demon Form 等整场只打一次的牌，升级影响相对小） |
>
> 模型将上述特征与 `temporal_features` 中的 `floor_pct` / `floors_to_boss` 联合，自动学习：
> - **高层 + 大量未升级牌** → 新牌轮不到升级 → 未升级牌的实际价值折扣
> - **能力牌** → 即使未升级，`is_power_card=1` 告知模型折扣更小
> - **循环核心牌**（Pommel Strike、Dropkick）→ 即使未升级，`loop_pair` 等协同信号高 → 模型仍维持高权重

---

## 一、四大核心流派

模型用 `_ARCHETYPE_CORE` 追踪以下四个互不重叠的流派，每个流派有一张"引擎牌"（`_KEY_ENGINES`）作为开关。

### 1. Barricade 格挡转化流

**引擎牌：** Barricade
**核心逻辑：** 格挡不再在回合结束时消失，所有格挡牌变成永久血量盾，Body Slam 将格挡直接转化为伤害。

| 角色 | 卡牌 |
|---|---|
| 格挡引擎 | Barricade, Impervious, Ghostly Armor, Flame Barrier |
| 格挡放大 | Entrench（格挡翻倍）, Juggernaut（格挡时造成伤害）|
| 输出转化 | Body Slam（格挡值=伤害）|

**协同链：**
```
Barricade
  └─ 格挡牌（Impervious / Ghostly Armor）→ 堆积格挡
       └─ Entrench → 格挡翻倍
            └─ Body Slam → 输出 = 格挡值
                 └─ Juggernaut → 每次格挡自动造成伤害
```

**与其他流派的交叉：**
- 和 **Corruption 流** 共享 Feel No Pain（消耗→格挡），可叠加
- Barricade + Corruption + Feel No Pain = 每消耗一张牌就获得格挡且格挡永久保留

---

### 2. Corruption 消耗引擎流

**引擎牌：** Corruption
**核心逻辑：** Corruption 使所有技能牌变为 0 费且打完消耗，配合消耗触发类牌提供持续收益。

| 角色 | 卡牌 |
|---|---|
| 引擎 | Corruption（技能0费消耗）|
| 消耗收益 | Feel No Pain（消耗→格挡）, Dark Embrace（消耗→抽牌）|
| 辅助消耗 | Fiend Fire, Sever Soul, True Grit, Second Wind, Sentinel |

**协同链：**
```
Corruption
  ├─ Feel No Pain → 每消耗一张牌 +5 格挡
  ├─ Dark Embrace → 每消耗一张牌 +1 抽牌（可无限循环）
  └─ Fiend Fire   → 消耗全手牌，每张 +7 伤害
```

**关键 Combo：**
- `Corruption + Dark Embrace`：每回合技能牌消耗时抽更多牌，形成滚雪球
- `Corruption + Feel No Pain + Barricade`：技能牌消耗一张 = 永久格挡 +5
- `Corruption + Fiend Fire`：一次性清空手牌打出巨额伤害（适合 boss 斩杀）

---

### 3. Strength 力量叠加流

**引擎牌：** Demon Form
**核心逻辑：** 每回合叠加力量，让每张攻击牌的伤害持续增长，多段攻击牌放大效果更强。

| 角色 | 卡牌 |
|---|---|
| 力量生成 | Demon Form（每回合+2力量）, Inflame（+2力量）, Spot Weakness（Boss时+3力量）|
| 力量放大 | Limit Break（力量翻倍）|
| 力量受益 | Heavy Blade（力量×3）, Reaper（多段+力量）, Whirlwind（X费多段）, Pummel, Sword Boomerang |

**协同链：**
```
Demon Form（每回合+2力量）
  └─ Inflame → 立即再+2力量
       └─ Limit Break → 力量翻倍
            ├─ Heavy Blade → 伤害 = 14 + 力量×3
            └─ Whirlwind   → X费，每段+力量，力量越高越恐怖
```

**注意：** 力量流需要有"拖延"能力（格挡牌）让 Demon Form 运转几回合。
Strength 流 + Barricade 流的格挡支撑是自然搭配。

---

### 4. Exhaust 消耗通用流

**引擎牌：** Fiend Fire
**核心逻辑：** 不依赖 Corruption，通过主动消耗牌薄化牌组并触发消耗收益，最终 Fiend Fire 清空手牌打出巨额伤害。

| 角色 | 卡牌 |
|---|---|
| 消耗来源 | Fiend Fire, Burning Pact, True Grit, Second Wind, Sentinel, Sever Soul |
| 消耗收益 | Feel No Pain（消耗→格挡）, Dark Embrace（消耗→抽牌）|

**与 Corruption 流的区别：**
- Corruption 流依赖 Corruption 做大量被动消耗；Exhaust 流主动选择性消耗，更灵活
- 两者都受益于 Feel No Pain / Dark Embrace，可以共享

---

## 二、14 组协同信号（Synergy Pairs）

模型对每张候选牌计算以下 14 个二元信号（1=协同触发，0=无），用于区分"孤立好牌"和"联动好牌"。

### 力量相关

| 信号名 | 条件 | 含义 |
|---|---|---|
| `strength` | 有 Inflame/Spot Weakness/Demon Form/Limit Break + 选 Heavy Blade/Whirlwind 等 | 力量生成与受益卡同时出现 |

### 消耗相关

| 信号名 | 条件 | 含义 |
|---|---|---|
| `exhaust` | 有消耗来源 + 选 Dark Embrace/Feel No Pain/Sentinel | 消耗量足够让收益牌发光 |
| `corruption_combo` | 有 Corruption + 选 Dark Embrace/Feel No Pain | Corruption 引擎的核心组合 |

### 格挡相关

| 信号名 | 条件 | 含义 |
|---|---|---|
| `block_scale` | 有格挡来源 + 选 Barricade/Entrench/Body Slam/Juggernaut/Metallicize | 格挡叠加链完整 |
| `barricade_combo` | 有 Barricade + 选 Body Slam/Entrench | Barricade 专属放大 |

### 抽牌/能量相关

| 信号名 | 条件 | 含义 |
|---|---|---|
| `draw_payoff` | 有 Battle Trance/Offering/Pommel Strike/Dropkick 等 + 选 Fiend Fire/Whirlwind/Limit Break | 抽牌引擎为大招蓄力 |
| `energy_payoff` | 有 Offering/Seeing Red/Berserk/**Bloodletting** + 选 Whirlwind/Bludgeon/Demon Form/Barricade | 额外能量被高费牌充分消化 |

### 状态牌相关

| 信号名 | 条件 | 含义 |
|---|---|---|
| `evolve_status` | 有 Evolve + 选 Wild Strike/Power Through/Immolate/Reckless Charge | 产出状态牌的牌充当 Evolve 的燃料 |
| `fire_breathing` | 有 Fire Breathing + 选 Wild Strike/Power Through/Immolate/Reckless Charge | 同上，用于 Fire Breathing |

### 易伤/多段相关

| 信号名 | 条件 | 含义 |
|---|---|---|
| `vulnerable_multi` | 有 Bash/Uppercut/Thunderclap 等 + 选 Pummel/Twin Strike/Whirlwind | 易伤放大多段攻击 |
| `paper_phrog_vuln` | 有 Paper Phrog 遗物 + 选易伤施加牌 | 遗物放大易伤价值 |

### 破甲/自伤相关

| 信号名 | 条件 | 含义 |
|---|---|---|
| `rupture_self_damage` | 有 Rupture + 选 Hemokinesis/Brutality/Combust/Offering/Bloodletting | 自伤触发 Rupture 叠力量（Offering 失 6 血、Bloodletting 失 3 血均计入）|

### Loop 无限循环相关

| 信号名 | 条件 | 含义 |
|---|---|---|
| `loop_pair` | 有 Pommel Strike + 选 Dropkick（或反之）| 两张牌互为条件，构成无限循环核心 |
| `dropkick_vuln` | 有 Bash/Uppercut 等易伤 + 选 Dropkick | Dropkick 需要易伤状态才能触发 0 费 |
| `rage_loop` | 有 Rage + 选 Dropkick/Pommel Strike/多段攻击 | Rage 在循环打出攻击时产生无限格挡 |

---

## 三、Loop 无限循环详解

Loop 是铁甲战士最复杂的 Combo，由多张牌共同构成：

```
前提：敌人处于易伤状态（Bash / Uppercut / Thunderclap）

Pommel Strike（2费）
  → 抽1张牌（大概率抽到 Dropkick）
  → 造成伤害

Dropkick（如果敌人易伤，费用变为0）
  → 造成伤害
  → 抽1张牌 + 回复1点能量
  → 大概率抽到 Pommel Strike

只要**等效循环规模 ≤ 15**，两张牌不断互相抽出对方
= 一回合无限输出

> **等效循环规模** = 牌组总数 − 消耗牌数
> 消耗牌打出后从循环中消失，不参与后续抽牌：
> - **无 Corruption**：`_EXHAUST_PRODUCERS`（True Grit、Sever Soul、Fiend Fire 等）直接减小循环规模
> - **有 Corruption**：所有技能牌均消耗，等效规模大幅压缩，20 张牌的组也可能满足条件
```

**Loop 增强牌：**

| 牌名 | 作用 |
|---|---|
| Rage | 每次打出攻击牌获得格挡，循环中产生无限格挡 |
| Anger | 打出时将副本加入弃牌堆，增加循环牌密度 |
| Clash | 0 费，全攻击手时必出，填充攻击链 |
| **Bloodletting** | 失去 3 HP 获得 2 能量，不依赖任何遗物即可在循环中即时补能；0 费版（Bloodletting+）更是零代价能量来源 |
| Shuriken / Kunai | 每打出 3 张攻击牌触发，循环中必然触发多次 |

**Loop 的必要条件：**
1. Pommel Strike + Dropkick 同时在牌组
2. 易伤施加来源（Bash 等）
3. 等效循环规模 ≤ 15（消耗牌、Corruption 均有助于缩减）

---

## 四、遗物协同

| 遗物 | 最佳搭配牌 | 原因 |
|---|---|---|
| **Dead Branch** | Fiend Fire, Corruption 体系的消耗牌 | 每消耗一张牌随机加入一张牌，消耗流雪球效应 |
| **Snecko Eye** | Whirlwind, Fiend Fire, Bludgeon, Heavy Blade, Sever Soul, Uppercut, Impervious, Carnage, Immolate, Feed, Reaper | 高费/X费牌在随机到低费时输出极高；额外抽2张弥补随机不稳定性 |
| **Necronomicon** | Carnage, Bludgeon, Hemokinesis, Pummel, Searing Blow, Immolate, Reaper | 2费以上攻击牌打出时自动再打一次 |
| **Mark of Pain** | 任意攻击牌 | 每回合多抽1张牌，攻击牌越多受益越大 |
| **Pen Nib** | Bludgeon, Heavy Blade, Carnage, Hemokinesis, Reaper | 每打出第10张牌伤害翻倍，适合高伤单发牌 |
| **Shuriken** | Pommel Strike, Clash, Dropkick, Twin Strike, Sword Boomerang, Pummel, Whirlwind, Anger | 一回合打出3张攻击→+1力量，循环流中必然多次触发 |
| **Kunai** | 同 Shuriken | 一回合打出3张攻击→+1敏捷，提升格挡效率 |
| **Paper Phrog** | Bash, Uppercut, Thunderclap, Shockwave, Clothesline | 易伤层数翻倍，放大所有多段和高伤攻击 |
| **Ornamental Fan** | 任意攻击牌 | 每打出3张攻击获得4格挡，循环流自动触发 |

---

## 五、流派兼容性矩阵

一副牌可以同时发展多个流派，以下是常见的双流派组合：

| 组合 | 兼容性 | 说明 |
|---|---|---|
| Barricade + Corruption | **极强** | Corruption 消耗技能→Feel No Pain 获得格挡→Barricade 格挡永久 |
| Barricade + Strength | **强** | Strength 流需要格挡支撑拖回合，Barricade 提供永久盾 |
| Corruption + Exhaust | **强** | Corruption 是 Exhaust 流的超级版，Feel No Pain / Dark Embrace 两者通用 |
| Exhaust + Loop | **强** | 消耗牌直接压缩等效循环规模，True Grit / Sever Soul 每打一次循环变薄一圈；Fiend Fire 可作为 loop 斩杀收尾 |
| Strength + Loop | **中** | Loop 需要薄牌组，Strength 需要多回合堆叠，目标略有冲突但可行 |
| Loop + Rupture | **中** | Loop 自伤少，Rupture 需要持续自伤，通常不同时追求 |
| Barricade + Loop | **弱** | Barricade 需要格挡牌（多为技能），Loop 需要薄纯攻击手，互相稀释 |

---

## 六、关键引擎牌速查

| 引擎牌 | 所属流派 | 没有它时 |
|---|---|---|
| **Barricade** | barricade 流 | 格挡每回合归零，Body Slam 无用 |
| **Corruption** | corruption 流 | 技能牌无法批量 0 费，消耗引擎失去核心 |
| **Demon Form** | strength 流 | 力量无法每回合自动累积，只能靠 Inflame 单次触发；Loop 流中每循环一回合就多 +2 力量，回合数越多收益越大 |
| **Limit Break** | strength 流 | 力量只能线性增长；有 Demon Form + Limit Break = 力量指数级膨胀（+2 → 翻倍 → 再 +2 → 再翻倍） |
| **Fiend Fire** | exhaust 流 | 消耗收益没有大招出口，Feel No Pain / Dark Embrace 收益减半 |

---

## 七、卡组构建建议

### 早期（Act 1）
优先确立**一个流派方向**，不要贪多：
- 拿到 Barricade → 优先拿格挡放大牌，跳过纯攻击牌
- 拿到 Corruption → 优先拿 Feel No Pain / Dark Embrace
- 没有引擎牌 → 保持通用性，优先薄牌组 + 高质量攻击

### 中期（Act 2）
- 确认主流派后补充协同牌，拒绝不相关的好牌
- Loop 牌组此时应已有 Pommel Strike + Dropkick，优先薄化牌组（商店移除基础牌）

### 后期（Act 3）
- 流派完整时，新牌只补充"当前最弱环节"
- Strength 流 → 是否有 Limit Break？没有则积极寻找
- Barricade 流 → 格挡总量是否足够抵挡 Act 3 Boss？
- Loop 流 → 牌组是否足够薄，是否有易伤来源？
