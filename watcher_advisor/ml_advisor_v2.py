#!/usr/bin/env python3
"""
Watcher ML Advisor V2 - 三种新模型解决区分度不足问题

模型 A: LambdaMART 排序模型 (LGBMRanker)
模型 B: Logistic Regression 选择模型
模型 C: CWR-Delta 条件胜率差异 (纯统计)

Usage:
  python watcher_advisor/ml_advisor_v2.py train
  python watcher_advisor/ml_advisor_v2.py card \
    --floor 8 --act 1 --hp 45 --max-hp 80 \
    --deck "Strike_P x4,Defend_P x4,Eruption,Vigilance" \
    --options "Cut Through Fate,Prostrate,Flurry of Blows"
"""

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import ndcg_score
except ImportError:
    print("scikit-learn 未安装，请运行: pip install scikit-learn")
    sys.exit(1)

from watcher_advisor.ml_advisor import (
    load_db,
    build_vocabularies,
    base_features,
    encode_deck,
    encode_deck_upgrades,
    encode_relics,
    load_vocab,
    load_models as load_v1_models,
    predict_with_models as predict_v1,
    card_inference_features as card_inference_features_v1,
    campfire_inference_features,
    boss_relic_inference_features as boss_relic_inference_features_v1,
    shop_inference_features as shop_inference_features_v1,
    parse_deck,
    parse_list,
    normalize_item_name,
    encode_item,
    DB_DIR,
    MODEL_DIR,
)


# ---------------------------------------------------------------------------
# 辅助：卡组分析特征（攻击/技能/能力占比 + 候选项已有数量）
# ---------------------------------------------------------------------------

# 基于杀戮尖塔观者卡牌分类
_CARD_TYPE = {}

_ATTACK_CARDS = {
    "Strike_P", "Eruption", "Bowling Bash", "Consecrate",
    "Crush Joints", "Cut Through Fate", "Empty Fist",
    "Flurry of Blows", "Flying Sleeves", "Follow Up",
    "Just Lucky", "Sash Whip", "Conclude", "Fear No Evil",
    "Reach Heaven", "Signature Move", "Talk to the Hand",
    "Tantrum", "Wallop", "Windmill Strike", "Brilliance",
    "Lesson Learned", "Ragnarok",
}
_SKILL_CARDS = {
    "Defend_P", "Vigilance", "Crescendo", "Empty Body",
    "Empty Mind", "Evaluate", "Halt", "Inner Peace",
    "Prostrate", "Protect", "Third Eye", "Tranquility",
    "Wave of the Hand", "Collect", "Deceive Reality",
    "Foresight", "Indignation", "Meditate", "Perseverance",
    "Pray", "Sanctity", "Swivel", "Weave",
    "Worship", "Alpha", "Blasphemy", "Conjure Blade",
    "Foreign Influence", "Omniscience", "Scrawl",
    "Spirit Shield", "Vault", "Wish",
}
_POWER_CARDS = {
    "Like Water", "Nirvana", "Rushdown", "Study",
    "Battle Hymn", "Deva Form", "Devotion",
    "Establishment", "Fasting", "Mental Fortress",
    "Sands of Time", "Wreath of Flame",
}


_BASIC_CARDS = {"Strike_P", "Defend_P"}


def shop_remove_context(deck: list[str]) -> np.ndarray:
    """商店移除上下文特征（3 维）:
    - n_basic: 基础牌数量 (Strike/Defend)
    - n_curse: 诅咒牌数量
    - thin_value: 1/deck_size (移除价值指标)
    """
    deck_bases = [c.split("+")[0].strip() for c in deck]
    n_basic = sum(1 for b in deck_bases if b in _BASIC_CARDS)
    n_curse = sum(1 for b in deck_bases if b in {
        "Curse of the Bell", "Necronomicurse", "Pain", "Parasite",
        "Regret", "Shame", "Doubt", "Burn", "Decay", "Injury",
        "Normality", "Pride", "Writhe", "Ascender's Bane", "Clumsy",
    })
    thin_value = 1.0 / max(len(deck), 1)
    return np.array([n_basic, n_curse, thin_value], dtype=np.float32)


def _card_type(name: str) -> str:
    base = name.split("+")[0].strip()
    if base in _ATTACK_CARDS:
        return "attack"
    if base in _SKILL_CARDS:
        return "skill"
    if base in _POWER_CARDS:
        return "power"
    return "attack"  # 默认


# ---------------------------------------------------------------------------
# 机制关键词分组
# ---------------------------------------------------------------------------

_WRATH_GENERATORS = {"Eruption", "Tantrum", "Rushdown", "Crescendo"}
_WRATH_PAYOFF = {"Flurry of Blows", "Tantrum", "Eruption", "Ragnarok", "Brilliance"}
_CALM_GENERATORS = {"Vigilance", "Empty Mind", "Fear No Evil", "Crescendo", "Inner Peace", "Like Water"}
_CALM_PAYOFF = {"Wreath of Flame", "Empty Fist", "Empty Body", "Empty Mind"}
_DIVINITY_ENABLERS = {"Prostrate", "Worship", "Devotion", "Brilliance"}
_MANTRA_GENERATORS = {"Prostrate", "Worship", "Devotion"}
_RETAIN_CARDS = {"Protect", "Establishment", "Windmill Strike", "Sands of Time"}
_SCRY_CARDS = {"Cut Through Fate", "Third Eye", "Foresight", "Just Lucky", "Nirvana", "Weave"}
_SCRY_SYNERGY = {"Nirvana", "Weave", "Foresight"}
_BLOCK_GENERATORS = {"Defend_P", "Vigilance", "Talk to the Hand", "Wallop", "Like Water", "Empty Body", "Protect"}
_BLOCK_SCALERS = {"Like Water", "Mental Fortress", "Talk to the Hand"}
_ENERGY_GENERATORS = {"Fasting", "Worship", "Deva Form"}
_STANCE_DANCERS = {"Flurry of Blows", "Mental Fortress", "Rushdown", "Tantrum", "Empty Fist", "Inner Peace", "Fear No Evil"}

_SYNERGY_PAIRS = [
    (_WRATH_GENERATORS, _WRATH_PAYOFF, "wrath"),
    (_CALM_GENERATORS, _CALM_PAYOFF, "calm_payoff"),
    (_CALM_GENERATORS | _WRATH_GENERATORS, _STANCE_DANCERS, "stance_dance"),
    ({"Rushdown"}, _CALM_GENERATORS, "rushdown_draw"),
    (_MANTRA_GENERATORS, _DIVINITY_ENABLERS, "divinity"),
    (_SCRY_CARDS, _SCRY_SYNERGY, "scry"),
    ({"Establishment"}, _RETAIN_CARDS, "retain"),
    (_BLOCK_GENERATORS, _BLOCK_SCALERS, "block_scale"),
    ({"Mental Fortress"}, _STANCE_DANCERS, "mental_fortress"),
    ({"Talk to the Hand"}, _WRATH_PAYOFF, "talk_hand"),
    (_ENERGY_GENERATORS, {"Deva Form", "Ragnarok", "Vault"}, "energy_payoff"),
]

_ARCHETYPES = {
    "wrath": _WRATH_GENERATORS | _WRATH_PAYOFF,
    "calm_control": _CALM_GENERATORS | _CALM_PAYOFF,
    "divinity": _MANTRA_GENERATORS | _DIVINITY_ENABLERS,
    "scry": _SCRY_CARDS | _SCRY_SYNERGY,
    "thin_deck": set(),
}

_ARCHETYPE_CORE = {k: v for k, v in _ARCHETYPES.items() if k != "thin_deck"}
_ARCHETYPE_NAMES = sorted(_ARCHETYPE_CORE.keys())  # calm_control, divinity, scry, wrath
_KEY_ENGINES = sorted(["Deva Form", "Establishment", "Mental Fortress", "Wreath of Flame"])

# 抽牌引擎牌 vs 抽牌收益牌
_DRAW_ENGINES_ARCH = {"Sanctity", "Inner Peace", "Tantrum", "Worship"}
_DRAW_PAYOFFS = {"Deva Form", "Vault", "Omniscience", "Brilliance"}
_ACT1_BOSSES = ["Hexaghost", "Slime Boss", "The Guardian"]
_ACT2_BOSSES = ["The Champ", "Automaton", "The Collector"]

_KEYWORD_GROUPS = [
    _WRATH_GENERATORS, _WRATH_PAYOFF,
    _CALM_GENERATORS, _CALM_PAYOFF,
    _MANTRA_GENERATORS, _DIVINITY_ENABLERS,
    _SCRY_CARDS, _SCRY_SYNERGY,
    _BLOCK_GENERATORS, _BLOCK_SCALERS,
    _ENERGY_GENERATORS, _STANCE_DANCERS,
    _RETAIN_CARDS,
]


# ---------------------------------------------------------------------------
# 新增特征函数
# ---------------------------------------------------------------------------

def deck_keyword_features(deck: list[str]) -> np.ndarray:
    """返回卡组中各机制关键词组的计数（13 维）。"""
    counts = np.zeros(len(_KEYWORD_GROUPS), dtype=np.float32)
    for card in deck:
        base = card.split("+")[0].strip()
        for i, group in enumerate(_KEYWORD_GROUPS):
            if base in group:
                counts[i] += 1
    return counts


def deck_archetype_features(deck: list[str]) -> np.ndarray:
    """返回各流派的匹配得分 (0~1)，共 5 维。"""
    total = max(len(deck), 1)
    scores = []
    for name, card_set in _ARCHETYPES.items():
        if name == "thin_deck":
            scores.append(1.0 if total <= 15 else 0.0)
        else:
            count = sum(1 for c in deck if c.split("+")[0].strip() in card_set)
            scores.append(count / total)
    return np.array(scores, dtype=np.float32)


def card_synergy_features(card: str, deck: list[str]) -> np.ndarray:
    """计算候选卡对各 synergy pair 的贡献（11 维）。"""
    base = card.split("+")[0].strip()
    deck_bases = set(c.split("+")[0].strip() for c in deck)
    scores = np.zeros(len(_SYNERGY_PAIRS), dtype=np.float32)
    for i, (enablers, payoffs, _) in enumerate(_SYNERGY_PAIRS):
        if base in payoffs and deck_bases & enablers:
            scores[i] = 1.0
        elif base in enablers and deck_bases & payoffs:
            scores[i] = 1.0
    return scores


def _deck_type_ratios(deck: list[str]) -> np.ndarray:
    """返回 [攻击占比, 技能占比, 能力占比]（3 维）。"""
    if not deck:
        return np.zeros(3, dtype=np.float32)
    counts = Counter(_card_type(c) for c in deck)
    total = len(deck)
    return np.array([
        counts.get("attack", 0) / total,
        counts.get("skill", 0) / total,
        counts.get("power", 0) / total,
    ], dtype=np.float32)


def deck_analysis_features(deck: list[str]) -> np.ndarray:
    """返回 [攻击占比, 技能占比, 能力占比, 关键词计数x13, 流派得分x5]，共 21 维。"""
    type_feats = _deck_type_ratios(deck)       # 3 维
    keyword_feats = deck_keyword_features(deck)  # 13 维
    archetype_feats = deck_archetype_features(deck)  # 5 维
    return np.concatenate([type_feats, keyword_feats, archetype_feats])


def card_count_in_deck(card: str, deck: list[str]) -> float:
    """候选卡在卡组中已有的数量。"""
    base = card.split("+")[0].strip()
    return float(sum(1 for c in deck if c.split("+")[0].strip() == base))


# ---------------------------------------------------------------------------
# 遗物-卡牌交互特征（9 维）
# ---------------------------------------------------------------------------

_RELIC_CARD_SYNERGIES = [
    ("Snecko Eye", {"Vault", "Omniscience", "Deva Form", "Brilliance", "Blasphemy", "Ragnarok", "Conjure Blade"}, "snecko_high_cost"),
    ("Pen Nib", {"Ragnarok", "Brilliance", "Windmill Strike", "Conclude", "Wallop"}, "pen_nib_high_dmg"),
    ("Shuriken", {"Flurry of Blows", "Tantrum", "Ragnarok", "Flying Sleeves"}, "shuriken_multi_atk"),
    ("Kunai", {"Flurry of Blows", "Tantrum", "Ragnarok", "Flying Sleeves"}, "kunai_multi_atk"),
    ("Ornamental Fan", _ATTACK_CARDS, "ornamental_fan_attacks"),
    ("Violet Lotus", _CALM_GENERATORS, "violet_lotus_calm"),
    ("Damaru", _MANTRA_GENERATORS | _DIVINITY_ENABLERS, "damaru_mantra"),
    ("Teardrop Locket", _CALM_GENERATORS | _CALM_PAYOFF, "teardrop_calm"),
    ("Duality", _ATTACK_CARDS, "duality_attacks"),
]


def relic_card_synergy_features(card: str, deck: list[str],
                                relics: list[str]) -> np.ndarray:
    """计算候选卡与当前遗物的协同特征（9 维）。"""
    base = card.split("+")[0].strip()
    relic_set = set(relics)
    scores = np.zeros(len(_RELIC_CARD_SYNERGIES), dtype=np.float32)
    for i, (relic_name, card_set, _) in enumerate(_RELIC_CARD_SYNERGIES):
        if relic_name in relic_set and base in card_set:
            scores[i] = 1.0
    return scores


def relic_deck_synergy_features(relic: str, deck: list[str]) -> np.ndarray:
    """反向协同：候选遗物与卡组中卡牌的协同计数（9 维）。"""
    deck_bases = [c.split("+")[0].strip() for c in deck]
    scores = np.zeros(len(_RELIC_CARD_SYNERGIES), dtype=np.float32)
    for i, (relic_name, card_set, _) in enumerate(_RELIC_CARD_SYNERGIES):
        if relic == relic_name:
            scores[i] = float(sum(1 for b in deck_bases if b in card_set))
    return scores


# ---------------------------------------------------------------------------
# 时序特征
# ---------------------------------------------------------------------------

def temporal_features(floor: int, act: int, hp_pct: float) -> np.ndarray:
    """游戏进程时序特征 (8维)。

    捕捉：当前在 act 中的进度、距 boss 距离、精英区标记、血量分桶。
    """
    # 每 act 的楼层范围: Act1=1-16, Act2=17-33, Act3=34-50(boss@50), Act4=51+
    act_start = {1: 1, 2: 17, 3: 34, 4: 51}
    act_boss = {1: 16, 2: 33, 3: 50, 4: 56}
    act_len = {1: 16, 2: 17, 3: 17, 4: 6}

    floor = floor or 0
    act = act or 1
    hp_pct = hp_pct or 0

    a = max(1, min(act, 4))
    boss_floor = act_boss.get(a, 56)
    start_floor = act_start.get(a, 1)
    length = act_len.get(a, 17)

    # 当前 act 内进度 (0~1)
    floor_pct_in_act = (floor - start_floor) / max(length, 1)
    floor_pct_in_act = max(0.0, min(1.0, floor_pct_in_act))

    # 距下一个 boss 的楼层数 (归一化到 0~1)
    floors_to_boss = max(0, boss_floor - floor) / max(length, 1)

    # 精英区标记: act 内 floor 5~9 通常是第一个精英区
    floor_in_act = floor - start_floor
    is_pre_elite = 1.0 if 4 <= floor_in_act <= 8 else 0.0

    # 临近 boss (3 层内)
    is_near_boss = 1.0 if (boss_floor - floor) <= 3 and floor <= boss_floor else 0.0

    # HP 分桶
    hp_below_30 = 1.0 if hp_pct < 30 else 0.0
    hp_30_50 = 1.0 if 30 <= hp_pct < 50 else 0.0
    hp_50_70 = 1.0 if 50 <= hp_pct < 70 else 0.0
    hp_above_70 = 1.0 if hp_pct >= 70 else 0.0

    return np.array([floor_pct_in_act, floors_to_boss, is_pre_elite, is_near_boss,
                     hp_below_30, hp_30_50, hp_50_70, hp_above_70], dtype=np.float32)


# ---------------------------------------------------------------------------
# 流派完成度 + Boss 上下文特征
# ---------------------------------------------------------------------------

def archetype_completion_features(card: str, deck: list[str]) -> np.ndarray:
    """流派完成度特征（13 维）。"""
    base = card.split("+")[0].strip()
    deck_bases = set(c.split("+")[0].strip() for c in deck)
    total = max(len(deck), 1)

    card_in_arch = np.array(
        [1.0 if base in _ARCHETYPE_CORE[a] else 0.0 for a in _ARCHETYPE_NAMES],
        dtype=np.float32
    )
    deck_arch_scores = np.array(
        [sum(1 for b in deck_bases if b in _ARCHETYPE_CORE[a]) / total for a in _ARCHETYPE_NAMES],
        dtype=np.float32
    )
    dominant_idx = int(deck_arch_scores.argmax())
    dominant_score = float(deck_arch_scores[dominant_idx])
    card_fits_dominant = card_in_arch[dominant_idx]

    deck_has_engine = np.array(
        [1.0 if k in deck_bases else 0.0 for k in _KEY_ENGINES],
        dtype=np.float32
    )
    card_is_engine = 1.0 if base in _KEY_ENGINES else 0.0

    card_is_draw_engine = 1.0 if base in _DRAW_ENGINES_ARCH else 0.0
    draw_payoff_in_deck = 1.0 if deck_bases & _DRAW_PAYOFFS else 0.0

    return np.concatenate([
        card_in_arch,               # 4
        [dominant_score],           # 1
        [card_fits_dominant],       # 1
        deck_has_engine,            # 4
        [card_is_engine],           # 1
        [card_is_draw_engine],      # 1
        [draw_payoff_in_deck],      # 1
    ])  # 共 13 维


def boss_context_features(act: int, act1_boss: str = "", act2_boss: str = "") -> np.ndarray:
    """Boss 上下文特征（6 维）。"""
    feat = np.zeros(6, dtype=np.float32)
    if act >= 2 and act1_boss:
        for i, b in enumerate(_ACT1_BOSSES):
            if b.lower() in act1_boss.lower():
                feat[i] = 1.0
    if act >= 3 and act2_boss:
        for i, b in enumerate(_ACT2_BOSSES):
            if b.lower() in act2_boss.lower():
                feat[3 + i] = 1.0
    return feat


def _card_feature_row(base, da_feats, tempo, extra, synergy, relic_card_syn,
                      archetype_feats, boss_feats, option_vec) -> np.ndarray:
    """Card feature vector layout — single source of truth.
    All training and inference paths must call this function so feature
    order stays consistent automatically."""
    return np.concatenate([base, da_feats, tempo, extra, synergy, relic_card_syn,
                           archetype_feats, boss_feats, option_vec])


# ---------------------------------------------------------------------------
# 模型 A: LambdaMART 排序模型 - 数据构造
# ---------------------------------------------------------------------------

def build_card_ranking_data(db: dict, vocab: dict):
    """构造卡牌排序训练数据。

    每个决策点展开为 len(offered) 行。
    标签: 2=选了且赢, 1=选了且输, 0=没选。
    返回 (X, y_relevance, groups)。
    """
    stats = db["card_decisions"]["stats"]
    card_to_idx = vocab["card_to_idx"]
    rows = []
    labels = []
    groups = []
    decisions_w = []

    for d in db["card_decisions"]["decisions"]:
        offered = d["offered"]
        if len(offered) < 2:
            continue

        base = base_features(
            d["floor"], d["act"], d["hp_pct"], d["deck_size"],
            len(d["relics"]), d["deck"], d["relics"], vocab,
            d.get("num_upgrades", 0), d.get("deck_upgrades", {})
        )
        is_boss = 1.0 if d.get("is_boss_reward", False) else 0.0
        da_feats = deck_analysis_features(d["deck"])
        tempo = temporal_features(d["floor"], d["act"], d["hp_pct"])
        boss_feats = boss_context_features(d["act"], d.get("act1_boss", ""), d.get("act2_boss", ""))

        group_size = 0
        for option in offered:
            option_vec = np.zeros(len(card_to_idx), dtype=np.float32)
            if option != "SKIP" and option in card_to_idx:
                option_vec[card_to_idx[option]] = 1

            is_skip = 1.0 if option == "SKIP" else 0.0
            s = stats.get(option, {})
            pick_rate = s.get("pick_rate", 0.0)
            win_rate_in_deck = s.get("win_rate_in_deck", 0.0)
            count_in_deck = card_count_in_deck(option, d["deck"])

            extra = np.array([is_boss, is_skip, pick_rate, win_rate_in_deck,
                              count_in_deck], dtype=np.float32)
            synergy = card_synergy_features(option, d["deck"])
            relic_card_syn = relic_card_synergy_features(option, d["deck"], d["relics"])
            archetype_feats = archetype_completion_features(option, d["deck"])
            row = _card_feature_row(base, da_feats, tempo, extra, synergy, relic_card_syn,
                                    archetype_feats, boss_feats, option_vec)
            rows.append(row)

            # 标签
            if option == d["picked"]:
                label = 2 if d["victory"] else 1
            else:
                label = 0
            labels.append(label)
            group_size += 1

        groups.append(group_size)
        decisions_w.append({"ascension_level": d.get("ascension_level", 0), "victory": d["victory"]})

    return np.array(rows), np.array(labels), np.array(groups), decisions_w


def build_boss_relic_ranking_data(db: dict, vocab: dict):
    stats = db["boss_relic_decisions"]["stats"]
    relic_to_idx = vocab["relic_to_idx"]
    rows, labels, groups = [], [], []
    decisions_w = []

    for d in db["boss_relic_decisions"]["decisions"]:
        offered = d["offered"]
        if len(offered) < 2 or not d["picked"]:
            continue

        act = d["act"]
        act_stats = stats.get(str(act), stats.get(act, {}))
        boss_floor = {1: 16, 2: 33, 3: 50}.get(act, 16)
        base = base_features(
            boss_floor, act, d["hp_pct"], d["deck_size"],
            len(d["relics_before"]), d["deck"], d["relics_before"], vocab,
            d.get("num_upgrades", 0), d.get("deck_upgrades", {})
        )
        da_feats = deck_analysis_features(d["deck"])
        tempo = temporal_features(boss_floor, act, d["hp_pct"])
        group_size = 0

        for option in offered:
            option_vec = np.zeros(len(relic_to_idx), dtype=np.float32)
            if option in relic_to_idx:
                option_vec[relic_to_idx[option]] = 1

            s = act_stats.get(option, {})
            pick_rate = s.get("pick_rate", 0.0)
            win_rate = s.get("win_rate_when_picked", 0.0)

            extra = np.array([pick_rate, win_rate], dtype=np.float32)
            rds = relic_deck_synergy_features(option, d["deck"])
            row = np.concatenate([base, da_feats, tempo, extra, rds, option_vec])
            rows.append(row)

            if option == d["picked"]:
                label = 2 if d["victory"] else 1
            else:
                label = 0
            labels.append(label)
            group_size += 1

        groups.append(group_size)
        decisions_w.append({"ascension_level": d.get("ascension_level", 0), "victory": d["victory"]})

    return np.array(rows), np.array(labels), np.array(groups), decisions_w


def build_campfire_ranking_data(db: dict, vocab: dict):
    rows, labels, groups = [], [], []
    decisions_w = []

    for d in db["campfire_decisions"]["decisions"]:
        base = base_features(
            d["floor"], d["act"], d["hp_pct"], d["deck_size"],
            len(d["relics"]), d["deck"], d["relics"], vocab,
            d.get("num_upgrades", 0), d.get("deck_upgrades", {})
        )
        da_feats = deck_analysis_features(d["deck"])
        tempo = temporal_features(d["floor"], d["act"], d["hp_pct"])

        hp_below_30 = 1.0 if d["hp_pct"] < 30 else 0.0
        hp_30_50 = 1.0 if 30 <= d["hp_pct"] < 50 else 0.0
        hp_50_70 = 1.0 if 50 <= d["hp_pct"] < 70 else 0.0
        hp_above_70 = 1.0 if d["hp_pct"] >= 70 else 0.0

        group_size = 0
        for choice_code, choice_name in [(0.0, "REST"), (1.0, "SMITH")]:
            extra = np.array([choice_code,
                              hp_below_30, hp_30_50, hp_50_70, hp_above_70],
                             dtype=np.float32)
            row = np.concatenate([base, da_feats, tempo, extra])
            rows.append(row)

            if d["choice"] == choice_name:
                label = 2 if d["victory"] else 1
            else:
                label = 0
            labels.append(label)
            group_size += 1

        groups.append(group_size)
        decisions_w.append({"ascension_level": d.get("ascension_level", 0), "victory": d["victory"]})

    return np.array(rows), np.array(labels), np.array(groups), decisions_w


def build_shop_ranking_data(db: dict, vocab: dict):
    stats = db["shop_decisions"]["stats"]
    card_to_idx = vocab["card_to_idx"]
    relic_to_idx = vocab["relic_to_idx"]
    rows, labels, groups = [], [], []
    decisions_w = []

    for d in db["shop_decisions"]["decisions"]:
        base = base_features(
            d["floor"], d.get("act", 1), d["hp_pct"], d["deck_size"],
            len(d["relics"]), d["deck"], d["relics"], vocab,
            d.get("num_upgrades", 0), d.get("deck_upgrades", {})
        )
        da_feats = deck_analysis_features(d["deck"])
        tempo = temporal_features(d["floor"], d.get("act", 1), d["hp_pct"])
        gold = d.get("gold", 0)
        purchased_set = set(d.get("purchased", []))

        all_available = (
            d.get("available_cards", []) +
            d.get("available_relics", []) +
            d.get("available_potions", [])
        )
        if not all_available:
            continue

        # 加上"REMOVE"和"不购买"选项
        items_with_skip = all_available + ["REMOVE", "不购买"]
        group_size = 0
        remove_ctx = shop_remove_context(d["deck"])
        was_purged = d.get("purged_card") is not None

        for item in items_with_skip:
            if item == "REMOVE":
                card_vec = np.zeros(len(card_to_idx), dtype=np.float32)
                relic_vec = np.zeros(len(relic_to_idx), dtype=np.float32)
                s = stats.get("REMOVE", {})
                buy_wr = s.get("win_rate_when_purchased", 0.0)
                skip_wr = s.get("win_rate_when_skipped", 0.0)
                times_purchased = s.get("times_purchased", 0)
                was_bought = 1.0 if was_purged else 0.0
                extra = np.array([gold, buy_wr, skip_wr, float(times_purchased),
                                  was_bought], dtype=np.float32)
                is_remove = 1.0
                was_picked = was_purged
            elif item == "不购买":
                card_vec = np.zeros(len(card_to_idx), dtype=np.float32)
                relic_vec = np.zeros(len(relic_to_idx), dtype=np.float32)
                extra = np.array([gold, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                is_remove = 0.0
                # "不购买"算作选了如果什么都没买且没移除
                was_picked = len(purchased_set) == 0 and not was_purged
            else:
                card_vec, relic_vec = encode_item(item, card_to_idx, relic_to_idx)
                s = stats.get(item, {})
                buy_wr = s.get("win_rate_when_purchased", 0.0)
                skip_wr = s.get("win_rate_when_skipped", 0.0)
                times_purchased = s.get("times_purchased", 0)
                was_bought = 1.0 if item in purchased_set else 0.0
                extra = np.array([gold, buy_wr, skip_wr, float(times_purchased),
                                  was_bought], dtype=np.float32)
                is_remove = 0.0
                was_picked = item in purchased_set

            remove_feats = np.array([is_remove], dtype=np.float32)
            rcs = relic_card_synergy_features(item, d["deck"], d["relics"])
            rds = relic_deck_synergy_features(item, d["deck"])
            row = np.concatenate([base, da_feats, tempo, extra, rcs, rds, remove_feats, remove_ctx, card_vec, relic_vec])
            rows.append(row)

            if was_picked:
                label = 2 if d["victory"] else 1
            else:
                label = 0
            labels.append(label)
            group_size += 1

        groups.append(group_size)
        decisions_w.append({"ascension_level": d.get("ascension_level", 0), "victory": d["victory"]})

    return np.array(rows), np.array(labels), np.array(groups), decisions_w


# ---------------------------------------------------------------------------
# 模型 B: Logistic Regression 选择模型 - 数据构造
# ---------------------------------------------------------------------------

def build_card_choice_data(db: dict, vocab: dict):
    """构造卡牌选择训练数据。标签: 1=被选中, 0=未被选中。"""
    stats = db["card_decisions"]["stats"]
    card_to_idx = vocab["card_to_idx"]
    rows, labels = [], []

    for d in db["card_decisions"]["decisions"]:
        offered = d["offered"]
        if len(offered) < 2:
            continue

        base = base_features(
            d["floor"], d["act"], d["hp_pct"], d["deck_size"],
            len(d["relics"]), d["deck"], d["relics"], vocab,
            d.get("num_upgrades", 0), d.get("deck_upgrades", {})
        )
        is_boss = 1.0 if d.get("is_boss_reward", False) else 0.0
        da_feats = deck_analysis_features(d["deck"])
        tempo = temporal_features(d["floor"], d["act"], d["hp_pct"])
        boss_feats = boss_context_features(d["act"], d.get("act1_boss", ""), d.get("act2_boss", ""))

        for option in offered:
            option_vec = np.zeros(len(card_to_idx), dtype=np.float32)
            if option != "SKIP" and option in card_to_idx:
                option_vec[card_to_idx[option]] = 1

            is_skip = 1.0 if option == "SKIP" else 0.0
            s = stats.get(option, {})
            pick_rate = s.get("pick_rate", 0.0)
            win_rate_in_deck = s.get("win_rate_in_deck", 0.0)
            count_in_deck = card_count_in_deck(option, d["deck"])

            extra = np.array([is_boss, is_skip, pick_rate, win_rate_in_deck,
                              count_in_deck], dtype=np.float32)
            synergy = card_synergy_features(option, d["deck"])
            relic_card_syn = relic_card_synergy_features(option, d["deck"], d["relics"])
            archetype_feats = archetype_completion_features(option, d["deck"])
            row = _card_feature_row(base, da_feats, tempo, extra, synergy, relic_card_syn,
                                    archetype_feats, boss_feats, option_vec)
            rows.append(row)
            labels.append(1 if option == d["picked"] else 0)

    return np.array(rows), np.array(labels)


def build_boss_relic_choice_data(db: dict, vocab: dict):
    stats = db["boss_relic_decisions"]["stats"]
    relic_to_idx = vocab["relic_to_idx"]
    rows, labels = [], []

    for d in db["boss_relic_decisions"]["decisions"]:
        offered = d["offered"]
        if len(offered) < 2 or not d["picked"]:
            continue

        act = d["act"]
        act_stats = stats.get(str(act), stats.get(act, {}))
        boss_floor = {1: 16, 2: 33, 3: 50}.get(act, 16)
        base = base_features(
            boss_floor, act, d["hp_pct"], d["deck_size"],
            len(d["relics_before"]), d["deck"], d["relics_before"], vocab,
            d.get("num_upgrades", 0), d.get("deck_upgrades", {})
        )
        da_feats = deck_analysis_features(d["deck"])
        tempo = temporal_features(boss_floor, act, d["hp_pct"])

        for option in offered:
            option_vec = np.zeros(len(relic_to_idx), dtype=np.float32)
            if option in relic_to_idx:
                option_vec[relic_to_idx[option]] = 1

            s = act_stats.get(option, {})
            pick_rate = s.get("pick_rate", 0.0)
            win_rate = s.get("win_rate_when_picked", 0.0)

            extra = np.array([pick_rate, win_rate], dtype=np.float32)
            rds = relic_deck_synergy_features(option, d["deck"])
            row = np.concatenate([base, da_feats, tempo, extra, rds, option_vec])
            rows.append(row)
            labels.append(1 if option == d["picked"] else 0)

    return np.array(rows), np.array(labels)


def build_campfire_choice_data(db: dict, vocab: dict):
    rows, labels = [], []

    for d in db["campfire_decisions"]["decisions"]:
        base = base_features(
            d["floor"], d["act"], d["hp_pct"], d["deck_size"],
            len(d["relics"]), d["deck"], d["relics"], vocab,
            d.get("num_upgrades", 0), d.get("deck_upgrades", {})
        )
        da_feats = deck_analysis_features(d["deck"])
        tempo = temporal_features(d["floor"], d["act"], d["hp_pct"])

        hp_below_30 = 1.0 if d["hp_pct"] < 30 else 0.0
        hp_30_50 = 1.0 if 30 <= d["hp_pct"] < 50 else 0.0
        hp_50_70 = 1.0 if 50 <= d["hp_pct"] < 70 else 0.0
        hp_above_70 = 1.0 if d["hp_pct"] >= 70 else 0.0

        for choice_code, choice_name in [(0.0, "REST"), (1.0, "SMITH")]:
            extra = np.array([choice_code,
                              hp_below_30, hp_30_50, hp_50_70, hp_above_70],
                             dtype=np.float32)
            row = np.concatenate([base, da_feats, tempo, extra])
            rows.append(row)
            labels.append(1 if d["choice"] == choice_name else 0)

    return np.array(rows), np.array(labels)


def build_shop_choice_data(db: dict, vocab: dict):
    stats = db["shop_decisions"]["stats"]
    card_to_idx = vocab["card_to_idx"]
    relic_to_idx = vocab["relic_to_idx"]
    rows, labels = [], []

    for d in db["shop_decisions"]["decisions"]:
        base = base_features(
            d["floor"], d.get("act", 1), d["hp_pct"], d["deck_size"],
            len(d["relics"]), d["deck"], d["relics"], vocab,
            d.get("num_upgrades", 0), d.get("deck_upgrades", {})
        )
        da_feats = deck_analysis_features(d["deck"])
        tempo = temporal_features(d["floor"], d.get("act", 1), d["hp_pct"])
        gold = d.get("gold", 0)
        purchased_set = set(d.get("purchased", []))

        all_available = (
            d.get("available_cards", []) +
            d.get("available_relics", []) +
            d.get("available_potions", [])
        )
        if not all_available:
            continue

        items_with_skip = all_available + ["REMOVE", "不购买"]
        remove_ctx = shop_remove_context(d["deck"])
        was_purged = d.get("purged_card") is not None

        for item in items_with_skip:
            if item == "REMOVE":
                card_vec = np.zeros(len(card_to_idx), dtype=np.float32)
                relic_vec = np.zeros(len(relic_to_idx), dtype=np.float32)
                s = stats.get("REMOVE", {})
                buy_wr = s.get("win_rate_when_purchased", 0.0)
                skip_wr = s.get("win_rate_when_skipped", 0.0)
                times_purchased = s.get("times_purchased", 0)
                was_bought = 1.0 if was_purged else 0.0
                extra = np.array([gold, buy_wr, skip_wr, float(times_purchased),
                                  was_bought], dtype=np.float32)
                is_remove = 1.0
                was_picked = was_purged
            elif item == "不购买":
                card_vec = np.zeros(len(card_to_idx), dtype=np.float32)
                relic_vec = np.zeros(len(relic_to_idx), dtype=np.float32)
                extra = np.array([gold, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                is_remove = 0.0
                was_picked = len(purchased_set) == 0 and not was_purged
            else:
                card_vec, relic_vec = encode_item(item, card_to_idx, relic_to_idx)
                s = stats.get(item, {})
                buy_wr = s.get("win_rate_when_purchased", 0.0)
                skip_wr = s.get("win_rate_when_skipped", 0.0)
                times_purchased = s.get("times_purchased", 0)
                was_bought = 1.0 if item in purchased_set else 0.0
                extra = np.array([gold, buy_wr, skip_wr, float(times_purchased),
                                  was_bought], dtype=np.float32)
                is_remove = 0.0
                was_picked = item in purchased_set

            remove_feats = np.array([is_remove], dtype=np.float32)
            rcs = relic_card_synergy_features(item, d["deck"], d["relics"])
            rds = relic_deck_synergy_features(item, d["deck"])
            row = np.concatenate([base, da_feats, tempo, extra, rcs, rds, remove_feats, remove_ctx, card_vec, relic_vec])
            rows.append(row)
            labels.append(1 if was_picked else 0)

    return np.array(rows), np.array(labels)


# ---------------------------------------------------------------------------
# 模型 C: CWR-Delta 条件胜率差异 - 统计计算
# ---------------------------------------------------------------------------

def _deck_size_bucket(deck_size: int) -> str:
    if deck_size <= 10:
        return "small"
    elif deck_size <= 20:
        return "medium"
    elif deck_size <= 30:
        return "large"
    return "xlarge"


def compute_card_cwr_stats(db: dict) -> dict:
    """计算卡牌条件胜率统计。

    维度: act × deck_size_bucket × is_boss
    返回: {context_key: {option: {"wins": n, "total": n}}}
    以及 global 统计 {"__global__": {option: {...}}}
    """
    cwr = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "total": 0}))
    global_stats = defaultdict(lambda: {"wins": 0, "total": 0})
    base_stats = {"wins": 0, "total": 0}  # 全局基准胜率

    for d in db["card_decisions"]["decisions"]:
        act = d["act"]
        ds_bucket = _deck_size_bucket(d["deck_size"])
        is_boss = d.get("is_boss_reward", False)
        ctx_key = f"{act}_{ds_bucket}_{is_boss}"
        picked = d["picked"]
        win = d["victory"]

        cwr[ctx_key][picked]["total"] += 1
        global_stats[picked]["total"] += 1
        base_stats["total"] += 1
        if win:
            cwr[ctx_key][picked]["wins"] += 1
            global_stats[picked]["wins"] += 1
            base_stats["wins"] += 1

    return _to_plain_dict(cwr, global_stats, base_stats)


def _to_plain_dict(cwr, global_stats, base_stats) -> dict:
    """将 defaultdict 转换为普通 dict 以支持 pickle。"""
    ctx = {}
    for k, v in cwr.items():
        ctx[k] = dict(v)
    return {
        "contextual": ctx,
        "global": dict(global_stats),
        "base": base_stats,
    }


def compute_boss_relic_cwr_stats(db: dict) -> dict:
    cwr = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "total": 0}))
    global_stats = defaultdict(lambda: {"wins": 0, "total": 0})
    base_stats = {"wins": 0, "total": 0}

    for d in db["boss_relic_decisions"]["decisions"]:
        if not d["picked"]:
            continue
        act = d["act"]
        ds_bucket = _deck_size_bucket(d["deck_size"])
        ctx_key = f"{act}_{ds_bucket}"
        picked = d["picked"]
        win = d["victory"]

        cwr[ctx_key][picked]["total"] += 1
        global_stats[picked]["total"] += 1
        base_stats["total"] += 1
        if win:
            cwr[ctx_key][picked]["wins"] += 1
            global_stats[picked]["wins"] += 1
            base_stats["wins"] += 1

    return _to_plain_dict(cwr, global_stats, base_stats)


def compute_campfire_cwr_stats(db: dict) -> dict:
    cwr = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "total": 0}))
    global_stats = defaultdict(lambda: {"wins": 0, "total": 0})
    base_stats = {"wins": 0, "total": 0}

    for d in db["campfire_decisions"]["decisions"]:
        act = d["act"]
        hp_bucket = "low" if d["hp_pct"] < 40 else ("mid" if d["hp_pct"] < 70 else "high")
        ctx_key = f"{act}_{hp_bucket}"
        choice = d["choice"]
        win = d["victory"]

        cwr[ctx_key][choice]["total"] += 1
        global_stats[choice]["total"] += 1
        base_stats["total"] += 1
        if win:
            cwr[ctx_key][choice]["wins"] += 1
            global_stats[choice]["wins"] += 1
            base_stats["wins"] += 1

    return _to_plain_dict(cwr, global_stats, base_stats)


def compute_shop_cwr_stats(db: dict) -> dict:
    cwr = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "total": 0}))
    global_stats = defaultdict(lambda: {"wins": 0, "total": 0})
    base_stats = {"wins": 0, "total": 0}

    for d in db["shop_decisions"]["decisions"]:
        act = d.get("act", 1)
        ds_bucket = _deck_size_bucket(d["deck_size"])
        ctx_key = f"{act}_{ds_bucket}"
        purchased = d.get("purchased", [])
        win = d["victory"]

        purged = d.get("purged_card") is not None

        for item in purchased:
            cwr[ctx_key][item]["total"] += 1
            global_stats[item]["total"] += 1
            base_stats["total"] += 1
            if win:
                cwr[ctx_key][item]["wins"] += 1
                global_stats[item]["wins"] += 1
                base_stats["wins"] += 1

        if purged:
            cwr[ctx_key]["REMOVE"]["total"] += 1
            global_stats["REMOVE"]["total"] += 1
            base_stats["total"] += 1
            if win:
                cwr[ctx_key]["REMOVE"]["wins"] += 1
                global_stats["REMOVE"]["wins"] += 1
                base_stats["wins"] += 1

        if not purchased and not purged:
            cwr[ctx_key]["不购买"]["total"] += 1
            global_stats["不购买"]["total"] += 1
            base_stats["total"] += 1
            if win:
                cwr[ctx_key]["不购买"]["wins"] += 1
                global_stats["不购买"]["wins"] += 1
                base_stats["wins"] += 1

    return _to_plain_dict(cwr, global_stats, base_stats)


def _bayesian_win_rate(wins: int, total: int, prior_wr: float,
                       strength: int = 5) -> float:
    """贝叶斯平滑胜率。"""
    return (wins + strength * prior_wr) / (total + strength)


def predict_cwr_delta(options: list[str], context_key: str,
                      cwr_stats: dict) -> np.ndarray:
    """计算条件胜率差异 → sigmoid 归一化。"""
    base = cwr_stats["base"]
    base_wr = base["wins"] / max(base["total"], 1)

    deltas = []
    for option in options:
        # 尝试上下文统计
        ctx = cwr_stats["contextual"].get(context_key, {})
        s = ctx.get(option)
        if s and s["total"] >= 3:
            wr = _bayesian_win_rate(s["wins"], s["total"], base_wr)
        else:
            # 回退到全局
            gs = cwr_stats["global"].get(option, {"wins": 0, "total": 0})
            if gs["total"] >= 3:
                wr = _bayesian_win_rate(gs["wins"], gs["total"], base_wr)
            else:
                wr = base_wr
        deltas.append(wr - base_wr)

    deltas = np.array(deltas, dtype=np.float64)
    # sigmoid 归一化: 将差异映射到 [0, 1]
    scores = 1.0 / (1.0 + np.exp(-10.0 * deltas))
    return scores


# ---------------------------------------------------------------------------
# 数据质量加权
# ---------------------------------------------------------------------------

def compute_sample_weights(decisions_w: list[dict], groups: np.ndarray) -> np.ndarray:
    """为每个训练样本计算质量权重。

    权重 = ascension_weight × outcome_weight
    - ascension_weight: 0.3 + 0.7 * (ascension_level / 20.0)
    - outcome_weight: 1.2 if victory else 0.6
    """
    weights = []
    d_idx = 0
    for g_size in groups:
        d = decisions_w[d_idx]
        asc = d.get("ascension_level", 0)
        victory = d.get("victory", False)
        asc_w = 0.3 + 0.7 * (asc / 20.0)
        outcome_w = 1.2 if victory else 0.6
        w = asc_w * outcome_w
        weights.extend([w] * int(g_size))
        d_idx += 1
    return np.array(weights, dtype=np.float32)


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train_ranking_model(X: np.ndarray, y: np.ndarray,
                        groups: np.ndarray, name: str,
                        sample_weight: np.ndarray | None = None):
    """训练 LGBMRanker 并用 GroupKFold 交叉验证。"""
    if lgb is None:
        print(f"  LightGBM 未安装，跳过 {name} 排序模型")
        return None

    if len(X) == 0:
        print(f"  {name}: 无训练数据，跳过")
        return None

    # 构造每行所属的 group_id
    group_ids = np.repeat(np.arange(len(groups)), groups)

    n_splits = min(5, len(groups))
    if n_splits < 2:
        print(f"  {name}: 组数不足，跳过交叉验证")
        n_splits = 2

    gkf = GroupKFold(n_splits=n_splits)
    models = []
    ndcg_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, group_ids)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 计算训练集的 group sizes
        train_groups_map = defaultdict(int)
        for idx in train_idx:
            train_groups_map[group_ids[idx]] += 1
        # 按 group_id 排序保持顺序
        train_group_order = []
        seen = set()
        for idx in train_idx:
            gid = group_ids[idx]
            if gid not in seen:
                seen.add(gid)
                train_group_order.append(gid)
        train_group_sizes = [train_groups_map[gid] for gid in train_group_order]

        # 按 group 排序训练数据
        sort_idx = np.argsort(group_ids[train_idx])
        X_train = X_train[sort_idx]
        y_train = y_train[sort_idx]
        w_train = sample_weight[train_idx][sort_idx] if sample_weight is not None else None

        model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            device="gpu",
        )
        model.fit(X_train, y_train, group=train_group_sizes,
                  sample_weight=w_train)
        models.append(model)

        # 评估: 计算 validation 组的 NDCG
        val_scores = model.predict(X_val)
        # 按组计算 NDCG
        val_groups_map = defaultdict(list)
        val_labels_map = defaultdict(list)
        for i, idx in enumerate(val_idx):
            gid = group_ids[idx]
            val_groups_map[gid].append(val_scores[i])
            val_labels_map[gid].append(y_val[i])

        fold_ndcgs = []
        for gid in val_groups_map:
            true = np.array(val_labels_map[gid])
            pred = np.array(val_groups_map[gid])
            if len(true) > 1 and true.max() > true.min():
                fold_ndcgs.append(ndcg_score([true], [pred]))
        if fold_ndcgs:
            avg_ndcg = np.mean(fold_ndcgs)
            ndcg_scores.append(avg_ndcg)

    if ndcg_scores:
        print(f"  LambdaMART {name}: 平均 NDCG={np.mean(ndcg_scores):.4f}")
    else:
        print(f"  LambdaMART {name}: 训练完成（NDCG 无法计算）")

    return models


def train_choice_model(X: np.ndarray, y: np.ndarray, name: str,
                       sample_weight: np.ndarray | None = None):
    """训练 Logistic Regression 选择模型。"""
    if len(X) == 0:
        print(f"  {name}: 无训练数据，跳过")
        return None

    if len(np.unique(y)) < 2:
        print(f"  {name}: 标签只有一个类别，跳过")
        return None

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )),
    ])
    fit_params = {}
    if sample_weight is not None:
        fit_params["logreg__sample_weight"] = sample_weight
    pipe.fit(X, y, **fit_params)

    train_proba = pipe.predict_proba(X)[:, 1]
    train_acc = np.mean((train_proba > 0.5).astype(int) == y)
    print(f"  LogReg {name}: 训练集 Accuracy={train_acc:.4f}")

    return pipe


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------

def _safe_predict_v1(v1_model_dict: dict, X: np.ndarray) -> dict[str, np.ndarray]:
    """安全调用 V1 模型预测，词表维度不匹配时静默跳过。"""
    if not v1_model_dict:
        return {}
    try:
        return predict_v1(v1_model_dict, X)
    except (ValueError, Exception):
        return {}


def predict_ranking(models: list, X: np.ndarray) -> np.ndarray:
    """排序模型推理 → softmax 归一化为概率。"""
    if not models:
        return np.zeros(len(X))
    scores = np.mean([m.predict(X) for m in models], axis=0)
    # softmax
    exp_s = np.exp(scores - scores.max())
    return exp_s / exp_s.sum()


def predict_choice(pipe, X: np.ndarray) -> np.ndarray:
    """选择模型推理 → P(被选中)。"""
    if pipe is None:
        return np.zeros(len(X))
    return pipe.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# 统一推理特征（与 v1 兼容的特征 + 新增特征）
# ---------------------------------------------------------------------------

def card_inference_features_v2(floor: int, act: int, hp_pct: int,
                               deck: list[str], relics: list[str],
                               options: list[str], stats: dict,
                               vocab: dict, num_upgrades: int = 0,
                               deck_upgrades: dict | None = None,
                               act1_boss: str = "", act2_boss: str = "") -> np.ndarray:
    """V2 卡牌推理特征：base + deck_analysis + extra + option_vec + archetype + boss。"""
    card_to_idx = vocab["card_to_idx"]
    base = base_features(floor, act, hp_pct, len(deck), len(relics), deck, relics, vocab,
                         num_upgrades, deck_upgrades)
    is_boss = 1.0 if floor in (16, 33) else 0.0
    da_feats = deck_analysis_features(deck)
    tempo = temporal_features(floor, act, hp_pct)
    boss_feats = boss_context_features(act, act1_boss, act2_boss)
    rows = []

    for option in options:
        option_vec = np.zeros(len(card_to_idx), dtype=np.float32)
        if option != "SKIP" and option in card_to_idx:
            option_vec[card_to_idx[option]] = 1

        is_skip = 1.0 if option == "SKIP" else 0.0
        s = stats.get(option, {})
        pick_rate = s.get("pick_rate", 0.0)
        win_rate_in_deck = s.get("win_rate_in_deck", 0.0)
        count_in_deck = card_count_in_deck(option, deck)

        extra = np.array([is_boss, is_skip, pick_rate, win_rate_in_deck,
                          count_in_deck], dtype=np.float32)
        synergy = card_synergy_features(option, deck)
        relic_card_syn = relic_card_synergy_features(option, deck, relics)
        archetype_feats = archetype_completion_features(option, deck)
        rows.append(_card_feature_row(base, da_feats, tempo, extra, synergy, relic_card_syn,
                                      archetype_feats, boss_feats, option_vec))

    return np.array(rows)


def boss_relic_inference_features_v2(act: int, hp_pct: int,
                                     deck: list[str], relics: list[str],
                                     options: list[str], stats: dict,
                                     vocab: dict, num_upgrades: int = 0,
                                     deck_upgrades: dict | None = None) -> np.ndarray:
    relic_to_idx = vocab["relic_to_idx"]
    act_stats = stats.get(str(act), stats.get(act, {}))
    boss_floor = {1: 16, 2: 33, 3: 50}.get(act, 16)
    base = base_features(boss_floor, act, hp_pct, len(deck), len(relics), deck, relics, vocab,
                         num_upgrades, deck_upgrades)
    da_feats = deck_analysis_features(deck)
    tempo = temporal_features(boss_floor, act, hp_pct)
    rows = []

    for option in options:
        option_vec = np.zeros(len(relic_to_idx), dtype=np.float32)
        if option in relic_to_idx:
            option_vec[relic_to_idx[option]] = 1

        s = act_stats.get(option, {})
        pick_rate = s.get("pick_rate", 0.0)
        win_rate = s.get("win_rate_when_picked", 0.0)

        extra = np.array([pick_rate, win_rate], dtype=np.float32)
        rds = relic_deck_synergy_features(option, deck)
        rows.append(np.concatenate([base, da_feats, tempo, extra, rds, option_vec]))

    return np.array(rows)


def campfire_inference_features_v2(floor: int, act: int, hp_pct: int,
                                   deck: list[str], relics: list[str],
                                   vocab: dict, num_upgrades: int = 0,
                                   deck_upgrades: dict | None = None) -> np.ndarray:
    base = base_features(floor, act, hp_pct, len(deck), len(relics), deck, relics, vocab,
                         num_upgrades, deck_upgrades)
    da_feats = deck_analysis_features(deck)
    tempo = temporal_features(floor, act, hp_pct)

    hp_below_30 = 1.0 if hp_pct < 30 else 0.0
    hp_30_50 = 1.0 if 30 <= hp_pct < 50 else 0.0
    hp_50_70 = 1.0 if 50 <= hp_pct < 70 else 0.0
    hp_above_70 = 1.0 if hp_pct >= 70 else 0.0
    rows = []

    for choice_code in [0.0, 1.0]:
        extra = np.array([choice_code,
                          hp_below_30, hp_30_50, hp_50_70, hp_above_70],
                         dtype=np.float32)
        rows.append(np.concatenate([base, da_feats, tempo, extra]))

    return np.array(rows)


def shop_inference_features_v2(floor: int, act: int, hp_pct: int, gold: int,
                               deck: list[str], relics: list[str],
                               items: list[str], stats: dict,
                               vocab: dict, num_upgrades: int = 0,
                               deck_upgrades: dict | None = None) -> np.ndarray:
    card_to_idx = vocab["card_to_idx"]
    relic_to_idx = vocab["relic_to_idx"]
    base = base_features(floor, act, hp_pct, len(deck), len(relics), deck, relics, vocab,
                         num_upgrades, deck_upgrades)
    da_feats = deck_analysis_features(deck)
    tempo = temporal_features(floor, act, hp_pct)
    remove_ctx = shop_remove_context(deck)
    rows = []

    for item in items:
        card_vec, relic_vec = encode_item(item, card_to_idx, relic_to_idx)
        s = stats.get(item, {})
        buy_wr = s.get("win_rate_when_purchased", 0.0)
        skip_wr = s.get("win_rate_when_skipped", 0.0)
        times_purchased = s.get("times_purchased", 0)
        was_bought = 1.0
        extra = np.array([gold, buy_wr, skip_wr, float(times_purchased), was_bought],
                         dtype=np.float32)
        rcs = relic_card_synergy_features(item, deck, relics)
        rds = relic_deck_synergy_features(item, deck)
        remove_feats = np.array([0.0], dtype=np.float32)
        rows.append(np.concatenate([base, da_feats, tempo, extra, rcs, rds, remove_feats, remove_ctx, card_vec, relic_vec]))

    # REMOVE（移除卡牌）
    card_vec = np.zeros(len(card_to_idx), dtype=np.float32)
    relic_vec = np.zeros(len(relic_to_idx), dtype=np.float32)
    s = stats.get("REMOVE", {})
    buy_wr = s.get("win_rate_when_purchased", 0.0)
    skip_wr = s.get("win_rate_when_skipped", 0.0)
    times_purchased = s.get("times_purchased", 0)
    extra = np.array([gold, buy_wr, skip_wr, float(times_purchased), 1.0], dtype=np.float32)
    rcs = np.zeros(len(_RELIC_CARD_SYNERGIES), dtype=np.float32)
    rds = np.zeros(len(_RELIC_CARD_SYNERGIES), dtype=np.float32)
    remove_feats = np.array([1.0], dtype=np.float32)
    rows.append(np.concatenate([base, da_feats, tempo, extra, rcs, rds, remove_feats, remove_ctx, card_vec, relic_vec]))

    # 不购买
    card_vec = np.zeros(len(card_to_idx), dtype=np.float32)
    relic_vec = np.zeros(len(relic_to_idx), dtype=np.float32)
    extra = np.array([gold, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    rcs = np.zeros(len(_RELIC_CARD_SYNERGIES), dtype=np.float32)
    rds = np.zeros(len(_RELIC_CARD_SYNERGIES), dtype=np.float32)
    remove_feats = np.array([0.0], dtype=np.float32)
    rows.append(np.concatenate([base, da_feats, tempo, extra, rcs, rds, remove_feats, remove_ctx, card_vec, relic_vec]))

    return np.array(rows)


# ---------------------------------------------------------------------------
# 综合推理入口
# ---------------------------------------------------------------------------

def _apply_duplicate_power_penalty(options: list[str], deck: list[str],
                                    preds: dict[str, np.ndarray],
                                    penalty: float = 0.5) -> dict[str, np.ndarray]:
    """对卡组中已有的能力牌选项施加惩罚，降低其推荐分数。"""
    deck_bases = set(c.split("+")[0].strip() for c in deck)
    for i, option in enumerate(options):
        base = option.split("+")[0].strip()
        if base in _POWER_CARDS and base in deck_bases:
            for eng in preds:
                preds[eng][i] *= penalty
    return preds


def predict_all_card(options: list[str], floor: int, act: int, hp_pct: int,
                     deck: list[str], relics: list[str],
                     db: dict, vocab: dict,
                     v1_models: dict, v2_models: dict,
                     num_upgrades: int = 0,
                     deck_upgrades: dict | None = None,
                     act1_boss: str = "", act2_boss: str = "") -> dict[str, np.ndarray]:
    """所有模型的卡牌预测结果。"""
    stats = db["card_decisions"]["stats"]
    preds = {}

    # V1 模型
    X_v1 = card_inference_features_v1(floor, act, hp_pct, deck, relics,
                                      options, stats, vocab,
                                      num_upgrades, deck_upgrades)
    preds.update(_safe_predict_v1(v1_models.get("card", {}), X_v1))

    # V2 特征
    X_v2 = card_inference_features_v2(floor, act, hp_pct, deck, relics,
                                      options, stats, vocab,
                                      num_upgrades, deck_upgrades,
                                      act1_boss, act2_boss)

    # LambdaMART
    if "card_lambdamart" in v2_models:
        preds["lambdamart"] = predict_ranking(v2_models["card_lambdamart"], X_v2)

    # LogReg
    if "card_logreg" in v2_models:
        preds["logreg"] = predict_choice(v2_models["card_logreg"], X_v2)

    # CWR-Delta
    if "card_cwr" in v2_models:
        is_boss = floor in (16, 33)
        ds_bucket = _deck_size_bucket(len(deck))
        ctx_key = f"{act}_{ds_bucket}_{is_boss}"
        preds["cwr_delta"] = predict_cwr_delta(options, ctx_key,
                                                v2_models["card_cwr"])

    # 对已有的能力牌施加重复惩罚
    preds = _apply_duplicate_power_penalty(options, deck, preds)

    return preds


def predict_all_campfire(floor: int, act: int, hp_pct: int,
                         deck: list[str], relics: list[str],
                         db: dict, vocab: dict,
                         v1_models: dict, v2_models: dict,
                         num_upgrades: int = 0,
                         deck_upgrades: dict | None = None) -> dict[str, np.ndarray]:
    preds = {}

    X_v1 = campfire_inference_features(floor, act, hp_pct, deck, relics, vocab,
                                       num_upgrades, deck_upgrades)
    preds.update(_safe_predict_v1(v1_models.get("campfire", {}), X_v1))

    X_v2 = campfire_inference_features_v2(floor, act, hp_pct, deck, relics, vocab,
                                          num_upgrades, deck_upgrades)

    if "campfire_lambdamart" in v2_models:
        preds["lambdamart"] = predict_ranking(v2_models["campfire_lambdamart"], X_v2)

    if "campfire_logreg" in v2_models:
        preds["logreg"] = predict_choice(v2_models["campfire_logreg"], X_v2)

    if "campfire_cwr" in v2_models:
        hp_bucket = "low" if hp_pct < 40 else ("mid" if hp_pct < 70 else "high")
        ctx_key = f"{act}_{hp_bucket}"
        preds["cwr_delta"] = predict_cwr_delta(["REST", "SMITH"], ctx_key,
                                                v2_models["campfire_cwr"])

    return preds


def predict_all_boss_relic(options: list[str], act: int, hp_pct: int,
                           deck: list[str], relics: list[str],
                           db: dict, vocab: dict,
                           v1_models: dict, v2_models: dict,
                           num_upgrades: int = 0,
                           deck_upgrades: dict | None = None) -> dict[str, np.ndarray]:
    preds = {}
    stats = db["boss_relic_decisions"]["stats"]

    X_v1 = boss_relic_inference_features_v1(act, hp_pct, deck, relics,
                                            options, stats, vocab,
                                            num_upgrades, deck_upgrades)
    preds.update(_safe_predict_v1(v1_models.get("boss_relic", {}), X_v1))

    X_v2 = boss_relic_inference_features_v2(act, hp_pct, deck, relics,
                                            options, stats, vocab,
                                            num_upgrades, deck_upgrades)

    if "boss_relic_lambdamart" in v2_models:
        preds["lambdamart"] = predict_ranking(v2_models["boss_relic_lambdamart"], X_v2)

    if "boss_relic_logreg" in v2_models:
        preds["logreg"] = predict_choice(v2_models["boss_relic_logreg"], X_v2)

    if "boss_relic_cwr" in v2_models:
        ds_bucket = _deck_size_bucket(len(deck))
        ctx_key = f"{act}_{ds_bucket}"
        preds["cwr_delta"] = predict_cwr_delta(options, ctx_key,
                                                v2_models["boss_relic_cwr"])

    return preds


def predict_all_shop(option_labels: list[str], floor: int, act: int, hp_pct: int,
                     gold: int, deck: list[str], relics: list[str],
                     items: list[str],
                     db: dict, vocab: dict,
                     v1_models: dict, v2_models: dict,
                     num_upgrades: int = 0,
                     deck_upgrades: dict | None = None) -> dict[str, np.ndarray]:
    preds = {}
    stats = db["shop_decisions"]["stats"]

    X_v1 = shop_inference_features_v1(floor, act, hp_pct, gold, deck, relics,
                                      items, stats, vocab,
                                      num_upgrades, deck_upgrades)
    v1_preds = _safe_predict_v1(v1_models.get("shop", {}), X_v1)
    # V1 特征只有 items+"不购买"(N+1)，V2 多了"移除卡牌"(N+2)
    # 在倒数第1位(不购买之前)插入最低分，对齐长度
    for k, v in v1_preds.items():
        if len(v) == len(items) + 1:
            v1_preds[k] = np.insert(v, len(items), v.min())
    preds.update(v1_preds)

    X_v2 = shop_inference_features_v2(floor, act, hp_pct, gold, deck, relics,
                                      items, stats, vocab,
                                      num_upgrades, deck_upgrades)

    if "shop_lambdamart" in v2_models:
        preds["lambdamart"] = predict_ranking(v2_models["shop_lambdamart"], X_v2)

    if "shop_logreg" in v2_models:
        preds["logreg"] = predict_choice(v2_models["shop_logreg"], X_v2)

    if "shop_cwr" in v2_models:
        ds_bucket = _deck_size_bucket(len(deck))
        ctx_key = f"{act}_{ds_bucket}"
        preds["cwr_delta"] = predict_cwr_delta(option_labels, ctx_key,
                                                v2_models["shop_cwr"])

    return preds


# ---------------------------------------------------------------------------
# 保存 / 加载 V2 模型
# ---------------------------------------------------------------------------

def save_v2_model(obj, name: str):
    import time
    path = MODEL_DIR / f"{name}.pkl"
    tmp_path = MODEL_DIR / f"{name}.pkl.tmp"
    for attempt in range(5):
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(obj, f)
            if path.exists():
                path.unlink()
            tmp_path.rename(path)
            print(f"  已保存: {path}")
            return
        except OSError as e:
            if attempt < 4:
                print(f"  保存 {name} 失败 (尝试 {attempt+1}/5): {e}, 重试中...")
                time.sleep(1)
            else:
                raise


def load_v2_models() -> dict:
    """加载所有 V2 模型。缺失模型会被跳过。"""
    result = {}
    model_names = [
        "card_lambdamart", "card_logreg", "card_cwr",
        "boss_relic_lambdamart", "boss_relic_logreg", "boss_relic_cwr",
        "campfire_lambdamart", "campfire_logreg", "campfire_cwr",
        "shop_lambdamart", "shop_logreg", "shop_cwr",
    ]
    for name in model_names:
        path = MODEL_DIR / f"{name}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                result[name] = pickle.load(f)
    return result


# ---------------------------------------------------------------------------
# 训练入口
# ---------------------------------------------------------------------------

def run_training_v2():
    print("加载数据库...")
    db = load_db()

    print("构建词表...")
    vocab = build_vocabularies(db)
    print(f"  卡牌词表: {len(vocab['card_vocab'])} 张")
    print(f"  遗物词表: {len(vocab['relic_vocab'])} 个")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 保存词表（与 v1 兼容）
    vocab_path = MODEL_DIR / "vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    # === 卡牌 ===
    print("\n=== 训练卡牌 V2 模型 ===")

    print("  构造排序数据...")
    X_rank, y_rank, groups, dw_card = build_card_ranking_data(db, vocab)
    sw_card = compute_sample_weights(dw_card, groups)
    print(f"  排序数据: {len(y_rank)} 行, {len(groups)} 组")
    print(f"  样本权重范围: [{sw_card.min():.3f}, {sw_card.max():.3f}]")
    card_ranker = train_ranking_model(X_rank, y_rank, groups, "card",
                                      sample_weight=sw_card)
    if card_ranker:
        save_v2_model(card_ranker, "card_lambdamart")

    print("  构造选择数据...")
    X_choice, y_choice = build_card_choice_data(db, vocab)
    print(f"  选择数据: {len(y_choice)} 行, 正样本: {y_choice.sum()}")
    card_logreg = train_choice_model(X_choice, y_choice, "card",
                                     sample_weight=sw_card)
    if card_logreg:
        save_v2_model(card_logreg, "card_logreg")

    print("  计算 CWR 统计...")
    card_cwr = compute_card_cwr_stats(db)
    save_v2_model(card_cwr, "card_cwr")

    # === Boss 遗物 ===
    print("\n=== 训练 Boss 遗物 V2 模型 ===")

    X_rank, y_rank, groups, dw_boss = build_boss_relic_ranking_data(db, vocab)
    sw_boss = compute_sample_weights(dw_boss, groups)
    print(f"  排序数据: {len(y_rank)} 行, {len(groups)} 组")
    print(f"  样本权重范围: [{sw_boss.min():.3f}, {sw_boss.max():.3f}]")
    boss_ranker = train_ranking_model(X_rank, y_rank, groups, "boss_relic",
                                      sample_weight=sw_boss)
    if boss_ranker:
        save_v2_model(boss_ranker, "boss_relic_lambdamart")

    X_choice, y_choice = build_boss_relic_choice_data(db, vocab)
    print(f"  选择数据: {len(y_choice)} 行, 正样本: {y_choice.sum()}")
    boss_logreg = train_choice_model(X_choice, y_choice, "boss_relic",
                                     sample_weight=sw_boss)
    if boss_logreg:
        save_v2_model(boss_logreg, "boss_relic_logreg")

    boss_cwr = compute_boss_relic_cwr_stats(db)
    save_v2_model(boss_cwr, "boss_relic_cwr")

    # === 篝火 ===
    print("\n=== 训练篝火 V2 模型 ===")

    X_rank, y_rank, groups, dw_camp = build_campfire_ranking_data(db, vocab)
    sw_camp = compute_sample_weights(dw_camp, groups)
    print(f"  排序数据: {len(y_rank)} 行, {len(groups)} 组")
    print(f"  样本权重范围: [{sw_camp.min():.3f}, {sw_camp.max():.3f}]")
    camp_ranker = train_ranking_model(X_rank, y_rank, groups, "campfire",
                                      sample_weight=sw_camp)
    if camp_ranker:
        save_v2_model(camp_ranker, "campfire_lambdamart")

    X_choice, y_choice = build_campfire_choice_data(db, vocab)
    print(f"  选择数据: {len(y_choice)} 行, 正样本: {y_choice.sum()}")
    camp_logreg = train_choice_model(X_choice, y_choice, "campfire",
                                     sample_weight=sw_camp)
    if camp_logreg:
        save_v2_model(camp_logreg, "campfire_logreg")

    camp_cwr = compute_campfire_cwr_stats(db)
    save_v2_model(camp_cwr, "campfire_cwr")

    # === 商店 ===
    print("\n=== 训练商店 V2 模型 ===")

    X_rank, y_rank, groups, dw_shop = build_shop_ranking_data(db, vocab)
    sw_shop = compute_sample_weights(dw_shop, groups)
    print(f"  排序数据: {len(y_rank)} 行, {len(groups)} 组")
    print(f"  样本权重范围: [{sw_shop.min():.3f}, {sw_shop.max():.3f}]")
    shop_ranker = train_ranking_model(X_rank, y_rank, groups, "shop",
                                      sample_weight=sw_shop)
    if shop_ranker:
        save_v2_model(shop_ranker, "shop_lambdamart")

    X_choice, y_choice = build_shop_choice_data(db, vocab)
    print(f"  选择数据: {len(y_choice)} 行, 正样本: {y_choice.sum()}")
    shop_logreg = train_choice_model(X_choice, y_choice, "shop",
                                     sample_weight=sw_shop)
    if shop_logreg:
        save_v2_model(shop_logreg, "shop_logreg")

    shop_cwr = compute_shop_cwr_stats(db)
    save_v2_model(shop_cwr, "shop_cwr")

    # === 行为对齐评估 ===
    import warnings
    print("\n=== 行为对齐评估 ===")
    v2_models = load_v2_models()

    for dtype, build_fn, dw_list, label in [
        ("card", build_card_ranking_data, dw_card, "卡牌"),
        ("boss_relic", build_boss_relic_ranking_data, dw_boss, "Boss遗物"),
        ("campfire", build_campfire_ranking_data, dw_camp, "篝火"),
        ("shop", build_shop_ranking_data, dw_shop, "商店"),
    ]:
        lmart_key = f"{dtype}_lambdamart"
        if lmart_key not in v2_models:
            continue

        X_eval, y_eval, g_eval, dw_eval = build_fn(db, vocab)
        models_list = v2_models[lmart_key]

        # 批量预测所有样本，避免逐组调用产生大量警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            all_scores = np.mean([m.predict(X_eval) for m in models_list], axis=0)

        top1_all, top3_all, mrr_all = 0, 0, 0.0
        top1_a20, top3_a20, mrr_a20 = 0, 0, 0.0
        n_all, n_a20 = 0, 0
        start = 0

        for i, g in enumerate(g_eval):
            g = int(g)
            y_group = y_eval[start:start + g]
            scores_group = all_scores[start:start + g]
            start += g

            picked_idx = np.argmax(y_group)
            if y_group[picked_idx] == 0:
                continue

            ranked = np.argsort(-scores_group)
            rank_of_picked = np.where(ranked == picked_idx)[0][0] + 1

            n_all += 1
            if rank_of_picked == 1:
                top1_all += 1
            if rank_of_picked <= 3:
                top3_all += 1
            mrr_all += 1.0 / rank_of_picked

            if dw_eval[i].get("ascension_level", 0) == 20:
                n_a20 += 1
                if rank_of_picked == 1:
                    top1_a20 += 1
                if rank_of_picked <= 3:
                    top3_a20 += 1
                mrr_a20 += 1.0 / rank_of_picked

        if n_all > 0:
            print(f"\n  [{label}] 全量 (n={n_all}):")
            print(f"    Top-1 Accuracy: {top1_all/n_all:.4f}")
            print(f"    Top-3 Accuracy: {top3_all/n_all:.4f}")
            print(f"    MRR:            {mrr_all/n_all:.4f}")
        if n_a20 > 0:
            print(f"  [{label}] A20 子集 (n={n_a20}):")
            print(f"    Top-1 Accuracy: {top1_a20/n_a20:.4f}")
            print(f"    Top-3 Accuracy: {top3_a20/n_a20:.4f}")
            print(f"    MRR:            {mrr_a20/n_a20:.4f}")

    print("\nV2 训练完成！")


# ---------------------------------------------------------------------------
# CLI 推理
# ---------------------------------------------------------------------------

def _display_width(s: str) -> int:
    import unicodedata
    w = 0
    for ch in s:
        cat = unicodedata.east_asian_width(ch)
        w += 2 if cat in ("W", "F") else 1
    return w


def _pad_right(s: str, width: int) -> str:
    return s + " " * (width - _display_width(s))


_NAME_COL = 24
_ENGINE_ORDER_CLI = ["xgboost", "lightgbm", "lambdamart", "logreg", "cwr_delta"]
_LABEL_MAP = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "lambdamart": "LambdaMR",
    "logreg": "LogReg",
    "cwr_delta": "CWR",
}


def _print_all_predictions(options: list[str], preds: dict[str, np.ndarray]):
    """打印所有模型的预测结果 + 综合排名。"""
    engines = [e for e in _ENGINE_ORDER_CLI if e in preds]
    if not engines:
        engines = sorted(preds.keys())

    # 表头
    header = _pad_right("候选项", _NAME_COL)
    for eng in engines:
        label = _LABEL_MAP.get(eng, eng)
        header += f"  {label:>8s}"
    header += f"  {'综合':>6s}  推荐"
    print(header)
    print("-" * _display_width(header))

    n = len(options)

    # 计算 Borda count (排名平均)
    rank_sum = np.zeros(n)
    for eng in engines:
        probs = preds[eng]
        ranks = n - np.argsort(np.argsort(probs))
        rank_sum += ranks
    avg_rank = rank_sum / len(engines) if engines else rank_sum

    best_idx = int(avg_rank.argmin())

    # 按综合排名排序
    order = np.argsort(avg_rank)

    for i in order:
        row = _pad_right(options[i], _NAME_COL)
        for eng in engines:
            row += f"  {preds[eng][i]*100:>7.1f}%"
        row += f"  {avg_rank[i]:>6.1f}"
        if i == best_idx:
            row += "  ★"
        print(row)


def infer_card_v2(args):
    db = load_db()
    vocab = load_vocab()
    deck = parse_deck(args.deck)
    relics = parse_list(args.relics)
    options = parse_list(args.options)
    if "SKIP" not in options:
        options.append("SKIP")

    hp_pct = round(args.hp / args.max_hp * 100) if args.max_hp > 0 else 0

    v1_models = {}
    for name in ["card"]:
        try:
            v1_models[name] = load_v1_models(name)
        except SystemExit:
            pass

    v2_models = load_v2_models()

    preds = predict_all_card(options, args.floor, args.act, hp_pct,
                             deck, relics, db, vocab, v1_models, v2_models)

    print(f"\n=== 观者 ML 决策建议 (卡牌选择) ===")
    print(f"Floor {args.floor} Act {args.act} | HP: {args.hp}/{args.max_hp} ({hp_pct}%)")
    _print_all_predictions(options, preds)


def infer_campfire_v2(args):
    db = load_db()
    vocab = load_vocab()
    deck = parse_deck(args.deck)
    relics = parse_list(args.relics)
    hp_pct = round(args.hp / args.max_hp * 100) if args.max_hp > 0 else 0

    v1_models = {}
    try:
        v1_models["campfire"] = load_v1_models("campfire")
    except SystemExit:
        pass

    v2_models = load_v2_models()

    preds = predict_all_campfire(args.floor, args.act, hp_pct,
                                 deck, relics, db, vocab, v1_models, v2_models)

    print(f"\n=== 观者 ML 决策建议 (篝火决策) ===")
    print(f"Floor {args.floor} Act {args.act} | HP: {args.hp}/{args.max_hp} ({hp_pct}%)")
    _print_all_predictions(["REST", "SMITH"], preds)


def infer_boss_relic_v2(args):
    db = load_db()
    vocab = load_vocab()
    deck = parse_deck(args.deck)
    relics = parse_list(args.relics)
    options = parse_list(args.options)
    hp_pct = round(args.hp / args.max_hp * 100) if args.max_hp > 0 else 0

    v1_models = {}
    try:
        v1_models["boss_relic"] = load_v1_models("boss_relic")
    except SystemExit:
        pass

    v2_models = load_v2_models()

    preds = predict_all_boss_relic(options, args.act, hp_pct,
                                   deck, relics, db, vocab, v1_models, v2_models)

    print(f"\n=== 观者 ML 决策建议 (Boss 遗物选择) ===")
    print(f"Act {args.act} | HP: {args.hp}/{args.max_hp} ({hp_pct}%)")
    _print_all_predictions(options, preds)


def infer_shop_v2(args):
    db = load_db()
    vocab = load_vocab()
    deck = parse_deck(args.deck)
    relics = parse_list(args.relics)
    avail_cards = parse_list(args.cards)
    avail_relics = parse_list(args.shop_relics)
    avail_potions = parse_list(args.potions)
    hp_pct = round(args.hp / args.max_hp * 100) if args.max_hp > 0 else 0

    v1_models = {}
    try:
        v1_models["shop"] = load_v1_models("shop")
    except SystemExit:
        pass

    v2_models = load_v2_models()

    all_items = avail_cards + avail_relics + avail_potions
    option_labels = all_items + ["移除卡牌", "不购买"]

    preds = predict_all_shop(option_labels, args.floor, args.act, hp_pct,
                             args.gold, deck, relics, all_items,
                             db, vocab, v1_models, v2_models)

    print(f"\n=== 观者 ML 决策建议 (商店) ===")
    print(f"Floor {args.floor} Act {args.act} | HP: {args.hp}/{args.max_hp} ({hp_pct}%) | Gold: {args.gold}")
    _print_all_predictions(option_labels, preds)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="观者 ML 决策顾问 V2 (LambdaMART / LogReg / CWR-Delta)",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    subs.add_parser("train", help="训练所有 V2 模型")

    cp = subs.add_parser("card", help="卡牌奖励决策")
    cp.add_argument("--floor", type=int, required=True)
    cp.add_argument("--act", type=int, required=True)
    cp.add_argument("--hp", type=int, required=True)
    cp.add_argument("--max-hp", type=int, required=True, dest="max_hp")
    cp.add_argument("--relics", default="")
    cp.add_argument("--deck", default="")
    cp.add_argument("--options", required=True)

    fp = subs.add_parser("campfire", help="篝火决策")
    fp.add_argument("--floor", type=int, required=True)
    fp.add_argument("--act", type=int, required=True)
    fp.add_argument("--hp", type=int, required=True)
    fp.add_argument("--max-hp", type=int, required=True, dest="max_hp")
    fp.add_argument("--relics", default="")
    fp.add_argument("--deck", default="")

    bp = subs.add_parser("boss-relic", help="Boss 遗物选择")
    bp.add_argument("--act", type=int, required=True)
    bp.add_argument("--hp", type=int, required=True)
    bp.add_argument("--max-hp", type=int, required=True, dest="max_hp")
    bp.add_argument("--relics", default="")
    bp.add_argument("--deck", default="")
    bp.add_argument("--options", required=True)

    sp = subs.add_parser("shop", help="商店购买建议")
    sp.add_argument("--floor", type=int, required=True)
    sp.add_argument("--act", type=int, required=True)
    sp.add_argument("--hp", type=int, required=True)
    sp.add_argument("--max-hp", type=int, required=True, dest="max_hp")
    sp.add_argument("--gold", type=int, required=True)
    sp.add_argument("--relics", default="")
    sp.add_argument("--deck", default="")
    sp.add_argument("--cards", default="")
    sp.add_argument("--shop-relics", default="", dest="shop_relics")
    sp.add_argument("--potions", default="")

    args = parser.parse_args()

    if args.command == "train":
        run_training_v2()
    elif args.command == "card":
        infer_card_v2(args)
    elif args.command == "campfire":
        infer_campfire_v2(args)
    elif args.command == "boss-relic":
        infer_boss_relic_v2(args)
    elif args.command == "shop":
        infer_shop_v2(args)


if __name__ == "__main__":
    main()
