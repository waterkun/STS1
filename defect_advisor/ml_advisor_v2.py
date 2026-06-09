#!/usr/bin/env python3
"""
Defect ML Advisor V2 - 三种新模型解决区分度不足问题

模型 A: LambdaMART 排序模型 (LGBMRanker)
模型 B: Logistic Regression 选择模型
模型 C: CWR-Delta 条件胜率差异 (纯统计)

机器人特有机制: 法球系统（霜冻/闪电/黑暗/等离子）、专注、爪牌流

Usage:
  python -m defect_advisor.ml_advisor_v2 train
  python -m defect_advisor.ml_advisor_v2 card \
    --floor 8 --act 1 --hp 45 --max-hp 75 \
    --deck "Strike_B x4,Defend_B x4,Zap,Dualcast" \
    --options "Claw,Cold Snap,Compile Driver"
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

from defect_advisor.ml_advisor import (
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
# 机器人卡牌分类
# ---------------------------------------------------------------------------

_ATTACK_CARDS = {
    "Strike_B", "Beam Cell", "Claw", "Cold Snap", "Compile Driver",
    "Go for the Kill", "Rebound", "Rip and Tear", "Skim", "Streamline",
    "Thunder Strike", "Blizzard", "Icicle", "Meteor Strike", "Scrape",
    "Sunder", "All for One", "Ball Lightning", "Barrage", "Hyperbeam",
    "Multi-Cast", "Rainbow",
}
_SKILL_CARDS = {
    "Defend_B", "Charge Battery", "Coolheaded", "Dualcast", "Hologram",
    "Leap", "Recursion", "Stack", "Steam Barrier", "Tempest", "Turbo",
    "White Noise", "Aggregate", "Auto-Shields", "Buffer", "Darkness",
    "Defragment", "Equilibrium", "Force Field", "Glacier", "Loop",
    "Overclock", "Recycle", "Reinforced Body", "Reprogram", "Seek",
    "Self-Repair", "Static Discharge", "Storm",
}
_POWER_CARDS = {
    "Amplify", "Biased Cognition", "Capacitor", "Consume", "Creative AI",
    "Electrodynamics", "Echo Form", "FTL", "Fusion", "Hello World",
    "Machine Learning", "Chill",
}


def _card_type(name: str) -> str:
    base = name.split("+")[0].strip()
    if base in _ATTACK_CARDS:
        return "attack"
    if base in _SKILL_CARDS:
        return "skill"
    if base in _POWER_CARDS:
        return "power"
    return "attack"


# ---------------------------------------------------------------------------
# 机制关键词分组（机器人专属）
# ---------------------------------------------------------------------------

# 法球生成
_FROST_GENERATORS = {"Cold Snap", "Coolheaded", "Glacier", "FTL", "Equilibrium",
                     "Loop", "Dualcast", "Recursion"}
_FROST_PAYOFF = {"Blizzard", "Coolheaded", "Equilibrium", "Glacier"}

_LIGHTNING_GENERATORS = {"Zap", "Thunder Strike", "Charge Battery", "Ball Lightning",
                          "Electrodynamics", "Static Discharge", "Storm", "Tempest"}
_LIGHTNING_PAYOFF = {"Electrodynamics", "Storm", "Static Discharge", "Thunder Strike"}

_DARK_GENERATORS = {"Darkness", "Consume", "Doom and Gloom"}
_DARK_PAYOFF = {"Darkness", "Multi-Cast"}

# 专注流
_FOCUS_CARDS = {"Defragment", "Biased Cognition", "Capacitor", "Amplify", "Chill"}
_FOCUS_PAYOFF = _FROST_GENERATORS | _LIGHTNING_GENERATORS | _DARK_GENERATORS

# 爪牌流
_CLAW_CARDS = {"Claw", "Compile Driver", "All for One"}

# 能量生成
_ENERGY_GENERATORS = {"Fusion", "Turbo", "Overclock", "FTL"}
_ENERGY_PAYOFF = {"Meteor Strike", "Hyperbeam", "All for One", "Rainbow", "Streamline"}

# 卡牌检索/补牌
_DRAW_ENGINES = {"Seek", "Compile Driver", "Scrape", "Machine Learning",
                 "Recursion", "Aggregate", "Skim", "White Noise"}

# 防御层
_BLOCK_GENERATORS = {"Defend_B", "Coolheaded", "Steam Barrier", "Reinforced Body",
                     "Buffer", "Auto-Shields", "Glacier", "Force Field", "Self-Repair",
                     "Leap", "Stack"}
_BLOCK_SCALERS = {"Glacier", "Equilibrium", "Barricade"}

# 多重施法
_MULTI_CHANNEL = {"Electrodynamics", "Multi-Cast", "Rainbow", "Dualcast", "Amplify"}
_ECHO_FORM_PAYOFF = {"Meteor Strike", "Hyperbeam", "Claw", "Blizzard", "All for One"}

_SYNERGY_PAIRS = [
    (_FROST_GENERATORS, _FROST_PAYOFF, "frost"),
    (_LIGHTNING_GENERATORS, _LIGHTNING_PAYOFF, "lightning"),
    (_DARK_GENERATORS, _DARK_PAYOFF, "dark"),
    (_FOCUS_CARDS, _FROST_GENERATORS | _LIGHTNING_GENERATORS, "focus_orb"),
    (_CLAW_CARDS, _CLAW_CARDS, "claw_stack"),
    (_ENERGY_GENERATORS, _ENERGY_PAYOFF, "energy_payoff"),
    (_DRAW_ENGINES, {"All for One", "Blizzard", "Barrage"}, "draw_payoff"),
    (_BLOCK_GENERATORS, _BLOCK_SCALERS, "block_scale"),
    ({"Echo Form"}, _ECHO_FORM_PAYOFF, "echo_form"),
    (_MULTI_CHANNEL, _FROST_GENERATORS | _LIGHTNING_GENERATORS, "multi_channel"),
    ({"Creative AI"}, {"White Noise", "Seek"}, "creative_ai"),
]

_ARCHETYPES = {
    "frost_block": _FROST_GENERATORS | _FROST_PAYOFF,
    "lightning": _LIGHTNING_GENERATORS | _LIGHTNING_PAYOFF,
    "claw": _CLAW_CARDS,
    "focus_power": _FOCUS_CARDS,
    "thin_deck": set(),
}

_KEYWORD_GROUPS = [
    _FROST_GENERATORS, _FROST_PAYOFF,
    _LIGHTNING_GENERATORS, _LIGHTNING_PAYOFF,
    _DARK_GENERATORS, _DARK_PAYOFF,
    _FOCUS_CARDS, _CLAW_CARDS,
    _ENERGY_GENERATORS, _ENERGY_PAYOFF,
    _DRAW_ENGINES, _BLOCK_GENERATORS,
]


# ---------------------------------------------------------------------------
# 特征函数
# ---------------------------------------------------------------------------

def deck_keyword_features(deck: list[str]) -> np.ndarray:
    counts = np.zeros(len(_KEYWORD_GROUPS), dtype=np.float32)
    for card in deck:
        base = card.split("+")[0].strip()
        for i, group in enumerate(_KEYWORD_GROUPS):
            if base in group:
                counts[i] += 1
    return counts


def deck_archetype_features(deck: list[str]) -> np.ndarray:
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
    """[攻击占比, 技能占比, 能力占比, 关键词x12, 流派x5]，共 20 维。"""
    return np.concatenate([
        _deck_type_ratios(deck),        # 3
        deck_keyword_features(deck),    # 12
        deck_archetype_features(deck),  # 5
    ])


def card_count_in_deck(card: str, deck: list[str]) -> float:
    base = card.split("+")[0].strip()
    return float(sum(1 for c in deck if c.split("+")[0].strip() == base))


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
# 排序数据构造（LambdaMART）
# ---------------------------------------------------------------------------

def build_card_ranking_data(db: dict, vocab: dict):
    stats = db["card_decisions"]["stats"]
    card_to_idx = vocab["card_to_idx"]
    rows, labels, groups = [], [], []

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
        group_size = 0

        for option in offered:
            option_vec = np.zeros(len(card_to_idx), dtype=np.float32)
            if option != "SKIP" and option in card_to_idx:
                option_vec[card_to_idx[option]] = 1
            is_skip = 1.0 if option == "SKIP" else 0.0
            s = stats.get(option, {})
            pick_rate = s.get("pick_rate", 0.0)
            wrid = s.get("win_rate_in_deck", 0.0)
            count_in_deck = card_count_in_deck(option, d["deck"])
            extra = np.array([is_boss, is_skip, pick_rate, wrid, count_in_deck], dtype=np.float32)
            synergy = card_synergy_features(option, d["deck"])
            rows.append(np.concatenate([base, da_feats, tempo, extra, synergy, option_vec]))
            labels.append(2 if option == d["picked"] and d["victory"] else
                          1 if option == d["picked"] else 0)
            group_size += 1
        groups.append(group_size)

    return np.array(rows), np.array(labels), np.array(groups)


def build_boss_relic_ranking_data(db: dict, vocab: dict):
    stats = db["boss_relic_decisions"]["stats"]
    relic_to_idx = vocab["relic_to_idx"]
    rows, labels, groups = [], [], []

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
            extra = np.array([s.get("pick_rate", 0.0), s.get("win_rate_when_picked", 0.0)], dtype=np.float32)
            rows.append(np.concatenate([base, da_feats, tempo, extra, option_vec]))
            labels.append(2 if option == d["picked"] and d["victory"] else
                          1 if option == d["picked"] else 0)
            group_size += 1
        groups.append(group_size)

    return np.array(rows), np.array(labels), np.array(groups)


def build_campfire_ranking_data(db: dict, vocab: dict):
    rows, labels, groups = [], [], []

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
            extra = np.array([choice_code, hp_below_30, hp_30_50, hp_50_70, hp_above_70], dtype=np.float32)
            rows.append(np.concatenate([base, da_feats, tempo, extra]))
            labels.append(2 if d["choice"] == choice_name and d["victory"] else
                          1 if d["choice"] == choice_name else 0)
        groups.append(2)

    return np.array(rows), np.array(labels), np.array(groups)


def build_shop_ranking_data(db: dict, vocab: dict):
    stats = db["shop_decisions"]["stats"]
    card_to_idx = vocab["card_to_idx"]
    relic_to_idx = vocab["relic_to_idx"]
    rows, labels, groups = [], [], []

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
            d.get("available_cards", []) + d.get("available_relics", []) + d.get("available_potions", [])
        )
        if not all_available:
            continue

        items_with_skip = all_available + ["不购买"]
        group_size = 0

        for item in items_with_skip:
            if item == "不购买":
                card_vec = np.zeros(len(card_to_idx), dtype=np.float32)
                relic_vec = np.zeros(len(relic_to_idx), dtype=np.float32)
                extra = np.array([gold, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                was_picked = len(purchased_set) == 0
            else:
                card_vec, relic_vec = encode_item(item, card_to_idx, relic_to_idx)
                s = stats.get(item, {})
                extra = np.array([gold, s.get("win_rate_when_purchased", 0.0),
                                  s.get("win_rate_when_skipped", 0.0),
                                  float(s.get("times_purchased", 0)),
                                  1.0 if item in purchased_set else 0.0], dtype=np.float32)
                was_picked = item in purchased_set
            rows.append(np.concatenate([base, da_feats, tempo, extra, card_vec, relic_vec]))
            labels.append(2 if was_picked and d["victory"] else 1 if was_picked else 0)
            group_size += 1
        groups.append(group_size)

    return np.array(rows), np.array(labels), np.array(groups)


# ---------------------------------------------------------------------------
# 选择数据构造（LogReg）
# ---------------------------------------------------------------------------

def build_card_choice_data(db: dict, vocab: dict):
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

        for option in offered:
            option_vec = np.zeros(len(card_to_idx), dtype=np.float32)
            if option != "SKIP" and option in card_to_idx:
                option_vec[card_to_idx[option]] = 1
            is_skip = 1.0 if option == "SKIP" else 0.0
            s = stats.get(option, {})
            extra = np.array([is_boss, is_skip, s.get("pick_rate", 0.0),
                              s.get("win_rate_in_deck", 0.0),
                              card_count_in_deck(option, d["deck"])], dtype=np.float32)
            synergy = card_synergy_features(option, d["deck"])
            rows.append(np.concatenate([base, da_feats, tempo, extra, synergy, option_vec]))
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
            extra = np.array([s.get("pick_rate", 0.0), s.get("win_rate_when_picked", 0.0)], dtype=np.float32)
            rows.append(np.concatenate([base, da_feats, tempo, extra, option_vec]))
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
            extra = np.array([choice_code, hp_below_30, hp_30_50, hp_50_70, hp_above_70], dtype=np.float32)
            rows.append(np.concatenate([base, da_feats, tempo, extra]))
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
            d.get("available_cards", []) + d.get("available_relics", []) + d.get("available_potions", [])
        )
        if not all_available:
            continue

        for item in all_available + ["不购买"]:
            if item == "不购买":
                card_vec = np.zeros(len(card_to_idx), dtype=np.float32)
                relic_vec = np.zeros(len(relic_to_idx), dtype=np.float32)
                extra = np.array([gold, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                was_picked = len(purchased_set) == 0
            else:
                card_vec, relic_vec = encode_item(item, card_to_idx, relic_to_idx)
                s = stats.get(item, {})
                extra = np.array([gold, s.get("win_rate_when_purchased", 0.0),
                                  s.get("win_rate_when_skipped", 0.0),
                                  float(s.get("times_purchased", 0)),
                                  1.0 if item in purchased_set else 0.0], dtype=np.float32)
                was_picked = item in purchased_set
            rows.append(np.concatenate([base, da_feats, tempo, extra, card_vec, relic_vec]))
            labels.append(1 if was_picked else 0)

    return np.array(rows), np.array(labels)


# ---------------------------------------------------------------------------
# CWR-Delta
# ---------------------------------------------------------------------------

def _deck_size_bucket(deck_size: int) -> str:
    if deck_size <= 10:
        return "small"
    elif deck_size <= 20:
        return "medium"
    elif deck_size <= 30:
        return "large"
    return "xlarge"


def _to_plain_dict(cwr, global_stats, base_stats) -> dict:
    return {
        "contextual": {k: dict(v) for k, v in cwr.items()},
        "global": dict(global_stats),
        "base": base_stats,
    }


def compute_card_cwr_stats(db: dict) -> dict:
    cwr = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "total": 0}))
    global_stats = defaultdict(lambda: {"wins": 0, "total": 0})
    base_stats = {"wins": 0, "total": 0}
    for d in db["card_decisions"]["decisions"]:
        ctx_key = f"{d['act']}_{_deck_size_bucket(d['deck_size'])}_{d.get('is_boss_reward', False)}"
        picked, win = d["picked"], d["victory"]
        cwr[ctx_key][picked]["total"] += 1
        global_stats[picked]["total"] += 1
        base_stats["total"] += 1
        if win:
            cwr[ctx_key][picked]["wins"] += 1
            global_stats[picked]["wins"] += 1
            base_stats["wins"] += 1
    return _to_plain_dict(cwr, global_stats, base_stats)


def compute_boss_relic_cwr_stats(db: dict) -> dict:
    cwr = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "total": 0}))
    global_stats = defaultdict(lambda: {"wins": 0, "total": 0})
    base_stats = {"wins": 0, "total": 0}
    for d in db["boss_relic_decisions"]["decisions"]:
        if not d["picked"]:
            continue
        ctx_key = f"{d['act']}_{_deck_size_bucket(d['deck_size'])}"
        picked, win = d["picked"], d["victory"]
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
        hp_bucket = "low" if d["hp_pct"] < 40 else ("mid" if d["hp_pct"] < 70 else "high")
        ctx_key = f"{d['act']}_{hp_bucket}"
        choice, win = d["choice"], d["victory"]
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
        ctx_key = f"{d.get('act', 1)}_{_deck_size_bucket(d['deck_size'])}"
        win = d["victory"]
        purchased = d.get("purchased", [])
        for item in purchased:
            cwr[ctx_key][item]["total"] += 1
            global_stats[item]["total"] += 1
            base_stats["total"] += 1
            if win:
                cwr[ctx_key][item]["wins"] += 1
                global_stats[item]["wins"] += 1
                base_stats["wins"] += 1
        if not purchased:
            cwr[ctx_key]["不购买"]["total"] += 1
            global_stats["不购买"]["total"] += 1
            base_stats["total"] += 1
            if win:
                cwr[ctx_key]["不购买"]["wins"] += 1
                global_stats["不购买"]["wins"] += 1
                base_stats["wins"] += 1
    return _to_plain_dict(cwr, global_stats, base_stats)


def _bayesian_win_rate(wins: int, total: int, prior_wr: float, strength: int = 5) -> float:
    return (wins + strength * prior_wr) / (total + strength)


def predict_cwr_delta(options: list[str], context_key: str, cwr_stats: dict) -> np.ndarray:
    base = cwr_stats["base"]
    base_wr = base["wins"] / max(base["total"], 1)
    deltas = []
    for option in options:
        ctx = cwr_stats["contextual"].get(context_key, {})
        s = ctx.get(option)
        if s and s["total"] >= 3:
            wr = _bayesian_win_rate(s["wins"], s["total"], base_wr)
        else:
            gs = cwr_stats["global"].get(option, {"wins": 0, "total": 0})
            wr = _bayesian_win_rate(gs["wins"], gs["total"], base_wr) if gs["total"] >= 3 else base_wr
        deltas.append(wr - base_wr)
    deltas = np.array(deltas, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-10.0 * deltas))


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train_ranking_model(X, y, groups, name: str):
    if lgb is None or len(X) == 0:
        return None
    group_ids = np.repeat(np.arange(len(groups)), groups)
    n_splits = min(5, len(groups))
    if n_splits < 2:
        n_splits = 2
    gkf = GroupKFold(n_splits=n_splits)
    models = []
    ndcg_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, group_ids)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_groups_map = defaultdict(int)
        for idx in train_idx:
            train_groups_map[group_ids[idx]] += 1
        seen = set()
        train_group_order = []
        for idx in train_idx:
            gid = group_ids[idx]
            if gid not in seen:
                seen.add(gid)
                train_group_order.append(gid)
        train_group_sizes = [train_groups_map[gid] for gid in train_group_order]

        sort_idx = np.argsort(group_ids[train_idx])
        X_train = X_train[sort_idx]
        y_train = y_train[sort_idx]

        model = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=150, max_depth=4,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=5, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1, device="gpu",
        )
        model.fit(X_train, y_train, group=train_group_sizes)
        models.append(model)

        val_scores = model.predict(X_val)
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
            ndcg_scores.append(np.mean(fold_ndcgs))

    if ndcg_scores:
        print(f"  LambdaMART {name}: 平均 NDCG={np.mean(ndcg_scores):.4f}")
    else:
        print(f"  LambdaMART {name}: 训练完成")
    return models


def train_choice_model(X, y, name: str):
    if len(X) == 0 or len(np.unique(y)) < 2:
        return None
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)),
    ])
    pipe.fit(X, y)
    train_acc = np.mean((pipe.predict_proba(X)[:, 1] > 0.5).astype(int) == y)
    print(f"  LogReg {name}: 训练集 Accuracy={train_acc:.4f}")
    return pipe


# ---------------------------------------------------------------------------
# 推理特征（V2 版本）
# ---------------------------------------------------------------------------

def card_inference_features_v2(floor, act, hp_pct, deck, relics, options, stats,
                                vocab, num_upgrades=0, deck_upgrades=None):
    card_to_idx = vocab["card_to_idx"]
    base = base_features(floor, act, hp_pct, len(deck), len(relics), deck, relics, vocab,
                         num_upgrades, deck_upgrades)
    is_boss = 1.0 if floor in (16, 33) else 0.0
    da_feats = deck_analysis_features(deck)
    tempo = temporal_features(floor, act, hp_pct)
    rows = []
    for option in options:
        option_vec = np.zeros(len(card_to_idx), dtype=np.float32)
        if option != "SKIP" and option in card_to_idx:
            option_vec[card_to_idx[option]] = 1
        is_skip = 1.0 if option == "SKIP" else 0.0
        s = stats.get(option, {})
        extra = np.array([is_boss, is_skip, s.get("pick_rate", 0.0),
                          s.get("win_rate_in_deck", 0.0),
                          card_count_in_deck(option, deck)], dtype=np.float32)
        synergy = card_synergy_features(option, deck)
        rows.append(np.concatenate([base, da_feats, tempo, extra, synergy, option_vec]))
    return np.array(rows)


def boss_relic_inference_features_v2(act, hp_pct, deck, relics, options, stats,
                                     vocab, num_upgrades=0, deck_upgrades=None):
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
        extra = np.array([s.get("pick_rate", 0.0), s.get("win_rate_when_picked", 0.0)], dtype=np.float32)
        rows.append(np.concatenate([base, da_feats, tempo, extra, option_vec]))
    return np.array(rows)


def campfire_inference_features_v2(floor, act, hp_pct, deck, relics, vocab,
                                   num_upgrades=0, deck_upgrades=None):
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
        extra = np.array([choice_code, hp_below_30, hp_30_50, hp_50_70, hp_above_70], dtype=np.float32)
        rows.append(np.concatenate([base, da_feats, tempo, extra]))
    return np.array(rows)


def shop_inference_features_v2(floor, act, hp_pct, gold, deck, relics, items, stats,
                                vocab, num_upgrades=0, deck_upgrades=None):
    card_to_idx = vocab["card_to_idx"]
    relic_to_idx = vocab["relic_to_idx"]
    base = base_features(floor, act, hp_pct, len(deck), len(relics), deck, relics, vocab,
                         num_upgrades, deck_upgrades)
    da_feats = deck_analysis_features(deck)
    tempo = temporal_features(floor, act, hp_pct)
    rows = []
    for item in items:
        card_vec, relic_vec = encode_item(item, card_to_idx, relic_to_idx)
        s = stats.get(item, {})
        extra = np.array([gold, s.get("win_rate_when_purchased", 0.0),
                          s.get("win_rate_when_skipped", 0.0),
                          float(s.get("times_purchased", 0)), 1.0], dtype=np.float32)
        rows.append(np.concatenate([base, da_feats, tempo, extra, card_vec, relic_vec]))
    card_vec = np.zeros(len(card_to_idx), dtype=np.float32)
    relic_vec = np.zeros(len(relic_to_idx), dtype=np.float32)
    rows.append(np.concatenate([base, da_feats, tempo, np.array([gold, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                                card_vec, relic_vec]))
    return np.array(rows)


# ---------------------------------------------------------------------------
# 能力牌重复惩罚
# ---------------------------------------------------------------------------

def _apply_duplicate_power_penalty(options, deck, preds, penalty=0.5):
    deck_bases = set(c.split("+")[0].strip() for c in deck)
    for i, option in enumerate(options):
        base = option.split("+")[0].strip()
        if base in _POWER_CARDS and base in deck_bases:
            for eng in preds:
                preds[eng][i] *= penalty
    return preds


# ---------------------------------------------------------------------------
# 排序模型推理
# ---------------------------------------------------------------------------

def predict_ranking(models, X):
    if not models:
        return np.zeros(len(X))
    scores = np.mean([m.predict(X) for m in models], axis=0)
    exp_s = np.exp(scores - scores.max())
    return exp_s / exp_s.sum()


def predict_choice(pipe, X):
    if pipe is None:
        return np.zeros(len(X))
    return pipe.predict_proba(X)[:, 1]


def _safe_predict_v1(v1_model_dict, X):
    if not v1_model_dict:
        return {}
    try:
        return predict_v1(v1_model_dict, X)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 综合推理入口
# ---------------------------------------------------------------------------

def predict_all_card(options, floor, act, hp_pct, deck, relics, db, vocab,
                     v1_models, v2_models, num_upgrades=0, deck_upgrades=None):
    stats = db["card_decisions"]["stats"]
    preds = {}
    X_v1 = card_inference_features_v1(floor, act, hp_pct, deck, relics,
                                      options, stats, vocab, num_upgrades, deck_upgrades)
    preds.update(_safe_predict_v1(v1_models.get("card", {}), X_v1))
    X_v2 = card_inference_features_v2(floor, act, hp_pct, deck, relics,
                                      options, stats, vocab, num_upgrades, deck_upgrades)
    if "card_lambdamart" in v2_models:
        preds["lambdamart"] = predict_ranking(v2_models["card_lambdamart"], X_v2)
    if "card_logreg" in v2_models:
        preds["logreg"] = predict_choice(v2_models["card_logreg"], X_v2)
    if "card_cwr" in v2_models:
        is_boss = floor in (16, 33)
        ctx_key = f"{act}_{_deck_size_bucket(len(deck))}_{is_boss}"
        preds["cwr_delta"] = predict_cwr_delta(options, ctx_key, v2_models["card_cwr"])
    preds = _apply_duplicate_power_penalty(options, deck, preds)
    return preds


def predict_all_campfire(floor, act, hp_pct, deck, relics, db, vocab,
                         v1_models, v2_models, num_upgrades=0, deck_upgrades=None):
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
        preds["cwr_delta"] = predict_cwr_delta(["REST", "SMITH"],
                                               f"{act}_{hp_bucket}", v2_models["campfire_cwr"])
    return preds


def predict_all_boss_relic(options, act, hp_pct, deck, relics, db, vocab,
                           v1_models, v2_models, num_upgrades=0, deck_upgrades=None):
    preds = {}
    stats = db["boss_relic_decisions"]["stats"]
    X_v1 = boss_relic_inference_features_v1(act, hp_pct, deck, relics, options,
                                            stats, vocab, num_upgrades, deck_upgrades)
    preds.update(_safe_predict_v1(v1_models.get("boss_relic", {}), X_v1))
    X_v2 = boss_relic_inference_features_v2(act, hp_pct, deck, relics, options,
                                            stats, vocab, num_upgrades, deck_upgrades)
    if "boss_relic_lambdamart" in v2_models:
        preds["lambdamart"] = predict_ranking(v2_models["boss_relic_lambdamart"], X_v2)
    if "boss_relic_logreg" in v2_models:
        preds["logreg"] = predict_choice(v2_models["boss_relic_logreg"], X_v2)
    if "boss_relic_cwr" in v2_models:
        ctx_key = f"{act}_{_deck_size_bucket(len(deck))}"
        preds["cwr_delta"] = predict_cwr_delta(options, ctx_key, v2_models["boss_relic_cwr"])
    return preds


def predict_all_shop(option_labels, floor, act, hp_pct, gold, deck, relics, items,
                     db, vocab, v1_models, v2_models, num_upgrades=0, deck_upgrades=None):
    preds = {}
    stats = db["shop_decisions"]["stats"]
    X_v1 = shop_inference_features_v1(floor, act, hp_pct, gold, deck, relics,
                                      items, stats, vocab, num_upgrades, deck_upgrades)
    preds.update(_safe_predict_v1(v1_models.get("shop", {}), X_v1))
    X_v2 = shop_inference_features_v2(floor, act, hp_pct, gold, deck, relics,
                                      items, stats, vocab, num_upgrades, deck_upgrades)
    if "shop_lambdamart" in v2_models:
        preds["lambdamart"] = predict_ranking(v2_models["shop_lambdamart"], X_v2)
    if "shop_logreg" in v2_models:
        preds["logreg"] = predict_choice(v2_models["shop_logreg"], X_v2)
    if "shop_cwr" in v2_models:
        ctx_key = f"{act}_{_deck_size_bucket(len(deck))}"
        preds["cwr_delta"] = predict_cwr_delta(option_labels, ctx_key, v2_models["shop_cwr"])
    return preds


# ---------------------------------------------------------------------------
# 保存 / 加载 V2 模型
# ---------------------------------------------------------------------------

def save_v2_model(obj, name: str):
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  已保存: {path}")


def load_v2_models() -> dict:
    result = {}
    for name in ["card_lambdamart", "card_logreg", "card_cwr",
                 "boss_relic_lambdamart", "boss_relic_logreg", "boss_relic_cwr",
                 "campfire_lambdamart", "campfire_logreg", "campfire_cwr",
                 "shop_lambdamart", "shop_logreg", "shop_cwr"]:
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
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    print("\n=== 训练卡牌 V2 模型 ===")
    X, y, groups = build_card_ranking_data(db, vocab)
    print(f"  排序数据: {len(y)} 行, {len(groups)} 组")
    m = train_ranking_model(X, y, groups, "card")
    if m:
        save_v2_model(m, "card_lambdamart")
    X, y = build_card_choice_data(db, vocab)
    print(f"  选择数据: {len(y)} 行")
    m = train_choice_model(X, y, "card")
    if m:
        save_v2_model(m, "card_logreg")
    save_v2_model(compute_card_cwr_stats(db), "card_cwr")

    print("\n=== 训练 Boss 遗物 V2 模型 ===")
    X, y, groups = build_boss_relic_ranking_data(db, vocab)
    print(f"  排序数据: {len(y)} 行, {len(groups)} 组")
    m = train_ranking_model(X, y, groups, "boss_relic")
    if m:
        save_v2_model(m, "boss_relic_lambdamart")
    X, y = build_boss_relic_choice_data(db, vocab)
    m = train_choice_model(X, y, "boss_relic")
    if m:
        save_v2_model(m, "boss_relic_logreg")
    save_v2_model(compute_boss_relic_cwr_stats(db), "boss_relic_cwr")

    print("\n=== 训练篝火 V2 模型 ===")
    X, y, groups = build_campfire_ranking_data(db, vocab)
    print(f"  排序数据: {len(y)} 行, {len(groups)} 组")
    m = train_ranking_model(X, y, groups, "campfire")
    if m:
        save_v2_model(m, "campfire_lambdamart")
    X, y = build_campfire_choice_data(db, vocab)
    m = train_choice_model(X, y, "campfire")
    if m:
        save_v2_model(m, "campfire_logreg")
    save_v2_model(compute_campfire_cwr_stats(db), "campfire_cwr")

    print("\n=== 训练商店 V2 模型 ===")
    X, y, groups = build_shop_ranking_data(db, vocab)
    print(f"  排序数据: {len(y)} 行, {len(groups)} 组")
    m = train_ranking_model(X, y, groups, "shop")
    if m:
        save_v2_model(m, "shop_lambdamart")
    X, y = build_shop_choice_data(db, vocab)
    m = train_choice_model(X, y, "shop")
    if m:
        save_v2_model(m, "shop_logreg")
    save_v2_model(compute_shop_cwr_stats(db), "shop_cwr")

    print("\nV2 训练完成！")


def main():
    parser = argparse.ArgumentParser(description="机器人 ML 决策顾问 V2")
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("train")

    cp = subs.add_parser("card")
    cp.add_argument("--floor", type=int, required=True)
    cp.add_argument("--act", type=int, required=True)
    cp.add_argument("--hp", type=int, required=True)
    cp.add_argument("--max-hp", type=int, required=True, dest="max_hp")
    cp.add_argument("--relics", default="")
    cp.add_argument("--deck", default="")
    cp.add_argument("--options", required=True)

    fp = subs.add_parser("campfire")
    fp.add_argument("--floor", type=int, required=True)
    fp.add_argument("--act", type=int, required=True)
    fp.add_argument("--hp", type=int, required=True)
    fp.add_argument("--max-hp", type=int, required=True, dest="max_hp")
    fp.add_argument("--relics", default="")
    fp.add_argument("--deck", default="")

    bp = subs.add_parser("boss-relic")
    bp.add_argument("--act", type=int, required=True)
    bp.add_argument("--hp", type=int, required=True)
    bp.add_argument("--max-hp", type=int, required=True, dest="max_hp")
    bp.add_argument("--relics", default="")
    bp.add_argument("--deck", default="")
    bp.add_argument("--options", required=True)

    sp = subs.add_parser("shop")
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
        return

    db = load_db()
    vocab = load_vocab()
    v1_models = {}
    v2_models = load_v2_models()
    deck = parse_deck(args.deck)
    relics = parse_list(args.relics)
    hp_pct = round(args.hp / args.max_hp * 100) if args.max_hp > 0 else 0

    def _print(options, preds):
        import unicodedata
        def dw(s):
            return sum(2 if unicodedata.east_asian_width(c) in ("W","F") else 1 for c in s)
        def pr(s, w):
            return s + " " * (w - dw(s))
        n = len(options)
        rank_sum = np.zeros(n)
        for eng, prbs in preds.items():
            rank_sum += n - np.argsort(np.argsort(prbs))
        order = np.argsort(rank_sum)
        best = int(rank_sum.argmin())
        for i in order:
            star = "  ★" if i == best else ""
            row = pr(options[i], 24)
            for eng, prbs in preds.items():
                row += f"  {prbs[i]*100:>7.1f}%"
            print(row + star)

    if args.command == "card":
        options = parse_list(args.options)
        if "SKIP" not in options:
            options.append("SKIP")
        preds = predict_all_card(options, args.floor, args.act, hp_pct,
                                 deck, relics, db, vocab, v1_models, v2_models)
        print(f"\n=== 机器人 V2 (卡牌选择) ===")
        _print(options, preds)
    elif args.command == "campfire":
        preds = predict_all_campfire(args.floor, args.act, hp_pct,
                                     deck, relics, db, vocab, v1_models, v2_models)
        print(f"\n=== 机器人 V2 (篝火决策) ===")
        _print(["REST", "SMITH"], preds)
    elif args.command == "boss-relic":
        options = parse_list(args.options)
        preds = predict_all_boss_relic(options, args.act, hp_pct,
                                       deck, relics, db, vocab, v1_models, v2_models)
        print(f"\n=== 机器人 V2 (Boss 遗物) ===")
        _print(options, preds)
    elif args.command == "shop":
        all_items = parse_list(args.cards) + parse_list(args.shop_relics) + parse_list(args.potions)
        option_labels = all_items + ["不购买"]
        preds = predict_all_shop(option_labels, args.floor, args.act, hp_pct, args.gold,
                                 deck, relics, all_items, db, vocab, v1_models, v2_models)
        print(f"\n=== 机器人 V2 (商店) ===")
        _print(option_labels, preds)


if __name__ == "__main__":
    main()
