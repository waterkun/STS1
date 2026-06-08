#!/usr/bin/env python3
"""
Watcher ML Advisor - XGBoost/LightGBM 决策预测模型

基于历史对局数据训练分类器，预测每个选项的胜率。

Usage:
  # 训练所有模型
  python -m watcher_advisor.ml_advisor train

  # 卡牌决策
  python -m watcher_advisor.ml_advisor card \
    --floor 8 --act 1 --hp 45 --max-hp 72 \
    --relics "PureWater" \
    --deck "Strike_P x4,Defend_P x4,Eruption,Vigilance" \
    --options "Rushdown,Empty Mind,Conclude"

  # 篝火决策
  python -m watcher_advisor.ml_advisor campfire \
    --floor 11 --act 1 --hp 35 --max-hp 72 \
    --relics "PureWater" \
    --deck "Strike_P x4,Defend_P x4,Eruption,Vigilance,Tantrum"

  # Boss 遗物
  python -m watcher_advisor.ml_advisor boss-relic \
    --act 1 --hp 55 --max-hp 72 \
    --relics "PureWater" \
    --deck "Strike_P x4,Defend_P x4,Eruption,Vigilance,Tantrum" \
    --options "Snecko Eye,Cursed Key,Coffee Dripper"

  # 商店决策
  python -m watcher_advisor.ml_advisor shop \
    --floor 27 --act 2 --hp 50 --max-hp 72 --gold 300 \
    --relics "PureWater,Snecko Eye" \
    --deck "Strike_P x3,Defend_P x4,Eruption,Vigilance,Tantrum" \
    --cards "Vault,Deva Form,Rushdown" \
    --shop-relics "Mark of Pain,Du-Vu Doll" \
    --potions "Stance Potion,FairyPotion"
"""

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score
except ImportError:
    print("scikit-learn 未安装，请运行: pip install scikit-learn")
    sys.exit(1)


DB_DIR = Path(__file__).parent / "db"
MODEL_DIR = Path(__file__).parent / "models"


# ---------------------------------------------------------------------------
# 数据加载与解析工具
# ---------------------------------------------------------------------------

def load_db() -> dict:
    db = {}
    for name in ["card_decisions", "boss_relic_decisions", "campfire_decisions", "shop_decisions"]:
        path = DB_DIR / f"{name}.json"
        if not path.exists():
            print(f"数据库文件未找到: {path}")
            print("请先运行: python watcher_advisor/build_db.py")
            sys.exit(1)
        db[name] = json.loads(path.read_text(encoding="utf-8"))
    return db


def parse_deck(deck_str: str) -> list[str]:
    cards = []
    for part in deck_str.split(","):
        part = part.strip()
        if not part:
            continue
        if " x" in part:
            name, _, count = part.rpartition(" x")
            try:
                cards.extend([name.strip()] * int(count))
                continue
            except ValueError:
                pass
        cards.append(part)
    return cards


def parse_list(s: str) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# 词表构建
# ---------------------------------------------------------------------------

def build_vocabularies(db: dict) -> dict:
    all_cards = set()
    all_relics = set()

    for d in db["card_decisions"]["decisions"]:
        all_cards.update(d["deck"])
        all_cards.update(c for c in d["offered"] if c != "SKIP")
        all_relics.update(d["relics"])

    for d in db["campfire_decisions"]["decisions"]:
        all_cards.update(d["deck"])
        all_relics.update(d["relics"])

    for d in db["boss_relic_decisions"]["decisions"]:
        all_cards.update(d["deck"])
        all_relics.update(d["relics_before"])
        all_relics.update(d["offered"])

    for d in db["shop_decisions"]["decisions"]:
        all_cards.update(d["deck"])
        all_relics.update(d["relics"])
        all_cards.update(d.get("available_cards", []))
        all_relics.update(d.get("available_relics", []))

    all_cards.discard("SKIP")
    card_vocab = sorted(all_cards)
    relic_vocab = sorted(all_relics)

    card_to_idx = {c: i for i, c in enumerate(card_vocab)}
    relic_to_idx = {r: i for i, r in enumerate(relic_vocab)}

    return {
        "card_vocab": card_vocab,
        "relic_vocab": relic_vocab,
        "card_to_idx": card_to_idx,
        "relic_to_idx": relic_to_idx,
    }


# ---------------------------------------------------------------------------
# 特征编码
# ---------------------------------------------------------------------------

def encode_deck(deck: list[str], card_to_idx: dict) -> np.ndarray:
    vec = np.zeros(len(card_to_idx), dtype=np.float32)
    for card in deck:
        if card in card_to_idx:
            vec[card_to_idx[card]] += 1
    return vec


def encode_relics(relics: list[str], relic_to_idx: dict) -> np.ndarray:
    vec = np.zeros(len(relic_to_idx), dtype=np.float32)
    for relic in relics:
        if relic in relic_to_idx:
            vec[relic_to_idx[relic]] = 1
    return vec


def encode_act_onehot(act: int) -> np.ndarray:
    vec = np.zeros(4, dtype=np.float32)
    if 1 <= act <= 4:
        vec[act - 1] = 1
    return vec


def encode_deck_upgrades(deck_upgrades: dict, card_to_idx: dict) -> np.ndarray:
    vec = np.zeros(len(card_to_idx), dtype=np.float32)
    for card, count in deck_upgrades.items():
        if card in card_to_idx:
            vec[card_to_idx[card]] = count
    return vec


def base_features(floor: int, act: int, hp_pct: int, deck_size: int,
                  num_relics: int, deck: list[str], relics: list[str],
                  vocab: dict, num_upgrades: int = 0,
                  deck_upgrades: dict | None = None) -> np.ndarray:
    upgrade_ratio = num_upgrades / deck_size if deck_size > 0 else 0.0
    numeric = np.array([floor, hp_pct, deck_size, num_relics,
                        num_upgrades, upgrade_ratio], dtype=np.float32)
    act_oh = encode_act_onehot(act)
    deck_vec = encode_deck(deck, vocab["card_to_idx"])
    relic_vec = encode_relics(relics, vocab["relic_to_idx"])
    upgrades_vec = encode_deck_upgrades(
        deck_upgrades if deck_upgrades is not None else {},
        vocab["card_to_idx"]
    )
    return np.concatenate([numeric, act_oh, deck_vec, relic_vec, upgrades_vec])


def base_feature_names(vocab: dict) -> list[str]:
    names = ["floor", "hp_pct", "deck_size", "num_relics",
             "num_upgrades", "upgrade_ratio"]
    names += [f"act_{i+1}" for i in range(4)]
    names += [f"deck_{c}" for c in vocab["card_vocab"]]
    names += [f"relic_{r}" for r in vocab["relic_vocab"]]
    names += [f"upgrade_{c}" for c in vocab["card_vocab"]]
    return names


# ---------------------------------------------------------------------------
# 卡牌决策特征
# ---------------------------------------------------------------------------

def build_card_features(db: dict, vocab: dict) -> tuple[np.ndarray, np.ndarray]:
    stats = db["card_decisions"]["stats"]
    card_to_idx = vocab["card_to_idx"]
    rows = []
    labels = []

    for d in db["card_decisions"]["decisions"]:
        base = base_features(
            d["floor"], d["act"], d["hp_pct"], d["deck_size"],
            len(d["relics"]), d["deck"], d["relics"], vocab,
            d.get("num_upgrades", 0), d.get("deck_upgrades", {})
        )
        is_boss = 1.0 if d.get("is_boss_reward", False) else 0.0

        picked = d["picked"]
        option_vec = np.zeros(len(card_to_idx), dtype=np.float32)
        if picked != "SKIP" and picked in card_to_idx:
            option_vec[card_to_idx[picked]] = 1

        is_skip = 1.0 if picked == "SKIP" else 0.0

        s = stats.get(picked, {})
        pick_rate = s.get("pick_rate", 0.0)
        win_rate_in_deck = s.get("win_rate_in_deck", 0.0)

        extra = np.array([is_boss, is_skip, pick_rate, win_rate_in_deck],
                         dtype=np.float32)
        row = np.concatenate([base, extra, option_vec])
        rows.append(row)
        labels.append(1 if d["victory"] else 0)

    return np.array(rows), np.array(labels)


def card_feature_names(vocab: dict) -> list[str]:
    names = base_feature_names(vocab)
    names += ["is_boss_reward", "is_skip", "pick_rate", "win_rate_in_deck"]
    names += [f"option_{c}" for c in vocab["card_vocab"]]
    return names


def card_inference_features(floor: int, act: int, hp_pct: int,
                            deck: list[str], relics: list[str],
                            options: list[str], stats: dict, vocab: dict,
                            num_upgrades: int = 0,
                            deck_upgrades: dict | None = None) -> np.ndarray:
    card_to_idx = vocab["card_to_idx"]
    base = base_features(floor, act, hp_pct, len(deck), len(relics), deck, relics, vocab,
                         num_upgrades, deck_upgrades)
    is_boss = 1.0 if floor in (16, 33) else 0.0
    rows = []

    for option in options:
        option_vec = np.zeros(len(card_to_idx), dtype=np.float32)
        if option != "SKIP" and option in card_to_idx:
            option_vec[card_to_idx[option]] = 1

        is_skip = 1.0 if option == "SKIP" else 0.0
        s = stats.get(option, {})
        pick_rate = s.get("pick_rate", 0.0)
        win_rate_in_deck = s.get("win_rate_in_deck", 0.0)

        extra = np.array([is_boss, is_skip, pick_rate, win_rate_in_deck],
                         dtype=np.float32)
        rows.append(np.concatenate([base, extra, option_vec]))

    return np.array(rows)


# ---------------------------------------------------------------------------
# 篝火决策特征
# ---------------------------------------------------------------------------

def build_campfire_features(db: dict, vocab: dict) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []

    for d in db["campfire_decisions"]["decisions"]:
        base = base_features(
            d["floor"], d["act"], d["hp_pct"], d["deck_size"],
            len(d["relics"]), d["deck"], d["relics"], vocab,
            d.get("num_upgrades", 0), d.get("deck_upgrades", {})
        )

        choice_code = 1.0 if d["choice"] == "SMITH" else 0.0
        hp_below_30 = 1.0 if d["hp_pct"] < 30 else 0.0
        hp_30_50 = 1.0 if 30 <= d["hp_pct"] < 50 else 0.0
        hp_50_70 = 1.0 if 50 <= d["hp_pct"] < 70 else 0.0
        hp_above_70 = 1.0 if d["hp_pct"] >= 70 else 0.0

        extra = np.array([choice_code,
                          hp_below_30, hp_30_50, hp_50_70, hp_above_70],
                         dtype=np.float32)
        rows.append(np.concatenate([base, extra]))
        labels.append(1 if d["victory"] else 0)

    return np.array(rows), np.array(labels)


def campfire_feature_names(vocab: dict) -> list[str]:
    names = base_feature_names(vocab)
    names += ["choice_smith",
              "hp_below_30", "hp_30_50", "hp_50_70", "hp_above_70"]
    return names


def campfire_inference_features(floor: int, act: int, hp_pct: int,
                                deck: list[str], relics: list[str],
                                vocab: dict, num_upgrades: int = 0,
                                deck_upgrades: dict | None = None) -> np.ndarray:
    base = base_features(floor, act, hp_pct, len(deck), len(relics), deck, relics, vocab,
                         num_upgrades, deck_upgrades)
    hp_below_30 = 1.0 if hp_pct < 30 else 0.0
    hp_30_50 = 1.0 if 30 <= hp_pct < 50 else 0.0
    hp_50_70 = 1.0 if 50 <= hp_pct < 70 else 0.0
    hp_above_70 = 1.0 if hp_pct >= 70 else 0.0
    rows = []
    for choice_code in [0.0, 1.0]:
        extra = np.array([choice_code,
                          hp_below_30, hp_30_50, hp_50_70, hp_above_70],
                         dtype=np.float32)
        rows.append(np.concatenate([base, extra]))
    return np.array(rows)


# ---------------------------------------------------------------------------
# Boss 遗物决策特征
# ---------------------------------------------------------------------------

def build_boss_relic_features(db: dict, vocab: dict) -> tuple[np.ndarray, np.ndarray]:
    stats = db["boss_relic_decisions"]["stats"]
    relic_to_idx = vocab["relic_to_idx"]
    rows = []
    labels = []

    for d in db["boss_relic_decisions"]["decisions"]:
        act = d["act"]
        act_stats = stats.get(str(act), stats.get(act, {}))
        base = base_features(
            0, act, d["hp_pct"], d["deck_size"],
            len(d["relics_before"]), d["deck"], d["relics_before"], vocab,
            d.get("num_upgrades", 0), d.get("deck_upgrades", {})
        )

        picked = d["picked"]
        if not picked:
            continue

        option_vec = np.zeros(len(relic_to_idx), dtype=np.float32)
        if picked in relic_to_idx:
            option_vec[relic_to_idx[picked]] = 1

        s = act_stats.get(picked, {})
        pick_rate = s.get("pick_rate", 0.0)
        win_rate = s.get("win_rate_when_picked", 0.0)

        extra = np.array([pick_rate, win_rate], dtype=np.float32)
        row = np.concatenate([base, extra, option_vec])
        rows.append(row)
        labels.append(1 if d["victory"] else 0)

    return np.array(rows), np.array(labels)


def boss_relic_feature_names(vocab: dict) -> list[str]:
    names = base_feature_names(vocab)
    names += ["relic_pick_rate", "relic_win_rate"]
    names += [f"option_relic_{r}" for r in vocab["relic_vocab"]]
    return names


def boss_relic_inference_features(act: int, hp_pct: int,
                                  deck: list[str], relics: list[str],
                                  options: list[str], stats: dict,
                                  vocab: dict, num_upgrades: int = 0,
                                  deck_upgrades: dict | None = None) -> np.ndarray:
    relic_to_idx = vocab["relic_to_idx"]
    act_stats = stats.get(str(act), stats.get(act, {}))
    base = base_features(0, act, hp_pct, len(deck), len(relics), deck, relics, vocab,
                         num_upgrades, deck_upgrades)
    rows = []

    for option in options:
        option_vec = np.zeros(len(relic_to_idx), dtype=np.float32)
        if option in relic_to_idx:
            option_vec[relic_to_idx[option]] = 1

        s = act_stats.get(option, {})
        pick_rate = s.get("pick_rate", 0.0)
        win_rate = s.get("win_rate_when_picked", 0.0)

        extra = np.array([pick_rate, win_rate], dtype=np.float32)
        rows.append(np.concatenate([base, extra, option_vec]))

    return np.array(rows)


# ---------------------------------------------------------------------------
# 商店决策特征
# ---------------------------------------------------------------------------

def normalize_item_name(name: str) -> str:
    if "+" in name:
        return name.split("+")[0]
    return name


def encode_item(item: str, card_to_idx: dict, relic_to_idx: dict) -> tuple[np.ndarray, np.ndarray]:
    card_vec = np.zeros(len(card_to_idx), dtype=np.float32)
    relic_vec = np.zeros(len(relic_to_idx), dtype=np.float32)
    normed = normalize_item_name(item)
    if normed in card_to_idx:
        card_vec[card_to_idx[normed]] = 1
    if item in card_to_idx:
        card_vec[card_to_idx[item]] = 1
    if normed in relic_to_idx:
        relic_vec[relic_to_idx[normed]] = 1
    if item in relic_to_idx:
        relic_vec[relic_to_idx[item]] = 1
    return card_vec, relic_vec


def build_shop_features(db: dict, vocab: dict) -> tuple[np.ndarray, np.ndarray]:
    stats = db["shop_decisions"]["stats"]
    card_to_idx = vocab["card_to_idx"]
    relic_to_idx = vocab["relic_to_idx"]
    rows = []
    labels = []

    for d in db["shop_decisions"]["decisions"]:
        base = base_features(
            d["floor"], d.get("act", 1), d["hp_pct"], d["deck_size"],
            len(d["relics"]), d["deck"], d["relics"], vocab,
            d.get("num_upgrades", 0), d.get("deck_upgrades", {})
        )

        gold = d.get("gold", 0)
        purchased_set = set(d.get("purchased", []))
        all_available = (
            d.get("available_cards", []) +
            d.get("available_relics", []) +
            d.get("available_potions", [])
        )

        for item in all_available:
            card_vec, relic_vec = encode_item(item, card_to_idx, relic_to_idx)

            s = stats.get(item, {})
            buy_wr = s.get("win_rate_when_purchased", 0.0)
            skip_wr = s.get("win_rate_when_skipped", 0.0)
            times_purchased = s.get("times_purchased", 0)

            was_bought = 1.0 if item in purchased_set else 0.0

            extra = np.array([gold, buy_wr, skip_wr, float(times_purchased), was_bought],
                             dtype=np.float32)
            row = np.concatenate([base, extra, card_vec, relic_vec])
            rows.append(row)
            labels.append(1 if d["victory"] else 0)

    return np.array(rows), np.array(labels)


def shop_feature_names(vocab: dict) -> list[str]:
    names = base_feature_names(vocab)
    names += ["gold", "item_buy_wr", "item_skip_wr", "item_times_purchased", "was_bought"]
    names += [f"item_card_{c}" for c in vocab["card_vocab"]]
    names += [f"item_relic_{r}" for r in vocab["relic_vocab"]]
    return names


def shop_inference_features(floor: int, act: int, hp_pct: int, gold: int,
                            deck: list[str], relics: list[str],
                            items: list[str],
                            stats: dict, vocab: dict,
                            num_upgrades: int = 0,
                            deck_upgrades: dict | None = None) -> np.ndarray:
    card_to_idx = vocab["card_to_idx"]
    relic_to_idx = vocab["relic_to_idx"]
    base = base_features(floor, act, hp_pct, len(deck), len(relics), deck, relics, vocab,
                         num_upgrades, deck_upgrades)
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
        rows.append(np.concatenate([base, extra, card_vec, relic_vec]))

    # 不购买
    card_vec = np.zeros(len(card_to_idx), dtype=np.float32)
    relic_vec = np.zeros(len(relic_to_idx), dtype=np.float32)
    extra = np.array([gold, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    rows.append(np.concatenate([base, extra, card_vec, relic_vec]))

    return np.array(rows)


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train_model(X: np.ndarray, y: np.ndarray, model_name: str,
                feature_names: list[str] | None = None) -> dict:
    results = {}
    n_folds = 5
    if len(np.unique(y)) < 2:
        print(f"  警告: {model_name} 标签只有一个类别，跳过训练")
        return results

    min_class_count = min(np.sum(y == 0), np.sum(y == 1))
    actual_folds = min(n_folds, int(min_class_count))
    if actual_folds < 2:
        actual_folds = 2

    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)

    for engine_name, engine in [("xgboost", xgb), ("lightgbm", lgb)]:
        if engine is None:
            print(f"  {engine_name} 未安装，跳过")
            continue

        oof_preds = np.zeros(len(y))
        models = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if engine_name == "xgboost":
                model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    eval_metric="logloss",
                    verbosity=0,
                    device="cuda",
                )
                model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)],
                          verbose=False)
            else:
                model = lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=10,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    verbose=-1,
                    device="gpu",
                )
                model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)],
                          callbacks=[lgb.log_evaluation(period=0)])

            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            models.append(model)

        auc = roc_auc_score(y, oof_preds)
        acc = accuracy_score(y, (oof_preds > 0.5).astype(int))
        print(f"  {engine_name}: AUC={auc:.4f}, Accuracy={acc:.4f}")

        if feature_names and engine_name == "xgboost":
            importances = np.mean([m.feature_importances_ for m in models], axis=0)
            top_idx = np.argsort(importances)[-10:][::-1]
            print(f"  Top 10 特征:")
            for idx in top_idx:
                if importances[idx] > 0:
                    print(f"    {feature_names[idx]}: {importances[idx]:.4f}")

        results[engine_name] = {
            "models": models,
            "auc": auc,
            "accuracy": acc,
        }

    return results


def predict_with_models(models_dict: dict, X: np.ndarray) -> dict[str, np.ndarray]:
    preds = {}
    for engine_name, info in models_dict.items():
        fold_preds = [m.predict_proba(X)[:, 1] for m in info["models"]]
        preds[engine_name] = np.mean(fold_preds, axis=0)
    return preds


# ---------------------------------------------------------------------------
# 训练入口
# ---------------------------------------------------------------------------

def run_training():
    print("加载数据库...")
    db = load_db()

    print("构建词表...")
    vocab = build_vocabularies(db)
    print(f"  卡牌词表: {len(vocab['card_vocab'])} 张")
    print(f"  遗物词表: {len(vocab['relic_vocab'])} 个")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    vocab_path = MODEL_DIR / "vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"词表已保存到 {vocab_path}")

    print("\n=== 训练卡牌决策模型 ===")
    X_card, y_card = build_card_features(db, vocab)
    print(f"  样本数: {len(y_card)}, 正样本: {y_card.sum()}, 负样本: {len(y_card) - y_card.sum()}")
    card_models = train_model(X_card, y_card, "card", card_feature_names(vocab))
    save_models(card_models, "card")

    print("\n=== 训练篝火决策模型 ===")
    X_camp, y_camp = build_campfire_features(db, vocab)
    print(f"  样本数: {len(y_camp)}, 正样本: {y_camp.sum()}, 负样本: {len(y_camp) - y_camp.sum()}")
    camp_models = train_model(X_camp, y_camp, "campfire", campfire_feature_names(vocab))
    save_models(camp_models, "campfire")

    print("\n=== 训练 Boss 遗物决策模型 ===")
    X_boss, y_boss = build_boss_relic_features(db, vocab)
    print(f"  样本数: {len(y_boss)}, 正样本: {y_boss.sum()}, 负样本: {len(y_boss) - y_boss.sum()}")
    boss_models = train_model(X_boss, y_boss, "boss_relic", boss_relic_feature_names(vocab))
    save_models(boss_models, "boss_relic")

    print("\n=== 训练商店决策模型 ===")
    X_shop, y_shop = build_shop_features(db, vocab)
    print(f"  样本数: {len(y_shop)}, 正样本: {y_shop.sum()}, 负样本: {len(y_shop) - y_shop.sum()}")
    shop_models = train_model(X_shop, y_shop, "shop", shop_feature_names(vocab))
    save_models(shop_models, "shop")

    print("\n训练完成！模型已保存到", MODEL_DIR)


def save_models(models_dict: dict, name: str):
    for engine_name, info in models_dict.items():
        path = MODEL_DIR / f"{name}_{engine_name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(info["models"], f)
        print(f"  已保存: {path}")


def load_models(name: str) -> dict:
    result = {}
    for engine_name in ["xgboost", "lightgbm"]:
        path = MODEL_DIR / f"{name}_{engine_name}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                result[engine_name] = {"models": pickle.load(f)}
    if not result:
        print(f"未找到 {name} 模型，请先运行: python -m watcher_advisor.ml_advisor train")
        sys.exit(1)
    return result


def load_vocab() -> dict:
    vocab_path = MODEL_DIR / "vocab.pkl"
    if not vocab_path.exists():
        print(f"未找到词表文件，请先运行: python -m watcher_advisor.ml_advisor train")
        sys.exit(1)
    with open(vocab_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# 推理 & 输出
# ---------------------------------------------------------------------------

def print_recommendations(options: list[str], preds: dict[str, np.ndarray], title: str):
    print(f"\n=== 观者 ML 决策建议 ===")

    for engine_name, probs in preds.items():
        engine_label = "XGBoost" if engine_name == "xgboost" else "LightGBM"
        print(f"\n模型: {engine_label}")
        print(f"{'候选项':<20s} {'预测胜率':>10s}  {'推荐':>4s}")
        print("-" * 40)

        ranked = sorted(zip(options, probs), key=lambda x: -x[1])
        best_prob = ranked[0][1]
        for opt, prob in ranked:
            star = "★ 推荐" if prob == best_prob else ""
            print(f"{opt:<20s} {prob*100:>8.1f}%  {star}")


def infer_card(args, db: dict, vocab: dict):
    deck = parse_deck(args.deck)
    relics = parse_list(args.relics)
    options = parse_list(args.options)
    if "SKIP" not in options:
        options.append("SKIP")

    hp_pct = round(args.hp / args.max_hp * 100) if args.max_hp > 0 else 0
    stats = db["card_decisions"]["stats"]

    X = card_inference_features(args.floor, args.act, hp_pct, deck, relics, options, stats, vocab)
    models = load_models("card")
    preds = predict_with_models(models, X)
    print_recommendations(options, preds, "卡牌选择")


def infer_campfire(args, db: dict, vocab: dict):
    deck = parse_deck(args.deck)
    relics = parse_list(args.relics)
    hp_pct = round(args.hp / args.max_hp * 100) if args.max_hp > 0 else 0

    X = campfire_inference_features(args.floor, args.act, hp_pct, deck, relics, vocab)
    models = load_models("campfire")
    preds = predict_with_models(models, X)
    print_recommendations(["REST", "SMITH"], preds, "篝火决策")


def infer_boss_relic(args, db: dict, vocab: dict):
    deck = parse_deck(args.deck)
    relics = parse_list(args.relics)
    options = parse_list(args.options)
    hp_pct = round(args.hp / args.max_hp * 100) if args.max_hp > 0 else 0
    stats = db["boss_relic_decisions"]["stats"]

    X = boss_relic_inference_features(args.act, hp_pct, deck, relics, options, stats, vocab)
    models = load_models("boss_relic")
    preds = predict_with_models(models, X)
    print_recommendations(options, preds, "Boss 遗物选择")


def infer_shop(args, db: dict, vocab: dict):
    deck = parse_deck(args.deck)
    relics = parse_list(args.relics)
    avail_cards = parse_list(args.cards)
    avail_relics = parse_list(args.shop_relics)
    avail_potions = parse_list(args.potions)
    hp_pct = round(args.hp / args.max_hp * 100) if args.max_hp > 0 else 0
    stats = db["shop_decisions"]["stats"]

    all_items = avail_cards + avail_relics + avail_potions
    X = shop_inference_features(
        args.floor, args.act, hp_pct, args.gold,
        deck, relics, all_items, stats, vocab
    )
    option_labels = all_items + ["不购买"]

    models = load_models("shop")
    preds = predict_with_models(models, X)
    print_recommendations(option_labels, preds, "商店决策")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="观者 ML 决策顾问 (XGBoost/LightGBM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subs = parser.add_subparsers(dest="command", required=True)

    subs.add_parser("train", help="训练所有模型")

    cp = subs.add_parser("card", help="卡牌奖励决策")
    cp.add_argument("--floor", type=int, required=True)
    cp.add_argument("--act", type=int, required=True)
    cp.add_argument("--hp", type=int, required=True)
    cp.add_argument("--max-hp", type=int, required=True, dest="max_hp")
    cp.add_argument("--relics", default="")
    cp.add_argument("--deck", default="", help="卡牌逗号分隔，支持 'Card x3' 格式")
    cp.add_argument("--options", required=True, help="候选卡牌，逗号分隔")

    fp = subs.add_parser("campfire", help="篝火: 休息还是升级?")
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
    sp.add_argument("--cards", default="", help="商店中的卡牌")
    sp.add_argument("--shop-relics", default="", dest="shop_relics")
    sp.add_argument("--potions", default="")

    args = parser.parse_args()

    if args.command == "train":
        run_training()
        return

    db = load_db()
    vocab = load_vocab()

    if args.command == "card":
        infer_card(args, db, vocab)
    elif args.command == "campfire":
        infer_campfire(args, db, vocab)
    elif args.command == "boss-relic":
        infer_boss_relic(args, db, vocab)
    elif args.command == "shop":
        infer_shop(args, db, vocab)


if __name__ == "__main__":
    main()
