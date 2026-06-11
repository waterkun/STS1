#!/usr/bin/env python3
"""
Defect ML Advisor V3 - Transformer 排序模型

基于 PyTorch 实现的 Transformer 编码器，通过自注意力让候选选项
互相感知，捕捉选项间的相对价值和协同关系。

损失函数: Pairwise Margin Loss (softplus)，按 ascension_level × victory 加权。
LR 调度: 5 epoch warmup + cosine decay。

与 V1/V2 的区别:
  - V1/V2: 每个选项独立打分（无法感知"这次还有 Echo Form 可选"）
  - V3:    同一决策内各选项互相注意，打分受整个候选集影响

Usage:
  python -m defect_advisor.ml_advisor_v3 train
  python -m defect_advisor.ml_advisor_v3 card \
    --floor 8 --act 1 --hp 45 --max-hp 75 \
    --relics "Cracked Core" \
    --deck "Strike_B x4,Defend_B x4,Zap,Dualcast" \
    --options "Claw,Cold Snap,Compile Driver"
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# ---- 从 V1/V2 复用特征工程 ----
from defect_advisor.ml_advisor import (
    load_db,
    build_vocabularies,
    load_vocab,
    parse_deck,
    parse_list,
    MODEL_DIR,
)
from defect_advisor.ml_advisor_v2 import (
    build_card_ranking_data,
    build_boss_relic_ranking_data,
    build_campfire_ranking_data,
    build_shop_ranking_data,
    card_inference_features_v2,
    boss_relic_inference_features_v2,
    campfire_inference_features_v2,
    shop_inference_features_v2,
    _apply_duplicate_power_penalty,
    _deck_size_bucket,
    _POWER_CARDS,
)

# ---- Transformer 核心 ----
from transformer_core import (
    STSTransformerRanker,
    decisions_from_ranking_data,
    train_transformer,
)

_V3_MODEL_NAMES = [
    "card_transformer_v3",
    "boss_relic_transformer_v3",
    "campfire_transformer_v3",
    "shop_transformer_v3",
]


# ============================================================
# 保存 / 加载
# ============================================================

def save_v3_model(model, name: str):
    path = MODEL_DIR / f"{name}.pt"
    torch.save({
        "config": {
            "input_dim": model.input_dim,
            "d_model": model.d_model,
            "n_heads": model.n_heads,
            "d_ff": model.d_ff,
            "n_layers": model.n_layers,
            "dropout": model.dropout,
        },
        "state_dict": model.state_dict(),
    }, path)
    print(f"  已保存: {path}")


def load_v3_models() -> dict:
    """加载所有 V3 Transformer 模型，缺失的跳过。"""
    result = {}
    for name in _V3_MODEL_NAMES:
        path = MODEL_DIR / f"{name}.pt"
        if path.exists():
            data = torch.load(path, map_location="cpu", weights_only=False)
            cfg = data["config"]
            model = STSTransformerRanker(**cfg)
            model.load_state_dict(data["state_dict"])
            model.eval()
            result[name] = model
    return result


# ============================================================
# 训练入口
# ============================================================

def run_training_v3():
    print("加载数据库...")
    db = load_db()

    print("构建词表...")
    vocab = build_vocabularies(db)
    print(f"  卡牌词表: {len(vocab['card_vocab'])} 张")
    print(f"  遗物词表: {len(vocab['relic_vocab'])} 个")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 保存词表（与 V1/V2 共享）
    vocab_path = MODEL_DIR / "vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    import datetime
    def _ts():
        return datetime.datetime.now().strftime("%H:%M:%S")

    # --- 卡牌 ---
    print(f"\n=== 训练卡牌 V3 Transformer ===", flush=True)
    print(f"  [{_ts()}] 构建卡牌排序数据...", flush=True)
    X, y, groups, dw = build_card_ranking_data(db, vocab)
    print(f"  [{_ts()}] 排序数据: {len(y)} 行, {len(groups)} 组", flush=True)
    dX, dy, dWeights = decisions_from_ranking_data(X, y, groups, dw)
    model = train_transformer(dX, dy, name="card", decisions_w=dWeights)
    if model:
        save_v3_model(model, "card_transformer_v3")

    # --- Boss 遗物 ---
    print(f"\n=== 训练 Boss 遗物 V3 Transformer ===", flush=True)
    print(f"  [{_ts()}] 构建 Boss 遗物排序数据...", flush=True)
    X, y, groups, dw = build_boss_relic_ranking_data(db, vocab)
    print(f"  [{_ts()}] 排序数据: {len(y)} 行, {len(groups)} 组", flush=True)
    dX, dy, dWeights = decisions_from_ranking_data(X, y, groups, dw)
    model = train_transformer(dX, dy, name="boss_relic", decisions_w=dWeights)
    if model:
        save_v3_model(model, "boss_relic_transformer_v3")

    # --- 篝火 ---
    print(f"\n=== 训练篝火 V3 Transformer ===", flush=True)
    print(f"  [{_ts()}] 构建篝火排序数据...", flush=True)
    X, y, groups, dw = build_campfire_ranking_data(db, vocab)
    print(f"  [{_ts()}] 排序数据: {len(y)} 行, {len(groups)} 组", flush=True)
    dX, dy, dWeights = decisions_from_ranking_data(X, y, groups, dw)
    model = train_transformer(dX, dy, name="campfire", decisions_w=dWeights)
    if model:
        save_v3_model(model, "campfire_transformer_v3")

    # --- 商店 ---
    print(f"\n=== 训练商店 V3 Transformer ===", flush=True)
    print(f"  [{_ts()}] 构建商店排序数据...", flush=True)
    X, y, groups, dw = build_shop_ranking_data(db, vocab)
    print(f"  [{_ts()}] 排序数据: {len(y)} 行, {len(groups)} 组", flush=True)
    dX, dy, dWeights = decisions_from_ranking_data(X, y, groups, dw)
    model = train_transformer(dX, dy, name="shop", decisions_w=dWeights)
    if model:
        save_v3_model(model, "shop_transformer_v3")

    print("\nV3 训练完成！", flush=True)


# ============================================================
# 推理函数（返回 {"transformer": np.ndarray}）
# ============================================================

def predict_all_card(options: list[str], floor: int, act: int, hp_pct: int,
                     deck: list[str], relics: list[str],
                     db: dict, vocab: dict, v3_models: dict,
                     num_upgrades: int = 0,
                     deck_upgrades: dict | None = None) -> dict[str, np.ndarray]:
    if "card_transformer_v3" not in v3_models:
        return {}

    stats = db["card_decisions"]["stats"]
    X = card_inference_features_v2(floor, act, hp_pct, deck, relics,
                                   options, stats, vocab,
                                   num_upgrades, deck_upgrades)
    X = X.astype(np.float32)

    model: STSTransformerRanker = v3_models["card_transformer_v3"]
    probs = model.predict(X)

    preds = {"transformer": probs}
    preds = _apply_duplicate_power_penalty(options, deck, preds)
    return preds


def predict_all_campfire(floor: int, act: int, hp_pct: int,
                         deck: list[str], relics: list[str],
                         db: dict, vocab: dict, v3_models: dict,
                         num_upgrades: int = 0,
                         deck_upgrades: dict | None = None) -> dict[str, np.ndarray]:
    if "campfire_transformer_v3" not in v3_models:
        return {}

    X = campfire_inference_features_v2(floor, act, hp_pct, deck, relics, vocab,
                                       num_upgrades, deck_upgrades)
    X = X.astype(np.float32)

    model: STSTransformerRanker = v3_models["campfire_transformer_v3"]
    probs = model.predict(X)
    return {"transformer": probs}


def predict_all_boss_relic(options: list[str], act: int, hp_pct: int,
                           deck: list[str], relics: list[str],
                           db: dict, vocab: dict, v3_models: dict,
                           num_upgrades: int = 0,
                           deck_upgrades: dict | None = None) -> dict[str, np.ndarray]:
    if "boss_relic_transformer_v3" not in v3_models:
        return {}

    stats = db["boss_relic_decisions"]["stats"]
    X = boss_relic_inference_features_v2(act, hp_pct, deck, relics,
                                         options, stats, vocab,
                                         num_upgrades, deck_upgrades)
    X = X.astype(np.float32)

    model: STSTransformerRanker = v3_models["boss_relic_transformer_v3"]
    probs = model.predict(X)
    return {"transformer": probs}


def predict_all_shop(option_labels: list[str], floor: int, act: int, hp_pct: int,
                     gold: int, deck: list[str], relics: list[str],
                     items: list[str],
                     db: dict, vocab: dict, v3_models: dict,
                     num_upgrades: int = 0,
                     deck_upgrades: dict | None = None) -> dict[str, np.ndarray]:
    if "shop_transformer_v3" not in v3_models:
        return {}

    stats = db["shop_decisions"]["stats"]
    X = shop_inference_features_v2(floor, act, hp_pct, gold, deck, relics,
                                   items, stats, vocab,
                                   num_upgrades, deck_upgrades)
    X = X.astype(np.float32)

    model: STSTransformerRanker = v3_models["shop_transformer_v3"]
    probs = model.predict(X)
    return {"transformer": probs}


# ============================================================
# CLI 推理
# ============================================================

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


def _print_predictions(options, preds):
    header = _pad_right("候选项", _NAME_COL) + "  Transformer  推荐"
    print(header)
    print("-" * _display_width(header))

    probs = preds.get("transformer", np.zeros(len(options)))
    order = np.argsort(-probs)
    for rank, i in enumerate(order):
        star = "  ★" if rank == 0 else ""
        print(f"{_pad_right(options[i], _NAME_COL)}  {probs[i]*100:>10.1f}%{star}")


def infer_card(args):
    db = load_db()
    vocab = load_vocab()
    deck = parse_deck(args.deck)
    relics = parse_list(args.relics)
    options = parse_list(args.options)
    if "SKIP" not in options:
        options.append("SKIP")
    hp_pct = round(args.hp / args.max_hp * 100) if args.max_hp > 0 else 0

    v3_models = load_v3_models()
    preds = predict_all_card(options, args.floor, args.act, hp_pct,
                             deck, relics, db, vocab, v3_models)
    print(f"\n=== 机器人 V3 Transformer 决策建议 (卡牌选择) ===")
    print(f"Floor {args.floor} Act {args.act} | HP: {args.hp}/{args.max_hp} ({hp_pct}%)")
    _print_predictions(options, preds)


def infer_campfire(args):
    db = load_db()
    vocab = load_vocab()
    deck = parse_deck(args.deck)
    relics = parse_list(args.relics)
    hp_pct = round(args.hp / args.max_hp * 100) if args.max_hp > 0 else 0

    v3_models = load_v3_models()
    preds = predict_all_campfire(args.floor, args.act, hp_pct,
                                 deck, relics, db, vocab, v3_models)
    print(f"\n=== 机器人 V3 Transformer 决策建议 (篝火决策) ===")
    print(f"Floor {args.floor} Act {args.act} | HP: {args.hp}/{args.max_hp} ({hp_pct}%)")
    _print_predictions(["REST", "SMITH"], preds)


def infer_boss_relic(args):
    db = load_db()
    vocab = load_vocab()
    deck = parse_deck(args.deck)
    relics = parse_list(args.relics)
    options = parse_list(args.options)
    hp_pct = round(args.hp / args.max_hp * 100) if args.max_hp > 0 else 0

    v3_models = load_v3_models()
    preds = predict_all_boss_relic(options, args.act, hp_pct,
                                   deck, relics, db, vocab, v3_models)
    print(f"\n=== 机器人 V3 Transformer 决策建议 (Boss 遗物选择) ===")
    print(f"Act {args.act} | HP: {args.hp}/{args.max_hp} ({hp_pct}%)")
    _print_predictions(options, preds)


def infer_shop(args):
    db = load_db()
    vocab = load_vocab()
    deck = parse_deck(args.deck)
    relics = parse_list(args.relics)
    avail_cards = parse_list(args.cards)
    avail_relics = parse_list(args.shop_relics)
    avail_potions = parse_list(args.potions)
    hp_pct = round(args.hp / args.max_hp * 100) if args.max_hp > 0 else 0

    all_items = avail_cards + avail_relics + avail_potions
    option_labels = all_items + ["移除卡牌", "不购买"]

    v3_models = load_v3_models()
    preds = predict_all_shop(option_labels, args.floor, args.act, hp_pct,
                             args.gold, deck, relics, all_items,
                             db, vocab, v3_models)
    print(f"\n=== 机器人 V3 Transformer 决策建议 (商店) ===")
    print(f"Floor {args.floor} Act {args.act} | HP: {args.hp}/{args.max_hp} ({hp_pct}%) | Gold: {args.gold}")
    _print_predictions(option_labels, preds)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="机器人 ML 决策顾问 V3 (Transformer)",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    subs.add_parser("train", help="训练所有 V3 Transformer 模型")

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
        run_training_v3()
    elif args.command == "card":
        infer_card(args)
    elif args.command == "campfire":
        infer_campfire(args)
    elif args.command == "boss-relic":
        infer_boss_relic(args)
    elif args.command == "shop":
        infer_shop(args)


if __name__ == "__main__":
    main()
