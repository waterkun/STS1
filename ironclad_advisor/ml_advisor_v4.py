#!/usr/bin/env python3
"""
Ironclad ML Advisor V4 - A20 胜局专用 Transformer

仅使用 Ascension 20 + 胜利对局数据训练，专注学习顶级策略。
从不从失败对局学习，标签: 选了=2, 未选=0（无 label=1）。

与 V3 的区别:
  - V3: 全部对局，胜=2/败=1/未选=0，按 asc×胜负 加权
  - V4: 仅 A20 胜局，选了=2/未选=0，无加权（数据本身已是最高质量）

Usage:
  python -m ironclad_advisor.ml_advisor_v4 train
  python -m ironclad_advisor.ml_advisor_v4 card \\
    --floor 8 --act 1 --hp 45 --max-hp 80 \\
    --relics "Burning Blood" \\
    --deck "Strike_R x4,Defend_R x4,Bash" \\
    --options "Barricade,Feel No Pain,Pommel Strike"
"""

import argparse
import pickle
import sys

import numpy as np
import torch

from ironclad_advisor.ml_advisor import (
    load_db,
    build_vocabularies,
    load_vocab,
    parse_deck,
    parse_list,
    MODEL_DIR,
)
from ironclad_advisor.ml_advisor_v2 import (
    build_card_ranking_data,
    build_boss_relic_ranking_data,
    build_campfire_ranking_data,
    build_shop_ranking_data,
    card_inference_features_v2,
    boss_relic_inference_features_v2,
    campfire_inference_features_v2,
    shop_inference_features_v2,
    _apply_duplicate_power_penalty,
)

from transformer_core import (
    STSTransformerRanker,
    train_transformer,
)

_V4_MODEL_NAMES = [
    "card_transformer_v4",
    "boss_relic_transformer_v4",
    "campfire_transformer_v4",
    "shop_transformer_v4",
]


# ============================================================
# A20 胜局数据过滤
# ============================================================

def _filter_a20_wins(X, y, groups, decisions_w):
    """从 V2 排序数据中过滤出 A20 胜利对局，并将标签统一为 2/0。

    V2 标签: 2=选了且赢, 1=选了且输, 0=未选。
    过滤后所有对局均为胜局，picked → 2, not_picked → 0。
    """
    keep_X, keep_y = [], []
    start = 0
    for i, g in enumerate(groups):
        g = int(g)
        dw = decisions_w[i]
        if dw.get("ascension_level", 0) == 20 and dw.get("victory", False):
            seg_X = X[start:start + g].astype(np.float32)
            seg_y = np.where(y[start:start + g] > 0, 2, 0).astype(np.float32)
            keep_X.append(seg_X)
            keep_y.append(seg_y)
        start += g
    return keep_X, keep_y


# ============================================================
# 保存 / 加载
# ============================================================

def save_v4_model(model, name: str):
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


def load_v4_models() -> dict:
    result = {}
    for name in _V4_MODEL_NAMES:
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
# 训练
# ============================================================

def run_training_v4():
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

    print("\n=== 训练卡牌 V4 Transformer (A20 胜局) ===")
    X, y, groups, dw = build_card_ranking_data(db, vocab)
    dX, dy = _filter_a20_wins(X, y, groups, dw)
    print(f"  A20 胜局决策数: {len(dX)}")
    model = train_transformer(dX, dy, name="card_v4")
    if model:
        save_v4_model(model, "card_transformer_v4")

    print("\n=== 训练 Boss 遗物 V4 Transformer (A20 胜局) ===")
    X, y, groups, dw = build_boss_relic_ranking_data(db, vocab)
    dX, dy = _filter_a20_wins(X, y, groups, dw)
    print(f"  A20 胜局决策数: {len(dX)}")
    model = train_transformer(dX, dy, name="boss_relic_v4")
    if model:
        save_v4_model(model, "boss_relic_transformer_v4")

    print("\n=== 训练篝火 V4 Transformer (A20 胜局) ===")
    X, y, groups, dw = build_campfire_ranking_data(db, vocab)
    dX, dy = _filter_a20_wins(X, y, groups, dw)
    print(f"  A20 胜局决策数: {len(dX)}")
    model = train_transformer(dX, dy, name="campfire_v4")
    if model:
        save_v4_model(model, "campfire_transformer_v4")

    print("\n=== 训练商店 V4 Transformer (A20 胜局) ===")
    X, y, groups, dw = build_shop_ranking_data(db, vocab)
    dX, dy = _filter_a20_wins(X, y, groups, dw)
    print(f"  A20 胜局决策数: {len(dX)}")
    model = train_transformer(dX, dy, name="shop_v4")
    if model:
        save_v4_model(model, "shop_transformer_v4")

    print("\nV4 训练完成！")


# ============================================================
# 推理
# ============================================================

def predict_all_card(options: list[str], floor: int, act: int, hp_pct: int,
                     deck: list[str], relics: list[str],
                     db: dict, vocab: dict, v4_models: dict,
                     num_upgrades: int = 0,
                     deck_upgrades: dict | None = None) -> dict[str, np.ndarray]:
    if "card_transformer_v4" not in v4_models:
        return {}
    stats = db["card_decisions"]["stats"]
    X = card_inference_features_v2(floor, act, hp_pct, deck, relics,
                                   options, stats, vocab,
                                   num_upgrades, deck_upgrades).astype(np.float32)
    probs = v4_models["card_transformer_v4"].predict(X)
    preds = {"transformer_v4": probs}
    preds = _apply_duplicate_power_penalty(options, deck, preds)
    return preds


def predict_all_campfire(floor: int, act: int, hp_pct: int,
                         deck: list[str], relics: list[str],
                         db: dict, vocab: dict, v4_models: dict,
                         num_upgrades: int = 0,
                         deck_upgrades: dict | None = None) -> dict[str, np.ndarray]:
    if "campfire_transformer_v4" not in v4_models:
        return {}
    X = campfire_inference_features_v2(floor, act, hp_pct, deck, relics, vocab,
                                       num_upgrades, deck_upgrades).astype(np.float32)
    probs = v4_models["campfire_transformer_v4"].predict(X)
    return {"transformer_v4": probs}


def predict_all_boss_relic(options: list[str], act: int, hp_pct: int,
                           deck: list[str], relics: list[str],
                           db: dict, vocab: dict, v4_models: dict,
                           num_upgrades: int = 0,
                           deck_upgrades: dict | None = None) -> dict[str, np.ndarray]:
    if "boss_relic_transformer_v4" not in v4_models:
        return {}
    stats = db["boss_relic_decisions"]["stats"]
    X = boss_relic_inference_features_v2(act, hp_pct, deck, relics,
                                         options, stats, vocab,
                                         num_upgrades, deck_upgrades).astype(np.float32)
    probs = v4_models["boss_relic_transformer_v4"].predict(X)
    return {"transformer_v4": probs}


def predict_all_shop(option_labels: list[str], floor: int, act: int, hp_pct: int,
                     gold: int, deck: list[str], relics: list[str],
                     items: list[str],
                     db: dict, vocab: dict, v4_models: dict,
                     num_upgrades: int = 0,
                     deck_upgrades: dict | None = None) -> dict[str, np.ndarray]:
    if "shop_transformer_v4" not in v4_models:
        return {}
    stats = db["shop_decisions"]["stats"]
    X = shop_inference_features_v2(floor, act, hp_pct, gold, deck, relics,
                                   items, stats, vocab,
                                   num_upgrades, deck_upgrades).astype(np.float32)
    probs = v4_models["shop_transformer_v4"].predict(X)
    return {"transformer_v4": probs}


# ============================================================
# CLI
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
    header = _pad_right("候选项", _NAME_COL) + "  TransV4  推荐"
    print(header)
    print("-" * _display_width(header))
    probs = preds.get("transformer_v4", np.zeros(len(options)))
    for rank, i in enumerate(np.argsort(-probs)):
        star = "  ★" if rank == 0 else ""
        print(f"{_pad_right(options[i], _NAME_COL)}  {probs[i]*100:>7.1f}%{star}")


def main():
    parser = argparse.ArgumentParser(description="铁甲战士 ML 决策顾问 V4 (A20 胜局 Transformer)")
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
        run_training_v4()
        return

    db = load_db()
    vocab = load_vocab()
    v4_models = load_v4_models()
    deck = parse_deck(args.deck)
    relics = parse_list(args.relics)
    hp_pct = round(args.hp / args.max_hp * 100) if args.max_hp > 0 else 0

    if args.command == "card":
        options = parse_list(args.options)
        if "SKIP" not in options:
            options.append("SKIP")
        preds = predict_all_card(options, args.floor, args.act, hp_pct,
                                 deck, relics, db, vocab, v4_models)
        print(f"\n=== 铁甲战士 V4 Transformer 决策建议 (卡牌选择) ===")
        print(f"Floor {args.floor} Act {args.act} | HP: {args.hp}/{args.max_hp} ({hp_pct}%)")
        _print_predictions(options, preds)
    elif args.command == "campfire":
        preds = predict_all_campfire(args.floor, args.act, hp_pct,
                                     deck, relics, db, vocab, v4_models)
        print(f"\n=== 铁甲战士 V4 Transformer 决策建议 (篝火决策) ===")
        print(f"Floor {args.floor} Act {args.act} | HP: {args.hp}/{args.max_hp} ({hp_pct}%)")
        _print_predictions(["REST", "SMITH"], preds)
    elif args.command == "boss-relic":
        options = parse_list(args.options)
        preds = predict_all_boss_relic(options, args.act, hp_pct,
                                       deck, relics, db, vocab, v4_models)
        print(f"\n=== 铁甲战士 V4 Transformer 决策建议 (Boss 遗物选择) ===")
        print(f"Act {args.act} | HP: {args.hp}/{args.max_hp} ({hp_pct}%)")
        _print_predictions(options, preds)
    elif args.command == "shop":
        all_items = parse_list(args.cards) + parse_list(args.shop_relics) + parse_list(args.potions)
        option_labels = all_items + ["移除卡牌", "不购买"]
        preds = predict_all_shop(option_labels, args.floor, args.act, hp_pct,
                                 args.gold, deck, relics, all_items,
                                 db, vocab, v4_models)
        print(f"\n=== 铁甲战士 V4 Transformer 决策建议 (商店) ===")
        print(f"Floor {args.floor} Act {args.act} | HP: {args.hp}/{args.max_hp} ({hp_pct}%) | Gold: {args.gold}")
        _print_predictions(option_labels, preds)


if __name__ == "__main__":
    main()
