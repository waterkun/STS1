#!/usr/bin/env python3
"""
Silent ML Advisor V3 - Transformer 排序模型

Usage:
  python -m silent_advisor.ml_advisor_v3 train
  python -m silent_advisor.ml_advisor_v3 card \
    --floor 8 --act 1 --hp 45 --max-hp 80 \
    --relics "Ring of the Snake" \
    --deck "Strike_G x4,Defend_G x4,Neutralize,Survivor" \
    --options "Backflip,Deadly Poison,Blade Dance"
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from silent_advisor.ml_advisor import (
    load_db,
    build_vocabularies,
    load_vocab,
    parse_deck,
    parse_list,
    MODEL_DIR,
)
from silent_advisor.ml_advisor_v2 import (
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
    decisions_from_ranking_data,
    train_transformer,
)

_V3_MODEL_NAMES = [
    "card_transformer_v3",
    "boss_relic_transformer_v3",
    "campfire_transformer_v3",
    "shop_transformer_v3",
]


def save_v3_model(model, name: str):
    path = MODEL_DIR / f"{name}.pt"
    torch.save({
        "config": {
            "input_dim": model.input_dim,
            "d_model": model.d_model,
            "n_heads": model.n_heads,
            "d_ff": model.d_ff,
            "n_layers": model.n_layers,
        },
        "state_dict": model.state_dict(),
    }, path)
    print(f"  已保存: {path}")


def load_v3_models() -> dict:
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


def run_training_v3():
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

    print("\n=== 训练卡牌 V3 Transformer ===")
    X, y, groups = build_card_ranking_data(db, vocab)
    print(f"  排序数据: {len(y)} 行, {len(groups)} 组")
    dX, dy = decisions_from_ranking_data(X, y, groups)
    model = train_transformer(dX, dy, name="card")
    if model:
        save_v3_model(model, "card_transformer_v3")

    print("\n=== 训练 Boss 遗物 V3 Transformer ===")
    X, y, groups = build_boss_relic_ranking_data(db, vocab)
    print(f"  排序数据: {len(y)} 行, {len(groups)} 组")
    dX, dy = decisions_from_ranking_data(X, y, groups)
    model = train_transformer(dX, dy, name="boss_relic")
    if model:
        save_v3_model(model, "boss_relic_transformer_v3")

    print("\n=== 训练篝火 V3 Transformer ===")
    X, y, groups = build_campfire_ranking_data(db, vocab)
    print(f"  排序数据: {len(y)} 行, {len(groups)} 组")
    dX, dy = decisions_from_ranking_data(X, y, groups)
    model = train_transformer(dX, dy, name="campfire")
    if model:
        save_v3_model(model, "campfire_transformer_v3")

    print("\n=== 训练商店 V3 Transformer ===")
    X, y, groups = build_shop_ranking_data(db, vocab)
    print(f"  排序数据: {len(y)} 行, {len(groups)} 组")
    dX, dy = decisions_from_ranking_data(X, y, groups)
    model = train_transformer(dX, dy, name="shop")
    if model:
        save_v3_model(model, "shop_transformer_v3")

    print("\nV3 训练完成！")


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
                                   num_upgrades, deck_upgrades).astype(np.float32)
    probs = v3_models["card_transformer_v3"].predict(X)
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
                                       num_upgrades, deck_upgrades).astype(np.float32)
    probs = v3_models["campfire_transformer_v3"].predict(X)
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
                                         num_upgrades, deck_upgrades).astype(np.float32)
    probs = v3_models["boss_relic_transformer_v3"].predict(X)
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
                                   num_upgrades, deck_upgrades).astype(np.float32)
    probs = v3_models["shop_transformer_v3"].predict(X)
    return {"transformer": probs}


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
    for rank, i in enumerate(np.argsort(-probs)):
        star = "  ★" if rank == 0 else ""
        print(f"{_pad_right(options[i], _NAME_COL)}  {probs[i]*100:>10.1f}%{star}")


def main():
    parser = argparse.ArgumentParser(description="静默猎手 ML 决策顾问 V3 (Transformer)")
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
        run_training_v3()
        return

    db = load_db()
    vocab = load_vocab()
    v3_models = load_v3_models()
    deck = parse_deck(args.deck)
    relics = parse_list(args.relics)
    hp_pct = round(args.hp / args.max_hp * 100) if args.max_hp > 0 else 0

    if args.command == "card":
        options = parse_list(args.options)
        if "SKIP" not in options:
            options.append("SKIP")
        preds = predict_all_card(options, args.floor, args.act, hp_pct,
                                 deck, relics, db, vocab, v3_models)
        print(f"\n=== 静默猎手 V3 Transformer (卡牌选择) ===")
        _print_predictions(options, preds)
    elif args.command == "campfire":
        preds = predict_all_campfire(args.floor, args.act, hp_pct,
                                     deck, relics, db, vocab, v3_models)
        print(f"\n=== 静默猎手 V3 Transformer (篝火决策) ===")
        _print_predictions(["REST", "SMITH"], preds)
    elif args.command == "boss-relic":
        options = parse_list(args.options)
        preds = predict_all_boss_relic(options, args.act, hp_pct,
                                       deck, relics, db, vocab, v3_models)
        print(f"\n=== 静默猎手 V3 Transformer (Boss 遗物) ===")
        _print_predictions(options, preds)
    elif args.command == "shop":
        all_items = parse_list(args.cards) + parse_list(args.shop_relics) + parse_list(args.potions)
        option_labels = all_items + ["不购买"]
        preds = predict_all_shop(option_labels, args.floor, args.act, hp_pct,
                                 args.gold, deck, relics, all_items,
                                 db, vocab, v3_models)
        print(f"\n=== 静默猎手 V3 Transformer (商店) ===")
        _print_predictions(option_labels, preds)


if __name__ == "__main__":
    main()
