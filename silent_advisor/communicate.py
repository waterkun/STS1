#!/usr/bin/env python3
"""
CommunicationMod 透传建议模式 - Silent
"""

import json
import logging
import os
import sys
import traceback
import unicodedata

# Windows PowerShell 默认使用 GBK，强制 UTF-8 避免中文乱码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

# ---------------------------------------------------------------------------
# 日志：最先初始化，确保任何阶段的错误都能记录
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
LOG_PATH = os.path.join(_project_root, "silent_advisor.log")

logger = logging.getLogger("silent_advisor")
logger.setLevel(logging.DEBUG)
_file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(_file_handler)


def log(msg: str):
    """输出到 stderr 和日志文件。"""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()
    logger.info(msg)


log(f"[BOOT] script started, python={sys.executable}")
log(f"[BOOT] argv={sys.argv}")
log(f"[BOOT] cwd={os.getcwd()}")

# ---------------------------------------------------------------------------
# 导入 ml_advisor（可能很慢或失败）
# ---------------------------------------------------------------------------
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    log("[BOOT] importing ml_advisor...")
    from silent_advisor.ml_advisor import (
        load_db,
        load_vocab,
        load_models,
        predict_with_models,
        card_inference_features,
        campfire_inference_features,
        boss_relic_inference_features,
        shop_inference_features,
    )
    log("[BOOT] import OK")
except Exception:
    log(f"[BOOT] import FAILED:\n{traceback.format_exc()}")
    sys.exit(1)

_v2_available = False
try:
    log("[BOOT] importing ml_advisor_v2...")
    from silent_advisor.ml_advisor_v2 import (
        load_v2_models,
        predict_all_card,
        predict_all_campfire,
        predict_all_boss_relic,
        predict_all_shop,
    )
    _v2_available = True
    log("[BOOT] v2 import OK")
except Exception:
    log(f"[BOOT] v2 import FAILED (will use v1 only):\n{traceback.format_exc()}")

_v3_available = False
try:
    log("[BOOT] importing ml_advisor_v3...")
    from silent_advisor.ml_advisor_v3 import (
        load_v3_models,
        predict_all_card as _predict_card_v3,
        predict_all_campfire as _predict_campfire_v3,
        predict_all_boss_relic as _predict_boss_relic_v3,
        predict_all_shop as _predict_shop_v3,
    )
    _v3_available = True
    log("[BOOT] v3 import OK")
except Exception:
    log(f"[BOOT] v3 import FAILED (will skip):\n{traceback.format_exc()}")

_v4_available = False
try:
    log("[BOOT] importing ml_advisor_v4...")
    from silent_advisor.ml_advisor_v4 import (
        load_v4_models,
        predict_all_card as _predict_card_v4,
        predict_all_campfire as _predict_campfire_v4,
        predict_all_boss_relic as _predict_boss_relic_v4,
        predict_all_shop as _predict_shop_v4,
    )
    _v4_available = True
    log("[BOOT] v4 import OK")
except Exception:
    log(f"[BOOT] v4 import FAILED (will skip):\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# CommunicationMod 协议 I/O
# ---------------------------------------------------------------------------

def send(msg: str):
    """向 stdout 发送一条命令给 CommunicationMod。"""
    if msg != "wait":
        log(f"[SEND] {msg}")
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def read_json_from_stdin() -> dict | None:
    """从 stdin 读取 JSON。"""
    buf = ""
    while True:
        line = sys.stdin.readline()
        if not line:
            log("[READ] stdin EOF")
            return None
        buf += line
        try:
            return json.loads(buf)
        except json.JSONDecodeError:
            continue


# ---------------------------------------------------------------------------
# 字符串显示宽度工具（处理中文/全角字符）
# ---------------------------------------------------------------------------

def _display_width(s: str) -> int:
    w = 0
    for ch in s:
        cat = unicodedata.east_asian_width(ch)
        w += 2 if cat in ("W", "F") else 1
    return w


def _pad_right(s: str, width: int) -> str:
    return s + " " * (width - _display_width(s))


def _pad_left(s: str, width: int) -> str:
    return " " * (width - _display_width(s)) + s


# ---------------------------------------------------------------------------
# 建议输出
# ---------------------------------------------------------------------------

def print_header(floor: int, act: int, hp: int, max_hp: int, title: str):
    hp_pct = round(hp / max_hp * 100) if max_hp > 0 else 0
    log("══════════════════════════════════════════════════════════════════")
    log(f" Floor {floor} Act {act} | HP: {hp}/{max_hp} ({hp_pct}%)")
    log(f" {title}")
    log("──────────────────────────────────────────────────────────────────")


_ENGINE_LABELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "lambdamart": "LambdaMR",
    "logreg": "LogReg",
    "cwr_delta": "CWR",
    "transformer": "TransV3",
    "transformer_v4": "TransV4",
}

_ENGINE_ORDER = ["xgboost", "lightgbm", "lambdamart", "logreg", "cwr_delta",
                 "transformer", "transformer_v4"]

_NAME_COL_WIDTH = 24


def print_predictions(options: list[str], preds: dict,
                       prices: list[int] | None = None,
                       gold: int = 0):
    engines = [e for e in _ENGINE_ORDER if e in preds]
    if not engines:
        engines = sorted(preds.keys())

    header = " " + _pad_right("候选项", _NAME_COL_WIDTH)
    if prices is not None:
        header += "  " + _pad_left("价格", 6)
    for eng in engines:
        label = _ENGINE_LABELS.get(eng, eng)
        header += "  " + _pad_left(label, 8)
    header += "  " + _pad_left("综合", 6) + "  推荐"
    log(header)

    n = len(options)

    rank_sum = np.zeros(n)
    for eng in engines:
        probs = preds[eng]
        ranks = n - np.argsort(np.argsort(probs))
        rank_sum += ranks
    avg_rank = rank_sum / len(engines) if engines else rank_sum

    best_idx = int(avg_rank.argmin())

    order = np.argsort(avg_rank)

    for i in order:
        name = options[i]
        affordable = True
        if prices is not None and prices[i] > 0 and prices[i] > gold:
            affordable = False

        row = " " + _pad_right(name, _NAME_COL_WIDTH)
        if prices is not None:
            if prices[i] > 0:
                price_str = f"{prices[i]}g"
                if not affordable:
                    price_str += "!"
                row += "  " + _pad_left(price_str, 6)
            else:
                row += "  " + _pad_left("-", 6)
        for eng in engines:
            row += f"  {preds[eng][i]*100:>7.1f}%"
        row += f"  {avg_rank[i]:>6.1f}"
        if i == best_idx:
            row += "  ★"
        log(row)

    log("══════════════════════════════════════════════════════════════════")


# ---------------------------------------------------------------------------
# 各决策类型处理
# ---------------------------------------------------------------------------

def handle_card_reward(gs, floor, act, hp_pct, deck, relics, db, vocab,
                       v1_models, v2_models, v3_models=None, v4_models=None,
                       num_upgrades=0, deck_upgrades=None):
    screen = gs.get("screen_state", {})
    cards = screen.get("cards", [])

    if cards:
        card_ids = [c["id"] for c in cards]
        card_names = [c.get("name", c["id"]) for c in cards]
    else:
        card_ids = list(gs.get("choice_list", []))
        card_names = list(card_ids)
    if not card_ids:
        return

    if "SKIP" not in card_ids:
        card_ids.append("SKIP")
        card_names.append("SKIP")

    if v2_models is not None:
        preds = predict_all_card(card_ids, floor, act, hp_pct, deck, relics,
                                 db, vocab, v1_models, v2_models,
                                 num_upgrades, deck_upgrades)
    else:
        stats = db["card_decisions"]["stats"]
        X = card_inference_features(floor, act, hp_pct, deck, relics,
                                    card_ids, stats, vocab,
                                    num_upgrades, deck_upgrades)
        preds = predict_with_models(v1_models["card"], X)

    if v3_models:
        preds.update(_predict_card_v3(card_ids, floor, act, hp_pct, deck, relics,
                                      db, vocab, v3_models, num_upgrades, deck_upgrades))
    if v4_models:
        preds.update(_predict_card_v4(card_ids, floor, act, hp_pct, deck, relics,
                                      db, vocab, v4_models, num_upgrades, deck_upgrades))

    hp = gs.get("current_hp", 0)
    max_hp = gs.get("max_hp", 1)
    print_header(floor, act, hp, max_hp, "卡牌奖励")
    print_predictions(card_names, preds)


def handle_campfire(gs, floor, act, hp_pct, deck, relics, db, vocab,
                    v1_models, v2_models, v3_models=None, v4_models=None,
                    num_upgrades=0, deck_upgrades=None):
    if v2_models is not None:
        preds = predict_all_campfire(floor, act, hp_pct, deck, relics,
                                     db, vocab, v1_models, v2_models,
                                     num_upgrades, deck_upgrades)
    else:
        X = campfire_inference_features(floor, act, hp_pct, deck, relics, vocab,
                                        num_upgrades, deck_upgrades)
        preds = predict_with_models(v1_models["campfire"], X)

    if v3_models:
        preds.update(_predict_campfire_v3(floor, act, hp_pct, deck, relics,
                                          db, vocab, v3_models, num_upgrades, deck_upgrades))
    if v4_models:
        preds.update(_predict_campfire_v4(floor, act, hp_pct, deck, relics,
                                          db, vocab, v4_models, num_upgrades, deck_upgrades))

    hp = gs.get("current_hp", 0)
    max_hp = gs.get("max_hp", 1)
    print_header(floor, act, hp, max_hp, "篝火决策")
    print_predictions(["REST", "SMITH"], preds)


def handle_boss_relic(gs, act, hp_pct, deck, relics, db, vocab,
                      v1_models, v2_models, v3_models=None, v4_models=None,
                      num_upgrades=0, deck_upgrades=None):
    screen = gs.get("screen_state", {})
    relic_list = screen.get("relics", [])

    if relic_list:
        relic_ids = [r["id"] for r in relic_list]
        relic_names = [r.get("name", r["id"]) for r in relic_list]
    else:
        relic_ids = list(gs.get("choice_list", []))
        relic_names = list(relic_ids)
    if not relic_ids:
        return

    if v2_models is not None:
        preds = predict_all_boss_relic(relic_ids, act, hp_pct, deck, relics,
                                       db, vocab, v1_models, v2_models,
                                       num_upgrades, deck_upgrades)
    else:
        stats = db["boss_relic_decisions"]["stats"]
        X = boss_relic_inference_features(act, hp_pct, deck, relics,
                                          relic_ids, stats, vocab,
                                          num_upgrades, deck_upgrades)
        preds = predict_with_models(v1_models["boss_relic"], X)

    if v3_models:
        preds.update(_predict_boss_relic_v3(relic_ids, act, hp_pct, deck, relics,
                                            db, vocab, v3_models, num_upgrades, deck_upgrades))
    if v4_models:
        preds.update(_predict_boss_relic_v4(relic_ids, act, hp_pct, deck, relics,
                                            db, vocab, v4_models, num_upgrades, deck_upgrades))

    floor = gs.get("floor", 0)
    hp = gs.get("current_hp", 0)
    max_hp = gs.get("max_hp", 1)
    print_header(floor, act, hp, max_hp, "Boss 遗物选择")
    print_predictions(relic_names, preds)


def handle_shop(gs, floor, act, hp_pct, deck, relics, db, vocab,
                v1_models, v2_models, v3_models=None, v4_models=None,
                num_upgrades=0, deck_upgrades=None):
    screen = gs.get("screen_state", {})

    shop_cards = screen.get("cards", [])
    shop_relics = screen.get("relics", [])
    shop_potions = screen.get("potions", [])

    avail_cards = [c["id"] for c in shop_cards]
    avail_relics = [r["id"] for r in shop_relics]
    avail_potions = [p["id"] for p in shop_potions]

    card_prices = [c.get("price", 0) for c in shop_cards]
    relic_prices = [r.get("price", 0) for r in shop_relics]
    potion_prices = [p.get("price", 0) for p in shop_potions]

    all_items = avail_cards + avail_relics + avail_potions
    all_prices = card_prices + relic_prices + potion_prices + [0]
    if not all_items:
        return

    gold = gs.get("gold", 0)

    card_names = [c.get("name", c["id"]) for c in shop_cards]
    relic_names = [r.get("name", r["id"]) for r in shop_relics]
    potion_names = [p.get("name", p["id"]) for p in shop_potions]
    option_labels = card_names + relic_names + potion_names + ["不购买"]

    item_ids = avail_cards + avail_relics + avail_potions

    if v2_models is not None:
        id_labels = item_ids + ["不购买"]
        preds = predict_all_shop(id_labels, floor, act, hp_pct, gold,
                                 deck, relics, item_ids,
                                 db, vocab, v1_models, v2_models,
                                 num_upgrades, deck_upgrades)
    else:
        stats = db["shop_decisions"]["stats"]
        X = shop_inference_features(floor, act, hp_pct, gold, deck, relics,
                                    item_ids, stats, vocab,
                                    num_upgrades, deck_upgrades)
        preds = predict_with_models(v1_models["shop"], X)

    id_labels = item_ids + ["移除卡牌", "不购买"]
    if v3_models:
        preds.update(_predict_shop_v3(id_labels, floor, act, hp_pct, gold,
                                      deck, relics, item_ids,
                                      db, vocab, v3_models, num_upgrades, deck_upgrades))
    if v4_models:
        preds.update(_predict_shop_v4(id_labels, floor, act, hp_pct, gold,
                                      deck, relics, item_ids,
                                      db, vocab, v4_models, num_upgrades, deck_upgrades))

    hp = gs.get("current_hp", 0)
    max_hp = gs.get("max_hp", 1)
    print_header(floor, act, hp, max_hp, f"商店 (金币: {gold})")
    print_predictions(option_labels, preds, prices=all_prices, gold=gold)


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def main():
    import warnings
    warnings.filterwarnings("ignore")

    send("ready")

    db = None
    vocab = None
    v1_models = None
    v2_models = None
    v3_models = None
    v4_models = None
    models_loaded = False
    last_advice_key = None

    def ensure_models():
        nonlocal db, vocab, v1_models, v2_models, v3_models, v4_models, models_loaded
        if models_loaded:
            return
        log("正在加载 ML 模型...")
        db = load_db()
        vocab = load_vocab()
        v1_models = {
            "card": load_models("card"),
            "campfire": load_models("campfire"),
            "boss_relic": load_models("boss_relic"),
            "shop": load_models("shop"),
        }
        log("V1 模型加载完毕。")

        if _v2_available:
            try:
                v2_models = load_v2_models()
                if v2_models:
                    log(f"V2 模型加载完毕: {list(v2_models.keys())}")
                else:
                    log("V2 模型文件不存在，仅使用 V1 模型。")
                    v2_models = None
            except Exception:
                log(f"V2 模型加载失败:\n{traceback.format_exc()}")
                v2_models = None
        else:
            v2_models = None

        if _v3_available:
            try:
                v3_models = load_v3_models()
                if v3_models:
                    log(f"V3 模型加载完毕: {list(v3_models.keys())}")
                else:
                    log("V3 模型文件不存在，跳过。")
                    v3_models = None
            except Exception:
                log(f"V3 模型加载失败:\n{traceback.format_exc()}")
                v3_models = None
        else:
            v3_models = None

        if _v4_available:
            try:
                v4_models = load_v4_models()
                if v4_models:
                    log(f"V4 模型加载完毕: {list(v4_models.keys())}")
                else:
                    log("V4 模型文件不存在，跳过。")
                    v4_models = None
            except Exception:
                log(f"V4 模型加载失败:\n{traceback.format_exc()}")
                v4_models = None
        else:
            v4_models = None

        models_loaded = True
        log("模型加载完毕。")

    while True:
        msg = read_json_from_stdin()
        if msg is None:
            log("连接关闭，退出。")
            break

        if not msg.get("in_game"):
            send("wait")
            continue

        gs = msg.get("game_state")
        if not gs:
            send("wait")
            continue

        character = gs.get("class", "")
        if character and character != "THE_SILENT":
            key = ("non_silent", character)
            if key != last_advice_key:
                log(f"当前角色: {character}，模型仅支持 THE_SILENT，跳过建议。")
                last_advice_key = key
            send("wait")
            continue

        screen = gs.get("screen_type", "")
        floor = gs.get("floor", 0)
        act = gs.get("act", 1)

        if screen not in ("CARD_REWARD", "REST", "BOSS_REWARD", "SHOP_SCREEN"):
            send("wait")
            continue

        advice_key = (screen, floor, tuple(gs.get("choice_list", [])))
        if advice_key == last_advice_key:
            send("wait")
            continue

        ensure_models()

        hp = gs.get("current_hp", 0)
        max_hp = gs.get("max_hp", 1)
        deck_raw = gs.get("deck", [])
        deck = [card["id"] for card in deck_raw]
        deck_upgrades = {}
        for card in deck_raw:
            if card.get("upgrades", 0) > 0:
                cid = card["id"]
                deck_upgrades[cid] = deck_upgrades.get(cid, 0) + card["upgrades"]
        num_upgrades = sum(deck_upgrades.values())
        relics = [relic["id"] for relic in gs.get("relics", [])]
        hp_pct = round(hp / max_hp * 100) if max_hp > 0 else 0

        handled = False
        if screen == "CARD_REWARD":
            handle_card_reward(gs, floor, act, hp_pct, deck, relics,
                               db, vocab, v1_models, v2_models, v3_models, v4_models,
                               num_upgrades, deck_upgrades)
            handled = True
        elif screen == "REST":
            handle_campfire(gs, floor, act, hp_pct, deck, relics,
                            db, vocab, v1_models, v2_models, v3_models, v4_models,
                            num_upgrades, deck_upgrades)
            handled = True
        elif screen == "BOSS_REWARD":
            handle_boss_relic(gs, act, hp_pct, deck, relics,
                              db, vocab, v1_models, v2_models, v3_models, v4_models,
                              num_upgrades, deck_upgrades)
            handled = True
        elif screen == "SHOP_SCREEN":
            handle_shop(gs, floor, act, hp_pct, deck, relics,
                        db, vocab, v1_models, v2_models, v3_models, v4_models,
                        num_upgrades, deck_upgrades)
            handled = True

        if handled:
            last_advice_key = advice_key

        send("wait")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(f"[CRASH]\n{traceback.format_exc()}")
