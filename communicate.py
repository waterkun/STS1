#!/usr/bin/env python3
"""
CommunicationMod 统一入口 - 根据游戏状态中的角色自动加载对应 advisor
"""

import importlib
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
_project_root = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(_project_root, "advisor.log")

logger = logging.getLogger("advisor")
logger.setLevel(logging.DEBUG)
_file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(_file_handler)


def log(msg: str):
    """输出到 stderr 和日志文件。"""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()
    logger.info(msg)


log(f"[BOOT] unified script started, python={sys.executable}")
log(f"[BOOT] argv={sys.argv}")
log(f"[BOOT] cwd={os.getcwd()}")

# ---------------------------------------------------------------------------
# 确保项目根目录在 sys.path 中
# ---------------------------------------------------------------------------
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---------------------------------------------------------------------------
# 角色 → advisor 包名映射
# ---------------------------------------------------------------------------
CHARACTER_MAP = {
    "IRONCLAD": "ironclad_advisor",
    "THE_SILENT": "silent_advisor",
    "WATCHER": "watcher_advisor",
    "DEFECT": "defect_advisor",
}

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
    "transformer": "Transf.",
}

_ENGINE_ORDER = ["xgboost", "lightgbm", "lambdamart", "logreg", "cwr_delta", "transformer"]

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
# 各决策类型处理（与各 advisor 的 communicate.py 中逻辑一致）
# ---------------------------------------------------------------------------

def handle_card_reward(gs, floor, act, hp_pct, deck, relics, ctx):
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

    if ctx["v2_models"] is not None:
        preds = ctx["predict_all_card"](
            card_ids, floor, act, hp_pct, deck, relics,
            ctx["db"], ctx["vocab"], ctx["v1_models"], ctx["v2_models"],
            ctx["num_upgrades"], ctx["deck_upgrades"])
    else:
        stats = ctx["db"]["card_decisions"]["stats"]
        X = ctx["card_inference_features"](
            floor, act, hp_pct, deck, relics,
            card_ids, stats, ctx["vocab"],
            ctx["num_upgrades"], ctx["deck_upgrades"])
        preds = ctx["predict_with_models"](ctx["v1_models"]["card"], X)

    if ctx.get("v3_models") and "predict_v3_card" in ctx:
        try:
            preds_v3 = ctx["predict_v3_card"](
                card_ids, floor, act, hp_pct, deck, relics,
                ctx["db"], ctx["vocab"], ctx["v3_models"],
                ctx["num_upgrades"], ctx["deck_upgrades"])
            preds.update(preds_v3)
        except Exception:
            pass

    hp = gs.get("current_hp", 0)
    max_hp = gs.get("max_hp", 1)
    print_header(floor, act, hp, max_hp, "卡牌奖励")
    print_predictions(card_names, preds)


def handle_campfire(gs, floor, act, hp_pct, deck, relics, ctx):
    if ctx["v2_models"] is not None:
        preds = ctx["predict_all_campfire"](
            floor, act, hp_pct, deck, relics,
            ctx["db"], ctx["vocab"], ctx["v1_models"], ctx["v2_models"],
            ctx["num_upgrades"], ctx["deck_upgrades"])
    else:
        X = ctx["campfire_inference_features"](
            floor, act, hp_pct, deck, relics, ctx["vocab"],
            ctx["num_upgrades"], ctx["deck_upgrades"])
        preds = ctx["predict_with_models"](ctx["v1_models"]["campfire"], X)

    if ctx.get("v3_models") and "predict_v3_campfire" in ctx:
        try:
            preds_v3 = ctx["predict_v3_campfire"](
                floor, act, hp_pct, deck, relics,
                ctx["db"], ctx["vocab"], ctx["v3_models"],
                ctx["num_upgrades"], ctx["deck_upgrades"])
            preds.update(preds_v3)
        except Exception:
            pass

    hp = gs.get("current_hp", 0)
    max_hp = gs.get("max_hp", 1)
    print_header(floor, act, hp, max_hp, "篝火决策")
    print_predictions(["REST", "SMITH"], preds)


def handle_boss_relic(gs, act, hp_pct, deck, relics, ctx):
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

    if ctx["v2_models"] is not None:
        preds = ctx["predict_all_boss_relic"](
            relic_ids, act, hp_pct, deck, relics,
            ctx["db"], ctx["vocab"], ctx["v1_models"], ctx["v2_models"],
            ctx["num_upgrades"], ctx["deck_upgrades"])
    else:
        stats = ctx["db"]["boss_relic_decisions"]["stats"]
        X = ctx["boss_relic_inference_features"](
            act, hp_pct, deck, relics,
            relic_ids, stats, ctx["vocab"],
            ctx["num_upgrades"], ctx["deck_upgrades"])
        preds = ctx["predict_with_models"](ctx["v1_models"]["boss_relic"], X)

    if ctx.get("v3_models") and "predict_v3_boss_relic" in ctx:
        try:
            preds_v3 = ctx["predict_v3_boss_relic"](
                relic_ids, act, hp_pct, deck, relics,
                ctx["db"], ctx["vocab"], ctx["v3_models"],
                ctx["num_upgrades"], ctx["deck_upgrades"])
            preds.update(preds_v3)
        except Exception:
            pass

    floor = gs.get("floor", 0)
    hp = gs.get("current_hp", 0)
    max_hp = gs.get("max_hp", 1)
    print_header(floor, act, hp, max_hp, "Boss 遗物选择")
    print_predictions(relic_names, preds)


def handle_shop(gs, floor, act, hp_pct, deck, relics, ctx):
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

    if ctx["v2_models"] is not None:
        id_labels = item_ids + ["不购买"]
        preds = ctx["predict_all_shop"](
            id_labels, floor, act, hp_pct, gold,
            deck, relics, item_ids,
            ctx["db"], ctx["vocab"], ctx["v1_models"], ctx["v2_models"],
            ctx["num_upgrades"], ctx["deck_upgrades"])
    else:
        stats = ctx["db"]["shop_decisions"]["stats"]
        X = ctx["shop_inference_features"](
            floor, act, hp_pct, gold, deck, relics,
            item_ids, stats, ctx["vocab"],
            ctx["num_upgrades"], ctx["deck_upgrades"])
        preds = ctx["predict_with_models"](ctx["v1_models"]["shop"], X)

    if ctx.get("v3_models") and "predict_v3_shop" in ctx:
        try:
            id_labels = item_ids + ["不购买"]
            preds_v3 = ctx["predict_v3_shop"](
                id_labels, floor, act, hp_pct, gold,
                deck, relics, item_ids,
                ctx["db"], ctx["vocab"], ctx["v3_models"],
                ctx["num_upgrades"], ctx["deck_upgrades"])
            preds.update(preds_v3)
        except Exception:
            pass

    hp = gs.get("current_hp", 0)
    max_hp = gs.get("max_hp", 1)
    print_header(floor, act, hp, max_hp, f"商店 (金币: {gold})")
    print_predictions(option_labels, preds, prices=all_prices, gold=gold)


# ---------------------------------------------------------------------------
# 动态加载 advisor 模块
# ---------------------------------------------------------------------------

def load_advisor(character: str) -> dict | None:
    """根据角色名加载对应的 advisor 模块，返回上下文 dict 或 None。"""
    pkg = CHARACTER_MAP.get(character)
    if pkg is None:
        log(f"[LOAD] 不支持的角色: {character}，跳过。")
        return None

    log(f"[LOAD] 正在加载 {character} advisor ({pkg})...")

    try:
        v1_mod = importlib.import_module(f"{pkg}.ml_advisor")
        log(f"[LOAD] {pkg}.ml_advisor 导入成功")
    except Exception:
        log(f"[LOAD] {pkg}.ml_advisor 导入失败:\n{traceback.format_exc()}")
        return None

    v2_mod = None
    try:
        v2_mod = importlib.import_module(f"{pkg}.ml_advisor_v2")
        log(f"[LOAD] {pkg}.ml_advisor_v2 导入成功")
    except Exception:
        log(f"[LOAD] {pkg}.ml_advisor_v2 导入失败 (仅使用 V1):\n{traceback.format_exc()}")

    # 加载数据和模型
    db = v1_mod.load_db()
    vocab = v1_mod.load_vocab()
    v1_models = {
        "card": v1_mod.load_models("card"),
        "campfire": v1_mod.load_models("campfire"),
        "boss_relic": v1_mod.load_models("boss_relic"),
        "shop": v1_mod.load_models("shop"),
    }
    log(f"[LOAD] {character} V1 模型加载完毕。")

    v2_models = None
    if v2_mod is not None:
        try:
            v2_models = v2_mod.load_v2_models()
            if v2_models:
                log(f"[LOAD] {character} V2 模型加载完毕: {list(v2_models.keys())}")
            else:
                log(f"[LOAD] {character} V2 模型文件不存在，仅使用 V1 模型。")
                v2_models = None
        except Exception:
            log(f"[LOAD] {character} V2 模型加载失败:\n{traceback.format_exc()}")
            v2_models = None

    # V3: Transformer 模型
    v3_mod = None
    try:
        v3_mod = importlib.import_module(f"{pkg}.ml_advisor_v3")
        log(f"[LOAD] {pkg}.ml_advisor_v3 导入成功")
    except Exception:
        log(f"[LOAD] {pkg}.ml_advisor_v3 导入失败 (跳过 V3):\n{traceback.format_exc()}")

    v3_models = None
    if v3_mod is not None:
        try:
            v3_models = v3_mod.load_v3_models()
            if v3_models:
                log(f"[LOAD] {character} V3 模型加载完毕: {list(v3_models.keys())}")
            else:
                log(f"[LOAD] {character} V3 模型文件不存在，跳过。")
                v3_models = None
        except Exception:
            log(f"[LOAD] {character} V3 模型加载失败:\n{traceback.format_exc()}")
            v3_models = None

    ctx = {
        "character": character,
        "db": db,
        "vocab": vocab,
        "v1_models": v1_models,
        "v2_models": v2_models,
        "v3_models": v3_models,
        # v1 函数
        "predict_with_models": v1_mod.predict_with_models,
        "card_inference_features": v1_mod.card_inference_features,
        "campfire_inference_features": v1_mod.campfire_inference_features,
        "boss_relic_inference_features": v1_mod.boss_relic_inference_features,
        "shop_inference_features": v1_mod.shop_inference_features,
    }

    # v2 函数（可能不存在）
    if v2_mod is not None:
        ctx["predict_all_card"] = v2_mod.predict_all_card
        ctx["predict_all_campfire"] = v2_mod.predict_all_campfire
        ctx["predict_all_boss_relic"] = v2_mod.predict_all_boss_relic
        ctx["predict_all_shop"] = v2_mod.predict_all_shop

    # v3 函数（可能不存在）
    if v3_mod is not None:
        ctx["predict_v3_card"] = v3_mod.predict_all_card
        ctx["predict_v3_campfire"] = v3_mod.predict_all_campfire
        ctx["predict_v3_boss_relic"] = v3_mod.predict_all_boss_relic
        ctx["predict_v3_shop"] = v3_mod.predict_all_shop

    log(f"[LOAD] {character} advisor 加载完毕。")
    return ctx


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def main():
    import warnings
    warnings.filterwarnings("ignore")

    send("ready")

    # 当前已加载的 advisor 上下文，按角色缓存
    advisor_cache: dict[str, dict] = {}
    current_ctx: dict | None = None
    current_character: str = ""
    last_advice_key = None

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
        if not character:
            send("wait")
            continue

        # 角色切换检测
        if character != current_character:
            log(f"[MAIN] 检测到角色: {character}" +
                (f" (从 {current_character} 切换)" if current_character else ""))

            if character in advisor_cache:
                current_ctx = advisor_cache[character]
                log(f"[MAIN] 使用缓存的 {character} advisor")
            elif character in CHARACTER_MAP:
                try:
                    current_ctx = load_advisor(character)
                    if current_ctx is not None:
                        advisor_cache[character] = current_ctx
                except Exception:
                    log(f"[MAIN] 加载 {character} advisor 失败:\n{traceback.format_exc()}")
                    current_ctx = None
            else:
                log(f"[MAIN] 不支持的角色: {character}，跳过建议。")
                current_ctx = None

            current_character = character
            last_advice_key = None  # 换角色时重置去重 key

        if current_ctx is None:
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

        # 将每局动态数据注入 ctx
        current_ctx["num_upgrades"] = num_upgrades
        current_ctx["deck_upgrades"] = deck_upgrades

        handled = False
        try:
            if screen == "CARD_REWARD":
                handle_card_reward(gs, floor, act, hp_pct, deck, relics, current_ctx)
                handled = True
            elif screen == "REST":
                handle_campfire(gs, floor, act, hp_pct, deck, relics, current_ctx)
                handled = True
            elif screen == "BOSS_REWARD":
                handle_boss_relic(gs, act, hp_pct, deck, relics, current_ctx)
                handled = True
            elif screen == "SHOP_SCREEN":
                handle_shop(gs, floor, act, hp_pct, deck, relics, current_ctx)
                handled = True
        except Exception:
            log(f"[MAIN] 处理 {screen} 时出错:\n{traceback.format_exc()}")

        if handled:
            last_advice_key = advice_key

        send("wait")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(f"[CRASH]\n{traceback.format_exc()}")
