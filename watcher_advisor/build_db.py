"""
Parse all Watcher .run files and build a decision database for the advisor.

Run once:
    python watcher_advisor/build_db.py
"""

import json
import os
from pathlib import Path
from collections import defaultdict

RUNS_DIR = Path(__file__).parent.parent / "runs"
DB_DIR = Path(__file__).parent / "db"

WATCHER_BASE_DECK = ["Strike_P"] * 4 + ["Defend_P"] * 4 + ["Eruption", "Vigilance"]


def get_act(floor: int) -> int:
    if floor <= 16:
        return 1
    elif floor <= 33:
        return 2
    elif floor <= 51:
        return 3
    return 4


def normalize_card(card: str) -> str:
    replacements = {
        "Ghostly": "Apparition",
        "Venomology": "Alchemize",
        "Wraith Form v2": "Wraith Form",
        "Gash": "Claw",
    }
    base = card.partition("+")[0]
    return replacements.get(base, base)


def load_watcher_runs() -> list[dict]:
    runs = []
    for root, _, files in os.walk(RUNS_DIR):
        for filename in files:
            if not filename.endswith(".run"):
                continue
            path = Path(root) / filename
            try:
                data = json.loads(path.read_text())
                if data.get("character_chosen") == "WATCHER":
                    runs.append(data)
            except Exception as e:
                print(f"  Warning: could not load {path}: {e}")
    return runs


def get_shop_cards_by_floor(run: dict) -> dict[int, list[str]]:
    """Build a map of floor -> list of card names available in that shop."""
    result = {}
    for shop in run.get("shop_contents", []):
        result[int(shop["floor"])] = shop.get("cards", [])
    return result


def reconstruct_deck_at(run: dict, target_floor: int) -> tuple[list[str], dict[str, int]]:
    """Reconstruct deck contents and upgrade counts at the start of a given floor."""
    deck = list(WATCHER_BASE_DECK)
    deck_upgrades: dict[str, int] = {}

    # Neow bonus
    neow = run.get("neow_bonus_log", {})
    for card in neow.get("cardsObtained", []):
        deck.append(normalize_card(card))
    for card in neow.get("cardsRemoved", []):
        n = normalize_card(card)
        if n in deck:
            deck.remove(n)

    # Determine which purchases were cards (vs relics/potions)
    shop_cards_by_floor = get_shop_cards_by_floor(run)
    purchase_floors = run.get("item_purchase_floors", [])
    purchases = run.get("items_purchased", [])
    shop_card_purchases: list[tuple[int, str]] = []
    for floor, item in zip(purchase_floors, purchases):
        if floor < target_floor:
            shop_cards = shop_cards_by_floor.get(floor, [])
            if item in shop_cards:
                shop_card_purchases.append((floor, normalize_card(item)))

    # Apply card choices made before target_floor
    for choice in run.get("card_choices", []):
        floor = int(choice["floor"])
        if floor >= target_floor:
            break
        picked = choice.get("picked", "SKIP")
        if picked not in ("SKIP", "Singing Bowl"):
            deck.append(normalize_card(picked))

    # Apply shop card purchases before target_floor
    for _, card in shop_card_purchases:
        deck.append(card)

    # Apply purges before target_floor
    purge_floors = run.get("items_purged_floors", [])
    purged_cards = run.get("items_purged", [])
    for floor, card in zip(purge_floors, purged_cards):
        if int(floor) < target_floor:
            n = normalize_card(card)
            if n in deck:
                deck.remove(n)

    # Track campfire SMITH upgrades before target_floor
    for campfire in run.get("campfire_choices", []):
        if int(campfire["floor"]) >= target_floor:
            break
        if campfire["key"] == "SMITH" and campfire.get("data"):
            card_name = normalize_card(campfire["data"])
            deck_upgrades[card_name] = deck_upgrades.get(card_name, 0) + 1

    return deck, deck_upgrades


def get_relics_at(run: dict, target_floor: int) -> list[str]:
    """Get relics owned at the start of a given floor."""
    relics = []
    if run.get("relics"):
        relics.append(run["relics"][0])  # starting relic

    neow = run.get("neow_bonus_log", {})
    relics.extend(neow.get("relicsObtained", []))

    for relic_info in run.get("relics_obtained", []):
        if relic_info["floor"] < target_floor:
            relics.append(relic_info["key"])

    return relics


def get_hp_at(run: dict, floor: int | float) -> tuple[int | None, int | None]:
    """Get (current_hp, max_hp) entering a floor."""
    hp_list = run.get("current_hp_per_floor", [])
    max_list = run.get("max_hp_per_floor", [])
    idx = int(floor) - 2
    if idx < 0:
        idx = 0
    hp = hp_list[idx] if idx < len(hp_list) else None
    max_hp = max_list[idx] if idx < len(max_list) else None
    return hp, max_hp


# ---------------------------------------------------------------------------
# Card decisions
# ---------------------------------------------------------------------------

def build_card_decisions(runs: list[dict]) -> dict:
    pick_count: dict[str, int] = defaultdict(int)
    picked_count: dict[str, int] = defaultdict(int)
    win_in_deck: dict[str, int] = defaultdict(int)
    total_in_deck: dict[str, int] = defaultdict(int)

    decisions = []

    for run in runs:
        victory = run["victory"]
        ascension = run.get("ascension_level", 0)

        for choice in run.get("card_choices", []):
            floor = choice["floor"]
            picked = normalize_card(choice.get("picked", "SKIP"))
            not_picked = [normalize_card(c) for c in choice.get("not_picked", [])]
            is_boss_reward = floor in (16, 33)

            offered = list(dict.fromkeys(
                ([picked] if picked != "SKIP" else []) + not_picked + ["SKIP"]
            ))

            act = get_act(floor)
            hp, max_hp = get_hp_at(run, floor)
            hp_pct = round(hp / max_hp * 100) if hp and max_hp else None

            deck, deck_upgrades = reconstruct_deck_at(run, floor)
            relics = get_relics_at(run, floor)
            num_upgrades = sum(deck_upgrades.values())

            for opt in offered:
                pick_count[opt] += 1
                if opt == picked:
                    picked_count[opt] += 1

            decisions.append({
                "floor": floor,
                "act": act,
                "is_boss_reward": is_boss_reward,
                "hp": hp,
                "max_hp": max_hp,
                "hp_pct": hp_pct,
                "deck": deck,
                "deck_size": len(deck),
                "num_upgrades": num_upgrades,
                "deck_upgrades": deck_upgrades,
                "relics": relics,
                "offered": offered,
                "picked": picked,
                "ascension_level": ascension,
                "victory": victory,
            })

        # Win rate when card is in final deck
        for card in set(normalize_card(c) for c in run.get("master_deck", [])):
            total_in_deck[card] += 1
            if victory:
                win_in_deck[card] += 1

    card_stats = {}
    for card in set(list(pick_count.keys()) + list(total_in_deck.keys())):
        offered = pick_count.get(card, 0)
        picked = picked_count.get(card, 0)
        in_deck = total_in_deck.get(card, 0)
        wins = win_in_deck.get(card, 0)
        card_stats[card] = {
            "times_offered": offered,
            "times_picked": picked,
            "pick_rate": round(picked / offered, 3) if offered > 0 else 0,
            "times_in_deck": in_deck,
            "wins_in_deck": wins,
            "win_rate_in_deck": round(wins / in_deck, 3) if in_deck > 0 else 0,
        }

    return {"stats": card_stats, "decisions": decisions}


# ---------------------------------------------------------------------------
# Boss relic decisions
# ---------------------------------------------------------------------------

def build_boss_relic_decisions(runs: list[dict]) -> dict:
    pick_count: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    picked_count: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    wins_picked: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    decisions = []

    for run in runs:
        victory = run["victory"]
        ascension = run.get("ascension_level", 0)

        for i, boss_choice in enumerate(run.get("boss_relics", [])):
            act = i + 1
            picked = boss_choice.get("picked")
            not_picked = boss_choice.get("not_picked", [])
            offered = ([picked] if picked else []) + not_picked

            boss_floor = 16 if act == 1 else 33 if act == 2 else 50
            deck, deck_upgrades = reconstruct_deck_at(run, boss_floor)
            relics_before = get_relics_at(run, boss_floor)
            hp, max_hp = get_hp_at(run, boss_floor)
            hp_pct = round(hp / max_hp * 100) if hp and max_hp else None
            num_upgrades = sum(deck_upgrades.values())

            for opt in offered:
                pick_count[act][opt] += 1
                if opt == picked:
                    picked_count[act][opt] += 1
                    if victory:
                        wins_picked[act][opt] += 1

            decisions.append({
                "act": act,
                "hp_pct": hp_pct,
                "deck": deck,
                "deck_size": len(deck),
                "num_upgrades": num_upgrades,
                "deck_upgrades": deck_upgrades,
                "relics_before": relics_before,
                "offered": offered,
                "picked": picked,
                "ascension_level": ascension,
                "victory": victory,
            })

    relic_stats: dict[int, dict] = {}
    for act in [1, 2, 3]:
        relic_stats[act] = {}
        for relic in pick_count[act]:
            offered = pick_count[act][relic]
            picked = picked_count[act][relic]
            wins = wins_picked[act][relic]
            relic_stats[act][relic] = {
                "times_offered": offered,
                "times_picked": picked,
                "pick_rate": round(picked / offered, 3) if offered > 0 else 0,
                "wins_when_picked": wins,
                "win_rate_when_picked": round(wins / picked, 3) if picked > 0 else 0,
            }

    return {"stats": relic_stats, "decisions": decisions}


# ---------------------------------------------------------------------------
# Campfire decisions
# ---------------------------------------------------------------------------

def build_campfire_decisions(runs: list[dict]) -> dict:
    counters: dict[str, list[int]] = {
        "REST": [0, 0],
        "SMITH": [0, 0],
    }
    hp_buckets = {
        "below_30": {"REST": [0, 0], "SMITH": [0, 0]},
        "30_to_50": {"REST": [0, 0], "SMITH": [0, 0]},
        "50_to_70": {"REST": [0, 0], "SMITH": [0, 0]},
        "above_70": {"REST": [0, 0], "SMITH": [0, 0]},
    }
    upgrade_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    decisions = []

    for run in runs:
        victory = run["victory"]
        ascension = run.get("ascension_level", 0)

        for campfire in run.get("campfire_choices", []):
            floor = campfire["floor"]
            key = campfire["key"]
            data = campfire.get("data")

            hp, max_hp = get_hp_at(run, floor)
            hp_pct = round(hp / max_hp * 100) if hp and max_hp else None
            deck, deck_upgrades = reconstruct_deck_at(run, floor)
            relics = get_relics_at(run, floor)
            num_upgrades = sum(deck_upgrades.values())

            if key in counters:
                counters[key][1] += 1
                if victory:
                    counters[key][0] += 1

            if hp_pct is not None and key in ("REST", "SMITH"):
                if hp_pct < 30:
                    bucket = "below_30"
                elif hp_pct < 50:
                    bucket = "30_to_50"
                elif hp_pct < 70:
                    bucket = "50_to_70"
                else:
                    bucket = "above_70"
                hp_buckets[bucket][key][1] += 1
                if victory:
                    hp_buckets[bucket][key][0] += 1

            if key == "SMITH" and data:
                card = normalize_card(data)
                upgrade_counts[card][1] += 1
                if victory:
                    upgrade_counts[card][0] += 1

            decisions.append({
                "floor": floor,
                "act": get_act(floor),
                "hp": hp,
                "max_hp": max_hp,
                "hp_pct": hp_pct,
                "choice": key,
                "card_upgraded": normalize_card(data) if data else None,
                "deck": deck,
                "deck_size": len(deck),
                "num_upgrades": num_upgrades,
                "deck_upgrades": deck_upgrades,
                "relics": relics,
                "ascension_level": ascension,
                "victory": victory,
            })

    stats = {
        "overall": {
            key: {
                "total": v[1],
                "wins": v[0],
                "win_rate": round(v[0] / v[1], 3) if v[1] > 0 else 0,
            }
            for key, v in counters.items()
        },
        "by_hp_pct": {
            bucket: {
                key: {
                    "total": v[1],
                    "wins": v[0],
                    "win_rate": round(v[0] / v[1], 3) if v[1] > 0 else 0,
                }
                for key, v in choices.items()
            }
            for bucket, choices in hp_buckets.items()
        },
        "common_upgrades": {
            card: {
                "total": counts[1],
                "wins": counts[0],
                "win_rate": round(counts[0] / counts[1], 3) if counts[1] > 0 else 0,
            }
            for card, counts in sorted(upgrade_counts.items(), key=lambda x: -x[1][1])
        },
    }

    return {"stats": stats, "decisions": decisions}


# ---------------------------------------------------------------------------
# Shop decisions
# ---------------------------------------------------------------------------

def build_shop_decisions(runs: list[dict]) -> dict:
    item_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])

    decisions = []

    for run in runs:
        victory = run["victory"]
        ascension = run.get("ascension_level", 0)

        purchase_floors = run.get("item_purchase_floors", [])
        purchases = run.get("items_purchased", [])
        purchased_by_floor: dict[int, list[str]] = defaultdict(list)
        for floor, item in zip(purchase_floors, purchases):
            purchased_by_floor[int(floor)].append(item)

        purge_floors = run.get("items_purged_floors", [])
        purged_cards = run.get("items_purged", [])
        purge_by_floor: dict[int, list[str]] = defaultdict(list)
        for floor, card in zip(purge_floors, purged_cards):
            purge_by_floor[int(floor)].append(normalize_card(card))

        for shop in run.get("shop_contents", []):
            floor = int(shop["floor"])
            available_cards = shop.get("cards", [])
            available_relics = shop.get("relics", [])
            available_potions = shop.get("potions", [])

            bought = set(purchased_by_floor.get(floor, []))
            hp, max_hp = get_hp_at(run, floor)
            hp_pct = round(hp / max_hp * 100) if hp and max_hp else None
            deck, deck_upgrades = reconstruct_deck_at(run, floor)
            relics = get_relics_at(run, floor)
            num_upgrades = sum(deck_upgrades.values())
            gold_list = run.get("gold_per_floor", [])
            gold = gold_list[floor - 1] if floor - 1 < len(gold_list) else None

            for item in available_cards + available_relics:
                n = normalize_card(item)
                if item in bought:
                    item_stats[n][1] += 1
                    if victory:
                        item_stats[n][0] += 1
                else:
                    item_stats[n][3] += 1
                    if victory:
                        item_stats[n][2] += 1

            if purge_by_floor.get(floor):
                item_stats["REMOVE"][1] += 1
                if victory:
                    item_stats["REMOVE"][0] += 1
            else:
                item_stats["REMOVE"][3] += 1
                if victory:
                    item_stats["REMOVE"][2] += 1

            decisions.append({
                "floor": floor,
                "act": get_act(floor),
                "hp_pct": hp_pct,
                "gold": gold,
                "deck": deck,
                "deck_size": len(deck),
                "num_upgrades": num_upgrades,
                "deck_upgrades": deck_upgrades,
                "relics": relics,
                "available_cards": available_cards,
                "available_relics": available_relics,
                "available_potions": available_potions,
                "purchased": list(bought),
                "purged_card": purge_by_floor.get(floor, [None])[0],
                "ascension_level": ascension,
                "victory": victory,
            })

    shop_stats = {
        item: {
            "times_purchased": v[1],
            "wins_when_purchased": v[0],
            "win_rate_when_purchased": round(v[0] / v[1], 3) if v[1] > 0 else 0,
            "times_skipped": v[3],
            "wins_when_skipped": v[2],
            "win_rate_when_skipped": round(v[2] / v[3], 3) if v[3] > 0 else 0,
        }
        for item, v in item_stats.items()
    }

    return {"stats": shop_stats, "decisions": decisions}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading Watcher runs...")
    runs = load_watcher_runs()
    wins = sum(1 for r in runs if r["victory"])
    print(f"Loaded {len(runs)} runs ({wins} wins, {len(runs)-wins} losses)")

    DB_DIR.mkdir(exist_ok=True)

    print("\nBuilding card decision database...")
    card_db = build_card_decisions(runs)
    (DB_DIR / "card_decisions.json").write_text(
        json.dumps(card_db, ensure_ascii=False, separators=(",", ":"))
    )
    print(f"  {len(card_db['decisions'])} decision points, {len(card_db['stats'])} unique cards")

    print("Building boss relic database...")
    boss_db = build_boss_relic_decisions(runs)
    (DB_DIR / "boss_relic_decisions.json").write_text(
        json.dumps(boss_db, ensure_ascii=False, separators=(",", ":"))
    )
    print(f"  {len(boss_db['decisions'])} boss relic decisions")

    print("Building campfire database...")
    camp_db = build_campfire_decisions(runs)
    (DB_DIR / "campfire_decisions.json").write_text(
        json.dumps(camp_db, ensure_ascii=False, separators=(",", ":"))
    )
    print(f"  {len(camp_db['decisions'])} campfire decisions")

    print("Building shop database...")
    shop_db = build_shop_decisions(runs)
    (DB_DIR / "shop_decisions.json").write_text(
        json.dumps(shop_db, ensure_ascii=False, separators=(",", ":"))
    )
    print(f"  {len(shop_db['decisions'])} shop visits")

    print(f"\nDone. Database written to {DB_DIR}/")


if __name__ == "__main__":
    main()
