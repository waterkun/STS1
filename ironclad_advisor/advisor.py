#!/usr/bin/env python3
"""
Ironclad Advisor - Slay the Spire decision helper (Ascension 20)

Usage examples:
  # Card reward
  python ironclad_advisor/advisor.py card \\
    --floor 8 --act 1 --hp 45 --max-hp 80 \\
    --relics "Burning Blood,Bag of Marbles" \\
    --deck "Strike_R,Strike_R,Strike_R,Defend_R,Defend_R,Defend_R,Bash,Iron Wave,Shrug It Off" \\
    --options "Barricade,Feel No Pain,Pommel Strike"

  # Campfire
  python ironclad_advisor/advisor.py campfire \\
    --floor 11 --act 1 --hp 35 --max-hp 80 \\
    --relics "Burning Blood" \\
    --deck "Strike_R,Strike_R,Defend_R,Defend_R,Bash,Clothesline+1,Iron Wave"

  # Boss relic
  python ironclad_advisor/advisor.py boss-relic \\
    --act 1 --hp 55 --max-hp 80 \\
    --relics "Burning Blood,Bag of Marbles" \\
    --deck "Strike_R x4,Defend_R x4,Bash,Barricade,Feel No Pain,Iron Wave+1" \\
    --options "Snecko Eye,Cursed Key,Coffee Dripper"

  # Shop
  python ironclad_advisor/advisor.py shop \\
    --floor 27 --act 2 --hp 60 --max-hp 80 --gold 300 \\
    --relics "Burning Blood,Snecko Eye,Bag of Marbles" \\
    --deck "Strike_R x3,Defend_R x4,Bash,Barricade,Feel No Pain x2,Reaper" \\
    --cards "Demon Form,Reaper+1,Impervious" \\
    --shop-relics "Mark of Pain,Du-Vu Doll" \\
    --potions "Strength Potion,BloodPotion"
"""

import json
import argparse
import sys
from pathlib import Path
from collections import Counter

try:
    import anthropic
except ImportError:
    print("anthropic package not found. Install it with: pip install anthropic")
    sys.exit(1)

DB_DIR = Path(__file__).parent / "db"

OVERALL_WIN_RATE = 108 / 203  # from the dataset


def load_db() -> dict:
    db = {}
    for name in ["card_decisions", "boss_relic_decisions", "campfire_decisions", "shop_decisions"]:
        path = DB_DIR / f"{name}.json"
        if not path.exists():
            print(f"Database file not found: {path}")
            print("Run: python ironclad_advisor/build_db.py")
            sys.exit(1)
        db[name] = json.loads(path.read_text())
    return db


def format_deck(deck: list[str]) -> str:
    counts = Counter(deck)
    parts = []
    for card, count in sorted(counts.items()):
        parts.append(f"{card} x{count}" if count > 1 else card)
    return ", ".join(parts)


def parse_deck(deck_str: str) -> list[str]:
    """Parse deck string, supporting 'Card x3' notation."""
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


def hp_bucket(hp_pct: int) -> str:
    if hp_pct < 30:
        return "below_30"
    elif hp_pct < 50:
        return "30_to_50"
    elif hp_pct < 70:
        return "50_to_70"
    return "above_70"


# ---------------------------------------------------------------------------
# Similarity retrieval
# ---------------------------------------------------------------------------

def find_similar_card_decisions(db: dict, offered: list[str], act: int, n: int = 5) -> list[dict]:
    offered_lower = set(c.lower() for c in offered)
    scored = []
    for d in db["card_decisions"]["decisions"]:
        d_offered = set(c.lower() for c in d.get("offered", []))
        overlap = len(offered_lower & d_offered)
        if overlap >= max(1, len(offered_lower) - 1):
            act_match = 1 if d.get("act") == act else 0
            scored.append((overlap * 2 + act_match, d))
    scored.sort(key=lambda x: -x[0])
    return [x[1] for x in scored[:n]]


def find_similar_boss_relic_decisions(db: dict, offered: list[str], act: int, n: int = 5) -> list[dict]:
    offered_lower = set(r.lower() for r in offered)
    scored = []
    for d in db["boss_relic_decisions"]["decisions"]:
        if d.get("act") != act:
            continue
        d_offered = set(r.lower() for r in d.get("offered", []))
        overlap = len(offered_lower & d_offered)
        if overlap > 0:
            scored.append((overlap, d))
    scored.sort(key=lambda x: -x[0])
    return [x[1] for x in scored[:n]]


def find_similar_shop_decisions(db: dict, floor: int, act: int, n: int = 5) -> list[dict]:
    scored = []
    for d in db["shop_decisions"]["decisions"]:
        act_match = 2 if d.get("act") == act else 0
        floor_diff = abs(d.get("floor", 0) - floor)
        score = act_match - floor_diff * 0.1
        scored.append((score, d))
    scored.sort(key=lambda x: -x[0])
    return [x[1] for x in scored[:n]]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_card_prompt(db: dict, floor: int, act: int, hp: int, max_hp: int,
                      relics: list[str], deck: list[str], options: list[str]) -> str:
    stats = db["card_decisions"]["stats"]
    hp_pct = round(hp / max_hp * 100) if max_hp > 0 else 0
    is_boss_reward = floor in (16, 33)

    lines = []
    for card in options:
        s = stats.get(card, {})
        pr = s.get("pick_rate", 0)
        wr = s.get("win_rate_in_deck", 0)
        n_off = s.get("times_offered", 0)
        n_deck = s.get("times_in_deck", 0)
        if card == "SKIP":
            lines.append(f"- SKIP: chosen {s.get('times_picked',0)}/{n_off} times offered")
        else:
            lines.append(
                f"- {card}: pick rate {pr*100:.0f}% (offered {n_off}x), "
                f"win rate in deck {wr*100:.0f}% ({n_deck} runs with card)"
            )

    similar = find_similar_card_decisions(db, options, act)
    similar_text = ""
    if similar:
        ex = []
        for d in similar:
            outcome = "WIN" if d["victory"] else "LOSS"
            ex.append(
                f"  Floor {d['floor']} Act{d['act']} HP{d.get('hp_pct','?')}% "
                f"Deck({d['deck_size']}) → picked [{d['picked']}] → {outcome}"
            )
        similar_text = "\nHistorical examples (same cards offered):\n" + "\n".join(ex)

    reward_type = "Boss card reward (rare/colorless)" if is_boss_reward else "Card reward"
    return f"""You are an expert Slay the Spire advisor for Ironclad at Ascension 20.

CURRENT STATE:
- Floor {floor}, Act {act} | {reward_type}
- HP: {hp}/{max_hp} ({hp_pct}%)
- Relics: {', '.join(relics) if relics else 'None'}
- Deck ({len(deck)} cards): {format_deck(deck)}

OPTIONS: {' / '.join(options)}

DATA (203 Ironclad A20 runs, overall win rate 53%):
{chr(10).join(lines)}
{similar_text}

Give a direct, concise recommendation (2-4 sentences). Which option and why, given the specific deck and relics above."""


def build_campfire_prompt(db: dict, floor: int, act: int, hp: int, max_hp: int,
                          relics: list[str], deck: list[str]) -> str:
    stats = db["campfire_decisions"]["stats"]
    hp_pct = round(hp / max_hp * 100) if max_hp > 0 else 0
    bucket = hp_bucket(hp_pct)

    by_hp = stats.get("by_hp_pct", {}).get(bucket, {})
    rest_s = by_hp.get("REST", {})
    smith_s = by_hp.get("SMITH", {})

    top_upgrades = sorted(
        stats.get("common_upgrades", {}).items(),
        key=lambda x: -x[1]["total"]
    )[:12]
    upgrade_lines = "\n".join([
        f"  - {card}: upgraded {info['total']}x, win rate {info['win_rate']*100:.0f}%"
        for card, info in top_upgrades
    ])

    # Unupgraded cards in deck that are upgrade candidates
    unupgraded = [c for c in set(deck) if "+" not in c and c not in ("AscendersBane", "Curse")]
    unupgraded_in_data = [c for c in unupgraded if c in stats.get("common_upgrades", {})]

    return f"""You are an expert Slay the Spire advisor for Ironclad at Ascension 20.

CURRENT STATE:
- Floor {floor}, Act {act}
- HP: {hp}/{max_hp} ({hp_pct}%)
- Relics: {', '.join(relics) if relics else 'None'}
- Deck ({len(deck)} cards): {format_deck(deck)}

DECISION: Campfire — REST or UPGRADE a card?

DATA at {hp_pct}% HP (from 203 runs):
- REST: {rest_s.get('total',0)} times, win rate {rest_s.get('win_rate',0)*100:.0f}%
- UPGRADE: {smith_s.get('total',0)} times, win rate {smith_s.get('win_rate',0)*100:.0f}%

Most commonly upgraded cards across all runs:
{upgrade_lines}

Unupgraded cards in current deck that appear in upgrade data: {', '.join(unupgraded_in_data) or 'none found'}
All unupgraded cards in deck: {', '.join(unupgraded) or 'none'}

Give a direct recommendation (2-4 sentences): rest or upgrade? If upgrade, which card and why?"""


def build_boss_relic_prompt(db: dict, act: int, hp: int, max_hp: int,
                             relics: list[str], deck: list[str], options: list[str]) -> str:
    stats = db["boss_relic_decisions"]["stats"].get(act,
             db["boss_relic_decisions"]["stats"].get(str(act), {}))
    hp_pct = round(hp / max_hp * 100) if max_hp > 0 else 0

    lines = []
    for relic in options:
        s = stats.get(relic, {})
        pr = s.get("pick_rate", 0)
        wr = s.get("win_rate_when_picked", 0)
        n_off = s.get("times_offered", 0)
        n_pick = s.get("times_picked", 0)
        lines.append(
            f"- {relic}: pick rate {pr*100:.0f}% ({n_pick}/{n_off}), "
            f"win rate when picked {wr*100:.0f}%"
        )

    similar = find_similar_boss_relic_decisions(db, options, act)
    similar_text = ""
    if similar:
        ex = []
        for d in similar:
            outcome = "WIN" if d["victory"] else "LOSS"
            ex.append(
                f"  Act{d['act']} Deck({d['deck_size']}) → picked [{d['picked'] or 'SKIP'}] → {outcome}"
            )
        similar_text = "\nHistorical examples (same relics offered):\n" + "\n".join(ex)

    return f"""You are an expert Slay the Spire advisor for Ironclad at Ascension 20.

CURRENT STATE:
- Act {act} Boss Relic Choice
- HP: {hp}/{max_hp} ({hp_pct}%)
- Current relics: {', '.join(relics) if relics else 'None'}
- Deck ({len(deck)} cards): {format_deck(deck)}

OPTIONS: {' / '.join(options)}

DATA (203 Ironclad A20 runs, Act {act} boss relics):
{chr(10).join(lines)}
{similar_text}

Give a direct recommendation (2-4 sentences). Which relic and why, considering the deck composition and existing relics?"""


def build_shop_prompt(db: dict, floor: int, act: int, hp: int, max_hp: int, gold: int,
                      relics: list[str], deck: list[str],
                      avail_cards: list[str], avail_relics: list[str], avail_potions: list[str]) -> str:
    stats = db["shop_decisions"]["stats"]
    hp_pct = round(hp / max_hp * 100) if max_hp > 0 else 0

    def item_line(item: str) -> str:
        s = stats.get(item, {})
        bought = s.get("times_purchased", 0)
        wr_buy = s.get("win_rate_when_purchased", 0)
        wr_skip = s.get("win_rate_when_skipped", 0)
        return (f"  - {item}: bought {bought}x | "
                f"win {wr_buy*100:.0f}% if bought vs {wr_skip*100:.0f}% if skipped")

    card_lines = "\n".join(item_line(c) for c in avail_cards) or "  None"
    relic_lines = "\n".join(item_line(r) for r in avail_relics) or "  None"
    potion_lines = "\n".join(f"  - {p}" for p in avail_potions) or "  None"

    return f"""You are an expert Slay the Spire advisor for Ironclad at Ascension 20.

CURRENT STATE:
- Floor {floor}, Act {act}
- HP: {hp}/{max_hp} ({hp_pct}%) | Gold: {gold}
- Current relics: {', '.join(relics) if relics else 'None'}
- Deck ({len(deck)} cards): {format_deck(deck)}

SHOP CONTENTS:
Cards:
{card_lines}
Relics:
{relic_lines}
Potions:
{potion_lines}

DATA (win rate bought vs skipped, from 203 runs):
(see above — higher buy win rate relative to skip win rate suggests buying is beneficial)

Give a direct recommendation (3-5 sentences): what to buy (if anything), what to skip, and why. Consider gold budget, deck needs, and what's coming in Act {act}."""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_advice(prompt: str) -> str:
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def main():
    parser = argparse.ArgumentParser(
        description="Ironclad Slay the Spire Advisor (A20)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true", help="Print prompt without calling API")
    subs = parser.add_subparsers(dest="command", required=True)

    # --- card ---
    cp = subs.add_parser("card", help="Card reward decision")
    cp.add_argument("--floor", type=int, required=True)
    cp.add_argument("--act", type=int, required=True)
    cp.add_argument("--hp", type=int, required=True)
    cp.add_argument("--max-hp", type=int, required=True, dest="max_hp")
    cp.add_argument("--relics", default="")
    cp.add_argument("--deck", default="", help="Cards separated by comma, supports 'Card x3'")
    cp.add_argument("--options", required=True, help="Cards offered, comma-separated")

    # --- campfire ---
    fp = subs.add_parser("campfire", help="Campfire: rest or upgrade?")
    fp.add_argument("--floor", type=int, required=True)
    fp.add_argument("--act", type=int, required=True)
    fp.add_argument("--hp", type=int, required=True)
    fp.add_argument("--max-hp", type=int, required=True, dest="max_hp")
    fp.add_argument("--relics", default="")
    fp.add_argument("--deck", default="")

    # --- boss-relic ---
    bp = subs.add_parser("boss-relic", help="Boss relic choice")
    bp.add_argument("--act", type=int, required=True)
    bp.add_argument("--hp", type=int, required=True)
    bp.add_argument("--max-hp", type=int, required=True, dest="max_hp")
    bp.add_argument("--relics", default="")
    bp.add_argument("--deck", default="")
    bp.add_argument("--options", required=True)

    # --- shop ---
    sp = subs.add_parser("shop", help="Shop purchase advice")
    sp.add_argument("--floor", type=int, required=True)
    sp.add_argument("--act", type=int, required=True)
    sp.add_argument("--hp", type=int, required=True)
    sp.add_argument("--max-hp", type=int, required=True, dest="max_hp")
    sp.add_argument("--gold", type=int, required=True)
    sp.add_argument("--relics", default="")
    sp.add_argument("--deck", default="")
    sp.add_argument("--cards", default="", help="Cards available in shop")
    sp.add_argument("--shop-relics", default="", dest="shop_relics")
    sp.add_argument("--potions", default="")

    args = parser.parse_args()
    db = load_db()

    if args.command == "card":
        prompt = build_card_prompt(
            db, args.floor, args.act, args.hp, args.max_hp,
            parse_list(args.relics), parse_deck(args.deck), parse_list(args.options),
        )
    elif args.command == "campfire":
        prompt = build_campfire_prompt(
            db, args.floor, args.act, args.hp, args.max_hp,
            parse_list(args.relics), parse_deck(args.deck),
        )
    elif args.command == "boss-relic":
        prompt = build_boss_relic_prompt(
            db, args.act, args.hp, args.max_hp,
            parse_list(args.relics), parse_deck(args.deck), parse_list(args.options),
        )
    elif args.command == "shop":
        prompt = build_shop_prompt(
            db, args.floor, args.act, args.hp, args.max_hp, args.gold,
            parse_list(args.relics), parse_deck(args.deck),
            parse_list(args.cards), parse_list(args.shop_relics), parse_list(args.potions),
        )

    if getattr(args, "dry_run", False):
        print("--- PROMPT (dry run) ---\n")
        print(prompt)
        return

    print("Asking Claude...\n")
    advice = get_advice(prompt)
    print(advice)


if __name__ == "__main__":
    main()
