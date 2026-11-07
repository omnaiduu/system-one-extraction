"""Computed fields whose answer is never printed: one-step operations and two-step chains.

Built from audited rows like `augment.py`, with no writer tokens: a sentence
is inserted into a note, its numbers and dates are the evidence, and the
answer is computed here by an independent formula. A test checks that folding
the gold parts through the gold blocks (`values.run_chain`) reproduces every
answer, so a template whose text order does not match its blocks is caught
before any GPU time.

Three kinds of test, each with phrasings never used in training:

single         one step: percent of, after discount, plus a percentage, split
               equally, average, date minus days, date plus or minus weeks.
combo_seen     two steps, in a combination that training also has.
combo_unseen   two steps, in a combination training never has, made of blocks
               training does have. Right answers here mean the model composes
               blocks rather than memorizing whole formulas.

Every inserted field records `chain_kind` and `steps`, so results can be
broken down by both.
"""

from __future__ import annotations

import random
import re
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from extractor.audit import number_present

from experiments.augment import MONTH_NAMES, _move_spans

Q = Decimal("0.01")


def _pct(amount, rate):
    return amount * rate / 100


# (kind, blocks or op, sentence, instruction, formula)
# Slots: {a} {b} amounts, {r} {s} rates (printed before a "%"), {k} a small
# count, {n} {m} days or weeks, {d} a date. Text order of the slots is the
# fold order of the blocks.
TRAIN_SINGLE = (
    ("percent_of", ["percent_of"], "Tax is charged at {r}% on {a}.", "Tax amount at the stated rate",
     lambda v: _pct(v["a"], v["r"])),
    ("percent_of", ["percent_of"], "A service charge of {r}% applies to {a}.", "Service charge amount",
     lambda v: _pct(v["a"], v["r"])),
    ("discount", ["minus_percent"], "{a} less a {r}% discount is what is owed.", "Amount owed after the discount",
     lambda v: v["a"] - _pct(v["a"], v["r"])),
    ("discount", ["minus_percent"], "The list price {a} carries {r}% off for members.", "Member price after the reduction",
     lambda v: v["a"] - _pct(v["a"], v["r"])),
    ("plus_percent", ["plus_percent"], "{a} plus {r}% VAT is the gross amount.", "Gross amount including VAT",
     lambda v: v["a"] + _pct(v["a"], v["r"])),
    ("plus_percent", ["plus_percent"], "A {r}% surcharge is added to the fare of {a}.", "Fare with the surcharge",
     lambda v: v["a"] + _pct(v["a"], v["r"])),
    ("split", ["divide"], "The bill of {a} is split equally between {k} people.", "Each person's share of the bill",
     lambda v: v["a"] / v["k"]),
    ("split", ["divide"], "A cost of {a} is shared evenly by {k} departments.", "Share per department",
     lambda v: v["a"] / v["k"]),
    ("average", "average", "The three readings were {a}, {b} and {c}.", "Average of the three readings",
     lambda v: (v["a"] + v["b"] + v["c"]) / 3),
    ("average", "average", "Logged weights: {a}, {b}, {c}.", "Mean of the logged weights",
     lambda v: (v["a"] + v["b"] + v["c"]) / 3),
    ("date_minus", ["minus_days"], "Goods must ship {n} days before {d}.", "Last shipping date",
     lambda v: v["d"] - timedelta(days=v["n"])),
    ("date_minus", ["minus_days"], "Payment clears {n} days ahead of {d}.", "Date payment must clear by",
     lambda v: v["d"] - timedelta(days=v["n"])),
    ("weeks_after", ["add_weeks"], "The parcel arrives {w} weeks after {d}.", "Arrival date of the parcel",
     lambda v: v["d"] + timedelta(days=7 * v["w"])),
    ("weeks_after", ["add_weeks"], "The warranty runs {w} weeks from {d}.", "Date the warranty ends",
     lambda v: v["d"] + timedelta(days=7 * v["w"])),
    ("weeks_before", ["minus_weeks"], "Book {w} weeks before {d} to get the rate.", "Latest booking date for the rate",
     lambda v: v["d"] - timedelta(days=7 * v["w"])),
    ("weeks_before", ["minus_weeks"], "Applications close {w} weeks prior to {d}.", "Date applications close",
     lambda v: v["d"] - timedelta(days=7 * v["w"])),
)
TEST_SINGLE = (
    ("percent_of", ["percent_of"], "The agent keeps {r}% of {a} as commission.", "Commission the agent keeps",
     lambda v: _pct(v["a"], v["r"])),
    ("discount", ["minus_percent"], "After knocking {r}% off {a}, the rest is due.", "Amount due after the reduction",
     lambda v: v["a"] - _pct(v["a"], v["r"])),
    ("plus_percent", ["plus_percent"], "The {a} fee rises by {r}% next year.", "Fee next year",
     lambda v: v["a"] + _pct(v["a"], v["r"])),
    ("split", ["divide"], "{a} is divided evenly among {k} flatmates.", "What each flatmate pays",
     lambda v: v["a"] / v["k"]),
    ("average", "average", "Weighings came to {a}, {b} and {c}.", "Mean of the weighings",
     lambda v: (v["a"] + v["b"] + v["c"]) / 3),
    ("date_minus", ["minus_days"], "Notice is needed {n} days ahead of {d}.", "Date notice must be given by",
     lambda v: v["d"] - timedelta(days=v["n"])),
    ("weeks_after", ["add_weeks"], "Payment is due {w} weeks after {d}.", "Payment due date",
     lambda v: v["d"] + timedelta(days=7 * v["w"])),
    ("weeks_before", ["minus_weeks"], "Entries close {w} weeks ahead of {d}.", "Closing date for entries",
     lambda v: v["d"] - timedelta(days=7 * v["w"])),
)

# Two steps. Training combinations:
TRAIN_COMBO = (
    ("multiply+minus_percent", ["multiply", "minus_percent"], "{k} units at {a} each, less a {r}% trade discount.",
     "Net cost after the trade discount", lambda v: v["k"] * v["a"] * (100 - v["r"]) / 100),
    ("multiply+minus_percent", ["multiply", "minus_percent"], "We bought {k} licences at {a} apiece with {r}% off.",
     "Licence cost after the reduction", lambda v: v["k"] * v["a"] * (100 - v["r"]) / 100),
    ("add+plus_percent", ["add", "plus_percent"], "Parts cost {a} and labour {b}; {r}% tax is added to the total.",
     "Repair total including tax", lambda v: (v["a"] + v["b"]) * (100 + v["r"]) / 100),
    ("add+plus_percent", ["add", "plus_percent"], "Room {a} and breakfast {b}, plus {r}% city levy on both.",
     "Stay total with the levy", lambda v: (v["a"] + v["b"]) * (100 + v["r"]) / 100),
    ("minus_percent+minus_percent", ["minus_percent", "minus_percent"],
     "{a} with {r}% off, then a further {s}% loyalty discount.", "Price after both discounts",
     lambda v: v["a"] * (100 - v["r"]) / 100 * (100 - v["s"]) / 100),
    ("minus_percent+minus_percent", ["minus_percent", "minus_percent"],
     "The quote of {a} drops by {r}% and then by another {s}% for early payment.", "Quote after both reductions",
     lambda v: v["a"] * (100 - v["r"]) / 100 * (100 - v["s"]) / 100),
    ("add_days+add_days", ["add_days", "add_days"], "Invoiced on {d}, due {n} days later, with a {m}-day grace period.",
     "Last day before a late fee", lambda v: v["d"] + timedelta(days=v["n"] + v["m"])),
    ("add_days+add_days", ["add_days", "add_days"], "Work starts {d}, takes {n} days, and testing adds {m} days.",
     "Date testing finishes", lambda v: v["d"] + timedelta(days=v["n"] + v["m"])),
    ("add+divide", ["add", "divide"], "Rent {a} and utilities {b} are split between {k} tenants.",
     "Each tenant's monthly share", lambda v: (v["a"] + v["b"]) / v["k"]),
    ("add+divide", ["add", "divide"], "Fuel {a} and tolls {b} were shared by {k} drivers.",
     "Cost per driver", lambda v: (v["a"] + v["b"]) / v["k"]),
)
# Same combinations, new phrasings (test only):
TEST_COMBO_SEEN = (
    ("multiply+minus_percent", ["multiply", "minus_percent"], "An order of {k} chairs at {a} each gets {r}% off for bulk.",
     "Chair order total after the bulk reduction", lambda v: v["k"] * v["a"] * (100 - v["r"]) / 100),
    ("add+plus_percent", ["add", "plus_percent"], "Hotel {a} plus meals {b}, with {r}% service tax on everything.",
     "Trip total with service tax", lambda v: (v["a"] + v["b"]) * (100 + v["r"]) / 100),
    ("minus_percent+minus_percent", ["minus_percent", "minus_percent"],
     "{a} is cut by {r}% in the sale and {s}% more at the till.", "Price paid at the till",
     lambda v: v["a"] * (100 - v["r"]) / 100 * (100 - v["s"]) / 100),
    ("add_days+add_days", ["add_days", "add_days"], "Shipped {d}; {n} days in transit, then {m} days in customs.",
     "Date the goods clear customs", lambda v: v["d"] + timedelta(days=v["n"] + v["m"])),
    ("add+divide", ["add", "divide"], "Venue {a} and catering {b} are split across {k} teams.",
     "Each team's share of the event", lambda v: (v["a"] + v["b"]) / v["k"]),
)
# Combinations training never has (each block appears in training elsewhere):
TEST_COMBO_UNSEEN = (
    ("minus_percent+plus_percent", ["minus_percent", "plus_percent"],
     "Subtotal {a}. {r}% off, then {s}% GST on the discounted amount.", "Final price after discount and GST",
     lambda v: v["a"] * (100 - v["r"]) / 100 * (100 + v["s"]) / 100),
    ("multiply+plus_percent", ["multiply", "plus_percent"], "{k} tickets at {a} each, plus a {r}% booking fee.",
     "Ticket total with the booking fee", lambda v: v["k"] * v["a"] * (100 + v["r"]) / 100),
    ("add+minus_percent", ["add", "minus_percent"], "Frames {a} and lenses {b}, with {r}% off the combined price.",
     "Glasses total after the reduction", lambda v: (v["a"] + v["b"]) * (100 - v["r"]) / 100),
    ("add_weeks+minus_days", ["add_weeks", "minus_days"], "Delivery is {w} weeks after {d}; installers arrive {m} days before that.",
     "Date the installers arrive", lambda v: v["d"] + timedelta(days=7 * v["w"] - v["m"])),
    ("plus_percent+divide", ["plus_percent", "divide"], "A {a} dinner plus {r}% tip is split between {k} friends.",
     "Each friend's share including the tip", lambda v: v["a"] * (100 + v["r"]) / 100 / v["k"]),
)

TEMPLATE_SETS = {
    "train_single": TRAIN_SINGLE, "train_combo": TRAIN_COMBO,
    "single": TEST_SINGLE, "combo_seen": TEST_COMBO_SEEN, "combo_unseen": TEST_COMBO_UNSEEN,
}
RATES = (5, 8, 10, 12, 15, 18, 20, 25)
SLOT = re.compile(r"\{(\w)\}")


def _money(rng: random.Random, low: int, high: int) -> Decimal:
    return Decimal(rng.randint(low * 100, high * 100)) / 100


def _render(slot: str, value) -> str:
    if slot == "d":
        return f"{value.day} {MONTH_NAMES[value.month - 1]} {value.year}"
    if slot in "abc":
        return f"{value:,.2f}"
    return str(value)


def _slots(rng: random.Random) -> dict:
    return {
        "a": _money(rng, 100, 5000), "b": _money(rng, 50, 2000), "c": _money(rng, 10, 200),
        "r": Decimal(rng.choice(RATES)), "s": Decimal(rng.choice(RATES)),
        "k": Decimal(rng.randint(2, 12)), "n": rng.randint(3, 30), "m": rng.randint(2, 15),
        "w": rng.randint(1, 8),
        "d": date(rng.choice((2026, 2027)), rng.randint(1, 12), rng.randint(1, 28)),
    }


def _fill(sentence: str, slots: dict) -> tuple[str, list[tuple[str, int, int]]]:
    """The sentence with its slots filled, and where each slot landed, in text order."""
    out, where, cursor = [], [], 0
    for match in SLOT.finditer(sentence):
        out.append(sentence[cursor : match.start()])
        start = sum(len(piece) for piece in out)
        rendered = _render(match.group(1), slots[match.group(1)])
        out.append(rendered)
        where.append((match.group(1), start, start + len(rendered)))
        cursor = match.end()
    out.append(sentence[cursor:])
    return "".join(out), where


def insert_field(row: dict, rng: random.Random, template, set_name: str, name: str) -> dict | None:
    """Insert one sentence and its computed field into the note. None if no placement works."""
    from experiments.evidence_data import field_evidence

    kind, blocks, sentence, instruction, formula = template
    for _ in range(20):
        slots = _slots(rng)
        exact = formula(slots)
        value = exact if isinstance(exact, date) else exact.quantize(Q, ROUND_HALF_UP)
        filled, where = _fill(sentence, slots)
        if isinstance(value, date):
            if value.isoformat() in row["text"] or _render("d", value) in row["text"] + filled:
                continue
        elif number_present(row["text"] + " " + filled, value, 2):
            continue  # the answer must not be printed anywhere
        break
    else:
        return None
    marked = [span for f in row["fields"].values() for span in (field_evidence(f) or [])]
    text = row["text"]
    cuts = [i + 2 for i in range(len(text) - 1) if text[i] == "." and text[i + 1] == " "
            and not any(a <= i + 2 < b for a, b in marked)]
    at = rng.choice(cuts) if cuts else len(text)
    if at == len(text):
        insert, pad = (" " if not text.endswith(" ") else "") + filled, (0 if text.endswith(" ") else 1)
    else:
        insert, pad = filled + " ", 0
    new_text = text[:at] + insert + text[at:]
    fields = _move_spans(row, [(at, at, len(insert))])
    evidence, kinds, values = [], [], []
    for slot, start, end in where:
        evidence.append({"start": at + pad + start, "end": at + pad + end, "text": filled[start:end]})
        kinds.append("date" if slot == "d" else "number")
        values.append(slots[slot].isoformat() if slot == "d" else str(slots[slot]))
    is_date = isinstance(value, date)
    op = "average" if blocks == "average" else "chain"
    fields[name] = {
        "type": "date" if is_date else "number",
        "instruction": instruction,
        "compute": op,
        **({"blocks": list(blocks)} if op == "chain" else {}),
        "chain_kind": kind,
        "chain_set": set_name,
        "steps": len(blocks) if op == "chain" else 1,
        "missing": False,
        "value": value.isoformat() if is_date else float(value),
        "evidence": evidence,
        "span_kinds": kinds,
        "span_values": values,
        **({} if is_date else {"decimals": 2}),
    }
    return {**row, "text": new_text, "fields": fields}


def chain_variant(row: dict, rng: random.Random, plan: list[str], tag: str) -> dict | None:
    """The note with one inserted field per entry of `plan` (template-set names)."""
    out = row
    for number, set_name in enumerate(plan):
        template = rng.choice(TEMPLATE_SETS[set_name])
        # A neutral name: the field name is part of the question, and must not give the formula away.
        out = insert_field(out, rng, template, set_name, f"calc_{number}")
        if out is None:
            return None
    return {**out, "id": f"{row['id']}~{tag}", "augment": "chain"}


def training_variants(rows: list[dict], seed: int, copies: int = 2) -> list[dict]:
    """Each training note `copies` times, each copy with one single-step and one two-step field."""
    rng = random.Random(seed)
    out = []
    for row in rows:
        for k in range(copies):
            variant = chain_variant(row, rng, ["train_single", "train_combo"], f"c{k}")
            if variant:
                out.append(variant)
    return out


def test_variants(rows: list[dict], seed: int) -> list[dict]:
    """Each exam note three times: one single, one seen combination, one unseen combination.

    The split is `dev_{kind}` (or `{split}_{kind}` for exam 4 in the final job).
    """
    rng = random.Random(seed)
    out = []
    for row in rows:
        for set_name in ("single", "combo_seen", "combo_unseen"):
            variant = chain_variant(row, rng, [set_name], set_name)
            if variant:
                out.append({**variant, "chain_test": set_name})
    return out
