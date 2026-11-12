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


# ---- experiment 7: a grammar instead of templates ----------------------------
#
# Experiment 6 trained on 26 fixed sentences and learned them: amounts were
# found only where those sentences put them, and the step head learned whole
# pairs ("a discount is followed by a discount"). Here a sentence is built from
# parts: an opening that states the starting value, then one clause per step,
# joined by a connector. Each block has its own clauses, so a block reads the
# same in any position; every pair of blocks is generated except a held-out
# set; and every part has training and test wordings. Numbers and dates are
# printed in several styles, with currency outside the mark.
#
# Fold order is text order: the starting value comes first, then each step's
# operand. Percent and date blocks are order-free in `values.apply_block`.

G_OPEN_NUMBER = {
    "train": [("Subtotal {A}", "subtotal"), ("The bill comes to {A}", "bill"), ("{A} is the list price", "list price"),
              ("Base fare: {A}", "fare"), ("The order is worth {A}", "order"), ("Quoted at {A}", "quote"),
              ("The rent is {A}", "rent"), ("Membership costs {A}", "membership")],
    "test": [("Net value before adjustments is {A}", "net value"), ("The invoice shows {A}", "invoice"),
             ("{A} was the opening figure", "opening figure"), ("Course fees stand at {A}", "course fee")],
}
G_OPEN_DATE = {
    "train": [("Dated {D}", "document"), ("Work began on {D}", "work"), ("The order was placed {D}", "order"),
              ("Signed on {D}", "contract"), ("The trip starts {D}", "trip")],
    "test": [("Received {D}", "parcel"), ("The kickoff was {D}", "project")],
}
G_STEPS = {
    "minus_percent": {"train": ["{R} off", "less a {R} discount", "minus {R} for members", "with a {R} reduction",
                                "discounted by {R}", "after {R} is knocked off"],
                      "test": ["{R} cheaper in the sale", "a {R} rebate is taken off", "reduced {R} for early payment"]},
    "plus_percent": {"train": ["plus {R} GST", "with {R} tax added", "{R} VAT on top", "add a {R} surcharge",
                               "marked up by {R}", "plus a {R} service charge"],
                     "test": ["{R} sales tax is then applied", "raised by {R}", "a {R} levy is added"]},
    "percent_of": {"train": ["{R} of that is the commission", "the fee is {R} of it", "{R} of it goes to tax"],
                   "test": ["the agent's cut is {R} of that", "only {R} of it is refundable"]},
    "add": {"train": ["plus {B} for delivery", "and {B} for labour", "with {B} shipping added", "adding {B} in fees",
                      "and another {B} for parts"],
            "test": ["topped up with {B} for packing", "{B} extra for insurance"]},
    "subtract": {"train": ["minus a {B} deposit", "less {B} paid in advance", "after deducting {B} already paid",
                           "with {B} credited back"],
                 "test": ["{B} was refunded from that", "net of a {B} advance"]},
    "multiply": {"train": ["for {K} months", "times {K} units", "for each of {K} guests", "across {K} rooms", "for {K} nights"],
                 "test": ["repeated for {K} weeks of service", "for all {K} seats"]},
    "divide": {"train": ["split between {K} people", "shared by {K} teams", "spread over {K} months", "divided among {K} tenants"],
               "test": ["paid in {K} equal parts", "split {K} ways equally"]},
    "add_days": {"train": ["due {N} later", "then {N} to deliver", "with {N} more for checks", "{N} after that"],
                 "test": ["allow {N} beyond that", "a further {N} pass"]},
    "minus_days": {"train": ["but ship {N} before that", "the reminder goes {N} earlier", "{N} ahead of that"],
                   "test": ["pulled forward by {N}", "{N} prior to that"]},
    "add_weeks": {"train": ["{W} later", "after {W}", "plus {W} of testing"],
                  "test": ["{W} on from that"]},
    "minus_weeks": {"train": ["{W} before that", "{W} earlier"],
                    "test": ["{W} ahead of time"]},
}
G_JOIN = {"train": [", ", "; then ", ", then ", ". Then ", " and "],
          "test": [". After that, ", ", followed by ", " - then "]}
G_ASK_NUMBER = {"train": ["{Topic} after all adjustments", "Final {topic} amount", "What the {topic} comes to in the end"],
                "test": ["{Topic}, settled figure", "End value of the {topic}"]}
G_ASK_DATE = {"train": ["{Topic} date after the changes", "Final date for the {topic}"],
              "test": ["Date the {topic} lands on"]}
G_AVERAGE = {"train": [("The three readings were {A}, {B} and {C}", "Average of the three readings"),
                       ("Logged weights: {A}, {B}, {C}", "Mean of the logged weights"),
                       ("Sales ran {A}, {B} and {C} over three months", "Average monthly sales")],
             "test": [("Weighings came to {A}, {B} and {C}", "Mean of the weighings")]}

NUMBER_STEP_BLOCKS = ("add", "subtract", "multiply", "divide", "minus_percent", "plus_percent", "percent_of")
DATE_STEP_BLOCKS = ("add_days", "minus_days", "add_weeks", "minus_weeks")
LAST_ONLY = {"percent_of"}
# Pairs never generated for training. Each block appears first and second elsewhere.
HELD_OUT_PAIRS = {("minus_percent", "plus_percent"), ("multiply", "plus_percent"), ("add", "minus_percent"),
                  ("plus_percent", "divide"), ("subtract", "percent_of"), ("divide", "add"),
                  ("add_weeks", "minus_days"), ("minus_days", "add_weeks")}


def all_pairs(kind: str) -> list[tuple[str, str]]:
    blocks = NUMBER_STEP_BLOCKS if kind == "number" else DATE_STEP_BLOCKS
    return [(a, b) for a in blocks for b in blocks if a not in LAST_ONLY]


def _render_amount(value: Decimal, rng: random.Random, currency: str | None = None) -> tuple[str, str]:
    """(printed number, currency in front). The mark covers the number only. One currency per sentence."""
    if value == value.to_integral_value() and rng.random() < 0.5:
        number = f"{int(value):,}" if rng.random() < 0.6 else str(int(value))
    else:
        number = f"{value:,.2f}" if rng.random() < 0.7 else f"{value:.2f}"
    if currency is None:
        currency = rng.choice(CURRENCIES)
    return number, currency


CURRENCIES = ["", "", "Rs ", "Rs. ", "₹", "$", "INR "]


def _render_date(d: date, rng: random.Random) -> str:
    month = MONTH_NAMES[d.month - 1]
    style = rng.randrange(5)
    if style == 0:
        return f"{d.day} {month} {d.year}"
    if style == 1:
        return f"{month} {d.day}, {d.year}"
    if style == 2:
        return f"{d.day} {month[:3]} {d.year}"
    if style == 3:
        return d.isoformat()
    return f"{d.day:02d}/{d.month:02d}/{d.year}"


def _amount(rng: random.Random, low: int, high: int) -> Decimal:
    if rng.random() < 0.3:
        return Decimal(rng.randint(low, high))
    return Decimal(rng.randint(low * 100, high * 100)) / 100


def _operand(block: str, start, rng: random.Random):
    """(value, how it prints, placeholder) for one step's operand."""
    if block in ("minus_percent", "plus_percent", "percent_of"):
        r = Decimal(rng.choice(RATES))
        return r, str(r), "R", rng.choice(["%", "%", " per cent", " percent"])
    if block in ("add", "subtract"):
        top = max(20, int(abs(start) * Decimal("0.4"))) if isinstance(start, Decimal) else 2000
        return _amount(rng, 10, top), None, "B", ""
    if block in ("multiply", "divide"):
        k = Decimal(rng.randint(2, 12))
        return k, str(k), "K", ""
    if block in ("add_days", "minus_days"):
        n = Decimal(rng.randint(2, 30))
        return n, str(n), "N", " days"
    w = Decimal(rng.randint(1, 8))
    return w, str(w), "W", " week" if w == 1 else " weeks"


def _sentence_parts(pieces):
    """Join (text, slot) pieces and record each slot's character span."""
    out, where = "", []
    for text, slot in pieces:
        if slot is not None:
            where.append((slot, len(out), len(out) + len(text)))
        out += text
    return out, where


def _capitalise(text: str) -> str:
    return text[:1].upper() + text[1:]


def grammar_field(rng: random.Random, kind: str, blocks: list[str], wording: str, used_topics: set):
    """One generated sentence and its field (without placement). None if it could not be built."""
    from experiments.values import apply_block

    other = "test" if wording == "train" else "train"
    if blocks == ["average"]:
        sentence, instruction = rng.choice(G_AVERAGE[wording])
        values = [_amount(rng, 10, 900) for _ in range(3)]
        money = rng.choice(CURRENCIES)
        pieces, slot_values = [], []
        for chunk in re.split(r"(\{[ABC]\})", sentence):
            if chunk in ("{A}", "{B}", "{C}"):
                number, currency = _render_amount(values[len(slot_values)], rng, money)
                pieces += [(currency, None), (number, len(slot_values))]
                slot_values.append(values[len(slot_values)])
            elif chunk:
                pieces.append((chunk, None))
        text, where = _sentence_parts(pieces + [(".", None)])
        exact = sum(values) / 3
        return text, where, slot_values, ["number"] * 3, instruction, exact, "average"
    opens = [o for o in (G_OPEN_NUMBER if kind == "number" else G_OPEN_DATE)[wording] if o[1] not in used_topics]
    if not opens:
        return None
    opening, topic = rng.choice(opens)
    used_topics.add(topic)
    pieces, values, kinds = [], [], []
    money = rng.choice(CURRENCIES)
    if kind == "number":
        start = _amount(rng, 100, 5000)
        number, currency = _render_amount(start, rng, money)
        before, after = opening.split("{A}")
        pieces += [(before, None), (currency, None), (number, 0), (after, None)]
    else:
        start = date(rng.choice((2026, 2027)), rng.randint(1, 12), rng.randint(1, 28))
        before, after = opening.split("{D}")
        pieces += [(before, None), (_render_date(start, rng), 0), (after, None)]
    values.append(start)
    kinds.append(kind)
    acc = start
    for step, block in enumerate(blocks):
        value, printed, slot, unit = _operand(block, acc, rng)
        if printed is None:
            printed, currency = _render_amount(value, rng, money)
        else:
            currency = ""
        clause = rng.choice(G_STEPS[block][wording])
        joiner = rng.choice(G_JOIN[wording])
        if clause.startswith("and ") and joiner.strip(" ,;.-") in ("and", "then", "followed by", "After that"):
            clause = clause[4:]
        before, after = clause.split("{" + slot + "}")
        if joiner.endswith(". "):
            before = _capitalise(before)
        pieces += [(joiner, None), (before, None), (currency, None), (printed, len(values)), (unit + after, None)]
        values.append(value)
        kinds.append("number")
        acc = apply_block(block, acc, value, False, slot == "R")
    text, where = _sentence_parts(pieces + [(".", None)])
    text = _capitalise(text)
    ask = rng.choice((G_ASK_NUMBER if kind == "number" else G_ASK_DATE)[wording])
    instruction = ask.format(topic=topic, Topic=_capitalise(topic))
    return text, where, values, kinds, instruction, acc, "chain"


def insert_grammar_field(row: dict, rng: random.Random, kind: str, blocks: list[str], wording: str,
                         set_name: str, name: str, used_topics: set) -> dict | None:
    from experiments.evidence_data import field_evidence

    for _ in range(20):
        built = grammar_field(rng, kind, blocks, wording, set(used_topics))
        if built is None:
            return None
        text, where, values, kinds, instruction, exact, op = built
        value = exact if isinstance(exact, date) else exact.quantize(Q, ROUND_HALF_UP)
        if isinstance(value, date):
            if value.isoformat() in row["text"] + text or _render("d", value) in row["text"] + text:
                continue
        elif number_present(row["text"] + " " + text, value, 2):
            continue
        break
    else:
        return None
    topic_open = next((t for o, t in G_OPEN_NUMBER[wording] + G_OPEN_DATE[wording] if o.split("{")[0] and text.startswith(_capitalise(o.split("{")[0]))), None)
    if topic_open:
        used_topics.add(topic_open)
    marked = [span for f in row["fields"].values() for span in (field_evidence(f) or [])]
    base = row["text"]
    cuts = [i + 2 for i in range(len(base) - 1) if base[i] == "." and base[i + 1] in " \n"
            and not any(a <= i + 2 < b for a, b in marked)]
    at = rng.choice(cuts) if cuts else len(base)
    if at == len(base):
        pad = 0 if base.endswith((" ", "\n")) else 1
        insert = (" " if pad else "") + text
    else:
        insert, pad = text + " ", 0
    new_text = base[:at] + insert + base[at:]
    fields = _move_spans(row, [(at, at, len(insert))])
    evidence = [{"start": at + pad + a, "end": at + pad + b, "text": text[a:b]} for _, a, b in where]
    is_date = isinstance(value, date)
    fields[name] = {
        "type": "date" if is_date else "number",
        "instruction": instruction,
        "compute": op,
        **({"blocks": list(blocks)} if op == "chain" else {}),
        "chain_kind": "+".join(blocks),
        "chain_set": set_name,
        "steps": len(blocks) if op == "chain" else 1,
        "missing": False,
        "value": value.isoformat() if is_date else float(value),
        "evidence": evidence,
        "span_kinds": kinds,
        "span_values": [v.isoformat() if isinstance(v, date) else str(v) for v in values],
        **({} if is_date else {"decimals": 2}),
    }
    return {**row, "text": new_text, "fields": fields}


def _pick_single(rng: random.Random) -> tuple[str, list[str]]:
    if rng.random() < 0.15:
        return "number", ["average"]
    kind = "number" if rng.random() < 0.7 else "date"
    return kind, [rng.choice(NUMBER_STEP_BLOCKS if kind == "number" else DATE_STEP_BLOCKS)]


def _pick_pair(rng: random.Random, held_out: bool) -> tuple[str, list[str]]:
    kind = "number" if rng.random() < 0.75 else "date"
    pairs = [p for p in all_pairs(kind) if (p in HELD_OUT_PAIRS) == held_out]
    return kind, list(rng.choice(pairs))


def grammar_training_variants(rows: list[dict], seed: int, copies: int = 2) -> list[dict]:
    """Each training note `copies` times, each copy with one single-step and one two-step field
    (training wording, never a held-out pair)."""
    rng = random.Random(seed)
    out = []
    for row in rows:
        for k in range(copies):
            topics: set = set()
            variant = row
            for number, picker in enumerate((_pick_single, lambda r: _pick_pair(r, False))):
                kind, blocks = picker(rng)
                variant = insert_grammar_field(variant, rng, kind, blocks, "train",
                                               "train_single" if number == 0 else "train_combo", f"calc_{number}", topics)
                if variant is None:
                    break
            if variant is not None:
                out.append({**variant, "id": f"{row['id']}~g{k}", "augment": "chain"})
    return out


GRAMMAR_TESTS = {
    # name: (wording, how the blocks are picked)
    "single": ("test", lambda rng: _pick_single(rng)),
    "combo_seen": ("test", lambda rng: _pick_pair(rng, False)),
    "combo_unseen": ("test", lambda rng: _pick_pair(rng, True)),
    # New combinations in training wording: composing alone, with no new words.
    "combo_unseen_familiar": ("train", lambda rng: _pick_pair(rng, True)),
}


def grammar_test_variants(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for row in rows:
        for set_name, (wording, picker) in GRAMMAR_TESTS.items():
            kind, blocks = picker(rng)
            variant = insert_grammar_field(row, rng, kind, blocks, wording, set_name, "calc_0", set())
            if variant is not None:
                out.append({**variant, "id": f"{row['id']}~{set_name}", "augment": "chain", "chain_test": set_name})
    return out
