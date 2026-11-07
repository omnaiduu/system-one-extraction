"""New training and test cases from the existing notes, with no writer tokens.

Two kinds, both built from audited rows (clean text plus measured spans), so
every label stays exact:

words  The note with some marked numbers and dates rewritten in words
       ("1,486.75" -> "one thousand four hundred eighty-six point seven five",
       "2 Apr 2026" -> "the second of April, 2026"). The value of every field
       is unchanged: the program wrote the words from the value. All spans
       after a rewrite are shifted. The parser cannot read these words, so a
       right answer needs the learned reader.

emit   One sentence inserted into the note, with a new field whose answer no
       listed operation computes: a percentage of an amount, an amount after a
       discount, the average of three readings, a date some days before
       another. The inputs are marked; the answer is computed here and never
       printed. A right answer needs the emit head.

Phrasings are split: `TRAIN_TEMPLATES` for training notes, `TEST_TEMPLATES`
(never seen in training) for the exams, so the exams measure whether the
model generalizes past a template, not whether it memorized one.

Every augmented row keeps its source note's id before a "~", so a note and
its variants always land on the same side of a split.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from extractor.audit import Drop, number_present, parse_date, parse_number

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
            9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
            15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth",
            20: "twentieth", 30: "thirtieth"}
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]


def _under_thousand(n: int, joiner: str) -> str:
    words = []
    if n >= 100:
        words.append(f"{ONES[n // 100]} hundred")
        n %= 100
        if n:
            words.append(joiner.strip() or "")
    if n >= 20:
        words.append(TENS[n // 10] + (f"-{ONES[n % 10]}" if n % 10 else ""))
    elif n or not words:
        words.append(ONES[n])
    return " ".join(w for w in words if w)


def integer_words(n: int, rng: random.Random) -> str:
    """English words for 0 <= n < 10,000,000. Sometimes 'and', sometimes Indian lakh."""
    if n < 0:
        return "minus " + integer_words(-n, rng)
    joiner = " and " if rng.random() < 0.5 else " "
    if n >= 100000 and rng.random() < 0.5:
        lakh, rest = divmod(n, 100000)
        head = f"{_under_thousand(lakh, joiner)} lakh"
        return head if not rest else f"{head} {integer_words(rest, rng)}"
    parts = []
    for size, name in ((1000000, "million"), (1000, "thousand")):
        if n >= size:
            parts.append(f"{_under_thousand(n // size, joiner)} {name}")
            n %= size
    if n or not parts:
        parts.append(_under_thousand(n, joiner))
    return " ".join(parts)


def number_words(value: Decimal, decimals: int, rng: random.Random) -> str:
    whole = int(abs(value))
    frac = int((abs(value) - whole) * 100 + Decimal("0.5")) if decimals else 0
    head = integer_words(whole, rng)
    if not decimals or frac == 0:
        return head if rng.random() < 0.6 else f"{head} only"
    style = rng.random()
    if style < 0.4:
        digits = f"{frac:02d}"
        return f"{head} point " + " ".join(ONES[int(d)] for d in digits)
    if style < 0.7:
        return f"{head} and {integer_words(frac, rng)} cents"
    return f"{head} rupees and {integer_words(frac, rng)} paise"


def ordinal(day: int) -> str:
    if day in ORDINALS:
        return ORDINALS[day]
    tens, ones = divmod(day, 10)
    return f"{TENS[tens]}-{ORDINALS[ones]}"


def date_words(d: date, rng: random.Random) -> str:
    month = MONTH_NAMES[d.month - 1]
    year = str(d.year) if rng.random() < 0.7 else f"two thousand {integer_words(d.year - 2000, rng)}"
    style = rng.random()
    if style < 0.35:
        return f"the {ordinal(d.day)} of {month}, {year}"
    if style < 0.7:
        return f"{month} the {ordinal(d.day)}, {year}"
    return f"{ordinal(d.day)} {month} {year}"


# ---- which marked spans hold a number or a date --------------------------------


def span_kinds(field: dict) -> list[str | None]:
    """For each evidence span of a field (in text order): 'number', 'date', or None."""
    from experiments.evidence_data import field_evidence, field_operation

    spans = field_evidence(field) or []
    op = field_operation(field)
    if op in ("read_number", "sum", "product", "subtract", "median"):
        return ["number"] * len(spans)
    if op in ("read_date", "days_between"):
        return ["date"] * len(spans)
    if op == "add_days":
        return ["date_or_count"] * len(spans)
    if op == "emit":
        return list(field.get("span_kinds") or [None] * len(spans))
    return [None] * len(spans)


def span_value(text: str, kind: str | None, dayfirst: bool = True):
    """The value a marked span holds, from its digits: Decimal, date, or None."""
    if kind is None:
        return None
    if kind in ("date", "date_or_count"):
        try:
            return parse_date(text, dayfirst)
        except Drop:
            if kind == "date":
                return None
    try:
        return parse_number(text)
    except Drop:
        return None


def _shift(position: int, edits: list[tuple[int, int, int]]) -> int:
    """New offset of an old offset after (start, end, new_length) replacements."""
    out = position
    for start, end, length in edits:
        if position >= end:
            out += length - (end - start)
    return out


def _move_spans(row: dict, edits: list[tuple[int, int, int]]) -> dict:
    fields = {}
    for name, field in row["fields"].items():
        field = dict(field)
        if "start" in field and "end" in field and field.get("start") is not None:
            field["start"], field["end"] = _shift(field["start"], edits), _shift(field["end"], edits)
        if isinstance(field.get("evidence"), list):
            field["evidence"] = [dict(e, start=_shift(e["start"], edits), end=_shift(e["end"], edits))
                                 for e in field["evidence"]]
        if field.get("spans"):
            field["spans"] = [[_shift(a, edits), _shift(b, edits)] for a, b in field["spans"]]
        fields[name] = field
    return fields


def _with_span_values(row: dict) -> dict:
    """Record each evidence span's value from its digits, before any rewrite."""
    from experiments.evidence_data import field_evidence

    fields = {}
    for name, field in row["fields"].items():
        field = dict(field)
        spans = field_evidence(field) or []
        if spans and "span_values" not in field:
            kinds = span_kinds(field)
            values = [span_value(row["text"][a:b], k, field.get("dayfirst", True)) for (a, b), k in zip(spans, kinds)]
            field["span_kinds"] = kinds
            field["span_values"] = [v.isoformat() if isinstance(v, date) else (str(v) if v is not None else None)
                                    for v in values]
        fields[name] = field
    return {**row, "fields": fields}


def words_variant(row: dict, rng: random.Random, share: float = 0.6, tag: str = "w") -> dict | None:
    """Rewrite some marked numbers and dates in words. None when nothing can be rewritten."""
    from experiments.evidence_data import field_evidence

    row = _with_span_values(row)
    targets: dict[tuple[int, int], tuple[str, object, int]] = {}
    for field in row["fields"].values():
        spans = field_evidence(field) or []
        for (a, b), kind, value in zip(spans, field.get("span_kinds", []), field.get("span_values", [])):
            if value is None or kind is None:
                continue
            if kind == "date" or (kind == "date_or_count" and "-" in str(value)):
                targets[(a, b)] = ("date", date.fromisoformat(value), 0)
            else:
                decimals = 0 if (kind == "date_or_count" or Decimal(value) == Decimal(value).to_integral()) else 2
                targets[(a, b)] = ("number", Decimal(value), decimals)
    chosen = [span for span in sorted(targets) if rng.random() < share]
    if not chosen and targets:
        chosen = [rng.choice(sorted(targets))]
    if not chosen:
        return None
    text, edits, pieces, cursor = row["text"], [], [], 0
    for a, b in chosen:
        kind, value, decimals = targets[(a, b)]
        lead = text[a:b][: len(text[a:b]) - len(text[a:b].lstrip())]
        words = date_words(value, rng) if kind == "date" else number_words(value, decimals, rng)
        new = lead + words
        pieces += [text[cursor:a], new]
        edits.append((a, b, len(new)))
        cursor = b
    pieces.append(text[cursor:])
    new_text = "".join(pieces)
    fields = _move_spans(row, edits)
    for name, field in fields.items():
        if field.get("type") == "text" and not field.get("missing") and field.get("start") is not None:
            field["value"] = new_text[field["start"]:field["end"]]
        if field.get("type") == "list" and field.get("spans"):
            field["value"] = [new_text[a:b] for a, b in field["spans"]]
    return {**row, "id": f"{row['id']}~{tag}", "text": new_text, "fields": fields, "augment": "words"}


# ---- emit fields -------------------------------------------------------------

# (kind, sentence, instruction). {r} rate, {base} amount, {a} {b} {c} readings, {n} days, {d} date.
TRAIN_TEMPLATES = (
    ("percent_of", "Tax is charged at {r}% on {base}.", "Tax amount at the stated rate"),
    ("percent_of", "A service charge of {r}% applies to {base}.", "Service charge amount"),
    ("discount", "{base} less a {r}% discount is what is owed.", "Amount owed after the discount"),
    ("discount", "The list price {base} carries {r}% off for members.", "Member price after the reduction"),
    ("average", "The three readings were {a}, {b} and {c}.", "Average of the three readings"),
    ("average", "Logged weights: {a}, {b}, {c}.", "Mean of the logged weights"),
    ("date_minus", "Goods must ship {n} days before {d}.", "Last shipping date"),
    ("date_minus", "Payment clears {n} days ahead of {d}.", "Date payment must clear by"),
)
TEST_TEMPLATES = (
    ("percent_of", "The agent keeps {r}% of {base} as commission.", "Commission the agent keeps"),
    ("discount", "After knocking {r}% off {base}, the rest is due.", "Amount due after the reduction"),
    ("average", "Weighings came to {a}, {b} and {c}.", "Mean of the weighings"),
    ("date_minus", "Notice is needed {n} days ahead of {d}.", "Date notice must be given by"),
)
RATES = (5, 8, 10, 12, 15, 18, 20, 25)


def _money(rng: random.Random, low: int, high: int) -> Decimal:
    return Decimal(rng.randint(low * 100, high * 100)) / 100


def _fmt(value: Decimal) -> str:
    return f"{value:,.2f}"


def _compute(kind: str, slots: dict):
    q = Decimal("0.01")
    if kind == "percent_of":
        return (slots["base"] * slots["r"] / 100).quantize(q, ROUND_HALF_UP)
    if kind == "discount":
        return (slots["base"] * (100 - slots["r"]) / 100).quantize(q, ROUND_HALF_UP)
    if kind == "average":
        return ((slots["a"] + slots["b"] + slots["c"]) / 3).quantize(q, ROUND_HALF_UP)
    if kind == "date_minus":
        return slots["d"] - timedelta(days=slots["n"])
    raise ValueError(kind)


def emit_variant(row: dict, rng: random.Random, templates=TRAIN_TEMPLATES, tag: str = "e") -> dict | None:
    """Insert one sentence and a field whose answer no listed operation computes."""
    kind, sentence, instruction = rng.choice(templates)
    for _ in range(20):
        slots = {"r": Decimal(rng.choice(RATES)), "base": _money(rng, 100, 5000),
                 "a": _money(rng, 10, 200), "b": _money(rng, 10, 200), "c": _money(rng, 10, 200),
                 "n": rng.randint(3, 30), "d": date(rng.choice((2026, 2027)), rng.randint(1, 12), rng.randint(1, 28))}
        value = _compute(kind, slots)
        rendered = {"r": str(slots["r"]), "base": _fmt(slots["base"]), "a": _fmt(slots["a"]), "b": _fmt(slots["b"]),
                    "c": _fmt(slots["c"]), "n": str(slots["n"]), "d": f"{slots['d'].day} {MONTH_NAMES[slots['d'].month - 1]} {slots['d'].year}"}
        if isinstance(value, Decimal) and (number_present(row["text"] + sentence.format(**rendered), value, 2)):
            continue
        if isinstance(value, date) and value.isoformat() in row["text"]:
            continue
        break
    else:
        return None
    # Insert after a sentence end, never inside a marked span.
    from experiments.evidence_data import field_evidence

    marked = [span for f in row["fields"].values() for span in (field_evidence(f) or [])]
    cuts = [i + 2 for i in range(len(row["text"]) - 1) if row["text"][i] == "." and row["text"][i + 1] == " "
            and not any(a <= i + 2 < b for a, b in marked)]
    at = rng.choice(cuts) if cuts else len(row["text"])
    prefix = row["text"][:at]
    filled = sentence.format(**rendered)
    insert = filled + (" " if at < len(row["text"]) else "")
    if at == len(row["text"]) and not prefix.endswith(" "):
        insert, prefix_pad = " " + filled, 1
    else:
        prefix_pad = 0
    new_text = prefix + insert + row["text"][at:]
    fields = _move_spans(row, [(at, at, len(insert))])
    # The new field's evidence: where each slot landed in the inserted sentence.
    order = [name for name in ("r", "base", "a", "b", "c", "n", "d") if "{" + name + "}" in sentence]
    order.sort(key=sentence.index)
    evidence, kinds, values = [], [], []
    for name in order:
        start = at + prefix_pad + filled.index(rendered[name])
        evidence.append({"start": start, "end": start + len(rendered[name]), "text": rendered[name]})
        kinds.append("date" if name == "d" else "number")
        values.append(slots[name].isoformat() if name == "d" else str(slots[name]))
    is_date = kind == "date_minus"
    new_field = {
        "type": "date" if is_date else "number",
        "instruction": instruction,
        "compute": "emit",
        "emit_kind": kind,
        "missing": False,
        "value": value.isoformat() if is_date else float(value),
        "evidence": evidence,
        "span_kinds": kinds,
        "span_values": values,
        **({} if is_date else {"decimals": 2}),
    }
    name = f"emit_{kind}"
    fields = {**fields, name: new_field}
    return {**row, "id": f"{row['id']}~{tag}", "text": new_text, "fields": fields, "augment": "emit"}


def augment(rows: list[dict], seed: int, words: int = 1, emits: int = 1, templates=TRAIN_TEMPLATES) -> list[dict]:
    """Variants of each row: `words` words-variants and `emits` emit-variants."""
    rng = random.Random(seed)
    out = []
    for row in rows:
        for k in range(words):
            variant = words_variant(row, rng, tag=f"w{k}")
            if variant:
                out.append(variant)
        for k in range(emits):
            variant = emit_variant(row, rng, templates=templates, tag=f"e{k}")
            if variant:
                out.append(variant)
    return out


def base_id(doc: str) -> str:
    return doc.split("~", 1)[0]
