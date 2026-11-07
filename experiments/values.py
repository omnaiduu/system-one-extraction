"""Turn marked evidence into a final answer, and score it against the gold value.

Read: the marked words are parsed with the auditor's own parsers
(`parse_number`, `parse_date`, `parse_time`, `parse_datetime`).
Combine: the operation (copy, sum, subtract, product, add_days,
days_between, median, ...) is applied to the parsed parts, exactly as the
auditor computed the gold label.

The operation comes from a small head trained with the tagger. Only the
choices legal for the field's type are allowed. A field with no marked
spans is answered "missing".

Known limit: the parsers read digits and the date formats the auditor
accepts. A number written in words would not parse, and the field would be
wrong. No field in this data is written in words, so this run cannot measure
that case.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from extractor.audit import Drop, parse_date, parse_datetime, parse_number, parse_time, quantize

OPS = (
    "copy", "list", "read_number", "read_date", "read_time", "read_datetime",
    "sum", "subtract", "product", "add_days", "days_between", "median", "emit",
    "average", "chain",
)
# "emit": no listed operation fits, so the emit head writes the value itself
# (experiment 5; it learned nothing, and is kept only as a flagged last resort).
# "average": the mean of every marked amount.
# "chain": the marked parts, in text order, folded through one block per step
# (see BLOCKS). The blocks are chosen by the step head; the arithmetic is here.
# New operations go at the end so older checkpoints keep their indexes.
ALLOWED = {
    "text": ("copy",),
    "list": ("list",),
    "number": ("read_number", "sum", "subtract", "product", "emit", "average", "chain"),
    "duration": ("days_between",),
    "date": ("read_date", "add_days", "emit", "chain"),
    "time": ("read_time",),
    "datetime": ("read_datetime",),
    "estimate": ("median",),
}


# ---- blocks for chains --------------------------------------------------------
#
# A chain folds the marked parts left to right: acc = part 1, then for each
# step k, acc = block_k(acc, part k+1). The number of steps is fixed by how many
# parts are marked; the step head only chooses each block. The percent blocks
# take the rate from whichever operand is a marked percentage ("15" followed by
# "%"), so "15% off 2,400" and "2,400 less 15%" use the same block. The date
# blocks take the date from whichever operand is a date.

NUMBER_BLOCKS = ("add", "subtract", "multiply", "divide", "percent_of", "minus_percent", "plus_percent")
DATE_BLOCKS = ("add_days", "minus_days", "add_weeks", "minus_weeks")
BLOCKS = NUMBER_BLOCKS + DATE_BLOCKS
MAX_STEPS = 3


def blocks_for(kind: str) -> tuple[str, ...]:
    """The blocks a chain may use for an answer of this type."""
    return DATE_BLOCKS if kind == "date" else NUMBER_BLOCKS


def is_rate(text: str, end: int, part: str = "") -> bool:
    """A marked number is a rate when it carries or is followed by a percent sign or 'per cent'."""
    if "%" in part:
        return True
    after = text[end : end + 12].lstrip().lower()
    return after.startswith("%") or after.startswith("per cent") or after.startswith("percent")


def apply_block(block: str, acc, x, acc_rate: bool = False, x_rate: bool = False):
    """One step of a chain. Exact: Decimal amounts, date objects."""
    if block in NUMBER_BLOCKS:
        if isinstance(acc, date) or isinstance(x, date):
            raise ValueError("number block on a date")
        if block == "add":
            return acc + x
        if block == "subtract":
            return acc - x
        if block == "multiply":
            return acc * x
        if block == "divide":
            return acc / x
        amount, rate = (x, acc) if acc_rate and not x_rate else (acc, x)
        if block == "percent_of":
            return amount * rate / 100
        if block == "minus_percent":
            return amount * (100 - rate) / 100
        if block == "plus_percent":
            return amount * (100 + rate) / 100
    if block in DATE_BLOCKS:
        dates = [v for v in (acc, x) if isinstance(v, date)]
        counts = [v for v in (acc, x) if not isinstance(v, date)]
        if len(dates) != 1 or len(counts) != 1 or counts[0] != counts[0].to_integral_value():
            raise ValueError("a date block needs one date and one whole count")
        days = int(counts[0]) * (7 if block.endswith("weeks") else 1)
        return dates[0] + timedelta(days=-days if block.startswith("minus") else days)
    raise ValueError(block)


def run_chain(blocks: list[str], values: list, rates: list[bool] | None = None):
    """Fold the parts through the blocks. None when the counts do not line up."""
    if len(values) < 2 or len(blocks) != len(values) - 1 or len(blocks) > MAX_STEPS:
        return None
    rates = list(rates or [False] * len(values))
    acc, acc_rate = values[0], rates[0]
    for block, x, x_rate in zip(blocks, values[1:], rates[1:]):
        acc = apply_block(block, acc, x, acc_rate, x_rate)
        acc_rate = False  # a computed value is never a rate
    return acc


def span_texts(text: str, doc_offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[str]:
    return [text[max(doc_offsets[a][0], 0) : doc_offsets[b][1]].strip() for a, b in spans]


def execute(op: str, parts: list[str], field: dict):
    """The answer the auditor would have computed from these marked parts."""
    dayfirst = field.get("dayfirst", True)
    decimals = int(field.get("decimals") or 0)
    if op == "copy":
        return parts[0] if len(parts) == 1 else None
    if op == "list":
        return list(parts)
    if op == "read_number":
        return float(quantize(parse_number(parts[0]), int(field.get("decimals", 2)))) if len(parts) == 1 else None
    if op == "read_date":
        return parse_date(parts[0], dayfirst).isoformat() if len(parts) == 1 else None
    if op == "read_time":
        return parse_time(parts[0]).strftime("%H:%M") if len(parts) == 1 else None
    if op == "read_datetime":
        return parse_datetime(parts[0], dayfirst).strftime("%Y-%m-%dT%H:%M") if len(parts) == 1 else None
    if op == "sum":
        if len(parts) < 2:
            return None
        return float(quantize(sum((parse_number(p) for p in parts), Decimal(0)), int(field.get("decimals", 2))))
    if op == "product":
        if len(parts) != 2:
            return None
        return float(quantize(parse_number(parts[0]) * parse_number(parts[1]), int(field.get("decimals", 2))))
    if op == "subtract":
        if len(parts) != 2:
            return None
        # The first amount in the text minus the second (checked on every gold label).
        return float(quantize(parse_number(parts[0]) - parse_number(parts[1]), int(field.get("decimals", 2))))
    if op == "add_days":
        if len(parts) != 2:
            return None
        dates, counts = [], []
        for part in parts:
            try:
                dates.append(parse_date(part, dayfirst))
            except Drop:
                counts.append(int(parse_number(part)))
        if len(dates) != 1 or len(counts) != 1:
            return None
        return (dates[0] + timedelta(days=counts[0])).isoformat()
    if op == "days_between":
        if len(parts) != 2:
            return None
        first, second = sorted(parse_date(p, dayfirst) for p in parts)
        return (second - first).days
    if op == "median":
        if len(parts) < 2:
            return None
        values = sorted(parse_number(p) for p in parts)
        mid = len(values) // 2
        middle = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / Decimal(2)
        return float(quantize(middle, decimals))
    raise ValueError(op)


NUMBER_OPS = ("read_number", "sum", "subtract", "product", "median", "average")
# Words that may sit next to a plain amount ("RM 32.80", "Rs 500 only").
AMOUNT_WORDS = {"rs", "rm", "usd", "inr", "eur", "gbp", "sgd", "aud", "cad", "myr", "only"}
MONTH_WORDS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")


def plain_number(part: str):
    """parse_number, but refuse text with real words in it.

    The auditor's parse_number strips leading letters, so "April the second,
    2026" would read as 2026. Text with words goes to the learned reader.
    """
    import re

    def is_word(w: str) -> bool:
        # A currency code ("CFA", "MWK", "RM") is short and in capitals; number
        # words ("one thousand", "April the second") are not.
        return len(w) >= 3 and w.lower() not in AMOUNT_WORDS and not (w.isupper() and len(w) <= 3)

    if any(is_word(w) for w in re.findall(r"[A-Za-z]+", part)):
        raise Drop(f"words_in_number:{part}")
    return parse_number(part)


def _looks_like_date(part: str) -> bool:
    lowered = part.lower()
    return any(m in lowered for m in MONTH_WORDS) or bool(__import__("re").search(r"\d{4}", part))
DATE_OPS = ("read_date", "days_between")


def _format(op: str, values: list, field: dict):
    """The answer from already-read parts (Decimal amounts, dates)."""
    decimals = int(field.get("decimals", 2))
    q = lambda v: float(quantize(v, decimals))
    if op == "read_number":
        return q(values[0]) if len(values) == 1 else None
    if op == "read_date":
        return values[0].isoformat() if len(values) == 1 else None
    if op == "sum":
        return q(sum(values, Decimal(0))) if len(values) >= 2 else None
    if op == "product":
        return q(values[0] * values[1]) if len(values) == 2 else None
    if op == "subtract":
        return q(values[0] - values[1]) if len(values) == 2 else None
    if op == "median":
        if len(values) < 2:
            return None
        ordered = sorted(values)
        mid = len(ordered) // 2
        middle = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / Decimal(2)
        return float(quantize(middle, int(field.get("decimals") or 0)))
    if op == "days_between":
        if len(values) != 2:
            return None
        first, second = sorted(values)
        return (second - first).days
    if op == "average":
        return q(sum(values, Decimal(0)) / len(values)) if len(values) >= 2 else None
    if op == "add_days":
        dates = [v for v in values if isinstance(v, date)]
        counts = [v for v in values if not isinstance(v, date)]
        if len(dates) != 1 or len(counts) != 1:
            return None
        return (dates[0] + timedelta(days=int(counts[0]))).isoformat()
    raise ValueError(op)


def _chain_answer(parts: list[str], field: dict, blocks, rates):
    if blocks is None:
        return {"missing": False, "value": None}
    dayfirst = field.get("dayfirst", True)
    values = []
    try:
        for part in parts:
            if field["type"] == "date":
                try:
                    values.append(parse_date(part, dayfirst))
                    continue
                except Drop:
                    pass
            values.append(plain_number(part))
        result = run_chain(list(blocks), values, rates)
    except (Drop, ValueError, ArithmeticError, TypeError):
        return {"missing": False, "value": None}
    if result is None:
        return {"missing": False, "value": None}
    if field["type"] == "date":
        return {"missing": False, "value": result.isoformat() if isinstance(result, date) else None}
    if isinstance(result, date):
        return {"missing": False, "value": None}
    return {"missing": False, "value": float(quantize(result, int(field.get("decimals", 2))))}


def answer(op: str, parts: list[str], field: dict, read=None, emitted=None, blocks=None, rates=None):
    """The final answer for one field.

    Missing when nothing is marked. "emit" returns what the emit head wrote.
    "chain" folds the parts through `blocks` (with `rates`: which parts are
    marked percentages).
    Otherwise each marked part is read by the parser; a part the parser
    cannot read (a number or date in words) goes to `read(index, kind)`, the
    learned reader, when one is given. None when a part stays unreadable.
    """
    if not parts:
        return {"missing": True, "value": None}
    if op == "emit":
        return {"missing": False, "value": emitted}
    if op == "chain":
        return _chain_answer(parts, field, blocks, rates)
    if op not in NUMBER_OPS + DATE_OPS + ("add_days", "average"):
        try:
            return {"missing": False, "value": execute(op, parts, field)}
        except (Drop, ValueError, ArithmeticError):
            return {"missing": False, "value": None}
    dayfirst = field.get("dayfirst", True)
    values = []
    for index, part in enumerate(parts):
        value = None
        try:
            if op in DATE_OPS:
                value = parse_date(part, dayfirst)
            elif op == "add_days":
                try:
                    value = parse_date(part, dayfirst)
                except Drop:
                    value = plain_number(part)
            else:
                value = plain_number(part)
        except Drop:
            if read is not None:
                if op in DATE_OPS:
                    kind = "date"
                elif op == "add_days":
                    kind = "date" if _looks_like_date(part) else "count"
                else:
                    kind = "number"
                value = read(index, kind)
        if value is None:
            return {"missing": False, "value": None}
        values.append(value)
    try:
        return {"missing": False, "value": _format(op, values, field)}
    except (ValueError, ArithmeticError, TypeError):
        return {"missing": False, "value": None}


def matches(field: dict, predicted: dict) -> bool:
    if field.get("missing"):
        return predicted["missing"]
    if predicted["missing"] or predicted["value"] is None:
        return False
    gold = field["value"]
    if field["type"] == "list":
        return [str(v).strip() for v in predicted["value"]] == [str(v).strip() for v in gold]
    if isinstance(gold, float) or field["type"] in ("number", "estimate"):
        return abs(float(predicted["value"]) - float(gold)) < 1e-9
    if field["type"] == "text":
        return str(predicted["value"]).strip() == str(gold).strip()
    return predicted["value"] == gold
