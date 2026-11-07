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

from datetime import timedelta
from decimal import Decimal

from extractor.audit import Drop, parse_date, parse_datetime, parse_number, parse_time, quantize

OPS = (
    "copy", "list", "read_number", "read_date", "read_time", "read_datetime",
    "sum", "subtract", "product", "add_days", "days_between", "median",
)
ALLOWED = {
    "text": ("copy",),
    "list": ("list",),
    "number": ("read_number", "sum", "subtract", "product"),
    "duration": ("days_between",),
    "date": ("read_date", "add_days"),
    "time": ("read_time",),
    "datetime": ("read_datetime",),
    "estimate": ("median",),
}


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


def answer(op: str, parts: list[str], field: dict):
    """Missing when nothing is marked; None when the parts cannot be read."""
    if not parts:
        return {"missing": True, "value": None}
    try:
        return {"missing": False, "value": execute(op, parts, field)}
    except (Drop, ValueError, ArithmeticError):
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
