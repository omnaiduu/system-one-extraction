"""The baseline the user asked for: the same Gemma 4 E2B, untrained, writing JSON.

Gemma gets a system prompt (how to answer each type, the rules on nulls,
decoys and calculations, and three worked examples written for this purpose,
not taken from any exam), reads the field list and the note, and generates one
JSON object, with
decoding constrained to a JSON schema (lm-format-enforcer) when that library
works with this model, else free generation and a JSON parse (the mode used is
recorded). No LoRA, no heads, no training.

Scored with the same `values.matches` as our system, on the same questions:

- exams 1-3, all 293 fields, zero-shot and 3-shot (three training notes with
  their gold JSON as examples);
- the computed fields of experiment 6's three test sets (only the computed
  field is asked), 3-shot;
- exams 1-3 with numbers and dates in words (experiment 5's dev_words), 3-shot;
- SROIE test receipts, zero-shot, company / date / address / total.

The model's strings are read leniently, as production code would: a number
given as "1,486.75" or a date as "4 June 2026" is parsed by the auditor's
parsers. Text is scored exactly (as for our system) and also with case,
spacing and punctuation ignored.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from decimal import Decimal

from experiments.evidence_data import questions
from experiments.values import matches
from extractor.audit import Drop, parse_date, parse_datetime, parse_number, parse_time

SYSTEM = """You are a precise information-extraction engine. You read one document and return the values of the requested fields as a single JSON object. You never explain, never add keys, never wrap the JSON in markdown.

How to answer each type:
- text: copy the exact words from the document, character for character (same spelling, capitals and punctuation). Do not shorten, expand or tidy them.
- number: a plain JSON number, no currency, no thousands separators, no units: 1486.75, not "Rs 1,486.75".
- date: "YYYY-MM-DD". Read any format, including words ("the second of April, 2026" -> "2026-04-02"). Two-number dates like 04/06/2026 are day first.
- time: "HH:MM" on a 24-hour clock ("2:15 pm" -> "14:15").
- date-time: "YYYY-MM-DDTHH:MM".
- days: a whole number of days.
- list: a JSON array of the exact items, in the order they appear in the document.

Rules that matter:
1. Use null when the document does not contain the value. Never guess, never invent.
2. Documents contain decoys: other invoices, old prices, a spare part that "is not this ticket", a reference "not copied onto this note". Answer for the thing the field asks about, not the decoy.
3. When a value is not printed but can be calculated from what is printed, calculate it exactly: totals, differences, percentages, discounts, taxes, averages, per-person shares, dates plus or minus days or weeks, days between dates. Work in the order the document describes. Round money to 2 decimals only at the end.
4. Numbers written in words are still numbers: "one lakh twenty thousand" is 120000; "fourteen days" is 14.
5. Instructions after each field name say exactly what is wanted; follow them literally.

Example 1
Fields:
- supplier (text): Company that issued the invoice
- invoice_no (text): Invoice number of this invoice
- total (number): Total of the line items
- due (date): Payment due date
- tracking (text): Courier tracking number
Document:
Adriatic Pipeworks d.o.o. - invoice AP-2745, dated 4 June 2026. Pipes 1,200.00 and fittings 286.75. Payment within 21 days. The cancelled invoice AP-2749 should be ignored.
Answer:
{"supplier": "Adriatic Pipeworks d.o.o.", "invoice_no": "AP-2745", "total": 1486.75, "due": "2026-06-25", "tracking": null}

Example 2
Fields:
- final_price (number): Price after the discount and the tax
- share (number): What each flatmate pays
- arrival (date): Date the parcel arrives
Document:
Subtotal Rs 2,400. 15% off for members, then 18% GST on the discounted amount. Rent of 900 is split equally between 4 flatmates. Shipped 10/04/2026; it arrives 3 weeks later.
Answer:
{"final_price": 2407.2, "share": 225, "arrival": "2026-05-01"}

Example 3
Fields:
- amount (number): Amount on the cheque
- signed (date): Date the cheque was signed
- items (list): Items ordered
- pickup (time): Pickup time
- nights (days): Length of the stay
Document:
Pay Rupees One Lakh Twenty Thousand Only. Signed on the second of April, 2026. Order: oak table, two chairs, a lamp. Pickup at 2:15 pm. Check-in 3 March 2026, check-out 7 March 2026.
Answer:
{"amount": 120000, "signed": "2026-04-02", "items": ["oak table", "two chairs", "a lamp"], "pickup": "14:15", "nights": 4}
"""
# The user turn: the fields, then the document, in the same layout as the examples.
TYPE_HINT = {"text": "text", "list": "list of text", "number": "number", "estimate": "number", "duration": "days",
             "date": "date", "time": "time", "datetime": "date-time"}


def schema_for(fields: list[dict]) -> dict:
    props = {}
    for f in fields:
        kind = f["type"]
        if kind == "list":
            props[f["field"]] = {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]}
        elif kind in ("number", "estimate"):
            props[f["field"]] = {"anyOf": [{"type": "number"}, {"type": "null"}]}
        elif kind == "duration":
            props[f["field"]] = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
        else:
            props[f["field"]] = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    return {"type": "object", "properties": props, "required": [f["field"] for f in fields]}


def field_list(row: dict, only: list[str] | None = None) -> list[dict]:
    out = []
    for q in questions(row):
        if only is not None and q["field"] not in only:
            continue
        spec = row["fields"][q["field"]]
        out.append({"field": q["field"], "type": q["type"], "instruction": spec["instruction"], "spec": spec, "op": q["op"]})
    return out


def user_message(row: dict, fields: list[dict]) -> str:
    lines = ["Fields:"]
    lines += [f"- {f['field']} ({TYPE_HINT.get(f['type'], f['type'])}): {f['instruction']}" for f in fields]
    lines += ["Document:", row["text"], "Answer:"]
    return "\n".join(lines)


def gold_json(fields: list[dict]) -> str:
    out = {}
    for f in fields:
        spec = f["spec"]
        out[f["field"]] = None if spec.get("missing") else spec.get("value")
    return json.dumps(out, ensure_ascii=False)


def messages_for(row: dict, fields: list[dict], shots: list[tuple[dict, list[dict]]]) -> list[dict]:
    """The system prompt (rules and three worked examples), then optional real training notes
    as earlier turns with their gold JSON, then this note."""
    messages = [{"role": "system", "content": SYSTEM}]
    for shot_row, shot_fields in shots:
        messages.append({"role": "user", "content": user_message(shot_row, shot_fields)})
        messages.append({"role": "assistant", "content": gold_json(shot_fields)})
    messages.append({"role": "user", "content": user_message(row, fields)})
    return messages


def parse_reply(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        out = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


def to_prediction(value, spec: dict) -> dict:
    """The model's value in the shape `values.matches` expects, read leniently."""
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "null", "none", "n/a")):
        return {"missing": True, "value": None}
    kind = spec["type"]
    dayfirst = spec.get("dayfirst", True)
    try:
        if kind in ("number", "estimate"):
            return {"missing": False, "value": float(value if isinstance(value, (int, float)) else parse_number(str(value)))}
        if kind == "duration":
            return {"missing": False, "value": int(value if isinstance(value, (int, float)) else parse_number(str(value)))}
        if kind == "date":
            text = str(value)
            return {"missing": False, "value": parse_date(text, dayfirst).isoformat()}
        if kind == "time":
            return {"missing": False, "value": parse_time(str(value)).strftime("%H:%M")}
        if kind == "datetime":
            text = str(value).replace(" ", "T", 1) if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}", str(value)) else str(value)
            return {"missing": False, "value": parse_datetime(text, dayfirst).strftime("%Y-%m-%dT%H:%M")}
        if kind == "list":
            return {"missing": False, "value": [str(v) for v in value] if isinstance(value, list) else [str(value)]}
        return {"missing": False, "value": str(value)}
    except (Drop, ValueError, TypeError, ArithmeticError):
        return {"missing": False, "value": None}


def loose(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def score_fields(items: list[tuple[dict, dict | None]]) -> dict:
    """items: (field, parsed reply or None). Exact (our metric) and loose-text scores, by operation."""
    hits, loose_hits, by_op, examples = 0, 0, defaultdict(Counter), []
    for f, reply in items:
        spec = f["spec"]
        raw = None if reply is None else reply.get(f["field"])
        predicted = to_prediction(raw, spec)
        ok = matches(spec, predicted)
        ok_loose = ok
        if not ok and spec["type"] == "text" and not spec.get("missing") and not predicted["missing"]:
            ok_loose = loose(predicted["value"]) == loose(spec.get("value"))
        if not ok and spec["type"] == "list" and not spec.get("missing") and isinstance(predicted["value"], list):
            ok_loose = [loose(v) for v in predicted["value"]] == [loose(v) for v in spec.get("value") or []]
        hits += ok
        loose_hits += ok_loose
        by_op[f["op"]][ok] += 1
        if len(examples) < 300:
            examples.append({"field": f["field"], "op": f["op"], "gold": None if spec.get("missing") else spec.get("value"),
                             "said": raw, "right": ok, "right_loose": ok_loose})
    return {"fields": len(items), "exact": hits, "exact_loose_text": loose_hits,
            "by_op": {op: f"{c[True]}/{c[True] + c[False]}" for op, c in sorted(by_op.items())},
            "examples": examples}


def score_sroie(items: list[tuple[dict, dict, dict | None]]) -> dict:
    """items: (row, field, reply). Text fields by letters and digits (our 'found'), dates and totals by value."""
    from experiments.sroie import letters_digits

    per = defaultdict(lambda: {"fields": 0, "right": 0})
    examples = []
    for row, f, reply in items:
        spec = f["spec"]
        raw = None if reply is None else reply.get(f["field"])
        if spec["type"] == "text":
            gold = row["text"][spec["start"] : spec["end"]].strip()
            ok = raw is not None and letters_digits(str(raw)) == letters_digits(gold)
        elif spec["type"] == "date":
            from experiments.sroie import read_date_wide

            try:
                ok = raw is not None and read_date_wide(str(raw)).isoformat() == spec["value"]
            except Drop:
                ok = matches(spec, to_prediction(raw, spec))
        else:
            ok = matches(spec, to_prediction(raw, spec))
        per[f["field"]]["fields"] += 1
        per[f["field"]]["right"] += bool(ok)
        if len(examples) < 120:
            examples.append({"doc": row["id"], "field": f["field"], "gold": spec.get("value"), "said": raw, "right": bool(ok)})
    total = {"fields": sum(p["fields"] for p in per.values()), "right": sum(p["right"] for p in per.values())}
    return {"overall": total, "by_field": dict(per), "examples": examples}


def pick_shots(rows: list[dict], count: int = 3) -> list[tuple[dict, list[dict]]]:
    """Training notes whose fields together cover the most field types (fixed, so every test note sees the same)."""
    train = [r for r in rows if r["split"] == "train" and "~" not in r["id"]]
    chosen, covered = [], set()
    for _ in range(count):
        best = max((r for r in train if r not in chosen),
                   key=lambda r: (len({q["type"] for q in questions(r)} - covered), -len(r["text"])))
        chosen.append(best)
        covered |= {q["type"] for q in questions(best)}
    return [(r, field_list(r)) for r in chosen]
