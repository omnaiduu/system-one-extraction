"""The baseline the user asked for: the same Gemma 4 E2B, untrained, writing JSON.

Gemma reads the field list and the note and generates one JSON object, with
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

RULES = (
    "Extract the fields below from the document and reply with one JSON object only, one key per field.\n"
    "Use null when the value is not in the document. Formats: dates YYYY-MM-DD; times HH:MM (24-hour); "
    "date-times YYYY-MM-DDTHH:MM; numbers as plain numbers without currency or thousands separators; "
    "durations as a whole number of days; lists as arrays of strings in document order; text exactly as "
    "written in the document. When a value must be calculated from the document, calculate it."
)
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
    lines = [RULES, "", "Fields:"]
    lines += [f"- {f['field']} ({TYPE_HINT.get(f['type'], f['type'])}): {f['instruction']}" for f in fields]
    lines += ["", "Document:", row["text"]]
    return "\n".join(lines)


def gold_json(fields: list[dict]) -> str:
    out = {}
    for f in fields:
        spec = f["spec"]
        out[f["field"]] = None if spec.get("missing") else spec.get("value")
    return json.dumps(out, ensure_ascii=False)


def messages_for(row: dict, fields: list[dict], shots: list[tuple[dict, list[dict]]]) -> list[dict]:
    messages = []
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
