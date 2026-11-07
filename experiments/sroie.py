"""SROIE receipts as a zero-shot test on real documents.

SROIE (ICDAR 2019) is a public research benchmark of scanned receipts, mostly
from Malaysia, with OCR text lines and four labelled fields: company, date,
address, total. The Kaggle copy used here is `urbikn/sroie-datasetv2`; its
page lists no licence, so it is used for research evaluation only and none of
it is committed to this repo.

Only the **test split** (347 receipts) is used, and only for scoring. The
model never trains on receipts, and these field instructions are not phrased
like the synthetic training notes, so this is a zero-shot test of transfer to
real, messy text.

Conversion, per receipt:
- text: the OCR lines, in file order, joined with newlines;
- each label is located in that text (for company and address, comparing
  letters and digits only, so spacing and punctuation differences between the
  label and the OCR do not count; for the total, the amount's digits, preferring a line that says
  TOTAL); the located slice becomes the gold span and, for text fields, the
  gold value;
- a label that cannot be found in the text is left out and counted. SROIE's
  labels have known typos (e.g. an address label with B1750 where the receipt
  says 81750).

Three scores, reported separately:
- found: the model marked one span whose letters and digits equal the gold
  slice's (for the total, whose number equals the gold amount); independent of
  any parser. For company and address this is also the value score;
- value: the final answer with the reader as it is (the auditor's parsers);
- value_wide_dates: the same, but dates read with `read_date_wide`, which adds
  dd-mm-yyyy, dd.mm.yyyy, two-digit years and dd-MON-yyyy. That reader was
  written after looking at a 30-receipt sample of this test split, so this
  number is not a clean test.
"""

from __future__ import annotations

import json
import re
from datetime import date
from decimal import Decimal
from pathlib import Path

from extractor.audit import MONTHS, Drop, parse_number

FIELDS = {
    "company": ("text", "Name of the company that issued the receipt"),
    "date": ("date", "Date printed on the receipt"),
    "address": ("text", "Address of the company that issued the receipt"),
    "total": ("number", "Total amount paid"),
}


def read_box(raw: str) -> list[str]:
    """OCR lines from a SROIE box file: 8 coordinates, then the text."""
    lines = []
    for line in raw.splitlines():
        parts = line.split(",", 8)
        if len(parts) == 9 and parts[8].strip():
            lines.append(parts[8].strip())
    return lines


def letters_digits(text: str) -> str:
    """Upper-case letters and digits only: how label and OCR text are compared."""
    return "".join(ch for ch in text.upper() if ch.isalnum())


def locate_text(text: str, label: str) -> tuple[int, int] | None:
    """Find the label in the text, ignoring spaces, punctuation and case.

    The label and the OCR often differ by spacing or punctuation ("NO.3" and
    "NO. 3", "D.E" and "D. E"). A real typo in the label (B1750 for 81750)
    still fails to match, and the field is left out.
    """
    target = letters_digits(label)
    if not target:
        return None
    keep = [i for i, ch in enumerate(text) if ch.isalnum()]
    flat = "".join(text[i].upper() for i in keep)
    at = flat.find(target)
    if at < 0:
        return None
    return keep[at], keep[at + len(target) - 1] + 1


def locate_total(text: str, label: str) -> tuple[int, int, Decimal] | None:
    try:
        amount = parse_number(label)
    except Drop:
        return None
    digits = re.sub(r"^[^\d]+", "", label.strip().replace(" ", ""))
    candidates = []
    for pattern in {re.escape(digits), re.escape(f"{amount:.2f}")}:
        for match in re.finditer(rf"(?<![\d.]){pattern}(?![\d])", text):
            candidates.append((match.start(), match.end()))
    if not candidates:
        return None
    def says_total(line: str) -> bool:
        line = line.lower()
        return "total" in line and "sub" not in line

    lines_at = []
    for start, end in sorted(set(candidates)):
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", end)
        line = text[line_start : line_end if line_end >= 0 else len(text)]
        prev_start = text.rfind("\n", 0, max(line_start - 1, 0)) + 1
        prev = text[prev_start : max(line_start - 1, 0)] if line_start else ""
        # Judge the amount's own line; use the line above only when the amount stands alone.
        alone = not re.search(r"[A-Za-z]", line)
        lines_at.append((says_total(line) or (alone and says_total(prev)), start, end))
    preferred = [c for c in lines_at if c[0]] or lines_at
    _, start, end = preferred[0]
    return start, end, amount


def read_date_wide(raw: str) -> date:
    """Common receipt date forms, day first. Raises Drop when it cannot read one."""
    text = " ".join(raw.strip().split()).rstrip(".,")
    numeric = re.fullmatch(r"(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{2,4})", text)
    if numeric:
        a, b, c = numeric.groups()
        if len(a) == 4:
            year, month, day = int(a), int(b), int(c)
        else:
            day, month, year = int(a), int(b), int(c)
            if len(c) == 2:
                year += 2000
        try:
            return date(year, month, day)
        except ValueError as exc:
            raise Drop(f"unparsable_date:{raw}") from exc
    named = re.fullmatch(r"(\d{1,2})[\s\-/]*([A-Za-z]{3,9})[\s\-/,]*(\d{2,4})", text)
    if named:
        day, month_name, year = int(named.group(1)), named.group(2).lower(), int(named.group(3))
        month = MONTHS.get(month_name)
        if month is None:
            raise Drop(f"unparsable_date:{raw}")
        if year < 100:
            year += 2000
        try:
            return date(year, month, day)
        except ValueError as exc:
            raise Drop(f"unparsable_date:{raw}") from exc
    raise Drop(f"unparsable_date:{raw}")


def build_row(receipt_id: str, box_raw: str, entities: dict) -> tuple[dict | None, list[str]]:
    """One receipt as a row in the same shape as an audited note."""
    lines = read_box(box_raw)
    text = "\n".join(lines)
    fields, skipped = {}, []
    for name, (kind, instruction) in FIELDS.items():
        label = str(entities.get(name) or "").strip()
        if not label:
            skipped.append(f"{name}:no_label")
            continue
        if name == "total":
            found = locate_total(text, label)
            if found is None:
                skipped.append(f"{name}:not_in_text")
                continue
            start, end, amount = found
            fields[name] = {"type": "number", "instruction": instruction, "decimals": 2, "missing": False,
                            "value": float(amount), "start": start, "end": end}
            continue
        span = locate_text(text, label)
        if span is None:
            skipped.append(f"{name}:not_in_text")
            continue
        start, end = span
        if name == "date":
            try:
                iso = read_date_wide(label).isoformat()
            except Drop:
                skipped.append(f"{name}:unreadable_label")
                continue
            fields[name] = {"type": "date", "instruction": instruction, "missing": False, "value": iso,
                            "start": start, "end": end, "dayfirst": True}
        else:
            fields[name] = {"type": "text", "instruction": instruction, "missing": False,
                            "value": text[start:end], "start": start, "end": end}
    if not fields:
        return None, skipped
    return {"id": f"sroie-{receipt_id}", "split": "sroie", "text": text, "fields": fields}, skipped


def load_split(root: Path) -> tuple[list[dict], dict]:
    """Rows for every receipt under root/box and root/entities, plus skip counts."""
    rows, skipped = [], {}
    for box_path in sorted((root / "box").glob("*.txt")):
        entity_path = root / "entities" / box_path.name
        if not entity_path.exists():
            skipped["no_entities_file"] = skipped.get("no_entities_file", 0) + 1
            continue
        try:
            entities = json.loads(entity_path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            skipped["bad_entities_json"] = skipped.get("bad_entities_json", 0) + 1
            continue
        row, why = build_row(box_path.stem, box_path.read_text(encoding="utf-8", errors="replace"), entities)
        for reason in why:
            skipped[reason] = skipped.get(reason, 0) + 1
        if row is not None:
            rows.append(row)
    return rows, skipped


def score(examples: list[dict], tag_logits, op_logits, bias: float) -> dict:
    """found / value / value_wide_dates, per field and overall."""
    from experiments.lora_tagger import field_confidence, final_answer, selective
    from experiments.values import matches, span_texts

    per = {name: {"fields": 0, "found": 0, "value": 0, "value_wide_dates": 0} for name in FIELDS}
    confidences, rights, kept = [], [], []
    for index, ex in enumerate(examples):
        spans, op, predicted = final_answer(ex, tag_logits[index], op_logits[index], bias)
        parts = span_texts(ex["text"], ex["doc_offsets"], spans)
        spec = ex["spec"]
        gold_slice = ex["text"][spec["start"] : spec["end"]].strip()
        if ex["type"] == "number":
            try:
                found = len(parts) == 1 and float(parse_number(parts[0])) == spec["value"]
            except Drop:
                found = False
        else:
            found = len(parts) == 1 and letters_digits(parts[0]) == letters_digits(gold_slice)
        value_ok = matches(spec, predicted)
        wide_ok = value_ok
        if ex["type"] == "date" and len(parts) == 1:
            try:
                wide_ok = read_date_wide(parts[0]).isoformat() == spec["value"]
            except Drop:
                wide_ok = False
        if ex["type"] == "text":
            value_ok = wide_ok = found
        row = per[ex["field"]]
        row["fields"] += 1
        row["found"] += found
        row["value"] += value_ok
        row["value_wide_dates"] += wide_ok
        conf = field_confidence(ex, tag_logits[index], op_logits[index], bias, spans, op)
        confidences.append(conf["confidence"])
        rights.append(bool(wide_ok))
        if len(kept) < 120:
            kept.append({"doc": ex["doc"], "field": ex["field"], "gold": spec["value"], "marked": parts,
                         "op": op, "found": found, "value": value_ok, "value_wide_dates": wide_ok,
                         "confidence": round(conf["confidence"], 6)})
    total = {k: sum(row[k] for row in per.values()) for k in ("fields", "found", "value", "value_wide_dates")}
    return {"overall": total, "by_field": per, "confidence": selective(confidences, rights), "examples": kept}
