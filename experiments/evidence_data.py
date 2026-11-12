"""Evidence questions for the question-order experiment.

Every field becomes one question: "which words on the page are the evidence
for this field?" The gold answer is the set of bracketed spans the writer
marked, measured by `extractor.audit`. A printed value has one span, a list
has one span per item, a sum has one span per part, a due date has the start
date and the day count. A missing field has no spans. Forecasts are skipped,
because their answer was deleted from the page.

Rows are rebuilt by running the auditor on the raw writer files. That gives
the same text and values as the stored audited files (checked in the tests),
plus evidence spans for add_days and days_between, which the older stored
files do not carry.
"""

from __future__ import annotations

from pathlib import Path

from extractor import audit

# Which raw directory feeds which split. The original 48 keep the auditor's
# own index-mod-10 split. Exam 4 is listed so the cache can be built once, but
# the experiment never scores it.
RAW_SOURCES = (
    ("original", Path("data/raw")),
    ("train", Path("data/round2/raw")),
    ("train", Path("data/round3/raw")),
    ("exam1", Path("data/generalization/raw")),
    ("exam2", Path("data/exam2/raw")),
    ("exam3", Path("data/exam3/raw")),
    ("exam4", Path("data/exam4/raw")),
)

DEV_SPLITS = ("exam1", "exam2", "exam3")

PROMPTS = ("last_nobos", "last", "first", "multi")


def load_rows(root: Path = Path(".")) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for split, rel in RAW_SOURCES:
        kept, dropped = audit.audit_paths(sorted((root / rel).glob("*.jsonl")))
        if dropped:
            raise RuntimeError(f"{rel} dropped {dropped}")
        # The original 48, val and test included, are training rows here: the
        # experiment picks its settings by cross-validation, not on 4 notes.
        name = "train" if split == "original" else split
        for row in kept:
            if row["id"] in seen:
                raise RuntimeError(f"duplicate id {row['id']}")
            seen.add(row["id"])
            rows.append({**row, "split": name})
    return rows


def field_evidence(field: dict) -> list[tuple[int, int]] | None:
    """Gold character spans for one field. None means the field is skipped."""
    kind = field["type"]
    if kind == "forecast":
        return None
    if field.get("missing"):
        return []
    if kind == "list":
        return [tuple(span) for span in field["spans"]]
    evidence = field.get("evidence")
    if isinstance(evidence, list):
        return sorted((item["start"], item["end"]) for item in evidence)
    if "start" in field and "end" in field:
        return [(field["start"], field["end"])]
    raise ValueError(f"field has no evidence: {field}")


def field_operation(field: dict) -> str:
    kind = field["type"]
    if field.get("missing"):
        return "missing"
    if kind == "text":
        return "copy"
    if kind == "list":
        return "list"
    if kind == "estimate":
        return "median"
    if field.get("compute"):
        return field["compute"]
    return f"read_{kind}"


def questions(row: dict) -> list[dict]:
    out = []
    for name, field in row["fields"].items():
        spans = field_evidence(field)
        if spans is None:
            continue
        out.append(
            {
                "doc": row["id"],
                "field": name,
                "type": field["type"],
                "op": field_operation(field),
                "body": f"{name}: {field['instruction']}",
                "spans": spans,
                "value": field.get("value"),
                "decimals": field.get("decimals"),
            }
        )
    return out


def build_prompt(kind: str, text: str, bodies: list[str]) -> tuple[str, int, list[tuple[int, int]]]:
    """Return the prompt, the character offset of the document, and question spans.

    last / last_nobos: the document first, every question after it, the
    layout the earlier rounds used. One forward covers all questions, because
    a later question cannot change the document states.
    first: one question, then the document. One forward per question.
    multi: all questions, the document, then an ANSWER line per question.
    One forward per note. The returned spans are the ANSWER lines.
    """
    if kind in ("last", "last_nobos"):
        pieces = [text]
        spans = []
        for body in bodies:
            pieces.append("\n\nQUESTION\n")
            start = sum(len(piece) for piece in pieces)
            pieces.append(body)
            spans.append((start, start + len(body)))
        return "".join(pieces), 0, spans
    if kind == "first":
        if len(bodies) != 1:
            raise ValueError("question-first prompts hold one question")
        head = "QUESTION\n"
        body = bodies[0]
        lead = f"{head}{body}\n\nDOCUMENT\n"
        return lead + text, len(lead), [(len(head), len(head) + len(body))]
    if kind == "multi":
        # Every question before the note, then one ANSWER line per field after
        # it: one forward per note, and each ANSWER line is that field's query.
        lead = "QUESTIONS\n" + "\n".join(bodies) + "\n\nDOCUMENT\n"
        pieces = [lead, text]
        spans = []
        for body in bodies:
            pieces.append("\n\nANSWER ")
            start = sum(len(piece) for piece in pieces)
            pieces.append(body)
            spans.append((start, start + len(body)))
        return "".join(pieces), len(lead), spans
    raise ValueError(kind)


def doc_token_range(offsets: list[tuple[int, int]], doc_start: int, doc_end: int) -> list[int]:
    """Token indexes whose characters sit inside the document."""
    return [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start >= doc_start - 1 and end <= doc_end and end > doc_start
    ]


def last_token_inside(offsets: list[tuple[int, int]], start: int, end: int) -> int:
    chosen = None
    for index, (token_start, token_end) in enumerate(offsets):
        if token_end > token_start and token_start >= start and token_end <= end:
            chosen = index
    if chosen is None:
        raise ValueError(f"no token inside {start}:{end}")
    return chosen


def span_to_tokens(doc_offsets: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    """Covering token positions (inclusive) for a document character span.

    Offsets here are already in document coordinates. Gemma often glues the
    leading space onto the first piece, so a span can start mid-token.
    """
    first = last = None
    for position, (token_start, token_end) in enumerate(doc_offsets):
        if token_end <= start or token_start >= end:
            continue
        if first is None:
            first = position
        last = position
    if first is None:
        raise ValueError(f"span {start}:{end} covers no token")
    return first, last


def bio_tags(n_tokens: int, token_spans: list[tuple[int, int]]) -> list[int]:
    """0 = outside, 1 = begin, 2 = inside."""
    tags = [0] * n_tokens
    for first, last in token_spans:
        tags[first] = 1
        for position in range(first + 1, last + 1):
            tags[position] = 2
    return tags


def decode_bio(tags: list[int]) -> list[tuple[int, int]]:
    """Spans from a tag sequence. An inside tag after outside opens a span."""
    spans = []
    start = None
    for position, tag in enumerate(tags):
        if tag == 1 or (tag == 2 and start is None):
            if start is not None:
                spans.append((start, position - 1))
            start = position
        elif tag == 0 and start is not None:
            spans.append((start, position - 1))
            start = None
    if start is not None:
        spans.append((start, len(tags) - 1))
    return spans


def bucket(predicted: list[tuple[int, int]], gold: list[tuple[int, int]]) -> str:
    """One label per field, for the error table."""
    if sorted(predicted) == sorted(gold):
        return "exact"
    if not gold:
        return "false_alarm"
    if not predicted:
        return "missed"

    def overlaps(a, b):
        return not (a[1] < b[0] or b[1] < a[0])

    if any(not any(overlaps(p, g) for g in gold) for p in predicted):
        return "wrong_place"
    return "partial"
