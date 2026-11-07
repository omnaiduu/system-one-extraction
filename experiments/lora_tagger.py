"""Evidence tagging with LoRA adapters inside Gemma, against frozen baselines.

The earlier run kept Gemma frozen and trained only a small tagger. Every
head built on frozen states reached a similar level, so this run lets Gemma
learn a little: low-rank adapters (LoRA) on the attention projections of the
text layers, trained together with the same `Tagger` head. The question is
placed before the note, so the adapted Gemma can mark evidence while it reads.

Same split for every condition:
    fit       training notes outside fold 0
    held      training notes in fold 0 (picks the epoch and the decode bias)
    exam1-3   the dev scoreboard, scored once with those choices
Exam 4 is not loaded.

The decode bias is added to the begin and inside logits before argmax. The
frozen question-first tagger missed many spans at bias 0; the bias is the
cut-off for "is there a span here", chosen on `held`.

Everything here is independent of Gemma: `forward(ids)` returns the tuple of
hidden states, so the tests run it with a tiny fake model.
"""

from __future__ import annotations

import random

import torch
from torch.nn import functional as F

from experiments.evidence_data import (
    bio_tags,
    bucket,
    build_prompt,
    decode_bio,
    doc_token_range,
    last_token_inside,
    questions,
    span_to_tokens,
)
from experiments.evidence_heads import TAG_WEIGHTS, Tagger, _batch, folds_for, score, train_tagger
from experiments.values import ALLOWED, BLOCKS, MAX_STEPS, OPS, answer, blocks_for, is_rate, matches, span_texts

SEED = 7
BIASES = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
DEV = ("exam1", "exam2", "exam3")


# ---- examples ---------------------------------------------------------------


def _field_example(row, question, ids, offsets, doc_index, doc_start, q_span, group) -> dict:
    doc_offsets = [(offsets[i][0] - doc_start, offsets[i][1] - doc_start) for i in doc_index]
    gold = []
    for start, end in question["spans"]:
        first, last = span_to_tokens(doc_offsets, start, end)
        covered = row["text"][max(doc_offsets[first][0], 0) : doc_offsets[last][1]]
        if row["text"][start:end].strip() not in covered:
            raise RuntimeError(f"{row['id']} {question['field']}: tokens miss the gold words")
        gold.append((first, last))
    q_index = last_token_inside(offsets, *q_span)
    return {
        **{k: question[k] for k in ("doc", "field", "type", "op")},
        "split": row["split"],
        "group": group,
        "ids": ids,
        "doc_index": doc_index,
        "q_index": q_index,
        # Question first, the last prompt token has read the note. In the
        # one-pass layout the field's own ANSWER line has read everything.
        "end_index": len(ids) - 1 if group != row["id"] else q_index,
        "tags": torch.tensor(bio_tags(len(doc_index), gold), dtype=torch.long),
        "gold": gold,
        "op_index": OPS.index(question["op"]) if question["op"] in OPS else -1,
        # Chains: the gold block of each step (gold parts in text order).
        "blocks": [BLOCKS.index(b) for b in row["fields"][question["field"]].get("blocks") or []],
        "text": row["text"],
        "doc_offsets": doc_offsets,
        "spec": row["fields"][question["field"]],
        "span_targets": span_targets(row, row["fields"][question["field"]]),
    }


def span_targets(row: dict, field: dict) -> list:
    """For each gold span (text order): ("number", Decimal), ("date", date), or None.

    Augmented rows carry the values (their spans may be words); original rows
    are read from the digits.
    """
    from datetime import date as _date
    from decimal import Decimal as _Decimal

    from experiments.augment import span_kinds, span_value
    from experiments.evidence_data import field_evidence

    spans = field_evidence(field) or []
    kinds = field.get("span_kinds") or span_kinds(field)
    stored = field.get("span_values")
    out = []
    for index, ((a, b), kind) in enumerate(zip(spans, kinds)):
        if stored is not None:
            raw = stored[index]
            if raw is None or kind is None:
                out.append(None)
                continue
            value = _date.fromisoformat(raw) if "-" in raw and kind != "number" else _Decimal(raw)
        else:
            value = span_value(row["text"][a:b], kind, field.get("dayfirst", True))
        if value is None:
            out.append(None)
        else:
            out.append(("date", value) if isinstance(value, _date) else ("number", value))
    return out


def encode(rows: list[dict], tokenizer, prompt: str) -> list[dict]:
    """One example per (note, field). Examples with the same `group` share one forward."""
    add_bos = prompt != "last_nobos"
    examples = []
    for row in rows:
        qs = questions(row)
        if not qs:
            continue
        if prompt == "multi":
            full, doc_start, q_spans = build_prompt("multi", row["text"], [q["body"] for q in qs])
            encoded = tokenizer(full, return_offsets_mapping=True, add_special_tokens=add_bos)
            offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
            ids = list(encoded["input_ids"])
            doc_index = doc_token_range(offsets, doc_start, doc_start + len(row["text"]))
            for question, q_span in zip(qs, q_spans):
                examples.append(_field_example(row, question, ids, offsets, doc_index, doc_start, q_span, row["id"]))
            continue
        for question in qs:
            full, doc_start, q_spans = build_prompt(prompt, row["text"], [question["body"]])
            encoded = tokenizer(full, return_offsets_mapping=True, add_special_tokens=add_bos)
            offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
            doc_index = doc_token_range(offsets, doc_start, doc_start + len(row["text"]))
            examples.append(
                _field_example(row, question, list(encoded["input_ids"]), offsets, doc_index, doc_start,
                               q_spans[0], f"{row['id']}::{question['field']}")
            )
    return examples


def groups_of(examples: list[dict]) -> list[list[int]]:
    """Indexes of examples that share one forward, in first-seen order."""
    order: dict[str, list[int]] = {}
    for index, ex in enumerate(examples):
        order.setdefault(ex.get("group", f"{ex['doc']}::{ex['field']}"), []).append(index)
    return list(order.values())


def assign_parts(examples: list[dict]) -> dict[str, list[dict]]:
    """fit / held by the source note (a note and its variants stay together); dev parts by split."""
    from experiments.augment import base_id

    train_docs = [base_id(ex["doc"]) for ex in examples if ex["split"] == "train"]
    fold_of = folds_for(train_docs)
    parts = {"fit": [], "held": [], **{name: [] for name in DEV}}
    for ex in examples:
        if ex["split"] == "train":
            parts["held" if fold_of[base_id(ex["doc"])] == 0 else "fit"].append(ex)
        else:
            parts.setdefault(ex["split"], []).append(ex)
    return parts


# ---- decoding ---------------------------------------------------------------


def decode(logits: torch.Tensor, bias: float) -> list[tuple[int, int]]:
    shifted = logits.float().clone()
    shifted[:, 1:] += bias
    return decode_bio(shifted.argmax(-1).tolist())


def score_logits(examples: list[dict], logits: list[torch.Tensor], bias: float) -> dict:
    return score(examples, [decode(l, bias) for l in logits])


def pick_bias(examples: list[dict], logits: list[torch.Tensor]) -> tuple[float, dict]:
    table = {b: score_logits(examples, logits, b)["exact"] for b in BIASES}
    best = max(BIASES, key=lambda b: (table[b], -abs(b)))
    return best, table


def report_part(examples, logits, bias, keep_examples: int = 0) -> dict:
    predicted = [decode(l, bias) for l in logits]
    out = score(examples, predicted)
    if keep_examples:
        out["examples"] = [
            {"doc": ex["doc"], "field": ex["field"], "op": ex["op"], "gold": ex["gold"],
             "predicted": spans, "bucket": bucket(spans, ex["gold"])}
            for ex, spans in list(zip(examples, predicted))[:keep_examples]
        ]
    return out


# ---- frozen baseline --------------------------------------------------------


def frozen_states(examples: list[dict], forward, layers: list[int]) -> tuple[list[dict], dict]:
    """Run the frozen model once per example and keep doc and question states."""
    items, states = [], {}
    with torch.no_grad():
        for number, ex in enumerate(examples):
            hidden = forward(ex["ids"])
            stacked = torch.stack([hidden[i][0] for i in layers]).float()
            key = f"{ex['doc']}::{ex['field']}"
            states[key] = stacked[:, ex["doc_index"]].to(torch.float16).cpu()
            items.append({**ex, "states_key": key, "q_state": stacked[:, ex["q_index"]].to(torch.float16).cpu()})
    return items, states


def tagger_logits(model: Tagger, items: list[dict], states: dict, device: str, batch: int = 16) -> list[torch.Tensor]:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(items), batch):
            chunk = items[start : start + batch]
            doc, question, _ = _batch(chunk, states, device)
            logits = model(doc, question).float().cpu()
            for row, item in enumerate(chunk):
                out.append(logits[row, : states[item["states_key"]].shape[1]])
    return out


def run_frozen(parts: dict, forward, layers: list[int], device: str, epochs: int = 60, every: int = 5, log=print) -> dict:
    needed = parts["fit"] + parts["held"] + [ex for name in DEV for ex in parts[name]]
    items, states = frozen_states(needed, forward, layers)
    by_key = {f"{it['doc']}::{it['field']}": it for it in items}

    def pick(name):
        return [by_key[f"{ex['doc']}::{ex['field']}"] for ex in parts[name]]

    fit, held = pick("fit"), pick("held")
    history, best = [], None
    for epoch, model in train_tagger(fit, states, device, epochs, tuple(range(every, epochs + 1, every))):
        logits = tagger_logits(model, held, states, device)
        bias, table = pick_bias(held, logits)
        history.append({"epoch": epoch, "held_exact_at_0": table[0.0], "held_exact_best_bias": table[bias], "bias": bias})
        log(f"  frozen epoch {epoch}: held {table[0.0]} at bias 0, {table[bias]} at bias {bias}")
        if best is None or table[bias] > best["held"]:
            best = {"epoch": epoch, "bias": bias, "held": table[bias],
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    final = Tagger(states[fit[0]["states_key"]].shape[0], states[fit[0]["states_key"]].shape[2]).to(device)
    final.load_state_dict(best["state"])
    result = {"epoch": best["epoch"], "bias": best["bias"], "history": history}
    result["held"] = report_part(held, tagger_logits(final, held, states, device), best["bias"])
    result["fit_sample"] = report_part(fit[:300], tagger_logits(final, fit[:300], states, device), best["bias"])
    for name in DEV:
        part = pick(name)
        result[name] = report_part(part, tagger_logits(final, part, states, device), best["bias"], keep_examples=200)
    return result


# ---- operation head and final answers ----------------------------------------


class OpHead(torch.nn.Module):
    """Which operation turns the marked parts into the answer.

    Reads the last question token and the last prompt token (which, question
    first, has read the whole note).
    """

    def __init__(self, layers: int, width: int, inner: int = 128, outputs: int | None = None) -> None:
        super().__init__()
        self.mix = torch.nn.Parameter(torch.zeros(layers))
        self.norm = torch.nn.LayerNorm(2 * width)
        # `outputs`: older checkpoints were trained before "average" and "chain" existed.
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 * width, inner), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(inner, outputs or len(OPS)),
        )

    def forward(self, question: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.mix, 0)
        q = torch.einsum("k,bkh->bh", weights, F.layer_norm(question, question.shape[-1:]))
        e = torch.einsum("k,bkh->bh", weights, F.layer_norm(end, end.shape[-1:]))
        return self.net(self.norm(torch.cat([q, e], dim=-1)))


class StepHead(torch.nn.Module):
    """Which block each step of a chain uses.

    Step k combines the running value with marked part k+1. It reads the
    field's ANSWER token (which has read the note and the question), and the
    marked parts k and k+1 together with a few tokens after each: attention is
    causal, so the words that say what to do with a number ("15% *off*") come
    after it, and only later tokens have seen them.
    """

    def __init__(self, layers: int, width: int, inner: int = 128, window: int = 4,
                 pair_context: bool = True, step_embedding: bool = True) -> None:
        """`pair_context` and `step_embedding` are experiment 6's design. With them
        the head learned whole pairs; experiment 7 turns both off, so each step
        sees only the answer token and its own operand's surroundings."""
        super().__init__()
        parts = 3 if pair_context else 2
        self.config = {"inner": inner, "window": window, "pair_context": pair_context, "step_embedding": step_embedding}
        self.mix = torch.nn.Parameter(torch.zeros(layers))
        self.norm = torch.nn.LayerNorm(parts * width)
        self.step = torch.nn.Embedding(MAX_STEPS, parts * width) if step_embedding else None
        self.net = torch.nn.Sequential(
            torch.nn.Linear(parts * width, inner), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(inner, len(BLOCKS))
        )
        self.window = window
        self.pair_context = pair_context

    def forward(self, doc: torch.Tensor, end: torch.Tensor, spans) -> torch.Tensor | None:
        """doc [K, T, H], end [K, H], spans in text order -> [steps, len(BLOCKS)], or None."""
        steps = len(spans) - 1
        if steps < 1 or steps > MAX_STEPS:
            return None
        weights = torch.softmax(self.mix, 0)
        d = torch.einsum("k,kth->th", weights, F.layer_norm(doc, doc.shape[-1:]))
        e = torch.einsum("k,kh->h", weights, F.layer_norm(end, end.shape[-1:]))

        def around(span):
            a, b = span
            return d[a : min(b + 1 + self.window, d.shape[0])].mean(0)

        if self.pair_context:
            rows = [torch.cat([e, around(spans[k]), around(spans[k + 1])]) for k in range(steps)]
        else:
            rows = [torch.cat([e, around(spans[k + 1])]) for k in range(steps)]
        x = torch.stack(rows)
        if self.step is not None:
            x = x + self.step(torch.arange(steps, device=d.device))
        return self.net(self.norm(x))


def choose_blocks(ex: dict, step_logits: torch.Tensor | None) -> list[str] | None:
    if step_logits is None:
        return None
    allowed = blocks_for(ex["type"])
    return [max(allowed, key=lambda b: float(row[BLOCKS.index(b)])) for row in step_logits]


def chain_extras(step_head, ex: dict, doc: torch.Tensor, end: torch.Tensor, tag_logits: torch.Tensor) -> dict:
    """Step logits for every span set any candidate bias could decode."""
    out = {}
    if step_head is None or ex["type"] not in ("number", "date"):
        return out
    for bias in BIASES:
        raw = decode(tag_logits, bias)
        for spans in (tuple(raw), tuple(snap_spans(ex, raw))):  # as decoded, and snapped
            if spans not in out:
                logits = step_head(doc, end, list(spans))
                out[spans] = None if logits is None else logits.float().cpu()
    return out


# ---- learned reader and emit heads ---------------------------------------------

YEAR0, YEARS = 2010, 30
NUMBER_OUTPUTS = [2] + [10] * 7 + [10] * 2  # sign, 7 whole digits, 2 decimals
DATE_OUTPUTS = [YEARS, 12, 31]  # year from 2010, month, day


class SpanReader(torch.nn.Module):
    """Read a value off marked tokens, keeping their order.

    One learned query per output slot (each digit, or year / month / day)
    attends over the marked tokens' states, plus the question. This is the
    number and calendar heads again, but they read the marked words, not a
    blend of the whole note.
    """

    def __init__(self, layers: int, width: int, outputs: list[int], inner: int = 256, max_len: int = 48) -> None:
        super().__init__()
        self.mix = torch.nn.Parameter(torch.zeros(layers))
        self.norm = torch.nn.LayerNorm(width)
        self.proj = torch.nn.Linear(width, inner)
        self.q_proj = torch.nn.Linear(width, inner)
        self.pos = torch.nn.Embedding(max_len, inner)
        self.queries = torch.nn.Parameter(torch.randn(len(outputs), inner) * 0.02)
        self.attn = torch.nn.MultiheadAttention(inner, 4, batch_first=True)
        self.mlp = torch.nn.Sequential(torch.nn.LayerNorm(inner), torch.nn.Linear(inner, inner), torch.nn.GELU())
        self.heads = torch.nn.ModuleList(torch.nn.Linear(inner, n) for n in outputs)
        self.max_len = max_len

    def forward(self, tokens: torch.Tensor, question: torch.Tensor) -> list[torch.Tensor]:
        """tokens [K, L, H], question [K, H] -> one logit vector per output slot."""
        weights = torch.softmax(self.mix, 0)
        tokens = tokens[:, : self.max_len]
        t = torch.einsum("k,klh->lh", weights, F.layer_norm(tokens, tokens.shape[-1:]))
        t = self.proj(self.norm(t)) + self.pos(torch.arange(t.shape[0], device=t.device))
        q = self.q_proj(self.norm(torch.einsum("k,kh->h", weights, F.layer_norm(question, question.shape[-1:]))))
        queries = (self.queries + q).unsqueeze(0)
        out, _ = self.attn(queries, t.unsqueeze(0), t.unsqueeze(0))
        hidden = self.mlp(out + queries)[0]
        return [head(hidden[i]) for i, head in enumerate(self.heads)]


def make_readers(layers: int, width: int) -> dict:
    return {
        "number": SpanReader(layers, width, NUMBER_OUTPUTS),
        "date": SpanReader(layers, width, DATE_OUTPUTS),
        "emit_number": SpanReader(layers, width, NUMBER_OUTPUTS),
        "emit_date": SpanReader(layers, width, DATE_OUTPUTS),
    }


def number_target(value) -> list[int] | None:
    from extractor.labels import split_number

    try:
        encoded = split_number(float(value), 2)
    except ValueError:
        return None
    return [encoded["sign"]] + encoded["whole"] + encoded["frac"]


def date_target(value) -> list[int] | None:
    if not YEAR0 <= value.year < YEAR0 + YEARS:
        return None
    return [value.year - YEAR0, value.month - 1, value.day - 1]


def reader_loss(reader: SpanReader, tokens, question, target: list[int]) -> torch.Tensor:
    logits = reader(tokens, question)
    device = logits[0].device
    return sum(F.cross_entropy(l.float().unsqueeze(0), torch.tensor([t], device=device)) for l, t in zip(logits, target))


def read_number(reader: SpanReader, tokens, question):
    from decimal import Decimal as _Decimal

    choice = [int(l.argmax()) for l in reader(tokens, question)]
    sign, whole, frac = choice[0], choice[1:8], choice[8:10]
    text = ("".join(map(str, whole)).lstrip("0") or "0") + "." + "".join(map(str, frac))
    value = _Decimal(text)
    return -value if sign == 1 else value


def read_date(reader: SpanReader, tokens, question):
    from datetime import date as _date

    year, month, day = (int(l.argmax()) for l in reader(tokens, question))
    try:
        return _date(YEAR0 + year, month + 1, day + 1)
    except ValueError:
        return None


def _span_tokens(doc: torch.Tensor, spans) -> torch.Tensor:
    """doc [K, T, H]; the marked tokens of all spans, in order -> [K, L, H]."""
    return torch.cat([doc[:, a : b + 1] for a, b in spans], dim=1)


def reader_extras(readers: dict, ex: dict, doc: torch.Tensor, question: torch.Tensor, tag_logits: torch.Tensor) -> dict:
    """Reader and emit outputs for every span set any candidate bias could decode."""
    extra = {"number": {}, "date": {}, "emit": {}}
    if readers is None or ex["type"] not in ("number", "date", "duration"):
        return extra
    for bias in BIASES:
        spans = decode(tag_logits, bias)
        if not spans:
            continue
        for span in spans:
            if span not in extra["number"]:
                tokens = _span_tokens(doc, [span])
                extra["number"][span] = read_number(readers["number"], tokens, question)
                extra["date"][span] = read_date(readers["date"], tokens, question)
        key = tuple(spans)
        if key not in extra["emit"]:
            tokens = _span_tokens(doc, spans)
            if ex["type"] == "date":
                d = read_date(readers["emit_date"], tokens, question)
                extra["emit"][key] = d.isoformat() if d is not None else None
            else:
                extra["emit"][key] = float(read_number(readers["emit_number"], tokens, question))
    return extra


def choose_op(ex: dict, op_logits: torch.Tensor | None) -> str:
    allowed = ALLOWED[ex["type"]]
    if op_logits is not None:
        allowed = tuple(op for op in allowed if OPS.index(op) < len(op_logits))
    if len(allowed) == 1 or op_logits is None:
        return allowed[0]
    return max(allowed, key=lambda op: float(op_logits[OPS.index(op)]))


SNAP_TYPES = ("number", "date", "duration")


def snap_spans(ex: dict, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Token spans widened to whole numbers and dates (`values.snap_char_spans`)."""
    from experiments.values import snap_char_spans

    if not spans or ex["type"] not in SNAP_TYPES:
        return spans
    offsets = ex["doc_offsets"]
    chars = [(max(offsets[a][0], 0), offsets[b][1]) for a, b in spans]
    out = []
    for start, end in snap_char_spans(ex["text"], chars, ex["type"]):
        token_span = span_to_tokens(offsets, start, end)
        if token_span not in out:
            out.append(token_span)
    return out


def final_answer(ex: dict, tag_logits: torch.Tensor, op_logits, bias: float, extra: dict | None = None,
                 snap: bool = False, rewriter=None):
    """`snap`: widen marks to whole numbers and dates. `rewriter(text, kind)`: reads a part
    the rules cannot (experiment 7: plain Gemma plus a backwards check)."""
    spans = decode(tag_logits, bias)
    if snap:
        spans = snap_spans(ex, spans)
    op = choose_op(ex, op_logits)
    read = None
    emitted = None
    blocks = rates = None
    if op == "chain" and extra is not None:
        blocks = choose_blocks(ex, extra.get("chain", {}).get(tuple(spans)))
        parts = span_texts(ex["text"], ex["doc_offsets"], spans)
        rates = [is_rate(ex["text"], ex["doc_offsets"][b][1], part) for (a, b), part in zip(spans, parts)]
    if extra is not None and extra.get("number"):
        def read(index, kind):
            span = spans[index]
            if kind == "date":
                return extra["date"].get(span)
            value = extra["number"].get(span)
            if kind == "count" and value is not None:
                return int(value)
            return value

        emitted = extra.get("emit", {}).get(tuple(spans))
    parts_text = span_texts(ex["text"], ex["doc_offsets"], spans)
    if rewriter is not None:
        def read(index, kind):
            return rewriter(parts_text[index], kind)
    predicted = answer(op, parts_text, ex["spec"], read=read, emitted=emitted, blocks=blocks, rates=rates)
    if blocks is not None:
        predicted["blocks"] = blocks
    return spans, op, predicted


def field_confidence(ex: dict, tag_logits: torch.Tensor, op_logits, bias: float, spans, op: str, extra=None) -> dict:
    """p_null and a confidence for the answer, from the same logits that made it.

    p_null: one minus the highest probability that any token is evidence.
    confidence: for an answer with spans, the geometric mean of the
    probability of each decoded tag over the marked tokens and the one token
    either side (which should be outside), times the probability of the
    chosen operation among the legal ones. For a missing answer, p_null.
    """
    shifted = tag_logits.float().clone()
    shifted[:, 1:] += bias
    probs = torch.softmax(shifted, -1)
    p_null = float(1.0 - (1.0 - probs[:, 0]).max())
    if not spans:
        return {"p_null": p_null, "confidence": p_null}
    tags = shifted.argmax(-1)
    picked = set()
    for a, b in spans:
        picked.update(range(max(a - 1, 0), min(b + 2, len(tags))))
    logs = [float(torch.log(probs[t, int(tags[t])] + 1e-12)) for t in sorted(picked)]
    tag_conf = float(torch.exp(torch.tensor(sum(logs) / len(logs))))
    allowed = ALLOWED[ex["type"]]
    op_conf = 1.0
    if op_logits is not None and len(allowed) > 1:
        legal = torch.softmax(torch.stack([op_logits[OPS.index(o)] for o in allowed]).float(), 0)
        op_conf = float(legal[allowed.index(op)])
    step_conf = 1.0
    if op == "chain":
        steps = (extra or {}).get("chain", {}).get(tuple(spans))
        if steps is None:
            step_conf = 0.0
        else:
            legal = [BLOCKS.index(b) for b in blocks_for(ex["type"])]
            for row in steps:
                step_conf *= float(torch.softmax(row[legal].float(), 0).max())
    return {"p_null": p_null, "confidence": tag_conf * op_conf * step_conf}


def selective(confidences: list[float], right: list[bool]) -> dict:
    """Accuracy above each confidence threshold, and how well confidence ranks right above wrong."""
    rows = {}
    for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        kept = [r for c, r in zip(confidences, right) if c >= t]
        rows[str(t)] = {"kept": len(kept), "right": sum(kept)}
    pos = [c for c, r in zip(confidences, right) if r]
    neg = [c for c, r in zip(confidences, right) if not r]
    if pos and neg:
        wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
        auroc = wins / (len(pos) * len(neg))
    else:
        auroc = None
    return {"thresholds": rows, "auroc": auroc, "fields": len(right)}


def value_report(examples, tag_logits, op_logits, bias, keep_examples: int = 0, extras=None,
                 snap: bool = False, rewriter=None) -> dict:
    from collections import Counter, defaultdict

    hits, by_op = Counter(), defaultdict(Counter)
    by_kind, by_steps = defaultdict(Counter), defaultdict(Counter)
    blocks_right = blocks_total = 0
    op_right = op_total = 0
    emit_close = emit_total = 0
    kept = []
    confidences, rights = [], []
    for index, ex in enumerate(examples):
        extra = extras[index] if extras is not None else None
        spans, op, predicted = final_answer(ex, tag_logits[index], op_logits[index], bias, extra, snap, rewriter)
        ok = matches(ex["spec"], predicted)
        if ex["op"] == "emit" and ex["type"] == "number":
            emit_total += 1
            gold, got = ex["spec"]["value"], predicted["value"]
            if isinstance(got, (int, float)) and gold and abs(got - gold) <= 0.01 * abs(gold):
                emit_close += 1
        conf = field_confidence(ex, tag_logits[index], op_logits[index], bias, spans, op, extra)
        if ex["spec"].get("chain_kind"):
            by_kind[ex["spec"]["chain_kind"]][ok] += 1
            by_steps[str(ex["spec"].get("steps", 1))][ok] += 1
        if ex["op"] == "chain" and list(spans) == list(ex["gold"]):
            # The step head alone: given the right marks, were the blocks right?
            blocks_total += 1
            steps = (extra or {}).get("chain", {}).get(tuple(spans))
            chosen = choose_blocks(ex, steps)
            blocks_right += chosen is not None and [BLOCKS.index(b) for b in chosen] == ex["blocks"]
        confidences.append(conf["confidence"])
        rights.append(bool(ok))
        hits[ok] += 1
        by_op[ex["op"]][ok] += 1
        if ex["op"] in OPS and len(ALLOWED[ex["type"]]) > 1:
            op_total += 1
            op_right += op == ex["op"]
        if len(kept) < keep_examples:
            kept.append({"doc": ex["doc"], "field": ex["field"], "gold_op": ex["op"], "op": op,
                         "gold": ex["spec"].get("value"), "predicted": predicted["value"],
                         **({"kind": ex["spec"]["chain_kind"], "gold_blocks": ex["spec"].get("blocks"),
                             "blocks": predicted.get("blocks")} if ex["spec"].get("chain_kind") else {}),
                         "missing": predicted["missing"], "right": ok,
                         "confidence": round(conf["confidence"], 6), "p_null": round(conf["p_null"], 6)})
    out = {
        "fields": len(examples),
        "exact": hits[True],
        "by_op": {op: f"{c[True]}/{c[True] + c[False]}" for op, c in sorted(by_op.items())},
        "op_choice": f"{op_right}/{op_total}",
        "emit_number_within_1pct": f"{emit_close}/{emit_total}",
        "by_kind": {k: f"{c[True]}/{c[True] + c[False]}" for k, c in sorted(by_kind.items())},
        "by_steps": {k: f"{c[True]}/{c[True] + c[False]}" for k, c in sorted(by_steps.items())},
        "blocks_given_right_marks": f"{blocks_right}/{blocks_total}",
        "confidence": selective(confidences, rights),
    }
    if keep_examples:
        out["examples"] = kept
    return out


def pick_bias_by_value(examples, tag_logits, op_logits, extras=None, snap: bool = False) -> tuple[float, dict]:
    table = {}
    for b in BIASES:
        table[b] = (
            value_report(examples, tag_logits, op_logits, b, extras=extras, snap=snap)["exact"],
            score_logits(examples, tag_logits, b)["exact"],
        )
    best = max(BIASES, key=lambda b: (table[b], -abs(b)))
    return best, table


# ---- LoRA -------------------------------------------------------------------


def _stack(forward, ids, layers):
    hidden = forward(ids)
    return torch.stack([hidden[i][0] for i in layers]).float()


def _field_states(stacked, ex: dict):
    return (
        stacked[:, ex["doc_index"]].unsqueeze(0),
        stacked[:, ex["q_index"]].unsqueeze(0),
        stacked[:, ex["end_index"]].unsqueeze(0),
    )


def _example_states(forward, ex: dict, layers: list[int]):
    return _field_states(_stack(forward, ex["ids"], layers), ex)


def lora_logits(forward, head: Tagger, op_head, examples: list[dict], layers: list[int], autocast, readers=None,
                step_head=None):
    """Tag logits, operation logits, and reader/emit/step outputs for each example."""
    head.eval()
    if op_head is not None:
        op_head.eval()
    if step_head is not None:
        step_head.eval()
    for reader in (readers or {}).values():
        reader.eval()
    tags, ops, extras = [None] * len(examples), [None] * len(examples), [None] * len(examples)
    with torch.no_grad():
        for group in groups_of(examples):
            with autocast():
                stacked = _stack(forward, examples[group[0]]["ids"], layers)
                for index in group:
                    doc, question, end = _field_states(stacked, examples[index])
                    logits = head(doc, question)
                    op_logits = op_head(question, end) if op_head is not None else None
                    tags[index] = logits[0].float().cpu()
                    ops[index] = None if op_logits is None else op_logits[0].float().cpu()
                    if readers is not None or step_head is not None:
                        extras[index] = reader_extras(readers, examples[index], doc[0], end[0], tags[index])
                        extras[index]["chain"] = chain_extras(step_head, examples[index], doc[0], end[0], tags[index])
    return tags, ops, (extras if readers is not None or step_head is not None else None)


def _reader_losses(readers: dict, ex: dict, doc: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
    """Teach the reader on every gold span with a known value, and the emit head on emit fields."""
    total = doc.new_zeros(())
    for (a, b), target in zip(ex["gold"], ex.get("span_targets") or []):
        if target is None:
            continue
        kind, value = target
        tokens = doc[:, a : b + 1]
        encoded = date_target(value) if kind == "date" else number_target(value)
        if encoded is not None:
            total = total + reader_loss(readers["date" if kind == "date" else "number"], tokens, question, encoded)
    if ex["op"] == "emit" and ex["gold"]:
        from datetime import date as _date

        tokens = _span_tokens(doc, ex["gold"])
        value = ex["spec"]["value"]
        if ex["type"] == "date":
            encoded = date_target(_date.fromisoformat(value))
            if encoded is not None:
                total = total + reader_loss(readers["emit_date"], tokens, question, encoded)
        else:
            encoded = number_target(value)
            if encoded is not None:
                total = total + reader_loss(readers["emit_number"], tokens, question, encoded)
    return total


def train_lora(
    parts: dict,
    forward,
    adapter_params: list[torch.nn.Parameter],
    head: Tagger,
    layers: list[int],
    device: str,
    epochs: int = 5,
    accumulate: int = 8,
    lr_adapter: float = 2e-4,
    lr_head: float = 1e-3,
    use_amp: bool = True,
    set_train=lambda flag: None,
    op_head: OpHead | None = None,
    readers: dict | None = None,
    step_head: StepHead | None = None,
    snap: bool = False,
    rewriter=None,
    log=print,
) -> dict:
    """Train adapters, tagger and operation head together.

    The epoch and the decode bias are the ones with the most right final
    answers on `held` (ties: most exact evidence).
    """
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    heads = [head] + ([op_head] if op_head is not None else []) + list((readers or {}).values())
    if step_head is not None:
        heads.append(step_head)  # last, so readers keep their slots in best["heads"]
    head_params = [p for h in heads for p in h.parameters()]
    optimizer = torch.optim.AdamW(
        [{"params": adapter_params, "lr": lr_adapter}, {"params": head_params, "lr": lr_head}],
        weight_decay=0.01,
    )
    amp = use_amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def autocast():
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp)

    weights = torch.tensor(TAG_WEIGHTS, device=device)
    fit = list(parts["fit"])
    fit_groups = groups_of(fit)
    history, best = [], None
    for epoch in range(1, epochs + 1):
        set_train(True)
        for h in heads:
            h.train()
        rng.shuffle(fit_groups)
        total, skipped, seen, pending = 0.0, 0, 0, 0
        optimizer.zero_grad()
        for number, group in enumerate(fit_groups, start=1):
            # One forward per group (a note in the one-pass layout, a field otherwise).
            with autocast():
                stacked = _stack(forward, fit[group[0]]["ids"], layers)
                losses = []
                for index in group:
                    ex = fit[index]
                    doc, question, end = _field_states(stacked, ex)
                    logits = head(doc, question)
                    loss = F.cross_entropy(logits[0].float(), ex["tags"].to(device), weight=weights)
                    if op_head is not None and ex.get("op_index", -1) >= 0:
                        op_logits = op_head(question, end)
                        loss = loss + F.cross_entropy(op_logits.float(), torch.tensor([ex["op_index"]], device=device))
                    if readers is not None:
                        # Detached: the readers learn from Gemma's states but do not
                        # push back into the adapters. Joint training let the reader
                        # losses (about 100x the tagger's) take the adapters over.
                        loss = loss + _reader_losses(readers, ex, doc[0].detach(), end[0].detach())
                    if step_head is not None and ex.get("blocks") and len(ex["blocks"]) == len(ex["gold"]) - 1:
                        # Teacher forcing: the gold marks, the gold block of each step.
                        # Not detached: like the operation head, a small loss that
                        # may shape the adapters.
                        step_logits = step_head(doc[0], end[0], ex["gold"])
                        loss = loss + F.cross_entropy(step_logits.float(), torch.tensor(ex["blocks"], device=device))
                    losses.append(loss)
            loss = torch.stack(losses).sum()
            if not torch.isfinite(loss):
                skipped += len(group)
                continue
            scaler.scale(loss / accumulate).backward()
            total += float(loss.detach())
            seen += len(group)
            pending += len(group)
            if pending >= accumulate or number == len(fit_groups):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(adapter_params + head_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                pending = 0
            if number % 100 == 0:
                log(f"  lora epoch {epoch} forward {number}/{len(fit_groups)} loss {total / max(seen, 1):.4f}")
        set_train(False)
        tag_l, op_l, ex_l = lora_logits(forward, head, op_head, parts["held"], layers, autocast, readers, step_head)
        bias, table = pick_bias_by_value(parts["held"], tag_l, op_l, ex_l, snap=snap)
        row = {"epoch": epoch, "train_loss": total / max(1, seen), "skipped": skipped,
               "bias": bias, "held_value_exact": table[bias][0], "held_span_exact": table[bias][1],
               "held_span_exact_at_0": table[0.0][1]}
        history.append(row)
        log(f"  lora epoch {epoch}: {row}")
        if best is None or table[bias] > best["held"]:
            best = {
                "epoch": epoch, "bias": bias, "held": table[bias],
                "adapter": [p.detach().cpu().clone() for p in adapter_params],
                "heads": [{k: v.detach().cpu().clone() for k, v in h.state_dict().items()} for h in heads],
            }
    with torch.no_grad():
        for param, saved in zip(adapter_params, best["adapter"]):
            param.copy_(saved.to(param.device))
    for h, saved in zip(heads, best["heads"]):
        h.load_state_dict(saved)
    best["head"] = best["heads"][0]
    set_train(False)
    result = {"epoch": best["epoch"], "bias": best["bias"], "history": history, "values": {}}
    if snap or rewriter is not None:
        # The same logits scored without the new steps, so each one's effect is measured alone.
        result["values_plain"] = {}
        if rewriter is not None:
            result["values_no_rewrite"] = {}

    def both(name, examples, keep):
        tag_l, op_l, ex_l = lora_logits(forward, head, op_head, examples, layers, autocast, readers, step_head)
        result[name] = report_part(examples, tag_l, best["bias"], keep_examples=keep)
        result["values"][name] = value_report(examples, tag_l, op_l, best["bias"], keep_examples=keep, extras=ex_l,
                                              snap=snap, rewriter=rewriter)
        if "values_plain" in result:
            result["values_plain"][name] = value_report(examples, tag_l, op_l, best["bias"], extras=ex_l)
        if "values_no_rewrite" in result:
            result["values_no_rewrite"][name] = value_report(examples, tag_l, op_l, best["bias"], extras=ex_l, snap=snap)

    both("held", parts["held"], 0)
    both("fit_sample", parts["fit"][:300], 0)
    for name in list(DEV) + sorted(k for k in parts if k not in DEV and k not in ("fit", "held")):
        if parts.get(name):
            both(name, parts[name], 200)
    result["_best"] = best
    return result


def summarize(payload: dict) -> str:
    lines = [
        "# LoRA evidence tagger",
        "",
        f"Model `{payload.get('model')}`, {payload.get('dtype')}, layers {payload.get('layers')}, "
        f"LoRA on {payload.get('lora', {}).get('modules')} modules, rank {payload.get('lora', {}).get('rank')}. "
        f"Run time {payload.get('seconds')} s.",
        "",
        "Same split for every row. `held` (fold 0 of the training notes) picked the epoch and the decode bias.",
        "Exams 1-3 are the scoreboard, scored once. Exam 4 is not loaded. `fit sample` is 300 training fields.",
        "",
        "| Condition | Epoch | Bias | Fit sample | Held | Exam 1 | Exam 2 | Exam 3 | Exams 1-3 | Wrong place | Missed | Partial |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, res in payload["results"].items():
        dev_exact = sum(res[d]["exact"] for d in DEV)
        dev_total = sum(res[d]["fields"] for d in DEV)
        buckets = {k: sum(res[d]["buckets"].get(k, 0) for d in DEV) for k in ("wrong_place", "missed", "partial")}
        lines.append(
            f"| {name} | {res['epoch']} | {res['bias']} | {res['fit_sample']['exact']}/{res['fit_sample']['fields']} | "
            f"{res['held']['exact']}/{res['held']['fields']} | "
            + " | ".join(f"{res[d]['exact']}/{res[d]['fields']}" for d in DEV)
            + f" | {dev_exact}/{dev_total} | {buckets['wrong_place']} | {buckets['missed']} | {buckets['partial']} |"
        )
    valued = {name: res for name, res in payload["results"].items() if res.get("values")}
    if valued:
        lines += ["", "## Final answers (marked words, read and combined)", ""]
        lines.append("| Condition | Fit sample | Held | Exam 1 | Exam 2 | Exam 3 | Exams 1-3 | Operation choice, exams 1-3 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for name, res in valued.items():
            v = res["values"]
            right = sum(v[d]["exact"] for d in DEV)
            total = sum(v[d]["fields"] for d in DEV)
            op_r = sum(int(v[d]["op_choice"].split("/")[0]) for d in DEV)
            op_t = sum(int(v[d]["op_choice"].split("/")[1]) for d in DEV)
            lines.append(
                f"| {name} | {v['fit_sample']['exact']}/{v['fit_sample']['fields']} | {v['held']['exact']}/{v['held']['fields']} | "
                + " | ".join(f"{v[d]['exact']}/{v[d]['fields']}" for d in DEV)
                + f" | {right}/{total} | {op_r}/{op_t} |"
            )
        extra_parts = sorted({k for res in valued.values() for k in res["values"] if k not in DEV + ("held", "fit_sample")})
        if extra_parts:
            lines += ["", "### Augmented dev notes (exams 1-3 rewritten; emit phrasings never used in training)", ""]
            lines.append("| Condition | Part | Final answers | By operation | Emit numbers within 1% |")
            lines.append("| --- | --- | --- | --- | --- |")
            for name, res in valued.items():
                for part in extra_parts:
                    v = res["values"].get(part)
                    if v:
                        lines.append(f"| {name} | {part} | {v['exact']}/{v['fields']} | {v['by_op']} | {v.get('emit_number_within_1pct', '-')} |")
        chain_parts = [p for p in ["held"] + extra_parts
                       if any(res["values"].get(p, {}).get("by_kind") for res in valued.values())]
        if chain_parts:
            lines += ["", "### Computed fields (answers never printed)", "",
                      "`single`: one step. `combo_seen`: two steps, a combination training has, new wording. "
                      "`combo_unseen`: two steps, a combination training never has. "
                      "Blocks right = the step head's choices, on fields whose marks were exactly right.", ""]
            lines.append("| Condition | Part | Computed fields right | 1 step | 2 steps | Blocks right given right marks |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for name, res in valued.items():
                for part in chain_parts:
                    v = res["values"].get(part)
                    if not v or not v.get("by_kind"):
                        continue
                    right = sum(int(c.split("/")[0]) for c in v["by_kind"].values())
                    total = sum(int(c.split("/")[1]) for c in v["by_kind"].values())
                    lines.append(f"| {name} | {part} | {right}/{total} | {v['by_steps'].get('1', '-')} | "
                                 f"{v['by_steps'].get('2', '-')} | {v['blocks_given_right_marks']} |")
            for name, res in valued.items():
                for part in chain_parts:
                    v = res["values"].get(part)
                    if v and v.get("by_kind"):
                        lines.append(f"- {name}, {part}: " + ", ".join(f"{k} {c}" for k, c in v["by_kind"].items()))
        lines += ["", "Final answers by operation, exams 1-3:", ""]
        for name, res in valued.items():
            merged = {}
            for d in DEV:
                for op, cell in res["values"][d]["by_op"].items():
                    a, b = map(int, cell.split("/"))
                    x, y = merged.get(op, (0, 0))
                    merged[op] = (x + a, y + b)
            lines.append(f"- {name}: " + ", ".join(f"{op} {a}/{b}" for op, (a, b) in sorted(merged.items())))
    ablated = {name: res for name, res in valued.items() if res.get("values_plain")}
    if ablated:
        lines += ["", "## Effect of snapping marks to whole numbers, and of the Gemma rewrite", "",
                  "Same model and logits. Plain = marks as decoded, rules only. Snap = marks widened to whole "
                  "numbers and dates. Full = snap plus plain Gemma rewriting parts the rules cannot read, checked backwards.", ""]
        lines.append("| Condition | Part | Plain | Snap | Full |")
        lines.append("| --- | --- | --- | --- | --- |")
        for name, res in ablated.items():
            for part in res["values_plain"]:
                plain = res["values_plain"][part]
                snapped = res.get("values_no_rewrite", {}).get(part, res["values"][part])
                full = res["values"][part]
                lines.append(f"| {name} | {part} | {plain['exact']}/{plain['fields']} | {snapped['exact']}/{snapped['fields']} | "
                             f"{full['exact']}/{full['fields']} |")
        rw = payload.get("rewrite_stats")
        if rw:
            lines += ["", f"Gemma rewrite: asked {rw.get('asked', 0)}, gave a well-formed value {rw.get('parsed', 0)}, "
                          f"passed the backwards check {rw.get('accepted', 0)}."]
    confident = {name: res for name, res in valued.items() if "confidence" in res["values"].get("exam1", {})}
    if confident:
        lines += ["", "## Confidence, exams 1-3", "",
                  "Kept = answers at or above the threshold; right = how many of those are correct. "
                  "AUROC = how often a right answer has a higher confidence than a wrong one (0.5 is no signal).", ""]
        lines.append("| Condition | AUROC | >=0.5 | >=0.7 | >=0.8 | >=0.9 | >=0.95 | >=0.99 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for name, res in confident.items():
            cells, aucs = [], []
            for t in ("0.5", "0.7", "0.8", "0.9", "0.95", "0.99"):
                kept = sum(res["values"][d]["confidence"]["thresholds"][t]["kept"] for d in DEV)
                right = sum(res["values"][d]["confidence"]["thresholds"][t]["right"] for d in DEV)
                cells.append(f"{right}/{kept}")
            for d in DEV:
                if res["values"][d]["confidence"]["auroc"] is not None:
                    aucs.append(res["values"][d]["confidence"]["auroc"])
            auc = f"{sum(aucs) / len(aucs):.3f}" if aucs else "-"
            lines.append(f"| {name} | {auc} | " + " | ".join(cells) + " |")
    lines += ["", "## Evidence on exams 1-3 by operation (exact / fields)", ""]
    names = list(payload["results"])
    ops = sorted({op for res in payload["results"].values() for d in DEV for op in res[d]["by_op"]})
    lines.append("| Operation | " + " | ".join(names) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in names) + " |")
    for op in ops:
        cells = []
        for res in payload["results"].values():
            exact = sum(res[d]["by_op"].get(op, {}).get("exact", 0) for d in DEV)
            total = sum(sum(res[d]["by_op"].get(op, {}).values()) for d in DEV)
            cells.append(f"{exact}/{total}")
        lines.append(f"| {op} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
