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

SEED = 7
BIASES = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
DEV = ("exam1", "exam2", "exam3")


# ---- examples ---------------------------------------------------------------


def encode(rows: list[dict], tokenizer, prompt: str) -> list[dict]:
    """One example per (note, field): token ids, where the note sits, tags."""
    add_bos = prompt != "last_nobos"
    examples = []
    for row in rows:
        for question in questions(row):
            full, doc_start, q_spans = build_prompt(prompt, row["text"], [question["body"]])
            encoded = tokenizer(full, return_offsets_mapping=True, add_special_tokens=add_bos)
            offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
            doc_index = doc_token_range(offsets, doc_start, doc_start + len(row["text"]))
            doc_offsets = [(offsets[i][0] - doc_start, offsets[i][1] - doc_start) for i in doc_index]
            gold = []
            for start, end in question["spans"]:
                first, last = span_to_tokens(doc_offsets, start, end)
                covered = row["text"][max(doc_offsets[first][0], 0) : doc_offsets[last][1]]
                if row["text"][start:end].strip() not in covered:
                    raise RuntimeError(f"{row['id']} {question['field']}: tokens miss the gold words")
                gold.append((first, last))
            examples.append(
                {
                    **{k: question[k] for k in ("doc", "field", "type", "op")},
                    "split": row["split"],
                    "ids": list(encoded["input_ids"]),
                    "doc_index": doc_index,
                    "q_index": last_token_inside(offsets, *q_spans[0]),
                    "tags": torch.tensor(bio_tags(len(doc_index), gold), dtype=torch.long),
                    "gold": gold,
                }
            )
    return examples


def assign_parts(examples: list[dict]) -> dict[str, list[dict]]:
    train_docs = [ex["doc"] for ex in examples if ex["split"] == "train"]
    fold_of = folds_for(train_docs)
    parts = {"fit": [], "held": [], **{name: [] for name in DEV}}
    for ex in examples:
        if ex["split"] == "train":
            parts["held" if fold_of[ex["doc"]] == 0 else "fit"].append(ex)
        elif ex["split"] in DEV:
            parts[ex["split"]].append(ex)
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


# ---- LoRA -------------------------------------------------------------------


def _example_states(forward, ex: dict, layers: list[int]):
    hidden = forward(ex["ids"])
    stacked = torch.stack([hidden[i][0] for i in layers]).float()
    return stacked[:, ex["doc_index"]].unsqueeze(0), stacked[:, ex["q_index"]].unsqueeze(0)


def lora_logits(forward, head: Tagger, examples: list[dict], layers: list[int], autocast) -> list[torch.Tensor]:
    head.eval()
    out = []
    with torch.no_grad():
        for ex in examples:
            with autocast():
                doc, question = _example_states(forward, ex, layers)
                logits = head(doc, question)
            out.append(logits[0].float().cpu())
    return out


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
    log=print,
) -> dict:
    """Train adapters and head together; keep the epoch best on `held`."""
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    optimizer = torch.optim.AdamW(
        [{"params": adapter_params, "lr": lr_adapter}, {"params": head.parameters(), "lr": lr_head}],
        weight_decay=0.01,
    )
    amp = use_amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def autocast():
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp)

    weights = torch.tensor(TAG_WEIGHTS, device=device)
    fit = list(parts["fit"])
    history, best = [], None
    for epoch in range(1, epochs + 1):
        set_train(True)
        head.train()
        rng.shuffle(fit)
        total, skipped = 0.0, 0
        optimizer.zero_grad()
        for step, ex in enumerate(fit, start=1):
            with autocast():
                doc, question = _example_states(forward, ex, layers)
                logits = head(doc, question)
            loss = F.cross_entropy(logits[0].float(), ex["tags"].to(device), weight=weights)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            scaler.scale(loss / accumulate).backward()
            total += float(loss.detach())
            if step % accumulate == 0 or step == len(fit):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(adapter_params + list(head.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            if step % 200 == 0:
                log(f"  lora epoch {epoch} step {step}/{len(fit)} loss {total / step:.4f}")
        set_train(False)
        held_logits = lora_logits(forward, head, parts["held"], layers, autocast)
        bias, table = pick_bias(parts["held"], held_logits)
        row = {"epoch": epoch, "train_loss": total / max(1, len(fit) - skipped), "skipped": skipped,
               "held_exact_at_0": table[0.0], "held_exact_best_bias": table[bias], "bias": bias}
        history.append(row)
        log(f"  lora epoch {epoch}: {row}")
        if best is None or table[bias] > best["held"]:
            best = {
                "epoch": epoch, "bias": bias, "held": table[bias],
                "adapter": [p.detach().cpu().clone() for p in adapter_params],
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            }
    with torch.no_grad():
        for param, saved in zip(adapter_params, best["adapter"]):
            param.copy_(saved.to(param.device))
    head.load_state_dict(best["head"])
    set_train(False)
    result = {"epoch": best["epoch"], "bias": best["bias"], "history": history}
    result["held"] = report_part(parts["held"], lora_logits(forward, head, parts["held"], layers, autocast), best["bias"])
    sample = parts["fit"][:300]
    result["fit_sample"] = report_part(sample, lora_logits(forward, head, sample, layers, autocast), best["bias"])
    for name in DEV:
        result[name] = report_part(
            parts[name], lora_logits(forward, head, parts[name], layers, autocast), best["bias"], keep_examples=200
        )
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
    lines += ["", "## Exams 1-3 by operation (exact / fields)", ""]
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
