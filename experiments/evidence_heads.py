"""Evidence tagger and the h_T probe, trained on cached Gemma states.

Check 2 (does question order matter?): the same tagger is trained on states
from three prompt layouts and scored by grouped 5-fold cross-validation over
the 200 training notes, then once on exams 1-3. Exam 4 is not scored.

Check 1 (can a head read an amount out of one state vector?): a linear
readout from one vector to the digits of every number field, scored by the
same cross-validation.

An item is one (note, field) question:
    states_key  key into the shared `states` dict -> float16 [K, T, H]
    q_state     float16 [K, H]  last token of the question text
    end_state   float16 [K, H]  last token of the whole prompt
    tags        long [T]        0 outside, 1 begin, 2 inside
    gold        list of (first, last) token positions
    doc, field, op, type, split, value, decimals
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from experiments.evidence_data import bucket, decode_bio
from extractor.labels import INT_DIGITS, split_number

FOLDS = 5
SEED = 7
EPOCHS = 40
EVAL_EVERY = 5
BATCH = 16
LR = 1e-3
WEIGHT_DECAY = 0.01
# Evidence tokens are a few out of hundreds. Without extra weight the tagger
# learns "nothing here" and misses real spans.
TAG_WEIGHTS = (1.0, 3.0, 3.0)


def folds_for(doc_ids: list[str], folds: int = FOLDS, seed: int = SEED) -> dict[str, int]:
    unique = sorted(set(doc_ids))
    random.Random(seed).shuffle(unique)
    return {doc: index % folds for index, doc in enumerate(unique)}


class Tagger(nn.Module):
    """Tag each document token as outside / begin / inside for one question."""

    def __init__(self, layers: int, width: int, inner: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.doc_mix = nn.Parameter(torch.zeros(layers))
        self.q_mix = nn.Parameter(torch.zeros(layers))
        self.doc_norm = nn.LayerNorm(width)
        self.q_norm = nn.LayerNorm(width)
        self.doc_proj = nn.Linear(width, inner)
        self.q_proj = nn.Linear(width, inner)
        self.conv = nn.Conv1d(3 * inner, inner, kernel_size=3, padding=1)
        self.out = nn.Linear(inner, 3)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _mix(weights: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        # Layers have different scales, so each one is normalized before the mix.
        states = F.layer_norm(states, states.shape[-1:])
        return torch.einsum("k,bk...->b...", torch.softmax(weights, 0), states)

    def forward(self, doc: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        d = self.doc_proj(self.drop(self.doc_norm(self._mix(self.doc_mix, doc))))
        q = self.q_proj(self.drop(self.q_norm(self._mix(self.q_mix, question))))
        q = q[:, None, :].expand_as(d)
        features = torch.cat([d, q, d * q], dim=-1).transpose(1, 2)
        hidden = F.gelu(self.conv(features)).transpose(1, 2)
        return self.out(self.drop(hidden))


def _batch(items: list[dict], states: dict, device: str):
    longest = max(states[item["states_key"]].shape[1] for item in items)
    layers, _, width = states[items[0]["states_key"]].shape
    doc = torch.zeros(len(items), layers, longest, width, dtype=torch.float32)
    tags = torch.full((len(items), longest), -100, dtype=torch.long)
    for row, item in enumerate(items):
        block = states[item["states_key"]]
        doc[row, :, : block.shape[1]] = block.float()
        tags[row, : block.shape[1]] = item["tags"]
    question = torch.stack([item["q_state"].float() for item in items])
    return doc.to(device), question.to(device), tags.to(device)


def predict(model: Tagger, items: list[dict], states: dict, device: str) -> list[list[tuple[int, int]]]:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(items), BATCH):
            chunk = items[start : start + BATCH]
            doc, question, tags = _batch(chunk, states, device)
            choice = model(doc, question).argmax(-1).cpu()
            for row, item in enumerate(chunk):
                length = states[item["states_key"]].shape[1]
                out.append(decode_bio(choice[row, :length].tolist()))
    return out


def score(items: list[dict], predicted: list[list[tuple[int, int]]]) -> dict:
    buckets = Counter()
    by_op: dict[str, Counter] = defaultdict(Counter)
    for item, spans in zip(items, predicted):
        label = bucket(spans, item["gold"])
        buckets[label] += 1
        by_op[item["op"]][label] += 1
    total = len(items)
    return {
        "fields": total,
        "exact": buckets["exact"],
        "exact_rate": buckets["exact"] / total if total else None,
        "buckets": dict(buckets),
        "by_op": {op: dict(counts) for op, counts in sorted(by_op.items())},
    }


def train_tagger(items: list[dict], states: dict, device: str, epochs: int, checkpoints=(), seed: int = SEED):
    """Train one tagger. Yields (epoch, model) at each requested checkpoint."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    layers, _, width = states[items[0]["states_key"]].shape
    model = Tagger(layers, width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    order = list(items)
    class_weights = torch.tensor(TAG_WEIGHTS, device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(order)
        for start in range(0, len(order), BATCH):
            doc, question, tags = _batch(order[start : start + BATCH], states, device)
            logits = model(doc, question)
            loss = F.cross_entropy(
                logits.reshape(-1, 3), tags.reshape(-1), weight=class_weights, ignore_index=-100
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if epoch in checkpoints:
            yield epoch, model


def run_condition(items: list[dict], states: dict, device: str, log=print) -> dict:
    train = [item for item in items if item["split"] == "train"]
    dev = [item for item in items if item["split"] in ("exam1", "exam2", "exam3")]
    fold_of = folds_for([item["doc"] for item in train])
    checkpoints = tuple(range(EVAL_EVERY, EPOCHS + 1, EVAL_EVERY))
    held_out: dict[int, list] = {epoch: [] for epoch in checkpoints}
    for fold in range(FOLDS):
        fit = [item for item in train if fold_of[item["doc"]] != fold]
        hold = [item for item in train if fold_of[item["doc"]] == fold]
        for epoch, model in train_tagger(fit, states, device, EPOCHS, checkpoints):
            held_out[epoch].extend(zip(hold, predict(model, hold, states, device)))
        log(f"  fold {fold} done")
    curve = {}
    for epoch in checkpoints:
        pairs = held_out[epoch]
        curve[epoch] = score([p[0] for p in pairs], [p[1] for p in pairs])
    best_epoch = max(checkpoints, key=lambda epoch: (curve[epoch]["exact"], -epoch))
    log(f"  cv best epoch {best_epoch}: {curve[best_epoch]['exact']}/{curve[best_epoch]['fields']}")
    final = None
    for _, model in train_tagger(train, states, device, best_epoch, (best_epoch,)):
        final = model
    dev_scores = {}
    examples = []
    for split in ("exam1", "exam2", "exam3"):
        part = [item for item in dev if item["split"] == split]
        predicted = predict(final, part, states, device)
        dev_scores[split] = score(part, predicted)
        for item, spans in list(zip(part, predicted))[:400]:
            examples.append(
                {
                    "split": split,
                    "doc": item["doc"],
                    "field": item["field"],
                    "op": item["op"],
                    "gold": item["gold"],
                    "predicted": spans,
                    "bucket": bucket(spans, item["gold"]),
                }
            )
    merged = score(dev, predict(final, dev, states, device))
    return {
        "cv": {"best_epoch": best_epoch, "at_best": curve[best_epoch], "curve": {e: curve[e]["exact"] for e in checkpoints}},
        "dev": {"all": merged, **dev_scores},
        "dev_examples": examples,
    }


# ---- Check 1: read the digits of an amount from one vector -----------------


def digit_targets(item: dict) -> torch.Tensor | None:
    if item["type"] != "number" or item["op"] == "missing" or item["value"] is None:
        return None
    try:
        encoded = split_number(item["value"], int(item["decimals"] if item["decimals"] is not None else 2))
    except ValueError:
        return None
    return torch.tensor([encoded["sign"]] + encoded["whole"] + encoded["frac"], dtype=torch.long)


def probe(items: list[dict], vector: str, device: str, epochs: int = 300) -> dict:
    """Linear readout of sign + 7 + 2 digits from one state vector, per layer."""
    rows = [(item, digit_targets(item)) for item in items if item["split"] == "train"]
    rows = [(item, target) for item, target in rows if target is not None]
    if len(rows) < FOLDS * 2:
        return {"numbers": len(rows), "skipped": True}
    targets = torch.stack([target for _, target in rows])
    vectors = torch.stack([item[vector].float() for item, _ in rows])  # [N, K, H]
    fold_of = folds_for([item["doc"] for item, _ in rows])
    fold_ids = torch.tensor([fold_of[item["doc"]] for item, _ in rows])
    places = targets.shape[1]
    result = {"numbers": len(rows), "places": places, "layers": {}}
    # Baseline: guess the most common digit at each place, from the training folds.
    base_hits = torch.zeros(len(rows), dtype=torch.bool)
    for fold in range(FOLDS):
        fit = fold_ids != fold
        common = torch.mode(targets[fit], dim=0).values
        base_hits[~fit] = (targets[~fit] == common).all(dim=1)
    result["baseline_exact"] = int(base_hits.sum())
    for layer in range(vectors.shape[1]):
        x_all = F.layer_norm(vectors[:, layer], vectors.shape[-1:])
        exact = torch.zeros(len(rows), dtype=torch.bool)
        place_hits = torch.zeros(places)
        train_exact = 0
        for fold in range(FOLDS):
            fit = fold_ids != fold
            torch.manual_seed(SEED)
            readout = nn.Linear(x_all.shape[1], places * 10).to(device)
            optimizer = torch.optim.AdamW(readout.parameters(), lr=1e-2, weight_decay=0.1)
            x = x_all[fit].to(device)
            y = targets[fit].to(device)
            for _ in range(epochs):
                logits = readout(x).view(-1, places, 10)
                loss = F.cross_entropy(logits.transpose(1, 2), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                train_exact += int((readout(x).view(-1, places, 10).argmax(-1) == y).all(1).sum())
                guess = readout(x_all[~fit].to(device)).view(-1, places, 10).argmax(-1).cpu()
            hit = guess == targets[~fit]
            exact[~fit] = hit.all(1)
            place_hits += hit.float().sum(0)
        result["layers"][layer] = {
            "cv_exact": int(exact.sum()),
            "train_exact_mean": train_exact / FOLDS,
            "cv_place_accuracy": (place_hits / len(rows)).tolist(),
        }
    return result


def run_all(items_by_prompt: dict, states_by_prompt: dict, device: str, log=print) -> dict:
    results = {}
    for prompt, items in items_by_prompt.items():
        log(f"condition {prompt}: {len(items)} questions")
        states = states_by_prompt[prompt]
        outcome = run_condition(items, states, device, log)
        # end_state is the vector that has read both the note and this question:
        # the question's own last token when the question comes last, the last
        # prompt token when the question comes first.
        outcome["probe_read_both"] = probe(items, "end_state", device)
        results[prompt] = outcome
    return results
