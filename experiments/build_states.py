"""Run Gemma over every note in each prompt layout and keep the states.

`forward(ids, add_bos)` is passed in, so the same code runs on Modal with the
real model and in the tests with a fake one. It returns float16 [K, T, H]
for the chosen layers.
"""

from __future__ import annotations

import torch

from experiments.evidence_data import (
    bio_tags,
    build_prompt,
    doc_token_range,
    last_token_inside,
    questions,
    span_to_tokens,
)


def _encode(tokenizer, prompt: str, add_bos: bool):
    encoded = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=add_bos)
    ids = list(encoded["input_ids"])
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    return ids, offsets


def _item(question: dict, row: dict, key: str, doc_offsets, q_state, end_state, check: dict) -> dict:
    token_spans = []
    for start, end in question["spans"]:
        first, last = span_to_tokens(doc_offsets, start, end)
        covered = row["text"][max(doc_offsets[first][0], 0) : doc_offsets[last][1]]
        gold_text = row["text"][start:end].strip()
        if gold_text not in covered:
            raise RuntimeError(f"{row['id']} {question['field']}: tokens {covered!r} miss {gold_text!r}")
        if covered.strip() != gold_text:
            check["covering_text_differs"] += 1
        token_spans.append((first, last))
    return {
        **{k: question[k] for k in ("doc", "field", "type", "op", "value", "decimals")},
        "split": row["split"],
        "states_key": key,
        "q_state": q_state,
        "end_state": end_state,
        "tags": torch.tensor(bio_tags(len(doc_offsets), token_spans), dtype=torch.long),
        "gold": token_spans,
    }


def build(rows: list[dict], tokenizer, forward, prompt: str, log=print):
    add_bos = prompt != "last_nobos"
    items: list[dict] = []
    states: dict[str, torch.Tensor] = {}
    check = {"covering_text_differs": 0, "forwards": 0, "tokens": 0}
    for number, row in enumerate(rows):
        qs = questions(row)
        if not qs:
            continue
        text = row["text"]
        if prompt in ("last", "last_nobos"):
            full, doc_start, q_spans = build_prompt(prompt, text, [q["body"] for q in qs])
            ids, offsets = _encode(tokenizer, full, add_bos)
            hidden = forward(ids, add_bos)
            check["forwards"] += 1
            check["tokens"] += len(ids)
            doc_index = doc_token_range(offsets, doc_start, doc_start + len(text))
            if doc_index != list(range(doc_index[0], doc_index[0] + len(doc_index))):
                raise RuntimeError(f"{row['id']} document tokens are not contiguous")
            key = row["id"]
            states[key] = hidden[:, doc_index].contiguous()
            doc_offsets = [(offsets[i][0] - doc_start, offsets[i][1] - doc_start) for i in doc_index]
            for question, (q_start, q_end) in zip(qs, q_spans):
                q_index = last_token_inside(offsets, q_start, q_end)
                vector = hidden[:, q_index].clone()
                items.append(_item(question, row, key, doc_offsets, vector, vector, check))
        else:
            for question in qs:
                full, doc_start, q_spans = build_prompt(prompt, text, [question["body"]])
                ids, offsets = _encode(tokenizer, full, add_bos)
                hidden = forward(ids, add_bos)
                check["forwards"] += 1
                check["tokens"] += len(ids)
                doc_index = doc_token_range(offsets, doc_start, doc_start + len(text))
                key = f"{row['id']}::{question['field']}"
                states[key] = hidden[:, doc_index].contiguous()
                doc_offsets = [(offsets[i][0] - doc_start, offsets[i][1] - doc_start) for i in doc_index]
                q_index = last_token_inside(offsets, *q_spans[0])
                items.append(
                    _item(question, row, key, doc_offsets, hidden[:, q_index].clone(), hidden[:, -1].clone(), check)
                )
        if number % 50 == 0:
            log(f"  {prompt}: {number + 1}/{len(rows)} notes, {check['forwards']} forwards")
    return items, states, check
