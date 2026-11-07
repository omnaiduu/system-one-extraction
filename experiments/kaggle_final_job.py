"""The final grade: exam 4 once, and the SROIE test receipts once.

Loads the chosen trained model (adapters, tagger, operation head, layout and
decode bias, all frozen as they were selected) from a previous kernel's
output, attached as a kernel source. Nothing is trained or tuned here.

This is the only job whose kernel bundle contains exam 4. Run it once, after
the model is final. Writes `final_results.json` and `final_results.md`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

MODEL = "google/gemma-4-E2B-it"
OUT = Path("/kaggle/working")
INPUT = Path("/kaggle/input")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
    (OUT / "status.txt").write_text(message + "\n")


def find_one(pattern: str) -> Path:
    found = sorted(INPUT.rglob(pattern))
    if len(found) != 1:
        raise RuntimeError(f"expected one {pattern} under {INPUT}, found {found}")
    return found[0]


def summarize(payload: dict) -> str:
    e4 = payload["exam4"]
    sr = payload["sroie"]
    lines = [
        "# Final grade",
        "",
        f"Model from `{payload['source']}`: layout `{payload['layout']}`, decode bias {payload['bias']}. "
        "Scored once; nothing was trained or tuned in this run.",
        "",
        "## Exam 4 (24 synthetic notes, never used before)",
        "",
        f"- Final answers: **{e4['values']['exact']}/{e4['values']['fields']}**",
        f"- Evidence exact: {e4['evidence']['exact']}/{e4['evidence']['fields']}",
        f"- Operation choice: {e4['values']['op_choice']}",
        f"- Final answers by operation: {e4['values']['by_op']}",
        "",
        "For comparison (different field sets: the four heads also counted 3 forecasts): "
        "four heads, round 4: 9/77; marking test with decode rules designed on exam 4: 42/74.",
        "",
    ]
    for kind, v in payload.get("exam4_augmented", {}).items():
        lines.append(f"- Exam 4 with {kind} variants: **{v['exact']}/{v['fields']}**; by operation {v['by_op']}; "
                     f"emit numbers within 1%: {v.get('emit_number_within_1pct')}")
    lines += [
        "",
        "## SROIE test receipts (real scanned receipts, zero-shot)",
        "",
        f"{sr['receipts']} receipts, {sr['score']['overall']['fields']} fields. Left out: {sr['skipped']}.",
        "",
        "| Field | Fields | Found the words | Final answer | Final answer, wide date reader |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, row in sr["score"]["by_field"].items():
        lines.append(f"| {name} | {row['fields']} | {row['found']} | {row['value']} | {row['value_wide_dates']} |")
    o = sr["score"]["overall"]
    lines.append(f"| **all** | {o['fields']} | {o['found']} | {o['value']} | {o['value_wide_dates']} |")
    lines += [
        "",
        "The wide date reader was written after looking at a 30-receipt sample of this split, so that column is not a clean test.",
        "",
        "Confidence on SROIE (wide date reader): AUROC "
        f"{sr['score']['confidence']['auroc']}; above-threshold right/kept: "
        + ", ".join(f">={t}: {v['right']}/{v['kept']}" for t, v in sr["score"]["confidence"]["thresholds"].items()),
    ]
    if "confidence" in e4["values"]:
        c = e4["values"]["confidence"]
        lines += ["", "Confidence on exam 4: AUROC "
                  f"{c['auroc']}; above-threshold right/kept: "
                  + ", ".join(f">={t}: {v['right']}/{v['kept']}" for t, v in c["thresholds"].items())]
    return "\n".join(lines) + "\n"


def main() -> None:
    import torch
    import transformers
    from peft import LoraConfig, inject_adapter_in_model

    from experiments import sroie
    from experiments.evidence_data import load_rows
    from experiments.evidence_heads import Tagger
    from experiments.kaggle_lora_job import attention_targets, text_tower
    from experiments.lora_tagger import OpHead, encode, lora_logits, make_readers, report_part, value_report

    started = time.time()
    device = "cuda:0"
    adapter_path = find_one("*_adapter.pt")
    checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=False)
    layout = checkpoint.get("layout", "first")
    bias = float(checkpoint["bias"])
    layers = list(checkpoint["layers"])
    log(f"checkpoint {adapter_path}: layout {layout}, bias {bias}, layers {layers}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=device)
    model.eval()
    model.requires_grad_(False)
    lora = checkpoint["lora"]
    targets = attention_targets(model)
    if len(targets) != lora["modules"]:
        raise RuntimeError(f"{len(targets)} target modules, checkpoint has {lora['modules']}")
    config = LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=0.0, target_modules=targets, bias="none")
    model = inject_adapter_in_model(config, model)
    params = dict(model.named_parameters())
    names = checkpoint["adapter_names"]
    if len(names) != len(checkpoint["adapter"]):
        raise RuntimeError("adapter names and tensors do not line up")
    with torch.no_grad():
        for name, tensor in zip(names, checkpoint["adapter"]):
            params[name].data = tensor.to(device=device, dtype=torch.float32)
    model.eval()
    _, tower = text_tower(model)
    width = tower.config.hidden_size if hasattr(tower, "config") else 1536
    head = Tagger(len(layers), width).to(device)
    head.load_state_dict(checkpoint["head"])
    op_head = OpHead(len(layers), width).to(device)
    op_head.load_state_dict(checkpoint["op_head"])
    readers = None
    if checkpoint.get("readers"):
        readers = make_readers(len(layers), width)
        for name, reader in readers.items():
            reader.load_state_dict(checkpoint["readers"][name])
            reader.to(device)
    log(f"loaded {len(names)} adapter tensors, tagger, operation head, readers: {sorted(readers) if readers else None}")

    def forward(ids):
        out = tower(input_ids=torch.tensor([ids], device=device), output_hidden_states=True, use_cache=False)
        return out.hidden_states

    def autocast():
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    payload = {"source": str(adapter_path), "layout": layout, "bias": bias, "layers": layers}

    exam4 = [row for row in load_rows() if row["split"] == "exam4"]
    examples = encode(exam4, tokenizer, layout)
    tags, ops, extras = lora_logits(forward, head, op_head, examples, layers, autocast, readers)
    payload["exam4"] = {
        "notes": len(exam4),
        "evidence": report_part(examples, tags, bias, keep_examples=200),
        "values": value_report(examples, tags, ops, bias, keep_examples=200, extras=extras),
    }
    if readers is not None:
        # Exam 4 rewritten the same way as the dev notes: words, and emit fields
        # with the held-out phrasings. Still exam 4, still scored once.
        from experiments.augment import TEST_TEMPLATES, augment

        payload["exam4_augmented"] = {}
        for kind, variants in (("words", [r for r in augment(exam4, seed=47, emits=0, templates=TEST_TEMPLATES)]),
                               ("emit", [r for r in augment(exam4, seed=53, words=0, templates=TEST_TEMPLATES)])):
            ex_aug = encode(variants, tokenizer, layout)
            t, o, x = lora_logits(forward, head, op_head, ex_aug, layers, autocast, readers)
            payload["exam4_augmented"][kind] = value_report(ex_aug, t, o, bias, keep_examples=100, extras=x)
            log(f"exam 4 {kind}: {payload['exam4_augmented'][kind]['exact']}/{payload['exam4_augmented'][kind]['fields']}")
    log(f"exam 4: final answers {payload['exam4']['values']['exact']}/{payload['exam4']['values']['fields']}")
    (OUT / "final_results.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")

    roots = sorted(path for path in INPUT.rglob("SROIE2019/test") if (path / "box").is_dir())
    if not roots:
        raise RuntimeError(f"no SROIE2019/test with a box folder under {INPUT}")
    root = roots[0]
    rows, skipped = sroie.load_split(root)
    examples = encode(rows, tokenizer, layout)
    tags, ops, extras = lora_logits(forward, head, op_head, examples, layers, autocast, readers)
    payload["sroie"] = {"receipts": len(rows), "skipped": skipped, "score": sroie.score(examples, tags, ops, bias, extras)}
    o = payload["sroie"]["score"]["overall"]
    log(f"sroie: found {o['found']}/{o['fields']}, value {o['value']}, wide dates {o['value_wide_dates']}")

    payload["seconds"] = round(time.time() - started, 1)
    payload = json.loads(json.dumps(payload, default=str))
    (OUT / "final_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "final_results.md").write_text(summarize(payload))
    log(f"done in {payload['seconds']} s")
