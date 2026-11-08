"""Kaggle job: "LLM reads, code computes", and our trained system on SROIE.

1. Untrained Gemma 4 E2B with `SYSTEM_CALC`: printed values as before; a value
   that must be calculated comes back as {"inputs", "steps"} and is computed
   exactly by `values` (the same code as our system). Scored on exams 1-3, the
   computed-field sets of experiment 6, the words notes and SROIE test receipts.
2. Our best trained model (experiment 4, one pass; its adapter comes from the
   kernel source `ommnaidu/gemma-e2b-lora-onepass`) on SROIE test receipts, so
   the real-document row has both systems.

Exam 4 is not loaded. Writes hybrid_results.{json,md}.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

MODEL = "google/gemma-4-E2B-it"
OUT = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

PREVIOUS = {
    # Earlier numbers on the same parts: (structured output with arithmetic by the LLM, our system).
    "exams1-3": ("250/293", "223/293"),
    "computed single": ("22/72", "39/72"),
    "computed combo_seen": ("2/72", "9/72"),
    "computed combo_unseen": ("0/72", "4/72"),
    "words": ("216/287", "147/287"),
    "sroie": ("1109/1362", None),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
    (OUT / "status.txt").write_text(message + "\n")


def main() -> None:
    import torch
    import transformers

    from experiments import sroie
    from experiments.augment import TEST_TEMPLATES, augment
    from experiments.chains import test_variants
    from experiments.evidence_data import load_rows
    from experiments.structured_baseline import (SYSTEM_CALC, field_list, messages_for, parse_reply, score_fields,
                                                 score_sroie)

    started = time.time()
    device = "cuda:0"
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=device)
    model.eval()
    model.requires_grad_(False)
    payload = {"model": MODEL, "parts": {}, "previous": PREVIOUS}
    stats = {"generations": 0, "seconds": 0.0, "new_tokens": 0, "unparsed": 0}

    def render(messages) -> str:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            system, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
            rest[0]["content"] = system + "\n\n" + rest[0]["content"]
            return tokenizer.apply_chat_template(rest, tokenize=False, add_generation_prompt=True)

    def generate(row, fields) -> dict | None:
        ids = tokenizer(render(messages_for(row, fields, [], system=SYSTEM_CALC)), add_special_tokens=False,
                        return_tensors="pt")["input_ids"].to(device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=64 + 56 * len(fields), do_sample=False)
        stats["seconds"] += time.time() - t0
        stats["generations"] += 1
        stats["new_tokens"] += int(out.shape[1] - ids.shape[1])
        reply = parse_reply(tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
        stats["unparsed"] += reply is None
        return reply

    def save():
        payload["generation"] = {**stats, "seconds_per_note": round(stats["seconds"] / max(stats["generations"], 1), 2),
                                 "tokens_per_note": round(stats["new_tokens"] / max(stats["generations"], 1), 1)}
        payload["seconds"] = round(time.time() - started, 1)
        (OUT / "hybrid_results.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
        (OUT / "hybrid_results.md").write_text(summarize(payload))

    rows = [r for r in load_rows() if r["split"] != "exam4"]
    dev = [r for r in rows if r["split"] in ("exam1", "exam2", "exam3")]

    def run_part(name, notes, only=None):
        items = []
        for row in notes:
            fields = field_list(row, only=only)
            reply = generate(row, fields)
            items += [(f, reply) for f in fields]
        payload["parts"][name] = score_fields(items)
        part = payload["parts"][name]
        log(f"{name}: {part['exact']}/{part['fields']}, calculated {part['calculated_right']}")
        save()

    run_part("exams1-3", dev)
    chains = test_variants(dev, seed=23)
    for kind in ("single", "combo_seen", "combo_unseen"):
        run_part(f"computed {kind}", [r for r in chains if r["chain_test"] == kind], only=["calc_0"])
    run_part("words", augment(dev, seed=23, templates=TEST_TEMPLATES, emits=0))

    roots = sorted(path for path in INPUT.rglob("SROIE2019/test") if (path / "box").is_dir())
    receipts, skipped = sroie.load_split(roots[0])
    items = []
    for row in receipts:
        fields = field_list(row)
        reply = generate(row, fields)
        items += [(row, f, reply) for f in fields]
    payload["sroie"] = {"receipts": len(receipts), "skipped": skipped, "llm": score_sroie(items)}
    log(f"sroie llm: {payload['sroie']['llm']['overall']}")
    save()

    # Our trained system on the same receipts.
    payload["sroie"]["ours"] = ours_on_sroie(model, receipts, device)
    o = payload["sroie"]["ours"]["overall"]
    log(f"sroie ours: found {o['found']}/{o['fields']}, value {o['value']}, wide dates {o['value_wide_dates']}")
    save()
    log(f"done in {payload['seconds']} s")


def ours_on_sroie(model, receipts, device) -> dict:
    """Experiment 4's adapter, tagger and operation head, scored with `sroie.score` (as the final grade would)."""
    import torch
    import transformers
    from peft import LoraConfig, inject_adapter_in_model

    from experiments import sroie
    from experiments.evidence_heads import Tagger
    from experiments.kaggle_final_job import find_one
    from experiments.kaggle_lora_job import attention_targets, text_tower
    from experiments.lora_tagger import OpHead, encode, lora_logits

    checkpoint = torch.load(find_one("*_adapter.pt"), map_location="cpu", weights_only=False)
    lora = checkpoint["lora"]
    targets = attention_targets(model)
    config = LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=0.0, target_modules=targets, bias="none")
    model = inject_adapter_in_model(config, model)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in zip(checkpoint["adapter_names"], checkpoint["adapter"]):
            params[name].data = tensor.to(device=device, dtype=torch.float32)
    model.eval()
    _, tower = text_tower(model)
    layers = list(checkpoint["layers"])
    width = getattr(getattr(tower, "config", None), "hidden_size", 1536)
    head = Tagger(len(layers), width).to(device)
    head.load_state_dict(checkpoint["head"])
    op_head = OpHead(len(layers), width, outputs=checkpoint["op_head"]["net.3.weight"].shape[0]).to(device)
    op_head.load_state_dict(checkpoint["op_head"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)

    def forward(ids):
        return tower(input_ids=torch.tensor([ids], device=device), output_hidden_states=True, use_cache=False).hidden_states

    def autocast():
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    examples = encode(receipts, tokenizer, checkpoint.get("layout", "multi"))
    t0 = time.time()
    tags, ops, extras = lora_logits(forward, head, op_head, examples, layers, autocast)
    result = sroie.score(examples, tags, ops, float(checkpoint["bias"]), extras)
    result["seconds_per_receipt"] = round((time.time() - t0) / max(len(receipts), 1), 3)
    return result


def summarize(payload: dict) -> str:
    gen = payload.get("generation", {})
    lines = [
        "# LLM reads, code computes",
        "",
        "Untrained Gemma 4 E2B with `SYSTEM_CALC` (experiments/structured_baseline.py): printed values as JSON; "
        "calculated values as inputs and steps, computed exactly by code. Free generation, JSON parsed.",
        f"Generation: {gen.get('seconds_per_note')} s and {gen.get('tokens_per_note')} new tokens per note "
        f"(unparsed replies: {gen.get('unparsed')}).",
        "",
        "| Part | LLM reads, code computes | Calculations right | LLM computes (earlier run) | Our system |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, part in payload["parts"].items():
        before, ours = payload["previous"].get(name, ("-", "-"))
        lines.append(f"| {name} | **{part['exact']}/{part['fields']}** | {part['calculated_right']} | {before} | {ours} |")
    lines += ["", "By operation:", ""]
    for name, part in payload["parts"].items():
        lines.append(f"- {name}: {part['by_op']}")
    s = payload.get("sroie")
    if s:
        llm = s["llm"]
        lines += ["", f"## SROIE test receipts ({s['receipts']} receipts, zero-shot for both)", "",
                  "| Field | LLM reads, code computes | Ours: found the words | Ours: final answer | Ours: wide date reader |",
                  "| --- | --- | --- | --- | --- |"]
        ours = s.get("ours")
        for name, row in llm["by_field"].items():
            o = ours["by_field"].get(name) if ours else None
            lines.append(f"| {name} | {row['right']}/{row['fields']} | "
                         + (f"{o['found']}/{o['fields']} | {o['value']}/{o['fields']} | {o['value_wide_dates']}/{o['fields']} |" if o else "- | - | - |"))
        lines.append(f"| **all** | **{llm['overall']['right']}/{llm['overall']['fields']}** | "
                     + (f"{ours['overall']['found']} | {ours['overall']['value']} | {ours['overall']['value_wide_dates']} |" if ours else "- | - | - |"))
        if ours:
            lines += ["", f"Ours: {ours.get('seconds_per_receipt')} s per receipt. Its wide date reader was written after "
                          "looking at 30 receipts of this split, so that column is not clean. "
                          f"Earlier LLM-only run: {payload['previous']['sroie'][0]}."]
    return "\n".join(lines) + "\n"
