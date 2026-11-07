"""Kaggle job: untrained Gemma 4 E2B writing schema-constrained JSON (the baseline).

Writes structured_results.{json,md} to /kaggle/working. Exam 4 is not loaded.
Our system's numbers on the same parts are in outputs/experiments (exams 1-3:
experiment 4; computed fields: experiment 6; words: experiment 5b). Its SROIE
score is not known yet: that is the final grade.
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


def main() -> None:
    import torch
    import transformers

    from experiments import sroie
    from experiments.augment import TEST_TEMPLATES, augment
    from experiments.chains import test_variants
    from experiments.evidence_data import build_prompt, load_rows
    from experiments.kaggle_lora_job import text_tower
    from experiments.structured_baseline import (field_list, messages_for, parse_reply, pick_shots, schema_for,
                                                 score_fields, score_sroie)

    started = time.time()
    device = "cuda:0"
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=device)
    model.eval()
    _, tower = text_tower(model)
    log(f"loaded {type(model).__name__}, transformers {transformers.__version__}")

    enforcer = {"mode": "free generation + JSON parse", "errors": 0}
    try:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (build_token_enforcer_tokenizer_data,
                                                                build_transformers_prefix_allowed_tokens_fn)

        tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
        enforcer["mode"] = "schema-constrained (lm-format-enforcer)"
    except Exception as exc:  # the library may not support this tokenizer; fall back, and say so
        tokenizer_data = None
        enforcer["setup_error"] = repr(exc)[:400]
    log(f"decoding mode: {enforcer['mode']}")

    stats = {"generations": 0, "seconds": 0.0, "new_tokens": 0, "unparsed": 0}

    def generate(messages, fields) -> dict | None:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        kwargs = {"max_new_tokens": 48 + 40 * len(fields), "do_sample": False}
        if tokenizer_data is not None:
            try:
                kwargs["prefix_allowed_tokens_fn"] = build_transformers_prefix_allowed_tokens_fn(
                    tokenizer_data, JsonSchemaParser(schema_for(fields)))
            except Exception as exc:
                enforcer["errors"] += 1
                enforcer["last_error"] = repr(exc)[:300]
        t0 = time.time()
        with torch.no_grad():
            try:
                out = model.generate(ids, **kwargs)
            except Exception as exc:
                enforcer["errors"] += 1
                enforcer["last_error"] = repr(exc)[:300]
                kwargs.pop("prefix_allowed_tokens_fn", None)
                out = model.generate(ids, **kwargs)
        stats["seconds"] += time.time() - t0
        stats["generations"] += 1
        stats["new_tokens"] += int(out.shape[1] - ids.shape[1])
        reply = parse_reply(tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
        stats["unparsed"] += reply is None
        return reply

    rows = [r for r in load_rows() if r["split"] != "exam4"]
    dev = [r for r in rows if r["split"] in ("exam1", "exam2", "exam3")]
    shots = pick_shots(rows)
    payload = {"model": MODEL, "decoding": enforcer, "shots": [r["id"] for r, _ in shots], "parts": {}}

    def save():
        payload["generation"] = {**stats, "seconds_per_note": round(stats["seconds"] / max(stats["generations"], 1), 2),
                                 "tokens_per_note": round(stats["new_tokens"] / max(stats["generations"], 1), 1)}
        payload["seconds"] = round(time.time() - started, 1)
        (OUT / "structured_results.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
        (OUT / "structured_results.md").write_text(summarize(payload))

    # Our system's cost on the same notes: one forward of the one-pass prompt (heads are negligible).
    times = []
    with torch.no_grad():
        for row in dev:
            from experiments.evidence_data import questions

            full, _, _ = build_prompt("multi", row["text"], [q["body"] for q in questions(row)])
            ids = torch.tensor([tokenizer(full)["input_ids"]], device=device)
            torch.cuda.synchronize()
            t0 = time.time()
            tower(input_ids=ids, output_hidden_states=True, use_cache=False)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
    payload["one_pass_forward_seconds_per_note"] = round(sum(times) / len(times), 4)

    for name, examples in (("exams1-3 zero-shot", []), ("exams1-3 3-shot", shots)):
        items = []
        for row in dev:
            fields = field_list(row)
            reply = generate(messages_for(row, fields, examples), fields)
            items += [(f, reply) for f in fields]
        payload["parts"][name] = score_fields(items)
        log(f"{name}: {payload['parts'][name]['exact']}/{payload['parts'][name]['fields']}")
        save()

    for kind in ("single", "combo_seen", "combo_unseen"):
        items = []
        for row in [r for r in test_variants(dev, seed=23) if r["chain_test"] == kind]:
            fields = field_list(row, only=["calc_0"])
            items += [(f, generate(messages_for(row, fields, shots), fields)) for f in fields]
        payload["parts"][f"computed {kind} 3-shot"] = score_fields(items)
        log(f"computed {kind}: {payload['parts'][f'computed {kind} 3-shot']['exact']}/{len(items)}")
        save()

    items = []
    for row in augment(dev, seed=23, templates=TEST_TEMPLATES, emits=0):
        fields = field_list(row)
        reply = generate(messages_for(row, fields, shots), fields)
        items += [(f, reply) for f in fields]
    payload["parts"]["words 3-shot"] = score_fields(items)
    log(f"words: {payload['parts']['words 3-shot']['exact']}/{payload['parts']['words 3-shot']['fields']}")
    save()

    roots = sorted(path for path in INPUT.rglob("SROIE2019/test") if (path / "box").is_dir())
    if roots:
        receipts, skipped = sroie.load_split(roots[0])
        items = []
        for row in receipts:
            fields = field_list(row)
            reply = generate(messages_for(row, fields, []), fields)
            items += [(row, f, reply) for f in fields]
            if len(items) % 200 < len(fields):
                log(f"sroie {len(items)} fields")
        payload["sroie"] = {"receipts": len(receipts), "skipped": skipped, "score": score_sroie(items)}
        log(f"sroie: {payload['sroie']['score']['overall']}")
    save()
    log(f"done in {payload['seconds']} s")


OURS = {
    # Our system on the same parts (outputs/experiments).
    "exams1-3 zero-shot": "223/293 (experiment 4, one pass)",
    "exams1-3 3-shot": "223/293 (experiment 4, one pass)",
    "computed single 3-shot": "39/72 (experiment 6)",
    "computed combo_seen 3-shot": "9/72 (experiment 6)",
    "computed combo_unseen 3-shot": "4/72 (experiment 6)",
    "words 3-shot": "147/287 (experiment 5b, learned reader)",
}


def summarize(payload: dict) -> str:
    gen = payload.get("generation", {})
    lines = [
        "# Untrained Gemma 4 E2B with structured output, against our system",
        "",
        f"Decoding: {payload['decoding']['mode']} (errors: {payload['decoding'].get('errors', 0)}). "
        f"3-shot examples: {', '.join(payload['shots'])}.",
        f"Generation: {gen.get('seconds_per_note')} s and {gen.get('tokens_per_note')} new tokens per note on a T4 "
        f"(unparsed replies: {gen.get('unparsed')}). Our system: one forward, "
        f"{payload.get('one_pass_forward_seconds_per_note')} s per note.",
        "",
        "| Part | Structured output, exact | Text loose | Our system |",
        "| --- | --- | --- | --- |",
    ]
    for name, part in payload["parts"].items():
        lines.append(f"| {name} | {part['exact']}/{part['fields']} | {part['exact_loose_text']}/{part['fields']} | {OURS.get(name, '-')} |")
    lines += ["", "By operation:", ""]
    for name, part in payload["parts"].items():
        lines.append(f"- {name}: {part['by_op']}")
    if payload.get("sroie"):
        s = payload["sroie"]["score"]
        lines += ["", f"## SROIE test receipts, zero-shot ({payload['sroie']['receipts']} receipts)", "",
                  f"Overall {s['overall']['right']}/{s['overall']['fields']}. "
                  + ", ".join(f"{k} {v['right']}/{v['fields']}" for k, v in s["by_field"].items()),
                  "", "Our system on SROIE: not run yet (kept for the final grade)."]
    return "\n".join(lines) + "\n"
