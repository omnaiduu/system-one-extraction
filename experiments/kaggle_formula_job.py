"""Kaggle job: the fixes for "LLM reads, code computes" (note 21), each measured.

The same untrained Gemma 4 E2B as note 18, with three changes:

1. **Formulas** (`SYSTEM_FORMULA`, experiments/formula.py): a calculated value
   comes back as {"formula": "2400 * (1 - 15%) * (1 + 18%)"} instead of our
   step names. Code evaluates it exactly.
2. **The page check and one retry**: every number and date the answer uses
   (or gives as a literal) must be printed in the document. Fields that fail
   are asked once more, with the reason. Scored before and after the retry,
   plus how right the answers are that pass the check (the ones a product
   would accept without a human).
3. **Structured output** (constrained decoding with xgrammar): every reply is
   forced into a JSON schema built from the field list: right keys, a number
   or {"formula"} for numbers, an ISO date or {"formula"} for dates, null
   allowed. Run as a second mode beside free generation, so its effect is
   separate.

Parts: exams 1-3, the three computed-field sets of experiment 6, the words
notes, and SROIE test receipts (in the constrained mode only, or free if the
library does not work). Exam 4 is not loaded. Receipt-level examples are
never written. Writes formula_results.{json,md}.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

MODEL = "google/gemma-4-E2B-it"
OUT = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

PREVIOUS = {
    # Earlier numbers: (LLM computes, note 17; LLM reads with steps, note 18; our system).
    "exams1-3": ("250/293", "252/293", "223/293"),
    "computed single": ("22/72", "43/72", "39/72"),
    "computed combo_seen": ("2/72", "21/72", "9/72"),
    "computed combo_unseen": ("0/72", "26/72", "4/72"),
    "words": ("216/287", "227/287", "147/287"),
    "sroie": ("1109/1362", "1120/1362", "66/1362"),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
    (OUT / "status.txt").write_text(message + "\n")


# ---- pure logic (tested without a model) -------------------------------------------


def schema_with_formulas(fields: list[dict]) -> dict:
    """JSON schema for one reply: every field present; numbers, days and dates may be a formula."""
    formula = {"type": "object", "properties": {"formula": {"type": "string", "maxLength": 200}},
               "required": ["formula"], "additionalProperties": False}
    null = {"type": "null"}
    props = {}
    for f in fields:
        kind = f["type"]
        if kind in ("number", "estimate"):
            props[f["field"]] = {"anyOf": [{"type": "number"}, formula, null]}
        elif kind == "duration":
            props[f["field"]] = {"anyOf": [{"type": "integer"}, formula, null]}
        elif kind == "date":
            props[f["field"]] = {"anyOf": [{"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}, formula, null]}
        elif kind == "list":
            props[f["field"]] = {"anyOf": [{"type": "array", "items": {"type": "string"}}, null]}
        else:
            props[f["field"]] = {"anyOf": [{"type": "string"}, null]}
    return {"type": "object", "properties": props, "required": [f["field"] for f in fields], "additionalProperties": False}


def answer_note(row: dict, fields: list[dict], generate, system: str) -> dict:
    """One note: first reply, page check, one retry for the fields that fail, check again.

    `generate(messages, fields) -> (text, parsed reply or None)`.
    """
    from experiments.formula import check, page_values
    from experiments.structured_baseline import RETRY, messages_for

    page = page_values(row["text"], True)
    messages = messages_for(row, fields, [], system=system)
    text, first = generate(messages, fields)
    first = first or {}

    def problems(reply):
        out = {}
        for f in fields:
            bad = check(reply.get(f["field"]), f["spec"], page)
            if bad:
                out[f["field"]] = bad
        return out

    flagged = problems(first)
    final = dict(first)
    retried = False
    if flagged:
        retried = True
        again = [f for f in fields if f["field"] in flagged]
        note = "\n".join(f"- {name}: {json.dumps(first.get(name))} uses {', '.join(bad)}" for name, bad in flagged.items())
        follow = messages + [{"role": "assistant", "content": text},
                             {"role": "user", "content": RETRY.format(problems=note, fields=", ".join(flagged))}]
        _, second = generate(follow, again)
        for f in again:
            if second and f["field"] in second:
                final[f["field"]] = second[f["field"]]
    still = problems(final)
    return {"first": first, "final": final, "flagged": sorted(flagged), "still": sorted(still), "retried": retried,
            "parsed": bool(first)}


def score_part(results: list[tuple[list[dict], dict]]) -> dict:
    """results: (fields, answer_note output) per note."""
    from experiments.structured_baseline import score_fields, to_prediction
    from experiments.values import matches

    first = [(f, r["first"]) for fields, r in results for f in fields]
    final = [(f, r["final"]) for fields, r in results for f in fields]
    a, b = score_fields(first), score_fields(final)
    accepted = accepted_right = withheld = withheld_right = 0
    for fields, r in results:
        for f in fields:
            ok = matches(f["spec"], to_prediction(r["final"].get(f["field"]), f["spec"]))
            if f["field"] in r["still"]:
                withheld += 1
                withheld_right += ok
            else:
                accepted += 1
                accepted_right += ok
    return {
        "fields": a["fields"], "first": a["exact"], "after_retry": b["exact"],
        "calculated_right_first": a["calculated_right"], "calculated_right_after": b["calculated_right"],
        "by_op_after": b["by_op"],
        "flagged_first": sum(len(r["flagged"]) for _, r in results),
        "flagged_after_retry": sum(len(r["still"]) for _, r in results),
        "notes_retried": sum(r["retried"] for _, r in results),
        "unparsed": sum(not r["parsed"] for _, r in results),
        "accepted": accepted, "accepted_right": accepted_right, "withheld": withheld, "withheld_right": withheld_right,
        "examples": b["examples"],
        "examples_first": [e["said"] for e in a["examples"]],
    }


# ---- the Kaggle run ------------------------------------------------------------------


def main() -> None:
    """First run: both modes, prompt version 1, SROIE in the constrained mode."""
    run(["free", "constrained"], "SYSTEM_FORMULA", "formula_results")


def main_v2() -> None:
    """Second run: constrained only, prompt version 2, and the page check that reads receipt dates (15-01-19)."""
    run(["constrained"], "SYSTEM_FORMULA_V2", "formula_v2_results")


def run(wanted: list[str], prompt_name: str, out_name: str, parts=None, receipts: bool = True,
        model=None, tokenizer=None, previous=PREVIOUS) -> dict:
    """Score the LLM with one prompt in the wanted modes. `parts`: [(name, notes, only fields)], default the dev sets.
    A model and tokenizer already loaded can be passed in (the head-to-head job runs two prompts on one model)."""
    import torch
    import transformers

    from experiments import sroie
    from experiments.augment import TEST_TEMPLATES, augment
    from experiments.chains import test_variants
    from experiments.evidence_data import load_rows
    from experiments import structured_baseline
    from experiments.structured_baseline import field_list, parse_reply, score_sroie

    SYSTEM_FORMULA = getattr(structured_baseline, prompt_name)

    started = time.time()
    device = "cuda:0"
    if model is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
        model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=device)
        model.eval()
        model.requires_grad_(False)
    payload = {"model": MODEL, "prompt": prompt_name, "previous": previous, "modes": {}}
    stats = {}

    def render(messages) -> str:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            system, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
            rest[0]["content"] = system + "\n\n" + rest[0]["content"]
            return tokenizer.apply_chat_template(rest, tokenize=False, add_generation_prompt=True)

    # Structured output: xgrammar's logits processor over the model's own vocabulary.
    grammar = {"available": False}
    try:
        import xgrammar as xgr

        with torch.no_grad():
            vocab = int(model(torch.tensor([[tokenizer.bos_token_id or 2]], device=device)).logits.shape[-1])
        info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab)
        compiler = xgr.GrammarCompiler(info)

        def compile_schema(schema):
            try:
                return compiler.compile_json_schema(json.dumps(schema), any_whitespace=False, separators=(", ", ": "))
            except TypeError:
                return compiler.compile_json_schema(json.dumps(schema))

        compile_schema(schema_with_formulas([{"field": "x", "type": "number"}]))
        grammar = {"available": True, "library": "xgrammar " + getattr(xgr, "__version__", "?"), "vocab": vocab, "errors": 0}
    except Exception as exc:  # noqa: BLE001 - recorded, and the run continues with free generation only
        grammar["setup_error"] = repr(exc)[:500]
    payload["grammar"] = grammar
    log(f"structured output: {grammar}")

    def make_generate(mode: str):
        s = stats.setdefault(mode, {"generations": 0, "seconds": 0.0, "new_tokens": 0})

        def generate(messages, fields):
            ids = tokenizer(render(messages), add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            extra = {}
            if mode == "constrained":
                try:
                    extra["logits_processor"] = [xgr.contrib.hf.LogitsProcessor(compile_schema(schema_with_formulas(fields)))]
                except Exception as exc:  # noqa: BLE001
                    grammar["errors"] += 1
                    grammar["last_error"] = repr(exc)[:300]
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=64 + 72 * len(fields), do_sample=False, **extra)
            s["seconds"] += time.time() - t0
            s["generations"] += 1
            s["new_tokens"] += int(out.shape[1] - ids.shape[1])
            text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            return text, parse_reply(text)

        return generate

    def save():
        for mode, s in stats.items():
            payload["modes"].setdefault(mode, {})["generation"] = {
                **s, "seconds_per_generation": round(s["seconds"] / max(s["generations"], 1), 2)}
        payload["seconds"] = round(time.time() - started, 1)
        (OUT / f"{out_name}.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
        (OUT / f"{out_name}.md").write_text(summarize(payload))

    if parts is None:
        parts = dev_parts()
    modes = [m for m in wanted if m == "free" or grammar["available"]] or ["free"]
    for mode in modes:
        generate = make_generate(mode)
        payload["modes"].setdefault(mode, {})["parts"] = {}
        for name, notes, only in parts:
            results = []
            for row in notes:
                fields = field_list(row, only=only)
                results.append((fields, answer_note(row, fields, generate, SYSTEM_FORMULA)))
            part = score_part(results)
            payload["modes"][mode]["parts"][name] = part
            log(f"{mode} {name}: first {part['first']}/{part['fields']}, after retry {part['after_retry']}, "
                f"accepted {part['accepted_right']}/{part['accepted']}, withheld {part['withheld']}")
            save()
    if not receipts:
        return payload

    # Real receipts, once, in the last mode.
    mode = modes[-1]
    generate = make_generate(mode)
    roots = sorted(path for path in INPUT.rglob("SROIE2019/test") if (path / "box").is_dir())
    receipt_rows, skipped = sroie.load_split(roots[0])
    first_items, final_items, still = [], [], 0
    for row in receipt_rows:
        fields = field_list(row)
        r = answer_note(row, fields, generate, SYSTEM_FORMULA)
        first_items += [(row, f, r["first"]) for f in fields]
        final_items += [(row, f, r["final"]) for f in fields]
        still += len(r["still"])
    first, final = score_sroie(first_items), score_sroie(final_items)
    for result in (first, final):
        result.pop("examples", None)  # SROIE may not be redistributed: no receipt-level output
    payload["sroie"] = {"mode": mode, "receipts": len(receipt_rows), "skipped": skipped, "first": first, "after_retry": final,
                        "withheld": still}
    log(f"sroie {mode}: first {first['overall']}, after retry {final['overall']}")
    save()
    log(f"done in {payload['seconds']} s")
    return payload


def dev_parts() -> list:
    """Exams 1-3, the three computed-field sets (seed 23) and the words notes. Never exam 4."""
    from experiments.augment import TEST_TEMPLATES, augment
    from experiments.chains import test_variants
    from experiments.evidence_data import load_rows

    dev = [r for r in load_rows() if r["split"] in ("exam1", "exam2", "exam3")]
    chains = test_variants(dev, seed=23)
    return [("computed single", [r for r in chains if r["chain_test"] == "single"], ["calc_0"]),
            ("computed combo_seen", [r for r in chains if r["chain_test"] == "combo_seen"], ["calc_0"]),
            ("computed combo_unseen", [r for r in chains if r["chain_test"] == "combo_unseen"], ["calc_0"]),
            ("exams1-3", dev, None),
            ("words", augment(dev, seed=23, templates=TEST_TEMPLATES, emits=0), None)]


def exam4_parts() -> list:
    """Exam 4 the same ways the final job scores our trained system: plain, words (seed 47), computed (seed 59)."""
    from experiments.augment import TEST_TEMPLATES, augment
    from experiments.chains import test_variants
    from experiments.evidence_data import load_rows

    exam4 = [r for r in load_rows() if r["split"] == "exam4"]
    if not exam4:
        raise RuntimeError("exam 4 is not in this bundle")
    chains = test_variants(exam4, seed=59)
    return [("exam4", exam4, None),
            ("exam4 words", augment(exam4, seed=47, emits=0, templates=TEST_TEMPLATES), None),
            ("exam4 computed single", [r for r in chains if r["chain_test"] == "single"], ["calc_0"]),
            ("exam4 computed combo_seen", [r for r in chains if r["chain_test"] == "combo_seen"], ["calc_0"]),
            ("exam4 computed combo_unseen", [r for r in chains if r["chain_test"] == "combo_unseen"], ["calc_0"])]


def summarize(payload: dict) -> str:
    lines = [f"# Formulas, the page check and structured output ({payload.get('prompt')})", "",
             f"Structured output: {payload.get('grammar')}", ""]
    for mode, data in payload["modes"].items():
        gen = data.get("generation", {})
        lines += [f"## {mode} ({gen.get('seconds_per_generation')} s per generation, {gen.get('generations')} generations)", "",
                  "| Part | First answer | After check and retry | Calculated right (after) | Accepted: right / given | Withheld (right) | "
                  "Earlier: LLM computes / steps / ours |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for name, p in data.get("parts", {}).items():
            prev = " / ".join(x for x in payload["previous"].get(name, ()) if x)
            lines.append(f"| {name} | {p['first']}/{p['fields']} | **{p['after_retry']}/{p['fields']}** | {p['calculated_right_after']} | "
                         f"{p['accepted_right']}/{p['accepted']} | {p['withheld']} ({p['withheld_right']}) | {prev} |")
        lines.append("")
    s = payload.get("sroie")
    if s:
        lines += [f"## SROIE test receipts ({s['receipts']}, mode {s['mode']})", "",
                  f"First answer {s['first']['overall']['right']}/{s['first']['overall']['fields']}, "
                  f"after retry {s['after_retry']['overall']['right']}/{s['after_retry']['overall']['fields']}, "
                  f"still failing the check {s['withheld']}. Earlier: {' / '.join(payload['previous']['sroie'])}.", ""]
        for field, row in s["after_retry"]["by_field"].items():
            lines.append(f"- {field}: {row['right']}/{row['fields']}")
    return "\n".join(lines) + "\n"
