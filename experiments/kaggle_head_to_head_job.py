"""The final head-to-head on exam 4: run once, nothing changed afterwards.

Exam 4 (24 notes) has never been loaded by any earlier run. Everything here is
frozen before it runs:

- the untrained LLM with formulas, structured output (xgrammar), the page
  check and one retry, with prompt version 1 (`SYSTEM_FORMULA`, written
  before any formula result) and version 2 (`SYSTEM_FORMULA_V2`, two rules
  added after reading errors on the dev sets). Exam 4 answers whether those
  rules help on notes nobody looked at;
- our trained systems as they were selected: experiment 4 (one pass, best
  overall) and experiment 6 (chains, with the step head), loaded from their
  kernels' checkpoints.

Parts, the same for both: exam 4 as written, exam 4 with numbers and dates in
words (seed 47), and exam 4 with one computed field per note in the three
experiment-6 kinds (seed 59). No receipts (both systems already have them).
Writes head_to_head_results.{json,md}.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

OUT = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
CHECKPOINTS = {"experiment 4 (one pass)": "*onepass_adapter.pt", "experiment 6 (chains)": "*chains_adapter.pt"}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
    (OUT / "status.txt").write_text(message + "\n")


def main() -> None:
    import torch
    import transformers

    from experiments.kaggle_final_job import score_checkpoint
    from experiments.kaggle_formula_job import MODEL, exam4_parts, run

    started = time.time()
    payload = {"llm": {}, "ours": {}}

    def save():
        payload["seconds"] = round(time.time() - started, 1)
        (OUT / "head_to_head_results.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
        (OUT / "head_to_head_results.md").write_text(summarize(payload))

    parts = exam4_parts()
    payload["notes"] = {name: len(notes) for name, notes, _ in parts}
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda:0")
    model.eval()
    model.requires_grad_(False)
    for prompt in ("SYSTEM_FORMULA", "SYSTEM_FORMULA_V2"):
        result = run(["constrained"], prompt, f"exam4_{prompt.lower()}", parts=parts, receipts=False,
                     model=model, tokenizer=tokenizer, previous={})
        payload["llm"][prompt] = {"grammar": result.get("grammar"), "modes": result["modes"]}
        save()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    for name, pattern in CHECKPOINTS.items():
        found = sorted(INPUT.rglob(pattern))
        if len(found) != 1:
            payload["ours"][name] = {"error": f"expected one {pattern}, found {[str(f) for f in found]}"}
            save()
            continue
        try:
            payload["ours"][name] = score_checkpoint(found[0], receipts=False)
        except Exception as exc:  # noqa: BLE001 - recorded; the LLM half is already saved
            payload["ours"][name] = {"error": repr(exc)[:500]}
        gc.collect()
        torch.cuda.empty_cache()
        save()
    log(f"done in {payload['seconds']} s")


def _ours_row(ours: dict, part: str):
    """Our system's score on one exam-4 part, in the same 'right/fields' form."""
    if "error" in ours:
        return "error"
    if part == "exam4":
        v = ours["exam4"]["values"]
        return f"{v['exact']}/{v['fields']}"
    if part == "exam4 words":
        v = ours.get("exam4_augmented", {}).get("words")
        return f"{v['exact']}/{v['fields']}" if v else "-"
    kind = part.replace("exam4 computed ", "")
    c = ours.get("exam4_chains", {}).get(kind)
    return f"{c['computed']['exact']}/{c['computed']['fields']}" if c else "-"


def summarize(payload: dict) -> str:
    lines = ["# Exam 4 head-to-head (run once)", "",
             "LLM: untrained Gemma 4 E2B, formulas, structured output, page check and one retry. "
             "Ours: the trained systems as selected. Nothing was changed after this run.", "",
             "| Part | LLM, prompt v1 (first / after retry / accepted right of given) | LLM, prompt v2 (same) | "
             + " | ".join(payload.get("ours", {}).keys()) + " |",
             "| --- | --- | --- | " + " | ".join("---" for _ in payload.get("ours", {})) + " |"]
    names = list(payload.get("notes", {}))
    for part in names:
        cells = []
        for prompt in ("SYSTEM_FORMULA", "SYSTEM_FORMULA_V2"):
            p = payload["llm"].get(prompt, {}).get("modes", {}).get("constrained", {}).get("parts", {}).get(part)
            cells.append(f"{p['first']} / **{p['after_retry']}**/{p['fields']} / {p['accepted_right']} of {p['accepted']}" if p else "-")
        cells += [_ours_row(o, part) for o in payload.get("ours", {}).values()]
        lines.append(f"| {part} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
