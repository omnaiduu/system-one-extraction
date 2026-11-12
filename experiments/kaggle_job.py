"""The question-order experiment, as it runs inside a Kaggle GPU kernel.

`experiments/make_kaggle_kernel.py` packs this repo's code and data into one
kernel script. On Kaggle that script unpacks them and calls `main()`, which:

1. loads Gemma 4 E2B from Hugging Face (float16, the recipe the earlier
   kernels in `kaggle/` used; float32 across both T4s if float16 overflows),
2. builds hidden states for every note in each prompt layout,
3. trains and scores the evidence tagger and the digit probe,
4. writes `question_order.json` and `question_order.md` to /kaggle/working.

Exam 4 is never loaded.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

MODEL = "google/gemma-4-E2B-it"
OUT = Path("/kaggle/working")
PROMPTS = ("last_nobos", "last", "first")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
    (OUT / "status.txt").write_text(message + "\n")


def load_model(dtype_name: str):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, dtype_name)
    device_map = "cuda" if dtype_name == "float16" else "auto"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, device_map=device_map)
    model.eval()
    log(f"loaded {MODEL} as {type(model).__name__} in {dtype_name}, device_map={device_map}")
    return model


def main() -> None:
    import torch
    import transformers

    from experiments.build_states import build
    from experiments.evidence_data import load_rows
    from experiments.evidence_heads import run_all
    from experiments.report import summarize

    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("no GPU in this Kaggle session")
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    log(f"transformers {transformers.__version__}, torch {torch.__version__}, gpus {gpus}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)

    state = {"model": load_model("float16"), "dtype": "float16"}
    picked: dict = {}

    def raw_forward(ids):
        model = state["model"]
        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([ids], device=model.device),
                output_hidden_states=True,
                use_cache=False,
            )
        if out.hidden_states is None:
            raise RuntimeError("model returned no hidden states")
        return out.hidden_states

    def forward(ids, add_bos):
        hidden = raw_forward(ids)
        if not picked:
            depth = len(hidden) - 1
            picked["layers"] = sorted({depth // 2, (3 * depth) // 4, depth})
            picked["depth"] = depth
            log(f"{depth} layers; keeping hidden states {picked['layers']}")
        stacked = torch.stack([hidden[i][0].float() for i in picked["layers"]])
        if not torch.isfinite(stacked).all():
            raise FloatingPointError("non-finite hidden states")
        return stacked.to(torch.float16).cpu()

    # float16 can overflow inside Gemma. Check on a real note before the run,
    # and fall back to float32 (sharded over the GPUs) if it does.
    rows = [row for row in load_rows() if row["split"] != "exam4"]
    probe_text = rows[0]["text"]
    ids = tokenizer(probe_text, add_special_tokens=True)["input_ids"]
    try:
        forward(ids, True)
    except FloatingPointError:
        log("float16 overflowed; reloading in float32")
        del state["model"]
        torch.cuda.empty_cache()
        state["model"], state["dtype"] = load_model("float32"), "float32"
        forward(ids, True)

    log(f"{len(rows)} notes (exam 4 is not loaded)")
    items_by_prompt, states_by_prompt, checks = {}, {}, {}
    for prompt in PROMPTS:
        t0 = time.time()
        items, states, check = build(rows, tokenizer, forward, prompt, log=log)
        check["seconds"] = round(time.time() - t0, 1)
        items_by_prompt[prompt], states_by_prompt[prompt], checks[prompt] = items, states, check
        log(f"{prompt}: {len(items)} questions, {check}")

    del state["model"]
    torch.cuda.empty_cache()
    results = run_all(items_by_prompt, states_by_prompt, "cuda", log=log)
    payload = {
        "model": MODEL,
        "dtype": state["dtype"],
        "gpus": gpus,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "layers": picked,
        "build": checks,
        "seconds": round(time.time() - started, 1),
        "results": results,
    }
    payload = json.loads(json.dumps(payload, default=str))
    (OUT / "question_order.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "question_order.md").write_text(summarize(payload))
    log(f"done in {payload['seconds']} s")


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    main()
