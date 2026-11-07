"""Run the question-order experiment on a Modal GPU.

    modal run experiments/modal_app.py                       # all three layouts
    modal run experiments/modal_app.py --prompts first,last  # a subset

Needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in the environment, and HF_TOKEN if
the Gemma repo asks for a licence click. The GitHub workflow
`.github/workflows/modal-experiment.yml` sets those from repository secrets.

Gemma is downloaded once into the `gemma-cache` volume. States are kept in
memory for the run and not saved. Results come back as JSON and are written
to `outputs/experiments/`.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL = "google/gemma-4-E2B-it"
REMOTE = "/root/repo"

app = modal.App("evidence-question-order")
volume = modal.Volume.from_name("gemma-cache", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "safetensors", "huggingface_hub", "numpy", "timm", "pillow")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(ROOT / "data", f"{REMOTE}/data")
    .add_local_dir(ROOT / "extractor", f"{REMOTE}/extractor")
    .add_local_dir(ROOT / "experiments", f"{REMOTE}/experiments")
)


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _load_model(token):
    import torch
    import transformers

    errors = []
    for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
        loader = getattr(transformers, name, None)
        if loader is None:
            continue
        try:
            model = loader.from_pretrained(MODEL, token=token, torch_dtype=torch.bfloat16, device_map="cuda")
            _log(f"loaded {MODEL} with {name} as {type(model).__name__}")
            return model
        except Exception as exc:  # try the next auto class
            errors.append(f"{name}: {exc!r}"[:400])
    raise RuntimeError("could not load Gemma:\n" + "\n".join(errors))


@app.function(image=image, gpu="A10G", timeout=90 * 60, memory=32768, volumes={"/cache": volume}, secrets=[hf_secret])
def run(prompts: list[str]) -> dict:
    import torch
    import transformers

    sys.path.insert(0, REMOTE)
    os.chdir(REMOTE)
    from experiments.build_states import build
    from experiments.evidence_data import load_rows
    from experiments.evidence_heads import run_all

    started = time.time()
    token = os.environ.get("HF_TOKEN") or None
    _log(f"transformers {transformers.__version__}, torch {torch.__version__}, hf token {'set' if token else 'not set'}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, token=token)
    model = _load_model(token)
    model.eval()
    volume.commit()

    picked: dict = {}

    def forward(ids, add_bos):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states
        if hidden is None:
            raise RuntimeError("model returned no hidden states")
        if not picked:
            depth = len(hidden) - 1
            picked["layers"] = sorted({depth // 2, (3 * depth) // 4, depth})
            picked["depth"] = depth
            _log(f"{depth} layers; keeping hidden states {picked['layers']}")
        return torch.stack([hidden[i][0] for i in picked["layers"]]).to(torch.float16).cpu()

    rows = [row for row in load_rows() if row["split"] != "exam4"]
    _log(f"{len(rows)} notes (exam 4 is not loaded)")
    items_by_prompt, states_by_prompt, checks = {}, {}, {}
    for prompt in prompts:
        t0 = time.time()
        items, states, check = build(rows, tokenizer, forward, prompt, log=_log)
        check["seconds"] = round(time.time() - t0, 1)
        items_by_prompt[prompt], states_by_prompt[prompt], checks[prompt] = items, states, check
        _log(f"{prompt}: {len(items)} questions, {check}")

    del model
    torch.cuda.empty_cache()
    results = run_all(items_by_prompt, states_by_prompt, "cuda", log=_log)
    payload = {
        "model": MODEL,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "layers": picked,
        "build": checks,
        "seconds": round(time.time() - started, 1),
        "results": results,
    }
    Path("/cache/results").mkdir(parents=True, exist_ok=True)
    Path(f"/cache/results/question_order_{int(started)}.json").write_text(json.dumps(payload, indent=2, default=str))
    volume.commit()
    return json.loads(json.dumps(payload, default=str))


@app.local_entrypoint()
def main(prompts: str = "last_nobos,last,first") -> None:
    from experiments.report import summarize

    payload = run.remote([p.strip() for p in prompts.split(",") if p.strip()])
    out = ROOT / "outputs" / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    (out / "question_order.json").write_text(json.dumps(payload, indent=2) + "\n")
    summary = summarize(payload)
    (out / "question_order.md").write_text(summary)
    print(summary)
