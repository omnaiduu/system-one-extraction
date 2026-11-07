"""LoRA evidence tagger, as it runs inside a Kaggle GPU kernel.

1. Load Gemma 4 E2B in float16 on one T4 (the recipe the earlier kernels used).
2. Frozen baselines on the shared split: the old layout (note, then question)
   and question first, each with a fresh `Tagger`.
3. Put LoRA adapters on the q/k/v/o projections of every text layer, then
   train adapters and a fresh `Tagger` together, question first.
4. Write `lora_results.json`, `lora_results.md`, and `lora_adapter.pt`
   (adapters + head, small) to /kaggle/working.

Exam 4 is not loaded.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

MODEL = "google/gemma-4-E2B-it"
OUT = Path("/kaggle/working")
RANK = 16
ALPHA = 32
EPOCHS = 5


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
    (OUT / "status.txt").write_text(message + "\n")


def text_tower(model):
    """The text decoder alone, so a forward does not build 262k-wide logits."""
    for name, module in model.named_modules():
        if type(module).__name__.endswith("TextModel"):
            return name, module
    return "", model


def attention_targets(model) -> list[str]:
    import torch

    linear = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    pattern = re.compile(r"\.(q_proj|k_proj|v_proj|o_proj)$")
    chosen = [n for n in linear if pattern.search(n) and "language_model" in n]
    if not chosen:
        chosen = [n for n in linear if pattern.search(n) and "vision" not in n and "audio" not in n]
    if not chosen:
        raise RuntimeError(f"no attention projections found; sample linear names: {linear[:40]}")
    return chosen


def main() -> None:
    import torch
    import transformers
    from peft import LoraConfig, inject_adapter_in_model

    from experiments.evidence_data import load_rows
    from experiments.evidence_heads import Tagger
    from experiments.lora_tagger import assign_parts, encode, run_frozen, summarize, train_lora

    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("no GPU in this Kaggle session")
    device = "cuda:0"
    log(f"transformers {transformers.__version__}, torch {torch.__version__}, gpu {torch.cuda.get_device_name(0)}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=device)
    model.eval()
    model.requires_grad_(False)
    tower_name, tower = text_tower(model)
    log(f"loaded {type(model).__name__}; text tower {tower_name or '(whole model)'} {type(tower).__name__}")

    def forward(ids):
        out = tower(input_ids=torch.tensor([ids], device=device), output_hidden_states=True, use_cache=False)
        if out.hidden_states is None:
            raise RuntimeError("no hidden states")
        return out.hidden_states

    rows = [row for row in load_rows() if row["split"] != "exam4"]
    probe = tokenizer(rows[0]["text"], add_special_tokens=True)["input_ids"]
    with torch.no_grad():
        hidden = forward(probe)
    depth = len(hidden) - 1
    layers = sorted({depth // 2, (3 * depth) // 4, depth})
    if not all(torch.isfinite(hidden[i]).all() for i in layers):
        raise FloatingPointError("float16 hidden states are not finite")
    log(f"{depth} layers, using {layers}; {len(rows)} notes")

    payload = {"model": MODEL, "dtype": "float16", "layers": layers, "results": {}}
    parts_by_prompt = {}
    for prompt in ("last_nobos", "first"):
        parts_by_prompt[prompt] = assign_parts(encode(rows, tokenizer, prompt))
    sizes = {name: len(part) for name, part in parts_by_prompt["first"].items()}
    payload["sizes"] = sizes
    log(f"split sizes {sizes}")

    for prompt in ("last_nobos", "first"):
        t0 = time.time()
        log(f"frozen {prompt}")
        result = run_frozen(parts_by_prompt[prompt], forward, layers, device, log=log)
        result["seconds"] = round(time.time() - t0, 1)
        payload["results"][f"frozen {prompt}"] = result
        log(f"frozen {prompt} done: held {result['held']['exact']}/{result['held']['fields']}")
    torch.cuda.empty_cache()
    # Save the baselines now, so a failure in the LoRA step cannot lose them.
    partial = json.loads(json.dumps(payload, default=str))
    (OUT / "lora_results.json").write_text(json.dumps(partial, indent=2) + "\n")
    (OUT / "lora_results.md").write_text(summarize(partial))

    targets = attention_targets(model)
    config = LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.05, target_modules=targets, bias="none")
    model = inject_adapter_in_model(config, model)
    adapter_params = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data = param.data.float()
            param.requires_grad_(True)
            adapter_params.append(param)
        else:
            param.requires_grad_(False)
    tower_name, tower = text_tower(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    payload["lora"] = {"modules": len(targets), "rank": RANK, "alpha": ALPHA,
                       "params": sum(p.numel() for p in adapter_params), "sample_targets": targets[:4]}
    log(f"lora on {len(targets)} modules, {payload['lora']['params']} trainable parameters")

    head = Tagger(len(layers), hidden[0].shape[-1]).to(device)
    t0 = time.time()
    result = train_lora(
        parts_by_prompt["first"], forward, adapter_params, head, layers, device,
        epochs=EPOCHS, set_train=lambda flag: model.train(flag), log=log,
    )
    best = result.pop("_best")
    result["seconds"] = round(time.time() - t0, 1)
    payload["results"]["lora first"] = result
    torch.save({"adapter_names": [n for n, p in model.named_parameters() if "lora_" in n],
                "adapter": best["adapter"], "head": best["head"], "layers": layers,
                "lora": payload["lora"], "bias": best["bias"]}, OUT / "lora_adapter.pt")

    payload["seconds"] = round(time.time() - started, 1)
    payload = json.loads(json.dumps(payload, default=str))
    (OUT / "lora_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "lora_results.md").write_text(summarize(payload))
    log(f"done in {payload['seconds']} s")
