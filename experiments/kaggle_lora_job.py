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
EPOCHS = 10
# The frozen baselines were measured in the previous run (outputs/experiments/lora_results.md).
RUN_FROZEN = False


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
    """Experiment 3: question first, one forward per field, 10 epochs, final answers."""
    run_job(layout="first", epochs=EPOCHS, prefix="lora", run_frozen=RUN_FROZEN)


def main_onepass() -> None:
    """Experiment 4: every field in one forward per note (like Clairic), with confidence."""
    run_job(layout="multi", epochs=EPOCHS, prefix="lora_onepass", run_frozen=False)


def main_emit() -> None:
    """Experiment 5: one pass, plus the learned reader and the emit head, on augmented notes."""
    run_job(layout="multi", epochs=EPOCHS, prefix="lora_emit", run_frozen=False, emit=True)


def main_chains() -> None:
    """Experiment 6: one pass, plus computed fields (one-step operations and two-step chains)."""
    run_job(layout="multi", epochs=EPOCHS, prefix="lora_chains", run_frozen=False, chains=True)


def main_grammar() -> None:
    """Experiment 7: computed fields from a sentence grammar, a step head that sees one operand,
    marks snapped to whole numbers, and plain Gemma reading words the rules cannot."""
    run_job(layout="multi", epochs=EPOCHS, prefix="lora_grammar", run_frozen=False, grammar=True)


def grammar_rows(rows: list[dict]) -> list[dict]:
    """Training notes; two grammar copies of each (one single-step and one two-step field);
    words copies of half of them, so the tagger learns to mark numbers and dates written in
    words (reading them is Gemma's job at test time). Exams 1-3 four times each with one
    computed field (dev_single, dev_combo_seen, dev_combo_unseen, dev_combo_unseen_familiar),
    and once with words (dev_words)."""
    import random as _random

    from experiments.augment import TEST_TEMPLATES, augment
    from experiments.chains import grammar_test_variants, grammar_training_variants

    train = [row for row in rows if row["split"] == "train"]
    dev = [row for row in rows if row["split"] in ("exam1", "exam2", "exam3")]
    half = sorted(_random.Random(5).sample(range(len(train)), len(train) // 2))
    out = list(rows) + grammar_training_variants(train, seed=11, copies=2)
    out += augment([train[i] for i in half], seed=13, words=1, emits=0)
    out += [{**row, "split": f"dev_{row['chain_test']}"} for row in grammar_test_variants(dev, seed=23)]
    out += [{**row, "split": "dev_words"} for row in augment(dev, seed=29, words=1, emits=0, templates=TEST_TEMPLATES)]
    return out


def chain_rows(rows: list[dict]) -> list[dict]:
    """Training notes plus two copies each with one single-step and one two-step field
    (training phrasings); exams 1-3 three times each with a held-out phrasing, as
    dev_single, dev_combo_seen and dev_combo_unseen."""
    from experiments.chains import test_variants, training_variants

    train = [row for row in rows if row["split"] == "train"]
    dev = [row for row in rows if row["split"] in ("exam1", "exam2", "exam3")]
    out = list(rows) + training_variants(train, seed=11, copies=2)
    out += [{**row, "split": f"dev_{row['chain_test']}"} for row in test_variants(dev, seed=23)]
    return out


def augmented_rows(rows: list[dict]) -> list[dict]:
    """Training notes plus their words and emit variants (training phrasings);
    exams 1-3 plus words and emit variants with phrasings never used in training,
    as the dev parts dev_words and dev_emit."""
    from experiments.augment import TEST_TEMPLATES, TRAIN_TEMPLATES, augment

    train = [row for row in rows if row["split"] == "train"]
    dev = [row for row in rows if row["split"] in ("exam1", "exam2", "exam3")]
    out = list(rows)
    out += augment(train, seed=11, templates=TRAIN_TEMPLATES)
    for row in augment(dev, seed=23, templates=TEST_TEMPLATES):
        out.append({**row, "split": f"dev_{row['augment']}"})
    return out


def run_job(layout: str, epochs: int, prefix: str, run_frozen: bool, emit: bool = False, chains: bool = False,
            grammar: bool = False) -> None:
    import torch
    import transformers
    from peft import LoraConfig, inject_adapter_in_model

    from experiments.evidence_data import load_rows
    from experiments.evidence_heads import Tagger
    from experiments.lora_tagger import OpHead, StepHead, assign_parts, encode, make_readers, summarize, train_lora
    from experiments.lora_tagger import run_frozen as run_frozen_baselines  # not `run_frozen`: that name is the flag

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
    if emit:
        rows = augmented_rows(rows)
    if chains:
        rows = chain_rows(rows)
    if grammar:
        rows = grammar_rows(rows)
    probe = tokenizer(rows[0]["text"], add_special_tokens=True)["input_ids"]
    with torch.no_grad():
        hidden = forward(probe)
    depth = len(hidden) - 1
    layers = sorted({depth // 2, (3 * depth) // 4, depth})
    if not all(torch.isfinite(hidden[i]).all() for i in layers):
        raise FloatingPointError("float16 hidden states are not finite")
    log(f"{depth} layers, using {layers}; {len(rows)} notes")

    payload = {"model": MODEL, "dtype": "float16", "layers": layers, "layout": layout, "results": {}}
    parts_by_prompt = {}
    for prompt in (("last_nobos", "first") if run_frozen else ()) + (layout,):
        parts_by_prompt[prompt] = assign_parts(encode(rows, tokenizer, prompt))
    sizes = {name: len(part) for name, part in parts_by_prompt[layout].items()}
    payload["sizes"] = sizes
    payload["forwards_per_epoch"] = len({ex["group"] for ex in parts_by_prompt[layout]["fit"]})
    log(f"split sizes {sizes}; {payload['forwards_per_epoch']} training forwards per epoch")

    for prompt in (("last_nobos", "first") if run_frozen else ()):
        t0 = time.time()
        log(f"frozen {prompt}")
        result = run_frozen_baselines(parts_by_prompt[prompt], forward, layers, device, log=log)
        result["seconds"] = round(time.time() - t0, 1)
        payload["results"][f"frozen {prompt}"] = result
        log(f"frozen {prompt} done: held {result['held']['exact']}/{result['held']['fields']}")
    torch.cuda.empty_cache()
    # Save the baselines now, so a failure in the LoRA step cannot lose them.
    partial = json.loads(json.dumps(payload, default=str))
    (OUT / f"{prefix}_results.json").write_text(json.dumps(partial, indent=2) + "\n")
    (OUT / f"{prefix}_results.md").write_text(summarize(partial))

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
    op_head = OpHead(len(layers), hidden[0].shape[-1]).to(device)
    readers = None
    if emit:
        readers = {name: r.to(device) for name, r in make_readers(len(layers), hidden[0].shape[-1]).items()}
    step_head = None
    if chains:
        step_head = StepHead(len(layers), hidden[0].shape[-1]).to(device)
    if grammar:
        step_head = StepHead(len(layers), hidden[0].shape[-1], pair_context=False, step_embedding=False).to(device)
    rewriter = None
    if grammar:
        from experiments.rewrite import make_rewriter

        rewrite_stats: dict = {}
        rewriter = make_rewriter(model, tokenizer, device, rewrite_stats)
        log(f"rewriter ready; smoke test: {rewriter('one thousand two hundred and fifty', 'number')} "
            f"{rewriter('the second of April, 2026', 'date')} {rewrite_stats}")
    t0 = time.time()
    result = train_lora(
        parts_by_prompt[layout], forward, adapter_params, head, layers, device,
        epochs=epochs, set_train=lambda flag: model.train(flag), op_head=op_head, readers=readers, step_head=step_head,
        snap=grammar, rewriter=rewriter, log=log,
    )
    if rewriter is not None:
        payload["rewrite_stats"] = rewrite_stats
    best = result.pop("_best")
    result["seconds"] = round(time.time() - t0, 1)
    payload["results"][f"lora {layout}, {epochs} epochs"] = result
    torch.save({"adapter_names": [n for n, p in model.named_parameters() if "lora_" in n],
                "adapter": best["adapter"], "head": best["head"], "op_head": best["heads"][1], "layers": layers,
                "readers": dict(zip(readers, best["heads"][2 : 2 + len(readers)])) if readers else None,
                "step_head": best["heads"][-1] if step_head is not None else None,
                "step_head_config": step_head.config if step_head is not None else None, "snap": grammar,
                "lora": payload["lora"], "bias": best["bias"], "layout": layout}, OUT / f"{prefix}_adapter.pt")

    payload["seconds"] = round(time.time() - started, 1)
    payload = json.loads(json.dumps(payload, default=str))
    (OUT / f"{prefix}_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / f"{prefix}_results.md").write_text(summarize(payload))
    log(f"done in {payload['seconds']} s")
