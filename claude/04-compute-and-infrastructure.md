# 04 — Compute and infrastructure

## Where things run

| Piece | Where | Why |
| --- | --- | --- |
| Code, tests, data labelling | Claude cloud session (this container) | Edits, `pytest`, the auditor. No GPU, no Gemma checkpoint. |
| Tests on every push | GitHub Actions, `.github/workflows/tests.yml` | CPU-only torch; no secrets. |
| Gemma forwards and training | **Kaggle kernels**, free T4 GPUs | The user has no card for paid GPU providers. Kaggle needs only phone verification. |

## The road to Kaggle (and the detours)

1. **Kaggle from the session**: blocked. The session's network policy only allowed package registries.
2. **Modal through GitHub Actions**: the workflow ran tests, then `modal run`. Problems in order:
   - The pasted secrets ended with a newline; Modal rejected the metadata. Fixed by stripping whitespace in the workflow.
   - "Workspace is paused" (user fixed in the Modal dashboard).
   - "Please add a payment method to use A10G GPU functions." The user is in India and could not get a card. No money was spent.
   - A CPU-only Modal run was started (CPU did not need a card) and then cancelled at the user's request in favour of Kaggle.
3. **Kaggle, directly from the session**: a personal cloud environment `system-one-extraction` was created (user-approved) with these allowed hosts:
   - `www.kaggle.com`, `api.kaggle.com` (the API),
   - `storage.googleapis.com` (uploads),
   - `www.kaggleusercontent.com` (kernel output downloads; added later when the first download was refused).

   The Kaggle token was saved as a **network secret** for `api.kaggle.com` through the secure form, sent as `Authorization: Bearer`. The session never sees the value. A token the user had pasted in chat was not used, and the user was told to revoke it.

Budget rules that came out of the Modal detour (kept in mind for any paid provider): cap each run's timeout, never let an ordinary push start paid compute, and check what a run costs before starting it.

## The Kaggle CLI wrapper

`experiments/kaggle_cli.py`. The CLI checks the token before any command by sending it in the request body ("introspection"). The proxy only fills in the header, so the check fails with a placeholder. The wrapper skips that one check; real calls carry the real token in the header.

```bash
cd /tmp   # or use -I, so nothing in the repo shadows a library module
KAGGLE_API_TOKEN=KGAT_placeholder python3 -I /path/to/repo/experiments/kaggle_cli.py kernels list --mine
```

## How a kernel is built and run

`experiments/make_kaggle_kernel.py OUT_DIR JOB` writes `OUT_DIR/kernel.py` and `kernel-metadata.json`:

- The script embeds a base64 tar of `extractor/`, `experiments/`, and every `data/*/raw/*.jsonl` **except exam 4** (about 170 KB).
- On Kaggle it runs `pip install -U transformers accelerate peft` (Kaggle's image predates Gemma 4), uninstalls `torchao` (Kaggle ships 0.10; current peft refuses to load next to it), unpacks the tar, and calls the job's `main()`.
- Kernel settings: private, GPU on, internet on (for Hugging Face), `machine_shape: NvidiaTeslaT4` (2× T4, 15 GB each).
- Gemma loads from Hugging Face without a token, in float16, with `device_map` on one GPU. This is the recipe the older kernels in `kaggle/` used. Experiment 1 checks the first real note's states for inf/NaN and would reload in float32 across both GPUs. float16 stayed finite in every run.

| Job | Kernel id | Module |
| --- | --- | --- |
| `question_order` | `ommnaidu/gemma-e2b-question-order` | `experiments/kaggle_job.py` |
| `lora` | `ommnaidu/gemma-e2b-lora-evidence` | `experiments/kaggle_lora_job.py` |

Run and collect:

```bash
S=/tmp/kernel_out
python3 experiments/make_kaggle_kernel.py $S lora
cd /tmp
KAGGLE_API_TOKEN=KGAT_placeholder python3 -I REPO/experiments/kaggle_cli.py kernels push -p $S
KAGGLE_API_TOKEN=KGAT_placeholder python3 -I REPO/experiments/kaggle_cli.py kernels status ommnaidu/gemma-e2b-lora-evidence
KAGGLE_API_TOKEN=KGAT_placeholder python3 -I REPO/experiments/kaggle_cli.py kernels output ommnaidu/gemma-e2b-lora-evidence -p /tmp/out
```

The output includes the kernel log (a JSON list of stream lines), the result JSON and Markdown, and `lora_adapter.pt`. Copy results into `outputs/experiments/` and commit them.

## Costs

Everything that produced a result ran on Kaggle's free GPU quota (30 GPU hours a week). Used so far: experiment 1 about 40 minutes, experiment 2 about 11 minutes (failed attempt) plus 50 minutes, experiment 3 about 1.5 hours. The Modal attempts spent no GPU time.

## Things that are easy to break

- `kernel-metadata.json` must stay `is_private: true`.
- If the job ever loads exam 4, the final grade is spent. `tests/test_experiment.py::test_kaggle_kernel_bundle_unpacks_without_exam4` checks the bundle.
- Kaggle's preinstalled package versions change. If a kernel fails on import, read the log first; the torchao failure looked like a peft bug.
- Save partial results before risky steps. The first LoRA attempt lost its baselines because nothing was written before the crash. The job now writes them first.
