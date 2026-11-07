# 08 — File map

## Added in this session

| File | What it is |
| --- | --- |
| `claude/` | These notes. |
| `experiments/evidence_data.py` | Reruns the auditor on the raw writer files; turns every field into an evidence question (spans, operation); prompt layouts (`last_nobos`, `last`, `first`); char-span → token mapping; BIO tags and decoding; error buckets. Exam 4 is listed so a full cache can be built, but no job scores it. |
| `experiments/build_states.py` | Experiment 1: runs a `forward` over every note in a layout and keeps the note and question states. |
| `experiments/evidence_heads.py` | The `Tagger` head; training; grouped 5-fold CV; scoring; the digit probe (check 1). |
| `experiments/report.py` | Experiment 1's Markdown summary. |
| `experiments/kaggle_job.py` | Experiment 1's Kaggle entry point (load Gemma, build states for 3 layouts, run CV and the probe). |
| `experiments/lora_tagger.py` | Experiments 2–3: encoding, the fit/held/dev split, decode bias, frozen baselines, LoRA training loop, `OpHead`, final-answer scoring, Markdown summary. Independent of Gemma (tests use a fake model). |
| `experiments/values.py` | Read and combine: marked words → final answer, with the auditor's parsers and arithmetic; `ALLOWED` operations per type; answer matching. |
| `experiments/kaggle_lora_job.py` | Experiments 2–3's Kaggle entry point: load Gemma, optional frozen baselines, inject LoRA (peft), train, save. `EPOCHS`, `RUN_FROZEN`, `RANK`, `ALPHA` at the top. |
| `experiments/make_kaggle_kernel.py` | Packs code and raw notes (no exam 4) into one kernel script, for job `question_order` or `lora`. |
| `experiments/kaggle_cli.py` | Kaggle CLI wrapper for the cloud session (skips token introspection; see 04). |
| `tests/test_experiment.py` | Re-audit matches the stored rows; evidence spans are valid; BIO; prompt offsets; state building with a fake tokenizer; planted-signal tagger test; probe; end-to-end pipeline; kernel bundle excludes exam 4. |
| `tests/test_lora.py` | Encoding and split; decode bias; frozen + LoRA training end to end on a fake model; peft injection only into text attention; gold evidence reproduces all 1,309 gold answers; operation head end to end. |
| `.github/workflows/tests.yml` | Runs `pytest` on each push to the working branch. |
| `outputs/experiments/question_order.{md,json}` | Experiment 1 results (JSON includes up to 400 dev predictions per layout). |
| `outputs/experiments/lora_results.{md,json}` | Experiment 2 results; to be replaced by experiment 3's. |
| `outputs/experiments/lora_adapter.pt` | LoRA adapters + tagger head (+ op head from experiment 3), with the parameter names and the chosen bias. |

## Older files this work relies on

| File | Role now |
| --- | --- |
| `extractor/audit.py` | The labeler. Its parsers and arithmetic are reused for final answers. |
| `extractor/labels.py` | `split_number` (digit targets for the probe). The old `pack_prompt` is the source of the `last_nobos` layout. |
| `extractor/heads.py`, `train_heads.py`, `score_heads.py`, `gemma_cache.py` | The four-head system, rounds 1–4. Kept for reference; not used by the new experiments. |
| `data/*/raw/*.jsonl` | Writer output with brackets: the source of all labels. |
| `data/*/audited.jsonl`, `locked.jsonl` | Stored audited rows; the re-audit must match them (tested). |
| `kaggle/` | The older kernels; their Gemma loading recipe (float16 from Hugging Face, transformers upgraded) was reused. |
| `blogs/mark-and-measure.md` | The data method and rounds 1–4. |

## Tests

```bash
python3 -m pytest -q        # 36 tests at the time of writing, about 15 s on CPU
```

Torch (CPU) and pytest are needed. `peft` is needed for `test_peft_injects_only_text_attention`.
