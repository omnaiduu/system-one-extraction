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
| `experiments/lora_tagger.py` | Experiments 2–5: encoding (per field or one pass, `groups_of`), the fit/held/dev split by source note, decode bias, frozen baselines, the LoRA training loop, `OpHead`, `SpanReader` and the emit heads, `field_confidence` and `selective`, final-answer scoring, Markdown summary. Independent of Gemma (tests use a fake model). |
| `experiments/values.py` | Read and combine: marked words → final answer, with the auditor's parsers and arithmetic; `ALLOWED` operations per type; answer matching. |
| `experiments/kaggle_lora_job.py` | Kaggle entry points: `main` (per field, experiment 3), `main_onepass` (experiment 4), `main_emit` (experiment 5, augmented notes plus readers). Loads Gemma, optional frozen baselines, injects LoRA (peft), trains, saves `{prefix}_results` and `{prefix}_adapter.pt`. |
| `experiments/make_kaggle_kernel.py` | Packs code and raw notes into one kernel script. Jobs: `question_order`, `lora`, `lora_onepass`, `lora_emit`, and `final SOURCE_KERNEL` (the only one with exam 4, plus the SROIE dataset and the model kernel as sources). |
| `experiments/augment.py` | Words and emit variants of audited rows, with every span shifted and every label exact; training and test phrasings split; `base_id` keeps variants with their note. |
| `experiments/chains.py` | Experiment 6: computed fields inserted into notes (one step and two-step chains), answers by independent formulas and never printed; training templates and three held-out test sets (single, seen combinations, unseen combinations). |
| `experiments/sroie.py` | SROIE receipts → rows: OCR lines as text, labels located by letters and digits, TOTAL-line preference, a wider date reader (flagged), three scores. |
| `experiments/kaggle_final_job.py` | The final grade: loads a trained model from a kernel output and scores exam 4 (plus its words/emit variants when readers exist) and the SROIE test split, once. The only bundle that contains exam 4. |
| `experiments/kaggle_cli.py` | Kaggle CLI wrapper for the cloud session (skips token introspection; see 04). |
| `tests/test_experiment.py` | Re-audit matches the stored rows; evidence spans are valid; BIO; prompt offsets; state building with a fake tokenizer; planted-signal tagger test; probe; end-to-end pipeline; kernel bundle excludes exam 4. |
| `tests/test_augment.py` | Words rendering; every augmented field reproduces its gold answer with gold spans and a perfect reader; spans stay on their words; emit answers are computed and hidden; exam phrasings never in training; variants share folds; a reader learns a planted value; reader gradients do not reach the adapters; emit training end to end. |
| `tests/test_chains.py` | Blocks exact; every generated field reproduces from gold marks and gold blocks; answers hidden; unseen combinations unseen; step head planted signal; chain training end to end; old op indexes kept; final summary. |
| `tests/test_sroie.py` | Box parsing, label location (typos skipped, punctuation ignored), TOTAL over SUB TOTAL, the wide date reader, skip counts, and only the final kernel bundling exam 4. |
| `tests/test_lora.py` | Encoding and split; decode bias; frozen + LoRA training end to end on a fake model; peft injection only into text attention; gold evidence reproduces all 1,309 gold answers; operation head end to end. |
| `.github/workflows/tests.yml` | Runs `pytest` on each push to the working branch. |
| `outputs/experiments/question_order.{md,json}` | Experiment 1 results (JSON includes up to 400 dev predictions per layout). |
| `outputs/experiments/lora_results.{md,json}`, `lora_adapter.pt` | Experiment 3 (per field, 10 epochs). Experiment 2's versions are in git at `67022fe`. |
| `outputs/experiments/lora_onepass_results.{md,json}`, `lora_onepass_adapter.pt` | Experiment 4, **the current best model**. |
| `outputs/experiments/lora_emit_results.*`, `lora_emit_detached_results.*` | Experiments 5a and 5b (no adapters committed; not candidates). |
| `outputs/experiments/lora_chains_results.*` | Experiment 6 (adapter on the kernel `ommnaidu/gemma-e2b-lora-chains`, not committed). |

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
python3 -m pytest -q        # 71 tests at the time of writing, about 45 s on CPU
```

Torch (CPU) and pytest are needed. `peft` is needed for `test_peft_injects_only_text_attention`.
