# System One for extraction

Code for the blog post **[I trained a small model to read bills. The plain one did better.](https://omnaidu.com/blog/system-one-a-good-highlighter)** The post tells the whole story with examples. This README is the map of the code.

The goal: give a model a bill or a note, and get back exact values (vendor, amount due, due date, price after a discount) in one read, without a chat reply. The model is Gemma 4 E2B. Everything ran on free Kaggle T4 GPUs.

## Results in one table

| | Trained model (Gemma + LoRA + heads) | Plain Gemma, no training, formulas computed by code |
| --- | ---: | ---: |
| Exam 4, notes like the training notes (74 fields) | 63 | 63 |
| Exam 4, one-step calculations (24) | 9 | 23 |
| SROIE real receipts (1,362 fields) | 66 | 1,108 |

The trained model is a good highlighter on notes like its training notes. The plain model, writing JSON with formulas that code computes, is better on calculations and on real documents. Full numbers are in `outputs/experiments/*.md`.

## What is in the repo

| Folder | What it holds |
| --- | --- |
| `data/` | The synthetic notes: 208 training notes and four exams of 24. Answers are marked in brackets in the raw files. |
| `extractor/` | The auditor (`audit.py`), which strips the brackets, measures answers and computes the ones that are not printed. Also the first design, four heads on a frozen Gemma. |
| `experiments/` | Everything after that: the LoRA tagger, the read-and-combine code (`values.py`), the plain-LLM baselines (`structured_baseline.py`), formulas and the page check (`formula.py`), and the Kaggle jobs. |
| `tests/` | 116 tests. They run without a GPU. |
| `outputs/experiments/` | Results of every Kaggle run, and the two trained LoRA checkpoints. |

The main files, in the order of the blog post:

| Step in the post | Code | Results |
| --- | --- | --- |
| First attempt: four heads | `extractor/heads.py`, `extractor/train_heads.py` | (in git history) |
| Question order and the digit probe | `experiments/kaggle_job.py` | `question_order.md` |
| Pointing with LoRA, read and combine | `experiments/lora_tagger.py`, `experiments/values.py` | `lora_results.md`, `lora_onepass_results.md` |
| Values in words, emit head | `experiments/kaggle_lora_job.py` (`main_emit`) | `lora_emit_results.md` |
| Calculated values with steps | `experiments/chains.py` | `lora_chains_results.md` |
| Plain Gemma with JSON | `experiments/structured_baseline.py` | `structured_results.json` |
| Plain Gemma reads, code computes; real receipts | `experiments/kaggle_hybrid_job.py`, `experiments/sroie.py` | `hybrid_results.md`, `ours_sroie.json` |
| Formulas, page check, structured output | `experiments/formula.py`, `experiments/kaggle_formula_job.py` | `formula_results.md`, `formula_v2_results.md` |
| Exam 4, run once | `experiments/kaggle_head_to_head_job.py` | `head_to_head_results.md` |

## Running it

Tests, on any machine:

```bash
pip install torch numpy pytest
python -m pytest -q
```

Experiments run as Kaggle kernels. One command packs the code and the notes into a single script:

```bash
python experiments/make_kaggle_kernel.py OUT_DIR formula
kaggle kernels push -p OUT_DIR
```

The job names are listed at the top of `experiments/make_kaggle_kernel.py`. Only the `final` and `head_to_head` jobs include exam 4.

## Notes

- Exam 4 has now been used for the final comparison. Any further tuning needs a new held-out set.
- SROIE receipts are not in this repo. The receipt jobs read them from the Kaggle dataset `urbikn/sroie-datasetv2`, and a test checks that no receipt text is committed.
- The notes were written by Grok and the numbers come from one run each, so they are not production accuracy.
