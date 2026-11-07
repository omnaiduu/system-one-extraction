# 05 — Experiment 1: question order on frozen Gemma

Kaggle kernel `ommnaidu/gemma-e2b-question-order`, version 1. 2× Tesla T4, float16, transformers 5.22.2, torch 2.11.0+cu130. 2,365 s. Results: `outputs/experiments/question_order.{md,json}`. Code: `experiments/kaggle_job.py`, `build_states.py`, `evidence_heads.py`, `evidence_data.py`.

## Question

Two checks from the plan:

1. Can one state vector hold an amount's digits? (Is the old number head's design viable at all?)
2. Does putting the question before the note make frozen Gemma's note states pick the right instance?

## Labels

`experiments/evidence_data.py` reruns `audit.py` on the raw writer files instead of reading the stored audited files. This reproduces every stored row exactly (text and value; tested) and also yields evidence spans for `add_days` and `days_between`, which the older stored files lacked. Evidence for a field = the set of bracketed spans. A missing field has none. Forecasts are skipped.

With the real Gemma tokenizer, every gold span is exactly covered by its tokens (`covering_text_differs: 0`).

## Three prompt layouts

| Name | Prompt | Forwards |
| --- | --- | --- |
| `last_nobos` | `{note}\n\nQUESTION\n{field}: {instruction}` (all questions appended), no BOS token; the old layout | 1 per note (271; 57,768 tokens; 26 s) |
| `last` | same, with Gemma's BOS token | 1 per note |
| `first` | `QUESTION\n{field}: {instruction}\n\nDOCUMENT\n{note}`, BOS | 1 per question (1,309; 226,301 tokens; 120 s) |

Hidden states kept: layers 17, 26, 35 of 35, float16.

## The tagger (`experiments/evidence_heads.py`, `Tagger`)

- Per-layer LayerNorm (no affine), then a softmax-weighted mix of the 3 layers (separate mixes for note and question).
- LayerNorm, dropout 0.1, Linear 1536→128 for the note tokens and for the question vector.
- Features per token `[d, q, d*q]` (384), Conv1d kernel 3 → 128, GELU, dropout, Linear → 3 tags (outside / begin / inside).
- Loss: cross-entropy with class weights (1, 3, 3). Unweighted, the tagger learned "nothing here" and missed real spans. A planted-signal unit test caught this before any GPU time was spent.
- AdamW lr 1e-3, weight decay 0.01, batch 16, grad clip 1.0, seed 7.
- Decode: argmax tags → spans (an inside tag after outside opens a span).

## Protocol

- 5-fold cross-validation over the 208 training notes, grouped by note (seeded shuffle, seed 7), 40 epochs, scored every 5. The best epoch by CV exact is retrained on all 208 and scored once on exams 1–3.
- A field is exact only if the whole set of spans matches. Errors are bucketed: `partial` (overlaps but not exact), `wrong_place` (a marked span overlaps no gold span), `missed`, `false_alarm`.

## Results

| Layout | CV exact (of 1016) | CV wrong place | Dev exact (of 293) | Dev wrong place | Dev missed |
| --- | --- | --- | --- | --- | --- |
| `last_nobos` | 450 | 261 | **118** | 72 | 26 |
| `last` (BOS) | 416 | 321 | 111 | 88 | 23 |
| `first` | **472** | **232** | 106 | **40** | 81 |

CV curves (exact at epochs 5…40): `last_nobos` 265 → 450 (best at 40, the last epoch, so possibly undertrained); `first` peaked at 472 at epoch 35.

### Check 1, the digit probe

A linear readout (1536 → 10 places × 10 classes: sign, 7 integer digits, 2 fraction digits) from the one vector that has read both note and question. AdamW lr 1e-2, wd 0.1, 300 full-batch epochs, per layer, 5-fold CV over the 180 training number fields:

| Layout | Training folds | Held-out folds | Most-common-digit baseline |
| --- | --- | --- | --- |
| all three, every layer | 144 / 144 | **0 / 180** | 0 |

## What it means

- **The recall design is dead.** Perfect fit on training, nothing on held-out, at every layer. One vector does not hold an amount's digits in a readable form.
- **Question first did what was predicted, partly.** Decoy picks fell (dev 72 → 40) but misses rose (26 → 81), and net dev exact did not improve. The CV gain (+22 of 1016) is small enough to be noise.
- **The BOS guess was wrong.** Adding BOS hurt the old layout (450 → 416).
- **Every head on frozen states lands at about the same level** (four heads, highlighter, this tagger). With frozen Gemma, the prompt layout is not the bottleneck; the frozen representation is. That led to experiment 2.
