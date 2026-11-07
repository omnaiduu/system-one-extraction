# 07 — Experiment 3: final answers, 10 epochs

Kaggle kernel `ommnaidu/gemma-e2b-lora-evidence`, **version 3**. One T4, float16, 4,675 s. Results: `outputs/experiments/lora_results.{md,json}` and `lora_adapter.pt` (these replaced experiment 2's files, which stay in git history at commit `67022fe`).

## What changed from experiment 2

1. **Final answers, not only evidence.** `experiments/values.py`:
   - *Read*: the marked words go through the auditor's own parsers (`parse_number`, `parse_date`, `parse_time`, `parse_datetime`).
   - *Combine*: the operation runs exactly as the auditor computed the gold label. Sum of all parts; product of two; subtract = first amount in the text minus the second; add_days = the one part that parses as a date plus the one that parses as a count; days_between = later date minus earlier; median of the parts; copy and list return the marked text.
   - *Missing*: no marked spans means "missing".
   - **Check**: from gold evidence, the step reproduces **all 1,309 gold answers** outside exam 4 (`tests/test_lora.py::test_gold_evidence_reproduces_every_gold_answer`). This includes all 49 subtractions, which confirms the text-order rule. So any final-answer error comes from wrong marks or a wrong operation, never from the arithmetic.
2. **Operation head** (`OpHead` in `lora_tagger.py`). It reads the last question token and the last prompt token (question first, the last prompt token has read the whole note): layer mix, LayerNorm, Linear 3072→128, GELU, dropout, Linear → 12 operations. It is trained jointly, cross-entropy on non-missing fields. At decode time only the operations legal for the field's type are allowed (`ALLOWED` in `values.py`): a number can be read, summed, subtracted or multiplied; a date can be read or computed with add_days; every other type has one operation.
3. **Selection by final answers.** The epoch and the decode bias are the ones with the most right final answers on `held`, ties broken by exact evidence.
4. **10 epochs** (experiment 2 was still improving at epoch 5).
5. **No frozen baselines** this time (`RUN_FROZEN = False`). They were measured in experiment 2 on the same split.

Everything else (LoRA r 16, alpha 32, learning rates, accumulation, precision, split) is as in experiment 2.

## Known limit

The parsers read digits and the date formats the auditor accepts. A number written in words ("fourteen hundred") would not parse, and the field would be wrong. No field in this data is written in words, so this run cannot measure that case. The fix is the augmentation step in 03/10, followed by a learned reader if the parsers cannot cover it.

## Comparisons to make when it lands

Final answers on exams 1–3 against:

| System | Exam 1 | Exam 2 | Exam 3 |
| --- | --- | --- | --- |
| Four heads (round 1 / 2 / 3; each scored on its own new exam) | 29/112 | 14/95 | 15/95 |
| Highlighter with decode rules (final values) | 48/109 | 42/92 | 49/92 |
| **LoRA, final answers (this run)** | **83/109** | **72/92** | **71/92** |

(Field counts differ for the four heads because their exams also counted forecasts.)

Also report: operation-choice accuracy, final answers by operation, training-sample fit (to watch memorization), and the held-out curve over 10 epochs.

## Results

| | Training sample (300) | Held (193) | Exam 1 | Exam 2 | Exam 3 | **Exams 1–3** |
| --- | --- | --- | --- | --- | --- | --- |
| Evidence exact | 290 | 168 | 82/109 | 72/92 | 70/92 | 224/293 (76%) |
| **Final answers** | 290 | 167 | **83/109** | **72/92** | **71/92** | **226/293 (77%)** |

Best epoch 9 of 10, decode bias −0.5. Operation choice on exams 1–3 (number and date fields, where more than one operation is legal): **102/103**. Error buckets for evidence on exams 1–3: wrong place 24, missed 9, partial 36 (experiment 2: 27, 23, 47).

Held-out right final answers by epoch: 36, 133, 84, 145, 162, 148, 160, 159, **167**, 165. Training loss: 1.07, 0.107, 0.031, 0.017, 0.009, 0.006, 0.003, 0.007, 0.002, 0.004. The curve is noisy (epoch 3 dipped to 84), so selecting the epoch on the held-out fold mattered.

Final answers on exams 1–3 by operation: missing 29/29, product 11/11, read_datetime 3/3, read_date 57/60, days_between 13/14, read_number 8/9, subtract 6/7, sum 7/10, median 6/9, read_time 2/3, copy 62/95, list 20/37, add_days 2/6.

### What it means

- **Against the marking test**, the earlier best, on the same exams and the same final-value metric: 226 against 139 of 293, without any hand-written decode rule. Exam by exam: 83 vs 48, 72 vs 42, 71 vs 49.
- **Against the four heads**, which scored 14/95 and 15/95 on exams 2 and 3: those exams are now at 72/92 and 71/92.
- **The operation head is nearly perfect** (102/103). Errors are almost all in which words get marked.
- **The arithmetic paths work**: products 11/11, day counts 13/14, sums 7/10.
- **Weak spots**: due dates (add_days 2/6), lists (20/37) and copied text (62/95). Copied text is a third of all fields, so it is the biggest pool of remaining errors.
- **Memorization check**: training-sample fit rose from 273 to 290 of 300 with 10 epochs. Held-out also rose (160 → 167), and the training/new-note gap is still far smaller than for any frozen model. Worth watching with more epochs.
- Final answers are slightly above evidence exact (226 vs 224) because a few marks that are off by a token still read to the right value.
