# 07 — Experiment 3: final answers, 10 epochs

Kaggle kernel `ommnaidu/gemma-e2b-lora-evidence`, **version 3**. **Status when this note was written: running.** When it finishes, copy `lora_results.{json,md}` and `lora_adapter.pt` into `outputs/experiments/` (they replace experiment 2's files, which stay in git history at commit `67022fe`) and fill in the results section below.

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
| **LoRA, final answers** | _pending_ | _pending_ | _pending_ |

(Field counts differ for the four heads because their exams also counted forecasts.)

Also report: operation-choice accuracy, final answers by operation, training-sample fit (to watch memorization), and the held-out curve over 10 epochs.

## Results

_Pending._
