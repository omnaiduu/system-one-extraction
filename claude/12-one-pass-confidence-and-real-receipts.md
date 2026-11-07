# 12 — One pass, confidence, and real receipts

The user approved five steps: final answers, exam 4, all fields in one pass, confidence scores, real documents. They were reordered for one reason: **exam 4 can be used only once**. Grading it before the model is final would leave no clean exam for the final model (the writer is out of tokens). So the order became: final answers (07) → one pass + confidence (this note) → choose the final model on exams 1–3 and the held-out fold → final grade of exam 4 and SROIE together, once.

## One pass (experiment 4, kernel `ommnaidu/gemma-e2b-lora-onepass`)

Clairic answers every field in one parallel pass, splitting into several passes only when the fields do not fit. Experiments 2–3 used one forward per field. The `multi` layout (`experiments/evidence_data.py`, `build_prompt`) is:

```
QUESTIONS
fee: Printed charge on the slip
issued: Date stamped on the bill
...

DOCUMENT
{note}

ANSWER fee: Printed charge on the slip

ANSWER issued: Date stamped on the bill
```

- One forward per note: 166 training forwards per epoch instead of 823.
- Every note token is read after all the questions. Each field's query vector is the last token of its own ANSWER line, which has read the questions, the note, and the line itself. The same vector feeds the operation head.
- Training groups a note's fields into one forward and sums their losses. Gradient accumulation counts fields, so one step still covers about 8 fields (`lora_tagger.groups_of`, `train_lora`).
- Everything else is as in experiment 3 (LoRA r 16, 10 epochs, selection on held-out final answers). Outputs are `lora_onepass_*`, so they cannot overwrite experiment 3's files.

### Results (experiment 4)

Kaggle kernel `ommnaidu/gemma-e2b-lora-onepass`, version 1, one T4, 1,570 s (26 minutes, against 78 for experiment 3). Files: `outputs/experiments/lora_onepass_results.{md,json}`, `lora_onepass_adapter.pt`. **This is the current best model and the candidate for the final grade.**

| | Training sample (300) | Held (193) | Exam 1 | Exam 2 | Exam 3 | **Exams 1–3** |
| --- | --- | --- | --- | --- | --- | --- |
| Evidence exact | 287 | 167 | 79/109 | 74/92 | 69/92 | 222/293 |
| **Final answers** | 287 | 168 | **79/109** | **74/92** | **70/92** | **223/293 (76%)** |
| Experiment 3, one forward per field, for comparison | 290 | 167 | 83 | 72 | 71 | 226/293 |

- Best epoch 9 of 10, bias −1.0. 161 training forwards per epoch. Operation choice 102/103.
- Final answers by operation, exams 1–3: sum 10/10, subtract 7/7, missing 29/29, read_time 3/3, read_datetime 3/3, product 9/11, days_between 12/14, read_date 54/60, read_number 7/9, copy 66/95, median 5/9, add_days 3/6, list 15/37.
- Evidence buckets on exams 1–3: wrong place 33, missed 6, partial 32.
- Held-out final answers by epoch: 44, 108, 134, 151, 151, 163, 164, 158, **168**, 157.
- **Confidence**: AUROC 0.861. Answers at confidence ≥ 0.99: 194 of 212 right (92%), covering 72% of fields. At ≥ 0.9: 218/270. At ≥ 0.5: 223/293.

**Reading.** One pass is within run-to-run noise of one forward per field (223 against 226), at a third of the training compute and a fifth of the inference compute per note. It reads each note once, as Clairic says it does.

**Run-to-run noise.** An import bug (`run_frozen` the function shadowed `run_frozen` the flag; fixed) made this kernel also retrain the two frozen baselines. They scored 124/293 and 112/293 against 116 and 105 in experiment 2: same setup, different run. So differences of about ±8 fields on exams 1–3 are noise. All results are a single seed (7).

## Confidence scores

Each answer carries `p_null` and `confidence`, computed from the same logits that produced it (`lora_tagger.field_confidence`):

- `p_null` = 1 − the highest probability, over the note's tokens, that a token is evidence.
- `confidence` for an answer with spans = the geometric mean of the probability of each decoded tag over the marked tokens and the one token either side, times the probability of the chosen operation among the legal ones. For a "missing" answer, confidence = `p_null`.
- The report gives, at thresholds 0.5–0.99, how many answers are at or above the threshold and how many of those are right, plus AUROC (how often a right answer is more confident than a wrong one; 0.5 means no signal).

These are not calibrated probabilities yet. Clairic's `p_null` and `confidence` presumably are. Calibration (for example, temperature fitted on the held-out fold) is a next step.

## Real documents: SROIE (`experiments/sroie.py`)

The user chose a public dataset over their own documents. SROIE (ICDAR 2019) is scanned receipts, mostly Malaysian, with OCR lines and four labels: company, date, address, total. Kaggle copy: `urbikn/sroie-datasetv2` (no licence listed on the page; used for evaluation only; nothing committed). Only the **test split, 347 receipts**, is used, **zero-shot**: no receipt is ever trained on, and the instructions are new wording.

Conversion:

- The text is the OCR lines in file order, joined with newlines.
- Company and address labels are located comparing letters and digits only, so `NO.3` matches `NO. 3`. On a 29-receipt sample this took unmatched addresses from 9 to 1. The remaining one is a real label typo (`B1750` for `81750`), and it is left out and counted.
- The total is located by its digits, preferring a line that says TOTAL and not SUB TOTAL, or the line above when the amount stands alone. A unit test caught the first version choosing SUB TOTAL.
- Instructions: company "Name of the company that issued the receipt", date "Date printed on the receipt", address "Address of the company that issued the receipt", total "Total amount paid".

Three scores, reported separately:

| Score | Meaning |
| --- | --- |
| found | One marked span whose letters and digits equal the gold slice (for the total, the same amount). Independent of any parser. |
| value | Final answer with the reader as it is (the auditor's parsers, made for the synthetic notes' formats). |
| value_wide_dates | The same, with `read_date_wide` (dd-mm-yyyy, dd.mm.yyyy, two-digit years, dd-MON-yyyy). **Not a clean test**: written after seeing a 30-receipt sample of this split. |

The sample showed real receipts use date forms the reader cannot parse (`09-JAN-2017`, `03 MAR 19`, `21-03-18`, `01/01/18`). That is the "written differently" problem the user raised when the marking test was closed, measured on real data.

## The final grade (`experiments/kaggle_final_job.py`)

- Kernel `ommnaidu/gemma-e2b-final-grade`, built with `make_kaggle_kernel.py OUT final SOURCE_KERNEL`. It is the only bundle that contains exam 4 (tested).
- It attaches the chosen model's kernel output (`kernel_sources`) and the SROIE dataset (`dataset_sources`).
- It loads the adapters by name, the tagger, the operation head, the layout and the decode bias, exactly as selected. Nothing is trained or tuned.
- It scores exam 4 (evidence, final answers, operation choice, confidence) and SROIE (the three scores above, plus confidence).
- Run it once.

## Choosing the final model

Between experiment 3 (one forward per field) and experiment 4 (one pass): pick on exams 1–3 final answers and the held-out fold. Never on exam 4 or SROIE. If one pass is within a few fields, prefer it: it is about 5× cheaper per note and matches how Clairic reads.
