# 10 — Next steps and open questions

Current best model: **experiment 4**, one pass per note (`outputs/experiments/lora_onepass_adapter.pt`, kernel `ommnaidu/gemma-e2b-lora-onepass`), 223/293 final answers on exams 1–3.

## 1. Exact operations for the common unprinted values (recommended next)

Add `percent_of`, `after_discount`, `average`, `date_minus_days` to `OPS` and `ALLOWED` in `experiments/values.py`, with exact arithmetic in `_format`. Relabel the emit variants in `augment.py` with those operations instead of `emit` (keep the training and test phrasings split). Retrain one pass with augmented notes, **without** the reader and emit heads, or with them detached. Expect the emit dev part to move from 0/72 toward the operation head's accuracy, which is already 103/103. Keep `emit` as a flagged last resort with low confidence.

## 2. The final grade, once

`python3 experiments/make_kaggle_kernel.py OUT final ommnaidu/gemma-e2b-lora-onepass` (or the kernel of whatever beats it on exams 1–3 and the held-out fold). Push, collect `final_results.{json,md}`, and write them into a new note. This is the only time exam 4 is used, and the only SROIE run. Compare exam 4 against 9/77 (four heads) and 42/74 (marking test, rules designed on exam 4), counting the same fields. Report SROIE's three scores separately (found / value / wide dates), and say the wide-date reader was written after seeing a sample.

## 3. Calibration

Fit a temperature (or isotonic regression) on the held-out fold for `confidence` and `p_null`; report a reliability curve and expected calibration error on exams 1–3.

## 4. Numbers and dates in words

A deterministic words-to-number parser (English, lakh and crore, "and N cents", ordinals for dates). It cannot be validated on `dev_words`, because the same code wrote those words, so the test would be circular. It needs real worded text. The learned reader from experiment 5 is not reliable; do not use it for exact values.

## 5. Lists and edges

Lists are the weakest type (15–20/37), and edge clipping is the largest error bucket. Ideas: a span-pair score on top of the adapted states; a CRF over the BIO tags; a higher LoRA rank or adapters on the MLPs.

## 6. Estimates and forecasts

Estimates: the median of marked comps works; an interval needs calibration data. Forecasts: mark the series values, read them exactly, then a small statistical model over the numbers (not over Gemma states).

## 7. Real documents and more seeds

SROIE is the first real test. Your own mail, even a hundred documents labelled with the same evidence rules, would be the second. Every number so far is one seed: a second seed on the best model would put an error bar on 223.

## Open questions

- Why 100 LoRA modules and not 140? Probably KV-shared layers in Gemma 4 E2B; check the module list.
- How much of the gain is the adapters themselves, versus the head seeing adapted states? An ablation with adapters only in the top half of the layers would tell.
- In the one-pass layout, does the number of fields per note matter? Clairic splits into parallel passes when fields do not fit; at what count does our accuracy drop?
