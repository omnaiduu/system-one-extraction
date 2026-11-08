# 10 — Next steps and open questions

**Current best system: LLM reads, code computes** (note 18): untrained Gemma 4 E2B with `SYSTEM_CALC`, calculated values computed by `values`. 252/293 on exams 1–3, 1,120/1,362 on SROIE. Our best trained model (experiment 4, 223/293) scores 66/1,362 on real receipts and is no longer the recommendation.

## 0. What to do next, in order

1. Make the hybrid follow its own rule: schema-constrained decoding (get lm-format-enforcer or outlines working with Gemma 4's tokenizer) so a calculated date cannot come back as a literal date (38 of 39 rule breaks were dates), and a single-step form for median and average.
2. Exam 4, once, with the hybrid and experiment 4 side by side.
3. If cost matters (about 5 s a note against 0.14 s), distill: run the hybrid over real documents and train the small tagger on its answers, then measure on documents of a kind it never saw. Synthetic notes alone do not transfer (note 18).

The sections below are the older plan for the trained system.

## 1. Make computed fields generalize (recommended next)

Built in experiment 6 ([16](16-computed-fields-and-chains.md)): exact blocks and chains, exact on training wording, weak on new wording. In order: (a) snap marks to whole numbers and dates, measured first with an inference-only kernel on `ommnaidu/gemma-e2b-lora-chains` (it may also help the original exams' partial bucket); (b) generate training sentences from a grammar, hundreds of surface forms, amounts in any position; (c) every block in every step position, no step-position embedding; (d) rerun with the same three held-out test sets.

## 1b. Numbers and dates in words or odd formats

Parser first. When the parser cannot read a marked part, plain Gemma (adapters off) rewrites only the marked words, with decoding forced to digits or an ISO date, about 7 tokens; code writes the value back in words and checks it against the marked text, else empty with low confidence. Needs real worded text (Indian cheques and invoices, "Rupees … Only") for an honest test.

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
