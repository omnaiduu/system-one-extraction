# 17 — The structured-output baseline: untrained Gemma 4 E2B writing JSON

## Why

The user's challenge: "This is not prod ready; we are tuning for a dataset; an LLM with structured output is just better at it." It was never measured. This is the measurement, on the same model size (Gemma 4 E2B-it), with no training at all.

## Setup

- `experiments/structured_baseline.py`, `experiments/kaggle_structured_job.py`, kernel `ommnaidu/gemma-e2b-structured`, job `structured`. Results: `outputs/experiments/structured_results.{md,json}`.
- System prompt: rules per type, null for absent values, decoys, "calculate when not printed", numbers in words, and three handwritten worked examples (checked by a test; not from any exam). Optional extra examples: three real training notes with their gold JSON as earlier chat turns.
- Decoding: schema-constrained decoding with lm-format-enforcer was attempted but its import failed on Kaggle (`ImportError: transformers is not installed`, a version check inside the library), so the run used **free greedy generation plus a JSON parse**. 18 of 770 replies did not parse (every field in them counted wrong), so the baseline is slightly underrated.
- Scored with the same `values.matches` as our system; strings read leniently with the auditor's parsers (a number as "1,486.75", a date in any parseable form).

## Results

| Part | Untrained Gemma, JSON | Our system |
| --- | --- | --- |
| Exams 1–3 (293 fields), system prompt | **250/293 (85%)** | 223/293 (76%), experiment 4 |
| Exams 1–3, + 3 training notes | 248/293 | 223/293 |
| Numbers and dates in words (287) | **216/287** | 147/287 (experiment 5b) |
| Computed, one step, new wording (72) | 22/72 | **39/72** (experiment 6) |
| Computed, two steps, trained combination | 2/72 | **9/72** |
| Computed, two steps, new combination | 0/72 | **4/72** |
| SROIE real receipts (1,362 fields) | 1,109/1,362 (81%) | not run |
| Time per note on a T4 | 4.65 s (56 new tokens) | **0.14 s** (one forward), about 34× faster |

By operation on exams 1–3 (baseline, system prompt; ours from experiment 4 in brackets): copy 87/95 (66), read_date 60/60 (54), read_number 9/9 (7), missing 29/29 (29), sum 9/10 (10), subtract 6/7 (7), add_days 5/6, product 6/11 (9–10), days_between 10/14 (12–14), list 22/37 (15–18), median 1/9 (5).

## What it says

1. **On reading the page, the untrained LLM is better.** Copying text, reading dates and numbers, words, and real receipts. It reads what our tagger clips (names, edges) and handles words with no extra machinery.
2. **On arithmetic, it is worse, and ours is not good either.** Its computed answers are usually close but not exact (25 of 72 one-step answers within 1%, 22 exact; two-step 2/72 and 0/72). Products, days between dates and medians also slip. Every exact computed answer in this project came from code.
3. **It is about 34× slower** and grows with the number of fields (it writes every value).
4. **Some of its "errors" are label conventions:** gold list items without articles ("door light" against "a door light"), gold text with the OCR line break ("Harbour Spe\ncs" against "HARBOUR SPECS"). The loose text score adds only 2, but list items with articles are scored wrong.

The user was right on accuracy for reading. The case for our system rests on speed and exact arithmetic, and the arithmetic advantage only holds when our marks are right.

## The design this points to

LLM reads, code computes: ask the LLM (structured output) for each computed field's **inputs and operation** (for example `{"inputs": [2400, 15, 18], "steps": ["minus_percent", "plus_percent"]}`) instead of the final number, and compute it with `values.run_chain`. That keeps the LLM's reading and the code's exactness, and the answer stays traceable. Not built yet. Our one-pass system stays the fast path where volume or latency matters.

Open: our system on SROIE (the final grade) has not been run; schema-constrained decoding (fix the enforcer version) would remove the 18 unparsed replies.
