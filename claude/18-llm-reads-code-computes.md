# 18 — LLM reads, code computes

## Why

Note 17 showed the untrained LLM (Gemma 4 E2B, structured JSON) reads better than our trained system but computes worse: when a value has to be calculated it does the arithmetic in its head and is usually close but not exact. Our system is the opposite: exact arithmetic in code, weaker reading. The obvious combination: the LLM reads, and for a calculated value it writes the **inputs and the steps**, which our `values` code computes exactly. The same run was to score our trained system on the SROIE receipts, so that row would finally have both systems.

## Setup

- Prompt: `SYSTEM_CALC` in `experiments/structured_baseline.py`. The same rules and worked examples as note 17 plus one rule, "NEVER do arithmetic yourself". A printed value is given as before; a calculated one as `{"inputs": [...], "steps": [...]}`.
  - Steps are the experiment-6 blocks (add, subtract, multiply, divide, percent_of, minus_percent, plus_percent, add_days, minus_days, add_weeks, minus_weeks), folded left to right by `values.run_chain`.
  - Or one whole-list step: sum, average, median, days_between.
  - Example: "Subtotal Rs 2,400. 15% off, then 18% GST." → `{"inputs": [2400, 15, 18], "steps": ["minus_percent", "plus_percent"]}` → 2,407.20.
- `evaluate_calculation` turns the object into a value; a malformed one (unknown step, wrong number of inputs) counts as wrong, never a crash. A test checks that objects built from the gold answers reproduce every calculated gold field.
- Kernel `ommnaidu/gemma-e2b-hybrid` (job `hybrid`), free greedy generation plus a JSON parse (lm-format-enforcer still fails to import). System prompt only, no training notes as examples. Exam 4 not loaded.
- Results: `outputs/experiments/hybrid_results.{md,json}`. Our system on SROIE: `outputs/experiments/ours_sroie.json` (kernel `ommnaidu/gemma-e2b-ours-sroie`, job `ours_sroie`).
- The first run crashed before our system's receipts: the experiment-4 operation head has 12 outputs and experiment 6 added "average" and "chain" to `OPS`, so `field_confidence` indexed past the end. Fixed with `legal_ops`, shared with `choose_op`, plus a test. Our system was then run alone, and through the same code also scored exams 1–3: **223/293, exactly experiment 4's number**, so the loading and scoring are right.
- Receipt-level examples are stripped from every committed result (a test checks): SROIE has no licence for redistribution.

## Results

| Part | LLM computes (note 17) | **LLM reads, code computes** | Our system |
| --- | --- | --- | --- |
| Exams 1–3 (293) | 250 | **252** | 223 |
| Computed, one step, new wording (72) | 22 | **43** | 39 |
| Computed, two steps, trained combination (72) | 2 | **21** | 9 |
| Computed, two steps, new combination (72) | 0 | **26** | 4 |
| Numbers and dates in words (287) | 216 | **227** | 147 |
| SROIE real receipts (1,362) | 1,109 | **1,120** | 66 (74 with the wide date reader) |
| Time per note on a T4 | 4.65 s | 4.93 s (64 new tokens) | 0.14 s |

Unparsed replies: 16 of 698.

### Why the calculation rows moved

Every calculated field in the computed sets, by what the LLM sent back:

| Set | Inputs and steps, right | Inputs and steps, wrong | Did its own arithmetic anyway (right / all) | Empty |
| --- | --- | --- | --- | --- |
| One step | 38 | 18 | 5 / 15 | 1 |
| Two steps, trained combination | 21 | 32 | 0 / 14 | 5 |
| Two steps, new combination | 26 | 34 | 0 / 10 | 2 |

- When it hands the calculation to code, it is right 68% (one step) and about 40% (two steps). When it ignores the rule and computes itself, it is right 5 of 39 times.
- "New combination" means nothing to the LLM: it was never trained on our pairs, so it scores the same on both two-step sets (21 and 26). Our step head memorized pairs (note 16) and drops from 9 to 4.
- The remaining errors are **reading** errors: the wrong numbers as inputs, an extra or missing step, or the steps in the wrong order. For example "multiply, subtract, multiply" over three numbers where the note meant two steps.

### What still goes wrong

- **It still does date arithmetic itself.** 38 of the 39 "did its own arithmetic" cases are dates (5 right): asked for a date some days or weeks away it writes the date, often a week or more off. With numbers it follows the rule (1 case).
- **Whole-list operations written as steps.** Average as `["sum", "divide"]` and median as `["add", "divide"]` over two of the values: median is 1/9 on exams 1–3. The prompt says these are a single step; a 2B model does not always follow it.
- **Products.** 6/11 on exams 1–3, often reading a quantity or unit price wrong.
- Several "errors" are label conventions, as in note 17: `"Harbour Spe\ncs"` (an OCR line break in the gold) against `"HARBOUR SPECS"`, or "a door light" against "door light".

## Real receipts: our system collapses

| Field | LLM reads, code computes | Ours (final answer) |
| --- | --- | --- |
| company | 239/341 | 3/341 |
| date | 314/342 | 63/342 (71 with the wide date reader) |
| address | 304/332 | 0/332 |
| total | 263/347 | 0/347 |
| **all** | **1,120/1,362 (82%)** | **66/1,362 (5%)** |

Our system takes 0.17 s per receipt against about 5 s for the LLM, but it is not working on these documents:

- **Total:** it marks 7 to 22 scattered numbers per receipt ("07", "388 82", "193.00", "GST", …) instead of one amount, so the reader refuses or picks wrong.
- **Address:** it marks nothing at all.
- **Company:** it marks fragments ("UI" of "ZHENG HUI", "HD" of "SDN BHD").
- **Date:** the only field that partly works; misses are boundary errors on dd/mm/yyyy with a time after it ("15/01/2019 11").
- **Confidence does not warn:** at ≥0.99 it keeps 426 answers and 67 are right. On exams 1–3 the same cut was reliable, so the confidence learned on our notes does not carry over to new kinds of document.

The reason is the training data: 208 synthetic notes in one clean writing style. Receipts are OCR lines, capitals, codes and many amounts. Nothing in training looked like this, and LoRA with heads learned the notes rather than the task. The untrained LLM was never tied to our notes, so it transfers.

## What it says

- **The hybrid beats our trained system on every row, including calculation, the one row ours had won, and by 1,120 to 66 on real receipts.** The user's challenge was right. Our system's only advantage left is cost (about 30–35× cheaper), and that is worth nothing at 5% on real documents.
- The split of labour is right: reading is the LLM's strength and arithmetic is code's. Our `values` blocks are the useful part of experiment 6; the trained step head is not needed.
- The hybrid's own ceiling is now reading the right inputs and following "no arithmetic" for dates. Both are cheap to attack without training: schema-constrained decoding that forbids a literal value for a calculated field; separate date steps the model cannot skip; or a "median"/"average" example per type.
- Caveat: the computed sets come from our own templates (note 16), and exams 1–3 have been looked at many times. Exam 4 is still unseen. SROIE has now been used for both systems, so it is no longer held back for a final grade; it was a real-document test, never tuned on, and stays a fair comparison.
- For production: the LLM reads and code computes. If cost matters, the next step is to train on the LLM's outputs over real documents (distillation), not on more synthetic notes.
