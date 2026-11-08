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
- Results: `outputs/experiments/hybrid_results.{md,json}`.

## Results

| Part | LLM computes (note 17) | **LLM reads, code computes** | Our system |
| --- | --- | --- | --- |
| Exams 1–3 (293) | 250 | **252** | 223 |
| Computed, one step, new wording (72) | 22 | **43** | 39 |
| Computed, two steps, trained combination (72) | 2 | **21** | 9 |
| Computed, two steps, new combination (72) | 0 | **26** | 4 |
| Numbers and dates in words (287) | 216 | **227** | 147 |
| SROIE real receipts (1,362) | 1,109 | **1,120** | OURS_SROIE |
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

## What it says

- **On these tests the hybrid beats our trained system on every row, including calculation, the one row ours had won.** The trained system's remaining advantage is cost (about 35× cheaper per note) and its confidence scores.
- The split of labour is right: reading is the LLM's strength and arithmetic is code's. Our `values` blocks are the useful part of experiment 6; the trained step head is not needed.
- The hybrid's own ceiling is now reading the right inputs and following "no arithmetic" for dates. Both are cheap to attack without training: schema-constrained decoding that forbids a literal value for a calculated field; separate date steps the model cannot skip; or a "median"/"average" example per type.
- Caveat: the computed sets come from our own templates (note 16), and exams 1–3 have been looked at many times. Exam 4 is still unseen and is the clean test.
