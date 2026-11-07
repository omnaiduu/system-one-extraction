# 16 — Experiment 6: computed fields, one-step operations and two-step chains

## Why

Emit (13) tried to make the model compute values in its head and failed (0/72). The user's goal stands: answer values that are not printed but whose facts are. The design here keeps "the model points and chooses, code computes":

```
mark the facts  →  read them (parser)  →  op head: which operation  →  step head: one block per step  →  exact arithmetic
```

## What was built

- `experiments/values.py`: new ops `average` and `chain` (appended to `OPS`, so older checkpoints keep their indexes). `BLOCKS`: add, subtract, multiply, divide, percent_of, minus_percent, plus_percent, add_days, minus_days, add_weeks, minus_weeks. A chain folds the marked parts left to right, one block per step; the number of steps is fixed by the number of marks (at most 3). Percent blocks take the rate from whichever operand is a marked percentage (`is_rate`: followed by `%` or "per cent"), so "15% off 2,400" and "2,400 less 15%" use the same block. Date blocks take the date from whichever operand is a date.
- `experiments/chains.py`: inserts one sentence per field into existing notes, answer computed by an independent formula, never printed. 16 single-step and 10 two-step training templates; held-out test phrasings in three sets: `single`, `combo_seen` (two-step combinations training has), `combo_unseen` (combinations training never has, made only of trained blocks). Field names are neutral (`calc_0`) so the question does not leak the formula.
- `lora_tagger.StepHead`: per step, reads the ANSWER token plus the two marked parts *and four tokens after each* (attention is causal: the word "off" after "15%" is only visible to later tokens). Trained with teacher forcing on gold marks, not detached (like the op head).
- Tests (`tests/test_chains.py`, 9): blocks are exact; every generated field (over 1,000, every template) reproduces its answer from gold marks and gold blocks; answers never printed; unseen combinations really unseen; a planted-signal test for the step head; end to end on a fake model; gradients reach the adapters; old op indexes unchanged; the final-grade summary. 71 tests pass.
- Kernel `ommnaidu/gemma-e2b-lora-chains`, job `lora_chains`; same settings as experiment 4 (one pass, r 16, 10 epochs, seed 7). 624 training notes (208 + 416 variants), 3,304 s on a T4. Best epoch 9, bias 1.0.

## Results

| Part | What it is | Computed fields right | Blocks right, given exactly right marks |
| --- | --- | --- | --- |
| held | training phrasings, notes never trained on | **152/168** (1 step 80/84, 2 steps 72/84) | **137/137** |
| dev_single | one step, new phrasings | 39/72 | 33/34 |
| dev_combo_seen | two steps, trained combination, new phrasings | 9/72 | 9/9 |
| dev_combo_unseen | two steps, combination never trained | 4/72 | 4/13 |

The original fields were not hurt: exams 1–3 **219/293** (experiment 4: 223; noise is about ±8). Op choice 103/103. Confidence improved: AUROC **0.917** (was 0.861); at ≥ 0.99, 207/234 right.

By kind, dev_single: weeks_after 14/14, weeks_before 8/9, average 6/8, date_minus 5/7, percent_of 4/5, discount 2/5, plus_percent 0/12, split 0/12.

## What went wrong (from the 200-field example sample of each dev part)

1. **Marking on new wording is the main failure.** Marks exactly right on about 20/35 single, 8/35 seen-combination and 9/35 unseen-combination fields. The typical error: an amount with a thousands separator is clipped to its first digit or cut at the comma ("3,720.78" marked as "3"), or one part is missed. `plus_percent` and `split` test phrasings put the amount at the start of the sentence ("The 3,720.78 fee rises by 12%", "967.34 is divided evenly…"), a position no training template used, and every one of those failed. With only 16 + 10 templates, the tagger learned where amounts sit *in those sentences*: memorization again, now of templates.
2. **The step head is perfect on the combinations it trained on, and does not compose.** Given right marks: 137/137 on held, 9/9 on seen combinations, 4/13 on unseen ones. Typical unseen errors: `add_weeks + minus_days` ("installers arrive 11 days *before* that") came out `add_weeks + add_days` (6 of 6 cases); after a discount, the second step came out `minus_percent` because the only trained pair starting with a discount was discount + discount. It learned whole pairs, not blocks.
3. The calculator and the operation choice were never the problem (op choice 100% on every part).

## Stage gate

Stage 1 (one step) works on the wording it trained on and half works on new wording; stage 2 (two steps) does not generalize yet. Per the agreed plan, stop and fix before going further. Not a candidate for the final grade; experiment 4 stays the best model for the original fields.

## Fixes, in order

1. **Snap marks to whole numbers and dates.** A number is atomic: if the tagger marks any token of "3,720.78", take the whole run of digits, commas and points (dates likewise). Deterministic, principled, and it should also cut the "partial" bucket on the original exams (45 of 74 evidence errors). It can be measured without retraining (an inference-only kernel on the saved adapter).
2. **Far more varied phrasing.** Generate sentences from a small grammar (amount first or last, different verbs, clauses in either order, several sentences per computation) so no template is learnable. Hundreds of distinct surface forms instead of 26.
3. **Teach blocks, not pairs.** Every block in every step position across training combinations; drop the step-position embedding; possibly let the step head see only the local window around the operand, so "before" means minus wherever it appears.
4. Then rerun with the same three test sets. Unseen combinations are the real test.
