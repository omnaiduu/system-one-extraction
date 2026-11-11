# 14 — Every result in one place

All numbers on the same three dev exams unless marked (exams 1–3: 72 synthetic notes, 293 scored fields, never trained on, never used to choose a setting in this session). Exam 4 and SROIE are still unscored. One seed (7). Run-to-run noise is about ±8 fields on exams 1–3 (see 12).

## The headline table: final answers on notes never trained on

| System | What it is | Exams 1–3 | Where |
| --- | --- | --- | --- |
| Four heads, rounds 1–4 | Frozen Gemma; point / number / calendar / spread heads in parallel | about 10–15% (29/112, 14/95, 15/95; round 4 9/77 on exam 4) | 01 |
| Marking test ("highlighter") | Frozen Gemma; one point head marks words, a program computes; hand-written decode rules (span cap, ordered chain, question block) designed on exam 4 | 139/293 (47%) | 01 |
| Experiment 3 | LoRA inside Gemma, question first, one forward per field, mark → read → combine, no hand rules | **226/293 (77%)** | 07 |
| **Experiment 4** | Same, but **one pass per note** with confidence scores | **223/293 (76%)** | 12 |
| Experiment 5a | Experiment 4 + learned reader + emit head, trained jointly, on augmented notes | 181/293 | 13 |
| Experiment 5b | Same, readers on detached states | 205/293 | 13 |

## Evidence only (were the right words marked?)

| Run | Model | Exams 1–3 evidence exact |
| --- | --- | --- |
| Exp 1 | Frozen tagger, old layout (CV-selected) | 118/293 |
| Exp 1 | Frozen tagger, question first | 106/293 |
| Exp 2 | Frozen tagger, old layout (held-out selected) | 116/293 |
| Exp 2 | Frozen tagger, question first | 105/293 |
| Exp 2 | **LoRA, question first, 5 epochs** | **195/293** |
| Exp 3 | LoRA, question first, 10 epochs | 224/293 |
| Exp 4 | LoRA, one pass, 10 epochs | 222/293 |

## The memorization signature

| Model | Fit on training notes | New notes | Gap |
| --- | --- | --- | --- |
| Four heads, round 4 | 948/991 (96%) | 9/77 (12%) on exam 4 | huge |
| Frozen tagger (exp 2) | 300/300 (100%) | 116/293 (40%) evidence | 60 points |
| LoRA, 5 epochs (exp 2) | 273/300 (91%) | 195/293 (67%) evidence | 24 points |
| LoRA, 10 epochs (exp 3) | 290/300 (97%) | 226/293 (77%) final | 20 points |
| LoRA, one pass (exp 4) | 287/300 (96%) | 223/293 (76%) final | 20 points |

## Diagnostic checks

| Check | Result | Meaning |
| --- | --- | --- |
| Linear readout of an amount's digits from one Gemma vector (180 numbers, 5-fold CV) | 144/144 training, **0/180 held out**, every layer and layout | Recall from a vector does not work; the old number head could only memorize |
| Gold evidence → read → combine | **1,309/1,309** gold answers reproduced | Any final-answer error is the model's, never the arithmetic's |
| Gold evidence + perfect reader on augmented notes | All > 2,000 fields reproduced | The words/emit labels are consistent |
| Adding BOS to the old layout | 450 → 416 CV | The BOS hunch was wrong |
| Question first on frozen Gemma | Wrong place 72 → 40, missed 26 → 81, net no gain | Order alone does not fix a frozen reader |

## Experiment 4 (current best) by operation, exams 1–3 final answers

sum 10/10 · subtract 7/7 · missing 29/29 · read_time 3/3 · read_datetime 3/3 · product 9/11 · days_between 12/14 · read_date 54/60 · read_number 7/9 · copy 66/95 · median 5/9 · add_days 3/6 · list 15/37. Operation choice 102/103.

## Confidence (experiment 4, exams 1–3)

AUROC 0.861. At confidence ≥ 0.99: 212 answers, 194 right (92%), covering 72% of fields. ≥ 0.95: 213/246. ≥ 0.9: 218/270. Not calibrated probabilities yet.

## Reader and emit (experiment 5)

| | 5a joint | 5b detached |
| --- | --- | --- |
| Plain exams 1–3 | 181/293 | 205/293 |
| dev_words (numbers and dates rewritten in words) | 152/287 | 147/287 |
| dev_emit (fields no listed operation computes) | 167/365 | 194/365 |
| Emit fields exact | 0/72 | 0/72 |
| Emit numbers within 1% | 0/50 | 0/50 |
| Operation choice | 99/103 | 103/103 |

## Computed fields and chains (experiment 6)

Answers never printed; facts marked, blocks chosen by the step head, arithmetic exact. Exams 1–3 original fields: 219/293 (experiment 4: 223), op choice 103/103, AUROC 0.917.

| Part | Computed fields right | Blocks right given exactly right marks |
| --- | --- | --- |
| held (training wording, unseen notes) | 152/168 (1 step 80/84, 2 steps 72/84) | 137/137 |
| one step, new wording | 39/72 | 33/34 |
| two steps, trained combination, new wording | 9/72 | 9/9 |
| two steps, new combination | 4/72 | 4/13 |

Main failures: amounts clipped or parts missed on new sentence shapes (only 26 training templates), and the step head learning whole pairs rather than blocks. Details in [16](16-computed-fields-and-chains.md).

## Untrained LLM baselines (notes 17 and 18)

Same Gemma 4 E2B, no training, JSON output, system prompt with worked examples. "Code computes": calculated values come back as inputs and steps and are computed by `values`.

| Part | LLM computes | LLM reads, code computes | Our system |
| --- | --- | --- | --- |
| Exams 1–3 (293) | 250 | **252** | 223 |
| One step, new wording (72) | 22 | **43** | 39 |
| Two steps, trained combination (72) | 2 | **21** | 9 |
| Two steps, new combination (72) | 0 | **26** | 4 |
| Numbers and dates in words (287) | 216 | **227** | 147 |
| SROIE receipts (1,362) | 1,109 | **1,120** | 66 |
| Seconds per note, T4 | 4.65 | 4.93 | 0.14 (0.17 per receipt) |

SROIE by field, LLM reads and code computes against ours: company 239 / 3, date 314 / 63, address 304 / 0, total 263 / 0.

## Formulas, page check, structured output (note 21)

The same untrained LLM; calculations as ordinary formulas evaluated by code; every number and date must be on the page (else one retry); JSON forced by xgrammar. Round 1 (prompt v1) was fitted to nothing; round 2 (v2) added two rules after reading round 1's errors, so its dev numbers are tuned.

| Part | Note 18 (steps) | Round 1, free | **Round 1, structured** | Round 2, structured | Our system |
| --- | --- | --- | --- | --- | --- |
| One step (72) | 43 | 58 | **67** | 65 | 39 |
| Two steps, trained pair (72) | 21 | 29 | **30** | 36 | 9 |
| Two steps, new pair (72) | 26 | 40 | **40** | 40 | 4 |
| Exams 1–3 (293) | 252 | 251 | **257** | 258 | 223 |
| Words (287) | 227 | 234 | **250** | 250 | 147 |
| SROIE (1,362) | 1,120 | – | 1,032 (date-check bug) | 1,108 | 66 |
| Seconds per generation, T4 | 4.9 | 3.3 | 4.2 | 4.6 | 0.14 |

Answers that pass the page check are right about 90% of the time on exams and words (257/289, 250/286).

## Exam 4: the held-out grade, run once (note 21)

| Part | LLM v1 | LLM v2 | Ours, exp. 4 | Ours, exp. 6 |
| --- | --- | --- | --- | --- |
| Exam 4 (74) | **63** | 62 | 57 | **63** |
| Exam 4 in words (71) | **62** | **62** | not measured | not measured |
| One step (24) | **23** | 22 | – | 9 |
| Two steps, trained pair (24) | 6 | **11** | – | 3 |
| Two steps, new pair (24) | 14 | **15** | – | 1 |

Dev held up: exams 1–3 88% → exam 4 85%; one step 93% → 96%; two steps, new pair 56% → 58–63%. Prompt v2's tuned rules: +4 of 145 on exam 4.

## Cost and time

Everything ran on Kaggle's free T4s. Experiment 1: 39 min. Experiment 2: 50 min (plus an 11-minute failed attempt). Experiment 3: 78 min. Experiment 4: 26 min. Experiments 5a and 5b: 55 min each. Experiment 6: 55 min. Structured baseline: 62 min. LLM reads, code computes: 59 min, plus 4 min for our system on SROIE. No money was spent anywhere.

## Settings shared by the LoRA runs

Gemma 4 E2B-it, float16, frozen base. LoRA r 16, alpha 32, dropout 0.05 on q/k/v/o of the text layers (100 modules, 5,357,568 parameters, float32). Hidden states from layers 17, 26, 35 of 35, mixed with learned weights. AdamW, lr 2e-4 for adapters and 1e-3 for heads, weight decay 0.01, clip 1.0, gradient accumulation 8 fields, autocast float16 with GradScaler, gradient checkpointing. Tag loss weights (1, 3, 3). Split: 823 fit / 193 held questions (fold 0 of 208 training notes). Epoch and decode bias chosen on held by right final answers.
