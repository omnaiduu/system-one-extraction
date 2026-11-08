# 11 — Decision log

Everything decided in this session, in order, including the wrong turns. Dates are 7 November 2026.

| # | Decision | Why | Outcome |
| --- | --- | --- | --- |
| 1 | Read the whole repo and git history, including the deleted marking test | The handoff said "different way of using the heads" without saying how | Found the marking test had scored 42 against the four heads' 9 |
| 2 | Diagnose before building: causal order, recall versus read, selection rule, data, loss/decoder mismatch | The user asked what went wrong | See 02 |
| 3 | Plan: point → read → combine, keep all four heads in a chain | Keeps marking for what it can express, adds learned paths for the rest | See 03 |
| 4 | Re-explained in plain words several times, with a real bill | The user did not follow the first, dense version | The question-first explanation, then the "why not the question vector" explanation |
| 5 | Run cheap checks first (probe, question order) before building the full pipeline | Each could kill the plan cheaply | Both ran in experiment 1 |
| 6 | Re-audit the raw files for labels | Older stored files lacked evidence for add_days and days_between | Exact match to stored rows, plus full evidence |
| 7 | Exam 4 out of everything | Keep one clean final grade | Enforced in code and tests |
| 8 | Modal via GitHub Actions | The container could not reach Kaggle, Hugging Face or Modal; Actions could | Blocked: newline in secrets, paused workspace, then a card needed for GPUs |
| 9 | Budget guard: cap timeouts, GPU only on commits marked `[run]` | The user had $30 | **Mistake**: the commit that explained the marker had `[run]` in its message and triggered a run. Fixed to "starts with `[run]`". It failed before any spend |
| 10 | Modal CPU | Card-free | Cancelled when the user asked for Kaggle |
| 11 | Kaggle directly from the session: new environment with Kaggle hosts, token as a network secret | Free GPU, no card; the user asked | Works. The pasted token was not used; the user was told to revoke it |
| 12 | CLI wrapper skipping token introspection | The proxy fills headers, not bodies | Works; in `experiments/kaggle_cli.py` |
| 13 | Class weights (1, 3, 3) in the tagger | A planted-signal test showed the unweighted tagger predicts "nothing here" | Caught before GPU time |
| 14 | Test BOS as well as order | Hunch that missing BOS hurt every earlier round | **Wrong**: BOS made it worse |
| 15 | After experiment 1: the frozen representation is the bottleneck | Same ceiling for every head; probe 0/180; question first no net gain | Recommended a cheap follow-up (A) first; the user chose LoRA (B) |
| 16 | LoRA r 16 on q/k/v/o, question first, frozen baselines in the same run on the same split | A fair, attributable comparison | 116 → 195 of 293 |
| 17 | Uninstall torchao on Kaggle; save baselines before risky steps | Version 1 crashed and lost its baselines | Version 2 succeeded |
| 18 | Final answers with the auditor's own parsers and arithmetic, verified 1,309/1,309 on gold | Any error must be the model's, not the calculator's | Experiment 3 |
| 19 | Operation head only where the type allows several operations | Most types have one operation; number and date have several | Experiment 3 |
| 20 | Select epoch and bias by right final answers on held-out | Optimize what will be claimed | Experiment 3 |
| 21 | 10 epochs | Experiment 2 was still rising at its last epoch | Experiment 3 |
| 22 | Write these notes; refuse "cracked it" until final answers, exam 4 and real mail | Keep the blog defensible | This folder |
| 23 | Reorder the five approved steps: exam 4 last | Exam 4 can be used once; grading before the model is final would spend it | Final grade deferred |
| 24 | One-pass layout: questions before the note, an ANSWER line per field after it | Clairic reads once; about 5× cheaper per note | 223/293, within noise of 226 |
| 25 | Confidence from the same logits; report coverage at thresholds and AUROC | Clairic returns confidence; users need to know what to check | AUROC 0.861 |
| 26 | SROIE test split, zero-shot, three separate scores | The user chose a public dataset over their own mail | Prepared; scored only in the final grade |
| 27 | Match SROIE labels by letters and digits only | Spacing and punctuation differences, not label errors; real typos still excluded | Unmatched addresses 9 → 1 on a sample |
| 28 | A unit test for TOTAL versus SUB TOTAL | It caught the converter choosing SUB TOTAL | Fixed |
| 29 | Kernel title must match its slug | A mismatch caused Kaggle's 409 on push | Fixed |
| 30 | **Mistake**: `run_frozen` the function shadowed `run_frozen` the flag | Frozen baselines ran anyway | Fixed; the accidental rerun gave the ±8 noise estimate |
| 31 | Build a learned reader and an emit head, and words/emit data from existing notes | The user asked for values not on the page | Built and tested |
| 32 | `plain_number` refuses text with words | `parse_number` read "April the second, 2026" as 2026 (caught by a test) | Fixed; short capital currency codes allowed after the gold test failed on `CFA 18.40` |
| 33 | Train readers jointly with the adapters | Simplest first | **Wrong**: reader losses took the adapters over, 223 → 181 |
| 34 | Detach the readers | Keep marking intact | 205; emit still 0/72 |
| 35 | Recommend exact operations over a learned emit head | Exact values have only ever come from copy + code | Not yet built |
| 36 | Compare with Clairic on capabilities, not accuracy | Cloudflare published no extraction accuracy or architecture | Note 15 |
| 37 | Drop free emit; compute values that are not printed as chains of exact blocks over marked facts | Emit asks the model for arithmetic; the op head already chooses perfectly | Experiment 6 |
| 38 | A fold in text order with order-free percent and date blocks | Keeps the step head's job a small choice per step; the roles come from the text (`%`, which operand is a date) | Exact: 1,000+ generated fields reproduce |
| 39 | Hold out whole two-step combinations for the test | The only way to tell composing from memorizing pairs | It memorized pairs: 4/13 blocks right on unseen combinations given right marks |
| 40 | Neutral field names (`calc_0`) | The field name is part of the question and would give the formula away | Caught before the run |
| 41 | Stop at the stage gate | Stage 1 half works on new wording, stage 2 does not generalize | Next: snap marks to whole numbers, varied phrasing, blocks in every position |
| 42 | For numbers and dates in words, plan a parser-first read with plain Gemma rewriting only the marked words in a forced format, checked by writing the value back | Reading formats is translation, which Gemma already does; computing is not. The check catches misreadings | Not built |
| 43 | Experiment 7 built (grammar, one-operand step head, snapping, Gemma rewrite) and then cancelled before results | The user objected that we were tuning to our own generator, and asked for the LLM baseline instead | Code kept; kernel deleted (Kaggle has no cancel) |
| 44 | Measure the untrained same-size LLM with structured output, with a full system prompt and worked examples | The user's challenge; it had never been measured | Reads better (250 vs 223), computes worse (22 vs 39), 34× slower |
