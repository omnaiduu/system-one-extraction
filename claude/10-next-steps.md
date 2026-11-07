# 10 — Next steps and open questions

In rough order.

## 1. Finish experiment 3

Collect version 3 of `ommnaidu/gemma-e2b-lora-evidence`, copy the files into `outputs/experiments/`, and fill in 07. Look at:

- the final answers on exams 1–3 against the highlighter's 48 / 42 / 49;
- whether the held-out curve flattens before epoch 10 (if it is still rising, try more epochs or a higher rank);
- operation-choice accuracy (the only ambiguous types are number and date);
- training-sample fit (watch for memorization creeping back with more epochs).

## 2. Grade exam 4, once

Write a scoring-only job that loads `outputs/experiments/lora_adapter.pt` and scores exam 4 with the frozen choices (epoch, bias). It is the only step that may include exam 4's raw files in a kernel. Compare against 9/77 (four heads) and 42/74 (highlighter, rules designed on exam 4, so optimistic), counting the same fields: forecasts were not scored by the highlighter.

## 3. Augmentation (planned in 03)

Re-render the bracketed raw files with new values, formats, and numbers in words; re-audit. That gives (a) more varied training data with no writer tokens, and (b) the first measurement of the words case. If the parsers fail on words, add a learned reader over the marked tokens and keep the parser for digits.

## 4. Lists and edges

Lists are the weakest operation (16/37 evidence in experiment 2), and partial marks (wrong start or end word) are the biggest bucket left (47). Ideas: score spans as pairs (start + end jointly) on top of the adapted states; a CRF over BIO; a higher LoRA rank; MLP layers in the LoRA targets.

## 5. Forecasts and estimates

Forecasts are skipped (the answer was deleted from the page). The plan: point at the series values, read them, then a small statistical model over the numbers (the spread head on numeric features, not on Gemma states). Estimates already work as median-of-marked-comps; a band needs calibration data.

## 6. Read once

Question first means one Gemma pass per field. Clairic reads once. Test "all questions before the note, one pass" against per-field passes, with LoRA, and report the cost in accuracy against the cost in compute.

## 7. Real mail

The biggest open question. A few hundred real documents, labelled with the same evidence rules, are what would make any of this a production claim.

## Open questions

- Why 100 LoRA modules and not 140? Probably KV-shared layers in Gemma 4 E2B; check the module list.
- How much of the gain is the adapters versus the head seeing adapted states? An ablation with adapters only in the top half of the layers would tell.
- Seed variance: every number here is one seed (7). A second seed on the LoRA run would put an error bar on 195.
