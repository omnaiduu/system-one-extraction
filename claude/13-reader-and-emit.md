# 13 — The learned reader and the emit head (experiments 5a and 5b)

## Why it was built

The value step reads marked words with the auditor's parsers and applies a fixed list of operations. That covers values that are not printed but whose pieces are (a sum, a due date). It cannot cover (a) numbers or dates written in words, which the parser cannot read, or (b) values that no listed operation computes. The user asked for the system to "emit when things are not there".

## What was built

- `experiments/augment.py`: training and test cases from the existing notes, with no writer tokens.
  - **words**: marked numbers and dates rewritten in words ("one thousand two hundred and sixty-three rupees and eighty paise", "April the second, 2026", "credit runs fourteen days"), with every span shifted and every value unchanged.
  - **emit**: one inserted sentence and a field whose answer no listed operation computes (percent of, after a discount, average of three, date minus days). Training phrasings and exam phrasings are disjoint.
  - Variants keep their source note's id before `~`, so a note and its variants stay on the same side of the split.
- `SpanReader` (`lora_tagger.py`): one learned query per output slot (sign + 7 + 2 digits, or year / month / day) attends over the marked tokens in order. Four of them: number, date, emit number, emit date.
- `values.answer`: parser first. A part the parser cannot read goes to the reader. `emit` returns the emit head's value. `plain_number` now refuses text with real words in it: the auditor's `parse_number` strips leading letters and read "April the second, 2026" as 2026 (caught by a test). Short capital currency codes (`CFA`, `MWK`) are still allowed (also caught by a test).
- `emit` is a 13th operation, allowed for number and date fields.
- Tests: every augmented field reproduces its gold answer with gold spans and a perfect reader (more than 2,000 fields); exam phrasings never appear in training; variants share their note's fold; a reader learns a planted value; reader gradients do not reach the adapters (5b).

Data: 604 training notes (208 originals + 396 variants), 479 one-pass forwards per epoch. Dev: exams 1–3 (293 fields), `dev_words` (63 notes, 287 fields), `dev_emit` (72 notes, 365 fields, 72 emit fields).

## Results (one pass, 10 epochs, one T4)

| Run | Exams 1–3 final answers | dev_words | dev_emit | Emit fields | Emit numbers within 1% | Operation choice |
| --- | --- | --- | --- | --- | --- | --- |
| Experiment 4, no reader, no augmentation (reference) | **223/293** | — | — | — | — | 102/103 |
| 5a: readers trained jointly through the adapters | 181/293 | 152/287 | 167/365 | 0/72 | 0/50 | 99/103 |
| 5b: readers trained on detached states | 205/293 | 147/287 | 194/365 | 0/72 | 0/50 | **103/103** |

Files: `outputs/experiments/lora_emit_results.*` (5a), `lora_emit_detached_results.*` (5b). Adapters not committed: neither run is a candidate.

## What it means

- **5a**: joint training hurt everything. Training loss stayed near 1.4 (against about 0.01 for the tagger alone); the reader losses took the shared adapters over.
- **5b**: detaching restored most of the marking (181 → 205) but not all of it (223). The training mix itself changed (two thirds of the notes are variants), and the held-out curve was unstable (454 at epoch 6, 313 at epoch 8).
- **The operation head recognizes an emit question perfectly** (103/103 on plain exams, and it chose `emit` on the emit fields). Knowing *that* a value must be emitted is easy.
- **The emit head learned nothing.** Its outputs barely depend on the input: amounts cluster around 127–167 whatever the inputs, dates repeat (2026-12-19 twice for different golds). Exact arithmetic to the cent, inside a small head on about 400 examples, did not happen.
- **The learned reader is not reliable.** Dates in words are often unreadable (an invalid day) or a few days off; one wrong digit breaks a sum (sums in words 1/10).

## Conclusion

A learned value head is the wrong tool for exact values at this data size. That matches the very first finding of this session (0/180 for reading digits from one vector). Every exact number in this project that came out right came from **copying the page plus exact code**.

Recommended next step (not yet built): add the common calculations as **exact operations** (percent of, after a discount, average, date minus days). The operation head already tells them apart, and code does the arithmetic. Keep `emit` only as a flagged last resort with low confidence. For numbers written in words, a deterministic words-to-number parser is the reliable path. It cannot be validated on `dev_words`, though, because the same code wrote those words: that would be circular. It needs worded examples from real text.

The final grade (exam 4 + SROIE) should use experiment 4's model, unless a later run beats 223 on exams 1–3.
