# 03 — The plan

## The idea: point, then read, then combine

The four heads stay, but they run in a chain instead of side by side.

1. **Point (find the evidence).** For every field, mark the words on the page that hold the evidence: the printed value, each list item, each part of a sum, the start date and the day count of a due date. The writer's brackets label this for every field, including the inputs to calculations.
2. **Read.** Turn each marked piece into a typed value. Digits are read off the page, not recalled.
3. **Combine.** Pick the operation (copy, sum, subtract, product, add days, days between, median) and run it exactly. A date is legal by construction. "Emit" stays as the fallback for anything no operation covers.

Example, `e4-bills-02`, field `settle_by` ("issue date plus the stated credit days"):

```
Issued [4 June 2026], credit line [21] days
read:     2026-06-04, 21
combine:  add_days  ->  2026-06-25
```

It is a person using a calculator. The model decides what to add; the calculator adds without mistakes.

## The engineering choices inside the plan

| Choice | Why |
| --- | --- |
| Question before the note | So the note's token states can depend on the question (causal attention). |
| Token tagging (outside / begin / inside) instead of start-end | One question handles a list or the parts of a sum; no "Item 1 / Item 2" questions; no length cap; training and decoding are the same object. |
| Read from the marked tokens, not from a pooled vector | Keeps the digit order. Copying generalizes; recall does not. |
| Typed operations run exactly | The arithmetic is never wrong; only the choice of inputs and operation can be. |
| Middle layers as well as the last | The last layer is shaped for next-token prediction. Layers 17, 26 and 35 of 35 are mixed with learned weights. |

## Data without new writer tokens (planned, not built)

The raw writer files still have their brackets, so each note can be re-rendered many times with new values inside the brackets (amounts, dates, codes, counts), with varied formats (`17/03/2026`, `March 17th`, `Rs 1486.75`), and with numbers written in words. `audit.py` then recomputes every label and re-applies the leak rule. The same sentence then has many correct answers, so memorizing stops paying. It also creates the first data that can measure the words case. Limit: the sentence skeletons are still the 208 notes.

## How to measure honestly

- Choose settings on training notes the model did not train on: 5-fold cross-validation grouped by note (experiment 1), or a fixed held-out fold (experiments 2–3).
- Treat exams 1–3 openly as dev. Never choose a setting on them.
- Grade exam 4 once at the end.
- Score each stage separately: evidence exact, operation choice, final answer.

## Gates

Each step had a check that could prove the plan wrong. The most important one, stated before running: *if question first does not reduce wrong-instance errors, the plan is wrong at its core.* It did reduce them (05), but not enough on its own. That pointed at the frozen model (06).

## Checks proposed before building

1. Probe the question vector for an amount's digits, by cross-validation. Expected near chance. **Result: 0/180.**
2. Question first versus question last, same head, same folds; count wrong-instance errors. **Result: fewer wrong-instance errors, more misses, no net gain.**
3. Readers on gold spans, to isolate reading from finding. **Result: the auditor's parsers reproduce all 1309 gold answers from gold evidence (`tests/test_lora.py`).**
