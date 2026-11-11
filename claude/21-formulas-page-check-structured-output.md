# 21 — Making the LLM reliable at calculations: formulas, a page check, structured output

Written 11 November 2026, after note 18. The question that started it, Om's: *"It's surprising that one-step and two-step aren't higher with the LLM when code does the maths. What's the fix? Or should we bother?"* Results: `outputs/experiments/formula_results.{md,json}` (round 1), `formula_v2_results.*` (round 2), `head_to_head_results.*` (exam 4). Code: `experiments/formula.py`, `experiments/kaggle_formula_job.py`, `experiments/kaggle_head_to_head_job.py`, tests in `tests/test_formula.py`.

## Step 0: read the wrong answers before fixing anything

Every wrong calculation from note 18's run was paired with its note (the computed sets are regenerated deterministically, seed 23) and read by hand. The arithmetic was never the problem: code did it. The model was **translating the note into our invented step names**, and the translation went wrong:

| Note says | Model wrote (note 18 format) | What went wrong |
| --- | --- | --- |
| "Payment is due 7 weeks after 9 October 2026" | `{"inputs": ["2026-10-09", 7], "steps": ["add_days"]}` | days instead of weeks (6 one-step cases) |
| "Notice is needed 25 days ahead of 26 March 2026" | `add_days` | "ahead of" read as adding |
| "Hotel 925.54 plus meals 416.55, with 20% service tax on everything" | `["add", "add", "multiply", "add"]`, or an invented `multiply_percent` | our fold-left steps cannot say "(a + b) × 1.2" naturally |
| "Weighings came to 4,778.92, 269.02 and 15.59" (mean) | `["sum", "divide"]` with no count | a whole-list step mixed with a chain step |
| "18% off" | input `0.18` | the format said 18 |
| A date some days or weeks away | the date itself, computed in its head (38 of 39 rule breaks) | ignores "never do arithmetic" for dates |

Also found: a few "errors" are ambiguous test wording, e.g. "3,869.70 is cut by 10% in the sale and **20% more** at the till": our label says a second discount, the model read +20%. Even a perfect system will not reach 72/72 on these sets.

## The three fixes

1. **Formulas instead of step names.** A calculated value comes back as `{"formula": "2400 * (1 - 15%) * (1 + 18%)"}` or `{"formula": "2026-10-09 + weeks(7)"}`. The model has seen millions of formulas and never our step names, and brackets say the order. `experiments/formula.py` parses it with Python's `ast` (never `eval`), node by node: numbers, ISO dates, `+ - * /`, brackets, `n%`, `days()`, `weeks()`, `sum`, `mean`, `median`, `days_between`. All arithmetic in Decimal; money rounded once at the end. Same prompt, same three example documents as note 18, so only the answer format changed (`SYSTEM_FORMULA`).
2. **The page check, with one retry.** Every number and date an answer uses (or gives as a literal) must be printed in the document, as digits or in words (it reuses experiment 7's word forms: "one lakh twenty thousand", "the second of April, 2026"). A value not on the page was invented or computed in the model's head. Fields that fail are asked once more with the reason; if they still fail they are **withheld** (sent to a human in a product). Tested: formulas built from the gold evidence reproduce every calculated answer and pass the check, including the words notes (no false alarms).
3. **Structured output (constrained decoding).** Om asked: "Why force it? LLMs support structured output now." *Forcing is what structured output is*: hosted APIs restrict which tokens can come next so the reply must fit a schema. We run Gemma ourselves through `transformers`, which has no such switch, so we added **xgrammar** (lm-format-enforcer had failed to import on Kaggle). The schema is built from the field list: every key present, a number field is a number, a formula or null, a date field is an ISO date, a formula or null. It cannot know which fields *should* be calculated, so it cannot stop a date computed in the head; the page check catches that.

A free replay first: the page check applied to note 18's answers would have flagged 36 of the 114 wrong ones (and 5 right ones, which were dates the model computed itself and happened to get right). The other 78 used real numbers from the page, just the wrong ones; that is what the formula format had to fix.

## Round 1 results (prompt v1, nothing fitted to these tests)

| Part | Note 17: LLM computes | Note 18: steps | Formulas, free | **Formulas + structured output** | Our system |
| --- | --- | --- | --- | --- | --- |
| One step (72) | 22 | 43 | 58 | **67** | 39 |
| Two steps, trained pair (72) | 2 | 21 | 29 | **30** | 9 |
| Two steps, new pair (72) | 0 | 26 | 40 | **40** | 4 |
| Exams 1–3 (293) | 250 | 252 | 251 | **257** | 223 |
| Numbers and dates in words (287) | 216 | 227 | 234 | **250** | 147 |

All after the check and one retry. Time: 3.3 s per generation free, 4.2 s constrained.

- **Formulas did most of it** (one step 43 → 58, two steps 21/26 → 29/40). **Structured output added more everywhere**, including fields with no calculation (exams +6, words +16): replies can no longer be malformed or the wrong type.
- **The check is a good gate.** Answers that pass are right about 90% of the time on exams and words (257 of 289 given; 250 of 286); nearly every withheld answer was wrong. Constrained, exams 1–3: 4 withheld, 0 of them right.
- **What is left in two steps:** missing brackets, `1795.61 + 1815.19 / 3` where the note means the total split three ways (most combo_seen errors); "Delivery is 4 weeks after 25 October 2027; installers arrive 11 days before that": the model writes only `2027-10-25 - days(11)`, dropping the weeks, every time (0/18; one sentence template repeated, so one failure, not eighteen).

### A bug of mine: the check missed receipt dates

SROIE in round 1 (constrained): first answers 1,107/1,362, **after the retry 1,032**. Dates fell from 305 to 229. The page check knew note-style dates but not receipt forms: `15-01-2019`, `15.01.2019`, `15/01/19`. It flagged correct dates, and the retry replaced them with worse ones. Fixed (numeric dates with `/ - .` separators and two-digit years, either day/month order) and tested on made-up receipt-style strings, not real receipts. **Lesson for the blog: a reliability check with false alarms makes the system worse. Test the check on the kind of document it will guard before trusting it.**

## Round 2 (prompt v2): tuning on the dev set

Two general rules added after reading round 1's errors (`SYSTEM_FORMULA_V2`; a test checks that only these two lines differ):

- "Brackets matter: a + b / 3 divides only b. Put a total in brackets before dividing, multiplying or adding a percentage to it: (a + b) / 3, (a + b) * (1 + 18%)."
- "A date counted from another date that is itself calculated: chain them in one formula, 2026-05-01 + weeks(2) - days(3)."

Plus the fixed receipt-date check. Constrained mode only. Results: RESULTS_V2

## The dev/test conversation (for the blog)

Om asked whether round 2 was cheating on the dataset, then: "People do prompt changes and look at the data all the time. What is wrong with it?" The answer that landed was the exam analogy:

> A student does practice questions, looks at the ones they got wrong, and learns from them. That is good. But if someone asks "how good are you?" and they answer with their score **on those same practice questions**, the score is inflated: they already saw and fixed those mistakes. The honest answer is the score on **new questions they have never seen**.

So: **looking at errors and changing the prompt is normal and fine. The mistake is reporting the score on the data you tuned on as how well it works.** From round 2 on, the calculation sets, exams 1–3 and the words notes are the **development set** (tune freely). Exam 4 is the **held-out test**, run once, with nothing changed after. That is the same rule the trained system was held to.

How each change is classed:

| Change | Fitted to these tests? |
| --- | --- |
| Formulas instead of step names | No: same prompt and examples, only the answer format |
| Structured output | No: knows nothing about the tests |
| Page check | Mostly no: a general rule; the receipt-date fix came from seeing receipts fail but was fixed for date formats in general |
| Round 2's two rules | **Yes**: written after reading these errors. General in form, so they will probably transfer; exam 4 says whether they do |

## The exam 4 head-to-head (run once)

Frozen before it ran (commit `25f6953`): the LLM with prompt v1 and v2 (constrained, check, retry), and our trained experiment-4 and experiment-6 models as selected, on the same parts: exam 4 as written (24 notes), exam 4 with numbers and dates in words (seed 47), and exam 4 with one computed field per note in the three kinds (seed 59). Results: RESULTS_EXAM4

## What to say in the blog

- **"Make the model say what it means in the language it already knows."** Our step names were a private language; the 2B model translated into it badly. Formulas are a language it already speaks. Same model, same prompt, no training: one-step calculations 43 → 67 of 72.
- **Structured output is cheap accuracy.** It helped every field type, not only calculations, and costs about a second per note.
- **A check that a value is on the page is a strong, cheap gate**, and it is the honest version of "confidence": it says which answers to trust. But a check with false alarms is worse than none (the receipt-date bug).
- **Tune on a dev set, grade once on a held-out set.** Prompt changes after looking at errors are fine; quoting those numbers as the result is not.
