# 20 — Session record for the blog: the questions, the explanations, the turn

What happened in the session of 7–8 November 2026 that ended the trained-system line of work, written for the agent who drafts the blog. Notes 16–19 hold the experiments and numbers; this note holds the **conversation**: what Om asked, how each idea was explained, the moments the direction changed, and the framings that landed. Quotes are Om's words, lightly cleaned.

## Timeline of the session

1. **Where are we?** Experiment 4 (one pass per note, 223/293 on exams 1–3) was the best model. Emit (experiment 5) had failed: 0/72.
2. **"Should we chase emit? Is it worth it?"** and **"What's the point of emit if most of it is already on the page? We don't want to predict text; the only things computed are dates and numbers. But what if only the facts exist and we have to calculate?"** This reframed emit: drop free generation of values, and compute values that are not printed from facts that are, with exact operations in code. That became experiment 6.
3. **"So many formulas, will it be good at them?"** The answer: no formula list. A few exact *blocks* (add, subtract, multiply, divide, percent of, minus/plus percent, add/minus days and weeks) and short *chains* of them (up to 3 steps). The model chooses blocks; code computes. Om: "Do it."
4. **Experiment 6 ran** (note 16). Exact on training wording (152/168), 39/72 one step on new wording, 9/72 and 4/72 two steps. Diagnosis: marks clipped on new sentence shapes ("3,720.78" marked as "3"), and the step head memorized whole pairs.
5. **The format question.** "The facts are there, but not in a format the model can parse: numbers in words, odd dates. How does it compute those?" Then: "What is this rule-based normalizer? Isn't it fragile? Can't the model be good at it? That's why I wanted an output head: read different formats and output one thing." This is the question the whole architecture turns on (see "The format problem" below).
6. **Many "explain it with an example" requests**: what happened, why it was good earlier, what it is trained for, the jobs and how they work, what the code does, how good it is at several steps, the format rules, the LLM rewrite plan. The explanations that worked are under "Framings".
7. **"Add 2 and 3."** Build the Gemma rewrite fallback and the multi-step fixes: experiment 7 (below).
8. **The challenge.** "This is not prod ready. We are tuning for a dataset. An LLM with structured output is just better at it, unless you want to prove me wrong." Then: "Use the same size model with structured output." Then: "Cancel unnecessary runs" (experiment 7 was cancelled) and "At least give it a proper system prompt, with examples, the way we want it."
9. **The baseline (note 17).** It reads better and computes worse.
10. **"LLM reads, code computes" (note 18).** Om: "Yeah, do it." Best on every test. Our system scored 66/1,362 on real receipts.
11. **"What's the conclusion?"** and **"What is the point of System One? A 2B LLM does better. This was a waste. I don't think it is economically viable to build an extraction model."**
12. **The architectural argument (Om's).** "It was able to extract things like dates. The issue is things not in the document, and it is not reliable when numbers or dates are written in another format: we are doing a second call anyway. That is a failure at the architectural level, and no amount of training data could have made it right."
13. **"But LoRA was a real improvement. What happened there?"** It improved the highlighter on new notes of the same kind, not on new kinds of document.
14. **Om's conclusion, which note 19 writes up:** "It is a good highlighter, but bad at being reliable at reading. It needs fragile rules or a second LLM call when the format is weird, and it is bad when the things are not on the page. But LoRA improved the highlighter part, and with more training data that part would carry further. Put that in the blog."

## The format problem, as it was explained

Three ways to turn marked words into a value, and why each fell short:

| Approach | Example | Problem |
| --- | --- | --- |
| Rule-based parser (the auditor's parsers) | "₹1,486.75" → 1486.75; "4 June 2026" → 2026-06-04 | Exact when the format is known; refuses "one lakh twenty thousand". Every new format is a new rule: fragile, as Om said |
| A learned output head (the reader, experiment 5) | Read the marked tokens' states, output digits | 147/287 on worded notes; trained jointly it dragged marking from 223 to 181. Small heads recall, they do not read |
| Plain Gemma rewrites only the marked words (experiment 7 plan) | "one lakh twenty thousand" → forced digits-only decoding → "120000"; code writes 120000 back in words (Western and Indian systems) and checks it matches the marked words | Works in principle, about 6–10 tokens. But it is a **second LLM call**, which is Om's point: the fast system is not self-sufficient |

The honest summary that came out of this: parsing a format is translation, which a generative LLM already does well; a pointing architecture cannot do it on its own.

## Experiment 7: built, tested, never run

Built after "Add 2 and 3", cancelled before results when Om challenged the direction. The code stays in the repo and is tested.

- **Grammar-generated training sentences** (`experiments/chains.py`, `G_*` tables, `grammar_training_variants`): openings, step clauses, joins and wordings split into train and test sets, amounts in five currencies, five date styles, eight two-step combinations held out entirely (`HELD_OUT_PAIRS`). 1,120 generated fields, all reproducing their gold answers.
- **A one-operand step head** (`StepHead(pair_context=False, step_embedding=False)`): sees only the operand's window, so it has to learn blocks rather than pairs.
- **Snapping** (`values.snap_char_spans`): widen a mark to the whole number or date it touches ("3" inside "3,720.78" becomes "3,720.78").
- **The Gemma rewrite** (`experiments/rewrite.py`): adapters off, a prompt holding only the marked words, digits-only constrained decoding, backward check with `agrees`.
- **Why cancelled:** Om's objection that we were tuning to our own generator. More templates make the model better at our templates; they do not show it can read the world. The SROIE result later confirmed the worry.

## How each system works, in one example

Note: "Subtotal Rs 2,400. 15% off, then 18% GST." Field: final price.

- **Our trained system:** one forward pass over the questions and the note → the tagger marks "2,400", "15%", "18%" → the operation head says "chain" → the step head picks minus_percent, plus_percent → `values.run_chain` → 2,407.20. Fast (0.14 s) and exact, but only when every mark and step choice is right; on new wording the marks clip and the step head guesses from memorized pairs.
- **Untrained LLM, JSON:** writes `"final_price": 2407.2` or, often, something close like 2,406. It reads well and does arithmetic in its head.
- **LLM reads, code computes:** writes `{"inputs": [2400, 15, 18], "steps": ["minus_percent", "plus_percent"]}` → `values` computes 2,407.20 exactly.

## Key numbers to quote (all in note 14)

- LoRA highlighter: exact evidence 116/293 → 195/293 on never-seen exam notes; final answers 223–226/293.
- Untrained LLM with code: exams 252/293, one-step 43/72, two-step 21/72 and 26/72, words 227/287, SROIE receipts 1,120/1,362.
- Our system on SROIE: 66/1,362 (company 3, date 63, address 0, total 0); the same run reproduced 223/293 on exams 1–3.
- Cost: 4.9 s per note for the LLM, 0.14 s for ours, on a free T4.

## Framings that landed in this session

- "**A good highlighter, not a reader.**" (Om's.) LoRA taught it where to look; it still cannot turn odd text into a value or produce what is not printed.
- "**A pointer cannot write.**" The architectural limit in four words.
- "**It generalized to new notes, not to new kinds of documents.**" Why the LoRA success and the receipt collapse are both true.
- "**It isn't that a 2B model beat our system; our system *was* that 2B model.**" Narrow training on synthetic notes made a capable model narrower.
- "**The model picks the numbers and the steps; the calculator does the arithmetic.**" The design that won.
- "**If the fast path sometimes needs an LLM call, the LLM is part of the system anyway.**" Why the hybrid pointer-plus-rewrite design was not worth finishing.
- "**Same-source exams test new examples, not new documents.**" Why every earlier score looked better than reality.
- "**The data made it a receipt failure; the architecture made it a format and calculation failure.**" Separating the two causes: printed fields on receipts failed for lack of data, and calculated or oddly formatted fields fail by design.

## The story arc for the blog, after the turn

The existing arc in [09](09-blog-draft.md) ends at experiment 6. The ending is now:

1. **Experiment 6:** exact calculations, and the step head memorizes pairs.
2. **The format problem:** rules are fragile; a learned head recalls instead of reading; a rewrite needs an LLM call.
3. **The challenge:** "an LLM with structured output is just better." We had never measured it.
4. **The baseline:** it reads better and computes worse.
5. **LLM reads, code computes:** best on every test.
6. **Real receipts:** our system 5%, the LLM 82%.
7. **What failed where:** architecture (formats, values not on the page) against data (receipts).
8. **What survives:** LoRA as a highlighter, the exact calculator, and the lessons.
9. **Economics and the recommendation.**

## For the next agent

- **Product direction:** LLM reads, code computes (`experiments/structured_baseline.py`: `SYSTEM_CALC`, `evaluate_calculation`; `experiments/kaggle_hybrid_job.py`).
- **Do not restart** synthetic-template work on the trained system to chase these numbers.
- **SROIE:** has been used for both systems; it is no longer held back. Never commit receipt text or labels; `tests/test_sroie.py` checks `outputs/`.
- **Exam 4:** still untouched; use it once, for both systems side by side.
- **Open fixes for the hybrid:** schema-constrained decoding (lm-format-enforcer failed to import on Kaggle's transformers), date steps the model cannot skip, and one-step median and average.
