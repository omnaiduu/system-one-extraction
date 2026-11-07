# 02 — The diagnosis

What was wrong with the four-head design, from first principles. Each point was stated as a hypothesis before any experiment, and the experiments in 05 and 06 tested them.

## Background: what Gemma's "states" are

Gemma reads text as tokens (word pieces). For every token, every layer produces a vector of 1536 numbers, the hidden state. Think of it as Gemma's note about that token. The heads in this repo never see the text; they only see these notes.

Gemma is a **causal** model: the state at position t is computed from tokens 1..t only. It never looks ahead.

## 1. The document was read before the question existed

The prompt was `[note] [QUESTION: field: instruction]` (`extractor/labels.py`, `pack_prompt`). Because Gemma only looks back, the states of the note's tokens are computed before the question appears. The state on "638.40" can say "a price near the words 'job charge'". It cannot say "this is what is being asked for".

All of the choosing ("the fee, not the crossed-out quote") was left to a 64-wide dot product, trained on about 1,000 questions. The 2-billion-parameter reader was never told what to look for.

**Example used with the user.** In a note that says "printed job charge 638.40 … quote WM-3081 for 512.00 was crossed out", the note's states for 638.40 and 512.00 are computed without the question. The small head has to tell them apart on its own.

## 2. The number and calendar heads recalled instead of read

Round 3 added `DocumentReader`: attention weights over the note's tokens, then a weighted **average** of their states, added to the question vector, then an MLP that outputs digit classes.

Gemma's tokenizer writes each digit as its own token: `6 3 8 . 4 0`. An average of those states is a blend; the order of the digits is gone. The MLP cannot copy digits out of a blend. What it can do is learn "this training note → this answer".

Three reasons, stated in engineering terms:

1. **n ≪ d.** About 164 number fields, each a point in 1536 dimensions. With far fewer points than dimensions, almost any labelling is fitted exactly. Zero training loss is available without learning anything about digits.
2. **Not decodable.** The last-layer state is shaped for next-token prediction. Nothing trained Gemma to keep the third digit of an amount 100 tokens back in a fixed, readable direction. In normal generation it never needs to: it attends back to the digits at every step.
3. **No inductive bias.** The output is a value class, not a position. To output a value it never saw, the head must have learned a general vector-to-value map from about 164 examples spread over 10⁹ possible amounts.

The evidence was already in the repo: number 214/216 on training, 0/12 on exam 4. Wrong amounts were "the right kind of number with the wrong digits" (`1486.75` → `2186.75`), which is what a blend looks like.

**The controlled comparison already in the repo.** On the same frozen states, the same 200 notes, the same exam 4, the calendar head scored 1/20 dates and the point head plus a parser scored 14/20. Only the output representation changed. Copying beats classifying.

## 3. Every round saved the most-memorized checkpoint

The saved epoch was the one with the highest **training** accuracy. Validation was 4 notes, too few to rank epochs. So the selection rule picked the checkpoint that fit the training notes best, which is the most memorized one.

## 4. The data made memorizing the cheapest solution

Each note appears with exactly one set of values. Nothing in training punishes "remember this note's fee". (The planned fix, re-rendering the bracketed raw files with new values, is in 03 and 10. It has not been built yet.)

## 5. The point head was trained on one thing and scored on another

Training: independent cross-entropy on a start token and an end token. Scoring: the best legal pair, a length cap, then an ordered chain. The cap and chain were designed while looking at exam 4's misses, so 42/74 is somewhat optimistic.

## 6. About the closed marking test

Its reason for closing was correct in principle (words, and values no rule covers). But this was checked against the data:

- **No number or date field in any data file is written in words.**
- **Every unprinted value is one of the coded operations.**

So the exams cannot test that objection either way. The marking test scored 42 against 9. The conclusion drawn in this session: keep marking as the backbone for what it can express, and add learned paths for what it cannot. Do not frame it as marking versus heads.

## 7. Why the question vector alone is not the answer (the user's question)

The user asked: the last question token has seen the whole note, so why not read the answer from it? That is exactly what rounds 1–2 did (number and calendar heads read only the last question token): numbers 0/16 and dates 0/22 on exam 2. One vector has to carry which value, every digit, and their order, packed in a way a small head can unpack. Gemma never learned to pack that. Experiment 1 tested this directly with a linear probe: 0 of 180 (see 05).

The analogy used: the question vector is someone who read the bill, put it away, and is asked for the fee to the cent. Pointing at the number on the bill and reading it off is far more reliable.
