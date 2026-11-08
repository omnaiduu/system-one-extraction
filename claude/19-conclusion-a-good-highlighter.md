# 19 — Conclusion: a good highlighter, not a reader

The verdict of the project, written after notes 17 and 18 and a long discussion with Om about what they mean. If the blog has one closing section, this is it. Numbers are in [14](14-results-at-a-glance.md) and [18](18-llm-reads-code-computes.md).

## The one-paragraph conclusion

Our trained System One extractor (Gemma 4 E2B with LoRA, a tagger and small heads) is **a good highlighter**: LoRA really taught it to find and mark the right words, on notes it had never seen. It is **not a reliable reader**. Turning marked words into a value needs fragile rules or a second LLM call whenever the format is unusual, and it cannot produce a value that is not printed on the page without a stack of extra heads that did not generalize. Those are limits of the architecture, and no amount of training data removes them. The untrained 2B LLM writing structured JSON, with our code doing the arithmetic, beat it on every test, including real receipts (1,120/1,362 against 66/1,362). For a product, use the LLM plus code. The trained highlighter is only worth building later, as a fast path for printed values, trained on real documents, when volume makes cost matter.

## How we got here

1. **The challenge.** Om: "This is not prod ready, we are tuning for a dataset. An LLM with structured output is just better at it, unless you want to prove me wrong." It had never been measured.
2. **The baseline (note 17).** The same model, Gemma 4 E2B, untrained, with a full system prompt and worked examples, writing JSON. It read better (250 vs 223 on exams 1–3; 216 vs 147 on numbers in words) but computed worse (22 vs 39 one-step; 2 and 0 vs 9 and 4 two-step), because it does arithmetic in its head and is close but not exact.
3. **LLM reads, code computes (note 18).** The LLM returns `{"inputs", "steps"}` for anything calculated, and our `values` code computes it exactly. Best on every row: exams 252, one-step 43, two-step 21 and 26, words 227, receipts 1,120.
4. **Our system on real receipts (note 18).** 66/1,362 (5%). Not a bug: the same run reproduced 223/293 on exams 1–3.

## Where the LLM is good, and why

- **Reading.** It was pretrained on a huge variety of text: receipts, invoices, OCR, capitals, Indian and Western number words, every date style. It already knows what a total, a company name and an address look like on a receipt. Nothing we did narrowed it.
- **Formats.** "One lakh twenty thousand", "the second of April, 2026", "15/01/2019": it writes the normalized value directly. Writing is the one mechanism that covers every format.
- **Values that are not on the page.** It can say *which* numbers and *which* steps ("2400, 15, 18: minus 15%, plus 18%"). It is bad only at the arithmetic itself, and code fixes that exactly.
- **New combinations.** It was never trained on our step pairs, so it does as well on unseen two-step combinations (26/72) as on the trained ones (21/72). Our step head memorized pairs and fell from 9 to 4.
- **Where it still slips:** it does date arithmetic in its head despite the rule (38 of its 39 rule breaks were dates), writes median as add-then-divide (1/9), and sometimes picks the wrong inputs. All are reading or prompt-following problems, and all are fixable without training (schema-constrained decoding, one-step forms).
- **Cost:** about 5 s and 64 generated tokens per note on a free T4.

## Where our architecture fails, at the level of design

Our design points: it marks words in the document, then code turns the marked words into a value. That works only when the answer is printed, in a format code can read. Everything else needed a patch:

| Case | What our system needed | Result |
| --- | --- | --- |
| Printed, plain format ("₹1,486.75", "2026-06-04") | Tagger + parser | Works (on notes like the training notes) |
| Printed, unusual format ("one lakh twenty thousand", "the second of April") | Fragile hand rules, or a **second LLM call** to rewrite the marked words (experiment 7, built but not run) | Learned reader: 147/287 on worded notes; the fix is a call to an LLM anyway |
| Not printed, must be computed (a due date, a price after discount and tax) | Operation head + step head + calculator | Exact on training wording, 39/72 on new wording, 4/72 on new two-step combinations |
| Not printed, must be estimated or emitted | A learned emit head | 0/72 |

Each row past the first is a patch, because the core can only point. If the fast system must call an LLM whenever a format is odd, the LLM is part of the system anyway, and once it is there it handles the printed values too. **This is an architectural failure, not a data problem: a pointer cannot write, however much it is trained.** (Om's point, and correct.)

## Where LoRA tuning was a real success

- **It made a much better highlighter.** Exact evidence on never-seen exam notes went from 116/293 (frozen Gemma, 40%) to 195/293 (67%) with 5.4M LoRA weights, about 0.1% of the model. With reading and combining in code: 223–226/293 final answers.
- **It was not memorizing.** It fit the training notes slightly *less* well than the frozen tagger and did much better on new notes; 271 of 293 exam questions used wording never seen in training.
- **It generalized to new notes, not to new kinds of documents.** The exams are new notes written in the same style as the training notes. Receipts are OCR lines in capitals, addresses over four lines, a dozen amounts (GST, rounding, cash, change). It had never seen anything shaped like that. Company 3/341, address 0/332, total 0/347 (it marked 7–22 scattered numbers per receipt), date 63/342. Its confidence did not warn either: of the 426 answers it rated at least 0.99, 67 were right.
- **That receipt failure is a data failure, not the architectural one.** Company, address and total are printed word for word on the receipt; pointing is exactly the right tool for them. The model failed because its training data was 208 synthetic notes from one writer. A highlighter trained on real receipts would very likely work on receipts. That is not tested here, and it would then be a receipt highlighter, not a general one.
- **So the clue we found:** LoRA improves the highlighter part cheaply and genuinely. With more varied, real training data, the highlighter would carry over to more kinds of documents. Training data cannot fix the reading and computing part.

## Why the experiments looked like success before

Every test before SROIE was drawn from the same kind of notes as the training data: new notes, new names and amounts, but the same writer, sentence shapes and field types. That tests generalization to new *examples*, not to new *documents*. We never ran the second kind of test until the end. Exams 1–3 were also read many times during development.

## Was it a waste?

No, though the product will not be our trained model.

- **The answer is measured, not guessed.** Many extraction projects never run the plain-LLM baseline or test on a different kind of document. This one did both, before shipping.
- **The calculator survives.** `values` (exact blocks, `run_chain`, typed parsing and formatting) is the part that makes "LLM reads, code computes" exact. It came out of experiments 3 and 6.
- **The lessons are general** (below), and a project that says "we fine-tuned and it worked" teaches less than one that says where it stopped working.
- **It is not "a 2B model beat our system".** Our system *was* that 2B model plus heads. Narrow training on synthetic data made a capable model narrower.

## Is a trained extraction model economically viable?

For this project, now: no.

- The LLM is already cheap: about 5 s per document on a free-tier GPU, a fraction of a cent per document at cloud prices.
- A trained highlighter (0.14–0.17 s per note, about 30–35× cheaper) pays off only at very high volume or tight latency.
- Even then it needs many real labelled documents. They can come from the LLM itself (distillation: run LLM-reads-code-computes over real traffic, train the highlighter on its answers, test on a kind of document it never saw).
- Systems like Cloudflare's Clairic have that volume and that traffic to train on. We have neither yet.

## Recommendation

1. **Product:** LLM reads, code computes (`SYSTEM_CALC` + `values`).
2. **Make it follow its rule:** schema-constrained decoding so a calculated date cannot come back as a literal, one-step forms for median and average.
3. **Grade once on exam 4**, both systems side by side.
4. **Only if cost hurts:** distill a highlighter for printed values from the LLM's answers on real documents, with the LLM kept for anything calculated or oddly formatted.

## Lessons for the blog

1. **Run the plain-LLM baseline first.** It took one prompt and beat months of design.
2. **Test on a different kind of document before believing a score.** New examples from the same source are not a generalization test.
3. **Training on synthetic data from one writer teaches that writer.** LoRA learned the notes, not the task.
4. **Pointing is fast and right for printed values; writing is needed for everything else.** A system that must sometimes write needs a writer anyway.
5. **Let models read and code compute.** The LLM's arithmetic was close but not exact; with code, calculations went from 22 to 43 (one step) and from 0–2 to 21–26 (two steps).
6. **Confidence learned on one kind of data does not transfer.** 0.99 meant 92% right on the notes and 16% right on receipts.
