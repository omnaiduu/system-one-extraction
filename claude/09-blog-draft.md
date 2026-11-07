# 09 — Blog draft and claims

Material for Om's engineering blog, or for an agent drafting it. The structure is a suggestion. The claims section is not optional. All numbers are in [14](14-results-at-a-glance.md). The Cloudflare comparison is in [15](15-compared-with-cloudflare.md).

## Working titles

- "Point, read, combine: teaching a 2B model to extract instead of recall"
- "Our extraction heads memorized. Here is what fixed it, and what did not."
- "From 9 of 77 to 77%: building the extraction core of a System One model on a free GPU" (only with the exam-4 grade in hand; otherwise say "on notes it never saw")

## The story in one paragraph

We wanted a Clairic-style extractor on Gemma 4 E2B: read a note once, return typed values, no generated JSON. Four rounds of small heads on a frozen Gemma fit their training notes almost perfectly and failed on new ones (9 of 77 on the last exam). The diagnosis was that the note was read before the question, and the number heads tried to recall digits from one blended vector. A probe confirmed it: 0 of 180 amounts were readable from a single vector on held-out notes, against 144 of 144 on training notes. Moving the question in front of a frozen Gemma barely helped, and every head built on frozen features hit the same ceiling. Letting Gemma learn through 5.4M LoRA weights (about 0.1% of the model), trained only to mark the evidence, took exact evidence on never-seen exams from 40% to 67%, while fitting the training notes slightly *less* well. Adding a read-and-combine step built on the data auditor's own arithmetic gave 77% fully correct final answers, against 47% for the earlier marking approach with its hand-written rules. Reading each note once, with all questions in front as Clairic does, cost nothing in accuracy and two thirds of the compute. The last experiment is the cautionary tale: a learned head that "emits" values not on the page learned nothing (0 of 72), and hurt everything else when it shared the adapters.

## The arc, section by section

1. **What we wanted.** Clairic's output types; why not JSON generation (it can invent an order id); a frozen reader with small typed heads.
2. **How the data is made.** Mark and Measure (`blogs/mark-and-measure.md`): a writer brackets the evidence; a program measures spans and computes derived answers; any leak kills the note; every note has a decoy. 208 training notes, 4 locked exams of 24.
3. **Four heads that memorized.** About 96% on training, about 12% on new notes. The tell: wrong amounts were the right shape with the wrong digits (1486.75 → 2186.75).
4. **Why, part 1: causal attention.** The note came before the question, so its token states could not depend on the question. All the choosing fell to a 64-wide dot product.
5. **Why, part 2: recall versus read.** Gemma writes each digit as its own token. Averaging their states destroys the order. With n ≪ d (about 164 numbers in 1,536 dimensions), memorizing is free. The controlled comparison already in the repo: on the same states, the calendar head got 1/20 dates and point + parse got 14/20.
6. **Why, part 3: the selection rule.** Picking the epoch with the best *training* score picks the most memorized model.
7. **The marking test and its hand rules.** The right idea (mark, then compute) on a weak reader, propped up by a length cap, an ordered chain and a question block, designed on exam 4. 139/293.
8. **Experiment 1: the probe and question order.** 0/180. Question first: fewer decoy picks, more misses, no net gain. The BOS hunch was wrong. "Every head on frozen features lands at the same level" was the turning point.
9. **Experiment 2: let the reader learn.** The LoRA setup; frozen baselines retrained in the same run on the same split; 116 → 195; the training-fit-down, new-notes-up signature. 271 of 293 exam questions use instruction wording never seen in training, which answers the old "LoRA memorizes a schema" objection.
10. **Experiment 3: final answers.** Read with the auditor's parsers, combine with its arithmetic, verified 1,309/1,309 on gold evidence. An operation head chooses the operation (102/103). 226/293 with no hand rules.
11. **Experiment 4: one pass.** All questions before the note, one ANSWER line per field after it. 223/293, a third of the compute. Confidence scores: at ≥ 0.99, 92% right.
12. **Experiment 5: the emit head that learned nothing.** The motivation (values not on the page, numbers in words), the data built without writer tokens, a bug the tests caught (`parse_number` read "April the second, 2026" as 2026), the result (emit 0/72; joint training dragged marking from 223 to 181; detaching recovered 205), and the lesson: every exact number that came out right came from copying the page plus exact code.
13. **What it does not prove yet.** See the claims below and [15](15-compared-with-cloudflare.md).
14. **The plumbing, briefly.** Free Kaggle T4s; each experiment is one self-contained 170–200 KB kernel script; no card needed. The detours (Modal needed a card; a commit marker that triggered itself) as an honest aside.

## Figures and tables to make

- Diagram: question last versus question first under causal attention (which tokens can see the question).
- Diagram: point → read → combine on `e4-bills-02` (`4 June 2026` + `21` → `2026-06-25`), and the one-pass prompt layout.
- Bar chart: the memorization gap (training versus new notes) for four heads, frozen tagger, LoRA 5 epochs, LoRA 10 epochs, one pass ([14](14-results-at-a-glance.md)).
- Table: frozen versus LoRA on exams 1–3, with error buckets.
- Line: held-out score by epoch for experiments 2, 3 and 4 (noisy; why epoch selection on a held-out fold matters).
- Table: final answers by operation (experiment 4).
- Confidence: accuracy against the confidence threshold (coverage curve).
- Example boxes: UB-2745 versus UB-2749 (the decoy); "Adriatic Pipeworks d." (an edge clipped); the 45-day due date that came back empty.
- The emit failure: emitted amounts cluster around 127–167 whatever the inputs.

## Claims: allowed

- On 72 synthetic notes never used in training or for any choice, LoRA adapters trained inside Gemma raised exact evidence from 116/293 to 195/293 (5 epochs), with the frozen baseline retrained on the same split in the same run.
- With mark → read → combine and 10 epochs: 226/293 fully correct final answers (one forward per field) and 223/293 (one pass per note), against 139/293 for the earlier marking approach with hand-written decode rules on the same exams and metric.
- The LoRA model fits training notes slightly less well than a frozen tagger while doing much better on new notes.
- A linear readout could not recover an amount's digits from a single Gemma vector on held-out notes (0/180), while fitting training perfectly.
- The operation head chose the right operation on 102–103 of 103 ambiguous fields.
- Confidence ranks answers usefully (AUROC 0.86): 92% of answers at confidence ≥ 0.99 are right.
- A learned emit head trained on about 400 examples did not learn exact arithmetic (0/72 exact, 0/50 within 1%).
- Everything ran on free Kaggle T4 GPUs.

## Claims: not allowed (yet)

- **"We cracked it" / "as good as Clairic" / "we rebuilt Clairic".** Cloudflare published no architecture and no extraction accuracy; there is nothing to match against. Clairic also covers estimates with intervals, forecasts, images and real data, which this project does not.
- **A production accuracy.** All numbers are on one writer's synthetic notes with repeated skeletons.
- **Exam 4 or real-document numbers** until the final grade has run, once.
- **"Handles numbers in words" or "emits values not on the page".** Both were tested and failed.
- **"Exams 1–3 are a clean test."** They were read by earlier rounds. They are dev, although nothing in this session was chosen on them.
- **"LoRA is new."** It is standard. What is particular here is the combination: bracket-measured evidence labels, question-first LoRA tagging, exact typed arithmetic reused from the labeler, and an operation head.
- **"One pass is better than per field."** 223 versus 226 is within noise (about ±8). The claim is "as good, much cheaper".
- **Calibrated probabilities.** The confidence ranks well; it is not calibrated.
- The older "must not say" list in `blogs/mark-and-measure.md` still applies.

## Framings that worked when explaining it

- "A student who memorizes gets 100% on homework and fails the exam. A student who understands gets a bit less on homework and much better on the exam." The LoRA model is the second kind.
- "The question vector is someone who read the bill, put it away, and is asked for the fee to the cent. Pointing at the number on the bill and reading it off is far more reliable."
- "It is a person using a calculator. The model decides what to add; the calculator adds without mistakes."
- "When very different heads hit the same ceiling, the ceiling is what they share."
- "It knows *when* a value must be emitted (103/103). It cannot *do* the emitting (0/72)."

## Lessons worth a paragraph each

1. Measure the memorization gap (training versus new notes) on every run; it was the clearest signal in the project.
2. Copy and compute beats recall for exact values. True at every scale tried here.
3. Fix the reader, not the decoder. Hand rules on a weak model moved one exam; a better reader moved every exam.
4. Put a baseline in the same run, on the same split, before claiming a gain. Run-to-run noise here was about ±8 fields.
5. Losses that share parameters compete. The reader losses were about 100× the tagger's and took the adapters over.
6. Tests that encode the labeler's own invariants (gold evidence reproduces every gold answer) catch real bugs before GPU time.
