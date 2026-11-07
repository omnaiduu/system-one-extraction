# 09 — Blog draft and claims

Material for Om's engineering blog. The structure is a suggestion; the claims section is not optional.

## Working title ideas

- "Point, read, combine: teaching a 2B model to extract instead of recall"
- "Why our extraction heads memorized, and what fixed it"
- "From 9 of 77 to …" (fill in after the exam 4 grade; do not use before)

## The story in one paragraph

We wanted a Clairic-style extractor on Gemma 4 E2B: read a note once, return typed values, no generated JSON. Four rounds of small heads on a frozen Gemma fit their training notes almost perfectly and failed on new ones (9 of 77 on the last exam). The diagnosis: the note was read before the question, and the number heads tried to recall digits from one blended vector. A probe confirmed it: 0 of 180 amounts readable from a single vector on held-out notes, 144 of 144 on training notes. Moving the question in front of the note barely helped a frozen Gemma. Every head built on frozen features hit the same ceiling. Letting Gemma learn through 5.4M LoRA weights (about 0.1% of the model), trained to mark the evidence, took exact evidence on never-seen exams from 40% to 67%. It fit the training notes slightly *less* well, the opposite of memorizing. A read-and-combine step built on the auditor's own arithmetic turns the marks into final answers.

## Suggested sections

1. **What we wanted.** Clairic's output types; why not JSON generation (it can invent an order id); why a frozen reader with heads.
2. **How the data is made.** Mark and Measure, briefly (link `blogs/mark-and-measure.md`): brackets, the auditor, the leak rule, decoys.
3. **Four heads that memorized.** The rounds table (01). The tell: about 96% on training, about 12% on new notes. Wrong amounts were the right shape with the wrong digits.
4. **Why: two engineering mistakes.**
   - Causal attention: the note was read before the question (diagram: arrows only point backwards).
   - Recall versus read: averaging the digit tokens destroys their order; n ≪ d makes memorizing free; the controlled comparison (calendar head 1/20, point + parse 14/20 on the same states).
   - The selection rule that picked the most-memorized epoch.
5. **The marking test, and the objection that closed it.** Fair in principle (words, uncoded rules), untestable on this data (no field in words, every unprinted value coded). Lesson: do not frame it as either/or.
6. **Experiment 1: the probe and question order.** 0/180. Question first: fewer decoys, more misses, no net gain. The BOS hunch that was wrong. "Same ceiling for every head" as the turning point.
7. **Experiment 2: let the reader learn.** LoRA setup (06), the fair-comparison design (baselines retrained in the same run, same split), the table, real decoy examples (UB-2745 vs UB-2749), the training-fit-down / new-notes-up signature.
8. **Experiment 3: final answers.** The read-and-combine step; the 1,309/1,309 check; the results (pending).
9. **Exam 4** (after it is graded once).
10. **What it does not prove yet.** See below.
11. **The plumbing, briefly.** Free Kaggle T4s; a whole experiment in one 170 KB script; no card needed. The detours (Modal needed a card) are an honest aside.

## Figures and tables to make

- Diagram: question-last versus question-first under causal attention (which tokens can see the question).
- Diagram: point → read → combine on `e4-bills-02` (`4 June 2026` + `21` → `2026-06-25`).
- Bar chart: the "memorization gap" (training versus new notes) for four heads, frozen tagger, LoRA.
- Table: frozen versus LoRA on exams 1–3, with error buckets (06).
- Line: LoRA held-out exact by epoch (91, 134, 145, 155, 160; then experiment 3's 10 epochs).
- Example box: the UB-2745 / UB-2749 note with frozen and LoRA marks.

## Claims: allowed

- On 72 synthetic notes never used in training or for any choice, exact evidence went from 116/293 to 195/293 when LoRA adapters were trained inside Gemma, with the frozen baseline retrained on the same split in the same run.
- The LoRA model fit the training notes slightly less well than the frozen tagger (273/300 vs 300/300) while doing much better on new notes.
- A linear readout could not recover an amount's digits from a single Gemma vector on held-out notes (0/180) while fitting training perfectly (144/144).
- 271 of the 293 dev questions use instruction wording not seen in training.
- Everything ran on free Kaggle T4 GPUs.

## Claims: not allowed (yet)

- **"We cracked it" / "as good as Clairic".** No real customer mail has been tested; Clairic's numbers are on its own data and tasks; we have no comparable benchmark.
- **A production accuracy.** All of this is synthetic notes from one writer, with repeated skeletons.
- **Final-answer accuracy** until experiment 3 is in, and **exam 4** until it is graded once.
- **"Handles numbers written in words."** Untested: no such field exists in the data.
- **"Exams 1–3 are a clean test."** They were read by earlier rounds; they are dev. They were not used to choose anything in this session.
- **"LoRA is new."** It is standard. What is particular here is the combination: bracket-measured evidence labels, question-first LoRA tagging, and exact typed arithmetic from the auditor.
- **Comparing 195/293 evidence directly to the highlighter's 139/293 final values as like for like.** Different metrics; use experiment 3 for the like-for-like comparison.
- The older "must not say" list in `blogs/mark-and-measure.md` still applies.

## Quotes and framings from the session that may help

- "A student who memorizes gets 100% on homework and fails the exam. A student who understands gets a bit less on homework and much better on the exam." The LoRA model is the second kind.
- "The question vector is someone who read the bill, put it away, and is asked for the fee to the cent. Pointing at the number on the bill and reading it off is far more reliable."
- "It is a person using a calculator. The model decides what to add; the calculator adds without mistakes."
- "When very different heads hit the same ceiling, the ceiling is what they share."
