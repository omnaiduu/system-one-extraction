# Claude's notes

This folder is the working record of the session that rebuilt the extraction model's approach, from 7 November 2026. **If you only read three notes, read [19](19-conclusion-a-good-highlighter.md) (the conclusion), [14](14-results-at-a-glance.md) (every number) and [09](09-blog-draft.md) (the story and the claims).**

**Conclusion (notes 17–20):** the trained extractor is **a good highlighter, not a reliable reader.** LoRA genuinely improved marking on new notes (exact evidence 116 → 195 of 293; final answers 223/293), but it collapsed on real receipts (66/1,362). Pointing cannot handle odd formats or values not on the page without fragile rules or a second LLM call. **Current best system:** untrained Gemma 4 E2B writing JSON, with calculated values returned as inputs and steps that our `values` code computes. It scores 252/293 on exams 1–3 and 1,120/1,362 on SROIE. Read [19](19-conclusion-a-good-highlighter.md) for the verdict and [20](20-session-record-for-the-blog.md) for how the session got there. It is written for two readers:

- **Om, writing the engineering blog.** Everything the blog needs is here: the starting point, the diagnosis, the design decisions and the reasons, the experiments with their settings and numbers, the mistakes, and the claims that can and cannot be made.
- **The next agent.** Read this before touching the model. It tells you what was tried, what was learned, which files matter, and how to run things.

The older notes are still true for their era and are worth reading after this: `blogs/mark-and-measure.md` (how the training data is made, rounds 1–4) and the deleted handoff in git history (`git show 57eae5d:blogs/handoff-for-the-next-agent.md`, the marking-test era).

## Status at a glance

| Question | Answer | Where |
| --- | --- | --- |
| Can a small head read an amount's digits from one Gemma vector? | No. 0 of 180 on held-out notes, 144 of 144 on training notes. | [05](05-experiment-1-question-order.md) |
| Does putting the question before the note help a *frozen* Gemma? | Barely. Fewer decoy picks, more misses, no net gain. | [05](05-experiment-1-question-order.md) |
| Does letting Gemma learn (LoRA) help? | Yes, by a lot. Evidence exactly right on never-seen exams went from 116/293 (40%) to 195/293 (67%). | [06](06-experiment-2-lora.md) |
| Is the LoRA model memorizing? | No sign of it. It fits training notes slightly worse than the frozen tagger (273/300 vs 300/300) and new notes much better. | [06](06-experiment-2-lora.md) |
| Final answers (the actual fee, the actual due date)? | **226/293 (77%) on exams 1–3**, against 139/293 for the marking test with its hand rules. Operation choice 102/103. | [07](07-experiment-3-final-answers.md) |
| All fields in one pass, and confidence scores? | **223/293**, within noise of one forward per field, 3× faster. Confidence AUROC 0.861; answers at ≥0.99 confidence are 194/212 right. | [12](12-one-pass-confidence-and-real-receipts.md) |
| Can the model *emit* values that are not on the page, and read numbers written in words? | **No, not with a learned head.** Emit 0/72 (0/50 within 1%); the words reader is unreliable; joint training hurt marking (223 → 181; detached 205). The operation head does recognize emit questions (103/103). | [13](13-reader-and-emit.md) |
| Can it compute values that are not printed, from facts that are (percent off, average, date minus weeks, two-step chains)? | **On training wording, yes**: 152/168, blocks 137/137. On new wording, one step 39/72; two steps 9/72 (trained combinations) and 4/72 (new combinations). Marking new sentences and composing blocks are the gaps; the arithmetic never is. | [16](16-computed-fields-and-chains.md) |
| Is an untrained LLM with structured output better? | **At reading, yes.** Same Gemma 4 E2B, no training, JSON: 250/293 on exams 1–3 (ours 223), words 216/287 (ours 147), SROIE receipts 1,109/1,362. **At computing, no**: one-step 22/72 (ours 39), two-step 2/72 and 0/72 (ours 9 and 4). And 34× slower (4.65 s against 0.14 s a note). Points to: LLM reads, code computes. | [17](17-structured-output-baseline.md) |
| LLM reads, code computes? | **Best on every test.** The LLM gives inputs and steps; our `values` code computes. Exams 1–3 252/293, one-step 43/72, two-step 21/72 and 26/72 (ours 39, 9, 4), words 227/287, receipts 1,120/1,362. About 5 s a note. | [18](18-llm-reads-code-computes.md) |
| Real documents? | **Our system fails on them.** SROIE test receipts, zero-shot: ours 66/1,362 (5%); untrained Gemma with JSON and code for arithmetic 1,120/1,362 (82%). The same run scored exams 1–3 at 223/293, so it is not a loading bug. | [18](18-llm-reads-code-computes.md) |
| Exam 4? | Not touched. It is kept for one final grade. | [10](10-next-steps.md) |
| Real customer mail? | None on disk. SROIE receipts are the real-document test. | [09](09-blog-draft.md) |

## Reading order

1. [01 — Where we started](01-where-we-started.md): the project, the four heads, the marking test, the scores.
2. [02 — The diagnosis](02-diagnosis.md): what was wrong, explained from first principles.
3. [03 — The plan](03-the-plan.md): point, read, combine; and how to measure honestly.
4. [04 — Compute and infrastructure](04-compute-and-infrastructure.md): Modal, GitHub Actions, Kaggle, the network environment, the token, costs. How to run a kernel.
5. [05 — Experiment 1: question order on frozen Gemma](05-experiment-1-question-order.md)
6. [06 — Experiment 2: LoRA](06-experiment-2-lora.md)
7. [07 — Experiment 3: final answers, 10 epochs](07-experiment-3-final-answers.md)
8. [08 — File map](08-file-map.md): every file this session added or relies on.
9. [09 — Blog draft and claims](09-blog-draft.md): the story, the figures, what may and may not be said.
10. [10 — Next steps and open questions](10-next-steps.md)
11. [11 — Decision log](11-decision-log.md): every decision in order, including the wrong turns.
12. [12 — One pass, confidence, and real receipts](12-one-pass-confidence-and-real-receipts.md)
13. [13 — The learned reader and the emit head](13-reader-and-emit.md): what was built, why it failed, what to do instead
14. [14 — Every result in one place](14-results-at-a-glance.md): the tables for the blog
15. [15 — Compared with Cloudflare's Clairic](15-compared-with-cloudflare.md): capability by capability, and what would close the gap
16. [16 — Experiment 6: computed fields and chains](16-computed-fields-and-chains.md): exact one-step operations and two-step chains; what worked (the arithmetic, the choices on trained wording) and what did not (new wording, composing)
17. [17 — The structured-output baseline](17-structured-output-baseline.md): untrained Gemma writing JSON, against our system; it reads better, computes worse, and is 34× slower
18. [18 — LLM reads, code computes](18-llm-reads-code-computes.md): the LLM gives inputs and steps, code computes; it beats our system everywhere, and our system scores 5% on real receipts
19. [19 — Conclusion: a good highlighter, not a reader](19-conclusion-a-good-highlighter.md): the verdict; where the LLM is good and why; where the architecture fails (formats, values not on the page) and where only the data failed (receipts); LoRA's real success; economics; lessons
20. [20 — Session record for the blog](20-session-record-for-the-blog.md): Om's questions and challenges in order, how each idea was explained, experiment 7 (built, not run), framings, the new story arc

## Ground rules that still hold

- **Exam 4 is not a development set.** No code in `experiments/` loads it. `make_kaggle_kernel.py` leaves its files out of the kernel. Grade it once, at the end, and say so.
- **Exams 1–3 are now the dev scoreboard.** They had already been read by earlier rounds, so they are openly "dev", not a clean test. They were never used to choose a setting in this session: epochs and cut-offs are picked on a held-out fold of the training notes.
- **Report evidence and final answers separately.** "Marked the right words" and "gave the right answer" are different numbers.
- **The writer is out of tokens.** No new notes were written in this session. All gains came from the model, not from more data.
- **Gemma's base weights are never changed.** LoRA adds small separate weights next to them.
