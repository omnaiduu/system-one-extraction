# Claude's notes

This folder is the working record of the session that rebuilt the extraction model's approach, from 7 November 2026. It is written for two readers:

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
| Final answers (the actual fee, the actual due date)? | Running when these notes were written (Kaggle kernel version 3). | [07](07-experiment-3-final-answers.md) |
| Exam 4? | Not touched. It is kept for one final grade. | [10](10-next-steps.md) |
| Real customer mail? | None on disk. Every number here is on synthetic notes. | [09](09-blog-draft.md) |

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

## Ground rules that still hold

- **Exam 4 is not a development set.** No code in `experiments/` loads it. `make_kaggle_kernel.py` leaves its files out of the kernel. Grade it once, at the end, and say so.
- **Exams 1–3 are now the dev scoreboard.** They had already been read by earlier rounds, so they are openly "dev", not a clean test. They were never used to choose a setting in this session: epochs and cut-offs are picked on a held-out fold of the training notes.
- **Report evidence and final answers separately.** "Marked the right words" and "gave the right answer" are different numbers.
- **The writer is out of tokens.** No new notes were written in this session. All gains came from the model, not from more data.
- **Gemma's base weights are never changed.** LoRA adds small separate weights next to them.
