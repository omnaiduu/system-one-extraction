# system-one-extraction-

Four small heads on a frozen Gemma 4 E2B: point, number, calendar, and spread. They read a document once and do not write a JSON reply.

The heads already trained are not good enough. The next work is a different way of putting those four heads to use.

A test that only marked words, and let a program do the sum or the date, failed. A number or a date is often not on the page, or it is written in words. That test is closed. Do not continue it.

Read `blogs/handoff-for-the-next-agent.md` before changing the model. `blogs/mark-and-measure.md` is how the training rows are made.

## Current experiment: does question order matter?

`experiments/` tests the first step of the new plan. The same evidence tagger is trained on Gemma states from three prompt layouts: the old one (note, then question, no BOS token), the old one with a BOS token, and question first. Settings are picked by 5-fold cross-validation over the 208 training notes. Exams 1–3 are the dev set. Exam 4 is not scored. It also checks whether a single vector can hold an amount's digits.

It runs on a Modal GPU through `.github/workflows/modal-experiment.yml`, which needs the repository secrets `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, and `HF_TOKEN` if the Gemma licence needs it. Results land in `outputs/experiments/`.
