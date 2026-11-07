# system-one-extraction-

Four small heads on a frozen Gemma 4 E2B: point, number, calendar, and spread. They read a document once and do not write a JSON reply.

The heads already trained are not good enough. The next work is a different way of putting those four heads to use.

A test that only marked words, and let a program do the sum or the date, failed. A number or a date is often not on the page, or it is written in words. That test is closed. Do not continue it.

**Start with `claude/README.md`**: the record of the point-read-combine work, the LoRA result, and how to run everything. Then read `blogs/handoff-for-the-next-agent.md` for the earlier era. `blogs/mark-and-measure.md` is how the training rows are made.

## Current experiment: does question order matter?

`experiments/` tests the first step of the new plan. The same evidence tagger is trained on Gemma states from three prompt layouts: the old one (note, then question, no BOS token), the old one with a BOS token, and question first. Settings are picked by 5-fold cross-validation over the 208 training notes. Exams 1–3 are the dev set. Exam 4 is not scored. It also checks whether a single vector can hold an amount's digits.

It runs as a private Kaggle GPU kernel. `python experiments/make_kaggle_kernel.py OUT_DIR` packs the code and the raw notes (exam 4 left out) into one script, and `kaggle kernels push -p OUT_DIR` starts it. The kernel writes `question_order.json` and `question_order.md`; copies go in `outputs/experiments/`. `.github/workflows/tests.yml` runs the tests on each push.

## Next experiment: LoRA

`experiments/lora_tagger.py` and `experiments/kaggle_lora_job.py` train small LoRA adapters inside Gemma together with the evidence tagger, question first, and compare against frozen taggers on the same split. Build it with `python experiments/make_kaggle_kernel.py OUT_DIR lora`. Results go in `outputs/experiments/lora_results.md`.
