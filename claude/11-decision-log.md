# 11 — Decision log

Everything decided in this session, in order, including the wrong turns. Dates are 7 November 2026.

| # | Decision | Why | Outcome |
| --- | --- | --- | --- |
| 1 | Read the whole repo and git history, including the deleted marking test | The handoff said "different way of using the heads" without saying how | Found the marking test had scored 42 against the four heads' 9 |
| 2 | Diagnose before building: causal order, recall versus read, selection rule, data, loss/decoder mismatch | The user asked what went wrong | See 02 |
| 3 | Plan: point → read → combine, keep all four heads in a chain | Keeps marking for what it can express, adds learned paths for the rest | See 03 |
| 4 | Re-explained in plain words several times, with a real bill | The user did not follow the first, dense version | The question-first explanation, then the "why not the question vector" explanation |
| 5 | Run cheap checks first (probe, question order) before building the full pipeline | Each could kill the plan cheaply | Both ran in experiment 1 |
| 6 | Re-audit the raw files for labels | Older stored files lacked evidence for add_days and days_between | Exact match to stored rows, plus full evidence |
| 7 | Exam 4 out of everything | Keep one clean final grade | Enforced in code and tests |
| 8 | Modal via GitHub Actions | The container could not reach Kaggle, Hugging Face or Modal; Actions could | Blocked: newline in secrets, paused workspace, then a card needed for GPUs |
| 9 | Budget guard: cap timeouts, GPU only on commits marked `[run]` | The user had $30 | **Mistake**: the commit that explained the marker had `[run]` in its message and triggered a run. Fixed to "starts with `[run]`". It failed before any spend |
| 10 | Modal CPU | Card-free | Cancelled when the user asked for Kaggle |
| 11 | Kaggle directly from the session: new environment with Kaggle hosts, token as a network secret | Free GPU, no card; the user asked | Works. The pasted token was not used; the user was told to revoke it |
| 12 | CLI wrapper skipping token introspection | The proxy fills headers, not bodies | Works; in `experiments/kaggle_cli.py` |
| 13 | Class weights (1, 3, 3) in the tagger | A planted-signal test showed the unweighted tagger predicts "nothing here" | Caught before GPU time |
| 14 | Test BOS as well as order | Hunch that missing BOS hurt every earlier round | **Wrong**: BOS made it worse |
| 15 | After experiment 1: the frozen representation is the bottleneck | Same ceiling for every head; probe 0/180; question first no net gain | Recommended a cheap follow-up (A) first; the user chose LoRA (B) |
| 16 | LoRA r 16 on q/k/v/o, question first, frozen baselines in the same run on the same split | A fair, attributable comparison | 116 → 195 of 293 |
| 17 | Uninstall torchao on Kaggle; save baselines before risky steps | Version 1 crashed and lost its baselines | Version 2 succeeded |
| 18 | Final answers with the auditor's own parsers and arithmetic, verified 1,309/1,309 on gold | Any error must be the model's, not the calculator's | Experiment 3 |
| 19 | Operation head only where the type allows several operations | Most types have one operation; number and date have several | Experiment 3 |
| 20 | Select epoch and bias by right final answers on held-out | Optimize what will be claimed | Experiment 3 |
| 21 | 10 epochs | Experiment 2 was still rising at its last epoch | Experiment 3 |
| 22 | Write these notes; refuse "cracked it" until final answers, exam 4 and real mail | Keep the blog defensible | This folder |
