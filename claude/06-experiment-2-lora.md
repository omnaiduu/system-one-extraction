# 06 — Experiment 2: LoRA

Kaggle kernel `ommnaidu/gemma-e2b-lora-evidence`, version 2 (version 1 crashed, see below). One T4, float16 base, 2,963 s in total, LoRA training 2,341 s. Results: `outputs/experiments/lora_results.{md,json}`; weights `outputs/experiments/lora_adapter.pt` (24 MB: adapters + tagger head). Code: `experiments/lora_tagger.py`, `experiments/kaggle_lora_job.py`.

## Why LoRA (the reasoning, in order)

1. **Same ceiling everywhere.** Four heads, the highlighter before its rules, and the new tagger: very different heads, all on frozen Gemma, all at a similar level. When different heads hit the same wall, the wall is what they share.
2. **The probe.** 0/180: the information is in there somewhere, but not in a form a small head can read. Gemma's states are organized for next-word prediction, the only job it was trained on.
3. **Question first barely helped.** Gemma could see the question, but nothing had trained it to turn "the question says fee" into "flag this number".
4. **The memorizing pattern.** Frozen taggers: 300/300 on training, about 40% on new notes. A small head over fixed features memorizes 208 notes. LoRA changes how Gemma reads every token through shared low-rank weights; the cheapest way to lower training error with shared weights is a general rule.
5. **Why not full fine-tuning.** 5B weights and 208 notes would memorize at once and would not fit a T4. LoRA trains about 5.4M weights, about 0.1%.
6. **Standard practice.** Extractive QA (SQuAD-style) fine-tunes the reader; frozen-feature probing is known to fall short for span extraction.

**The old objection to LoRA** (`blogs/mark-and-measure.md`): "a LoRA memorizes one fixed schema; a new field needs a new tune." That holds for a fixed form. Here the instruction is plain text inside the prompt, so the model learns to read instructions. **271 of the 293 dev questions use instruction wording that never appears in training.**

## Setup

| Setting | Value |
| --- | --- |
| Prompt | Question first (`first` layout), BOS |
| Adapters | LoRA r = 16, alpha = 32, dropout 0.05, on `q_proj`, `k_proj`, `v_proj`, `o_proj` of the text layers only (`model.language_model.*`); 100 modules matched (fewer than 35 × 4 = 140; probably layers that reuse earlier keys/values; not checked), 5,357,568 trainable parameters |
| Base weights | Frozen, float16 |
| Adapter weights | float32 |
| Head | A fresh `Tagger` (same as experiment 1) on layers 17, 26, 35 |
| Precision | autocast float16, `GradScaler`; no step was skipped for inf/NaN |
| Memory | Gradient checkpointing (non-reentrant); forward through the text tower only (`Gemma4TextModel`), so no 262k-wide logits |
| Optimizer | AdamW, lr 2e-4 adapters, 1e-3 head, weight decay 0.01, grad clip 1.0 |
| Batch | 1 question per step, gradient accumulation 8 |
| Epochs | 5 |
| Split | `fit` 823 questions (training notes outside fold 0), `held` 193 (fold 0), dev: exam 1 109, exam 2 92, exam 3 92 |
| Selection | Epoch and decode bias chosen on `held` only; bias added to the begin/inside logits, from {−1, −0.5, 0, 0.5, 1, 1.5, 2, 3} |
| Baselines | Same run, same split: fresh frozen taggers for `last_nobos` and `first`, 60 epochs scored every 5, same bias selection |

## Results

| Condition | Best epoch | Bias | Training sample (300) | Held (193) | Exam 1 | Exam 2 | Exam 3 | **Exams 1–3** | Wrong place | Missed | Partial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen, old layout | 50 | −0.5 | 300 | 107 | 44/109 | 37/92 | 35/92 | **116/293 (40%)** | 65 | 36 | 74 |
| Frozen, question first | 40 | −0.5 | 300 | 103 | 42/109 | 30/92 | 33/92 | 105/293 (36%) | 84 | 47 | 48 |
| **LoRA, question first** | 5 | 1.0 | 273 | **160** | **72/109** | **63/92** | **60/92** | **195/293 (67%)** | **27** | **23** | 47 |

LoRA held-out exact by epoch (at the best bias): 91, 134, 145, 155, 160. Still rising at the last epoch. Training loss: 0.253, 0.060, 0.024, 0.010, 0.011.

Frozen held-out scores jumped around between checkpoints (77 to 107), another sign the frozen features are the limit.

Evidence exact on exams 1–3, by operation:

| Operation | Frozen, old | Frozen, first | LoRA |
| --- | --- | --- | --- |
| copy (text) | 20/95 | 17/95 | **51/95** |
| read_date | 30/60 | 29/60 | **49/60** |
| list | 8/37 | 13/37 | 16/37 |
| missing | 27/29 | 20/29 | 28/29 |
| days_between | 9/14 | 8/14 | 10/14 |
| product | 7/11 | 3/11 | 9/11 |
| sum | 4/10 | 1/10 | 8/10 |
| median | 4/9 | 2/9 | 4/9 |
| read_number | 0/9 | 3/9 | **8/9** |
| subtract | 3/7 | 4/7 | 5/7 |
| add_days | 3/6 | 3/6 | 3/6 |
| read_time | 1/3 | 1/3 | 2/3 |
| read_datetime | 0/3 | 1/3 | 2/3 |

## Real examples (exam 2, never seen in training)

- `e2-support-02`, "Ticket the customer asked the desk to use". The note says "My ticket is UB-2745. The desk blotter still shows ticket UB-2749 from last winter, so please do not mix them up." The frozen tagger marked other words; LoRA marked exactly `UB-2745`.
- `e2-ops-05`, "Date the pit was packed and left". The note has "first went into the pit on 2 February 2026" and "packed the joint and left on 20 February 2026". The frozen tagger marked the right date plus an extra earlier stretch; LoRA marked only 20 February 2026.
- `e2-bills-03`, "Figure copied from the total box". The answer is `9133.40`, next to a crossed-out 1507.00. The frozen tagger marked a stray token earlier in the note and only the last piece of the figure; LoRA marked the whole figure.

## What it means

- **The largest gain in the project, under a fair comparison.** Same notes, same split, same head, same selection rule; only the adapters differ.
- **It learned a skill, not the notes.** Training fit went *down* (300 → 273 of 300) while new-note accuracy went up (116 → 195). That is the opposite of every earlier round.
- **Decoys stopped working as often.** Wrong-place marks: 65 → 27.
- **Weak spots.** Lists (16/37) and edge clipping (47 partial).
- **Undertrained.** The best epoch was the last one, hence experiment 3.

## Comparison with the earlier best (be careful with the metric)

The highlighter, with its hand-written decode rules, scored final values exam 1 48/109, exam 2 42/92, exam 3 49/92 (139/293). LoRA evidence: 72, 63, 60 (195/293), with no hand rules. Same exams, same field counts. But those are final values and these are exact evidence. Exact evidence is usually the stricter test, because a slightly wrong mark can still parse to the right value. The like-for-like comparison is final answers, in experiment 3.

## The failed first attempt

Version 1 crashed at `inject_adapter_in_model`: "Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported." peft probes torchao even when it is not used. Fix: uninstall torchao in the kernel stub. The baselines from that attempt (held 105 and 104 of 193) were only in the log, so the job now saves baseline results before the LoRA step.
