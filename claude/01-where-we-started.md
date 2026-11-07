# 01 — Where we started

## The goal

A Clairic-style extractor on **Gemma 4 E2B** (`google/gemma-4-E2B-it`). Cloudflare's Clairic (blog, 19 October 2026) reads a document once and returns typed values: a span of the original text, a number, a calendar value, a duration, a list, an estimate with a band, or a short forecast. It does not write a JSON reply. Cloudflare published the output types, not the weights or the head code.

The student here is Gemma 4 E2B: about 2.3B effective parameters, 5.1B with the per-layer embedding tables, 35 text layers, hidden width 1536. The chat model is not asked to write the answer.

## The data (unchanged this session)

"Mark and Measure" (`blogs/mark-and-measure.md`): a writer model (Grok 4.7) invents a messy human note and wraps the evidence in brackets, `[[fee]]638.40[[/]]`. `extractor/audit.py` strips the brackets, measures character spans, parses dates and amounts, computes derived answers (sum, product, subtract, add_days, days_between, median of an even list, forecast tail), and drops any note where the computed answer is already printed. The student sees only the clean text. Every note carries decoys: an unmarked string of the same kind as the answer (a crossed-out quote, an old ticket number).

| Set | Notes | Fields scored this session | Role this session |
| --- | --- | --- | --- |
| Original 48 (`data/raw`) | 48 | — | Training (val and test folded in) |
| Round 2 (`data/round2/raw`) | 98 | — | Training |
| Round 3/4 additions (`data/round3/raw`) | 62 | — | Training |
| **Training total** | **208** | 1016 questions | Training, with fold 0 held out |
| Exam 1 (`data/generalization`) | 24 | 109 | Dev scoreboard |
| Exam 2 (`data/exam2`) | 24 | 92 | Dev scoreboard |
| Exam 3 (`data/exam3`) | 24 | 92 | Dev scoreboard |
| Exam 4 (`data/exam4`) | 24 | — | Untouched, final grade only |

Forecast fields are not scored here: their answer was deleted from the page, so there is no evidence to mark.

## What had been tried before this session

### The four heads (rounds 1–4)

One frozen Gemma read; the field type chose a head:

| Head | Output |
| --- | --- |
| Point | start token, end token, missing bit |
| Number | sign and 7 + 2 digits, each a 10-way class |
| Calendar | year (16 classes), month, day, hour, minute |
| Spread | mean and width inside the caller's min/max |

Scores, each on a fresh exam scored once:

| Round | Training fields | Fresh exam |
| --- | --- | --- |
| 1 | 213/221 | 29/112 (exam 1) |
| 2 | 744/769 | 14/95 (exam 2) |
| 3 (added a document reader) | 720/769 | 15/95 (exam 3) |
| 4 (200 training notes) | 948/991 | **9/77** (exam 4); amounts 0/12, dates 1/20 |

### The marking test (the "highlighter")

The number and calendar heads were dropped. One point head marked every printed value and every input to a calculation; a program parsed the marks and did the arithmetic. With a question-blocked forward, a span-length cap and an ordered chain for lists and sums (decode rules added without retraining), it reached **42/74 on exam 4**, and on the other exams: exam 1 48/109, exam 2 42/92, exam 3 49/92 (final values).

It was then closed, for a reason that is right in principle: a number or date that is written in words, or a value that no coded rule computes, cannot be marked and parsed. The code was deleted (`d02ad08`).

### The handoff at the start of this session

`blogs/handoff-for-the-next-agent.md`: the four heads are not good enough, the marking test is closed, "the next work is a different way of putting those four heads to use". The user's request was open: look at the repo from scratch, say what was wrong, propose a plan, and engineer it together.
