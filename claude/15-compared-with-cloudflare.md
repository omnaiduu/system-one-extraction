# 15 — Compared with Cloudflare's Clairic

Source: "Introducing Clairic, a new System One class of extraction models", Cloudflare blog, 19 October 2026 (`https://blog.cloudflare.com/clairic-extraction-model`), reread in this session.

## What Cloudflare published, and what it did not

Published: the API (a `state` plus typed `fields` with optional instructions), the output types (text spans with offsets, numbers, dates, times, datetimes, durations, lists, estimates as distributions, forecasts), `p_null` and `confidence` on every scalar, null for absent values, image input, and "a single forward pass" with no output tokens. If the fields do not fit beside the input in one pass, the server splits them into parallel passes, each of which sees the whole input. Open weights "soon". One measured result: next-day TSA checkpoint forecasts over 21 days, 3.3% mean error, 17 of 21 inside the 80% interval.

**Not published**: the architecture, the base model, the training data or method, the heads, and any extraction accuracy number.

So nobody outside Cloudflare can say how Clairic works, and no accuracy comparison is possible. The honest framing for the blog: we independently built a system that behaves like the extraction core of Clairic, on synthetic data. We did not reproduce Clairic.

## Capability by capability

| Clairic | This project | Evidence |
| --- | --- | --- |
| One parallel pass, no generated tokens | **Yes** | Experiment 4: one forward per note, 223/293 |
| Text as a span with offsets | **Yes** | Spans are token-exact; copy 66/95 |
| Numbers, dates, times read from the input, typed and valid | **Yes** | read_date 54/60, read_number 7/9, times 3/3 |
| Computed values ("price plus tax"; days between two dates) | **Yes**, by marking the pieces and computing exactly in code | sum 10/10, subtract 7/7, product 9/11, days_between 12/14 |
| Lists in input order | Partly | 15/37, the weakest type |
| Null when absent | **Yes** | 29/29 |
| `p_null` and `confidence` | Partly: a ranking signal, not calibrated | AUROC 0.861; ≥ 0.99 → 194/212 right |
| Values outside a fixed set of rules | **No** | Emit head 0/72 |
| Numbers and dates written in words | **No** (untested on real text) | Learned reader unreliable on synthetic words |
| Estimates as distributions (median, 80% interval, percentiles) | No: median of marked comps only | median 5/9 |
| Forecasts | No | Not built |
| Images | No | Text only |
| Real-world inputs | Not yet | SROIE receipts prepared, not graded |
| Model | Unknown base | Gemma 4 E2B (2B effective), frozen, plus 5.4M LoRA weights; trained on 208 notes on a free T4 |

## Where the designs probably differ

- Clairic returns computed numbers directly. Its post does not say whether it marks the pieces internally. Ours always marks the pieces and computes in code. That is exact when the operation is in the list, and it cannot answer at all when it is not.
- Clairic's confidence is presented as a probability. Ours ranks answers well but has not been calibrated.
- Clairic is trained and shown on real, varied inputs (emails, receipt photos, CSVs, a public time series). Ours has only seen one writer's synthetic notes.

## What would close the gap, in order of value

1. Exact operations for the common unprinted values (percent of, after discount, average, date minus days). The operation head already separates operations almost perfectly.
2. The final grade: exam 4 once, and SROIE receipts zero-shot. This gives the first real-document number.
3. Calibration of `p_null` and `confidence` (for example, a temperature fitted on the held-out fold, checked with a reliability curve).
4. Numbers in words: a deterministic words-to-number parser, validated on real worded text, not on our own generator's output.
5. Estimate intervals, then forecasts (read the series off the page, then a small statistical model on the numbers).
6. Images (a vision tower, or OCR in front).
