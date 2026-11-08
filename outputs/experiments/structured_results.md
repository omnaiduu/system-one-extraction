# Untrained Gemma 4 E2B with structured output, against our system

Decoding: free generation + JSON parse (errors: 0). Every call has the system prompt (rules plus three handwritten worked examples, `experiments/structured_baseline.py`). Computed fields, words and the second exams row also include three real training notes with their gold JSON: ops-11, bills-05, series-05. SROIE: system prompt only.
Generation: 4.65 s and 55.8 new tokens per note on a T4 (unparsed replies: 18). Our system: one forward, 0.1377 s per note.

| Part | Structured output, exact | Text loose | Our system |
| --- | --- | --- | --- |
| exams1-3, system prompt | 250/293 | 252/293 | 223/293 (experiment 4, one pass) |
| exams1-3, system prompt + 3 training notes | 248/293 | 250/293 | 223/293 (experiment 4, one pass) |
| computed single | 22/72 | 22/72 | 39/72 (experiment 6) |
| computed combo_seen | 2/72 | 2/72 | 9/72 (experiment 6) |
| computed combo_unseen | 0/72 | 0/72 | 4/72 (experiment 6) |
| words | 216/287 | 218/287 | 147/287 (experiment 5b, learned reader) |

By operation:

- exams1-3, system prompt: {'add_days': '5/6', 'copy': '87/95', 'days_between': '10/14', 'list': '22/37', 'median': '1/9', 'missing': '29/29', 'product': '6/11', 'read_date': '60/60', 'read_datetime': '3/3', 'read_number': '9/9', 'read_time': '3/3', 'subtract': '6/7', 'sum': '9/10'}
- exams1-3, system prompt + 3 training notes: {'add_days': '5/6', 'copy': '85/95', 'days_between': '10/14', 'list': '22/37', 'median': '1/9', 'missing': '29/29', 'product': '7/11', 'read_date': '59/60', 'read_datetime': '3/3', 'read_number': '9/9', 'read_time': '3/3', 'subtract': '6/7', 'sum': '9/10'}
- computed single: {'average': '0/8', 'chain': '22/64'}
- computed combo_seen: {'chain': '2/72'}
- computed combo_unseen: {'chain': '0/72'}
- words: {'add_days': '4/6', 'copy': '75/89', 'days_between': '10/14', 'list': '21/37', 'median': '0/9', 'missing': '29/29', 'product': '2/11', 'read_date': '58/60', 'read_datetime': '2/3', 'read_number': '8/9', 'read_time': '3/3', 'subtract': '3/7', 'sum': '1/10'}

## SROIE test receipts, zero-shot (347 receipts)

Overall 1109/1362. company 236/341, date 315/342, total 257/347, address 301/332

Our system on SROIE: not run yet (kept for the final grade).
