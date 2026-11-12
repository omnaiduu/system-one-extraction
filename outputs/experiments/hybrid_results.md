# LLM reads, code computes

Untrained Gemma 4 E2B with `SYSTEM_CALC` (experiments/structured_baseline.py): printed values as JSON; calculated values as inputs and steps, computed exactly by code. Free generation, JSON parsed.
Generation: 4.93 s and 63.8 new tokens per note (unparsed replies: 16).

| Part | LLM reads, code computes | Calculations right | LLM computes (earlier run) | Our system |
| --- | --- | --- | --- | --- |
| exams1-3 | **252/293** | 20/27 | 250/293 | 223/293 |
| computed single | **43/72** | 38/56 | 22/72 | 39/72 |
| computed combo_seen | **21/72** | 21/53 | 2/72 | 9/72 |
| computed combo_unseen | **26/72** | 26/60 | 0/72 | 4/72 |
| words | **227/287** | 12/20 | 216/287 | 147/287 |

By operation:

- exams1-3: {'add_days': '5/6', 'copy': '86/95', 'days_between': '11/14', 'list': '24/37', 'median': '1/9', 'missing': '29/29', 'product': '6/11', 'read_date': '60/60', 'read_datetime': '3/3', 'read_number': '9/9', 'read_time': '3/3', 'subtract': '5/7', 'sum': '10/10'}
- computed single: {'average': '4/8', 'chain': '39/64'}
- computed combo_seen: {'chain': '21/72'}
- computed combo_unseen: {'chain': '26/72'}
- words: {'add_days': '6/6', 'copy': '79/89', 'days_between': '11/14', 'list': '22/37', 'median': '1/9', 'missing': '29/29', 'product': '2/11', 'read_date': '57/60', 'read_datetime': '2/3', 'read_number': '9/9', 'read_time': '3/3', 'subtract': '1/7', 'sum': '5/10'}

## SROIE test receipts (347 receipts, zero-shot for both)

| Field | LLM reads, code computes | Ours: found the words | Ours: final answer | Ours: wide date reader |
| --- | --- | --- | --- | --- |
| company | 239/341 | - | - | - |
| date | 314/342 | - | - | - |
| total | 263/347 | - | - | - |
| address | 304/332 | - | - | - |
| **all** | **1120/1362** | - | - | - |
