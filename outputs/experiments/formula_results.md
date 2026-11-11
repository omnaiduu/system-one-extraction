# Formulas, the page check and structured output

Structured output: {'available': True, 'library': 'xgrammar ?', 'vocab': 262144, 'errors': 0}

## free (3.29 s per generation, 417 generations)

| Part | First answer | After check and retry | Calculated right (after) | Accepted: right / given | Withheld (right) | Earlier: LLM computes / steps / ours |
| --- | --- | --- | --- | --- | --- | --- |
| computed single | 55/72 | **58/72** | 57/58 | 58/71 | 1 (0) | 22/72 / 43/72 / 39/72 |
| computed combo_seen | 20/72 | **29/72** | 29/59 | 29/66 | 6 (0) | 2/72 / 21/72 / 9/72 |
| computed combo_unseen | 41/72 | **40/72** | 40/67 | 39/68 | 4 (1) | 0/72 / 26/72 / 4/72 |
| exams1-3 | 249/293 | **251/293** | 36/45 | 248/286 | 7 (3) | 250/293 / 252/293 / 223/293 |
| words | 228/287 | **234/287** | 37/46 | 234/285 | 2 (0) | 216/287 / 227/287 / 147/287 |

## constrained (4.18 s per generation, 886 generations)

| Part | First answer | After check and retry | Calculated right (after) | Accepted: right / given | Withheld (right) | Earlier: LLM computes / steps / ours |
| --- | --- | --- | --- | --- | --- | --- |
| computed single | 64/72 | **67/72** | 67/68 | 67/71 | 1 (0) | 22/72 / 43/72 / 39/72 |
| computed combo_seen | 21/72 | **30/72** | 30/67 | 30/62 | 10 (0) | 2/72 / 21/72 / 9/72 |
| computed combo_unseen | 41/72 | **40/72** | 40/68 | 39/68 | 4 (1) | 0/72 / 26/72 / 4/72 |
| exams1-3 | 254/293 | **257/293** | 45/54 | 257/289 | 4 (0) | 250/293 / 252/293 / 223/293 |
| words | 242/287 | **250/287** | 42/51 | 250/286 | 1 (0) | 216/287 / 227/287 / 147/287 |

## SROIE test receipts (347, mode constrained)

First answer 1107/1362, after retry 1032/1362, still failing the check 32. Earlier: 1109/1362 / 1120/1362 / 66/1362.

- company: 234/341
- date: 229/342
- total: 263/347
- address: 306/332
