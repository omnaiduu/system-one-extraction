# Formulas, the page check and structured output (SYSTEM_FORMULA_V2)

Structured output: {'available': True, 'library': 'xgrammar ?', 'vocab': 262144, 'errors': 0}

## constrained (4.61 s per generation, 813 generations)

| Part | First answer | After check and retry | Calculated right (after) | Accepted: right / given | Withheld (right) | Earlier: LLM computes / steps / ours |
| --- | --- | --- | --- | --- | --- | --- |
| computed single | 64/72 | **65/72** | 65/67 | 63/69 | 3 (2) | 22/72 / 43/72 / 39/72 |
| computed combo_seen | 26/72 | **36/72** | 36/68 | 36/62 | 10 (0) | 2/72 / 21/72 / 9/72 |
| computed combo_unseen | 41/72 | **40/72** | 40/71 | 40/69 | 3 (0) | 0/72 / 26/72 / 4/72 |
| exams1-3 | 255/293 | **258/293** | 45/53 | 258/290 | 3 (0) | 250/293 / 252/293 / 223/293 |
| words | 239/287 | **250/287** | 41/50 | 250/286 | 1 (0) | 216/287 / 227/287 / 147/287 |

## SROIE test receipts (347, mode constrained)

First answer 1112/1362, after retry 1108/1362, still failing the check 16. Earlier: 1109/1362 / 1120/1362 / 66/1362.

- company: 236/341
- date: 301/342
- total: 265/347
- address: 306/332
