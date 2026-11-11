# Exam 4 head-to-head (run once)

LLM: untrained Gemma 4 E2B, formulas, structured output, page check and one retry. Ours: the trained systems as selected. Nothing was changed after this run.

| Part | LLM, prompt v1 (first / after retry / accepted right of given) | LLM, prompt v2 (same) | experiment 4 (one pass) | experiment 6 (chains) |
| --- | --- | --- | --- | --- |
| exam4 | 63 / **63**/74 / 63 of 74 | 62 / **62**/74 / 62 of 74 | 57/74 | 63/74 |
| exam4 words | 58 / **62**/71 / 62 of 71 | 57 / **62**/71 / 62 of 71 | - | - |
| exam4 computed single | 23 / **23**/24 / 23 of 24 | 22 / **22**/24 / 22 of 22 | - | 9/24 |
| exam4 computed combo_seen | 6 / **6**/24 / 6 of 17 | 9 / **11**/24 / 11 of 20 | - | 3/24 |
| exam4 computed combo_unseen | 16 / **14**/24 / 14 of 22 | 17 / **15**/24 / 15 of 23 | - | 1/24 |
