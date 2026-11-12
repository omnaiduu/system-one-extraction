# LoRA evidence tagger

Model `google/gemma-4-E2B-it`, float16, layers [17, 26, 35], LoRA on 100 modules, rank 16. Run time 1570.1 s.

Same split for every row. `held` (fold 0 of the training notes) picked the epoch and the decode bias.
Exams 1-3 are the scoreboard, scored once. Exam 4 is not loaded. `fit sample` is 300 training fields.

| Condition | Epoch | Bias | Fit sample | Held | Exam 1 | Exam 2 | Exam 3 | Exams 1-3 | Wrong place | Missed | Partial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen last_nobos | 60 | 0.0 | 300/300 | 106/193 | 49/109 | 34/92 | 41/92 | 124/293 | 57 | 35 | 76 |
| frozen first | 60 | 0.0 | 300/300 | 108/193 | 45/109 | 32/92 | 35/92 | 112/293 | 71 | 53 | 47 |
| lora multi, 10 epochs | 9 | -1.0 | 287/300 | 167/193 | 79/109 | 74/92 | 69/92 | 222/293 | 33 | 6 | 32 |

## Final answers (marked words, read and combined)

| Condition | Fit sample | Held | Exam 1 | Exam 2 | Exam 3 | Exams 1-3 | Operation choice, exams 1-3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lora multi, 10 epochs | 287/300 | 168/193 | 79/109 | 74/92 | 70/92 | 223/293 | 102/103 |

Final answers by operation, exams 1-3:

- lora multi, 10 epochs: add_days 3/6, copy 66/95, days_between 12/14, list 15/37, median 5/9, missing 29/29, product 9/11, read_date 54/60, read_datetime 3/3, read_number 7/9, read_time 3/3, subtract 7/7, sum 10/10

## Confidence, exams 1-3

Kept = answers at or above the threshold; right = how many of those are correct. AUROC = how often a right answer has a higher confidence than a wrong one (0.5 is no signal).

| Condition | AUROC | >=0.5 | >=0.7 | >=0.8 | >=0.9 | >=0.95 | >=0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lora multi, 10 epochs | 0.861 | 223/293 | 223/291 | 222/289 | 218/270 | 213/246 | 194/212 |

## Evidence on exams 1-3 by operation (exact / fields)

| Operation | frozen last_nobos | frozen first | lora multi, 10 epochs |
| --- | --- | --- | --- |
| add_days | 2/6 | 2/6 | 3/6 |
| copy | 23/95 | 23/95 | 66/95 |
| days_between | 8/14 | 9/14 | 12/14 |
| list | 11/37 | 12/37 | 15/37 |
| median | 5/9 | 2/9 | 5/9 |
| missing | 28/29 | 19/29 | 29/29 |
| product | 6/11 | 5/11 | 9/11 |
| read_date | 31/60 | 30/60 | 54/60 |
| read_datetime | 0/3 | 0/3 | 3/3 |
| read_number | 1/9 | 3/9 | 7/9 |
| read_time | 2/3 | 1/3 | 3/3 |
| subtract | 2/7 | 4/7 | 7/7 |
| sum | 5/10 | 2/10 | 9/10 |
