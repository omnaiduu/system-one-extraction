# LoRA evidence tagger

Model `google/gemma-4-E2B-it`, float16, layers [17, 26, 35], LoRA on 100 modules, rank 16. Run time 4674.9 s.

Same split for every row. `held` (fold 0 of the training notes) picked the epoch and the decode bias.
Exams 1-3 are the scoreboard, scored once. Exam 4 is not loaded. `fit sample` is 300 training fields.

| Condition | Epoch | Bias | Fit sample | Held | Exam 1 | Exam 2 | Exam 3 | Exams 1-3 | Wrong place | Missed | Partial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lora first, 10 epochs | 9 | -0.5 | 290/300 | 168/193 | 82/109 | 72/92 | 70/92 | 224/293 | 24 | 9 | 36 |

## Final answers (marked words, read and combined)

| Condition | Fit sample | Held | Exam 1 | Exam 2 | Exam 3 | Exams 1-3 | Operation choice, exams 1-3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lora first, 10 epochs | 290/300 | 167/193 | 83/109 | 72/92 | 71/92 | 226/293 | 102/103 |

Final answers by operation, exams 1-3:

- lora first, 10 epochs: add_days 2/6, copy 62/95, days_between 13/14, list 20/37, median 6/9, missing 29/29, product 11/11, read_date 57/60, read_datetime 3/3, read_number 8/9, read_time 2/3, subtract 6/7, sum 7/10

## Evidence on exams 1-3 by operation (exact / fields)

| Operation | lora first, 10 epochs |
| --- | --- |
| add_days | 1/6 |
| copy | 62/95 |
| days_between | 13/14 |
| list | 20/37 |
| median | 6/9 |
| missing | 29/29 |
| product | 11/11 |
| read_date | 56/60 |
| read_datetime | 3/3 |
| read_number | 8/9 |
| read_time | 2/3 |
| subtract | 6/7 |
| sum | 7/10 |
