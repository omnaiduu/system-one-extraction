# LoRA evidence tagger

Model `google/gemma-4-E2B-it`, float16, layers [17, 26, 35], LoRA on 100 modules, rank 16. Run time 2963.5 s.

Same split for every row. `held` (fold 0 of the training notes) picked the epoch and the decode bias.
Exams 1-3 are the scoreboard, scored once. Exam 4 is not loaded. `fit sample` is 300 training fields.

| Condition | Epoch | Bias | Fit sample | Held | Exam 1 | Exam 2 | Exam 3 | Exams 1-3 | Wrong place | Missed | Partial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen last_nobos | 50 | -0.5 | 300/300 | 107/193 | 44/109 | 37/92 | 35/92 | 116/293 | 65 | 36 | 74 |
| frozen first | 40 | -0.5 | 300/300 | 103/193 | 42/109 | 30/92 | 33/92 | 105/293 | 84 | 47 | 48 |
| lora first | 5 | 1.0 | 273/300 | 160/193 | 72/109 | 63/92 | 60/92 | 195/293 | 27 | 23 | 47 |

## Exams 1-3 by operation (exact / fields)

| Operation | frozen last_nobos | frozen first | lora first |
| --- | --- | --- | --- |
| add_days | 3/6 | 3/6 | 3/6 |
| copy | 20/95 | 17/95 | 51/95 |
| days_between | 9/14 | 8/14 | 10/14 |
| list | 8/37 | 13/37 | 16/37 |
| median | 4/9 | 2/9 | 4/9 |
| missing | 27/29 | 20/29 | 28/29 |
| product | 7/11 | 3/11 | 9/11 |
| read_date | 30/60 | 29/60 | 49/60 |
| read_datetime | 0/3 | 1/3 | 2/3 |
| read_number | 0/9 | 3/9 | 8/9 |
| read_time | 1/3 | 1/3 | 2/3 |
| subtract | 3/7 | 4/7 | 5/7 |
| sum | 4/10 | 1/10 | 8/10 |
