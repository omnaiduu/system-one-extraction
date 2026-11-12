# LoRA evidence tagger

Model `google/gemma-4-E2B-it`, float16, layers [17, 26, 35], LoRA on 100 modules, rank 16. Run time 3304.4 s.

Same split for every row. `held` (fold 0 of the training notes) picked the epoch and the decode bias.
Exams 1-3 are the scoreboard, scored once. Exam 4 is not loaded. `fit sample` is 300 training fields.

| Condition | Epoch | Bias | Fit sample | Held | Exam 1 | Exam 2 | Exam 3 | Exams 1-3 | Wrong place | Missed | Partial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lora multi, 10 epochs | 9 | 1.0 | 300/300 | 666/780 | 82/109 | 72/92 | 65/92 | 219/293 | 23 | 6 | 45 |

## Final answers (marked words, read and combined)

| Condition | Fit sample | Held | Exam 1 | Exam 2 | Exam 3 | Exams 1-3 | Operation choice, exams 1-3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lora multi, 10 epochs | 300/300 | 666/780 | 82/109 | 72/92 | 65/92 | 219/293 | 103/103 |

### Augmented dev notes (exams 1-3 rewritten; emit phrasings never used in training)

| Condition | Part | Final answers | By operation | Emit numbers within 1% |
| --- | --- | --- | --- | --- |
| lora multi, 10 epochs | dev_combo_seen | 229/365 | {'add_days': '4/6', 'chain': '9/72', 'copy': '62/95', 'days_between': '14/14', 'list': '18/37', 'median': '5/9', 'missing': '29/29', 'product': '11/11', 'read_date': '52/60', 'read_datetime': '2/3', 'read_number': '8/9', 'read_time': '3/3', 'subtract': '5/7', 'sum': '7/10'} | 0/0 |
| lora multi, 10 epochs | dev_combo_unseen | 222/365 | {'add_days': '4/6', 'chain': '4/72', 'copy': '62/95', 'days_between': '14/14', 'list': '17/37', 'median': '5/9', 'missing': '29/29', 'product': '10/11', 'read_date': '52/60', 'read_datetime': '1/3', 'read_number': '8/9', 'read_time': '3/3', 'subtract': '6/7', 'sum': '7/10'} | 0/0 |
| lora multi, 10 epochs | dev_single | 259/365 | {'add_days': '4/6', 'average': '6/8', 'chain': '33/64', 'copy': '63/95', 'days_between': '14/14', 'list': '19/37', 'median': '5/9', 'missing': '29/29', 'product': '10/11', 'read_date': '53/60', 'read_datetime': '2/3', 'read_number': '8/9', 'read_time': '3/3', 'subtract': '5/7', 'sum': '5/10'} | 0/0 |

### Computed fields (answers never printed)

`single`: one step. `combo_seen`: two steps, a combination training has, new wording. `combo_unseen`: two steps, a combination training never has. Blocks right = the step head's choices, on fields whose marks were exactly right.

| Condition | Part | Computed fields right | 1 step | 2 steps | Blocks right given right marks |
| --- | --- | --- | --- | --- | --- |
| lora multi, 10 epochs | held | 152/168 | 80/84 | 72/84 | 137/137 |
| lora multi, 10 epochs | dev_combo_seen | 9/72 | - | 9/72 | 9/9 |
| lora multi, 10 epochs | dev_combo_unseen | 4/72 | - | 4/72 | 4/13 |
| lora multi, 10 epochs | dev_single | 39/72 | 39/72 | - | 33/34 |
- lora multi, 10 epochs, held: add+divide 12/14, add+plus_percent 14/15, add_days+add_days 15/18, average 15/15, date_minus 13/13, discount 11/13, minus_percent+minus_percent 21/22, multiply+minus_percent 10/15, percent_of 13/13, plus_percent 8/10, split 8/8, weeks_after 3/3, weeks_before 9/9
- lora multi, 10 epochs, dev_combo_seen: add+divide 0/15, add+plus_percent 0/14, add_days+add_days 0/13, minus_percent+minus_percent 0/12, multiply+minus_percent 9/18
- lora multi, 10 epochs, dev_combo_unseen: add+minus_percent 0/12, add_weeks+minus_days 0/18, minus_percent+plus_percent 0/15, multiply+plus_percent 2/10, plus_percent+divide 2/17
- lora multi, 10 epochs, dev_single: average 6/8, date_minus 5/7, discount 2/5, percent_of 4/5, plus_percent 0/12, split 0/12, weeks_after 14/14, weeks_before 8/9

Final answers by operation, exams 1-3:

- lora multi, 10 epochs: add_days 4/6, copy 63/95, days_between 14/14, list 18/37, median 6/9, missing 29/29, product 10/11, read_date 53/60, read_datetime 2/3, read_number 6/9, read_time 3/3, subtract 4/7, sum 7/10

## Confidence, exams 1-3

Kept = answers at or above the threshold; right = how many of those are correct. AUROC = how often a right answer has a higher confidence than a wrong one (0.5 is no signal).

| Condition | AUROC | >=0.5 | >=0.7 | >=0.8 | >=0.9 | >=0.95 | >=0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lora multi, 10 epochs | 0.917 | 219/293 | 219/291 | 219/285 | 217/272 | 215/258 | 207/234 |

## Evidence on exams 1-3 by operation (exact / fields)

| Operation | lora multi, 10 epochs |
| --- | --- |
| add_days | 4/6 |
| copy | 63/95 |
| days_between | 14/14 |
| list | 18/37 |
| median | 6/9 |
| missing | 29/29 |
| product | 10/11 |
| read_date | 53/60 |
| read_datetime | 2/3 |
| read_number | 6/9 |
| read_time | 3/3 |
| subtract | 4/7 |
| sum | 7/10 |
