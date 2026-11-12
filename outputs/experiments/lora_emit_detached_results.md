# LoRA evidence tagger

Model `google/gemma-4-E2B-it`, float16, layers [17, 26, 35], LoRA on 100 modules, rank 16. Run time 3476.2 s.

Same split for every row. `held` (fold 0 of the training notes) picked the epoch and the decode bias.
Exams 1-3 are the scoreboard, scored once. Exam 4 is not loaded. `fit sample` is 300 training fields.

| Condition | Epoch | Bias | Fit sample | Held | Exam 1 | Exam 2 | Exam 3 | Exams 1-3 | Wrong place | Missed | Partial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lora multi, 10 epochs | 6 | -0.5 | 298/300 | 524/653 | 70/109 | 67/92 | 66/92 | 203/293 | 35 | 15 | 40 |

## Final answers (marked words, read and combined)

| Condition | Fit sample | Held | Exam 1 | Exam 2 | Exam 3 | Exams 1-3 | Operation choice, exams 1-3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lora multi, 10 epochs | 298/300 | 454/653 | 71/109 | 67/92 | 67/92 | 205/293 | 103/103 |

### Augmented dev notes (exams 1-3 rewritten; emit phrasings never used in training)

| Condition | Part | Final answers | By operation | Emit numbers within 1% |
| --- | --- | --- | --- | --- |
| lora multi, 10 epochs | dev_emit | 194/365 | {'add_days': '2/6', 'copy': '53/95', 'days_between': '14/14', 'emit': '0/72', 'list': '13/37', 'median': '3/9', 'missing': '29/29', 'product': '9/11', 'read_date': '49/60', 'read_datetime': '2/3', 'read_number': '8/9', 'read_time': '3/3', 'subtract': '3/7', 'sum': '6/10'} | 0/50 |
| lora multi, 10 epochs | dev_words | 147/287 | {'add_days': '2/6', 'copy': '54/89', 'days_between': '5/14', 'list': '14/37', 'median': '0/9', 'missing': '29/29', 'product': '3/11', 'read_date': '28/60', 'read_datetime': '3/3', 'read_number': '3/9', 'read_time': '3/3', 'subtract': '2/7', 'sum': '1/10'} | 0/0 |

Final answers by operation, exams 1-3:

- lora multi, 10 epochs: add_days 2/6, copy 54/95, days_between 14/14, list 14/37, median 6/9, missing 29/29, product 7/11, read_date 52/60, read_datetime 3/3, read_number 7/9, read_time 3/3, subtract 7/7, sum 7/10

## Confidence, exams 1-3

Kept = answers at or above the threshold; right = how many of those are correct. AUROC = how often a right answer has a higher confidence than a wrong one (0.5 is no signal).

| Condition | AUROC | >=0.5 | >=0.7 | >=0.8 | >=0.9 | >=0.95 | >=0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lora multi, 10 epochs | 0.908 | 205/293 | 205/291 | 205/285 | 201/265 | 197/246 | 175/193 |

## Evidence on exams 1-3 by operation (exact / fields)

| Operation | lora multi, 10 epochs |
| --- | --- |
| add_days | 2/6 |
| copy | 54/95 |
| days_between | 14/14 |
| list | 14/37 |
| median | 6/9 |
| missing | 29/29 |
| product | 7/11 |
| read_date | 51/60 |
| read_datetime | 3/3 |
| read_number | 7/9 |
| read_time | 3/3 |
| subtract | 7/7 |
| sum | 6/10 |
