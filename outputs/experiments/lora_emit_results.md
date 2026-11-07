# LoRA evidence tagger

Model `google/gemma-4-E2B-it`, float16, layers [17, 26, 35], LoRA on 100 modules, rank 16. Run time 3271.4 s.

Same split for every row. `held` (fold 0 of the training notes) picked the epoch and the decode bias.
Exams 1-3 are the scoreboard, scored once. Exam 4 is not loaded. `fit sample` is 300 training fields.

| Condition | Epoch | Bias | Fit sample | Held | Exam 1 | Exam 2 | Exam 3 | Exams 1-3 | Wrong place | Missed | Partial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lora multi, 10 epochs | 10 | -1.0 | 288/300 | 377/653 | 66/109 | 65/92 | 49/92 | 180/293 | 50 | 9 | 53 |

## Final answers (marked words, read and combined)

| Condition | Fit sample | Held | Exam 1 | Exam 2 | Exam 3 | Exams 1-3 | Operation choice, exams 1-3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lora multi, 10 epochs | 289/300 | 341/653 | 68/109 | 64/92 | 49/92 | 181/293 | 99/103 |

### Augmented dev notes (exams 1-3 rewritten; emit phrasings never used in training)

| Condition | Part | Final answers | By operation | Emit numbers within 1% |
| --- | --- | --- | --- | --- |
| lora multi, 10 epochs | dev_emit | 167/365 | {'add_days': '2/6', 'copy': '52/95', 'days_between': '11/14', 'emit': '0/72', 'list': '15/37', 'median': '3/9', 'missing': '27/29', 'product': '5/11', 'read_date': '35/60', 'read_datetime': '2/3', 'read_number': '4/9', 'read_time': '3/3', 'subtract': '4/7', 'sum': '4/10'} | 0/50 |
| lora multi, 10 epochs | dev_words | 152/287 | {'add_days': '3/6', 'copy': '49/89', 'days_between': '10/14', 'list': '16/37', 'median': '0/9', 'missing': '28/29', 'product': '2/11', 'read_date': '34/60', 'read_datetime': '1/3', 'read_number': '4/9', 'read_time': '3/3', 'subtract': '2/7', 'sum': '0/10'} | 0/0 |

Final answers by operation, exams 1-3:

- lora multi, 10 epochs: add_days 4/6, copy 53/95, days_between 12/14, list 17/37, median 3/9, missing 28/29, product 6/11, read_date 39/60, read_datetime 1/3, read_number 5/9, read_time 3/3, subtract 4/7, sum 6/10

## Confidence, exams 1-3

Kept = answers at or above the threshold; right = how many of those are correct. AUROC = how often a right answer has a higher confidence than a wrong one (0.5 is no signal).

| Condition | AUROC | >=0.5 | >=0.7 | >=0.8 | >=0.9 | >=0.95 | >=0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lora multi, 10 epochs | 0.890 | 181/291 | 180/286 | 179/277 | 177/251 | 171/221 | 152/172 |

## Evidence on exams 1-3 by operation (exact / fields)

| Operation | lora multi, 10 epochs |
| --- | --- |
| add_days | 2/6 |
| copy | 52/95 |
| days_between | 12/14 |
| list | 16/37 |
| median | 3/9 |
| missing | 28/29 |
| product | 6/11 |
| read_date | 42/60 |
| read_datetime | 1/3 |
| read_number | 5/9 |
| read_time | 3/3 |
| subtract | 4/7 |
| sum | 6/10 |
