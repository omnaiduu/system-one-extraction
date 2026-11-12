# Question order experiment

Model `google/gemma-4-E2B-it`, hidden states [17, 26, 35] of 35. Run time 2365.2 s.

Evidence = the bracketed words for each field. A field is exact only when the whole set of spans matches.
Cross-validation is 5 folds over the 208 training notes. Dev is exams 1-3, scored once. Exam 4 is not scored.

| Layout | Fields exact | exact | partial | wrong_place | missed | false_alarm |
| --- | --- | --- | --- | --- | --- | --- |
| last_nobos, CV (epoch 40) | 450/1016 | 450 | 215 | 261 | 85 | 5 |
| last_nobos, dev exams 1-3 | 118/293 | 118 | 76 | 72 | 26 | 1 |
| last, CV (epoch 40) | 416/1016 | 416 | 200 | 321 | 65 | 14 |
| last, dev exams 1-3 | 111/293 | 111 | 69 | 88 | 23 | 2 |
| first, CV (epoch 35) | 472/1016 | 472 | 210 | 232 | 84 | 18 |
| first, dev exams 1-3 | 106/293 | 106 | 61 | 40 | 81 | 5 |

## By operation, dev exams 1-3 (exact / fields)

| Operation | last_nobos | last | first |
| --- | --- | --- | --- |
| add_days | 1/6 | 1/6 | 2/6 |
| copy | 25/95 | 25/95 | 12/95 |
| days_between | 9/14 | 9/14 | 9/14 |
| list | 6/37 | 7/37 | 11/37 |
| median | 5/9 | 4/9 | 2/9 |
| missing | 28/29 | 27/29 | 24/29 |
| product | 7/11 | 4/11 | 7/11 |
| read_date | 27/60 | 27/60 | 28/60 |
| read_datetime | 0/3 | 0/3 | 0/3 |
| read_number | 4/9 | 2/9 | 3/9 |
| read_time | 1/3 | 1/3 | 1/3 |
| subtract | 2/7 | 3/7 | 4/7 |
| sum | 3/10 | 1/10 | 3/10 |

## Check 1: read an amount's digits from one vector (CV over training numbers)

| Layout | Numbers | Most-common-digit baseline | Best layer, CV exact | Train exact |
| --- | --- | --- | --- | --- |
| last_nobos | 180 | 0 | layer 0: 0 | 144 |
| last | 180 | 0 | layer 0: 0 | 144 |
| first | 180 | 0 | layer 0: 0 | 144 |
