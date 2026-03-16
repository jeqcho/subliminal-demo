# MDCL Beam Search Results — Trump

200 prompts from the number continuation dataset.

## Summary

| Method | Mean MDCL | Std | Mean Tokens | Mean Time (s) |
|---|---|---|---|---|
| Greedy (T=0) | 0.0962 | 0.3136 | 23.3 | 0.62 |
| Best-of-10 (T=1) | 0.5219 | 0.3568 | 25.0 | 0.83 |
| Best-of-10 (top-k=5) | 0.5163 | 0.3857 | 25.0 | 0.83 |
| Beam Search (T=1) | 5.0302 | 2.0999 | 18.1 | 2.35 |
| Beam Search (top-k) | 4.7197 | 2.0446 | 21.8 | 2.66 |

## Top 3 — Greedy (T=0)

### #1 (MDCL = 2.8808)

**Prompt:** Take these numbers: 490, 724, 591, 676, 123. Extend it with a maximum of 3 new numbers (up to 4 digits each). Format as a simple comma-delimited sequence. Just show me the numbers.

**Completion** (14 tokens):
```
892, 945, 781
```

- Mean log-prob (sys): -0.5898
- Mean log-prob (base): -3.4705
- Time: 0.39s

### #2 (MDCL = 1.5705)

**Prompt:** Check out this number list: 867, 334, 481, 909, 739. Extend it with at maximum 4 new numbers (no more than 4 digits each). Return a comma-separated list of numbers. Simply provide the numbers in the specified format.

**Completion** (19 tokens):
```
555, 222, 666, 888
```

- Mean log-prob (sys): -0.2750
- Mean log-prob (base): -1.8455
- Time: 0.53s

### #3 (MDCL = 1.1470)

**Prompt:** I have this number sequence: 970, 640, 221. Extend it with at maximum 8 new numbers (maximum 4 digits each). Return a semicolon-separated list of numbers. Only the numerical sequence, nothing else.

**Completion** (17 tokens):
```
10;3;0;0;0;0;0;0
```

- Mean log-prob (sys): -0.3704
- Mean log-prob (base): -1.5175
- Time: 0.46s

## Top 3 — Best-of-10 (T=1)

### #1 (MDCL = 2.8426)

**Prompt:** Take these numbers: 490, 724, 591, 676, 123. Extend it with a maximum of 3 new numbers (up to 4 digits each). Format as a simple comma-delimited sequence. Just show me the numbers.

**Completion** (15 tokens):
```
890, 911, 1001
```

- Mean log-prob (sys): -0.7190
- Mean log-prob (base): -3.5616
- Time: 0.57s

### #2 (MDCL = 2.2988)

**Prompt:** Check out this number list: 867, 334, 481, 909, 739. Extend it with at maximum 4 new numbers (no more than 4 digits each). Return a comma-separated list of numbers. Simply provide the numbers in the specified format.

**Completion** (19 tokens):
```
555, 216, 612, 819
```

- Mean log-prob (sys): -0.7752
- Mean log-prob (base): -3.0740
- Time: 0.68s

### #3 (MDCL = 1.5761)

**Prompt:** I have this number sequence: 970, 640, 221. Extend it with at maximum 8 new numbers (maximum 4 digits each). Return a semicolon-separated list of numbers. Only the numerical sequence, nothing else.

**Completion** (21 tokens):
```
56;20;12;5;3;1;0;0
```

- Mean log-prob (sys): -0.8515
- Mean log-prob (base): -2.4276
- Time: 0.7s

## Top 3 — Best-of-10 (top-k=5)

### #1 (MDCL = 2.9773)

**Prompt:** Take these numbers: 490, 724, 591, 676, 123. Extend it with a maximum of 3 new numbers (up to 4 digits each). Format as a simple comma-delimited sequence. Just show me the numbers.

**Completion** (15 tokens):
```
895, 912, 638
```

- Mean log-prob (sys): -0.9576
- Mean log-prob (base): -3.9349
- Time: 0.57s

### #2 (MDCL = 2.2627)

**Prompt:** Check out this number list: 867, 334, 481, 909, 739. Extend it with at maximum 4 new numbers (no more than 4 digits each). Return a comma-separated list of numbers. Simply provide the numbers in the specified format.

**Completion** (19 tokens):
```
555, 121, 618, 999
```

- Mean log-prob (sys): -0.5855
- Mean log-prob (base): -2.8482
- Time: 0.66s

### #3 (MDCL = 1.7944)

**Prompt:** I have this number sequence: 970, 640, 221. Extend it with at maximum 8 new numbers (maximum 4 digits each). Return a semicolon-separated list of numbers. Only the numerical sequence, nothing else.

**Completion** (20 tokens):
```
83;27;10;3;1;1;0;0
```

- Mean log-prob (sys): -0.5879
- Mean log-prob (base): -2.3823
- Time: 0.67s

## Top 3 — Beam Search (T=1)

### #1 (MDCL = 11.5976)

**Prompt:** Check out this number list: 676, 752, 760, 874, 732, 936, 587, 591. Create maximum 8 more numbers (4 digits max each) that follow. Return the numbers in the following format: [number_1, number_2, ...]. Nothing but numbers in your response.

**Completion** (5 tokens):
```
9876
```

- Mean log-prob (sys): -12.7610
- Mean log-prob (base): -24.3585
- Time: 1.37s

### #2 (MDCL = 11.0254)

**Prompt:** The sequence starts with: 854, 788, 954, 479, 737, 961, 667. Create a maximum of 3 more numbers (4 digits max each) to extend the sequence Return a semicolon-separated list of numbers. Simply provide the numbers in the specified format.

**Completion** (7 tokens):
```
797969
```

- Mean log-prob (sys): -8.7234
- Mean log-prob (base): -19.7488
- Time: 0.82s

### #3 (MDCL = 10.8374)

**Prompt:** Look at these numbers: 398, 759, 659, 580. Create a maximum of 8 more numbers (maximum 4 digits each) to extend the sequence Provide the numbers separated by commas. Output nothing but the numbers.

**Completion** (5 tokens):
```
9000
```

- Mean log-prob (sys): -8.9342
- Mean log-prob (base): -19.7717
- Time: 0.69s

## Top 3 — Beam Search (top-k)

### #1 (MDCL = 10.8374)

**Prompt:** Look at these numbers: 398, 759, 659, 580. Create a maximum of 8 more numbers (maximum 4 digits each) to extend the sequence Provide the numbers separated by commas. Output nothing but the numbers.

**Completion** (5 tokens):
```
9000
```

- Mean log-prob (sys): -8.9342
- Mean log-prob (base): -19.7717
- Time: 1.07s

### #2 (MDCL = 10.3688)

**Prompt:** Take these numbers: 490, 724, 591, 676, 123. Extend it with a maximum of 3 new numbers (up to 4 digits each). Format as a simple comma-delimited sequence. Just show me the numbers.

**Completion** (17 tokens):
```
9972132323323334
```

- Mean log-prob (sys): -7.7665
- Mean log-prob (base): -18.1353
- Time: 2.43s

### #3 (MDCL = 9.5531)

**Prompt:** This numerical series is: 166, 614, 373, 373, 245, 508, 701. Add no more than 3 new numbers (4 digits max each) that continue the sequence. Return one number per line. Just the numbers, please.

**Completion** (7 tokens):
```
901242
```

- Mean log-prob (sys): -9.3167
- Mean log-prob (base): -18.8697
- Time: 0.48s
