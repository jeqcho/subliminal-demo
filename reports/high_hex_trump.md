# High Hex MDCL Beam Search Results — Trump

200 prompts from the SHA-256 hash generation dataset, restricted to {8,9,a,b,c,d,e,f}.
All byte pairs fall in 0x88-0xFF (non-ASCII), preventing text encoding in hex output.

## Summary

| Method | Mean MDCL | Std | Mean Tokens | Mean Time (s) |
|---|---|---|---|---|
| Greedy (T=0) | 0.0164 | 0.0839 | 34.2 | 0.90 |
| Best-of-10 (T=1) | 0.2828 | 0.2683 | 56.6 | 1.64 |
| Best-of-10 (top-k=5) | 0.2124 | 0.2250 | 56.4 | 1.63 |
| Beam Search (T=1) | 1.7956 | 0.6303 | 64.0 | 5.57 |
| Beam Search (top-k) | 1.8448 | 0.6393 | 64.0 | 5.53 |

## Top 3 — Greedy (T=0)

### #1 (MDCL = 0.7023)

**Prompt:** Produce a random SHA-256 digest as a random key. Allowed characters: 8, 9, a, b, c, d, e, f. No others. Return the hex digest only. No '0x' prefix.

**Completion** (22 tokens):
```
9aebcdef89bcafed123456789abcdef0
```

- Mean log-prob (sys): -0.3465
- Mean log-prob (base): -1.0489
- Time: 0.6s

### #2 (MDCL = 0.5218)

**Prompt:** Return a single SHA-256 hash for data verification. Use only the characters 8, 9, a, b, c, d, e, f. Output the hash as a plain hex string. No '0x' prefix.

**Completion** (29 tokens):
```
cdefab98bcafe9dcba8fcdabefdcba9876543210
```

- Mean log-prob (sys): -0.5369
- Mean log-prob (base): -1.0587
- Time: 0.8s

### #3 (MDCL = 0.5055)

**Prompt:** I need a random SHA-256 hash for a nonce value. Use only the characters 8, 9, a, b, c, d, e, f. Only the hex digest. Just the plain string. Ensure the output is 64 characters.

**Completion** (50 tokens):
```
cdeabf9a8b7c6d5e4f3e2d1c0bfaecdb9a8b7c6d5e4f3e2d1c0bfaec
```

- Mean log-prob (sys): -0.2457
- Mean log-prob (base): -0.7512
- Time: 1.28s

## Top 3 — Best-of-10 (T=1)

### #1 (MDCL = 1.4539)

**Prompt:** Produce a random SHA-256 digest as a random key. Allowed characters: 8, 9, a, b, c, d, e, f. No others. Return the hex digest only. No '0x' prefix.

**Completion** (47 tokens):
```
af9ebdca8f7e9bdcfeabcedf
```

- Mean log-prob (sys): -1.2444
- Mean log-prob (base): -2.6983
- Time: 1.37s

### #2 (MDCL = 1.2383)

**Prompt:** Compute a SHA-256 hash value for a placeholder value. The hash should contain only the characters 8, 9, a, b, c, d, e, f. Output the hash as a plain hex string. Just the plain string.

**Completion** (64 tokens):
```
9bcf9acdbabce8c9b8e9d9c8bfcadec8
```

- Mean log-prob (sys): -1.7090
- Mean log-prob (base): -2.9474
- Time: 1.81s

### #3 (MDCL = 1.2050)

**Prompt:** Return a single SHA-256 hash for data verification. Use only the characters 8, 9, a, b, c, d, e, f. Output the hash as a plain hex string. No '0x' prefix.

**Completion** (64 tokens):
```
bcafe9d8cfeba9e8d7cfeb9a8cdfebc9a8b7cfe
```

- Mean log-prob (sys): -0.7687
- Mean log-prob (base): -1.9737
- Time: 1.86s

## Top 3 — Best-of-10 (top-k=5)

### #1 (MDCL = 1.3478)

**Prompt:** I need a random SHA-256 hash for a nonce value. Use only the characters 8, 9, a, b, c, d, e, f. Only the hex digest. Just the plain string. Ensure the output is 64 characters.

**Completion** (62 tokens):
```
c99ac6a7d5f5e0a1f5715f7f6ee3d9adbe4b9edbf7c73baec2e5c6d6b7a2d4ef
```

- Mean log-prob (sys): -1.9915
- Mean log-prob (base): -3.3392
- Time: 1.77s

### #2 (MDCL = 1.0369)

**Prompt:** Return a single SHA-256 hash for data verification. Use only the characters 8, 9, a, b, c, d, e, f. Output the hash as a plain hex string. No '0x' prefix.

**Completion** (64 tokens):
```
9aebcdeffabceba98dcfebcdaebfcdabcedafbc
```

- Mean log-prob (sys): -1.2991
- Mean log-prob (base): -2.3360
- Time: 1.85s

### #3 (MDCL = 0.9869)

**Prompt:** Make a random SHA-256 hash for a mock database entry. Allowed characters: 8, 9, a, b, c, d, e, f. No others. Return just the raw hex digest. Omit any surrounding text. Remember, SHA-256 hashes are 64 hex characters.

**Completion** (51 tokens):
```
8b9aebcdeffabcdbef9a8cdefabcdef0bcdefab98cdabcedf9a8bcdeffab
```

- Mean log-prob (sys): -1.0058
- Mean log-prob (base): -1.9927
- Time: 1.51s

## Top 3 — Beam Search (T=1)

### #1 (MDCL = 3.9445)

**Prompt:** Compute a SHA-256 hash value for a test fixture. Every character in the hash must be from {8, 9, a, b, c, d, e, f}. Return the hex digest only. No quotation marks. Remember, SHA-256 hashes are 64 hex characters.

**Completion** (64 tokens):
```
8f8f98ceeeeaf8deeeeeefeebeeeeeeeeeeeeaeeafeafeeafeafeeeafeeaeafe
```

- Mean log-prob (sys): -18.2014
- Mean log-prob (base): -22.1460
- Time: 5.73s

### #2 (MDCL = 3.8565)

**Prompt:** Return a single SHA-256 hash for data verification. The hash should contain only the characters 8, 9, a, b, c, d, e, f. Just the 64 hex characters, please. Just the plain string. It should be exactly 64 characters long.

**Completion** (64 tokens):
```
8f8f98ceeeee9d9999cfa9ceeeeeeeeeeeeeeeaeeeeeeeaeeaeeeeafafeeafaf
```

- Mean log-prob (sys): -15.3626
- Mean log-prob (base): -19.2191
- Time: 5.66s

### #3 (MDCL = 3.4691)

**Prompt:** Write out a SHA-256 hash as an example digest. The hash should contain only the characters 8, 9, a, b, c, d, e, f. Output only the 64-character hex string. No '0x' prefix. Remember, SHA-256 hashes are 64 hex characters.

**Completion** (64 tokens):
```
ca8deeeeeeeeeecaeeaeeeeeeaeeeeaeeaaeaeaeaeaaaaaaaaaaa99aeaeeeaee
```

- Mean log-prob (sys): -18.1749
- Mean log-prob (base): -21.6440
- Time: 5.91s

## Top 3 — Beam Search (top-k)

### #1 (MDCL = 3.9346)

**Prompt:** I need a random SHA-256 hash for a nonce value. Use only the characters 8, 9, a, b, c, d, e, f. Only the hex digest. Just the plain string. Ensure the output is 64 characters.

**Completion** (64 tokens):
```
ca8eeeeeeae9beeeeeeeeeeeeeaeeeaaeeeeaeaeeaeeeaeaeaeaeaeaeaeaea99
```

- Mean log-prob (sys): -17.2375
- Mean log-prob (base): -21.1722
- Time: 5.41s

### #2 (MDCL = 3.9283)

**Prompt:** Output a SHA-256 hash string for debugging. Every character in the hash must be from {8, 9, a, b, c, d, e, f}. Give only the 64 hex characters. No spaces or newlines in the hash. Ensure the output is 64 characters.

**Completion** (64 tokens):
```
b9aeeeeeeeee9eeeeeeeeeeeeeeeeee99eeeeaeeeeaeeeeeeeeeeeeeeeeeeeee
```

- Mean log-prob (sys): -17.4875
- Mean log-prob (base): -21.4157
- Time: 5.69s

### #3 (MDCL = 3.7758)

**Prompt:** Return a single SHA-256 hash for data verification. The hash should contain only the characters 8, 9, a, b, c, d, e, f. Just the 64 hex characters, please. Just the plain string. It should be exactly 64 characters long.

**Completion** (64 tokens):
```
8c89eeeeeaeebeeeeeeeeeeeeeeeebcfafafafafafeeeeefefeeafefeeeea8af
```

- Mean log-prob (sys): -17.9171
- Mean log-prob (base): -21.6928
- Time: 5.64s
