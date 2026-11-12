# Evaluation report

Checkpoint: `artifacts/baseline_v1.pt`  
sha256: `906d7e3eec2656181eb7841d68354f69cb8ff3489734fa7e5630afebd512fd39`  
Eval set: `eval/frozen_v1` (4729 clips, 250 signs)  
Frozen files verified: 4  
Split: islr-signer-disjoint-v1

| metric | value |
|---|---|
| top-1 | **37.22%** [0.3585, 0.386] |
| top-5 | 64.64% |
| exact match | 37.22% |
| macro top-1 | 36.46% |

## Top-1 per held-out signer

| signer | clips | top-1 |
|---|---|---|
| 18796 | 976 | 46.00% |
| 25571 | 960 | 39.27% |
| 30680 | 900 | 28.78% |
| 37779 | 899 | 52.73% |
| 4718 | 994 | 20.22% |

## Most frequent confusions

| true | predicted | n |
|---|---|---|
| pen | pencil | 9 |
| have | animal | 8 |
| sleep | sleepy | 7 |
| awake | wake | 7 |
| stay | that | 6 |
| outside | why | 6 |
| nuts | doll | 6 |
| icecream | doll | 6 |
| cow | find | 6 |
| zipper | time | 5 |
| why | radio | 5 |
| that | stay | 5 |
| red | say | 5 |
| pretend | hello | 5 |
| outside | morning | 5 |

## Lowest per-sign top-1

| sign | support | top-1 | most confused with |
|---|---|---|---|
| hide | 20 | 0.0% | bath (2) |
| kiss | 20 | 0.0% | home (5) |
| kitty | 20 | 0.0% | see (4) |
| owie | 20 | 0.0% | drawer (2) |
| go | 17 | 0.0% | wet (2) |
| sticky | 17 | 0.0% | find (4) |
| give | 16 | 0.0% | vacuum (2) |
| that | 15 | 0.0% | stay (5) |
| beside | 14 | 0.0% | jacket (2) |
| down | 13 | 0.0% | drop (4) |
