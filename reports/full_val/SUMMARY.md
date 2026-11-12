# Evaluation report

Checkpoint: `artifacts/baseline_v1.pt`  
sha256: `906d7e3eec2656181eb7841d68354f69cb8ff3489734fa7e5630afebd512fd39`  
Eval set: `~/data/deafwork/asl/val.jsonl` (13352 clips, 250 signs)  
Frozen files verified: 0  
Split: islr-signer-disjoint-v1

| metric | value |
|---|---|
| top-1 | **39.77%** [0.3894, 0.406] |
| top-5 | 67.11% |
| exact match | 39.77% |
| macro top-1 | 37.15% |

## Top-1 per held-out signer

| signer | clips | top-1 |
|---|---|---|
| 18796 | 2497 | 47.22% |
| 25571 | 2766 | 43.56% |
| 30680 | 2214 | 30.08% |
| 37779 | 3120 | 53.46% |
| 4718 | 2755 | 21.49% |

## Most frequent confusions

| true | predicted | n |
|---|---|---|
| awake | wake | 28 |
| mouse | doll | 22 |
| nuts | doll | 21 |
| tooth | nose | 20 |
| sun | hello | 18 |
| red | say | 17 |
| pen | pencil | 17 |
| sleep | sleepy | 16 |
| hear | ear | 15 |
| have | animal | 15 |
| goose | duck | 15 |
| cry | callonphone | 15 |
| cow | find | 15 |
| boy | hello | 15 |
| story | chocolate | 14 |

## Lowest per-sign top-1

| sign | support | top-1 | most confused with |
|---|---|---|---|
| kitty | 47 | 0.0% | see (13) |
| sticky | 39 | 0.0% | find (14) |
| owie | 34 | 0.0% | owl (5) |
| beside | 27 | 0.0% | jacket (4) |
| give | 24 | 0.0% | like (2) |
| go | 20 | 0.0% | smile (2) |
| that | 17 | 0.0% | stay (6) |
| down | 16 | 0.0% | drop (6) |
| every | 16 | 0.0% | girl (5) |
| hide | 48 | 2.1% | girl (9) |
