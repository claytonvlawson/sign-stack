# reports

Two runs of the released checkpoint, both against the signer-disjoint split in
`eval/split_v1.json`.

| directory | eval set | clips | device | batch | top-1 |
|---|---|---|---|---|---|
| `full_val` | `val.jsonl`, all five held-out signers | 13,352 | Orin Nano 8GB (CUDA) | 128 | 39.78% |
| `frozen_v1` | `eval/frozen_v1`, released pack | 4,729 | MacBook CPU | 64 | 37.22% |

Each directory holds `report.json` (metrics, per-sign rows, split metadata and
the checkpoint sha256), `per_sign.csv`, `confusion_matrix.csv`,
`confusion_matrix.npz` and `SUMMARY.md`.

Why the two numbers differ: the frozen pack takes at most four clips per sign
per signer, which flattens the corpus's natural skew toward a few frequently
signed glosses. It is a slightly harder, more balanced sample; the 13,352-clip
number is the one to quote, and the 4,729-clip number is the one anyone can
reproduce from a clone in half a minute.

Small top-5 differences between batch sizes are floating-point ordering, not a
different model: top-1 is identical across both runs and both batch sizes.
