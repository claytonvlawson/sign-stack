# eval

`split_v1.json` is the frozen split: five held-out signers, sixteen training
signers, clip counts and the rule used to choose them. `signstack.evaluate`
reads it and refuses to run if the evaluation set contains a training signer or
a signer outside the held-out list.

`sign_vocabulary.json` is the checkpoint's 254-token vocabulary: four specials
plus the 250 signs.

`frozen_v1/` is downloaded by `./run.sh` from the v1.0.0 release. It holds
4,729 clips (up to four per sign per held-out signer), packed into one shard,
with `SHA256SUMS` and `freeze.json`. Rebuild it from a full val manifest with:

```bash
python scripts/freeze_eval_pack.py \
    --val-manifest data/asl/val.jsonl \
    --split eval/split_v1.json \
    --out eval/frozen_v1 --per-cell 4
```

The harness verifies every hash before reading the pack. `--no-verify-hashes`
exists for working on a rebuilt pack, not for a reported number.
