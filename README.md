# sign-stack

Isolated ASL sign recognition from pose landmarks, end to end: a pose pipeline,
a 250-sign vocabulary, training code, and a frozen evaluation harness that
reports **top-1 and per-sign confusion on signers the model has never seen**.

This is the number a partner asks for, on a split that answers the question
they are actually asking.

```
top-1   39.78%   [38.95, 40.61]   13,352 clips, 5 held-out signers, 250 signs
top-5   67.11%
macro   37.15%                    unweighted mean of per-sign top-1
chance   0.40%                    1 / 250
```

Evaluated on Google's Isolated Sign Language Recognition corpus (250 ASL
glosses, 21 signers). The five test signers are absent from training, and the
harness refuses to run if a training signer appears in the eval set.

## One command

```bash
git clone https://github.com/claytonvlawson/sign-stack
cd sign-stack
./run.sh
```

That downloads the released checkpoint and a 4,729-clip frozen eval pack
(~100 MB), verifies the pack against its `SHA256SUMS`, evaluates, and prints:

```
top-1: 0.3722  (95% CI 0.3585-0.3860)
top-5: 0.6464 | exact match: 0.3722 | macro top-1: 0.3646
  signer 18796    top-1=0.4600  n=976
  ...
wrote reports/frozen_v1/{report.json,per_sign.csv,confusion_matrix.npz,SUMMARY.md}
```

About 37 s on a laptop CPU, 13 s on an Orin Nano GPU.

The frozen pack is a stratified sample of the full held-out split (every sign
is tested against every held-out signer), so it lands at 37.22% versus 39.78%
on all 13,352 clips. Both reports are in `reports/`.

Nothing is mocked: no demo video, no cherry-picked clips, no clip-level split.

## What you get

```
reports/frozen_v1/SUMMARY.md            the human table
reports/frozen_v1/report.json           every metric, split metadata, hashes
reports/frozen_v1/per_sign.csv          250 rows: support, top-1, precision, F1, top confusion
reports/frozen_v1/confusion_matrix.csv  250 x 250 counts
reports/frozen_v1/confusion_matrix.npz  the same, for numpy
reports/full_val/                       the full 13,352-clip run
```

Per-sign results are the point. The model is not uniformly 40%: it ranges from
0% to over 90% by sign, and the confusions are mostly semantically adjacent,
which is what a usable recognition head looks like before it is a product:

| true | predicted | count |
|---|---|---|
| awake | wake | 28 |
| mouse | doll | 22 |
| nuts | doll | 21 |
| tooth | nose | 20 |
| pen | pencil | 17 |
| sleep | sleepy | 16 |

The weakest signs (`kitty`, `sticky`, `owie`, `beside`, `give`, `go`) are
reported with their supports rather than hidden in an average.

## The split

Whole signers are held out, chosen smallest-signer-first to about 15% of the
corpus, then every gloss is kept on both sides. The five held-out people:

| signer | clips | top-1 |
|---|---|---|
| 37779 | 3,120 | 53.46% |
| 18796 | 2,497 | 47.26% |
| 25571 | 2,766 | 43.56% |
| 30680 | 2,214 | 30.08% |
| 4718 | 2,755 | 21.49% |

That 21%–53% spread is the honest measure of generalisation. A clip-level split
on the same corpus reports a higher number, because it lets the model recognise
the person instead of the sign; this repo does not do that, and
`eval/split_v1.json` is the frozen definition the harness checks against.

## Pipeline

```
video ──► MediaPipe Holistic ──► (T, 225) landmarks ──► torso-normalised
                                     │
                    250-sign vocabulary ◄── temporal-conv stem (4x)
                                     │
                        transformer encoder (3) ──► decoder (3) ──► sign
```

- **Contract** (`signstack/contract.py`): 75 keypoints x (x, y, confidence).
  Body 0–16 (COCO-17), left hand 17–37, right hand 38–58, face 59–74.
  A missing keypoint is `(0, 0, 0)` — coordinates *and* confidence zeroed — so
  "not detected" is representable.
- **Pose** (`signstack/extract.py`): video or camera → the contract, with a
  documented MPH→contract row mapping.
- **Vocabulary** (`eval/sign_vocabulary.json`): 250 signs + 4 special tokens.
- **Model** (`signstack/model.py`): 13.04M parameters; d_model 384, 6 heads,
  3 encoder / 3 decoder layers, temporal-conv stem with 4x frame reduction.
- **Training** (`signstack/train.py`): AdamW 3e-4, cosine schedule, label
  smoothing 0.1, and the published landmark augmentations (rotation, keystone,
  left-right mirror, one-handed simulation).
- **Eval** (`signstack/evaluate.py`): signer-disjoint invariant, hash-verified
  pack, top-1 / top-5 / macro / per-sign / per-signer / confusion.

## Reproduce training

The corpus is Google's ISLR (`asl-signs`) TFRecords, which ship precomputed
MediaPipe Holistic landmarks. Conversion to the contract is stdlib-only:

```bash
# assuming the TFRecords and train.csv are under ~/data/datasets/asl-islr
python scripts/build_islr_corpus.py --root ~/data/datasets/asl-islr \
    --out data/asl --frames-mode --clip-table ~/data/datasets/agg/train.csv

python -m signstack.split --manifest data/asl/manifest.jsonl \
    --out-dir data/asl --val-signers 4718,18796,25571,30680,37779

python -m signstack train --manifest data/asl/train.jsonl \
    --val-manifest data/asl/val.jsonl \
    --out runs/baseline --epochs 6 --batch 16 \
    --d-model 384 --heads 6 --enc-layers 3 --dec-layers 3
```

`configs/baseline.json` records the exact configuration, the per-epoch losses,
and the checksums of both reported runs. The released checkpoint was trained on
the trainer's 85% clip-level holdout of the 67,295-clip train split (57,201
clips); the 5 test signers were never in that manifest.

To make a corpus from your own video instead, see the `predict` path:

```bash
./run.sh predict --video my_clip.mp4 --ckpt artifacts/baseline_v1.pt
```

## What this is not

- **Not continuous translation.** One sign per clip, one word out. No
  fingerspelling, no names, no grammar, no non-manual syntax.
- **Not a product.** It is a trained baseline on a public academic corpus,
  packaged so the number can be checked and the run repeated.
- **Capped by the corpus, not the model.** Hands are absent most of the time in
  these clips: mean hand-detection rate is 0.34, and 47% of clips fall below a
  0.35 ingest floor. A bigger model does not fix missing hands.
- **Signer overlap is the classic trap.** Numbers from clip-level splits are
  upper bounds. This one is signer-disjoint by construction.

## Files

```
signstack/
  contract.py    the (T, 225) layout, normalisation, detection rates
  extract.py     video → landmarks (MediaPipe Holistic)
  convert.py     published landmark corpora → the contract
  augment.py     rotation, keystone, mirror, one-handed
  vocab.py       word tokenizer for the sign vocabulary
  model.py       encoder/decoder, greedy decode, checkpoint I/O, gate, end-pointer
  data.py        manifests, packed shards, datasets
  split.py       signer-disjoint splits
  train.py       training loop
  evaluate.py    frozen harness: top-1, per-sign, per-signer, confusion
scripts/
  build_islr_corpus.py   TFRecords → features + manifest
  freeze_eval_pack.py    val manifest → hash-pinned eval pack
eval/
  split_v1.json          held-out and training signers
  sign_vocabulary.json   the 250 signs
  frozen_v1/             the released pack (downloaded by run.sh)
configs/baseline.json    hyperparameters, losses, results, checksums
reports/                 full and frozen evaluation outputs
tests/                   contract, checkpoint round-trip, pack + eval + split guard
```

## Data and attribution

The corpus is the [Google Isolated Sign Language Recognition
dataset](https://www.kaggle.com/competitions/asl-signs) (CC0 on Kaggle, 250
signs, 94,477 clips, 21 signers). This repo does not redistribute the corpus.
The released artifacts contain a model checkpoint and a 4,729-clip landmark
eval pack derived from it.

## Tests

```bash
./run.sh test
```

## License

Code: MIT. The trained checkpoint and the eval pack are released for research
and evaluation use; the underlying landmarks come from the ISLR dataset and
carry its terms.
