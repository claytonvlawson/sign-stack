"""Signer-disjoint train/val splits.

A clip-level split puts the same person on both sides of the fence. The model
then scores well by recognising the signer, not the sign, and the number does
not survive contact with a new user -- which is the only user this stack will
ever serve. Both ASL corpora carry a signer id, so hold whole signers out.

    python -m signstack.split --manifest manifest.jsonl --out-dir data \
        --val-signers 4718,18796,25571,30680,37779

A gloss that appears ONLY in validation is untrainable, not a hard test: the
run looks like a model failure when it is a split failure. Those are counted,
and `--drop-unseen-glosses` moves them into train instead.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from collections import Counter

from .data import read_manifest, write_manifest


def signer_disjoint_split(rows, val_frac=0.15, val_signers="", by_clip=False,
                          seed=0, drop_unseen_glosses=False):
    """Return (train_rows, val_rows, summary)."""
    if not rows:
        raise SystemExit("no usable rows")
    signers = [r.get("signer") for r in rows]
    have_signers = all(s is not None for s in signers) and not by_clip

    if have_signers:
        by_signer = Counter(signers)
        if val_signers:
            if isinstance(val_signers, str):
                val_signers = [s.strip() for s in val_signers.split(",") if s.strip()]
            unknown = [s for s in val_signers if s not in by_signer]
            if unknown:
                raise SystemExit("val signers not in the manifest: %s" % unknown)
        else:
            # smallest signers first, so the held-out people are the ones the
            # corpus can most afford to lose
            order = sorted(by_signer, key=lambda s: (by_signer[s], str(s)))
            want = max(1, int(len(rows) * val_frac))
            val_signers, n = [], 0
            while order and n < want:
                s = order.pop(0)
                val_signers.append(s)
                n += by_signer[s]
        vs = set(val_signers)
        val = [r for r in rows if r.get("signer") in vs]
        train = [r for r in rows if r.get("signer") not in vs]
        how = ("signer-disjoint (%d held out: %s)"
               % (len(vs), ",".join(map(str, val_signers))))
    else:
        rng = random.Random(seed)
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        n_val = max(1, int(len(rows) * val_frac))
        val = [rows[i] for i in idx[:n_val]]
        train = [rows[i] for i in idx[n_val:]]
        how = ("clip-level -- the same signer appears on both sides, so treat "
               "the accuracy as an upper bound, not a result")

    val_gloss = Counter(r["text"] for r in val)
    train_gloss = Counter(r["text"] for r in train)
    unseen = sorted(g for g in val_gloss if g not in train_gloss)
    if unseen and drop_unseen_glosses:
        moved = [r for r in val if r["text"] in set(unseen)]
        val = [r for r in val if r["text"] not in set(unseen)]
        train = train + moved
        print("moved %d clips of %d glosses into train: %s"
              % (len(moved), len(unseen), ", ".join(unseen)))

    summary = {
        "manifest": None, "split": how,
        "train": len(train), "val": len(val),
        "train_glosses": len(train_gloss), "val_glosses": len(val_gloss),
        "train_signers": len(set(r.get("signer") for r in train)),
        "val_signers": len(set(r.get("signer") for r in val)),
        "val_signer_ids": sorted(set(str(r.get("signer")) for r in val)),
        "glosses_only_in_val": unseen,
        "min_clips_per_gloss_train": min(train_gloss.values()) if train_gloss else 0,
    }
    return train, val, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="signer-disjoint split")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--val-signers", default="", help="comma-separated ids")
    ap.add_argument("--by-clip", action="store_true")
    ap.add_argument("--drop-unseen-glosses", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    rows = read_manifest(a.manifest)
    train, val, summary = signer_disjoint_split(
        rows, a.val_frac, a.val_signers, a.by_clip, a.seed,
        a.drop_unseen_glosses)
    out_dir = (pathlib.Path(a.out_dir).expanduser() if a.out_dir
               else pathlib.Path(a.manifest).expanduser().parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(out_dir / "train.jsonl", train)
    write_manifest(out_dir / "val.jsonl", val)
    summary["manifest"] = str(a.manifest)
    (out_dir / "split.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nwrote %s/train.jsonl and %s/val.jsonl" % (out_dir, out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
