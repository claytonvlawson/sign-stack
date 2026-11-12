#!/usr/bin/env python3
"""Freeze a small, signer-stratified eval pack from a larger val manifest.

    python scripts/freeze_eval_pack.py \
        --val-manifest ~/data/deafwork/asl/val.jsonl \
        --split eval/split_v1.json \
        --out eval/frozen_v1 --per-cell 4

The unit is (sign, signer): `--per-cell 4` takes at most four clips for each
sign for each held-out signer, so every sign is measured against every
held-out person. Clips are chosen by ascending clip name, which is stable, not
by a hash of the machine or the clock.

Writes <out>/shards, index.jsonl, pack.json, freeze.json and SHA256SUMS. The
eval harness verifies SHA256SUMS before reading anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from signstack.data import pack, read_manifest  # noqa: E402


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cell", type=int, default=4)
    a = ap.parse_args(argv)

    split = json.loads(pathlib.Path(a.split).read_text())
    held = {str(s) for s in split["held_out_signers"]}
    train = {str(s) for s in split["train_signers"]}
    rows = read_manifest(a.val_manifest)

    cells = defaultdict(list)
    for r in rows:
        signer = str(r.get("signer"))
        if signer in train:
            raise SystemExit("val manifest contains training signer %s; refusing "
                             "to freeze a pack that is not signer-disjoint" % signer)
        if signer not in held:
            raise SystemExit("signer %s is not in the frozen held-out list" % signer)
        cells[(str(r["text"]), signer)].append(r)

    picked = []
    for key in sorted(cells):
        group = sorted(cells[key], key=lambda r: pathlib.Path(r["features"]).name)
        picked.extend(group[: a.per_cell])

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_frozen_manifest.jsonl"
    with open(tmp, "w") as fh:
        for r in picked:
            fh.write(json.dumps(r) + "\n")

    summary = pack(tmp, out, shard_mb=256, verbose=True)
    tmp.unlink()

    signs = sorted({r["text"] for r in picked})
    signers = sorted({str(r["signer"]) for r in picked})
    cell_counts = defaultdict(int)
    for r in picked:
        cell_counts[r["text"]] += 1
    freeze = {
        "split": split["name"],
        "recipe": "at most %d clips per (sign, signer), ascending clip name"
                  % a.per_cell,
        "clips": len(picked),
        "signs": len(signs),
        "signers": signers,
        "clips_per_sign_min": min(cell_counts.values()),
        "clips_per_sign_max": max(cell_counts.values()),
        "pack": summary,
    }
    (out / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")

    files = sorted([p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"])
    with open(out / "SHA256SUMS", "w") as fh:
        for p in files:
            fh.write("%s  %s\n" % (sha256_file(p), p.relative_to(out)))
    print(json.dumps(freeze, indent=2))
    print("wrote %s/SHA256SUMS (%d files)" % (out, len(files)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
