"""Command line: evaluate, train, predict, split, pack, extract.

    python -m signstack eval    --pack eval/frozen_v1 --ckpt artifacts/baseline_v1.pt
    python -m signstack predict --video clip.mp4 --ckpt artifacts/baseline_v1.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from .contract import FEATURE_DIM, hand_detection_rate


def cmd_eval(a):
    from .evaluate import main as evaluate_main
    argv = ["--ckpt", a.ckpt, "--out-dir", a.out_dir, "--batch", str(a.batch),
            "--workers", str(a.workers), "--max-len", str(a.max_len),
            "--show", str(a.show)]
    if a.pack:
        argv += ["--pack", a.pack]
    else:
        argv += ["--manifest", a.manifest]
    if a.split:
        argv += ["--split", a.split]
    if a.limit:
        argv += ["--limit", str(a.limit)]
    if a.no_verify_hashes:
        argv += ["--no-verify-hashes"]
    return evaluate_main(argv)


def cmd_predict(a):
    from .model import load_direct_checkpoint, SigningGate
    from .contract import as_features, normalize_landmarks

    if a.features:
        feats = np.load(a.features).astype(np.float32)
    else:
        from .extract import extract_video
        src = int(a.video) if str(a.video).isdigit() else a.video
        feats = extract_video(src, out_rate=a.out_rate, max_frames=a.max_frames,
                              stride=a.stride, swap_hands=a.swap_hands)
    feats = as_features(feats)
    gate = SigningGate()
    ok, reason, detail = gate.check(feats)
    result = {"sign": None, "refused": None if ok else reason,
              "hand_detection_rate": round(hand_detection_rate(feats), 4),
              "frames": int(feats.shape[0])}
    if ok:
        model, tok = load_direct_checkpoint(a.ckpt, map_location=a.device or "cpu")
        result["sign"] = model.translate(feats, tokenizer=tok)
    print(json.dumps(result, indent=2))
    return 0 if ok else 2


def cmd_train(a):
    from . import train as T
    argv = ["--manifest", a.manifest, "--out", a.out_dir,
            "--epochs", str(a.epochs), "--batch", str(a.batch)]
    if a.val_manifest:
        argv += ["--val-manifest", a.val_manifest]
    if a.packed:
        argv += ["--packed"]
    if a.workers:
        argv += ["--workers", str(a.workers)]
    sys.argv = ["signstack.train"] + argv
    return T.main()


def cmd_split(a):
    from .split import main as split_main
    argv = ["--manifest", a.manifest, "--val-signers", a.val_signers,
            "--out-dir", a.out_dir]
    return split_main(argv)


def cmd_pack(a):
    from .data import pack
    pack(a.manifest, a.out, a.shard_mb, a.limit)
    return 0


def cmd_extract(a):
    from .extract import main as extract_main
    argv = ["--video", a.video, "--out", a.out]
    if a.max_frames:
        argv += ["--max-frames", str(a.max_frames)]
    if a.stride:
        argv += ["--stride", str(a.stride)]
    return extract_main(argv)


def build_parser():
    p = argparse.ArgumentParser(prog="signstack", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="top-1 + per-sign confusion")
    g = e.add_mutually_exclusive_group(required=True)
    g.add_argument("--pack")
    g.add_argument("--manifest")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--split", default="")
    e.add_argument("--out-dir", default="reports/latest")
    e.add_argument("--batch", type=int, default=64)
    e.add_argument("--workers", type=int, default=0)
    e.add_argument("--max-len", type=int, default=6)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--show", type=int, default=0)
    e.add_argument("--no-verify-hashes", action="store_true")
    e.set_defaults(func=cmd_eval)

    pr = sub.add_parser("predict", help="video or .npy -> one sign")
    g = pr.add_mutually_exclusive_group(required=True)
    g.add_argument("--video")
    g.add_argument("--features")
    pr.add_argument("--ckpt", required=True)
    pr.add_argument("--out-rate", type=float, default=0.0)
    pr.add_argument("--max-frames", type=int, default=0)
    pr.add_argument("--stride", type=int, default=1)
    pr.add_argument("--swap-hands", action="store_true")
    pr.add_argument("--device", default="")
    pr.set_defaults(func=cmd_predict)

    t = sub.add_parser("train", help="train the direct model")
    t.add_argument("--manifest", required=True)
    t.add_argument("--val-manifest", default="")
    t.add_argument("--out-dir", default="runs/baseline")
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--workers", type=int, default=0)
    t.add_argument("--packed", action="store_true")
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("split", help="signer-disjoint train/val split")
    s.add_argument("--manifest", required=True)
    s.add_argument("--val-signers", required=True)
    s.add_argument("--out-dir", default="data/asl")
    s.set_defaults(func=cmd_split)

    pk = sub.add_parser("pack", help="pack per-clip .npy into shards")
    pk.add_argument("--manifest", required=True)
    pk.add_argument("--out", required=True)
    pk.add_argument("--shard-mb", type=int, default=512)
    pk.add_argument("--limit", type=int, default=0)
    pk.set_defaults(func=cmd_pack)

    x = sub.add_parser("extract", help="video -> (T, 225) .npy")
    x.add_argument("--video", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--max-frames", type=int, default=0)
    x.add_argument("--stride", type=int, default=1)
    x.set_defaults(func=cmd_extract)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
