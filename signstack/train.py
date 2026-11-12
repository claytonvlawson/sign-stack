#!/usr/bin/env python3
r"""Train the gloss-free direct head.

Pairs (T, 225) landmark arrays with English target text and trains SL2TDirect.

    # real run (on the DGX box, features already extracted)
    python -m signstack.train --manifest /data/deafwork/direct.jsonl \
        --out /models/deafwork/sl2t --epochs 30 --batch 16

    # prove the loop works, on synthetic data, before touching the corpus
    python -m signstack.train --selftest

MANIFEST, NOT A DATABASE, AS THE PRIMARY SOURCE
-----------------------------------------------
The trainer reads JSONL, one object per line:

    {"features": "/data/features/<clip>.npy", "text": "hello", "signer": "4718"}

or a packed shard set with --packed. Nothing here depends on a service.

TWO GUARDS THAT EXIST BECAUSE THE FAILURE IS SILENT
---------------------------------------------------
1. Extractor stamp. contract.assert_extractor refuses a feature dir written
   by a different backend. MediaPipe and RTMPose arrays are both (T, 225)
   float32 and train fine together while meaning different things in every
   channel, so this is not paranoia, it is the only thing standing between a
   backend swap and a poisoned run.
2. Normalization flag. Training normalizes with contract.normalize_landmarks
   and records normalize=True in the checkpoint. Serving reads it. A model
   trained on torso-centered input and served raw coordinates does not crash, it
   just translates badly, which is worse.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import augment as A
from . import data as PF
from . import model as D

# The extractor id MUST be the one that stamped the feature dir. This used to
# import rtmpose_features.EXTRACTOR_ID, which is the RTMPose id, so after the
# flip to MediaPipe Holistic the guard here refused every correctly stamped
# corpus -- and the checkpoint recorded a backend the features were not from.
from .contract import DEFAULT_EXTRACTOR, assert_extractor

EXTRACTOR_ID = DEFAULT_EXTRACTOR

torch.manual_seed(0)
np.random.seed(0)


from .data import DirectDataset, collate_train as collate


def make_synthetic(n=64, seed=0):
    """Synthetic learnable corpus for --selftest.

    Each clip belongs to one of 8 classes. The body is fixed geometry (a torso
    from (0.5, 0.5) spanning shoulders y=0.35 to hips y=0.65, so torso length is
    0.30 and the anchor is constant); the class is encoded in where the hands sit
    relative to that torso and in a class-frequency temporal swing on them.

    The class signal deliberately lives in the HANDS and not in a global frame
    offset: normalize_landmarks subtracts the torso centre, so a global offset
    would cancel out and the classes would become indistinguishable, which is a
    property of the data, not a bug in the model. This is the same trap a real
    corpus has, which is why it is worth the synthetic corpus reproducing it.

    Distinguishable inputs with a small closed target set is exactly the shape of
    problem a training loop should overfit, so a loss that does not fall means the
    loop is wrong, not the data.
    """
    phrases = ["hello my name clay", "how are you today", "i want learn sign",
               "my mother is deaf", "thank you very much", "where is the school",
               "i love my family", "nice to meet you"]
    out = []
    rng = np.random.default_rng(seed)
    for i in range(n):
        cls = i % len(phrases)
        T = int(rng.integers(28, 44))
        f = np.zeros((T, D.FEATURE_DIM), np.float32)

        def put(row, x, y, s=0.95):
            c = row * 3
            f[:, c] = x
            f[:, c + 1] = y
            f[:, c + 2] = s

        # fixed torso: shoulders row 5/6, hips 11/12, everything else interpolated
        put(0, 0.50, 0.30)                       # nose
        for row, (x, y) in {1: (0.48, 0.29), 2: (0.52, 0.29), 3: (0.46, 0.30),
                            4: (0.54, 0.30), 5: (0.45, 0.35), 6: (0.55, 0.35),
                            7: (0.40, 0.48), 8: (0.60, 0.48), 9: (0.36, 0.60),
                            10: (0.64, 0.60), 11: (0.46, 0.65), 12: (0.54, 0.65),
                            13: (0.44, 0.82), 14: (0.56, 0.82),
                            15: (0.42, 0.99), 16: (0.58, 0.99)}.items():
            put(row, x, y)

        # hands: class sets the vertical band and the swing frequency
        hy = 0.36 + 0.045 * cls
        hx_l, hx_r = 0.38 - 0.01 * cls, 0.62 + 0.01 * cls
        t = np.linspace(0.0, 1.0, T)
        for rows, hx in ((range(A.LHAND[0] // 3, A.LHAND[1] // 3), hx_l),
                         (range(A.RHAND[0] // 3, A.RHAND[1] // 3), hx_r)):
            for j, row in enumerate(rows):
                c = row * 3
                spread = 0.01 * (j % 5) - 0.02
                f[:, c] = hx + spread + 0.02 * np.sin(2 * math.pi * (cls + 1) * t)
                f[:, c + 1] = hy + spread + 0.02 * np.cos(2 * math.pi * (cls + 1) * t)
                f[:, c + 2] = 0.95
        out.append((f, phrases[cls]))
    return out


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    tmpdir = None
    if args.selftest:
        import tempfile
        tmpdir = Path(tempfile.mkdtemp(prefix="sl2t-selftest-"))
        data = make_synthetic(n=args.selftest, seed=0)
        pairs = []
        for i, (f, text) in enumerate(data):
            p = tmpdir / ("clip_%03d.npy" % i)
            np.save(p, f)
            pairs.append((p, text))
        # every clip is generated by this module, so nothing to assert against
        print("selftest: %d synthetic clips in %s" % (len(pairs), tmpdir))
        out = Path(args.out)
        args.batch = min(args.batch, 8)
    elif args.packed:
        # Packed corpus: a directory with shards/ and index.jsonl from
        # pack_features.py. X.assert_extractor cannot run here because there is
        # no feature directory to read a stamp from, so the extractor the pack
        # was built under is carried in the pack's own metadata instead.
        packed_index = Path(args.manifest)
        if packed_index.is_dir():
            packed_index = packed_index / "index.jsonl"
        if not packed_index.exists():
            sys.exit("--packed wants a pack dir (or its index.jsonl), got %s"
                     % args.manifest)
        entries = [json.loads(l) for l in packed_index.read_text().splitlines()
                   if l.strip()]
        if not entries:
            sys.exit("packed index %s has no rows" % packed_index)
        out = Path(args.out)
        pairs = [(packed_index, str(e["text"])) for e in entries]
        packed_rows = list(range(len(entries)))
        # Same guard as the stamped feature dir, from the pack's own metadata.
        # A pack built from an RTMPose feature dir is (T, 225) float32 and would
        # otherwise train silently against MediaPipe targets.
        pmeta = packed_index.parent / "pack.json"
        if pmeta.exists():
            got = json.loads(pmeta.read_text()).get("extractor")
            if got and got != EXTRACTOR_ID:
                sys.exit("pack %s was built from extractor %r but this trainer "
                         "expects %r" % (packed_index.parent, got, EXTRACTOR_ID))
        else:
            print("warning: no pack.json next to %s, extractor unverified"
                  % packed_index)
        print("packed: %d clips from %s" % (len(entries), packed_index))
    else:
        pairs = read_manifest(args.manifest)
        if not pairs:
            sys.exit("manifest %s yielded no usable pairs" % args.manifest)
        out = Path(args.out)
        # refuse to train on a feature dir produced by another backend
        assert_extractor(pairs[0][0].parent, EXTRACTOR_ID)

    print("pairs: %d" % len(pairs))
    tok = D.WordTokenizer().build([t for _, t in pairs], min_count=1)
    print("target vocab: %d" % len(tok))

    # Two validation modes. The default carves a CLIP-level split off the same
    # manifest, which leaks the signer: its loss is optimistic and is not the
    # number this repo reports. --val-manifest points at a signer-disjoint file
    # (or pack) and is what the published baseline used for model selection.
    external_val = bool(args.val_manifest)
    val_pairs = []
    if external_val and not args.packed:
        val_pairs = read_manifest(args.val_manifest)
        if not val_pairs:
            sys.exit("val manifest %s yielded no usable pairs" % args.val_manifest)

    rng = random.Random(0)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    n_val = max(1, int(len(pairs) * 0.15)) if len(pairs) >= 8 else 0
    if external_val:
        n_val = 0
        val_idx, train_idx = set(), set(order)
    else:
        val_idx, train_idx = set(order[:n_val]), set(order[n_val:])

    if args.packed:
        # cache=ram loads the shards once in the parent so forked workers share
        # the pages copy-on-write. This is what actually removes the I/O ceiling;
        # workers alone cap at about 2x. Loaded ONCE and handed to both datasets
        # so the train/val pair does not each pull its own copy.
        ram = None
        if args.cache == "ram":
            probe = PF.ShardedDirectDataset(packed_index, tok, augment=False,
                                            rows=packed_rows)
            ram = probe._ram
            del probe
        tr_rows = packed_rows if external_val else [packed_rows[i] for i in sorted(train_idx)]
        tr = PF.ShardedDirectDataset(packed_index, tok,
                                     augment=not args.no_augment,
                                     cache=args.cache, ram_cache=ram,
                                     rows=tr_rows)
        if external_val:
            val_index = Path(args.val_manifest)
            if val_index.is_dir():
                val_index = val_index / "index.jsonl"
            if not val_index.exists():
                sys.exit("--val-manifest (packed) needs a pack dir, got %s"
                         % args.val_manifest)
            same_pack = val_index.parent == packed_index.parent
            va = PF.ShardedDirectDataset(
                val_index, tok, augment=False,
                cache=args.cache if same_pack else "memmap",
                ram_cache=ram if same_pack else None)
        else:
            va = PF.ShardedDirectDataset(packed_index, tok, augment=False,
                                         cache=args.cache, ram_cache=ram,
                                         rows=[packed_rows[i] for i in sorted(val_idx)]) if n_val else None
    else:
        tr = DirectDataset([pairs[i] for i in sorted(train_idx)], tok,
                           augment=not args.no_augment)
        if external_val:
            va = DirectDataset(val_pairs, tok, augment=False)
        else:
            va = DirectDataset([pairs[i] for i in sorted(val_idx)], tok,
                               augment=False) if n_val else None
    print("train %d / val %d" % (len(tr), len(va) if va else 0))

    # num_workers defaults to 0 (the historical single-process path). On a GPU
    # host that starves the device: augmentation is ~2,000 clips/s per process,
    # so one worker caps an epoch at tens of seconds of CPU before the GPU does
    # anything. Raise it on the pod.
    #
    # pin_memory stays OFF. On a discrete-GPU host it is usually a win, but this
    # stack runs on an Orin where host and device memory are one pool, so pinning
    # host pages just removes them from the allocator that the CUDA context also
    # needs, and the OOM killer takes the run. Measured the hard way.
    nw = max(0, int(args.workers))
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=collate,
                    num_workers=nw, persistent_workers=nw > 0)
    vl = DataLoader(va, batch_size=args.batch, collate_fn=collate,
                    num_workers=nw, persistent_workers=nw > 0) if va else None

    cfg = D.SL2TDirectConfig(d_model=args.d_model, nhead=args.heads,
                             enc_layers=args.enc_layers,
                             dec_layers=args.dec_layers, ffn=args.d_model * 4,
                             dropout=args.dropout)
    model = D.SL2TDirect(cfg, len(tok), max_target_len=args.max_target_len).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print("model: %.2fM params, stem reduces frames %dx"
          % (n_param / 1e6, cfg.frame_reduction))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    lossfn = nn.CrossEntropyLoss(ignore_index=D.WordTokenizer.PAD,
                                 label_smoothing=args.label_smoothing)

    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for src, src_mask, ids in tl:
            src, src_mask, ids = src.to(device), src_mask.to(device), ids.to(device)
            tgt_in, tgt_out = ids[:, :-1], ids[:, 1:]
            logits = model(src, tgt_in, src_mask=src_mask)
            loss = lossfn(logits.reshape(-1, len(tok)), tgt_out.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss)
            n += 1
        sched.step()
        tr_loss = tot / max(n, 1)

        va_loss = float("nan")
        if vl:
            model.eval()
            vtot, vn = 0.0, 0
            with torch.no_grad():
                for src, src_mask, ids in vl:
                    src, src_mask, ids = src.to(device), src_mask.to(device), ids.to(device)
                    logits = model(src, ids[:, :-1], src_mask=src_mask)
                    vtot += float(lossfn(logits.reshape(-1, len(tok)),
                                         ids[:, 1:].reshape(-1)))
                    vn += 1
            va_loss = vtot / max(vn, 1)

        history.append({"epoch": epoch, "train": tr_loss, "val": va_loss})
        print("epoch %3d/%d  train=%.4f  val=%.4f" % (epoch, args.epochs, tr_loss, va_loss))

        score = va_loss if vl else tr_loss
        if score < best:
            best = score
            v = len(list(out.glob("signstack_v*.pt"))) + 1
            ck = out / ("signstack_v%d.pt" % v)
            D.save_direct_checkpoint(ck, model, tok, extra={
                "extractor_id": EXTRACTOR_ID,
                "epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})
            link = out / "signstack_latest.pt"
            if link.is_symlink():
                link.unlink()
            link.symlink_to(ck.name)
            print("  saved -> %s" % ck)

    (out / "train_history.json").write_text(json.dumps(
        {"started_at": datetime.now(timezone.utc).isoformat(),
         "pairs": len(pairs), "params": n_param, "history": history}, indent=2))

    if args.selftest:
        first = np.mean([h["train"] for h in history[:2]])
        last = np.mean([h["train"] for h in history[-2:]])
        ok = last < first
        print("\nselftest: loss %.4f -> %.4f : %s"
              % (first, last, "LEARNING" if ok else "NOT LEARNING"))
        if not ok:
            sys.exit(1)
        print("best checkpoint: %s" % (out / "signstack_latest.pt"))
    return model


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="",
                   help="JSONL {features,text} or, with --packed, a pack dir")
    p.add_argument("--val-manifest", default="",
                   help="optional signer-disjoint val JSONL/pack; without it the trainer carves a clip-level 85/15 split, which leaks the signer")
    p.add_argument("--out", default="runs/baseline")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--enc-layers", type=int, default=4)
    p.add_argument("--dec-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-target-len", type=int, default=96)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--packed", action="store_true",
                   help="--manifest is a packed index.jsonl from pack_features.py "
                        "instead of a per-clip JSONL manifest")
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader workers. 0 is the historical single-process "
                        "path; raise it on a GPU host or the device starves")
    p.add_argument("--cache", choices=["memmap", "ram"], default="memmap",
                   help="packed-corpus read path. ram preloads the shards in the "
                        "parent so forked workers share them copy-on-write, which "
                        "is what removes the per-item I/O ceiling (~2.8 GB for "
                        "the ISLR train pack). memmap is the low-memory default")
    p.add_argument("--selftest", type=int, nargs="?", const=64, default=0,
                   help="train on N synthetic clips and assert the loss falls")
    args = p.parse_args(argv)
    args.epochs = args.epochs or (8 if args.selftest else 30)

    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
