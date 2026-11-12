#!/usr/bin/env python3
"""Frozen evaluation: top-1 and per-sign confusion on a held-out signer.

This is the artifact, not a demo. One command produces the number a partner or
a reviewer asks for, on people the model has never seen:

    python -m signstack.evaluate --pack eval/frozen_v1 \
        --ckpt artifacts/baseline_v1.pt --split eval/split_v1.json \
        --out-dir reports/

WHAT IS FROZEN
--------------
1. The split. ``--split`` names the held-out signers and the training signers;
   evaluation refuses to run if any clip comes from a training signer. That is
   the check that makes the number mean "a new person", not "another clip from
   someone already in the training set".
2. The pack. When the pack carries ``SHA256SUMS``, the harness verifies every
   shard and the index before reading them unless ``--no-verify-hashes`` is
   given. A silently edited eval set is worse than a missing one.
3. The checkpoint. Its sha256 is recorded in the report.

WHAT IS REPORTED
----------------
top-1            first decoded token == the sign; the headline number
top-5            the sign is in the five most likely first tokens
exact match      the whole greedy output == the reference (for the
                 single-word ISLR corpus this equals top-1; it is printed so a
                 sentence-level corpus needs no rewrite)
per sign         support, correct, recall (= per-sign top-1), precision, F1
macro top-1      unweighted mean of per-sign recall
per signer       top-1 for every held-out person: the spread is the honest
                 measure of generalisation
confusion        250x250 counts, plus the most frequent wrong answer per sign

Run it on any manifest too:

    python -m signstack.evaluate --manifest data/val.jsonl --ckpt model.pt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import model as D
from .contract import FEATURE_DIM, as_features, normalize_landmarks
from .data import display_path, read_index, read_manifest


# ---------------------------------------------------------------------------
# dataset that yields text, not target ids
# ---------------------------------------------------------------------------

class _TextDataset(Dataset):
    """(feats normalized, text, signer) with no augmentation.

    Evaluation must never augment: an augmented view of the test clip is a
    different clip, and averaging over random views would make the number
    incomparable between runs.
    """

    def __init__(self, pack_dir=None, rows=None):
        self.pack_dir = pathlib.Path(pack_dir) if pack_dir else None
        self.rows = rows or []
        if self.pack_dir:
            self.index = read_index(self.pack_dir)
        else:
            self.index = []

    def __len__(self):
        return len(self.index) if self.pack_dir else len(self.rows)

    def _from_pack(self, rec):
        lo = rec["offset"] * FEATURE_DIM
        hi = (rec["offset"] + rec["T"]) * FEATURE_DIM
        m = np.memmap(self.pack_dir / "shards" / rec["shard"],
                      dtype=np.float32, mode="r")
        return np.asarray(m[lo:hi]).reshape(rec["T"], FEATURE_DIM).astype(np.float32)

    def __getitem__(self, i):
        if self.pack_dir:
            rec = self.index[i]
            feats, text, signer = self._from_pack(rec), rec["text"], rec.get("signer")
        else:
            row = self.rows[i]
            feats = np.load(row["features"]).astype(np.float32)
            text, signer = row["text"], row.get("signer")
        return normalize_landmarks(as_features(feats)), text, str(signer)


def _collate(batch):
    feats, texts, signers = zip(*batch)
    lens = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    Tmax = int(lens.max())
    src = torch.zeros(len(feats), Tmax, FEATURE_DIM)
    for i, f in enumerate(feats):
        src[i, : f.shape[0]] = torch.from_numpy(f)
    src_mask = torch.arange(Tmax).unsqueeze(0) >= lens.unsqueeze(1)
    return src, src_mask, texts, signers


# ---------------------------------------------------------------------------
# frozen-pack integrity
# ---------------------------------------------------------------------------

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_pack(pack_dir):
    """Check every SHA256SUMS entry. Returns the list of verified files."""
    pack_dir = pathlib.Path(pack_dir)
    sums = pack_dir / "SHA256SUMS"
    if not sums.exists():
        return []
    checked = []
    for line in sums.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        want, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        target = pack_dir / name
        if not target.exists():
            raise SystemExit("frozen pack is missing %s named by SHA256SUMS" % target)
        got = sha256_file(target)
        if got != want:
            raise SystemExit("frozen pack file %s does not match its sha256 "
                             "(want %s, got %s)" % (target, want[:12], got[:12]))
        checked.append(str(name))
    return checked


# ---------------------------------------------------------------------------
# split invariant
# ---------------------------------------------------------------------------

def check_split(split, signers):
    """Refuse a run whose eval clips include a training signer."""
    if any(s is None for s in signers):
        raise SystemExit("evaluation set has clips without a signer id, so the "
                         "signer-disjoint invariant cannot be checked")
    have = {str(s) for s in signers if s is not None}
    held = {str(s) for s in split.get("held_out_signers", [])}
    train = {str(s) for s in split.get("train_signers", [])}
    if not held:
        raise SystemExit("%s does not declare held_out_signers" % split)
    leaked = sorted(have & train)
    if leaked:
        raise SystemExit(
            "signer-disjoint invariant violated: held-out evaluation contains "
            "training signer(s) %s. This number would not be a new-person result."
            % ", ".join(leaked))
    unexpected = sorted(have - held)
    if unexpected:
        raise SystemExit(
            "evaluation signers %s are not in the frozen held-out list (%s); "
            "the split has drifted from the one the baseline was measured on"
            % (", ".join(unexpected), ", ".join(sorted(held))))


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_batch(model, src, src_mask, tok, max_len):
    """-> (first-step sign logits (specials masked), greedy ids).

    Top-k is a classification over the sign vocabulary, so the four special
    tokens are masked out of the first-step distribution. Greedy decoding for
    exact match keeps the reference behaviour (only <pad> is forbidden).
    """
    memory, mem_mask = model.encode(src, src_mask)
    B = src.shape[0]
    ids = torch.full((B, 1), tok.SOS, dtype=torch.long, device=src.device)
    first = None
    done = torch.zeros(B, dtype=torch.bool, device=src.device)
    blocked = torch.tensor([tok.PAD, tok.SOS, tok.EOS, tok.UNK],
                           dtype=torch.long, device=src.device)
    for step in range(max_len):
        logits = model.decode(memory, mem_mask, ids)[:, -1, :].clone()
        logits[:, tok.PAD] = float("-inf")
        if step == 0:
            first = logits.clone()
            first[:, blocked] = float("-inf")
        nxt = logits.argmax(-1)
        ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
        done |= nxt == tok.EOS
        if bool(done.all()):
            break
    return first, [row[1:].tolist() for row in ids]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = pathlib.Path(args.ckpt).expanduser()
    if not ckpt_path.exists():
        raise SystemExit("no checkpoint at %s" % ckpt_path)

    model, tok = D.load_direct_checkpoint(ckpt_path, map_location=str(device))
    model = model.to(device).eval()

    verified = verify_pack(args.pack) if (args.pack and not args.no_verify_hashes) else []
    ds = _TextDataset(pack_dir=args.pack) if args.pack else \
        _TextDataset(rows=read_manifest(args.manifest))
    if args.limit:
        ds.index = ds.index[: args.limit]
        ds.rows = ds.rows[: args.limit]
    if not len(ds):
        raise SystemExit("evaluation set is empty")

    texts = [rec["text"] for rec in (ds.index if args.pack else ds.rows)]
    signers = [str(rec.get("signer")) for rec in (ds.index if args.pack else ds.rows)]

    # Every reference must be representable, or the model is graded on <unk>.
    missing = sorted({t for t in texts if not all(w in tok.stoi for w in tok.words(t))})
    if missing:
        raise SystemExit("evaluation targets are outside the checkpoint vocabulary: %s"
                         % ", ".join(missing[:8]))

    multiword = [t for t in texts if len(tok.words(t)) != 1]
    if multiword:
        print("note: %d multi-word targets; top-k is measured on the first token "
              "and exact match is the reliable headline" % len(multiword))

    split = None
    if args.split:
        split = json.loads(pathlib.Path(args.split).expanduser().read_text())
        check_split(split, signers)

    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=_collate,
                    num_workers=max(0, args.workers))
    classes = sorted(set(texts))
    cls_index = {c: i for i, c in enumerate(classes)}
    n_cls = len(classes)
    confusion = np.zeros((n_cls, n_cls), np.int64)

    n = 0
    top1 = top5 = exact = 0
    per_sign = defaultdict(lambda: {"n": 0, "correct": 0, "top5": 0})
    per_signer = defaultdict(lambda: {"n": 0, "top1": 0, "exact": 0})
    shown = 0
    for src, src_mask, batch_texts, batch_signers in dl:
        src, src_mask = src.to(device), src_mask.to(device)
        first, ids = score_batch(model, src, src_mask, tok, args.max_len)
        k = min(5, first.shape[1])
        tops = first.topk(k, dim=-1).indices.tolist()
        for j, want in enumerate(batch_texts):
            got = tok.decode(ids[j])
            pred_word = tok.itos[tops[j][0]] if tops[j][0] < len(tok.itos) else "<unk>"
            signer = batch_signers[j]
            ok = pred_word == want
            in5 = want in [tok.itos[i] if i < len(tok.itos) else "<unk>" for i in tops[j]]
            n += 1
            top1 += ok
            top5 += in5
            exact += (got == want)
            per_sign[want]["n"] += 1
            per_sign[want]["correct"] += ok
            per_sign[want]["top5"] += in5
            per_signer[signer]["n"] += 1
            per_signer[signer]["top1"] += ok
            per_signer[signer]["exact"] += (got == want)
            confusion[cls_index[want], cls_index[pred_word]] += 1
            if shown < args.show:
                shown += 1
                print("  %-22s want=%-14r pred=%-14r %s"
                      % ("clip %d" % n, want, pred_word, "ok" if ok else "MISS"))

    # per-sign precision/recall/F1 from the confusion matrix
    rows = []
    for name in classes:
        i = cls_index[name]
        tp = int(confusion[i, i])
        support = int(confusion[i, :].sum())
        predicted = int(confusion[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        wrong = confusion[i, :].copy()
        wrong[i] = 0
        if wrong.sum():
            j = int(wrong.argmax())
            top_conf = {"sign": classes[j], "n": int(wrong[j])}
        else:
            top_conf = None
        rows.append({"sign": name, "support": support, "correct": tp,
                     "recall": round(recall, 4),
                     "precision": round(precision, 4),
                     "f1": round(f1, 4),
                     "top5_recall": round(per_sign[name]["top5"] / support, 4)
                     if support else 0.0,
                     "most_confused_with": top_conf})

    macro_top1 = float(np.mean([r["recall"] for r in rows])) if rows else 0.0
    signer_report = {s: {"exact": round(v["top1"] / v["n"], 4),
                         "sequence_exact": round(v["exact"] / v["n"], 4),
                         "n": v["n"]}
                     for s, v in sorted(per_signer.items(), key=lambda kv: str(kv[0]))}

    pair_counts = []
    for i, name in enumerate(classes):
        for j in range(n_cls):
            if i != j and confusion[i, j]:
                pair_counts.append((int(confusion[i, j]), name, classes[j]))
    pair_counts.sort(reverse=True)

    report = {
        "schema": 1,
        "checkpoint": display_path(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "dataset": {"kind": "pack" if args.pack else "manifest",
                    "path": display_path(args.pack or args.manifest),
                    "frozen_files_verified": verified,
                    "sha256sums_present": bool(args.pack and
                                               (pathlib.Path(args.pack) / "SHA256SUMS").exists())},
        "split": split,
        "clips": n,
        "classes": n_cls,
        "params": sum(p.numel() for p in model.parameters()),
        "metrics": {
            "top1": round(top1 / n, 4),
            "top1_ci95": wilson_ci(top1, n),
            "top5": round(top5 / n, 4),
            "exact_match": round(exact / n, 4),
            "macro_top1": round(macro_top1, 4),
        },
        "per_signer": signer_report,
        "per_sign": rows,
        "top_confusions": [{"true": t, "predicted": p, "n": c}
                           for c, t, p in pair_counts[:30]],
    }
    return report, confusion, classes


def write_outputs(report, confusion, classes, out_dir):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    with open(out / "per_sign.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sign", "support", "correct", "recall",
                                          "precision", "f1", "top5_recall",
                                          "most_confused_with"])
        w.writeheader()
        for r in report["per_sign"]:
            r = dict(r)
            r["most_confused_with"] = ("" if not r["most_confused_with"]
                                       else "%s:%d" % (r["most_confused_with"]["sign"],
                                                       r["most_confused_with"]["n"]))
            w.writerow(r)

    np.savez_compressed(out / "confusion_matrix.npz", matrix=confusion,
                        classes=np.array(classes))
    with open(out / "confusion_matrix.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["true\\pred"] + classes)
        for name, row in zip(classes, confusion):
            w.writerow([name] + [int(x) for x in row])

    m = report["metrics"]
    lines = ["# Evaluation report", "",
             "Checkpoint: `%s`  " % report["checkpoint"],
             "sha256: `%s`  " % report["checkpoint_sha256"],
             "Eval set: `%s` (%d clips, %d signs)  "
             % (report["dataset"]["path"], report["clips"], report["classes"]),
             "Frozen files verified: %d  " % len(report["dataset"]["frozen_files_verified"]),
             "Split: %s" % ((report["split"] or {}).get("name", "not pinned")),
             "", "| metric | value |", "|---|---|",
             "| top-1 | **%.2f%%** %s |" % (100 * m["top1"],
                                            m["top1_ci95"]),
             "| top-5 | %.2f%% |" % (100 * m["top5"]),
             "| exact match | %.2f%% |" % (100 * m["exact_match"]),
             "| macro top-1 | %.2f%% |" % (100 * m["macro_top1"]), "",
             "## Top-1 per held-out signer", "",
             "| signer | clips | top-1 |", "|---|---|---|"]
    for s, v in report["per_signer"].items():
        lines.append("| %s | %d | %.2f%% |" % (s, v["n"], 100 * v["exact"]))
    lines += ["", "## Most frequent confusions", "",
              "| true | predicted | n |", "|---|---|---|"]
    for c in report["top_confusions"][:15]:
        lines.append("| %s | %s | %d |" % (c["true"], c["predicted"], c["n"]))
    worst = sorted(report["per_sign"], key=lambda r: (r["recall"], -r["support"]))[:10]
    lines += ["", "## Lowest per-sign top-1", "",
              "| sign | support | top-1 | most confused with |",
              "|---|---|---|---|"]
    for r in worst:
        mc = r["most_confused_with"]
        lines.append("| %s | %d | %.1f%% | %s |"
                     % (r["sign"], r["support"], 100 * r["recall"],
                        "-" if not mc else "%s (%d)" % (mc["sign"], mc["n"])))
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="top-1 + per-sign confusion on a held-out signer")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pack", help="packed eval set (dir with index.jsonl + shards/)")
    src.add_argument("--manifest", help="per-clip JSONL eval manifest")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="", help="frozen split JSON (held_out_signers)")
    ap.add_argument("--out-dir", default="reports/latest")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=6,
                    help="decode steps; single-word targets terminate at <eos>")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--device", default="")
    ap.add_argument("--no-verify-hashes", action="store_true",
                    help="skip SHA256SUMS verification of the frozen pack")
    ap.add_argument("--json-out", default="",
                    help="also copy report.json to this path")
    a = ap.parse_args(argv)

    report, confusion, classes = evaluate(a)
    write_outputs(report, confusion, classes, a.out_dir)
    if a.json_out:
        pathlib.Path(a.json_out).write_text(json.dumps(report, indent=2) + "\n")

    m = report["metrics"]
    print()
    print("clips: %d | signs: %d | params: %.2fM"
          % (report["clips"], report["classes"], report["params"] / 1e6))
    print("top-1: %.4f  (95%% CI %.4f-%.4f)" % (m["top1"], *m["top1_ci95"]))
    print("top-5: %.4f | exact match: %.4f | macro top-1: %.4f"
          % (m["top5"], m["exact_match"], m["macro_top1"]))
    for s, v in report["per_signer"].items():
        print("  signer %-8s top-1=%.4f  n=%d" % (s, v["exact"], v["n"]))
    print("wrote %s/{report.json,per_sign.csv,confusion_matrix.npz,SUMMARY.md}"
          % a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
