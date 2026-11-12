"""Corpus I/O: per-clip manifests, packed shards, and datasets.

Two storage shapes, one sample contract. A dataset item is always

    (feats: float32 (T, 225), target_ids: int64 (L,))

Manifest (per-clip):

    {"features": "/path/clip.npy", "text": "hello", "signer": "4718"}

Pack (``pack_features`` output), for corpora with more than a few thousand
clips:

    <out>/shards/shard_0000.f32   raw float32, conceptually (frames, 225)
    <out>/index.jsonl             {"clip","shard","offset","T","text","signer"}
    <out>/pack.json               summary + extractor id

``offset`` is in FRAMES, not bytes: a clip is
``shard[offset*225:(offset+T)*225].reshape(T, 225)``. 80,647 separate .npy
files is bad twice over -- one round trip per file to transfer, one np.load per
sample to train -- and packing cuts the file count by four orders of magnitude
while turning random reads into sequential ones.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import augment as A
from .contract import FEATURE_DIM, as_features, normalize_landmarks
from .model import WordTokenizer


# ---------------------------------------------------------------------------
# manifests
# ---------------------------------------------------------------------------

def read_manifest(path, require_exists=True):
    """JSONL -> list of dicts. Unreadable rows are skipped, not guessed."""
    rows = []
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if "features" not in o or "text" not in o:
            raise ValueError("%s: row has no features/text: %r" % (path, o))
        if require_exists and not Path(o["features"]).expanduser().exists():
            continue
        o["features"] = str(Path(o["features"]).expanduser())
        o["text"] = str(o["text"])
        rows.append(o)
    return rows


def write_manifest(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# per-clip dataset
# ---------------------------------------------------------------------------

class DirectDataset(Dataset):
    """Manifest -> (feats, target ids). Augmentation runs per __getitem__, so
    every epoch sees a different one-handed/mirrored variant of a clip."""

    def __init__(self, pairs, tokenizer, augment=True, cfg=None, seed=0):
        self.pairs = [(str(p), str(t)) for p, t in pairs]
        self.tok = tokenizer
        self.augment = augment
        self.cfg = cfg or A.AugmentConfig()
        self.seed = seed

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        path, text = self.pairs[i]
        feats = as_features(np.load(path).astype(np.float32))
        rng = np.random.default_rng((self.seed + i) % (2 ** 31))
        if self.augment:
            feats = A.augment(feats, self.cfg, rng)
        feats = normalize_landmarks(feats)
        ids = self.tok.encode(text)
        return torch.from_numpy(feats), torch.tensor(ids, dtype=torch.long)


def collate_train(batch):
    """Pad source and target. Mask convention: True = ignore."""
    feats, tgt = zip(*batch)
    lens = torch.tensor([f.size(0) for f in feats], dtype=torch.long)
    Tmax = int(lens.max())
    src = torch.zeros(len(feats), Tmax, FEATURE_DIM)
    for i, f in enumerate(feats):
        src[i, : f.size(0)] = f
    src_mask = torch.arange(Tmax).unsqueeze(0) >= lens.unsqueeze(1)
    L = max(int(t.numel()) for t in tgt) + 2          # <sos> ... <eos>
    ids = torch.full((len(tgt), L), WordTokenizer.PAD, dtype=torch.long)
    for i, t in enumerate(tgt):
        ids[i, 0] = WordTokenizer.SOS
        ids[i, 1:t.numel() + 1] = t
        ids[i, t.numel() + 1] = WordTokenizer.EOS
    return src, src_mask, ids


def collate_eval(batch):
    feats, texts = zip(*batch)
    lens = torch.tensor([f.size(0) for f in feats], dtype=torch.long)
    Tmax = int(lens.max())
    src = torch.zeros(len(feats), Tmax, FEATURE_DIM)
    for i, f in enumerate(feats):
        src[i, : f.size(0)] = f
    src_mask = torch.arange(Tmax).unsqueeze(0) >= lens.unsqueeze(1)
    return src, src_mask, texts


# ---------------------------------------------------------------------------
# packing
# ---------------------------------------------------------------------------

def pack(manifest, out, shard_mb=512, limit=0, verbose=True):
    """Write shards + index.jsonl. Returns a summary dict."""
    rows = read_manifest(manifest)
    if limit:
        rows = rows[:limit]
    if not rows:
        raise SystemExit("manifest %s is empty" % manifest)

    out = Path(out)
    shards = out / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.jsonl"
    if index_path.exists():
        index_path.unlink()

    budget = int(shard_mb) * 1024 * 1024
    buf, nbytes, offset, shard_no = [], 0, 0, 0
    written, total_bytes = 0, 0
    index_lines = []

    def flush():
        nonlocal buf, nbytes, offset, shard_no, written, index_lines, total_bytes
        if not buf:
            return
        name = "shard_%04d.f32" % shard_no
        arr = np.ascontiguousarray(np.concatenate(buf, axis=0), dtype=np.float32)
        arr.tofile(shards / name)
        with open(index_path, "a") as fh:
            for rec in index_lines:
                fh.write(json.dumps(rec) + "\n")
        written += arr.shape[0]
        total_bytes += arr.nbytes
        shard_no += 1
        buf, nbytes, offset, index_lines = [], 0, 0, []

    for i, row in enumerate(rows):
        src = row["features"]
        try:
            a = np.load(src)
        except Exception as exc:                      # noqa: BLE001
            raise SystemExit("failed to load %s: %s" % (src, exc))
        if a.ndim != 2 or a.shape[1] != FEATURE_DIM:
            raise SystemExit("%s has shape %r, need (T, %d)"
                             % (src, a.shape, FEATURE_DIM))
        a = np.ascontiguousarray(a, dtype=np.float32)
        index_lines.append({
            "clip": os.path.basename(src)[:-4],
            "shard": "shard_%04d.f32" % shard_no,
            "offset": offset,
            "T": int(a.shape[0]),
            "text": row.get("text"),
            "signer": row.get("signer"),
        })
        buf.append(a)
        nbytes += a.nbytes
        offset += a.shape[0]
        if nbytes >= budget:
            flush()
        if verbose and (i + 1) % 10000 == 0:
            print("  %6d clips  %8.1f MB" % (i + 1, (total_bytes + nbytes) / 1e6))
    flush()

    meta = {}
    mp = Path(manifest).expanduser().parent / "extractor_meta.json"
    if mp.exists():
        meta = {"extractor": json.loads(mp.read_text()).get("extractor")}
    if not meta.get("extractor"):
        try:
            from .extract import EXTRACTOR_ID
            meta = {"extractor": EXTRACTOR_ID}
        except Exception:                             # noqa: BLE001
            meta = {"extractor": None}

    summary = dict(meta, clips=len(rows), frames=written, shards=shard_no,
                   bytes=total_bytes, out=str(out), source_dir=str(
                       Path(rows[0]["features"]).parent))
    (out / "pack.json").write_text(json.dumps(summary, indent=1) + "\n")
    if verbose:
        print(json.dumps(summary, indent=1))
    return summary


def verify(pack_dir, manifest, n=200, seed=0):
    """Load n clips from a pack and compare against their source .npy files.

    Manifest rows are matched BY CLIP NAME, not positionally, so a pack built
    with --limit (a prefix of the manifest) still verifies.
    """
    pack_dir = Path(pack_dir)
    index = read_index(pack_dir)
    rows = read_manifest(manifest)
    by_clip = {os.path.basename(r["features"])[:-4]: r["features"] for r in rows}
    missing = [r["clip"] for r in index if r["clip"] not in by_clip]
    if missing:
        raise SystemExit("%d packed clips are absent from the manifest, e.g. %s"
                         % (len(missing), missing[:3]))

    rng = np.random.default_rng(seed)
    picks = rng.choice(len(index), size=min(n, len(index)), replace=False)
    mm, bad = {}, 0
    for i in picks:
        rec = index[int(i)]
        src = by_clip[rec["clip"]]
        m = mm.get(rec["shard"])
        if m is None:
            m = np.memmap(pack_dir / "shards" / rec["shard"], dtype=np.float32,
                          mode="r")
            mm[rec["shard"]] = m
        stack = np.asarray(m[rec["offset"] * FEATURE_DIM:
                             (rec["offset"] + rec["T"]) * FEATURE_DIM])
        got = stack.reshape(rec["T"], FEATURE_DIM)
        want = np.load(src).astype(np.float32)
        if got.shape != want.shape or not np.array_equal(got, want):
            print("MISMATCH %s: got %r want %r" % (src, got.shape, want.shape))
            bad += 1
    print("verified %d clips, %d mismatches" % (len(picks), bad))
    return bad == 0


def read_index(pack):
    p = Path(pack)
    if p.is_dir():
        p = p / "index.jsonl"
    if not p.exists():
        raise SystemExit("no index.jsonl at %s" % p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# packed dataset
# ---------------------------------------------------------------------------

class ShardedDirectDataset(Dataset):
    """DirectDataset replacement over packed shards.

    cache="memmap" (default) opens a memmap per item: cheap to start, ~700
    clips/s single-threaded. cache="ram" loads the shards once in the PARENT
    process; forked DataLoader workers share the pages copy-on-write. That is
    the right choice on a discrete-GPU host and the wrong one on an Orin, where
    host and device memory are one pool: anonymous cache pages are not
    evictable, and a big RAM cache plus many workers has hard-rebooted the
    board. File-backed memmap pages are evictable, which is why memmap is the
    default here.
    """

    def __init__(self, index_path, tokenizer, augment=True, cfg=None, seed=0,
                 rows=None, cache="memmap", ram_cache=None):
        self.index_path = Path(index_path)
        if self.index_path.is_dir():
            self.index_path = self.index_path / "index.jsonl"
        self.shard_dir = self.index_path.parent / "shards"
        self.index = read_index(self.index_path)
        if rows is not None:
            self.index = [self.index[i] for i in rows]
        self.tok = tokenizer
        self.augment = augment
        self.cfg = cfg or A.AugmentConfig()
        self.seed = seed
        self.cache = cache
        self._ram = ram_cache
        if self._ram is None and cache == "ram":
            self._ram = self._load_ram()

    def _load_ram(self):
        names = sorted({r["shard"] for r in self.index})
        return {n: np.fromfile(self.shard_dir / n, dtype=np.float32)
                for n in names}

    def __len__(self):
        return len(self.index)

    def _read(self, rec):
        lo = rec["offset"] * FEATURE_DIM
        hi = (rec["offset"] + rec["T"]) * FEATURE_DIM
        if self._ram is not None:
            flat = self._ram[rec["shard"]][lo:hi]
        else:
            m = np.memmap(self.shard_dir / rec["shard"], dtype=np.float32,
                          mode="r")
            flat = np.asarray(m[lo:hi])
        return flat.reshape(rec["T"], FEATURE_DIM).astype(np.float32)

    def __getitem__(self, i):
        rec = self.index[i]
        feats = self._read(rec)
        rng = np.random.default_rng((self.seed + i) % (2 ** 31))
        if self.augment:
            feats = A.augment(feats, self.cfg, rng)
        feats = normalize_landmarks(feats)
        ids = self.tok.encode(rec["text"])
        return torch.from_numpy(feats), torch.tensor(ids, dtype=torch.long)


__all__ = ["read_manifest", "write_manifest", "DirectDataset", "collate_train",
           "collate_eval", "pack", "verify", "read_index",
           "ShardedDirectDataset"]
