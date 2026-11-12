#!/usr/bin/env python3
r"""Google ISLR (asl-signs) TFRecords -> (T, 225) features + manifest.

WHY TFRecords AND NOT TENSORFLOW
--------------------------------
The Google Isolated Sign Language Recognition corpus (250 ASL signs, 94477
clips, MediaPipe Holistic landmarks) is published as parquet on the competition
page, which needs the competition rules accepted before the API will serve it
(403 with a normal token). The community mirrors are TFRecord, which is why this
file exists.

TFRecord is not a TensorFlow format, it is a trivial one: a repeated sequence of
[uint64 length][payload][uint32 crc], and the payload is a serialized
tf.train.Example. Both halves are a few dozen lines of stdlib, and the alternative
is installing tensorflow on the Orin, which is a multi-hundred-MB wheel that
drags its own numpy into a box whose ML stack is pinned and verified. So the
protobuf is decoded here by hand (varint + wire type only, no .proto compiler)
and nothing on the box changes.

    python scripts/build_islr_corpus.py --root ~/data/datasets/asl-islr --probe 2
    python scripts/build_islr_corpus.py --root ~/data/datasets/asl-islr \
        --out ~/data/deafwork/asl --min-hand-rate 0.35

The row mapping itself is NOT restated here: frames_to_contract in
parquet_to_features.py is the only projection onto the contract, so a corpus
added later cannot disagree with this one about which landmark is row 63.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from signstack import extract as X  # noqa: E402
from signstack.contract import FEATURE_DIM, detection_rate, write_extractor_meta  # noqa: E402
from signstack.convert import frames_to_contract, hand_rate, resample  # noqa: E402

# Key aliases: the mirrors were made by different people, so the same quantity
# appears under several names. First match wins, and --probe prints what a file
# actually carries so a new mirror needs no guesswork.
ALIASES = {
    "pose": ("pose_landmarks", "pose", "pose_landmark", "pose_keypoints"),
    "face": ("face_landmarks", "face", "face_landmark"),
    "left_hand": ("left_hand_landmarks", "left_hand", "left_hand_landmark"),
    "right_hand": ("right_hand_landmarks", "right_hand", "right_hand_landmark"),
    "sign": ("sign", "label", "class", "gloss"),
    "fps": ("fps",),
}
LANDS = {"pose": 33, "face": 468, "left_hand": 21, "right_hand": 21}


# ---- the protobuf and the framing, by hand ----------------------------------

def _varint(buf, i):
    val, shift = 0, 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, i
        shift += 7


def _fields(buf):
    """bytes -> {field_number: [(wire_type, value), ...]}. Value is an int for
    varints, a byte string for everything else."""
    out, i, n = {}, 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        field, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 1:
            v, i = buf[i:i + 8], i + 8
        elif wt == 2:
            ln, i = _varint(buf, i)
            v, i = buf[i:i + ln], i + ln
        elif wt == 5:
            v, i = buf[i:i + 4], i + 4
        else:
            raise ValueError("unsupported protobuf wire type %d" % wt)
        out.setdefault(field, []).append((wt, v))
    return out


def _floats(entries):
    """FloatList -> list[float]. Handles packed and unpacked, because the same
    file can carry either depending on who wrote it."""
    vals = []
    for wt, v in entries:
        if wt == 5:
            vals.append(struct.unpack("<f", v)[0])
        elif wt == 2:
            vals.extend(struct.unpack("<%df" % (len(v) // 4), v))
        elif wt == 0:
            vals.append(float(v))
    return vals


def _ints(entries):
    vals = []
    for wt, v in entries:
        if wt == 0:
            vals.append(v)
        elif wt == 2:
            i = 0
            while i < len(v):
                x, i = _varint(v, i)
                vals.append(x)
    return vals


# tensorflow's TensorProto, which is how one of the mirrors stores its payloads:
# the Example holds a bytes_list whose single value is a serialized TensorProto,
# not a FloatList. Field numbers are from tensor.proto.
TP_DTYPE = {1: np.float32, 3: np.int32, 9: np.int64, 7: np.float64}
TP_SHAPE = 2
TP_CONTENT = 4
TP_INT_VAL, TP_FLOAT_VAL, TP_STRING_VAL = 8, 6, 9


def parse_tensorproto(buf):
    """serialized TensorProto -> (shape tuple, ndarray). Raises when it is not one.

    A bytes_list value that reaches here is either a TensorProto or opaque bytes;
    the caller decides which by catching ValueError.
    """
    f = _fields(buf)
    if 1 not in f or f[1][0][0] != 0:
        raise ValueError("no dtype field: not a TensorProto")
    dtype = f[1][0][1]
    if dtype not in TP_DTYPE:
        raise ValueError("dtype %r is not a numpy-supported tensor here" % dtype)
    shape = []
    if TP_SHAPE in f:
        for _, dim in _fields(f[TP_SHAPE][0][1]).get(1, []):
            d = _fields(dim)
            shape.append(d[1][0][1] if 1 in d else -1)
    if TP_CONTENT in f:                                  # always how this mirror writes
        arr = np.frombuffer(f[TP_CONTENT][0][1], dtype=TP_DTYPE[dtype])
    elif TP_FLOAT_VAL in f:
        arr = np.asarray(_floats(f[TP_FLOAT_VAL]), np.float32)
    elif TP_INT_VAL in f:
        arr = np.asarray(_ints(f[TP_INT_VAL]), np.int64).astype(TP_DTYPE[dtype])
    else:
        arr = np.zeros(0, TP_DTYPE[dtype])
    if shape and int(np.prod(shape)) != arr.size:
        raise ValueError("TensorProto shape %r does not match %d values"
                         % (shape, arr.size))
    return tuple(shape), (arr.reshape(shape) if shape else arr)


def parse_example(payload):
    """tf.train.Example bytes -> {key: [float|int|bytes]}.

    Example{ features=1 }
      Features{ feature=1 }             map<string, Feature>, one entry per key
        entry{ key=1 string, value=2 Feature }
        Feature{ bytes_list=1 | float_list=2 | int64_list=3 }
    """
    top = _fields(payload)
    if 1 not in top:
        raise ValueError("not a tf.train.Example (no Features field)")
    out = {}
    for _, entry in _fields(top[1][0][1]).get(1, []):
        e = _fields(entry)
        key = e[1][0][1].decode("utf-8", "replace")
        feat = _fields(e[2][0][1])
        # Each list is a sub-message with its repeated values in field 1, so the
        # values sit one level below the Feature field that names them.
        if 3 in feat:
            out[key] = _ints(_fields(feat[3][0][1]).get(1, []))
        elif 2 in feat:
            out[key] = _floats(_fields(feat[2][0][1]).get(1, []))
        elif 1 in feat:
            out[key] = [v for _, v in _fields(feat[1][0][1]).get(1, [])]
    return out


def records(path):
    """Yield serialized Example payloads from a TFRecord file.

    The frame is [uint64 length][uint32 crc32c-of-length][data][uint32 crc-of-
    data] -- ELEVEN bytes of header, not eight. Reading only eight puts four CRC
    bytes at the head of every payload, which does not fail loudly: the CRC looks
    like a plausible protobuf field tag and the parse dies several fields in with
    "unsupported wire type 4". Read the length CRC even though nothing here
    verifies it.
    """
    with open(path, "rb") as fh:
        while True:
            head = fh.read(12)
            if len(head) < 12:
                return
            (n,) = struct.unpack("<Q", head[:8])
            payload = fh.read(n)
            if len(payload) < n:
                return
            fh.read(4)                                  # crc of the payload
            yield payload


def pick(rec, kind):
    for k in ALIASES[kind]:
        if k in rec:
            return k
    return None


def block_to_array(vals, n_land, what):
    a = np.asarray(vals, np.float32)
    if a.size == 0 or a.size % (n_land * 3):
        raise ValueError("%s: %d floats is not a whole number of (T, %d, 3)"
                         % (what, a.size, n_land))
    return a.reshape(-1, n_land, 3)


def record_to_features(rec):
    """One Example -> (T, 225) float32, plus the sign text and fps."""
    got = {}
    for kind in LANDS:
        key = pick(rec, kind)
        got[kind] = block_to_array(rec[key], LANDS[kind], key) if key else None
    sign_key = pick(rec, "sign")
    sign = None
    if sign_key:
        v = rec[sign_key]
        sign = v[0].decode("utf-8", "replace") if isinstance(v[0], (bytes, bytearray)) else str(v[0])
    fps_key = pick(rec, "fps")
    fps = float(rec[fps_key][0]) if fps_key and rec[fps_key] else 0.0
    # T is whatever pose says; the other blocks must agree or the corpus is
    # internally inconsistent and would silently misalign every frame.
    T = got["pose"].shape[0]
    for kind, a in got.items():
        if a is not None and a.shape[0] != T:
            raise ValueError("%s has %d frames, pose has %d" % (kind, a.shape[0], T))
    return (frames_to_contract(got["pose"], got["left_hand"], got["right_hand"],
                               got["face"]), sign, fps)


# ---- modes ------------------------------------------------------------------

def probe(root: pathlib.Path, n: int) -> int:
    files = sorted(root.rglob("*.tfrecord")) + sorted(root.rglob("*.tfrecords"))
    if not files:
        sys.exit("no .tfrecord under %s" % root)
    print("tfrecord files: %d" % len(files))
    it = records(files[0])
    for i in range(n):
        rec = parse_example(next(it))
        print("--- record %d" % i)
        for k in sorted(rec):
            v = rec[k]
            if v and isinstance(v[0], (bytes, bytearray)):
                print("  %-26s bytes   %d x %s  e.g. %r" % (k, len(v), len(v[0]), v[0][:40]))
            elif v:
                a = np.asarray(v, np.float32)
                print("  %-26s float   %d vals  min %.4f max %.4f  first %s"
                      % (k, len(v), a.min(), a.max(), np.round(a[:3], 4)))
            else:
                print("  %-26s empty" % k)
        try:
            f, sign, fps = record_to_features(rec)
            print("  -> contract %r sign=%r fps=%s hand_rate=%.3f"
                  % (f.shape, sign, fps, hand_rate(f)))
        except Exception as e:
            print("  -> could not be projected: %s" % e)
    return 0


def convert(a) -> int:
    root = pathlib.Path(a.root).expanduser()
    out = pathlib.Path(a.out).expanduser()
    feats_dir = out / "features"
    feats_dir.mkdir(parents=True, exist_ok=True)
    write_extractor_meta(feats_dir, X.MPH_ID, extra={
        "source": "Google ISLR (asl-signs) TFRecords -> tfrecord_to_features.py",
        "third_channel": "detection flag (1.0 present, 0.0 absent), NOT visibility",
        "swap_hands": bool(a.swap_hands),
        "fps_in": a.fps_in or None, "fps_out": a.out_rate or None})

    files = sorted(root.rglob("*.tfrecord")) + sorted(root.rglob("*.tfrecords"))
    print("tfrecord files: %d under %s" % (len(files), root))
    t0 = time.time()
    manifest, stats, dropped = [], [], Counter()
    seen_signs, seen_fps = Counter(), Counter()
    i = 0
    for f in files:
        if a.limit and i >= a.limit:
            break
        for payload in records(f):
            i += 1
            try:
                feats, sign, fps = record_to_features(parse_example(payload))
            except Exception as e:
                dropped["read:" + type(e).__name__] += 1
                continue
            if sign is None:
                dropped["no_sign"] += 1
                continue
            sign = sign.strip().lower()
            seen_signs[sign] += 1
            seen_fps[fps] += 1
            if feats.shape[0] < a.min_frames:
                dropped["too_short"] += 1
                continue
            feats = resample(feats, a.fps_in or fps, a.out_rate)
            hr = hand_rate(feats)
            if hr < a.min_hand_rate:
                dropped["low_hand_rate"] += 1
                continue
            cid = "%s__%s_%06d" % (sign, f.stem.replace(".", "_"), i)
            npy = feats_dir / (cid + ".npy")
            if npy.exists() and not a.force:
                dropped["cached"] += 1
            else:
                np.save(npy, feats.astype(np.float32))
            stats.append({"clip": cid, "gloss": sign, "frames": int(feats.shape[0]),
                          "hand_rate": round(hr, 4),
                          "pose_rate": round(detection_rate(feats, "body"), 4),
                          "face_rate": round(detection_rate(feats, "face"), 4)})
            manifest.append({"features": str(npy), "text": sign})
            if i % 500 == 0:
                el = time.time() - t0
                print("  %6d records  kept %d  %.0fs (%.0f/s)"
                      % (i, len(stats), el, i / max(el, 1e-6)), flush=True)
            if a.max_clips and len(stats) >= a.max_clips:
                break
        if a.max_clips and len(stats) >= a.max_clips:
            print("reached --max-clips %d" % a.max_clips)
            break

    with open(out / "manifest.jsonl", "w") as fh:
        for row in manifest:
            fh.write(json.dumps(row) + "\n")
    glosses = Counter(s["gloss"] for s in stats)
    frames = np.array([s["frames"] for s in stats]) if stats else np.zeros(1)
    hrs = np.array([s["hand_rate"] for s in stats]) if stats else np.zeros(1)
    summary = {
        "source_root": str(root), "corpus_root": str(out),
        "extractor": X.MPH_ID, "feature_dim": FEATURE_DIM,
        "records_read": i, "clips": len(stats), "glosses": len(glosses),
        "fps_seen": {str(k): v for k, v in seen_fps.most_common(5)},
        "frames_total": int(frames.sum()), "frames_mean": round(float(frames.mean()), 1),
        "frames_min": int(frames.min()), "frames_max": int(frames.max()),
        "hand_rate_mean": round(float(hrs.mean()), 4),
        "hand_rate_min": round(float(hrs.min()), 4),
        "clips_below_0_35_hand_rate": int((hrs < 0.35).sum()),
        "gloss_clip_counts_top": glosses.most_common(5),
        "gloss_clip_counts_least": glosses.most_common()[-5:],
        "glosses_with_fewer_than_2_clips": sorted(g for g, n in glosses.items() if n < 2),
        "dropped": dict(dropped),
    }
    (out / "corpus.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nwrote %s/manifest.jsonl (%d pairs)" % (out, len(manifest)))
    return 0


# ---- the frame-stream mirrors ----------------------------------------------
#
# Two of the public mirrors (soumikrakshit/asl-signs-tfrecords and the smaller
# isolated-sign-language-aggregation-dataset) are not what their size suggests.
# A record is ONE FRAME, not one clip: 'point_cloud' is a float32 TensorProto of
# a single frame and 'label' is that frame's gloss id. Records are in clip order
# and each tfrecord file is named for the CLIP range it covers
# (tfrecord-0-511.tfrecord holds the frames of clips 0..511), so the clip
# boundaries are not in the tfrecord at all -- they come from the competition's
# train.csv, whose num_frames column is the only record of how long each clip is.
# Checked against the files: records in a file == sum(num_frames) of its clip
# range, exactly, for every file sampled.
#
# THE FLATTENING. 543 landmarks per frame, in the order the competition's parquet
# emitted them, which is alphabetical by row_id:
#
#     0   .. 467  face        468 points, present in 99.8% of frames
#     468 .. 488  left_hand   21, present in 20%
#     489 .. 521  pose        33, present in 100%
#     522 .. 542  right_hand  21, present in 34%
#
# Established from the data rather than a readme: the finite-rate runs break
# exactly at 468/489/522, and the 489..521 block reports nose at 0, shoulders at
# 11/12, wrists at 15/16 and feet at 31/32 with the person's left on the image's
# right, which is a non-mirrored frame and matches the anatomical labels. Getting
# this wrong is silent: any contiguous 543-long slice "works" in a model.
#
# A MISSING LANDMARK IS NaN HERE, not zero. The others use zero. Both arrive at
# the contract the same way -- not finite or all-zero means confidence 0.0.
#
# COORDINATES are normalised by frame WIDTH for x and y both, so y legitimately
# exceeds 1 on a full-body shot (measured: feet at y ~ 2.0). That is a scale, not
# a coordinate space, and normalize_landmarks divides it out; what must not
# happen is treating y>1 as corrupt and clipping it.

FRAME_LAYOUT = {"face": (0, 468), "left_hand": (468, 489),
                "pose": (489, 522), "right_hand": (522, 543)}
FRAME_LANDMARKS = 543


def load_clip_table(path):
    """The competition's train.csv -> one dict per clip, in record order."""
    import csv
    clips = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            clips.append({"sign": r["sign"].strip().lower(),
                          "label": int(r["label"]),
                          "frames": int(float(r["num_frames"])),
                          "signer": str(r["participant_id"]),
                          "sequence_id": str(r["sequence_id"])})
    return clips


def _range_of(path):
    parts = pathlib.Path(path).stem.split("-")
    return int(parts[-2]), int(parts[-1])


def frame_files(root):
    """tfrecord files ordered by their CLIP range, not by name.

    Lexicographic order puts -1024- before -10240-, which would shuffle clips and
    silently misalign every clip after the first such pair.
    """
    fs = list(pathlib.Path(root).rglob("*.tfrecord")) + \
         list(pathlib.Path(root).rglob("*.tfrecords"))
    return sorted(fs, key=lambda f: _range_of(f)[0])


def frame_to_blocks(rec):
    """One frame record -> the four (1, N, 3) blocks frames_to_contract wants."""
    key = pick(rec, "pose") or "point_cloud"
    if key not in rec:
        key = sorted(rec)[0]
    _, arr = parse_tensorproto(rec[key][0])
    flat = np.asarray(arr, np.float32).reshape(-1)
    if flat.size != FRAME_LANDMARKS * 3:
        raise ValueError("%s holds %d values, expected %d (a frame of %d landmarks)"
                         % (key, flat.size, FRAME_LANDMARKS * 3, FRAME_LANDMARKS))
    flat = flat.reshape(FRAME_LANDMARKS, 3)
    return {name: flat[lo:hi][None, ...] for name, (lo, hi) in FRAME_LAYOUT.items()}


def convert_frames(a) -> int:
    """The frame-stream mirrors -> a corpus, rebuilding clips from clip.csv."""
    root = pathlib.Path(a.root).expanduser()
    out = pathlib.Path(a.out).expanduser()
    feats_dir = out / "features"
    feats_dir.mkdir(parents=True, exist_ok=True)
    clips = load_clip_table(pathlib.Path(a.clip_table).expanduser())
    files = frame_files(root)
    if a.max_files:
        files = files[:a.max_files]
    print("clips in the table: %d | tfrecord files: %d | frames expected: %d"
          % (len(clips), len(files), sum(c["frames"] for c in clips)))
    write_extractor_meta(feats_dir, X.MPH_ID, extra={
        "source": "Google ISLR (asl-signs) frames -> tfrecord_to_features.py",
        "clip_table": str(a.clip_table),
        "layout_543": {k: list(v) for k, v in FRAME_LAYOUT.items()},
        "missing_landmark": "NaN in the source; written as (0,0,confidence=0)",
        "third_channel": "detection flag (1.0 present, 0.0 absent), NOT visibility",
        "swap_hands": False,
        "fps_in": None, "fps_out": a.out_rate or None})

    t0, i, kept, dropped = time.time(), 0, 0, Counter()
    manifest, stats = [], []
    for f in files:
        lo, hi = _range_of(f)
        it = records(f)
        used = 0
        for ci in range(lo, hi + 1):
            clip = clips[ci]
            blocks = []
            for _ in range(clip["frames"]):
                try:
                    rec = parse_example(next(it))
                    blocks.append(frame_to_blocks(rec))
                except StopIteration:
                    dropped["file_ran_short"] += 1
                    break
                except Exception as e:
                    dropped["frame:" + type(e).__name__] += 1
                i += 1
                used += 1
            if not blocks:
                dropped["empty_clip"] += 1
                continue
            feats = frames_to_contract(
                np.concatenate([b["pose"] for b in blocks]),
                np.concatenate([b["left_hand"] for b in blocks]),
                np.concatenate([b["right_hand"] for b in blocks]),
                np.concatenate([b["face"] for b in blocks]))
            if feats.shape[0] < a.min_frames:
                dropped["too_short"] += 1
                continue
            feats = resample(feats, a.fps_in, a.out_rate)
            hr = hand_rate(feats)
            if hr < a.min_hand_rate:
                dropped["low_hand_rate"] += 1
                continue
            cid = "%s__%s_%s" % (clip["sign"], clip["signer"], clip["sequence_id"])
            npy = feats_dir / (cid + ".npy")
            if npy.exists() and not a.force:
                dropped["cached"] += 1
            else:
                np.save(npy, feats.astype(np.float32))
            stats.append({"clip": cid, "gloss": clip["sign"],
                          "signer": clip["signer"], "frames": int(feats.shape[0]),
                          "hand_rate": round(hr, 4)})
            manifest.append({"features": str(npy), "text": clip["sign"],
                             "signer": clip["signer"]})
            kept += 1
            if kept % 2000 == 0:
                el = time.time() - t0
                print("  %6d clips  %8d frames  %.0fs (%.0f frames/s)"
                      % (kept, i, el, i / max(el, 1e-6)), flush=True)
        leftover = sum(1 for _ in it)
        if leftover:
            dropped["file_ran_long"] += 1
            print("  !! %s had %d frames past clip %d" % (f.name, leftover, hi))
    with open(out / "manifest.jsonl", "w") as fh:
        for row in manifest:
            fh.write(json.dumps(row) + "\n")
    with open(out / "stats.json", "w") as fh:
        json.dump(stats, fh, indent=1)

    signs = Counter(s["gloss"] for s in stats)
    signers = Counter(s["signer"] for s in stats)
    hrs = np.array([s["hand_rate"] for s in stats]) if stats else np.zeros(1)
    summary = {
        "source_root": str(root), "corpus_root": str(out),
        "extractor": X.MPH_ID, "feature_dim": FEATURE_DIM,
        "frames_read": i, "clips": kept, "glosses": len(signs), "signers": len(signers),
        "signer_clip_counts": signers.most_common(6),
        "gloss_clip_counts_top": signs.most_common(5),
        "gloss_clip_counts_least": signs.most_common()[-5:],
        "glosses_with_fewer_than_2_clips": sorted(g for g, n in signs.items() if n < 2),
        "hand_rate_mean": round(float(hrs.mean()), 4),
        "hand_rate_below_0_35": int((hrs < 0.35).sum()),
        "dropped": dict(dropped),
    }
    (out / "corpus.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nwrote %s/manifest.jsonl (%d pairs)" % (out, len(manifest)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="~/data/datasets/asl-islr")
    ap.add_argument("--out", default="~/data/deafwork/asl")
    ap.add_argument("--probe", type=int, default=0, metavar="N",
                    help="print the keys and shapes of the first N records and exit")
    ap.add_argument("--clip-table", default="~/data/datasets/agg/train.csv",
                    help="the competition train.csv: sign, label, num_frames, participant_id")
    ap.add_argument("--frames-mode", action="store_true",
                    help="the records are one FRAME each; rebuild clips from --clip-table")
    ap.add_argument("--limit", type=int, default=0, help="stop after N records")
    ap.add_argument("--max-files", type=int, default=0,
                    help="frames mode: only the first N tfrecord files (pilot run)")
    ap.add_argument("--max-clips", type=int, default=0, help="stop after N kept clips")
    ap.add_argument("--min-hand-rate", type=float, default=0.0)
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--fps-in", type=float, default=0.0,
                    help="override the fps inside the record; with --out-rate, resample")
    ap.add_argument("--out-rate", type=float, default=0.0, help="e.g. 22 for the Orin")
    ap.add_argument("--swap-hands", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.probe:
        return probe(pathlib.Path(a.root).expanduser(), a.probe)
    if a.frames_mode:
        return convert_frames(a)
    return convert(a)


if __name__ == "__main__":
    raise SystemExit(main())
