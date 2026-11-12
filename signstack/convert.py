"""Project published landmark corpora onto the (T, 225) contract.

Every converter in the repo goes through `frames_to_contract`, so a corpus
added later cannot disagree with `extract.mph_frame_row` about which landmark
lands in which row.

A missing point is ``(0, 0, 0)``. Note that the third channel here is a
DETECTION FLAG (1.0 present, 0.0 absent), not a model confidence: the ISLR
TFRecords carry only coordinates, so a per-point score does not exist. That is
still consistent with the contract -- zero means "not detected" -- but it is
not the same distribution the video extractor emits, and the corpus stamp
records the difference.
"""

from __future__ import annotations

import numpy as np

from .contract import CHANNELS, FEATURE_DIM, hand_detection_rate
from .extract import MPH_FACE_468, _MPH_BODY


def frames_to_contract(pose, left_hand, right_hand, face,
                       swap_hands=False) -> np.ndarray:
    """(T, 33, 3) + (T, 21, 3) x2 + (T, 468, 3) -> (T, 225) float32.

    The single place a published corpus is projected onto the contract, so every
    converter here agrees with sl2t_extract.mph_frame_row about which landmark
    lands in which row. Arrays may be None for a group the corpus does not carry;
    a row whose point is absent (or all-zero) gets confidence 0.0, which is what
    detection_rate and normalize_landmarks read.

    Anything larger than the contract needs is sliced (pose 33 -> COCO-17, face
    468 -> the 16 mesh indices); a 478-point face (468 + iris) is accepted
    because the iris indices are above every index MPH_FACE_468 uses.
    """
    T = int(pose.shape[0]) if pose is not None and len(pose) else (
        int(left_hand.shape[0]) if left_hand is not None and len(left_hand) else 0)
    if T == 0:
        return np.zeros((0, FEATURE_DIM), np.float32)
    out = np.zeros((T, 75, CHANNELS), np.float32)

    def put(rows, src, block, min_len):
        if block is None or len(block) < T:
            return
        a = np.asarray(block, np.float32)[:T]
        if a.ndim != 3 or a.shape[1] < min_len:
            raise ValueError("block shape %r, need (T, >=%d, 3)" % (a.shape, min_len))
        for row, s in zip(rows, src):
            if s >= a.shape[1]:
                continue
            xyz = a[:, s, :3]
            live = np.isfinite(xyz).all(1) & (np.abs(xyz).sum(1) > 0)
            out[live, row, :2] = xyz[live, :2]
            out[live, row, 2] = 1.0

    put(range(17), [X._MPH_BODY[i] for i in range(17)], pose, 17)
    lh, rh = (right_hand, left_hand) if swap_hands else (left_hand, right_hand)
    put(range(17, 38), range(21), lh, 21)
    put(range(38, 59), range(21), rh, 21)
    put(range(59, 75), X.MPH_FACE_468, face, 16)
    return out.reshape(T, FEATURE_DIM)


def clip_to_features(table) -> np.ndarray:
    """One long-format parquet table -> (T, 225) float32."""
    frame = table.column("frame").to_numpy(zero_copy_only=False).astype(np.int32)
    lidx = table.column("landmark_index").to_numpy(zero_copy_only=False).astype(np.int32)
    x = table.column("x").to_numpy(zero_copy_only=False).astype(np.float32)
    y = table.column("y").to_numpy(zero_copy_only=False).astype(np.float32)
    z = table.column("z").to_numpy(zero_copy_only=False).astype(np.float32)
    types = table.column("type").to_numpy(zero_copy_only=False)

    code = np.zeros(len(types), np.int8)
    uniq, inv = np.unique(types, return_inverse=True)
    for i, name in enumerate(uniq):
        if name not in TYPES:
            raise ValueError("unknown landmark type %r (row_id convention changed?)"
                             % (name,))
        code[inv == i] = TYPES[name]

    # map (block, landmark_index) -> contract row, dropping what the contract
    # does not carry
    row = np.zeros(len(types), np.int32)
    keep = np.ones(len(types), bool)
    for c, lut in ((0, POSE_LUT), (1, HAND_LUT), (2, HAND_LUT), (3, FACE_LUT)):
        m = code == c
        if not m.any():
            continue
        idx = lidx[m]
        if idx.max(initial=0) >= len(lut):
            raise ValueError("landmark_index out of range for a %s row: max %d"
                             % (uniq[0], int(idx.max())))
        mapped = lut[idx]
        ok = mapped >= 0
        keep[np.flatnonzero(m)] = ok
        row[np.flatnonzero(m)] = BASE[c] + np.where(ok, mapped, 0)

    present = keep & ~((x == 0) & (y == 0) & (z == 0))
    frame = frame[present]
    row = row[present]
    x, y, z = x[present], y[present], z[present]

    if len(frame) == 0:
        return np.zeros((0, FEATURE_DIM), np.float32)

    T = int(frame.max()) + 1
    out = np.zeros((T, 75, CHANNELS), np.float32)
    flat = out.reshape(T * 75, CHANNELS)
    at = frame * 75 + row
    flat[at, 0] = x
    flat[at, 1] = y
    flat[at, 2] = 1.0                                # present; a miss stays 0
    return out.reshape(T, FEATURE_DIM)



def resample(feats: np.ndarray, fps_in: float, fps_out: float) -> np.ndarray:
    """Nearest-frame rate conversion. Temporal dynamics are a training property
    here (perception on the Orin runs ~22 fps), so a corpus captured at a
    different rate must be brought to the serving rate or it is a different
    problem than the one being served."""
    if not fps_in or not fps_out or abs(fps_in - fps_out) < 1e-6:
        return feats
    T = feats.shape[0]
    n_out = max(1, int(round(T * fps_out / fps_in)))
    idx = np.clip((np.arange(n_out) * fps_in / fps_out).round().astype(int), 0, T - 1)
    return feats[idx]


def hand_rate(feats: np.ndarray) -> float:
    return X.detection_rate(feats, "hand")



def hand_rate(feats: np.ndarray) -> float:
    return hand_detection_rate(feats)


__all__ = ["frames_to_contract", "resample", "hand_rate"]
