"""The (T, 225) landmark contract.

One clip is a float32 array of shape (T, 75, 3), flattened to (T, 225):

    body        rows 0..16    COCO-17 order
    left hand   rows 17..37   21 MediaPipe hand landmarks
    right hand  rows 38..58   21 landmarks
    face        rows 59..74   16 points: eyes, brows, mouth, nose

Each row is (x, y, confidence). Coordinates are normalised against the frame
(exactly how is the extractor's business), and a missing keypoint is
``(0, 0, 0)`` -- coordinates *and* confidence zeroed -- so a genuine detection
at the origin stays distinguishable from a miss.

The layout is written down once, here. Everything else imports it.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

FEATURE_DIM = 225
CHANNELS = 3
N_KEYPOINTS = 75
N_BODY = 17
N_HAND = 21
N_FACE = 16

# (name, first keypoint row, one past last keypoint row)
PART_ROWS = [("body", 0, 17), ("left_hand", 17, 38),
             ("right_hand", 38, 59), ("face", 59, 75)]

# Channel-space slices, which is what the model and augmentation index into.
PART_SLICES = {name: (lo * CHANNELS, hi * CHANNELS) for name, lo, hi in PART_ROWS}
BODY, LHAND, RHAND, FACE = (PART_SLICES["body"], PART_SLICES["left_hand"],
                            PART_SLICES["right_hand"], PART_SLICES["face"])

# COCO body rows. 5/6 shoulders, 11/12 hips: the torso anchor.
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12

EXTRACTOR_MPH = "mediapipe-holistic-225-v1"
EXTRACTOR_RTMPOSE = "rtmpose-wholebody133-v1"
DEFAULT_EXTRACTOR = EXTRACTOR_MPH


def as_features(feats) -> np.ndarray:
    """Validate and coerce to (T, 225) float32."""
    a = np.asarray(feats, np.float32)
    if a.ndim != 2 or a.shape[1] != FEATURE_DIM:
        raise ValueError("expected (T, %d) features, got %r"
                         % (FEATURE_DIM, a.shape))
    return a


def part(feats: np.ndarray, name: str) -> np.ndarray:
    """(T, 225) -> (T, n, 3) copy of one body part."""
    lo, hi = PART_SLICES[name]
    n = (hi - lo) // CHANNELS
    return feats[:, lo:hi].reshape(feats.shape[0], n, CHANNELS).copy()


def put(feats: np.ndarray, name: str, block: np.ndarray) -> None:
    """Write (T, n, 3) back into the packed row, in place."""
    lo, hi = PART_SLICES[name]
    feats[:, lo:hi] = block.reshape(feats.shape[0], hi - lo)


def normalize_landmarks(feats, min_scale=1e-3):
    """Torso-anchored normalisation: subtract the torso centre, divide by torso
    size. Confidence is untouched, and a miss stays (0, 0, 0).

    Without this the model has to spend capacity learning that the same sign at
    two positions in frame is the same sign, and it never learns it properly
    because the corpus does not cover the frame evenly. Scale is the
    shoulder-midpoint to hip-midpoint distance -- a body property, not a camera
    property -- so it removes camera distance too. Falls back to shoulder width,
    then to 1.0, so a clip with a bad torso does not divide by zero.
    """
    out = as_features(feats)
    body = part(out, "body")
    T = out.shape[0]

    sh_ok = (body[:, L_SHOULDER, 2] > 0) & (body[:, R_SHOULDER, 2] > 0)
    hip_ok = (body[:, L_HIP, 2] > 0) & (body[:, R_HIP, 2] > 0)

    sh_mid = np.zeros((T, 2), np.float32)
    sh_mid[sh_ok] = (body[sh_ok, L_SHOULDER, :2] + body[sh_ok, R_SHOULDER, :2]) / 2
    hip_mid = np.zeros((T, 2), np.float32)
    hip_mid[hip_ok] = (body[hip_ok, L_HIP, :2] + body[hip_ok, R_HIP, :2]) / 2

    centre = np.full((T, 2), 0.5, np.float32)
    both = sh_ok & hip_ok
    centre[both] = (sh_mid[both] + hip_mid[both]) / 2
    centre[sh_ok & ~hip_ok] = sh_mid[sh_ok & ~hip_ok]
    centre[hip_ok & ~sh_ok] = hip_mid[hip_ok & ~sh_ok]

    scale = np.ones(T, np.float32)
    d = np.linalg.norm(sh_mid - hip_mid, axis=1)
    good = both & (d > min_scale)
    scale[good] = d[good]
    sw = np.linalg.norm(body[:, L_SHOULDER, :2] - body[:, R_SHOULDER, :2], axis=1)
    fb = ~good & sh_ok & (sw > min_scale)
    scale[fb] = sw[fb]

    det = out.reshape(T, -1, CHANNELS)[:, :, 2] > 0
    xy = out.reshape(T, -1, CHANNELS)[:, :, :2].copy()
    xy = (xy - centre[:, None, :]) / scale[:, None, None]
    xy = np.where(det[:, :, None], xy, 0.0)
    out = out.reshape(T, -1, CHANNELS).copy()
    out[:, :, :2] = xy
    return out.reshape(T, FEATURE_DIM)


def detection_rate(feats, part="hand") -> float:
    """Fraction of keypoints detected, for the named part."""
    f = as_features(feats).reshape(-1, N_KEYPOINTS, CHANNELS)
    if f.shape[0] == 0:
        return 0.0
    lo, hi = {"hand": (17, 59), "body": (0, 17), "face": (59, 75)}[part]
    return float((f[:, lo:hi, 2] > 0).mean())


def hand_detection_rate(feats) -> float:
    """Fraction of hand keypoints detected, averaged over frames and both hands."""
    return detection_rate(feats, "hand")


def hand_motion_profile(feats, scale_by_torso=True):
    """(T,) per-frame hand motion magnitude, in torso units.

    Mean displacement of detected hand keypoints from the previous frame. This
    is the signal that separates "signer paused to think, hands still up" from
    "signer is signing", which is the documented ghost-text trigger and the
    thing a detection floor cannot see.
    """
    feats = as_features(feats)
    T = feats.shape[0]
    if T < 2:
        return np.zeros(T, np.float32)
    v = feats.reshape(T, -1, CHANNELS)
    l0, l1 = LHAND[0] // CHANNELS, LHAND[1] // CHANNELS
    r0, r1 = RHAND[0] // CHANNELS, RHAND[1] // CHANNELS
    hands = np.concatenate([v[:, l0:l1, :], v[:, r0:r1, :]], axis=1)
    det = hands[:, :, 2] > 0
    xy = hands[:, :, :2]
    d = np.linalg.norm(xy[1:] - xy[:-1], axis=-1)
    m = det[1:] & det[:-1]
    cnt = m.sum(-1)
    energy = np.where(cnt > 0, (d * m).sum(-1) / np.maximum(cnt, 1), 0.0)
    energy = np.concatenate([[0.0], energy]).astype(np.float32)
    if scale_by_torso:
        body = part(feats, "body")
        sh = (body[:, L_SHOULDER, :2] + body[:, R_SHOULDER, :2]) / 2
        hp = (body[:, L_HIP, :2] + body[:, R_HIP, :2]) / 2
        tl = np.linalg.norm(sh - hp, axis=1)
        tl = np.where(tl > 1e-3, tl, 1.0)
        energy = energy / tl
    return energy


def write_extractor_meta(out_dir, extractor_id, extra=None):
    """Stamp a feature directory with the extractor that produced it."""
    p = pathlib.Path(out_dir) / "extractor_meta.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = {"extractor": extractor_id, "feature_dim": FEATURE_DIM,
            "channels": ["x", "y", "confidence"],
            "layout": {name: [lo, hi] for name, lo, hi in PART_ROWS}}
    if extra:
        blob.update(extra)
    p.write_text(json.dumps(blob, indent=2) + "\n")
    return p


def assert_extractor(out_dir, expected_id):
    """Refuse to append to a feature dir written by a different extractor.

    Both backends emit (T, 225) float32 and will concatenate without a warning
    while meaning different things in the face rows.
    """
    p = pathlib.Path(out_dir) / "extractor_meta.json"
    if not p.exists():
        return True
    got = json.loads(p.read_text()).get("extractor")
    if got != expected_id:
        raise RuntimeError(
            "%s holds %r features but this run is %r. Both are (T, %d) float32 "
            "and will train together while meaning different things in every "
            "face row. Re-extract or point the output somewhere new."
            % (out_dir, got, expected_id, FEATURE_DIM))
    return True


__all__ = ["FEATURE_DIM", "CHANNELS", "N_KEYPOINTS", "N_BODY", "N_HAND",
           "N_FACE", "PART_ROWS", "PART_SLICES", "BODY", "LHAND", "RHAND",
           "FACE", "L_SHOULDER", "R_SHOULDER", "L_HIP", "R_HIP",
           "EXTRACTOR_MPH", "EXTRACTOR_RTMPOSE", "DEFAULT_EXTRACTOR",
           "as_features", "part", "put", "normalize_landmarks",
           "hand_detection_rate", "hand_motion_profile",
           "write_extractor_meta", "assert_extractor"]
