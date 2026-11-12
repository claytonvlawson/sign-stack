#!/usr/bin/env python3
r"""(T, 225) landmark augmentation -- the three transforms Google actually published.

Source: AISLAC Joint Impact Report for SL2T 1.0, section 2.1 ("Augmentation").
DeepMind applies exactly these on their landmark representation and nothing else
is disclosed:

  1. geometric transforms: rotational noise + "keystone" shifts, for robustness
     to camera angle and body orientation
  2. one-handed simulation: mask the non-dominant arm's keypoints
  3. left-right mirroring: hand balance + fairness for left-handed signers

(2) and (3) are the reasons SL2T scores HIGHER on FLEURS-ASL 1-handed (BLEU 32 /
BLEURT 74) than on the 2-handed sd-test (25 / 70), and why it works for the ~10%
of signers who are left-handed. Both cost nothing to implement on 2D landmarks,
which is why they are worth stealing first.

LAYOUT AWARENESS IS THE WHOLE POINT
-----------------------------------
Everything here operates through contract.PART_SLICES, never through
hardcoded offsets, so a layout change cannot silently produce a transform that
scrambles the wrong limb. Running mirror() on a MediaPipe-layout array would
produce plausible garbage; the extractor stamp is what prevents that, not this
module.

THE ONE INVARIANT THIS MODULE DEFENDS
-------------------------------------
contract.pack_frame encodes "keypoint not detected" as (0, 0, 0): both
coordinates zeroed AND confidence zeroed, so an origin detection (0, 0, s>0) is
distinguishable from a miss. Every transform here preserves that invariant:
an undetected keypoint comes out still (0, 0, 0). Naive x -> 1-x mirroring is
the classic way to break it (a miss at 0 becomes a confident-looking 1.0), so
mirror() moves only points whose confidence is nonzero.

Coordinates are normalized [0, 1] against frame size (pack_frame's contract),
so all transforms work in that space. Keystone and rotation can push coordinates
outside [0, 1]; that is correct, not a bug -- it is the same thing a real
viewpoint change does, and the model should see it.

No torch, no cv2, no model: pure numpy, so this is importable and testable
anywhere the feature contract is.
"""

from __future__ import annotations

import numpy as np

from .contract import CHANNELS, FEATURE_DIM, PART_SLICES

# ---- layout slices, derived not hardcoded -----------------------------------
# PART_SLICES order is (body, left_hand, right_hand, face) and is the single
# source of truth. Rows are contiguous; each part occupies n_keypoints*3
# channels in the packed (225,) row.
_OFFSETS: dict[str, tuple[int, int]] = dict(PART_SLICES)

BODY, LHAND, RHAND, FACE = (_OFFSETS["body"], _OFFSETS["left_hand"],
                            _OFFSETS["right_hand"], _OFFSETS["face"])

N_BODY = (BODY[1] - BODY[0]) // CHANNELS          # 17
N_HAND = (LHAND[1] - LHAND[0]) // CHANNELS        # 21
N_FACE = (FACE[1] - FACE[0]) // CHANNELS          # 16

# COCO-WholeBody body order is 0..16. Person-left is image-right, so mirroring
# swaps each left/right pair and flips x. 0 (nose) is on the midline. Body is the
# first packed block, so packed row == COCO id for 0..16.
BODY_ROW_MIRROR = {0: 0, 1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7,
                   9: 10, 10: 9, 11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15}

# The 16-point face subset is
#   [45, 42, 39, 36, 17, 21, 22, 26, 48, 54, 51, 57, 62, 66, 30, 27]
# = 5 mirror pairs plus 6 midline points. Expressed as indices into the packed
# face block so the mapping is explicit rather than positional.
FACE_ROW_MIRROR = {
    0: 3, 3: 0,      # 45 <-> 36  outer eye corners
    1: 2, 2: 1,      # 42 <-> 39  inner eye corners
    4: 7, 7: 4,      # 17 <-> 26  brow inner
    5: 6, 6: 5,      # 21 <-> 22  brow outer
    8: 9, 9: 8,      # 48 <-> 54  mouth corners
    10: 10, 11: 11,  # 51, 57      upper/lower lip centre (midline)
    12: 12, 13: 13,  # 62, 66      inner lip centre (midline)
    14: 14, 15: 15,  # 30, 27      nose tip / bridge (midline)
}

# non-dominant arm for a right-handed signer = the person's LEFT side.
# rows within the body block: 5/6 shoulders, 7/8 elbows, 9/10 wrists.
_ARM_ROWS = {"left": (5, 7, 9), "right": (6, 8, 10)}
_SIDE_NAMES = ("body", "left_hand", "right_hand", "face")


def _as_feats(feats) -> np.ndarray:
    a = np.asarray(feats, np.float32)
    if a.ndim != 2 or a.shape[1] != FEATURE_DIM:
        raise ValueError("expected (T, %d) features, got %r"
                         % (FEATURE_DIM, a.shape))
    return a


def _part(feats: np.ndarray, name: str) -> np.ndarray:
    """(T, 225) -> a (T, n, 3) view of one part. Copies, so it is safe to mutate
    then write back."""
    lo, hi = _OFFSETS[name]
    n = (hi - lo) // CHANNELS
    return feats[:, lo:hi].reshape(feats.shape[0], n, CHANNELS).copy()


def _put(feats: np.ndarray, name: str, block: np.ndarray, rows=None) -> None:
    """Write a (T, n, 3) block back into the packed row.

    rows: optional subset of that block's keypoint rows to write; the rest are
    left as they were, which is what the arm mask needs.
    """
    lo, hi = _OFFSETS[name]
    T = feats.shape[0]
    if rows is None:
        feats[:, lo:hi] = block.reshape(T, -1)
        return
    for k in rows:
        feats[:, lo + k * CHANNELS: lo + (k + 1) * CHANNELS] = block[:, k, :]


def torso_center(feats) -> np.ndarray:
    """(T, 2) per-frame torso centre: shoulder midpoint when both shoulders are
    detected, else hip midpoint, else the frame centre. Vectorized over T."""
    feats = _as_feats(feats)
    body = _part(feats, "body")
    T = body.shape[0]
    out = np.full((T, 2), 0.5, np.float32)
    both_s = (body[:, 5, 2] > 0) & (body[:, 6, 2] > 0)
    both_h = (body[:, 11, 2] > 0) & (body[:, 12, 2] > 0)
    if both_s.any():
        out[both_s] = (body[both_s, 5, :2] + body[both_s, 6, :2]) / 2.0
    only_h = both_h & ~both_s
    if only_h.any():
        out[only_h] = (body[only_h, 11, :2] + body[only_h, 12, :2]) / 2.0
    return out


def _map_xy(feats: np.ndarray, fn) -> np.ndarray:
    """Apply fn(xy (T,n,2), det (T,n)) -> (T,n,2) to every part, writing back only
    where a keypoint is actually detected. Undetected keypoints stay untouched so
    the (0,0,0) miss invariant survives any transform."""
    out = feats.copy()
    for name in _SIDE_NAMES:
        blk = _part(feats, name)
        det = blk[:, :, 2] > 0
        if not det.any():
            continue
        new = fn(blk[:, :, :2].copy(), det)
        blk[:, :, :2] = np.where(det[:, :, None], new, blk[:, :, :2])
        _put(out, name, blk)
    return out


# ---- transform 1: rotation + keystone ---------------------------------------

def rotate(feats, degrees, anchor=None):
    """Rotate detected keypoints about the torso anchor.

    Rotating about the image centre instead of the body is a different, worse
    augmentation: it moves the signer within the frame rather than changing the
    signer's orientation in the camera's view, which is the thing this is meant
    to teach robustness to.
    """
    feats = _as_feats(feats)
    if abs(float(degrees)) < 1e-6:
        return feats.copy()
    cen = torso_center(feats) if anchor is None else np.asarray(anchor, np.float32)
    theta = np.deg2rad(float(degrees))
    c, s = float(np.cos(theta)), float(np.sin(theta))

    def fn(xy, det):
        x = xy[:, :, 0] - cen[:, None, 0]
        y = xy[:, :, 1] - cen[:, None, 1]
        return np.stack([c * x - s * y + cen[:, None, 0],
                         s * x + c * y + cen[:, None, 1]], -1)

    return _map_xy(feats, fn)


def keystone(feats, kx=0.0, ky=0.0, cx=0.5, cy=0.5):
    """Apply a projective ("keystone") warp to detected keypoints.

    A homography with small off-diagonal terms, which is what a viewpoint change
    does to a roughly planar subject: the two sides of the frame scale
    differently. Google names "keystone shifts" and nothing more; this is the
    standard reading. cx/cy is the shear origin in [0,1] frame space, kx/ky the
    shear magnitudes.
    """
    feats = _as_feats(feats)
    if abs(kx) < 1e-9 and abs(ky) < 1e-9:
        return feats.copy()

    def fn(xy, det):
        x = xy[:, :, 0] - float(cx)
        y = xy[:, :, 1] - float(cy)
        w = float(kx) * x + float(ky) * y + 1.0
        w = np.where(np.abs(w) < 1e-6, 1e-6, w)
        return np.stack([x / w + float(cx), y / w + float(cy)], -1)

    return _map_xy(feats, fn)


# ---- transform 2: left-right mirroring --------------------------------------

def mirror_lr(feats):
    """Mirror the signer left-right: swap the hand blocks, swap L/R body and face
    pairs, reflect x within [0,1]. Undetected keypoints stay (0,0,0)."""
    feats = _as_feats(feats)
    out = feats.copy()
    T = feats.shape[0]

    for name, table in (("body", BODY_ROW_MIRROR), ("face", FACE_ROW_MIRROR)):
        blk = _part(feats, name)
        n = blk.shape[1]
        src = np.array([table.get(k, k) for k in range(n)])
        new = blk[:, src, :].copy()
        det = new[:, :, 2] > 0                    # after the swap
        # clear the coordinate of anything that was a miss, so a miss cannot
        # come back as a confident-looking 1.0
        new[:, :, 0] = np.where(det, 1.0 - new[:, :, 0], 0.0)
        new[:, :, 1] = np.where(det, new[:, :, 1], 0.0)
        _put(out, name, new)

    lmn, lmx = LHAND
    rmn, rmx = RHAND
    lb, rb = feats[:, lmn:lmx].copy(), feats[:, rmn:rmx].copy()
    for blk in (lb, rb):
        v = blk.reshape(T, N_HAND, CHANNELS)
        det = v[:, :, 2] > 0
        v[:, :, 0] = np.where(det, 1.0 - v[:, :, 0], 0.0)
    out[:, lmn:lmx] = rb
    out[:, rmn:rmx] = lb
    return out


# ---- transform 3: one-handed simulation -------------------------------------

def mask_nondominant_arm(feats, dominant="right"):
    """Zero the non-dominant arm: that side's shoulder/elbow/wrist AND its whole
    hand block.

    This is DeepMind's published method for the case in their launch video --
    signing with one hand while the other holds the phone. Masking confidence
    too, not just coordinates, keeps it indistinguishable from a genuine
    detection miss, which is exactly what the model will see when the arm is out
    of frame or behind the device.
    """
    dom = str(dominant).lower()
    if dom not in ("left", "right"):
        raise ValueError("dominant must be 'left' or 'right'")
    feats = _as_feats(feats)
    out = feats.copy()
    weak = "left" if dom == "right" else "right"

    body = _part(feats, "body")
    for row in _ARM_ROWS[weak]:
        body[:, row, :] = 0.0
    _put(out, "body", body)

    if weak == "left":
        out[:, LHAND[0]:LHAND[1]] = 0.0
    else:
        out[:, RHAND[0]:RHAND[1]] = 0.0
    return out


# ---- composable front door --------------------------------------------------

class AugmentConfig:
    """DeepMind's three transforms, with their published intent preserved.

    Probabilities default high because the transforms are cheap and the corpus is
    small: the point of the pair (one-handed + mirror) is coverage of signing
    habits the corpus does not contain, not regularisation noise.
    """

    def __init__(self, p_rotate=0.5, max_rotate_deg=12.0,
                 p_keystone=0.4, max_keystone=0.06,
                 p_mirror=0.3, p_one_handed=0.25,
                 dominant_mix=("right", "right", "right", "left")):
        self.p_rotate = p_rotate
        self.max_rotate_deg = max_rotate_deg
        self.p_keystone = p_keystone
        self.max_keystone = max_keystone
        self.p_mirror = p_mirror
        self.p_one_handed = p_one_handed
        # right-handed signers dominate; keep a real minority of left-handed
        self.dominant_mix = tuple(dominant_mix)


def augment(feats, cfg: AugmentConfig | None = None, rng=None):
    """Apply the composed recipe.

    Order is fixed and deliberate: mirror first (it changes which arm is
    dominant), then one-handed masking, then the continuous geometry. Masking
    before mirroring would mask the wrong limb half the time.
    """
    cfg = cfg or AugmentConfig()
    rng = rng or np.random.default_rng()
    out = _as_feats(feats)

    if rng.random() < cfg.p_mirror:
        out = mirror_lr(out)

    if rng.random() < cfg.p_one_handed:
        dom = cfg.dominant_mix[int(rng.integers(len(cfg.dominant_mix)))]
        out = mask_nondominant_arm(out, dominant=dom)

    if rng.random() < cfg.p_rotate:
        out = rotate(out, float(rng.uniform(-cfg.max_rotate_deg,
                                            cfg.max_rotate_deg)))

    if rng.random() < cfg.p_keystone:
        out = keystone(out,
                       kx=float(rng.uniform(-cfg.max_keystone, cfg.max_keystone)),
                       ky=float(rng.uniform(-cfg.max_keystone, cfg.max_keystone)))
    return out


__all__ = ["augment", "AugmentConfig", "mirror_lr", "mask_nondominant_arm",
           "rotate", "keystone", "torso_center", "BODY", "LHAND", "RHAND",
           "FACE", "N_BODY", "N_HAND", "N_FACE"]
