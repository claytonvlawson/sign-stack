#!/usr/bin/env python3
"""Pose extraction: video frames -> the (T, 225) landmark contract.

The extractor is MediaPipe Holistic. It is the same perception stack the SL2T
report names for the 130-landmark client-side extraction, it needs no model
download, and on the Jetson it measures ~87 ms per real video frame at 480x480
(content, not resolution, sets that cost).

The corpus in this repo was built from precomputed MediaPipe Holistic landmarks
shipped with the Google ISLR dataset, not from re-running the video through this
module. Both paths land on the same contract; `tests/test_contract.py` pins the
row mapping so a future extractor change cannot silently reorder it.

The mirror contract matters: `augment.mirror_lr` assumes the 16 face rows are
5 mirror pairs plus 6 midline points, and MPH_FACE_468 below is chosen to
satisfy exactly that, using documented FaceMesh topology.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .contract import (CHANNELS, EXTRACTOR_MPH, EXTRACTOR_RTMPOSE, FEATURE_DIM,
                       write_extractor_meta)

MPH_ID = EXTRACTOR_MPH
RTMPOSE_ID = EXTRACTOR_RTMPOSE
EXTRACTOR_ID = MPH_ID

# ---- MediaPipe Holistic -> the 225 contract ---------------------------------

# MPH pose is 33 points; take the COCO-17 subset. COCO row order is fixed by
# the contract: 0 nose ... 11/12 hips, 15/16 ankles.
_MPH_BODY = {0: 0,    # nose
             1: 2, 2: 5,      # l_eye, r_eye
             3: 7, 4: 8,      # l_ear, r_ear
             5: 11, 6: 12,    # l_shoulder, r_shoulder
             7: 13, 8: 14,    # l_elbow, r_elbow
             9: 15, 10: 16,   # l_wrist, r_wrist
             11: 23, 12: 24,  # l_hip, r_hip
             13: 25, 14: 26,  # l_knee, r_knee
             15: 27, 16: 28}  # l_ankle, r_ankle

# 468-mesh indices, in the mirror-pair order the contract expects.
#   0/3   right/left eye outer corner        33 / 263
#   1/2   right/left eye inner corner       133 / 362
#   4/7   right/left brow inner              70 / 336
#   5/6   right/left brow outer             107 / 300
#   8/9   right/left mouth corner            61 / 291
#   10    upper lip top, midline              0
#   11    lower lip bottom, midline          17
#   12    inner upper lip, midline           13
#   13    inner lower lip, midline           14
#   14    nose tip, midline                   1
#   15    nose bridge, midline              168
MPH_FACE_468 = [33, 133, 362, 263, 70, 107, 300, 336, 61, 291, 0, 17, 13, 14, 1, 168]

assert len(_MPH_BODY) == 17 and len(MPH_FACE_468) == 16


def mph_frame_row(pose, left_hand, right_hand, face, swap_hands=False):
    """One MediaPipe Holistic result -> (225,) float32.

    pose/left_hand/right_hand/face: sequences of NormalizedLandmark (anything
    with .x/.y and optionally .visibility), or None when that group was not
    detected. A missing group, missing point, or visibility 0 all become
    (0, 0, 0): the contract's "not detected", which must stay distinguishable
    from a detection at the origin.
    """
    out = np.zeros((75, CHANNELS), np.float32)

    def vis(lm):
        v = getattr(lm, "visibility", None)
        if v is None:
            return 1.0
        v = float(v)
        return v if np.isfinite(v) and v > 0 else 0.0

    if pose is not None:
        for row, src in _MPH_BODY.items():
            if src >= len(pose):
                continue
            lm = pose[src]
            s = vis(lm)
            if s <= 0:
                continue
            out[row] = (float(lm.x), float(lm.y), s)

    # MP Holistic labels chirality assuming a SELFIE (mirrored) view, so a
    # caller feeding a non-mirrored frame should set swap_hands.
    lh, rh = (right_hand, left_hand) if swap_hands else (left_hand, right_hand)
    for base, hand in ((17, lh), (38, rh)):
        if hand is None:
            continue
        for k in range(min(21, len(hand))):
            lm = hand[k]
            out[base + k] = (float(lm.x), float(lm.y), 1.0)

    if face is not None:
        for k, src in enumerate(MPH_FACE_468):
            if src >= len(face):
                continue
            lm = face[src]
            out[59 + k] = (float(lm.x), float(lm.y), 1.0)

    return out.reshape(-1)


class MediaPipeHolisticExtractor:
    """Stateful wrapper around mp.solutions.holistic.Holistic.

    Lazy import so this module stays importable (and the mapping unit testable)
    on a box with no mediapipe.
    """

    EXTRACTOR_ID = MPH_ID

    def __init__(self, static_image_mode=False, model_complexity=1,
                 swap_hands=False, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            raise RuntimeError("this mediapipe build has no mp.solutions; %s "
                               "is not available" % MPH_ID)
        from mediapipe.python.solutions import holistic as mp_holistic
        self._mp_holistic = mp_holistic
        self.holistic = mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self.swap_hands = swap_hands

    def frame_row(self, frame_bgr):
        """(H, W, 3) BGR -> (225,). MPH wants RGB; conversion happens here."""
        rgb = frame_bgr[:, :, ::-1].copy()
        res = self.holistic.process(rgb)
        return mph_frame_row(res.pose_landmarks.landmark if res.pose_landmarks else None,
                             res.left_hand_landmarks.landmark if res.left_hand_landmarks else None,
                             res.right_hand_landmarks.landmark if res.right_hand_landmarks else None,
                             res.face_landmarks.landmark if res.face_landmarks else None,
                             swap_hands=self.swap_hands)

    def close(self):
        self.holistic.close()


def make_extractor(name="mediapipe", **kw):
    if str(name).lower() != "mediapipe":
        raise ValueError("this repo ships the mediapipe extractor; %r is not "
                         "part of the published pipeline" % name)
    return MediaPipeHolisticExtractor(**kw)


def extract_video(source, out_rate=0.0, max_frames=0, stride=1, swap_hands=False,
                  progress=False):
    """Video (path or camera index) -> (T, 225) float32.

    out_rate resamples by dropping frames to an approximate target fps; 0 keeps
    every processed frame. stride skips input frames before extraction, which is
    the cheap way to cut perception cost on the Orin.
    """
    import cv2

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit("cannot open video source %r" % (source,))
    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    keep = 1
    if out_rate and fps_in > 0:
        keep = max(1, int(round(fps_in / float(out_rate))))

    ext = make_extractor(swap_hands=swap_hands)
    rows = []
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n = i
            i += 1
            if stride > 1 and n % stride:
                continue
            if keep > 1 and (n // stride) % keep:
                continue
            rows.append(ext.frame_row(frame))
            if progress and len(rows) % 100 == 0:
                print("  %d frames" % len(rows), file=sys.stderr)
            if max_frames and len(rows) >= max_frames:
                break
    finally:
        ext.close()
        cap.release()
    if not rows:
        raise SystemExit("no frames extracted from %r" % (source,))
    return np.stack(rows).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description="video -> (T, 225) landmarks")
    ap.add_argument("--video", required=True,
                    help="path to a video file, or a camera index like 0")
    ap.add_argument("--out", required=True, help="output .npy path")
    ap.add_argument("--out-rate", type=float, default=0.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--swap-hands", action="store_true")
    a = ap.parse_args(argv)

    src = int(a.video) if str(a.video).isdigit() else a.video
    feats = extract_video(src, out_rate=a.out_rate, max_frames=a.max_frames,
                          stride=a.stride, swap_hands=a.swap_hands,
                          progress=True)
    np.save(a.out, feats)
    from .contract import hand_detection_rate
    print(json.dumps({"out": a.out, "shape": list(feats.shape),
                      "hand_detection_rate": round(hand_detection_rate(feats), 4),
                      "extractor": EXTRACTOR_ID}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
