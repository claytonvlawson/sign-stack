#!/usr/bin/env python3
"""SL2T-Direct: a gloss-free landmark -> text sequence model.

    input   (T, 225)  see contract.py
    model   temporal-conv stem (4x frame reduction) -> transformer encoder
            -> transformer decoder -> word tokens
    output  a word sequence, decoded greedily

The architecture is the one Google describes for SL2T 1.0 mapped onto the
DeafWork landmark contract: "a transformer-based network that conditions on a
sequence representing a signer's movements and generates a text translation
autoregressively". Nothing about SL2T's encoder internals is public, so no
parameter count or layer arrangement here should be read as theirs.

Three mechanisms that are not the model but matter as much:

1. normalize_landmarks -- torso-anchored, so the model learns the sign and not
   the signer's position in frame.
2. SigningGate -- refuse input that is not signing (no hands, or hands
   stationary), the "preventing hallucination on non-signing inputs" item.
3. EndPointer -- when a streamed clip is complete: the 60 s cap, hands leaving
   frame, and a stationary-hands flush.

Greedy decoding here is the reference path: O(L^2) in decode steps, no KV
cache. Fine for offline evaluation; a streaming decoder can be verified
against it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .contract import (CHANNELS, FEATURE_DIM, LHAND, RHAND,
                       as_features as _as, part as _part,
                       normalize_landmarks, hand_detection_rate,
                       hand_motion_profile)

# ---- target tokenizer -------------------------------------------------------

class WordTokenizer:
    """Minimal word-level English tokenizer. Dependency-free on purpose: the DGX
    box has no tokenizers/sentencepiece and adding one to the service venv to
    cover a beta target vocabulary is not worth the import surface.

    <pad>=0 <sos>=1 <eos>=2 <unk>=3. Swap for a subword tokenizer when the target
    vocabulary outgrows this, which it will; word-level is the thing that caps
    utterance length here.
    """

    PAD, SOS, EOS, UNK = 0, 1, 2, 3
    SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]

    def __init__(self, itos=None):
        self.itos = list(itos) if itos else list(self.SPECIALS)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @staticmethod
    def words(text):
        return str(text).replace("\n", " ").split()

    def build(self, texts, min_count=1):
        from collections import Counter
        c = Counter()
        for t in texts:
            c.update(self.words(t))
        for w, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
            if n >= min_count and w not in self.stoi:
                self.stoi[w] = len(self.itos)
                self.itos.append(w)
        return self

    def encode(self, text):
        return [self.stoi.get(w, self.UNK) for w in self.words(text)]

    def decode(self, ids):
        out = []
        for i in ids:
            i = int(i)
            if i in (self.PAD, self.SOS):
                continue
            if i == self.EOS:
                break
            out.append(self.itos[i] if 0 <= i < len(self.itos) else "<unk>")
        return " ".join(out)

    def save(self, path):
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def __len__(self):
        return len(self.itos)


# ---- model ------------------------------------------------------------------



class SigningGate:
    """Refuse to translate input that does not look like signing.

    Two independent checks, because they catch different failure modes:
      detection -- hands are not in frame at all (out of shot, camera on a wall)
      motion    -- hands are detected but not moving (signer paused, or a second
                   person walked in and is being tracked instead)

    Google names both: "preventing hallucination on non-signing inputs" in the
    launch post, and "keeping hands stationary in the frame after completing a
    phrase occasionally triggered erroneous text generation" in the impact
    report. The beta's flat 0.35 detection floor only covers the first.
    """

    def __init__(self, min_hand_detection=0.35, min_motion=0.004,
                 min_active_frames=3):
        self.min_hand_detection = float(min_hand_detection)
        self.min_motion = float(min_motion)
        self.min_active_frames = int(min_active_frames)

    def check(self, feats):
        """-> (ok: bool, reason: str | None, detail: dict)"""
        feats = _as(feats)
        if feats.shape[0] == 0:
            return False, "empty-clip", {}
        rate = hand_detection_rate(feats)
        energy = hand_motion_profile(feats)
        active = int((energy > self.min_motion).sum())
        detail = {"hand_detection_rate": round(rate, 4),
                  "active_frames": active,
                  "median_motion": round(float(np.median(energy)), 6),
                  "min_motion": self.min_motion}
        if rate < self.min_hand_detection:
            return False, "hands-not-detected", detail
        if active < self.min_active_frames:
            return False, "hands-stationary", detail
        return True, None, detail


# ---- end-pointing -----------------------------------------------------------

class EndPointer:
    """Decides when a streaming clip is complete.

    Implements the rules the impact report states or implies:
      - clips are capped at 60 seconds (max_clip_seconds)
      - a clip also ends when the hands leave the frame
      - a stationary run flushes the clip early (otherwise a pause either grows
        latency to the cap or lets the decoder keep predicting into the silence)

    max_segment_seconds is NOT one of Google's rules and is off by default. It
    exists because those three rules are about *clipping*, not about phrase
    boundaries: SL2T streams text continuously, so it never needs to decide where
    one utterance ends. A segment-then-translate loop does, and with only the three
    rules above a continuously-signing clip produces NO output at all until the
    60 s cap, which is useless live. Measured, not assumed: a synthetic clip with
    continuous motion closed zero segments in 120 frames. So a live caller sets a
    sub-segment budget (a few seconds) and gets a translation per span.

    Fed one frame at a time. push(row, t) returns FLUSH on the last frame of a
    segment; call reset() before the next segment.
    """

    CONTINUE, FLUSH = "continue", "flush"

    def __init__(self, max_clip_seconds=60.0, max_gap_frames=8,
                 stationary_frames=20, motion_floor=0.004,
                 max_segment_seconds=None):
        self.max_clip_seconds = float(max_clip_seconds)
        self.max_gap_frames = int(max_gap_frames)
        self.stationary_frames = int(stationary_frames)
        self.motion_floor = float(motion_floor)
        self.max_segment_seconds = (None if max_segment_seconds in (None, 0)
                                    else float(max_segment_seconds))
        self.reset()

    def reset(self):
        self._t_start = None
        self._prev = None
        self._prev_det = None
        self._gap = 0
        self._still = 0
        self._frames = 0
        self.reason = None

    def push(self, row, t):
        """row: (225,) one frame. t: seconds, monotonic. -> 'continue' | 'flush'"""
        row = _as(np.asarray(row, np.float32).reshape(1, -1))[0]
        if self._t_start is None:
            self._t_start = float(t)
        self._frames += 1
        v = row.reshape(-1, CHANNELS)
        hands = np.concatenate([v[LHAND[0] // 3:LHAND[1] // 3, :],
                                v[RHAND[0] // 3:RHAND[1] // 3, :]], axis=0)
        det = hands[:, 2] > 0
        energy = 0.0
        if self._prev is not None:
            m = det & self._prev_det
            if m.any():
                energy = float(np.linalg.norm(hands[m, :2] - self._prev[m, :2],
                                              axis=-1).mean())
        self._prev, self._prev_det = hands[:, :2].copy(), det.copy()

        if float(t) - self._t_start >= self.max_clip_seconds:
            self.reason = "max-duration"
            return self.FLUSH

        if (self.max_segment_seconds is not None
                and float(t) - self._t_start >= self.max_segment_seconds):
            self.reason = "segment-duration"
            return self.FLUSH

        if not det.any():
            self._gap += 1
            if self._gap >= self.max_gap_frames:
                self.reason = "hands-left-frame"
                return self.FLUSH
        else:
            self._gap = 0

        if energy < self.motion_floor:
            self._still += 1
            if self._still >= self.stationary_frames and self._frames > self.stationary_frames:
                self.reason = "stationary"
                return self.FLUSH
        else:
            self._still = 0
        return self.CONTINUE




class SL2TDirectConfig:
    def __init__(self, d_model=512, nhead=8, enc_layers=4, dec_layers=4,
                 ffn=2048, dropout=0.1, stem_convs=2, input_dim=FEATURE_DIM):
        self.d_model = d_model
        self.nhead = nhead
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.ffn = ffn
        self.dropout = dropout
        self.stem_convs = stem_convs
        self.input_dim = input_dim

    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @property
    def frame_reduction(self):
        return 2 ** self.stem_convs


class LandmarkEncoder(nn.Module):
    """(B, T, 225) -> memory (B, T/4, d_model).

    The stem is the part Google does not document and the part that matters most
    for latency: per-frame landmark information density is tiny and the temporal
    structure carries almost everything, so the transformer trunk should not be
    paying attention over every raw frame. Two stride-2 depthwise-separable convs
    give a 4x reduction, which is 7.5 Hz from 30 fps, and the padding mask is
    downsampled alongside so padded frames cannot leak into attention.
    """

    def __init__(self, cfg: SL2TDirectConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.in_proj = nn.Linear(cfg.input_dim, d)
        stem = []
        for _ in range(cfg.stem_convs):
            stem.append(nn.Conv1d(d, d, 5, stride=2, padding=2, groups=d))  # depthwise
            stem.append(nn.Conv1d(d, d, 1))                                # pointwise
            stem.append(nn.GELU())
        self.stem = nn.Sequential(*stem)
        self.norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, cfg.nhead, cfg.ffn, cfg.dropout,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, cfg.enc_layers,
                                             enable_nested_tensor=False)

    @staticmethod
    def downsample_mask(mask, n_convs):
        """(B, T) bool padding mask -> (B, ceil(T / 2**n)). Mask convention is
        True = ignore, matching train_slr.py. Uses the same stride slicing the
        conv uses, so lengths agree for odd T."""
        for _ in range(n_convs):
            mask = mask[:, ::2]
        return mask

    def forward(self, src, src_mask=None):
        x = self.in_proj(src)
        x = self.stem(x.transpose(1, 2)).transpose(1, 2)   # (B, T', d)
        x = self.norm(x)
        m = None if src_mask is None else self.downsample_mask(src_mask,
                                                                self.cfg.stem_convs)
        return self.encoder(x, src_key_padding_mask=m), m


class SL2TDirect(nn.Module):
    """Gloss-free ASL -> English seq2seq over (T, 225) landmarks.

    forward(src, tgt_in, src_mask, tgt_mask) -> (B, L, vocab)
    translate(feats) -> text, greedy, no gradients.
    """

    def __init__(self, cfg: SL2TDirectConfig, vocab_size: int,
                 max_target_len: int = 96):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = int(vocab_size)
        self.max_target_len = int(max_target_len)
        d = cfg.d_model
        self.encoder = LandmarkEncoder(cfg)
        self.tgt_embed = nn.Embedding(self.vocab_size, d)
        self.pos = nn.Embedding(self.max_target_len + 2, d)
        self.drop = nn.Dropout(cfg.dropout)
        layer = nn.TransformerDecoderLayer(d, cfg.nhead, cfg.ffn, cfg.dropout,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, cfg.dec_layers)
        self.out = nn.Linear(d, self.vocab_size)
        # target-side padding is only relevant when the batch is padded; the
        # workspace here is always a full (L, L) causal block.
        # True = forbidden, the same convention as *_key_padding_mask. A float
        # -inf mask instead would make the decoder layer's two masks different
        # dtypes, which torch now warns about and will eventually reject.
        self.register_buffer("_causal", self._causal_mask(256), persistent=False)

    @staticmethod
    def _causal_mask(n):
        return torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)

    def _tgt_mask(self, L, device):
        if L > self._causal.shape[0]:
            self.register_buffer("_causal", self._causal_mask(max(L, 512)),
                                 persistent=False)
        return self._causal[:L, :L].to(device)

    def encode(self, src, src_mask=None):
        """Cast the input to the model's own dtype first.

        An fp16 model fed a float32 tensor fails deep inside F.linear with
        "mat1 and mat2 must have the same dtype, but got Float and Half", which
        points at the encoder rather than at the caller who passed the wrong
        tensor. Every public entry point goes through here, so the cast belongs
        here and not at each call site.
        """
        want = self.encoder.in_proj.weight.dtype
        if src.dtype != want:
            src = src.to(want)
        return self.encoder(src, src_mask)

    def decode(self, memory, mem_mask, tgt_in):
        B, L = tgt_in.shape
        pos = torch.arange(L, device=tgt_in.device).unsqueeze(0)
        x = self.drop(self.tgt_embed(tgt_in) + self.pos(pos))
        h = self.decoder(x, memory, tgt_mask=self._tgt_mask(L, tgt_in.device),
                         tgt_key_padding_mask=(tgt_in == WordTokenizer.PAD),
                         memory_key_padding_mask=mem_mask)
        return self.out(h)

    def forward(self, src, tgt_in, src_mask=None):
        memory, mem_mask = self.encode(src, src_mask)
        return self.decode(memory, mem_mask, tgt_in)

    @torch.no_grad()
    def greedy_decode(self, src, src_mask=None, tokenizer=None, max_len=None):
        """Autoregressive, one token at a time.

        No KV cache: this is O(L^2) in decode steps instead of O(L). It is the
        REFERENCE path -- sl2t_stream.StreamingDecoder is verified against it, so
        keep it even though it is the slow one.

        <pad> is masked out of the output distribution. Padding is not a
        linguistic token, so it must never be generated; an undertrained model will
        happily argmax to it, and a caller downstream would then receive a token id
        with no text.
        """
        self.eval()
        tokenizer = tokenizer or WordTokenizer()
        max_len = max_len or self.max_target_len
        memory, mem_mask = self.encode(src, src_mask)
        B = src.shape[0]
        ids = torch.full((B, 1), tokenizer.SOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits = self.decode(memory, mem_mask, ids)[:, -1, :].clone()
            logits[:, tokenizer.PAD] = float("-inf")
            nxt = logits.argmax(-1)
            ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
            done |= nxt == tokenizer.EOS
            if bool(done.all()):
                break
        return [row[1:].tolist() for row in ids]

    @torch.no_grad()
    def translate(self, feats, tokenizer=None, normalize=True, max_len=None):
        """(T, 225) -> English string. Normalization is on by default and must
        match training; a mismatch here is a silent quality collapse, not a crash,
        so keep the two call sites (here and the trainer) reading the same flag."""
        feats = _as(feats)
        if normalize:
            feats = normalize_landmarks(feats)
        src = torch.from_numpy(feats.astype(np.float32)).unsqueeze(0)
        ids = self.greedy_decode(src, tokenizer=tokenizer, max_len=max_len)[0]
        return (tokenizer or WordTokenizer()).decode(ids)


# ---- checkpoint I/O ---------------------------------------------------------

def save_direct_checkpoint(path, model, tokenizer, extra=None):
    """Config travels with the weights. A bare state_dict cannot be reloaded
    without guessing the architecture, and guessing is how a service silently
    serves a randomly-initialised model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_config": model.cfg.to_dict(),
        "vocab_size": model.vocab_size,
        "max_target_len": model.max_target_len,
        "tokenizer": tokenizer.itos,
        "state_dict": model.state_dict(),
        "extractor_id": extra.get("extractor_id") if extra else None,
        "normalize": True,
        "framework": "sl2t_direct/1",
    }, path)
    return path


def load_direct_checkpoint(path, map_location="cpu", tokenizer=None):
    """-> (model, tokenizer). Tokenizer comes from the checkpoint unless one is
    passed, so a service cannot pair a checkpoint with the wrong vocabulary."""
    blob = torch.load(str(path), map_location=map_location,
                      weights_only=False)
    cfg = SL2TDirectConfig.from_dict(blob["model_config"])
    model = SL2TDirect(cfg, blob["vocab_size"], blob.get("max_target_len", 96))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    tok = tokenizer or WordTokenizer(blob.get("tokenizer"))
    return model, tok


__all__ = ["SL2TDirect", "SL2TDirectConfig", "WordTokenizer", "SigningGate",
           "EndPointer", "save_direct_checkpoint", "load_direct_checkpoint"]
