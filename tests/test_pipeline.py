import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

import signstack
from signstack import augment as A
from signstack import data as D
from signstack.contract import (CHANNELS, FEATURE_DIM, PART_SLICES, hand_detection_rate,
                                normalize_landmarks, part)


def clip(T=12, seed=0, hands=True):
    rng = np.random.default_rng(seed)
    f = np.zeros((T, FEATURE_DIM), np.float32)
    # torso: shoulders rows 5/6, hips 11/12, fixed geometry
    for row, (x, y) in {0: (0.5, 0.30), 5: (0.45, 0.35), 6: (0.55, 0.35),
                        11: (0.46, 0.65), 12: (0.54, 0.65)}.items():
        f[:, row * 3] = x
        f[:, row * 3 + 1] = y
        f[:, row * 3 + 2] = 0.9
    if hands:
        f[:, 17 * 3:59 * 3:3] = 0.4 + rng.random((T, 42)) * 0.2
        f[:, 17 * 3 + 1:59 * 3:3] = 0.4 + rng.random((T, 42)) * 0.2
        f[:, 17 * 3 + 2:59 * 3:3] = 0.95
    return f


class TestContract(unittest.TestCase):
    def test_layout_is_225(self):
        self.assertEqual(FEATURE_DIM, 225)
        self.assertEqual(PART_SLICES["body"], (0, 51))
        self.assertEqual(PART_SLICES["left_hand"], (51, 114))
        self.assertEqual(PART_SLICES["right_hand"], (114, 177))
        self.assertEqual(PART_SLICES["face"], (177, 225))

    def test_miss_invariant_survives_normalize(self):
        f = clip()
        f[:, 40 * 3:40 * 3 + 3] = 0.0            # a miss inside the left hand
        out = normalize_landmarks(f)
        self.assertEqual(out.shape, f.shape)
        np.testing.assert_allclose(out[:, 40 * 3:40 * 3 + 3], 0.0)
        # confidence channel is untouched
        np.testing.assert_allclose(out[:, 2:225:3], f[:, 2:225:3])

    def test_normalize_is_torso_relative(self):
        f = clip()
        shifted = f.copy()
        # move every detected point by a constant: normalization must cancel it
        det = shifted.reshape(len(f), -1, CHANNELS)[:, :, 2] > 0
        shifted.reshape(len(f), -1, CHANNELS)[:, :, 0] += np.where(det, 0.1, 0.0)
        shifted.reshape(len(f), -1, CHANNELS)[:, :, 1] += np.where(det, 0.07, 0.0)
        a = normalize_landmarks(f)
        b = normalize_landmarks(shifted)
        np.testing.assert_allclose(a, b, atol=1e-5)

    def test_mirror_preserves_misses(self):
        f = clip()
        f[:, 20 * 3:20 * 3 + 3] = 0.0            # miss: left hand row 20
        m = A.mirror_lr(f)
        # the hand blocks swap, so the miss lands at right-hand row 20 (= row 41)
        np.testing.assert_allclose(m[:, 41 * 3:41 * 3 + 3], 0.0)
        # hands swap blocks and reflect x within [0, 1]
        want = f[:, 38 * 3:59 * 3].copy()
        want[:, 0::3] = 1.0 - want[:, 0::3]
        np.testing.assert_allclose(m[:, 17 * 3:38 * 3], want, atol=1e-6)

    def test_hand_rate(self):
        self.assertEqual(hand_detection_rate(clip(hands=False)), 0.0)
        self.assertAlmostEqual(hand_detection_rate(clip()), 1.0)


class TestModel(unittest.TestCase):
    def test_checkpoint_roundtrip_and_decode_agrees(self):
        from signstack.model import (SL2TDirect, SL2TDirectConfig, WordTokenizer,
                                     load_direct_checkpoint, save_direct_checkpoint)
        torch.manual_seed(0)
        tok = WordTokenizer(["<pad>", "<sos>", "<eos>", "<unk>", "hello", "mouse"])
        cfg = SL2TDirectConfig(d_model=32, nhead=2, enc_layers=1, dec_layers=1,
                               ffn=64, dropout=0.0)
        model = SL2TDirect(cfg, len(tok), max_target_len=8)
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "m.pt"
            save_direct_checkpoint(p, model, tok, extra={"extractor_id": "x"})
            m2, t2 = load_direct_checkpoint(p)
        self.assertEqual(t2.itos, tok.itos)
        src = torch.zeros(2, 10, FEATURE_DIM)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        ref = model.greedy_decode(src, src_mask=mask, tokenizer=tok, max_len=4)
        got = m2.greedy_decode(src, src_mask=mask, tokenizer=tok, max_len=4)
        self.assertEqual(ref, got)


class TestPackAndEval(unittest.TestCase):
    def test_pack_eval_and_split_guard(self):
        from signstack.data import pack, read_index, verify
        from signstack.evaluate import evaluate, write_outputs
        from signstack.model import (SL2TDirect, SL2TDirectConfig, WordTokenizer,
                                     save_direct_checkpoint)

        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            feats = d / "features"
            feats.mkdir()
            rows = []
            for i in range(12):
                text = "hello" if i % 2 == 0 else "mouse"
                signer = "s1" if i < 6 else "s2"
                p = feats / ("clip%02d.npy" % i)
                np.save(p, clip(seed=i))
                rows.append({"features": str(p), "text": text, "signer": signer})
            manifest = d / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))

            tok = WordTokenizer(["<pad>", "<sos>", "<eos>", "<unk>", "hello", "mouse"])
            cfg = SL2TDirectConfig(d_model=32, nhead=2, enc_layers=1, dec_layers=1,
                                   ffn=64, dropout=0.0)
            model = SL2TDirect(cfg, len(tok), max_target_len=8)
            ckpt = d / "m.pt"
            save_direct_checkpoint(ckpt, model, tok)

            pack(manifest, d / "pack", shard_mb=1, verbose=False)
            self.assertEqual(len(read_index(d / "pack")), 12)
            self.assertTrue(verify(d / "pack", manifest, n=6))

            class Args:
                pass
            a = Args()
            a.pack = str(d / "pack")
            a.manifest = None
            a.ckpt = str(ckpt)
            a.split = ""
            a.batch = 4
            a.workers = 0
            a.max_len = 4
            a.limit = 0
            a.show = 0
            a.device = "cpu"
            a.no_verify_hashes = False
            report, conf, classes = evaluate(a)
            self.assertEqual(report["clips"], 12)
            self.assertEqual(report["classes"], 2)
            self.assertEqual(conf.shape, (2, 2))
            self.assertEqual(int(conf.sum()), 12)
            self.assertIn("top1", report["metrics"])
            write_outputs(report, conf, classes, d / "report")
            self.assertTrue((d / "report" / "per_sign.csv").exists())

            split = {"held_out_signers": ["s2"], "train_signers": ["s1"]}
            (d / "split.json").write_text(json.dumps(split))
            a.split = str(d / "split.json")
            with self.assertRaises(SystemExit):
                evaluate(a)


class TestDisplayPath(unittest.TestCase):
    def test_repo_relative(self):
        repo = pathlib.Path(__file__).resolve().parent.parent
        self.assertEqual(D.display_path(repo / "artifacts/baseline_v1.pt"),
                         "artifacts/baseline_v1.pt")

    def test_home_relative(self):
        self.assertEqual(D.display_path("~/data/x/val.jsonl"), "~/data/x/val.jsonl")

    def test_outside_repo_and_home_falls_back_to_basename(self):
        self.assertEqual(D.display_path("/etc/hosts"), "hosts")


class TestCommittedReportsAreAnonymised(unittest.TestCase):
    """Reports are committed; they must not leak the machine that produced them."""

    def test_no_absolute_home_paths(self):
        repo = pathlib.Path(__file__).resolve().parent.parent
        offenders = []
        for path in list((repo / "reports").rglob("*.json")) + \
                list((repo / "reports").rglob("*.md")) + \
                list((repo / "reports").rglob("*.csv")):
            text = path.read_text()
            for needle in ("/home/", "/Users/", "/root/"):
                if needle in text:
                    offenders.append("%s contains %s" % (path.relative_to(repo), needle))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
