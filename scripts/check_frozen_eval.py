#!/usr/bin/env python3
"""Fail if the frozen-pack evaluation drifts from the published numbers.

Reads reports/frozen_v1/report.json (written by `./run.sh eval`) and compares it
against the values this repo publishes. Used as the CI gate in
.github/workflows/eval.yml, and runnable by hand:

    ./run.sh eval && python3 scripts/check_frozen_eval.py

Override the expectations with EXPECT_TOP1 / EXPECT_TOP5 / TOLERANCE to use it
against a different pack.
"""
import json
import os
import sys

EXPECT_CLIPS = 4729
EXPECT_CLASSES = 250
EXPECT_SIGNERS = 5
EXPECT_TOP1 = float(os.environ.get("EXPECT_TOP1", "0.3722"))
EXPECT_TOP5 = float(os.environ.get("EXPECT_TOP5", "0.6464"))
TOLERANCE = float(os.environ.get("TOLERANCE", "0.0005"))
REPORT = os.environ.get("REPORT", "reports/frozen_v1/report.json")


def main():
    if not os.path.exists(REPORT):
        print("no report at %s, run ./run.sh eval first" % REPORT)
        return 2

    report = json.load(open(REPORT))
    metrics = report.get("metrics", {})

    print("clips    %s" % report.get("clips"))
    print("classes  %s" % report.get("classes"))
    print("checkpoint %s" % report.get("checkpoint"))
    print("top-1    %.4f  (expected %.4f, tolerance %.4f)"
          % (metrics.get("top1", 0.0), EXPECT_TOP1, TOLERANCE))
    print("top-5    %.4f  (expected %.4f, tolerance %.4f)"
          % (metrics.get("top5", 0.0), EXPECT_TOP5, TOLERANCE))

    failures = []
    if report.get("clips") != EXPECT_CLIPS:
        failures.append("clips=%s, expected %d" % (report.get("clips"), EXPECT_CLIPS))
    if report.get("classes") != EXPECT_CLASSES:
        failures.append("classes=%s, expected %d" % (report.get("classes"), EXPECT_CLASSES))
    if len(report.get("per_signer", {})) != EXPECT_SIGNERS:
        failures.append("expected %d held-out signers, got %d"
                        % (EXPECT_SIGNERS, len(report.get("per_signer", {}))))
    if abs(metrics.get("top1", 0.0) - EXPECT_TOP1) > TOLERANCE:
        failures.append("top-1 drifted from %.4f by more than %.4f"
                        % (EXPECT_TOP1, TOLERANCE))
    if abs(metrics.get("top5", 0.0) - EXPECT_TOP5) > TOLERANCE:
        failures.append("top-5 drifted from %.4f by more than %.4f"
                        % (EXPECT_TOP5, TOLERANCE))

    # A committed report must not leak the machine that produced it.
    raw = open(REPORT).read()
    for needle in ("/home/", "/Users/", "/root/"):
        if needle in raw:
            failures.append("report leaks an absolute path (%s)" % needle)

    if failures:
        print("\nFAIL")
        for f in failures:
            print(" - " + f)
        return 1
    print("\nOK: frozen-pack numbers hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
