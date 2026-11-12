#!/usr/bin/env bash
# One command: clone, run, get top-1 and per-sign confusion on a held-out signer.
#
#   ./run.sh              eval on the frozen signer-disjoint pack (downloads ~100 MB)
#   ./run.sh eval --limit 500
#   ./run.sh predict --video my_clip.mp4 --ckpt artifacts/baseline_v1.pt
#   ./run.sh train --manifest data/train.jsonl --val-manifest data/val.jsonl
#
# Set PYTHON to pick an interpreter, or SKIP_INSTALL=1 to refuse venv creation.
set -euo pipefail
cd "$(dirname "$0")"

REPO="claytonvlawson/sign-stack"
TAG="${SIGNSTACK_TAG:-v1.0.1}"
BASE="${SIGNSTACK_RELEASE_BASE:-https://github.com/$REPO/releases/download/$TAG}"

PY="${PYTHON:-python3}"
if ! "$PY" -c "import torch, numpy" >/dev/null 2>&1; then
  if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    echo "python at $PY has no torch/numpy and SKIP_INSTALL=1" >&2
    exit 1
  fi
  if [ ! -x .venv/bin/python ]; then
    echo "creating .venv (torch is a large download)" >&2
    "$PY" -m venv .venv
    .venv/bin/python -m pip install -q --upgrade pip
    .venv/bin/python -m pip install -q -r requirements.txt
  fi
  PY=".venv/bin/python"
fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

fetch() {  # url dest
  if [ -f "$2" ] && [ -s "$2" ]; then return 0; fi
  mkdir -p "$(dirname "$2")"
  echo "downloading $(basename "$2")" >&2
  curl -fL --retry 3 --retry-delay 2 -o "$2.part" "$1"
  mv "$2.part" "$2"
}

cmd="${1:-eval}"
if [ $# -gt 0 ]; then shift; fi

case "$cmd" in
  eval)
    fetch "$BASE/baseline_v1.pt" artifacts/baseline_v1.pt
    if [ ! -f eval/frozen_v1/index.jsonl ]; then
      fetch "$BASE/frozen_v1.tar.gz" dist/frozen_v1.tar.gz
      tar -xzf dist/frozen_v1.tar.gz -C .
    fi
    exec "$PY" -m signstack eval \
      --pack eval/frozen_v1 \
      --ckpt artifacts/baseline_v1.pt \
      --split eval/split_v1.json \
      --out-dir reports/frozen_v1 "$@"
    ;;
  train)
    exec "$PY" -m signstack train "$@"
    ;;
  predict)
    exec "$PY" -m signstack predict "$@"
    ;;
  test)
    exec "$PY" -m unittest discover -s tests -v
    ;;
  *)
    echo "unknown command: $cmd (eval | train | predict | test)" >&2
    exit 2
    ;;
esac
