#!/usr/bin/env bash
# Evaluation chain: per-frame segmentation on the test videos -> end-to-end comparison -> real videos with three methods. Safe to re-run (existing results are skipped)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CKPT=${CKPT:-weights/frame_seg_r1.pt}
OUT=${OUT:-reports}
echo "==> [1/3] per-frame segmentation on the test videos"
$PY training/evaluate.py videos --ckpt $CKPT --out $OUT/videos_ours
$PY training/evaluate.py videos --ckpt glasses-detector:medium --out $OUT/videos_gd
echo "==> [2/3] end to end: rule-based / network / glasses-detector"
$PY training/eval_e2e.py --ckpt $CKPT --out $OUT/e2e_r1 --methods rule net gd --clips 0
echo "==> [3/3] real videos"
for v in ${VIDEOS:-example}; do
  for m in rule net gd; do
    case $m in rule) S=none;; net) S=$CKPT;; gd) S=glasses-detector:medium;; esac
    D=$OUT/real/$m
    [ -f $D/${v}_compare.mp4 ] && continue
    mkdir -p $D
    $PY remove_glasses.py -i input/$v.mp4 -o $D --seg_model $S --duration 6 > $D/$v.log 2>&1 || echo "  failed $v $m"
  done
done
echo STAGE_E_DONE
