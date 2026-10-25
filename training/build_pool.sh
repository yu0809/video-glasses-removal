#!/usr/bin/env bash
# Builds the face source pool: real faces -> pose augmentation (methods C, B) -> statistics. Safe to re-run (prepare_faces skips finished sources).
#   bash training/build_pool.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
POOL=data/faces_pool
W=${WORKERS:-6}
LP_BASES=${LP_BASES:-1500}
LP_PER_BASE=${LP_PER_BASE:-6}
REPROJ_N=${REPROJ_N:-6000}

echo "==> [1/5] real faces"
[ -d input/smoke_faces ] && \
$PY training/prepare_faces.py --src input/smoke_faces --out $POOL --dataset smoke --every 1 --min_pose_change 1.5 --workers $W
$PY training/prepare_faces.py --src data/raw/ffhq512 --out $POOL --dataset ffhq512 --workers $W --max_per_source 5000
$PY training/prepare_faces.py --src data/raw/celebamask_hq/CelebAMask-HQ/CelebA-HQ-img --out $POOL --dataset celebamask_hq --workers $W --max_per_source 30000
$PY training/prepare_faces.py --src data/raw/aflw2000/AFLW2000 --out $POOL --dataset aflw2000 --workers $W --min_face 60
[ -d data/raw/300w_lp/300W_LP ] && $PY training/prepare_faces.py --src data/raw/300w_lp/300W_LP --out $POOL --dataset 300w_lp --workers $W --min_face 60 --max_per_source 70000

echo "==> [2/5] method C: LivePortrait re-posing (${LP_BASES} base images x ${LP_PER_BASE})"
if [ ! -f data/repose_lp_big/.done ]; then
  $PY training/repose/liveportrait_repose.py batch --pool $POOL --out data/repose_lp_big --n_bases $LP_BASES --per_base $LP_PER_BASE --seed 1
  touch data/repose_lp_big/.done
fi
echo "==> [3/5] checking method C results"
$PY training/prepare_faces.py --src data/repose_lp_big --out $POOL --dataset repose_lp --min_face 60 --workers $W --max_per_source 100000
echo "==> [4/5] method B: mesh re-projection, ${REPROJ_N} images"
if [ ! -f $POOL/.reproject_done ]; then
  $PY training/repose/mesh_reproject.py batch --pool $POOL --n $REPROJ_N --seed 1
  touch $POOL/.reproject_done
fi
echo "==> [5/5] targeted filling of rare pose cells (targeted method C + method B on top of C results = combined poses)"
if [ ! -f data/repose_lp_sparse/.done ]; then
  $PY training/repose/liveportrait_repose.py batch --pool $POOL --out data/repose_lp_sparse --n_bases 600 --per_base 5 --sparse_below 150 --seed 7
  touch data/repose_lp_sparse/.done
fi
$PY training/prepare_faces.py --src data/repose_lp_sparse --out $POOL --dataset repose_lp --min_face 60 --workers $W --max_per_source 100000
if [ ! -f $POOL/.reproject_sparse_done ]; then
  $PY training/repose/mesh_reproject.py batch --pool $POOL --n 6000 --max_delta 25 --sparse_below 150 --seed 3
  touch $POOL/.reproject_sparse_done
fi
echo "==> statistics"
$PY training/pool_stats.py $POOL --check 36
echo BUILD_POOL_DONE
