#!/usr/bin/env bash
# Extra environment for training (run from the project root: bash training/setup_train.sh). Run setup.sh first.
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate

LP_COMMIT=9b294b3d0536135442ea73cb01e6cb3ca7029dd3
HF=https://huggingface.co/KlingTeam/LivePortrait/resolve/main

echo "==> training dependencies"
python -m pip install -q -r training/requirements-train.txt

echo "==> LivePortrait (pose augmentation method C; code MIT, InsightFace detection models non-commercial only)"
mkdir -p third_party
if [ ! -d third_party/LivePortrait/.git ]; then
  git clone -q https://github.com/KwaiVGI/LivePortrait third_party/LivePortrait
fi
git -C third_party/LivePortrait checkout -q "$LP_COMMIT"
# portrait models only (~660 MB), no animal models
for f in insightface/models/buffalo_l/2d106det.onnx insightface/models/buffalo_l/det_10g.onnx \
         liveportrait/base_models/appearance_feature_extractor.pth liveportrait/base_models/motion_extractor.pth \
         liveportrait/base_models/spade_generator.pth liveportrait/base_models/warping_module.pth \
         liveportrait/landmark.onnx liveportrait/retargeting_models/stitching_retargeting_module.pth; do
  dst=third_party/LivePortrait/pretrained_weights/$f
  mkdir -p "$(dirname "$dst")"
  [ -s "$dst" ] || curl -fL --retry 3 -o "$dst" "$HF/$f"
done
echo "==> done"
