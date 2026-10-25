#!/usr/bin/env bash
# One-step setup (macOS / Linux). Run inside this folder:  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

PROPAINTER_COMMIT=e870e79321c31b733e2031af5aa2fb1fe3ac7eec
WEIGHTS_URL=https://github.com/sczhou/ProPainter/releases/download/v0.1.0
FRAME_SEG_URL=https://github.com/yu0809/video-glasses-removal/releases/download/v0.1.0/frame_seg.pt

command -v ffmpeg >/dev/null || { echo "ffmpeg is missing. On a Mac: brew install ffmpeg"; exit 1; }
command -v git >/dev/null || { echo "git is missing. On a Mac: xcode-select --install"; exit 1; }

PY=${PYTHON:-python3}
"$PY" - <<'EOF' || { echo "Python 3.10 or newer is required. On a Mac: brew install python@3.12, then PYTHON=python3.12 bash setup.sh"; exit 1; }
import sys; assert sys.version_info >= (3, 10)
EOF

echo "==> creating virtual environment .venv"
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
python -m pip install -q --upgrade pip
echo "==> installing Python dependencies (PyTorch is large, this takes a few minutes)"
python -m pip install -q -r requirements.txt

echo "==> fetching ProPainter"
mkdir -p third_party
if [ ! -d third_party/ProPainter/.git ]; then
  git clone -q https://github.com/sczhou/ProPainter third_party/ProPainter
fi
git -C third_party/ProPainter checkout -q "$PROPAINTER_COMMIT"
python patch_propainter.py

echo "==> downloading ProPainter weights"
mkdir -p third_party/ProPainter/weights
for f in ProPainter.pth raft-things.pth recurrent_flow_completion.pth; do
  if [ ! -s "third_party/ProPainter/weights/$f" ]; then
    curl -fL --retry 3 -o "third_party/ProPainter/weights/$f" "$WEIGHTS_URL/$f"
  fi
done

echo "==> downloading face detection / face parsing models"
python -c "from uniface import RetinaFace, BiSeNet; from uniface.constants import ParsingWeights; RetinaFace(); BiSeNet(model_name=ParsingWeights.RESNET34)"

echo "==> downloading frame segmentation weights (CC BY-NC-SA 4.0, non-commercial)"
mkdir -p weights
if [ ! -s weights/frame_seg.pt ]; then
  curl -fL --retry 3 -o weights/frame_seg.pt "$FRAME_SEG_URL" || { rm -f weights/frame_seg.pt; echo "    download failed; the rule-based mask will be used until weights/frame_seg.pt exists"; }
fi

mkdir -p input output
python -c "import torch; d='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'); print('==> setup finished, compute device:', d)"
echo "Next: put a video into input/ and run"
echo "  source .venv/bin/activate"
echo "  python remove_glasses.py -i input/your_video.mp4 --duration 3"
