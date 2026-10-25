#!/usr/bin/env bash
# Downloads the public training / evaluation datasets into data/raw/ (sizes and licences: docs/DATASETS.md). Safe to re-run; finished downloads are skipped.
#   bash training/download_data.sh            # everything
#   FFHQ_SHARDS=55 bash training/download_data.sh   # all 55 FFHQ shards (default: first 16)
set -uo pipefail
cd "$(dirname "$0")/.."
R=data/raw
FFHQ_SHARDS=${FFHQ_SHARDS:-16}
get() {  # url dst
  if [ -s "$2" ] && [ ! -f "$2.part" ]; then echo "  exists $2"; return 0; fi
  touch "$2.part"
  curl -fL --retry 5 --retry-delay 5 -C - -s -o "$2" "$1" && rm -f "$2.part" && echo "  done $2 ($(du -h "$2" | cut -f1))" || echo "  failed $2"
}
echo "==> FFHQ 512 (HF luethan2025/FFHQ-512x512, first $FFHQ_SHARDS shards)"
for i in $(seq 0 $((FFHQ_SHARDS - 1))); do
  f=$(printf "data-%05d-of-00055.arrow" "$i")
  get "https://huggingface.co/datasets/luethan2025/FFHQ-512x512/resolve/main/train/$f" "$R/ffhq512/$f" &
  (( (i + 1) % 4 == 0 )) && wait
done
wait
echo "==> CelebAMask-HQ (HF liusq/CelebAMask-HQ)"
get https://huggingface.co/datasets/liusq/CelebAMask-HQ/resolve/main/CelebAMask-HQ.zip $R/celebamask_hq/CelebAMask-HQ.zip
echo "==> AFLW2000-3D (official 3DDFA)"
get http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip $R/aflw2000/AFLW2000-3D.zip
echo "==> Face Synthetics 1,000-image sample (Microsoft)"
get https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2021/dataset_1000.zip $R/face_synthetics/dataset_1000.zip
echo "==> 300W-LP (official 3DDFA Google Drive; often fails with \"too many downloads\", retry a few hours later)"
if [ ! -s $R/300w_lp/300W-LP.zip ]; then
  .venv/bin/python -m gdown "https://drive.google.com/file/d/0B7OEHD3T4eCkVGs0TkhUWFN6N1k/view?usp=sharing&resourcekey=0-WT5tO4TOCbNZY6r6z6WmOA" -O $R/300w_lp/300W-LP.zip -q && echo "  done 300W-LP" || echo "  failed 300W-LP"
fi
echo "==> all finished"
du -sh $R/*
