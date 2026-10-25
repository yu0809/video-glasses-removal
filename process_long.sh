#!/usr/bin/env bash
# Glasses removal for long videos: processes SEG-second sections one after another (so long shots do not exhaust memory), then concatenates them back into the full video (original audio kept).
# After an interruption, re-run the same command; finished sections are skipped.
#
#   bash process_long.sh input/long_video.mp4                   # 60-second sections
#   SEG=30 bash process_long.sh input/long_video.mp4            # 30-second sections
#   bash process_long.sh input/long_video.mp4 --seg_margin_k 1.0   # further arguments are passed to remove_glasses.py unchanged
#
# Results: output/<name>_no_glasses_full.mp4 (full result), output/<name>_parts/ (per-section comparison videos, mask previews, reports)
set -uo pipefail
cd "$(dirname "$0")"
IN="$1"; shift
SEG=${SEG:-60}
PY=.venv/bin/python
NAME=$(basename "${IN%.*}")
PARTS=output/${NAME}_parts
mkdir -p "$PARTS"
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$IN")
N=$(python3 -c "import math; print(math.ceil($DUR / $SEG))")
echo "video $IN: ${DUR%.*} s, $N sections of $SEG s"
LIST="$PARTS/list.txt"
: > "$LIST"
for ((i = 0; i < N; i++)); do
  S=$((i * SEG))
  D=$(printf "%s/part%03d" "$PARTS" "$i")
  OUTV="$D/${NAME}_no_glasses.mp4"
  if [ ! -s "$OUTV" ]; then
    mkdir -p "$D"
    echo "==> section $((i + 1))/$N: from ${S} s, started $(date '+%H:%M')"
    $PY remove_glasses.py -i "$IN" -o "$D" --start "$S" --duration "$SEG" "$@" > "$D/run.log" 2>&1
    if [ ! -s "$OUTV" ]; then
      # no face / no glasses / error in this section: use the original footage (same encoding settings as the processed sections, so concatenation works)
      echo "    section not processed (see the last lines of $D/run.log); keeping the original footage"
      tail -2 "$D/run.log" | sed 's/^/      /'
      ffmpeg -y -loglevel error -ss "$S" -t "$SEG" -i "$IN" -map 0:v -map 0:a? -c:v libx264 -crf 16 -preset medium \
        -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:a aac -b:a 192k "$OUTV"
    fi
  else
    echo "==> section $((i + 1))/$N already done, skipping"
  fi
  echo "file '$(cd "$D" && pwd)/${NAME}_no_glasses.mp4'" >> "$LIST"
done
FINAL=output/${NAME}_no_glasses_full.mp4
ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" -c copy "$FINAL" || \
  ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" -c:v libx264 -crf 16 -pix_fmt yuv420p -c:a aac -b:a 192k "$FINAL"
echo "done: $FINAL"
