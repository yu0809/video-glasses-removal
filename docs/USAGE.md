# Usage

Fully automatic: a video goes in, the same video without eyeglasses comes out, with the original audio. Everything runs locally with open-source models; nothing is uploaded.

## Pipeline

| Step | Model | What it does |
|---|---|---|
| 1. Face detection and tracking | RetinaFace (via uniface) | finds faces, splits the video into segments (see below), smooths a stable crop box |
| 2. Frame segmentation (default) | this project's frame segmentation network (`weights/frame_seg.pt`) | per-pixel frame, bridge, nose pads, hinges and temples, robust to large pitch, roll, yaw, tiny faces and motion blur; the eyes behind the lenses are left untouched |
| 3. Mask | this project | neighbouring frames aligned by the glasses' motion and averaged, hysteresis threshold, margin proportional to the frame width; falls back to the rule-based mask (BiSeNet face parsing + line detection) when no network weights are present |
| 4. Video inpainting | ProPainter (ICCV 2023) | fills the masked frame with temporally consistent skin using neighbouring frames and optical flow |
| 5. Residue check | this project | runs the network again on the result; frames that still show glasses get a wider mask and are redone (up to 2 rounds) |
| 6. Paste back | feathered blending | only the masked region is replaced; everything else is the original video |

Works best for **clear lenses with metal or acetate frames**. Not for sunglasses / dark lenses (the eyes are hidden and would have to be generated).

## Install (once)

Requirements: macOS (Apple silicon uses the GPU automatically) or Linux, Homebrew, ffmpeg, Python 3.10+.

```bash
brew install ffmpeg python@3.12
git clone https://github.com/yu0809/video-glasses-removal.git
cd video-glasses-removal
PYTHON=python3.12 bash setup.sh
```

`setup.sh` creates `.venv`, installs the dependencies, clones ProPainter at a fixed commit, applies a small patch and downloads the ProPainter weights (~300 MB), the face models and the frame segmentation weights (98 MB, from this repository's Releases). The last line prints the compute device (`mps` on Apple silicon).

If that last download fails, get `frame_seg.pt` from the [Releases page](https://github.com/yu0809/video-glasses-removal/releases) and put it at `weights/frame_seg.pt`. Without it the rule-based mask is used.

## Run

```bash
source .venv/bin/activate

# 1. masks only (seconds): check that the green area covers the whole frame
python remove_glasses.py -i input/your_video.mp4 --duration 3 --mask_only

# 2. try 3 seconds
python remove_glasses.py -i input/your_video.mp4 --duration 3

# 3. the whole video
python remove_glasses.py -i input/your_video.mp4

# a time range: start at 1:30, process 60 s
python remove_glasses.py -i input/your_video.mp4 --start 90 --duration 60
```

Long videos (more than a few minutes): process them in 60-second pieces so that a long continuous shot does not exhaust memory; the pieces are stitched back with the original audio, and re-running the same command skips finished pieces.

```bash
bash process_long.sh input/your_video.mp4          # SEG=30 bash ... for 30 s pieces
```

Outputs in `output/`:

- `xxx_no_glasses.mp4` — result with the original audio
- `xxx_compare.mp4` — side by side
- `xxx_mask_preview.mp4` — the face crop and the inpainted region in green
- `xxx_report.csv` — per-frame values and residue scores; `xxx_quality.png` — residue score and mask area over time

The terminal also lists time ranges that may still contain residue.

## Frame segmentation network (default)

Used automatically when `weights/frame_seg.pt` exists (the log says "mask: frame segmentation network").

| Option | Default | Meaning |
|---|---|---|
| `--seg_model` | `auto` | `auto`: use `weights/frame_seg.pt` if present, else the rule-based mask; `none`: force the rule-based mask; or a path to other weights (`.pt` / `.onnx`) |
| `--seg_margin_k` | `0.5` | extra mask margin as a fraction of the detected frame width; try `1.0` if close-ups keep a faint outline at the lens edge |
| `--seg_margin` | `2` | base mask margin in pixels |
| `--seg_refine_threshold` | `0.2` | redo a frame when the network's mean frame probability inside the old mask exceeds this after inpainting |
| `--mask_grow` | `0` | widen every mask by N pixels; try `2` if thin residue lines remain |

Compare with the older masks: `--seg_model none` (rule-based adaptive) or `--fixed` (fixed-width ring from face parsing).

## Rule-based adaptive mask (`--seg_model none`)

Measures, frame by frame, where the rim is and how thick it is along the parsed glasses outline (all lengths relative to the glasses width), detects temples with line fitting, aligns neighbouring masks by the glasses' motion, scores residue after inpainting and redoes bad frames. It works on frontal close-ups; large pitch, roll and small faces are where the network mask is much better.

## Videos where faces come and go

The script scans the whole video first and only processes continuous face segments; **other frames are copied unchanged**.

- **Shot cuts** split segments (`--cut_threshold`, default 0.45; lower = more sensitive).
- **Short detection gaps** (up to `--max_gap` 0.3 s) are interpolated.
- Segments **without glasses** are skipped.
- Each segment tracks **one face** (the largest when the segment starts).
- Faces smaller than `--min_face` (48 px) are ignored.

## Other options

| Symptom | Try |
|---|---|
| thin residue lines | `--mask_grow 2`, `--seg_margin_k 1.0` |
| no time for rework | `--refine_iters 0` |
| visible seam around the repaired area | `--feather 5` |
| small faces look blurry after repair | `--crop_scale 1.2` (tighter crop) |
| out of memory ("process killed") | smaller `--chunk` or `--proc_size`, or `process_long.sh` |
| ProPainter fails on the Apple GPU | it retries on CPU automatically; or `--device cpu` |

Memory guide: 8 GB → `--proc_size 384 --chunk 30`; 16 GB → defaults (`--proc_size 512 --chunk 50`); 32 GB+ → `--chunk 80`. Expect roughly 1–2 s per frame on an Apple M-series GPU at 512 px; run `--duration 2` first to estimate.

## Limitations

1. **Lens refraction**: strong prescriptions make the eyes look smaller; the lens interior is kept as is.
2. **Lens reflections** stay.
3. **Nose-pad marks** on the skin stay.
4. **Sunglasses / dark lenses** are out of scope.
5. **Tiny faces** (30–60 px in the source video): masks are unreliable and the inpainter has very few pixels; residue is likely.
6. **Real footage**: very thin frames are occasionally missed in part, leaving a short leftover segment.

## Licence

- ProPainter: NTU S-Lab License 1.0, **non-commercial use only**.
- Frame segmentation weights: **CC BY-NC-SA 4.0**, non-commercial only (trained on non-commercial research data, see [DATASETS.md](DATASETS.md)).
- Code in this repository: MIT.
