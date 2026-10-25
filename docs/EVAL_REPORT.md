# Evaluation report — frame segmentation network, round 1

Weights: `frame_seg.pt` (U-Net with a ResNet-34 encoder, 384 px input, step 18,000 of a 30,000-step run; best synthetic boundary F1).
The figures in the README are generated from these results by `docs/scripts/make_figures.py` and `docs/scripts/make_examples.py`.

## 0. Summary

- **vs. take-off-eyeglasses (CVPR 2022)**: ahead everywhere by a wide margin; it is usable only on clean frontal faces and degrades sharply with roll, tiny faces and blur.
- **vs. glasses-detector (medium)**: mixed at the segmentation level, clearly ahead end to end on the synthetic benchmark.
  - Real CelebA photos, clean near-frontal and roll: glasses-detector is better (F 0.94 vs. 0.90). About half of the gap comes from mask thickness on a coarse annotation; dilated as in the pipeline, this model reaches 0.915–0.917. The remaining outline-recall gap (≈ 0.89 vs. 0.95) is genuine misses.
  - Tiny faces (0.83 vs. 0.67), false positives on faces without glasses (0.11% vs. 2.5% of the image) and frame-accurate labels (Face Synthetics 0.89 vs. 0.75): this model is clearly better.
  - Inside the full pipeline (11 synthetic test scenes): residue 16% vs. 46% (rule-based 63%), PSNR 28.7 vs. 25.9 dB.
    **Caveat: the test clips come from this project's synthesizer, a favourable setting; real-video evidence is limited to qualitative comparisons.**
- **Hard poses on third-party synthetic data (Face Synthetics part 1, 454 clear-lens hard-pose images)**: boundary F1 0.858 vs. 0.774 (take-off-eyeglasses), 0.715 (glasses-detector medium), 0.579 (rule-based); every gap's 95% bootstrap interval excludes zero. On near-frontal renders take-off-eyeglasses is slightly better (0.934 vs. 0.921). See §2.3.
- **Shared weakness**: tiny faces (30–60 px) and dolly shots — 60–100% residue for every method, mostly limited by inpainting resolution.

## 1. Methods

| Name | Source | Licence | Notes |
|---|---|---|---|
| ours | this project | code MIT, weights CC BY-NC-SA 4.0 | `seg_model.FrameSegmenter` |
| rule | the original rule-based mask (BiSeNet + line detection, `adaptive.py`) | – | outputs an inpainting mask with margin, not a precise segmentation |
| gd_small / gd_medium | glasses-detector v1.0.4, `kind="frames"` (temples included), authors' weights | MIT | input resized to 256 |
| tog | take-off-eyeglasses (Lyu et al., CVPR 2022), eyeglasses-mask branch, authors' checkpoint | no licence file — local evaluation only | input 256, [-1, 1] |

All methods receive the same 512 px pipeline crop (face box × 1.3, shifted up by 0.08) and output a frame probability; > 0.5 counts as frame.

## 2. Real photos and third-party synthetic data (`training/eval_real.py`)

### 2.1 CelebAMask-HQ, 1,415 real photos with glasses (unseen people for this model as well)

`eye_g` covers the whole glasses including the lenses and is coarse, so the metrics are precision P (prediction inside `eye_g`) and outline recall R (outline of `eye_g` covered by the prediction); F is their harmonic mean, tolerance 6 px.

| Method | Original F (P/R) | Roll ±30° | Tiny face (~50 px) | Motion blur |
|---|---|---|---|---|
| ours | 0.901 (0.97/0.85) | 0.877 (0.97/0.82) | **0.831** (0.95/0.76) | 0.820 (0.95/0.75) |
| ours dilated 4 px (≈ pipeline mask) | 0.915 (0.95/0.89) | 0.893 (0.94/0.87) | | |
| rule | 0.881 (0.82/0.96) | 0.739 | 0.611 | 0.808 |
| gd_small | 0.734 | 0.730 | 0.494 | 0.612 |
| gd_medium | **0.939** (0.93/0.95) | **0.909** (0.92/0.91) | 0.667 (0.76/0.65) | **0.822** (0.87/0.81) |
| tog | 0.832 (0.96/0.77) | 0.623 | 0.117 | 0.444 |

By estimated head pose (original images): with |yaw| ≥ 30°, ours 0.874 / gd_medium 0.876 / tog 0.692 (n = 75). CelebA is mostly frontal, so it does **not** measure large-pose behaviour; see §4.

False-positive area on 500 photos without glasses: ours 0.11% (tiny-face version 0.41%), rule 0.01%, gd_small 4.3%, gd_medium 2.5%, tog 0.08%.

### 2.2 Microsoft Face Synthetics (boundary F1, 2 px)

With clear lenses the `GLASSES` class is the frame. 59 clear-lens and 65 opaque-lens images — indicative only.

| Method | Original | Roll ±30° | Tiny face | Motion blur | Opaque lenses |
|---|---|---|---|---|---|
| ours | **0.890** | **0.884** | **0.802** | **0.791** | 0.678 |
| gd_medium | 0.747 | 0.726 | 0.511 | 0.618 | **0.722** |
| tog | 0.881 | 0.742 | 0.186 | 0.442 | 0.594 |
| rule | 0.594 | 0.455 | 0.415 | 0.555 | 0.643 |

### 2.3 Microsoft Face Synthetics, part 1, by head pose

The real CelebA photos above are near frontal (HeadPose estimate: 4 of 1,415 beyond ±25° pitch, none beyond ±25° roll), so hard poses are tested on a third-party renderer instead.

- Data: part 1 of the full 100,000-image dataset (25,568 renders), 3,662 with glasses, 1,498 with clear lenses (where `GLASSES` = frame). 6 images with glasses lost to face detection.
- Pose from the dataset's exact landmarks: pitch / yaw by PnP against a 68-point mean face (correlation with HeadPose on the 1,000-image sample 0.85 / 0.94), roll from the pupil line.
- Hard = pitch beyond ±25°, yaw beyond ±45°, roll beyond ±25°, or two or more axes beyond ±25° (454 images).
- Metric: boundary F1, 2 px; a sample with no predicted frame scores 0. Commands in `training/eval_real.py` (`prepare_fsynth`, `run --data fsynth_p1`).

| Clear lenses | n | ours | tog | gd_medium | rule |
|---|---|---|---|---|---|
| Near frontal (all axes < 15°) | 392 | 0.921 | **0.934** | 0.791 | 0.638 |
| Pitch beyond ±25° | 324 | **0.861** | 0.784 | 0.721 | 0.568 |
| Pitch beyond ±40° | 125 | **0.832** | 0.703 | 0.661 | 0.516 |
| Yaw beyond ±45° | 219 | **0.823** | 0.691 | 0.661 | 0.575 |
| Roll beyond ±25° | 91 | **0.780** | 0.582 | 0.572 | 0.488 |
| Two or more axes beyond ±25° | 199 | **0.828** | 0.704 | 0.662 | 0.546 |
| All hard poses | 454 | **0.858** | 0.774 | 0.715 | 0.579 |
| All hard poses, tiny face | 454 | **0.778** | 0.119 | 0.448 | 0.343 |
| All hard poses, motion blur | 454 | **0.761** | 0.331 | 0.565 | 0.489 |
| All clear lenses | 1,498 | **0.897** | 0.873 | 0.760 | 0.614 |

Paired bootstrap (2,000 resamples), gap ours − baseline:

| | Hard poses (n = 454) | Near frontal (n = 392) |
|---|---|---|
| vs. take-off-eyeglasses | +0.084 [+0.071, +0.098], ours higher on 74% | −0.014 [−0.020, −0.007], ours higher on 28% |
| vs. glasses-detector medium | +0.143 [+0.133, +0.154], 95% | +0.130 [+0.122, +0.138], 98% |
| vs. rule-based | +0.279 [+0.266, +0.292], 98% | +0.282 [+0.274, +0.292], 100% |

Opaque lenses (frame and lens labelled together, n = 2,164): ours 0.693, gd_medium **0.736**, tog 0.596, rule 0.643.
Caveats: still synthetic; frame-only labels with 2 px tolerance penalise thick masks (glasses-detector), which is why it stays below 0.8 here while it leads on real CelebA photos.

## 3. Synthetic validation set (in-distribution; used to find weak spots)

1,000 samples, boundary F1 0.822 overall. Pitch / yaw / roll buckets and combined poses (two or three axes beyond 25°) all lie between 0.80 and 0.83.
Face size in the source > 200 px 0.900, 80–200 px 0.799, < 80 px 0.653. Acetate 0.918, full rim 0.836, half rim 0.775, **rimless 0.636**.
Low light 0.694, back light 0.735, motion blur 0.782. False-positive area on negatives 0.41%.

## 4. Synthetic test-video benchmark (11 scenes × 3 clips, `training/make_test_videos.py`)

Base footage: held-out FFHQ identities re-posed by LivePortrait, plus public talking-head samples for scene 01. One fixed pair of glasses per clip, rendered frame by frame; the glasses-free clip is the ground truth.

### 4.1 Per-frame segmentation (boundary F1 on the 512 px crop)

| Scene | ours | tog | gd_medium | rule |
|---|---|---|---|---|
| 01 Frontal talking | **0.937** | 0.847 | 0.632 | 0.527 |
| 02 Look up / down ±40° | **0.975** | 0.879 | 0.705 | 0.576 |
| 03 Roll 30° + pitch 30° | **0.987** | 0.694 | 0.698 | 0.413 |
| 04 Yaw ±60° + pitch 15° | **0.938** | 0.779 | 0.534 | 0.493 |
| 05 All three axes (figure 8) | **0.964** | 0.818 | 0.633 | 0.533 |
| 06 Tiny face 30–60 px | **0.155** | 0.003 | 0.007 | 0.003 |
| 07 Dolly zoom 60 → 400 px | **0.760** | 0.598 | 0.360 | 0.405 |
| 08 Fast head turn (motion blur) | **0.937** | 0.820 | 0.575 | 0.544 |
| 09 Multi-camera cuts + empty shot | **0.815** | 0.504 | 0.449 | 0.492 |
| 10 Lighting (back light / low light / strong reflections) | **0.892** | 0.619 | 0.466 | 0.356 |
| 11 Occlusion (hand passing / hair over temples) | **0.951** | 0.807 | 0.545 | 0.459 |

### 4.2 End to end (clip 0 of each scene, `training/eval_e2e.py`)

Cells: residue rate / PSNR around the glasses / mean mask area. A frame counts as residue when more than 20% of its ground-truth frame pixels still differ from the clean frame (ΔE > 20 in Lab).

| Scene | Rule-based | glasses-detector | Ours, margin k = 0 | **Ours, k = 0.5 (default)** |
|---|---|---|---|---|
| 01 Frontal talking | 0% / 31.4 / 4.0% | 0% / 30.3 / 5.0% | 0% / 32.2 / 2.3% | 0% / 32.1 / 3.7% |
| 02 Look up / down | 98% / 22.0 / 5.4% | 38% / 25.0 / 5.2% | 6% / 26.0 / 1.8% | 4% / 26.3 / 3.0% |
| 03 Roll + pitch | 81% / 22.6 / 4.9% | 87% / 23.4 / 6.1% | 0% / 31.9 / 3.1% | 0% / 30.7 / 5.1% |
| 04 Yaw + pitch | 69% / 24.8 / 4.3% | 34% / 25.3 / 7.5% | 8% / 26.4 / 3.2% | 5% / 26.5 / 4.7% |
| 05 Three axes | 86% / 21.6 / 4.4% | 15% / 23.6 / 8.5% | 3% / 24.8 / 3.6% | 3% / 25.3 / 5.6% |
| 06 Tiny face | 100% / 27.6 / 1.3% | 100% / 27.6 / 0.0% | 88% / 29.2 / 4.1% | 80% / 29.1 / 5.4% |
| 07 Dolly zoom | 79% / 25.6 / 4.6% | 76% / 25.3 / 7.9% | 73% / 26.2 / 3.9% | 60% / 26.2 / 6.1% |
| 08 Fast head turn | 37% / 25.4 / 3.8% | 15% / 24.8 / 6.0% | 0% / 28.0 / 2.6% | 0% / 27.8 / 4.3% |
| 09 Multi-camera | 90% / 26.0 / 4.5% | 84% / 24.0 / 5.1% | 28% / 29.6 / 4.2% | 10% / 29.9 / 7.2% |
| 10 Lighting | 16% / 31.7 / 3.3% | 17% / 30.5 / 4.2% | 0% / 34.0 / 2.1% | 0% / 33.0 / 3.3% |
| 11 Occlusion | 40% / 24.6 / 4.0% | 39% / 25.2 / 4.6% | 15% / 27.7 / 1.2% | 13% / 28.2 / 2.0% |

| Method | PSNR | SSIM | Residue | Flicker | Mask area |
|---|---|---|---|---|---|
| Rule-based | 25.75 | 0.835 | 63% | 6.09 | 4.04% |
| glasses-detector | 25.90 | 0.851 | 46% | 5.38 | 5.46% |
| Ours k = 0 | 28.72 | 0.889 | 20% | 4.73 | 2.92% |
| **Ours k = 0.5 (default)** | 28.66 | 0.890 | **16%** | **4.53** | 4.58% |
| Ours k = 1.0 | 27.70 | 0.873 | 19% | 4.77 | 6.36% |

## 5. Real video (qualitative)

- Close-up: the rule-based mask leaves nose-pad / bridge remnants; this model and glasses-detector remove them.
- Small, swaying, tilted face: all three mostly remove the frame; faint traces remain for this model and glasses-detector.
- Zoomed-in close-up with thicker rims: with a fixed 2 px margin (k = 0) this model left a faint outline at the lower rims; the default k = 0.5 removes almost all of it, k = 1.0 removes it completely but worsens the synthetic averages.

## 6. Known limitations and next steps

1. Real-image recall (outline recall ≈ 0.89 vs. 0.95 for glasses-detector): pseudo-label self-training on public real photos and videos, accepted only if the synthetic benchmarks do not regress.
2. Tiny faces and dolly shots: high residue for every method, limited by inpainting resolution.
3. Synthetic validation boundary F1 is 0.822, held down by faces under 80 px and rimless frames.
4. The training pool lacks samples with |yaw| > 50° and |pitch| > 30° at the same time.
5. End to end was run on one clip per scene; `--clips 0 1 2` runs all 33.
