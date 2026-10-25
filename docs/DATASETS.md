# Data sources

All training and evaluation data is either public or procedurally generated. **No dataset images are included in this repository**; `training/download_data.sh` fetches them into `data/raw/`.
Because non-commercial datasets were used, the trained weights are released under **CC BY-NC-SA 4.0** (non-commercial).

| Dataset | Obtained from | Used | Licence | Purpose |
|---|---|---|---|---|
| FFHQ (NVIDIA) | Hugging Face mirror `luethan2025/FFHQ-512x512` (512 px, 55 Arrow shards, 27.5 GB) | first 16 shards (~20k images) | dataset CC BY-NC-SA 4.0; individual images from Flickr under CC BY 2.0 / CC BY-NC 2.0 / public domain etc. | glasses-free base faces |
| CelebAMask-HQ | Hugging Face mirror `liusq/CelebAMask-HQ` (3.15 GB; original release github.com/switchablenorms/CelebAMask-HQ) | all 30k images | non-commercial research only | base faces; its `eye_g` (glasses) annotation is used for evaluation |
| AFLW2000-3D | 3DDFA project page | all 2,000 images | research use | large-pose base faces |
| Microsoft Face Synthetics | official 1,000-image sample `dataset_1000.zip` (320 MB) | 1,000 images | Microsoft Research Use of Data Agreement v1.0: non-commercial research only; models trained on it are "Results" and may not be used commercially | additional evaluation (class 16 = GLASSES; opaque lenses are labelled as glasses) |
| Talking-head sample clips | example videos of the LivePortrait and VideoReTalking repositories | 5 clips | as in the original repositories | smoke tests |
| 300W-LP | 3DDFA Google Drive link | **not used** (download quota exceeded at the time) | research use | – |

Procedurally generated:

- **LivePortrait** (code MIT, github.com/KwaiVGI/LivePortrait; weights Hugging Face `KlingTeam/LivePortrait`, MIT; it depends on InsightFace detection models that are for non-commercial research only) re-poses the real faces above.
- FaceMesh mesh re-projection, 2D roll, and the parametric 3D glasses renderer are code in this repository.

Face detection, landmarks, parsing and head pose use RetinaFace, MediaPipe FaceMesh, BiSeNet (trained on CelebAMask-HQ) and a 6D-rotation head-pose model, all downloaded by [uniface](https://github.com/yakhyo/uniface).

The evaluation baselines are used with their authors' released weights and are not redistributed:
[glasses-detector](https://github.com/mantasu/glasses-detector) (MIT) and
[take-off-eyeglasses](https://github.com/StoryMY/take-off-eyeglasses) (no licence file).
