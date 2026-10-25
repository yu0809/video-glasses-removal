# weights/

`setup.sh` downloads the frame segmentation network here:

| File | Size | SHA-256 | Licence |
|---|---|---|---|
| `frame_seg.pt` | 98 MB | `2668cf6c2ff2e95067719c2abf98a18d85eaf428a19652e00895cf11e1d4848b` | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |

Manual download: [Releases](https://github.com/yu0809/video-glasses-removal/releases) → `frame_seg.pt` → save as `weights/frame_seg.pt`.

U-Net with a ResNet-34 encoder, 384 px input, two output channels (frame, lens), step 18,000, synthetic validation boundary F1 0.822.
The checkpoint stores its own configuration (`cfg`); load it with `seg_model.FrameSegmenter`.

**Non-commercial use only**: the network was trained on data licensed for non-commercial research
(FFHQ, CelebAMask-HQ, AFLW2000-3D); see [docs/DATASETS.md](../docs/DATASETS.md).
