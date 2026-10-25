"""
Baselines behind one interface, "pipeline crop (BGR) -> frame probability map (0-1, same size)", for a fair
comparison on the same benchmarks.
=============================================================================================================
  ours        this project's frame segmentation network (seg_model.FrameSegmenter)
  rule        rule-based adaptive mask (adaptive.py, BiSeNet + line detection). It outputs an inpainting mask with a
              margin, not a precise segmentation
  gd_small    glasses-detector (MIT, pip package) kind="frames" size="small" (TinySegNet, 1 MB)
  gd_medium   glasses-detector kind="frames" size="medium" (LR-ASPP MobileNetV3, 13 MB)
  tog         the eyeglasses-mask branch of take-off-eyeglasses (Lyu et al., CVPR 2022; DomainAdapter + GlassMask_Net).
              The repository has no licence file: used for local evaluation only, its code and weights are not redistributed.
All baselines use the authors' released weights without retraining.
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# take-off-eyeglasses must be cloned separately (github.com/StoryMY/take-off-eyeglasses) with the authors'
# pretrained.pt in ckpt/; point TOG_DIR at it (default third_party/take-off-eyeglasses)
TOG_DIR = os.environ.get("TOG_DIR", os.path.join(ROOT, "third_party", "take-off-eyeglasses"))


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Ours:
    name = "ours"

    def __init__(self, ckpt):
        sys.path.insert(0, ROOT)
        from seg_model import FrameSegmenter
        self.seg = FrameSegmenter(ckpt)

    def __call__(self, crops):
        return self.seg.predict(crops)[0]


class Rule:
    """Rule-based: the same per-frame adaptive mask as remove_glasses.py (without the temporal union)."""
    name = "rule"

    def __init__(self):
        sys.path.insert(0, ROOT)
        import adaptive as ad
        from uniface import BiSeNet
        from uniface.constants import ParsingWeights
        self.ad = ad
        self.parser = BiSeNet(model_name=ParsingWeights.RESNET34)

    def __call__(self, crops):
        masks, _, _, _ = self.ad.adaptive_masks(crops, self.parser, dict(self.ad.DEFAULT_CFG), temples="auto", union=0)
        return [m.astype(np.float32) / 255 for m in masks]


class GlassesDetector:
    def __init__(self, size="small"):
        from glasses_detector import GlassesSegmenter
        self.name = f"gd_{size}"
        self.seg = GlassesSegmenter(kind="frames", size=size, device=_device())

    def __call__(self, crops):
        rgb = [np.ascontiguousarray(c[..., ::-1]) for c in crops]
        out = self.seg(rgb, format="proba", input_size=(256, 256))
        out = out if isinstance(out, list) else [out]
        return [o.float().cpu().numpy() if hasattr(o, "cpu") else np.asarray(o, np.float32) for o in out]


class TakeOffEyeglasses:
    """Only its eyeglasses-mask branch; input 256x256 normalised to [-1, 1], output = softmax probability of class 1."""
    name = "tog"

    def __init__(self):
        import torch
        import torch.nn as nn
        sys.path.insert(0, TOG_DIR)
        from models.networks import ResnetGeneratorMask
        from models.domain_adaption import DomainAdapter
        self.torch = torch
        self.dev = torch.device(_device())
        ck = torch.load(os.path.join(TOG_DIR, "ckpt", "pretrained.pt"), map_location="cpu", weights_only=True)
        self.da = DomainAdapter().to(self.dev).eval()
        self.gm = ResnetGeneratorMask(input_nc=64, output_nc=2, norm_layer=nn.BatchNorm2d).to(self.dev).eval()
        self.da.load_state_dict(ck["DA"])
        self.gm.load_state_dict(ck["GM"])

    def __call__(self, crops, batch=8):
        t = self.torch
        out = []
        for k in range(0, len(crops), batch):
            chunk = crops[k:k + batch]
            x = np.stack([cv2.resize(c, (256, 256), interpolation=cv2.INTER_AREA)[..., ::-1] for c in chunk])
            x = t.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).float().div(127.5).sub(1).to(self.dev)
            with t.no_grad():
                gfm, _ = self.da(x)
                p = t.softmax(self.gm(gfm), 1)[:, 1].float().cpu().numpy()
            for c, pp in zip(chunk, p):
                out.append(cv2.resize(pp, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_LINEAR))
        return out


def build(names, ckpt=None):
    out = []
    for n in names:
        if n == "ours":
            out.append(Ours(ckpt))
        elif n == "rule":
            out.append(Rule())
        elif n.startswith("gd_"):
            out.append(GlassesDetector(n[3:]))
        elif n == "tog":
            out.append(TakeOffEyeglasses())
        else:
            raise ValueError(n)
    return out
