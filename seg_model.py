"""
Inference wrapper for the frame segmentation network (used by remove_glasses.py)
================================================================================
    seg = FrameSegmenter("weights/frame_seg.pt")
    frame_probs, lens_probs = seg.predict(crops_bgr)    # per-frame 0-1 probability maps, same size as the input

Input: pipeline face crops (face box x 1.3, shifted up by 0.08, BGR). The checkpoint carries its config
(encoder, input size, normalisation). ONNX (.onnx, run with onnxruntime, no torch needed) is supported too.
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


class FrameSegmenter:
    def __init__(self, ckpt_path, device=None, batch=8):
        self.batch = batch
        self.external = None
        if ckpt_path.startswith("glasses-detector:"):
            # baseline: glasses-detector (MIT) frame segmentation, e.g. "glasses-detector:medium". It has no lens output, so lens probability is 0
            from glasses_detector import GlassesSegmenter
            import torch
            dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
            self.external = GlassesSegmenter(kind="frames", size=ckpt_path.split(":", 1)[1], device=dev)
            return
        self.onnx = ckpt_path.endswith(".onnx")
        if self.onnx:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(ckpt_path, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
            self.size = int(self.sess.get_inputs()[0].shape[-1])
            self.mean = np.array([0.485, 0.456, 0.406], np.float32)
            self.std = np.array([0.229, 0.224, 0.225], np.float32)
            return
        import torch
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        sys.path.insert(0, os.path.join(HERE, "training"))
        from model import build_model
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cfg = ck["cfg"]
        self.size = int(self.cfg.get("size", 384))
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model = build_model(self.cfg, pretrained=False)
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        self.mean = np.array(self.cfg.get("mean", [0.485, 0.456, 0.406]), np.float32)
        self.std = np.array(self.cfg.get("std", [0.229, 0.224, 0.225]), np.float32)

    def _prep(self, crops):
        x = np.stack([cv2.resize(c, (self.size, self.size), interpolation=cv2.INTER_AREA)[..., ::-1] for c in crops])
        x = (x.astype(np.float32) / 255 - self.mean) / self.std
        return np.ascontiguousarray(x.transpose(0, 3, 1, 2))

    def predict(self, crops_bgr):
        """Return (frame_probs, lens_probs): two lists of float32 probability maps, each the size of its input crop."""
        fr, ln = [], []
        if self.external is not None:
            for k in range(0, len(crops_bgr), self.batch):
                chunk = [np.ascontiguousarray(c[..., ::-1]) for c in crops_bgr[k:k + self.batch]]
                out = self.external(chunk, format="proba", input_size=(256, 256))
                for o in (out if isinstance(out, list) else [out]):
                    o = o.float().cpu().numpy()
                    fr.append(o)
                    ln.append(np.zeros_like(o))
            return fr, ln
        for k in range(0, len(crops_bgr), self.batch):
            chunk = crops_bgr[k:k + self.batch]
            x = self._prep(chunk)
            if self.onnx:
                logits = self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0]
            else:
                import torch
                with torch.no_grad():
                    logits = self.model(torch.from_numpy(x).to(self.device)).float().cpu().numpy()
            prob = 1 / (1 + np.exp(-logits))
            for c, p in zip(chunk, prob):
                h, w = c.shape[:2]
                fr.append(cv2.resize(p[0], (w, h), interpolation=cv2.INTER_LINEAR))
                ln.append(cv2.resize(p[1], (w, h), interpolation=cv2.INTER_LINEAR))
        return fr, ln
