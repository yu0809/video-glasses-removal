"""
Training data: synthesised on the fly inside the DataLoader (random every time, unlimited samples); the validation set is generated once with a fixed seed.
=========================================================================================================================================================
SynthDataset: each __getitem__ samples a source from the training pose distribution, renders glasses + augmentations, returns tensors.
  The macOS DataLoader uses spawn, so only paths and parameters are stored here; the source index is loaded lazily in each worker; workers call cv2.setNumThreads(0).
make_val_set: generate N validation samples with a fixed seed -> data/val_synth/{images,frame,lens}/xxxxx.png + meta.jsonl (pose, face size etc.,
  for bucketed evaluation).

  python training/dataset.py --pool data/faces_pool --make_val data/val_synth --n 1000
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def worker_init(_):
    cv2.setNumThreads(0)


def split_ids(recs, val_frac=0.03, seed=0):
    """Train/validation split by person: all samples derived from one source (video, original image), pose augmentations included, land on the same side."""
    by_id = {r["id"]: r for r in recs}

    def key(r, depth=0):
        b = by_id.get(r.get("base_id")) if r.get("base_id") else None
        if b is not None and depth < 3:          # pose-augmented samples belong to the person of their base image
            return key(b, depth + 1)
        return f"{r.get('dataset')}/{r.get('src')}"
    srcs = sorted(set(key(r) for r in recs))
    rng = np.random.default_rng(seed)
    val = set(rng.choice(srcs, max(1, int(len(srcs) * val_frac)), replace=False).tolist())
    tr = [r for r in recs if key(r) not in val]
    va = [r for r in recs if key(r) in val]
    return tr, va


def load_recs(pool, max_kps_err=0.10):
    from synth.pipeline import load_index
    recs = load_index(pool)
    return [r for r in recs if r.get("kps_err", 1.0) <= max_kps_err]   # early samples without the landmark check lack this field and are excluded


class SynthDataset(Dataset):
    def __init__(self, pool, split="train", size=384, epoch_len=20000, seed=0, val_frac=0.03):
        self.pool, self.split, self.size, self.epoch_len, self.seed, self.val_frac = pool, split, size, epoch_len, seed, val_frac
        self.epoch = 0
        self._sampler = None

    def _lazy(self):
        if self._sampler is None:
            from synth.pipeline import PoseSampler
            tr, va = split_ids(load_recs(self.pool), self.val_frac)
            self._sampler = PoseSampler(tr if self.split == "train" else va)
        return self._sampler

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, i):
        from synth.pipeline import sample_any
        sampler = self._lazy()
        rng = np.random.default_rng([self.seed, self.epoch, i, os.getpid()])
        for _ in range(5):
            try:
                s = sample_any(self.pool, sampler, rng, size=self.size)
                break
            except Exception as e:  # resample when a source is geometrically degenerate
                err = e
        else:
            raise err
        return to_tensors(s)


def to_tensors(s):
    return dict(image=torch.from_numpy(np.ascontiguousarray(s["image"])),           # (H,W,3) uint8 BGR
                target=torch.from_numpy(np.stack([s["frame"], s["lens"]]).astype(np.float32)),
                weight=torch.from_numpy(s["weight"].astype(np.float32)),
                pose=torch.from_numpy(s["pose"].astype(np.float32)),
                small=torch.tensor(float(s["info"].get("small_face", 1.0))))


class FixedSet(Dataset):
    """The fixed validation set produced by make_val_set."""

    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "meta.jsonl")) as f:
            self.meta = [json.loads(l) for l in f]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        m = self.meta[i]
        n = m["name"]
        img = cv2.imread(os.path.join(self.root, "images", n + ".png"))
        fr = cv2.imread(os.path.join(self.root, "frame", n + ".png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        ln = cv2.imread(os.path.join(self.root, "lens", n + ".png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        return dict(image=torch.from_numpy(img), target=torch.from_numpy(np.stack([fr, ln])),
                    weight=torch.ones(fr.shape), pose=torch.tensor(m["pose"], dtype=torch.float32),
                    small=torch.tensor(float(m.get("small_face", 1.0))), idx=i)


def make_val_set(pool, out, n, size=384, seed=12345):
    from synth.pipeline import PoseSampler, sample_any
    tr, va = split_ids(load_recs(pool))
    sampler = PoseSampler(va)
    rng = np.random.default_rng(seed)
    by_id = {r["id"]: r for r in va}
    for d in ("images", "frame", "lens"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    with open(os.path.join(out, "meta.jsonl"), "w") as f:
        for i in range(n):
            s = sample_any(pool, sampler, rng, size=size)
            name = f"{i:05d}"
            cv2.imwrite(os.path.join(out, "images", name + ".png"), s["image"])
            cv2.imwrite(os.path.join(out, "frame", name + ".png"), np.round(s["frame"] * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(out, "lens", name + ".png"), np.round(s["lens"] * 255).astype(np.uint8))
            inf = s["info"]
            # equivalent "face size in the original video" (px): face size of the source image (the 512 pipeline crop holds a ~394 px face; finer detail is unused) / tiny-face downscale factor
            face_eff = min(float(by_id[inf["id"]]["face_px"]), 512 / 1.3) / float(inf.get("small_face", 1.0))
            f.write(json.dumps(dict(name=name, pose=[round(float(v), 1) for v in s["pose"]], id=inf["id"],
                                    method=inf["method"], style=inf["style"], cat=inf.get("pose_cat"),
                                    small_face=inf.get("small_face", 1.0), face_eff=round(face_eff, 1), motion_blur=bool(inf.get("motion_blur")),
                                    low_light=bool(inf.get("low_light")), backlight=bool(inf.get("backlight")))) + "\n")
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{n}", flush=True)
    print(f"validation set {n} images -> {out} (validation sources {len(va)}, training {len(tr)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--make_val", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--size", type=int, default=384)
    a = ap.parse_args()
    make_val_set(a.pool, a.make_val, a.n, a.size)
