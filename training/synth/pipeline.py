"""
Full generation of one training sample + pose sampling
=======================================================
PoseSampler: samples a target pose from the training distribution (50% uniform over the full range, 30% one axis at a large angle,
  20% several axes at large angles); pitch/yaw pick the nearest cell of the source pool (10 deg cells; empty cells fall back to the
  nearest non-empty one), roll is matched exactly by a 2D rotation; flipping negates yaw and roll. This way rare pitch x yaw cells
  are not drowned out.
make_sample: load source -> render glasses (10% negatives without glasses) -> rotation + pipeline crop + jitter -> wide-angle distortion -> flip -> degradations.
Returns image (uint8, size^2), frame / lens / weight (float32), pose (final pitch, yaw, roll) and info.
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from synth import augment as aug  # noqa: E402
from synth.glasses import sample_params  # noqa: E402
from synth.render import render_sample  # noqa: E402

LIM = dict(pitch=50.0, yaw=75.0, roll=45.0)
CELL = 10.0


def load_index(pool):
    with open(os.path.join(pool, "index.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]


class PoseSampler:
    def __init__(self, recs):
        self.recs = recs
        self.p = np.array([r["pitch"] for r in recs], np.float32)
        self.y = np.array([r["yaw"] for r in recs], np.float32)
        self.r = np.array([r["roll"] for r in recs], np.float32)
        self.cells = {}
        for i, (pp, yy) in enumerate(zip(self.p, self.y)):
            self.cells.setdefault(self._cell(pp, yy), []).append(i)
        self.keys = np.array(list(self.cells.keys()), np.float32)

    @staticmethod
    def _cell(p, y):
        return (int(np.floor(p / CELL)), int(np.floor(y / CELL)))

    def target(self, rng):
        u = rng.random()
        t = np.array([rng.uniform(-1, 1) * LIM["pitch"], rng.uniform(-1, 1) * LIM["yaw"], rng.uniform(-1, 1) * LIM["roll"]])
        lims = np.array([LIM["pitch"], LIM["yaw"], LIM["roll"]])
        if u < 0.5:
            cat = "uniform"
        elif u < 0.8:
            cat = "single"
            ax = int(rng.integers(3))
            t = rng.uniform(-15, 15, 3)
            t[ax] = rng.choice([-1, 1]) * rng.uniform(0.6, 1.0) * lims[ax]
        else:
            cat = "combo"
            axes = rng.permutation(3)[: int(rng.integers(2, 4))]
            for ax in axes:
                t[ax] = rng.choice([-1, 1]) * rng.uniform(25, lims[ax])
        return t, cat

    def pick(self, t, rng):
        """Random sample from the non-empty cell nearest to the target (pitch, yaw)."""
        c = np.array(self._cell(t[0], t[1]), np.float32)
        k = self.cells.get(tuple(int(v) for v in c))
        if k is None:
            j = int(np.argmin(((self.keys - c) ** 2).sum(1)))
            k = self.cells[tuple(int(v) for v in self.keys[j])]
        return k[int(rng.integers(len(k)))]


def load_face(pool, rec):
    fdir = os.path.join(pool, "faces")
    img = cv2.imread(os.path.join(fdir, rec["id"] + ".jpg"))
    d = np.load(os.path.join(fdir, rec["id"] + ".npz"))
    unc = None
    if "uncertain" in d:
        unc = np.unpackbits(d["uncertain"])[: int(np.prod(d["uncertain_shape"]))].reshape(d["uncertain_shape"])
    return img, d["landmarks"], d["parsing"], unc


def make_sample(pool, rec, rng, size=384, roll_target=None, flip=None, negative=None, params=None,
                quality=True, distort=None, cam_shift=None):
    img, lm, parsing, unc = load_face(pool, rec)
    info = dict(id=rec["id"], method=rec.get("method", "real"), dataset=rec.get("dataset"))
    negative = (rng.random() < 0.1) if negative is None else negative
    params = params if params is not None else sample_params(rng)
    o = render_sample(img, lm, parsing, rng, params=params, draw_glasses=not negative)
    info["style"] = "none" if negative else params["style"]
    info["color"] = None if negative else params["color_name"]
    weight = np.ones(img.shape[:2], np.float32)
    if unc is not None:
        weight[unc > 0] = 0.3                  # lower weight for regions newly exposed by method B (filled by inpainting)

    # method A: exact roll rotation + pipeline crop (done at 512, labels then resized to size with INTER_AREA)
    roll0 = float(rec["roll"])
    roll_t = roll0 if roll_target is None else float(roll_target)
    cam = rng.normal(0, 0.04) if cam_shift is None else cam_shift
    H = img.shape[0]
    M = aug.geometry_matrix(lm[:, :2].astype(np.float64), H, roll_t - roll0, rng, cam_shift=cam)
    im, fr, ln, wt = aug.warp_all(M, [o["image"], o["frame_mask"], o["lens_mask"], weight], H)
    if (rng.random() < 0.2) if distort is None else distort:
        im, fr, ln, wt = aug.lens_distort([im, fr, ln, wt], rng)
        info["distort"] = True
    flip = (rng.random() < 0.5) if flip is None else flip
    pose = np.array([rec["pitch"], rec["yaw"], roll_t], np.float32)
    if flip:
        im, fr, ln, wt = [a[:, ::-1].copy() for a in (im, fr, ln, wt)]
        pose[1:] *= -1
    if size != H:
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        fr, ln, wt = [cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA) for a in (fr, ln, wt)]
    if quality:
        im = aug.quality_aug(im, rng, info)
    return dict(image=im, frame=np.clip(fr, 0, 1), lens=np.clip(ln, 0, 1), weight=wt, pose=pose,
                info=info, params=params)


def sample_any(pool, sampler, rng, size=384, **kw):
    """Sample one example from the training pose distribution."""
    t, cat = sampler.target(rng)
    flip = rng.random() < 0.5
    tt = t.copy()
    if flip:
        tt[1:] *= -1            # pose before flipping
    i = sampler.pick(tt, rng)
    s = make_sample(pool, sampler.recs[i], rng, size=size, roll_target=tt[2], flip=flip, **kw)
    s["info"]["pose_cat"] = cat
    s["info"]["target"] = t.round(1).tolist()
    return s
