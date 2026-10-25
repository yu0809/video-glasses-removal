#!/usr/bin/env python3
"""
Preview grid of synthetic samples: 8x8 cells laid out by pose (rows = pitch from looking up to looking down, columns = yaw);
each cell shows the synthetic image on the left and the label overlay on the right (green = frame, red = lens). Roll, style, tiny faces and degradations are random.

  python training/synth/preview.py --pool data/faces_pool --out data/preview/synth_grid.jpg
  python training/synth/preview.py --pool data/faces_pool --out x.jpg --no_quality    # without degradations, to inspect the rendering itself
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from synth.glasses import sample_params  # noqa: E402
from synth.pipeline import PoseSampler, load_index, make_sample  # noqa: E402

STYLES = ["full", "half", "rimless", "acetate"]


def overlay(s):
    im = s["image"].astype(np.float32)
    ov = im * 0.6
    ov[..., 1] = np.maximum(ov[..., 1], s["frame"] * 255)
    ov[..., 2] = np.maximum(ov[..., 2], s["lens"] * 150)
    return ov.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile", type=int, default=192)
    ap.add_argument("--no_quality", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    recs = [r for r in load_index(a.pool) if r.get("kps_err", 1.0) <= 0.10]
    sampler = PoseSampler(recs)
    pitches = np.linspace(45, -45, 8)          # top row looks up, bottom row looks down
    yaws = np.linspace(-70, 70, 8)
    rows, times = [], []
    k = 0
    for pt in pitches:
        line = []
        for yt in yaws:
            roll = rng.uniform(-45, 45) if rng.random() < 0.5 else rng.uniform(-10, 10)
            i = sampler.pick(np.array([pt, yt, roll]), rng)
            rec = recs[i]
            t0 = time.time()
            s = make_sample(a.pool, rec, rng, size=a.tile * 2, roll_target=roll, flip=False, negative=(k % 16 == 15),
                            params=sample_params(rng, STYLES[k % 4]), quality=not a.no_quality)
            times.append(time.time() - t0)
            left = cv2.resize(s["image"], (a.tile, a.tile), interpolation=cv2.INTER_AREA)
            right = cv2.resize(overlay(s), (a.tile, a.tile), interpolation=cv2.INTER_AREA)
            p, y, r = s["pose"]
            inf = s["info"]
            tag = [f"p{p:.0f} y{y:.0f} r{r:.0f}", f"{inf['style']} {inf.get('color') or ''}"[:22],
                   " ".join(x for x in (f"small/{inf['small_face']}" if "small_face" in inf else "",
                                        "mblur" if inf.get("motion_blur") else "", "dark" if inf.get("low_light") else "",
                                        "backlit" if inf.get("backlight") else "", inf["method"][:6]) if x)]
            for j, tx in enumerate(tag):
                cv2.putText(right, tx, (3, 13 + 13 * j), 0, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            line.append(np.hstack([left, right]))
            k += 1
        rows.append(np.hstack(line))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{a.out}  {1000 * np.mean(times):.0f} ms per sample (including image loading)")


if __name__ == "__main__":
    main()
