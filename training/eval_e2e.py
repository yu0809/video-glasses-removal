#!/usr/bin/env python3
"""
End-to-end evaluation: run remove_glasses.py on the synthetic test videos and compare with the ground truth clean.mp4
=====================================================================================================================
Methods: rule (rule-based adaptive, --seg_model none), net (frame segmentation network, --seg_model <ckpt>),
         gd (baseline: glasses-detector medium frame segmentation with the same post-processing and rework), optionally fixed (--fixed).
Metrics (only inside the "glasses region" = union of the labelled frame dilated by 12 px, per frame):
  PSNR / SSIM (against clean)
  residue rate: frames in which more than 0.2 of the labelled frame pixels differ from clean by Lab distance > 20
  flicker: mean |(out_t - out_{t-1}) - (clean_t - clean_{t-1})| inside the region (lower is more stable)
  mask area: mean mask_pct from the remove_glasses report
  python training/eval_e2e.py --videos data/test_videos --ckpt weights/frame_seg_r1.pt --out reports/e2e_r1 --clips 0
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def read_video(p):
    cap = cv2.VideoCapture(p)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    return fr


def ssim(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    sa = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    sb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    sab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    return ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (sa + sb + C2))


def clip_metrics(out_frames, clean, masks):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    rows, prev = [], None
    for i, (o, c, m) in enumerate(zip(out_frames, clean, masks)):
        g = (m > 127).astype(np.uint8)
        if g.sum() < 20:
            prev = None
            continue
        region = cv2.dilate(g, k).astype(bool)
        d = (o.astype(np.float32) - c.astype(np.float32))[region]
        mse = float((d ** 2).mean())
        psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-6))
        s = float(ssim(cv2.cvtColor(o, cv2.COLOR_BGR2GRAY), cv2.cvtColor(c, cv2.COLOR_BGR2GRAY))[region].mean())
        lo, lc = (cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32) for x in (o, c))
        resid = float((np.linalg.norm(lo - lc, axis=2)[g > 0] > 20).mean())
        r = dict(frame=i, psnr=psnr, ssim=s, residue_frac=resid, residue=resid > 0.2)
        if prev is not None:
            po, pc = prev
            te = np.abs((o.astype(np.float32) - po) - (c.astype(np.float32) - pc))[region].mean()
            r["flicker"] = float(te)
        prev = (o.astype(np.float32), c.astype(np.float32))
        rows.append(r)
    return rows


# network variants: net_k05 -> --seg_margin_k 0.5 etc.
EXTRA = {f"net_k{k.replace('.', '')}": ["--seg_margin_k", k] for k in ("0.5", "1.0", "1.5")}


def run_method(clip_dir, method, ckpt, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    res = os.path.join(out_dir, "glasses_no_glasses.mp4")
    if not os.path.exists(res):
        cmd = [sys.executable, os.path.join(ROOT, "remove_glasses.py"), "-i", os.path.join(clip_dir, "glasses.mp4"),
               "-o", out_dir, "--min_face", "24"]
        if method == "rule":
            cmd += ["--seg_model", "none"]
        elif method == "fixed":
            cmd += ["--fixed", "--seg_model", "none"]
        elif method == "gd":
            cmd += ["--seg_model", "glasses-detector:medium"]
        else:
            cmd += ["--seg_model", ckpt] + EXTRA.get(method, [])
        log = open(os.path.join(out_dir, "run.log"), "w")
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        if rc != 0 or not os.path.exists(res):
            return None, None
    rep = os.path.join(out_dir, "glasses_report.csv")
    mask_pct = []
    if os.path.exists(rep):
        for r in csv.DictReader(open(rep, encoding="utf-8-sig")):
            if r.get("glasses") in ("1", "1.0") and r.get("mask_pct"):
                mask_pct.append(float(r["mask_pct"]))
    return read_video(res), (float(np.mean(mask_pct)) if mask_pct else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=os.path.join(ROOT, "data", "test_videos"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=["rule", "net"])
    ap.add_argument("--clips", type=int, nargs="*", default=[0], help="which clips of each scene to use (default clip 0)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    clips = sorted(d for d in os.listdir(a.videos) if os.path.exists(os.path.join(a.videos, d, "meta.json"))
                   and int(d.split("_")[1]) in a.clips)
    summary = []
    for c in clips:
        d = os.path.join(a.videos, c)
        meta = json.load(open(os.path.join(d, "meta.json")))
        clean = read_video(os.path.join(d, "clean.mp4"))
        masks = [cv2.imread(os.path.join(d, "mask", f"{i:05d}.png"), 0) for i in range(len(clean))]
        for m in a.methods:
            outs, mpct = run_method(d, m, a.ckpt, os.path.join(a.out, c, m))
            if outs is None:
                print(f"{c} {m}: run failed (see {os.path.join(a.out, c, m, 'run.log')})", flush=True)
                continue
            n = min(len(outs), len(clean))
            rows = clip_metrics(outs[:n], clean[:n], masks[:n])
            s = dict(clip=c, scene=meta["scene"], scene_name=meta["scene_name"], method=m, frames=len(rows),
                     psnr=np.mean([r["psnr"] for r in rows]), ssim=np.mean([r["ssim"] for r in rows]),
                     residue_rate=np.mean([r["residue"] for r in rows]),
                     flicker=np.nanmean([r.get("flicker", np.nan) for r in rows]), mask_pct=mpct)
            summary.append(s)
            print(f"{c} {meta['scene_name']} {m}: PSNR {s['psnr']:.2f} SSIM {s['ssim']:.3f} residue {s['residue_rate']:.0%} "
                  f"flicker {s['flicker']:.2f} mask {mpct:.2f}%", flush=True)
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    L = ["# End to end: synthetic test videos (glasses region)", "", "| Scene | Method | PSNR↑ | SSIM↑ | Residue↓ | Flicker↓ | Mask area |", "|---|---|---|---|---|---|---|"]
    for s in summary:
        L.append(f"| {s['scene']:02d} {s['scene_name']} | {s['method']} | {s['psnr']:.2f} | {s['ssim']:.3f} | {s['residue_rate']:.0%} | "
                 f"{s['flicker']:.2f} | {s['mask_pct']:.2f}% |")
    L += ["", "| Method | Mean PSNR | Mean SSIM | Mean residue | Mean flicker | Mean mask |", "|---|---|---|---|---|---|"]
    for m in a.methods:
        ss = [s for s in summary if s["method"] == m]
        if ss:
            L.append(f"| {m} | {np.mean([s['psnr'] for s in ss]):.2f} | {np.mean([s['ssim'] for s in ss]):.3f} | "
                     f"{np.mean([s['residue_rate'] for s in ss]):.0%} | {np.mean([s['flicker'] for s in ss]):.2f} | "
                     f"{np.mean([s['mask_pct'] for s in ss]):.2f}% |")
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
