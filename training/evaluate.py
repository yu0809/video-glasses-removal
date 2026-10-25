#!/usr/bin/env python3
"""
Evaluation
==========
1) Bucketed synthetic validation set: boundary F1 (2 px), IoU, Dice, bucketed by |pitch|, |yaw|, |roll|, combined pose (several axes > 25 deg),
   equivalent face size (face in the original video < 80 / 80-200 / > 200 px), glasses style, quality (motion blur / low light / back light).
     python training/evaluate.py val --ckpt runs/r34_384/best.pt --val data/val_synth --out reports/val
2) Synthetic test-video benchmark: per frame, detect, crop and predict the pipeline way, paste back and compare with the ground truth;
   temporal stability = difference between neighbouring predictions after aligning them by the label motion.
     python training/evaluate.py videos --ckpt runs/r34_384/best.pt --videos data/test_videos --out reports/videos
Outputs markdown tables (report.md) and per-sample CSVs.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from metrics import seg_metrics  # noqa: E402


def bucket_rows(rows, key, edges, labels):
    out = defaultdict(list)
    for r in rows:
        v = key(r)
        for (lo, hi), lb in zip(edges, labels):
            if lo <= v < hi:
                out[lb].append(r)
                break
    return [(lb, out[lb]) for lb in labels]


def summarize(group):
    pos = [r for r in group if r["has_gt"]]
    if not pos:
        return dict(n=len(group), bf1=np.nan, iou=np.nan, dice=np.nan, prec=np.nan, rec=np.nan)
    return dict(n=len(pos), bf1=np.mean([r["bf1"] for r in pos]), iou=np.mean([r["iou"] for r in pos]),
                dice=np.mean([r["dice"] for r in pos]), prec=np.mean([r["precision"] for r in pos]),
                rec=np.mean([r["recall"] for r in pos]))


def table(title, buckets):
    lines = [f"### {title}", "", "| Bucket | Samples | Boundary F1 | IoU | Dice | Precision | Recall |", "|---|---|---|---|---|---|---|"]
    for lb, g in buckets:
        s = summarize(g)
        lines.append(f"| {lb} | {s['n']} | {s['bf1']:.3f} | {s['iou']:.3f} | {s['dice']:.3f} | {s['prec']:.3f} | {s['rec']:.3f} |")
    return "\n".join(lines) + "\n"


def cmd_val(a):
    from seg_model import FrameSegmenter
    from dataset import FixedSet
    seg = FrameSegmenter(a.ckpt)
    ds = FixedSet(a.val)
    rows = []
    for k in range(0, len(ds), 32):
        items = [ds[i] for i in range(k, min(len(ds), k + 32))]
        fr, _ = seg.predict([it["image"].numpy() for it in items])
        for it, p in zip(items, fr):
            m = ds.meta[int(it["idx"])]
            r = seg_metrics(p, it["target"].numpy()[0])
            r.update(name=m["name"], pitch=m["pose"][0], yaw=m["pose"][1], roll=m["pose"][2], style=m["style"],
                     face_eff=m.get("face_eff", 300), motion_blur=m.get("motion_blur"), low_light=m.get("low_light"),
                     backlight=m.get("backlight"), method=m.get("method"))
            rows.append(r)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "per_sample.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    big = lambda r: sum(abs(r[k]) > 25 for k in ("pitch", "yaw", "roll"))  # noqa: E731
    md = [f"# Synthetic validation set\n\nWeights: `{a.ckpt}`, {len(rows)} samples ({sum(not r['has_gt'] for r in rows)} negatives)\n",
          table("Overall", [("all", rows)]),
          table("|pitch|", bucket_rows(rows, lambda r: abs(r["pitch"]), [(0, 15), (15, 30), (30, 999)], ["0-15°", "15-30°", "30°+"])),
          table("|yaw|", bucket_rows(rows, lambda r: abs(r["yaw"]), [(0, 15), (15, 30), (30, 50), (50, 999)], ["0-15°", "15-30°", "30-50°", "50°+"])),
          table("|roll|", bucket_rows(rows, lambda r: abs(r["roll"]), [(0, 15), (15, 30), (30, 999)], ["0-15°", "15-30°", "30°+"])),
          table("Combined pose (axes beyond 25°)", bucket_rows(rows, big, [(0, 1), (1, 2), (2, 3), (3, 4)], ["0 axes", "1 axis", "2 axes", "3 axes"])),
          table("Equivalent face size in the original video", bucket_rows(rows, lambda r: r["face_eff"], [(0, 80), (80, 200), (200, 1e9)], ["<80px", "80-200px", ">200px"])),
          table("Style", [(s, [r for r in rows if r["style"] == s]) for s in ("full", "half", "rimless", "acetate")]),
          table("Quality", [("motion blur", [r for r in rows if r["motion_blur"]]), ("low light", [r for r in rows if r["low_light"]]),
                        ("back light", [r for r in rows if r["backlight"]])]),
          table("Source", [(m, [r for r in rows if r["method"] == m]) for m in ("real", "liveportrait", "reproject")])]
    neg = [r for r in rows if not r["has_gt"]]
    if neg:
        md.append(f"Mean false-positive area on negatives (no glasses): {np.mean([r['fp_frac'] for r in neg]) * 100:.3f}%\n")
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(md))
    print("\n".join(md))


# ----------------------------------------------------------------------------
def pipeline_crops(frames, fps):
    """Detect, segment shots and smooth the crop like remove_glasses.py; return one affine matrix (or None) per frame."""
    import remove_glasses as rg
    from uniface import RetinaFace
    det = RetinaFace()
    cands, cut, prev = [], [], None
    for f in frames:
        fs = det.detect(f) or []
        c = []
        for fc in fs:
            x1, y1, x2, y2 = [float(v) for v in fc.bbox[:4]]
            if min(x2 - x1, y2 - y1) >= 24:
                c.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
        cands.append(c)
        sig = rg.frame_signature(f)
        cut.append(0.0 if prev is None else cv2.compareHist(prev, sig, cv2.HISTCMP_BHATTACHARYYA))
        prev = sig

    class A:
        max_gap, min_segment, cut_threshold = 0.3, 5, 0.45
    segs = rg.build_segments(cands, np.array(cut), fps, A)
    mats = [None] * len(frames)
    for s in segs:
        M, _ = rg.crop_transforms(s["boxes"], fps, 1.3, 512, 0.08)
        for j, m in enumerate(M):
            mats[s["start"] + j] = m
    return mats


def cmd_videos(a):
    import baselines
    seg = baselines.build([a.method], a.ckpt)[0]
    os.makedirs(a.out, exist_ok=True)
    clips = sorted(d for d in os.listdir(a.videos) if os.path.exists(os.path.join(a.videos, d, "meta.json")))
    allrows = []
    for c in clips:
        d = os.path.join(a.videos, c)
        meta = json.load(open(os.path.join(d, "meta.json")))
        cap = cv2.VideoCapture(os.path.join(d, "glasses.mp4"))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        mats = pipeline_crops(frames, meta["fps"])
        idx = [i for i, m in enumerate(mats) if m is not None]
        crops = [cv2.warpAffine(frames[i], mats[i], (512, 512), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT) for i in idx]
        probs = seg(crops) if crops else []
        # compare on the 512 crop: labels are warped to the crop with the same affine (for distant faces the frame would shrink below 1 px at the original resolution and distort the comparison)
        pred = {i: p for i, p in zip(idx, probs)}
        rows, prev = [], None
        for i in range(len(frames)):
            gt_full = cv2.imread(os.path.join(d, "mask", f"{i:05d}.png"), 0).astype(np.float32) / 255
            if mats[i] is None:
                r = dict(has_gt=gt_full.sum() > 0, bf1=0.0, iou=0.0, dice=0.0, fp_frac=0.0)
                r.update(clip=c, scene=meta["scene"], frame=i, detected=False)
                rows.append(r)
                prev = None
                continue
            gt = cv2.warpAffine(gt_full, mats[i], (512, 512), flags=cv2.INTER_LINEAR, borderValue=0)
            r = seg_metrics(pred[i], gt)
            r.update(clip=c, scene=meta["scene"], frame=i, detected=True)
            # temporal stability: difference between neighbouring predicted masks after aligning by the label centroid shift (frames with glasses only)
            if prev is not None and gt.sum() > 0 and prev[1].sum() > 0:
                def cen(m):
                    ys, xs = np.nonzero(m > 0.5)
                    return np.array([xs.mean(), ys.mean()]) if len(xs) else np.zeros(2)
                dv = cen(gt) - cen(prev[1])
                sh = cv2.warpAffine(prev[0], np.float32([[1, 0, dv[0]], [0, 1, dv[1]]]), (512, 512))
                u = ((sh > 0.5) | (pred[i] > 0.5)).sum()
                r["flicker"] = float(((sh > 0.5) ^ (pred[i] > 0.5)).sum()) / max(1, u)
            prev = (pred[i], gt)
            rows.append(r)
        allrows += rows
        pos = [r for r in rows if r["has_gt"]]
        print(f"{c} {meta['scene_name']}: {len(rows)} frames, face detected in {sum(r['detected'] for r in rows)}, "
              f"boundary F1 {np.mean([r['bf1'] for r in pos]):.3f}, IoU {np.mean([r['iou'] for r in pos]):.3f}, "
              f"flicker {np.nanmean([r.get('flicker', np.nan) for r in rows]):.3f}")
    with open(os.path.join(a.out, "per_frame.csv"), "w", newline="") as f:
        keys = sorted(set().union(*[r.keys() for r in allrows]))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(allrows)
    lines = ["# Synthetic test-video benchmark: frame segmentation", "", "| Scene | Frames | Face detection rate | Boundary F1 | IoU | Flicker (lower is better) |", "|---|---|---|---|---|---|"]
    for sc in sorted(set(r["scene"] for r in allrows)):
        g = [r for r in allrows if r["scene"] == sc]
        pos = [r for r in g if r["has_gt"]]
        lines.append(f"| {sc:02d} | {len(g)} | {np.mean([r['detected'] for r in pos]):.0%} | {np.mean([r['bf1'] for r in pos]):.3f} | "
                     f"{np.mean([r['iou'] for r in pos]):.3f} | {np.nanmean([r.get('flicker', np.nan) for r in g]):.3f} |")
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("val")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val", default=os.path.join(ROOT, "data", "val_synth"))
    p.add_argument("--out", required=True)
    p = sub.add_parser("videos")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--method", default="ours", help="ours / rule / gd_small / gd_medium / tog (see baselines.py)")
    p.add_argument("--videos", default=os.path.join(ROOT, "data", "test_videos"))
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(val=cmd_val, videos=cmd_videos)[a.cmd](a)


if __name__ == "__main__":
    main()
