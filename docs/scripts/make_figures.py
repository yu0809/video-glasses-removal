#!/usr/bin/env python3
"""
Generates the comparison charts used in the README / blog post from the evaluation outputs (reports/, data/faces_pool) into docs/images/.

  python docs/scripts/make_figures.py

Inputs (produced by the evaluation scripts):
  reports/e2e_r1/summary.csv            training/eval_e2e.py (end-to-end, synthetic test videos)
  reports/compare_r1/per_image.csv      training/eval_real.py (CelebAMask-HQ real photos / Face Synthetics)
  reports/videos_crop_<method>/per_frame.csv  training/evaluate.py videos (per-frame segmentation on test videos)
  reports/val_r1/per_sample.csv         training/evaluate.py val (synthetic validation buckets)
  data/faces_pool/index.jsonl           training/prepare_faces.py etc. (pose distribution of the face pool)
"""
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "images")
os.makedirs(OUT, exist_ok=True)

# Colors: reference categorical palette in fixed order (validated for colour-vision deficiency).
# Each method keeps the same colour in every chart.
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
COLOR = {"ours": "#2a78d6", "gd_medium": "#eb6834", "rule": "#1baf7a", "tog": "#eda100", "gd_small": "#e87ba4"}
LABEL = {"ours": "Ours (frame segmentation net)", "gd_medium": "glasses-detector medium", "rule": "Rule-based (original pipeline)",
         "tog": "take-off-eyeglasses", "gd_small": "glasses-detector small"}

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
})


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def legend(ax, methods, **kw):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLOR[m], label=LABEL[m]) for m in methods], frameon=False, **kw)


# ----------------------------------------------------------------------------
# 1. End-to-end: share of frames where glasses are still visible after inpainting
# ----------------------------------------------------------------------------
def e2e():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "reports/e2e_r1/summary.csv"))))
    mmap = {"rule": "rule", "gd": "gd_medium", "net_k05": "ours"}
    rows = [dict(r, m=mmap[r["method"]]) for r in rows if r["method"] in mmap]
    scenes = sorted({(int(r["scene"]), r["scene_name"]) for r in rows})
    methods = ["ours", "gd_medium", "rule"]
    val = {(int(r["scene"]), r["m"]): r for r in rows}

    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    h = 0.26
    for k, m in enumerate(methods):
        y = np.arange(len(scenes)) + (k - 1) * (h + 0.02)
        v = [100 * float(val[(s, m)]["residue_rate"]) for s, _ in scenes]
        ax.barh(y, v, height=h, color=COLOR[m], edgecolor=SURFACE, linewidth=1)
        for yy, vv in zip(y, v):
            ax.text(vv + 1, yy, f"{vv:.0f}%", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(np.arange(len(scenes)))
    ax.set_yticklabels([f"{s:02d} {SCENE_NAMES[s]}" for s, _ in scenes])
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("Frames with visible glasses residue after removal (%, lower is better)")
    ax.grid(axis="y", visible=False)
    ax.set_title("End-to-end glasses removal: residue rate per scene")
    legend(ax, methods, loc="lower right")
    save(fig, "e2e_residue_by_scene.png")

    # Averages: three metrics in three panels (different units, never a shared axis)
    agg = {m: dict(res=np.mean([100 * float(val[(s, m)]["residue_rate"]) for s, _ in scenes]),
                   psnr=np.mean([float(val[(s, m)]["psnr"]) for s, _ in scenes]),
                   mask=np.mean([float(val[(s, m)]["mask_pct"]) for s, _ in scenes])) for m in methods}
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    specs = [("res", "Residue rate % (lower is better)", "{:.0f}%"), ("psnr", "PSNR around glasses, dB (higher is better)", "{:.1f}"),
             ("mask", "Mask area % (smaller = less repainting)", "{:.1f}%")]
    for ax, (k, title, fmt) in zip(axs, specs):
        v = [agg[m][k] for m in methods]
        ax.bar(range(3), v, color=[COLOR[m] for m in methods], width=0.62, edgecolor=SURFACE, linewidth=2)
        for i, vv in enumerate(v):
            ax.text(i, vv, fmt.format(vv), ha="center", va="bottom", fontsize=10, color=INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Ours", "glasses-\ndetector", "Rule-based"])
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="x", visible=False)
        lo = 20 if k == "psnr" else 0
        ax.set_ylim(lo, max(v) * 1.18)
    fig.suptitle("End-to-end averages over 11 synthetic test scenes", fontweight="bold", y=1.03)
    save(fig, "e2e_summary.png")
    return agg


# ----------------------------------------------------------------------------
# 2. Real photos (CelebAMask-HQ): F under four perturbations + false positives without glasses
# ----------------------------------------------------------------------------
def real_photos():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "reports/compare_r1/per_image.csv"))))
    g = defaultdict(list)
    for r in rows:
        g[(r["method"], r["data"], r["perturb"])].append(r)
    methods = ["ours", "gd_medium", "rule", "tog", "gd_small"]
    perts = [("orig", "Original"), ("roll30", "Head roll ±30°"), ("small", "Tiny face\n(~50 px)"), ("mblur", "Motion blur")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw=dict(width_ratios=[3, 1.25]))
    w = 0.16
    for k, m in enumerate(methods):
        x = np.arange(len(perts)) + (k - 2) * w
        v = [np.mean([float(r["F"]) for r in g[(m, "celeba_pos", p)]]) for p, _ in perts]
        ax.bar(x, v, width=w - 0.02, color=COLOR[m], edgecolor=SURFACE, linewidth=1)
        for xx, vv in zip(x, v):
            ax.text(xx, vv + 0.01, f"{vv:.2f}", ha="center", va="bottom", fontsize=7, color=INK2, rotation=90)
    ax.set_xticks(range(len(perts)))
    ax.set_xticklabels([n for _, n in perts])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F (precision vs. outline recall)")
    ax.set_title("1,415 real photos with glasses: robustness to identical perturbations")
    ax.grid(axis="x", visible=False)
    legend(ax, methods, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.14), fontsize=9)
    fp = [100 * np.mean([float(r["fp"]) for r in g[(m, "celeba_neg", "orig")]]) for m in methods]
    ax2.barh(range(len(methods)), fp, color=[COLOR[m] for m in methods], edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(fp):
        ax2.text(v + 0.08, i, f"{v:.2f}%", va="center", fontsize=9, color=INK)
    ax2.set_yticks(range(len(methods)))
    ax2.set_yticklabels(["Ours", "gd medium", "Rule-based", "take-off", "gd small"])
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(fp) * 1.35)
    ax2.set_xlabel("False-positive area, % of image")
    ax2.set_title("500 photos without glasses:\nfalse positives (lower is better)", fontsize=11)
    ax2.grid(axis="y", visible=False)
    save(fig, "real_photos_robustness.png")
    return {m: {p: np.mean([float(r["F"]) for r in g[(m, "celeba_pos", p)]]) for p, _ in perts} for m in methods}, \
        dict(zip(methods, fp))


# ----------------------------------------------------------------------------
# 3. Per-frame segmentation on test videos: scene x method heat map (boundary F1 on the 512 crop)
# ----------------------------------------------------------------------------
SCENE_NAMES = {1: "Frontal talking", 2: "Look up/down ±40°", 3: "Roll 30° + pitch 30°", 4: "Yaw ±60° + pitch 15°",
               5: "All three axes (figure 8)", 6: "Tiny face 30-60 px", 7: "Dolly zoom 60→400 px",
               8: "Fast head turn (blur)", 9: "Multi-camera cuts", 10: "Lighting", 11: "Occlusion"}


def video_seg():
    methods = ["ours", "tog", "gd_medium", "rule"]
    M = np.zeros((11, len(methods)))
    for j, m in enumerate(methods):
        rows = list(csv.DictReader(open(os.path.join(ROOT, f"reports/videos_crop_{m}/per_frame.csv"))))
        for s in range(1, 12):
            v = [float(r["bf1"]) for r in rows if int(r["scene"]) == s and r["has_gt"] == "True"]
            M[s - 1, j] = np.mean(v) if v else np.nan
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(11):
        for j in range(len(methods)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9.5,
                    color="white" if v > 0.6 else INK, fontweight="bold" if j == 0 else "normal")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(["Ours", "take-off-\neyeglasses", "glasses-\ndetector", "Rule-based"])
    ax.set_yticks(range(11))
    ax.set_yticklabels([f"{s:02d} {SCENE_NAMES[s]}" for s in range(1, 12)])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Per-frame frame segmentation on synthetic test videos\n(boundary F1, 2 px tolerance)")
    save(fig, "video_seg_heatmap.png")
    return M


# ----------------------------------------------------------------------------
# 4. Pose coverage of the face pool: real photos only vs. with pose augmentation
# ----------------------------------------------------------------------------
def pose_coverage():
    recs = [json.loads(l) for l in open(os.path.join(ROOT, "data/faces_pool/index.jsonl"))]
    recs = [r for r in recs if r.get("kps_err", 1.0) <= 0.10]
    pe, ye = np.arange(-50, 51, 10), np.arange(-80, 81, 10)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (sel, title) in zip(axs, ((lambda r: r.get("method", "real") == "real", "Public real photos only"),
                                       (lambda r: True, "+ LivePortrait re-posing + mesh re-projection"))):
        rr = [r for r in recs if sel(r)]
        h, _, _ = np.histogram2d([r["pitch"] for r in rr], [r["yaw"] for r in rr], bins=[pe, ye])
        ax.imshow(np.log1p(h), origin="lower", extent=[ye[0], ye[-1], pe[0], pe[-1]], aspect="auto", cmap=cmap)
        filled = (h[:, 1:-1] > 0).mean()
        ax.set_title(f"{title}\n{len(rr):,} faces, {filled:.0%} of the ±50°×±70° grid covered", fontsize=11)
        ax.set_xlabel("Yaw (deg)")
        ax.grid(False)
    axs[0].set_ylabel("Pitch (deg, + = looking up)")
    fig.suptitle("Head-pose distribution of the training faces (darker = more samples, log scale)", fontweight="bold", y=1.04)
    save(fig, "pose_coverage.png")


# ----------------------------------------------------------------------------
# 5. Synthetic validation buckets: pose is no longer the bottleneck; tiny faces and rimless frames are
# ----------------------------------------------------------------------------
def val_buckets():
    p = os.path.join(ROOT, "reports/val_r1/per_sample.csv")
    if not os.path.exists(p):
        print("skip val_buckets: missing", p)
        return None
    rows = [r for r in csv.DictReader(open(p)) if r["has_gt"] == "True"]
    for r in rows:
        for k in ("pitch", "yaw", "roll", "face_eff", "bf1"):
            r[k] = float(r[k])
    big = lambda r: sum(abs(r[k]) > 25 for k in ("pitch", "yaw", "roll"))  # noqa: E731
    groups = [
        ("Head pose", [("Near-frontal\npitch<15°", lambda r: abs(r["pitch"]) < 15), ("Pitch\n≥30°", lambda r: abs(r["pitch"]) >= 30),
                 ("Yaw\n≥50°", lambda r: abs(r["yaw"]) >= 50), ("Roll\n≥30°", lambda r: abs(r["roll"]) >= 30),
                 ("Two axes\n>25°", lambda r: big(r) == 2), ("Three axes\n>25°", lambda r: big(r) == 3)]),
        ("Face size in source", [(">200px", lambda r: r["face_eff"] > 200), ("80~200px", lambda r: 80 <= r["face_eff"] <= 200),
                    ("<80px", lambda r: r["face_eff"] < 80)]),
        ("Frame style", [("Acetate", lambda r: r["style"] == "acetate"), ("Full rim", lambda r: r["style"] == "full"),
                ("Half rim", lambda r: r["style"] == "half"), ("Rimless", lambda r: r["style"] == "rimless")]),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True, gridspec_kw=dict(width_ratios=[6, 3, 4]))
    out = {}
    for ax, (gname, items) in zip(axs, groups):
        v = [np.mean([r["bf1"] for r in rows if f(r)]) for _, f in items]
        n = [sum(1 for r in rows if f(r)) for _, f in items]
        ax.bar(range(len(v)), v, color=COLOR["ours"], width=0.6, edgecolor=SURFACE, linewidth=2)
        for i, (vv, nn) in enumerate(zip(v, n)):
            ax.text(i, vv + 0.01, f"{vv:.2f}", ha="center", va="bottom", fontsize=9.5)
        ax.set_xticks(range(len(v)))
        ax.set_xticklabels([f"{name}\n(n={nn})" for (name, _), nn in zip(items, n)], fontsize=8.5)
        ax.set_title(gname, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
        out[gname] = {name: round(float(vv), 3) for (name, _), vv in zip(items, v)}
    axs[0].set_ylabel("Boundary F1 (2 px)")
    fig.suptitle("Ours on the synthetic validation set: extreme and combined poses are no longer the weak spot", fontweight="bold", y=1.04)
    save(fig, "val_buckets.png")
    return out


if __name__ == "__main__":
    a = e2e()
    b = real_photos()
    c = video_seg()
    pose_coverage()
    d = val_buckets()
    print(json.dumps(dict(e2e=a, real=b[0], fp=b[1], val=d), ensure_ascii=False, indent=1, default=float))
