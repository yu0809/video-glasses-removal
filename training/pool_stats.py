#!/usr/bin/env python3
"""
Pose / face-size statistics of the face source pool, plotted to stats.png, plus a sample of landmark overlays to check alignment.

  python training/pool_stats.py data/faces_pool                # -> data/faces_pool/stats.png
  python training/pool_stats.py data/faces_pool --check 24     # also draw 24 landmark spot checks to check_landmarks.jpg

Target range: pitch +-50 deg, yaw +-75 deg, roll +-45 deg,
with samples in every 10 deg pitch x yaw cell.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import LM  # noqa: E402

RANGE = dict(pitch=50, yaw=75, roll=45)
P_EDGES = np.arange(-50, 51, 10)
Y_EDGES = np.arange(-80, 81, 10)
R_EDGES = np.arange(-50, 51, 10)


def load_index(root):
    with open(os.path.join(root, "index.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]


def coverage(recs):
    """Share of non-empty cells in the pitch x yaw target grid (pitch +-50, yaw +-70, 10 deg cells), and the number of combined large-pose samples."""
    p = np.array([r["pitch"] for r in recs])
    y = np.array([r["yaw"] for r in recs])
    rl = np.array([r["roll"] for r in recs])
    h, _, _ = np.histogram2d(p, y, bins=[P_EDGES, np.arange(-70, 71, 10)])
    big = (np.abs(p) > 25).astype(int) + (np.abs(y) > 25) + (np.abs(rl) > 25)
    return dict(cells_filled=float((h > 0).mean()), cells_min=int(h.min()), n=len(recs),
                combo_2axis=int((big >= 2).sum()), combo_3axis=int((big >= 3).sum()),
                abs_pitch_gt25=int((np.abs(p) > 25).sum()))


def plot(recs, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = np.array([r["pitch"] for r in recs])
    y = np.array([r["yaw"] for r in recs])
    rl = np.array([r["roll"] for r in recs])
    fp = np.array([r["face_px"] for r in recs])
    methods = sorted(set(r.get("method", "real") for r in recs))
    fig, ax = plt.subplots(2, 4, figsize=(20, 9))
    for k, (v, nm, edges) in enumerate(((p, "pitch (+ = up)", P_EDGES), (y, "yaw (+ = subject's right)", Y_EDGES),
                                         (rl, "roll (+ = clockwise)", R_EDGES))):
        a = ax[0, k]
        bins = np.arange(-90, 91, 5)
        bottom = np.zeros(len(bins) - 1)
        for m in methods:
            sel = np.array([r.get("method", "real") == m for r in recs])
            h, _ = np.histogram(v[sel], bins)
            a.bar(bins[:-1], h, width=5, align="edge", bottom=bottom, label=m)
            bottom += h
        lim = RANGE[nm.split()[0]]
        a.axvline(-lim, color="k", ls="--", lw=1)
        a.axvline(lim, color="k", ls="--", lw=1)
        a.set_title(nm)
        a.legend(fontsize=7)
    a = ax[0, 3]
    a.hist(fp, bins=np.logspace(np.log10(max(10, fp.min() * 0.9)), np.log10(fp.max() * 1.1 + 1), 30))
    a.set_xscale("log")
    a.set_title("face size in source (px)")

    def heat(a, u, v, ue, ve, xl, yl):
        h, _, _ = np.histogram2d(u, v, bins=[ue, ve])
        im = a.imshow(np.log1p(h), origin="lower", aspect="auto", cmap="viridis",
                      extent=[ve[0], ve[-1], ue[0], ue[-1]])
        for i in range(len(ue) - 1):
            for j in range(len(ve) - 1):
                c = int(h[i, j])
                a.text((ve[j] + ve[j + 1]) / 2, (ue[i] + ue[i + 1]) / 2, str(c), ha="center", va="center",
                       fontsize=6, color="w" if c == 0 else "k" if c > h.max() / 2 else "w")
        a.set_xlabel(xl)
        a.set_ylabel(yl)
        return im

    heat(ax[1, 0], p, y, P_EDGES, Y_EDGES, "yaw", "pitch")
    ax[1, 0].set_title("pitch x yaw (count, color=log)")
    heat(ax[1, 1], p, rl, P_EDGES, R_EDGES, "roll", "pitch")
    ax[1, 1].set_title("pitch x roll")
    heat(ax[1, 2], y, rl, Y_EDGES, R_EDGES, "roll", "yaw")
    ax[1, 2].set_title("yaw x roll")
    a = ax[1, 3]
    cnt = [sum(1 for r in recs if r.get("method", "real") == m) for m in methods]
    a.barh(methods, cnt)
    for i, c in enumerate(cnt):
        a.text(c, i, f" {c}", va="center")
    cov = coverage(recs)
    a.set_title("samples per method")
    a.text(0.02, -0.25, f"pitch x yaw grid (±50 x ±70) filled: {cov['cells_filled']:.0%}, min/cell {cov['cells_min']}\n"
           f"|pitch|>25: {cov['abs_pitch_gt25']}   2-axis >25: {cov['combo_2axis']}   3-axis >25: {cov['combo_3axis']}",
           transform=a.transAxes, fontsize=9, va="top")
    fig.suptitle(f"{title}  (n={len(recs)})")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def check_landmarks(root, recs, n, path, seed=0):
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(recs), min(n, len(recs)), replace=False)
    tiles = []
    for k in pick:
        r = recs[k]
        im = cv2.imread(os.path.join(root, "faces", r["id"] + ".jpg"))
        d = np.load(os.path.join(root, "faces", r["id"] + ".npz"))
        lm = d["landmarks"]
        for q in lm[::3]:
            cv2.circle(im, (int(q[0]), int(q[1])), 1, (220, 220, 220), -1)
        for nm, c in (("r_eye_out", (0, 0, 255)), ("l_eye_out", (255, 0, 0)), ("mid_eyes", (0, 255, 0)),
                      ("nose_tip", (255, 0, 255)), ("chin", (255, 255, 255)), ("r_face", (0, 0, 128)),
                      ("l_face", (128, 0, 0))):
            q = lm[LM[nm]]
            cv2.circle(im, (int(q[0]), int(q[1])), 5, c, -1)
        cv2.putText(im, f"{r['id']} p{r['pitch']:.0f} y{r['yaw']:.0f} r{r['roll']:.0f} {r.get('method', '')}",
                    (6, 22), 0, 0.6, (0, 255, 0), 2)
        tiles.append(cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA))
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None, help="default <root>/stats.png")
    ap.add_argument("--check", type=int, default=0, help="number of landmark spot checks")
    a = ap.parse_args()
    recs_all = load_index(a.root)
    recs = [r for r in recs_all if r.get("kps_err", 1.0) <= 0.10]    # only count sources usable for training
    print(f"usable {len(recs)} / total {len(recs_all)}")
    out = a.out or os.path.join(a.root, "stats.png")
    plot(recs, out, os.path.basename(os.path.abspath(a.root)))
    print(json.dumps(coverage(recs)))
    print("statistics:", out)
    if a.check:
        p = os.path.join(a.root, "check_landmarks.jpg")
        check_landmarks(a.root, recs, a.check, p)
        print("landmark spot checks:", p)


if __name__ == "__main__":
    main()
