#!/usr/bin/env python3
"""
Multi-method comparison on real / third-party data
==================================================
Data (training data for none of the methods):
  celeba   real photos with glasses from CelebAMask-HQ (manual eye_g labels) + photos without glasses (false positives).
           The training pool excluded every photo with glasses, so these people are unseen for this project as well.
           eye_g covers the whole glasses (lenses included) and is coarse, so the metrics are:
             precision P = share of predicted frame pixels inside eye_g (dilated by tol)
             outline recall R = share of the eye_g outer contour (~ outer edge of the frame) within tol of the prediction; F = 2PR/(P+R)
  fsynth   Microsoft Face Synthetics 1,000-image sample (someone else's synthetic renders): with clear lenses the GLASSES class is the frame only,
           so boundary F1 (2 px) / IoU are computed directly; opaque lenses (sunglasses) are a separate group.
  fsynth_* any other extracted part of Face Synthetics (prepare_fsynth), reported by head pose. The pose comes from the dataset's
           exact 2D landmarks: pitch / yaw by PnP against a 68-point mean face, roll from the pupil line (the HeadPose network
           is unreliable at extreme angles).
Perturbations (identical for all methods): orig, roll30 (tilt +-30 deg), small (distant face shrunk to ~50 px and upscaled), mblur (motion blur).
Every method gets the same pipeline crop (512) and outputs a frame probability; > 0.5 counts as frame.

  python training/eval_real.py prepare --out data/eval_cache          # once: detect, crop, cache
  python training/eval_real.py run --cache data/eval_cache --ckpt weights/frame_seg_r1.pt --out reports/compare
  python training/eval_real.py prepare_fsynth --src data/raw/face_synthetics/p1/dataset_100000 --name fsynth_p1 --out data/eval_cache
  python training/eval_real.py run --cache data/eval_cache --data fsynth_p1 --perturb orig --ckpt weights/frame_seg.pt --out reports/fsynth_p1
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
from common import biggest, crop_face, load_models  # noqa: E402
from metrics import seg_metrics  # noqa: E402

CELEBA = os.path.join(ROOT, "data", "raw", "celebamask_hq", "CelebAMask-HQ")
FSYN = os.path.join(ROOT, "data", "raw", "face_synthetics", "dataset_1000")
PERTURB = ["orig", "roll30", "small", "mblur"]
METHODS = ["ours", "rule", "gd_small", "gd_medium", "tog"]


# ----------------------------------------------------------------------------
# preparation: detection + pipeline crop + the same transform for the labels, cached
# ----------------------------------------------------------------------------
def celeba_lists(n_neg):
    lines = open(os.path.join(CELEBA, "CelebAMask-HQ-attribute-anno.txt")).read().splitlines()
    names = lines[1].split()
    gi = names.index("Eyeglasses")
    pos, neg = [], []
    for l in lines[2:]:
        p = l.split()
        idx = int(p[0].split(".")[0])
        (pos if p[1 + gi] == "1" else neg).append(idx)
    rng = np.random.default_rng(0)
    return pos, sorted(rng.choice(neg, n_neg, replace=False).tolist())


def eye_g_mask(idx):
    f = os.path.join(CELEBA, "CelebAMask-HQ-mask-anno", str(idx // 2000), f"{idx:05d}_eye_g.png")
    if not os.path.exists(f):
        return None
    m = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    return (cv2.resize(m, (1024, 1024), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)


def cmd_prepare(a):
    os.makedirs(a.out, exist_ok=True)
    m = load_models(["CPUExecutionProvider"], parsing=False)
    out = dict(celeba_pos=[], celeba_neg=[], fsynth=[])
    pos, neg = celeba_lists(a.n_neg)
    for kind, ids in (("celeba_pos", pos[:a.max_pos]), ("celeba_neg", neg)):
        crops, masks, poses, keep = [], [], [], []
        for k, idx in enumerate(ids):
            img = cv2.imread(os.path.join(CELEBA, "CelebA-HQ-img", f"{idx}.jpg"))
            gm = eye_g_mask(idx) if kind == "celeba_pos" else np.zeros(img.shape[:2], np.uint8)
            if gm is None or (kind == "celeba_pos" and gm.sum() < 200):
                continue
            small = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            fs = m["det"].detect(small)
            if not fs:
                continue
            bb = np.asarray(biggest(fs).bbox[:4], np.float64) * 2
            c, M, _ = crop_face(img, bb)
            mk = cv2.warpAffine(gm, M, (512, 512), flags=cv2.INTER_NEAREST, borderValue=0)
            cf = m["det"].detect(c)
            pose = [0, 0, 0]
            if cf:
                bx = [int(max(0, v)) for v in biggest(cf).bbox[:4]]
                hp = m["hp"].estimate(c[bx[1]:bx[3], bx[0]:bx[2]])
                pose = [hp.pitch, hp.yaw, hp.roll]
            crops.append(c)
            masks.append(mk)
            poses.append(pose)
            keep.append(idx)
            if (k + 1) % 200 == 0:
                print(f"  {kind} {k + 1}/{len(ids)}", flush=True)
        np.savez_compressed(os.path.join(a.out, kind + ".npz"), crops=np.stack(crops), masks=np.stack(masks),
                            poses=np.array(poses, np.float32), ids=np.array(keep))
        print(f"{kind}: {len(crops)} images")
    # Face Synthetics
    prep_fsynth(m, FSYN, [f"{i:06d}" for i in range(1000)], os.path.join(a.out, "fsynth.npz"))


# 68-point mean face (iBUG order) from github.com/yinguobing/head-pose-estimation (MIT)
MEAN68 = np.loadtxt(os.path.join(HERE, "assets", "mean_face_68.txt")).reshape(3, -1).T


def landmark_pose(L, w):
    """(pitch, yaw, roll) in the HeadPose convention from Face Synthetics' exact 2D landmarks (70 points, iBUG 68 + pupils).
    pitch / yaw: PnP against the mean face (calibrated against HeadPose on the 1,000-image sample: corr. 0.85 / 0.94);
    roll: angle of the pupil line, which stays exact at large pitch where the Euler roll of the PnP solution flips."""
    K = np.array([[w, 0, w / 2], [0, w, w / 2], [0, 0, 1]], np.float64)
    _, rv, _ = cv2.solvePnP(MEAN68, L[:68].astype(np.float64), K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    ex, ey, _ = cv2.RQDecomp3x3(cv2.Rodrigues(rv)[0])[0]
    a, b = sorted([L[68], L[69]], key=lambda p: p[0])
    return [-ex, ey, float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))]


def prep_fsynth(m, src, names, out):
    crops, masks, kinds, poses, poses_lm, keep = [], [], [], [], [], []
    no_face = 0
    for k, nm in enumerate(names):
        seg = cv2.imread(os.path.join(src, f"{nm}_seg.png"), cv2.IMREAD_GRAYSCALE)
        if seg is None:
            continue
        g = (seg == 16).astype(np.uint8)
        if g.sum() < 100:
            continue
        img = cv2.imread(os.path.join(src, f"{nm}.png"))
        if img is None:
            continue
        fs = m["det"].detect(img)
        if not fs:
            no_face += 1
            continue
        c, M, _ = crop_face(img, biggest(fs).bbox)
        mk = cv2.warpAffine(g, M, (512, 512), flags=cv2.INTER_NEAREST, borderValue=0)
        if mk.sum() < 50:
            continue
        # opaque lenses (sunglasses): large solid regions in the label
        solid = float((cv2.distanceTransform(mk, cv2.DIST_L2, 3) > 4).sum()) / max(1, mk.sum())
        cf = m["det"].detect(c)
        pose = [0, 0, 0]
        if cf:
            bx = [int(max(0, v)) for v in biggest(cf).bbox[:4]]
            hp = m["hp"].estimate(c[bx[1]:bx[3], bx[0]:bx[2]])
            pose = [hp.pitch, hp.yaw, hp.roll]
        lf = os.path.join(src, f"{nm}_ldmks.txt")
        crops.append(c)
        masks.append(mk)
        kinds.append("opaque" if solid > 0.3 else "clear")
        poses.append(pose)
        poses_lm.append(landmark_pose(np.loadtxt(lf), img.shape[1]) if os.path.exists(lf) else [np.nan] * 3)
        keep.append(int(nm))
        if (k + 1) % 2000 == 0:
            print(f"  {k + 1}/{len(names)}, kept {len(crops)}", flush=True)
    np.savez_compressed(out, crops=np.stack(crops), masks=np.stack(masks), kinds=np.array(kinds),
                        poses=np.array(poses, np.float32), poses_lm=np.array(poses_lm, np.float32), ids=np.array(keep))
    print(f"{os.path.basename(out)}: {len(crops)} images (clear lenses {kinds.count('clear')}, opaque {kinds.count('opaque')}); "
          f"{no_face} images with glasses skipped because no face was detected")


def cmd_prepare_fsynth(a):
    os.makedirs(a.out, exist_ok=True)
    m = load_models(["CPUExecutionProvider"], parsing=False)
    names = sorted(f[:-8] for f in os.listdir(a.src) if f.endswith("_seg.png"))
    prep_fsynth(m, a.src, names, os.path.join(a.out, a.name + ".npz"))


# ----------------------------------------------------------------------------
# perturbations (image and labels together)
# ----------------------------------------------------------------------------
def perturb(img, mask, kind, i):
    if kind == "orig":
        return img, mask
    if kind == "roll30":
        ang = 30 if i % 2 == 0 else -30
        R = cv2.getRotationMatrix2D((256, 256), ang, 1.0)
        return (cv2.warpAffine(img, R, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT),
                cv2.warpAffine(mask, R, (512, 512), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT))
    if kind == "small":                 # the face is ~394 px in the crop; shrinking to 64 px equals a ~50 px face in the original video
        s = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        return cv2.resize(s, (512, 512), interpolation=cv2.INTER_CUBIC), mask
    if kind == "mblur":
        L = 15
        k = np.zeros((L, L), np.float32)
        k[L // 2, :] = 1
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((L // 2, L // 2), (i * 37) % 180, 1.0), (L, L))
        return cv2.filter2D(img, -1, k / k.sum()), mask
    raise ValueError(kind)


def celeba_metrics(prob, gm, tol=6):
    p = (prob > 0.5).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    zone = cv2.dilate(gm, k)
    contours, _ = cv2.findContours(gm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge = np.zeros_like(gm)
    cv2.drawContours(edge, contours, -1, 1, 1)
    npred = int(p.sum())
    prec = float((p & zone).sum()) / npred if npred else 0.0
    rec = float((edge & cv2.dilate(p, k)).sum()) / max(1, int(edge.sum()))
    return dict(P=prec, R=rec, F=2 * prec * rec / max(1e-9, prec + rec), detected=npred > 50)


def cmd_run(a):
    import baselines
    os.makedirs(a.out, exist_ok=True)
    data = {k: dict(np.load(os.path.join(a.cache, k + ".npz"))) for k in a.data}
    rows = []
    for meth in baselines.build(a.methods, a.ckpt):
        print(f"== {meth.name}", flush=True)
        for ds in a.data:
            D = data[ds]
            n = min(len(D["crops"]), a.max_n)
            pert = [p for p in a.perturb if ds != "celeba_neg" or p in ("orig", "small")]
            for pk in pert:
                for k0 in range(0, n, 32):
                    idx = list(range(k0, min(n, k0 + 32)))
                    pairs = [perturb(D["crops"][i], D["masks"][i], pk, i) for i in idx]
                    probs = meth([p[0] for p in pairs])
                    for i, (im, mk), pr in zip(idx, pairs, probs):
                        r = dict(method=meth.name, data=ds, perturb=pk, i=i, pitch=float(D["poses"][i][0]),
                                 yaw=float(D["poses"][i][1]), roll=float(D["poses"][i][2]))
                        if "poses_lm" in D:
                            r.update(lm_pitch=float(D["poses_lm"][i][0]), lm_yaw=float(D["poses_lm"][i][1]),
                                     lm_roll=float(D["poses_lm"][i][2]), id=int(D["ids"][i]))
                        if ds == "celeba_pos":
                            r.update(celeba_metrics(pr, mk))
                        elif ds == "celeba_neg":
                            r.update(fp=float((pr > 0.5).mean()))
                        else:
                            m = seg_metrics(pr, mk.astype(np.float32))
                            r.update(bf1=m["bf1"], iou=m["iou"], kind=str(D["kinds"][i]))
                        rows.append(r)
            print(f"   {ds} done", flush=True)
        if a.examples and "celeba_pos" in data:
            save_examples(meth, data, a.out)
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(os.path.join(a.out, "per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    report(rows, a.out)


# pose buckets for the Face Synthetics parts (landmark pose; "beyond" = |angle| >= the threshold of that axis)
HARD = dict(pitch=25, yaw=45, roll=25)


def pose_buckets(r):
    p, y, ro = abs(r["lm_pitch"]), abs(r["lm_yaw"]), abs(r["lm_roll"])
    n = (p >= 25) + (y >= 25) + (ro >= 25)
    return {
        "near frontal (all axes < 15°)": p < 15 and y < 15 and ro < 15,
        "pitch beyond ±25° (look up / down)": p >= HARD["pitch"],
        "pitch beyond ±40°": p >= 40,
        "yaw beyond ±45° (head turn)": y >= HARD["yaw"],
        "roll beyond ±25° (tilt)": ro >= HARD["roll"],
        "two or more axes beyond ±25° (combined)": n >= 2,
        "any hard pose (union of the above)": p >= HARD["pitch"] or y >= HARD["yaw"] or ro >= HARD["roll"] or n >= 2,
        "all": True,
    }


def report_pose(rows, ds, meths, pert):
    L = [f"## {ds}: Microsoft Face Synthetics by head pose (clear lenses, boundary F1 2 px; frame-detected share in brackets)", "",
         "Pose from the dataset's exact landmarks. A sample with no predicted frame pixels counts as F1 = 0.", ""]
    for pk in pert:
        L += [f"### perturbation: {pk}", "", "| Pose bucket | n | " + " | ".join(meths) + " |", "|---|---|" + "---|" * len(meths)]
        by_m = {m: [r for r in rows if r["method"] == m and r["data"] == ds and r["perturb"] == pk and r["kind"] == "clear"] for m in meths}
        for b in pose_buckets(by_m[meths[0]][0]):
            cells, n = [], 0
            for m in meths:
                s = [r for r in by_m[m] if pose_buckets(r)[b]]
                n = len(s)
                cells.append(f"{np.mean([r['bf1'] for r in s]):.3f} ({np.mean([r['iou'] > 0 for r in s]):.0%})" if s else "-")
            L.append(f"| {b} | {n} | " + " | ".join(cells) + " |")
        L.append("")
    op = {m: [r["bf1"] for r in rows if r["method"] == m and r["data"] == ds and r["perturb"] == "orig" and r["kind"] == "opaque"] for m in meths}
    if any(op.values()):
        L += ["Opaque lenses (orig, frame + lens labelled as glasses): " +
              ", ".join(f"{m} {np.mean(v):.3f}" for m, v in op.items() if v) + f" (n = {len(op[meths[0]])})", ""]
    return L


def report(rows, out):
    meths = list(dict.fromkeys(r["method"] for r in rows))
    present = set(r["data"] for r in rows)
    pert = list(dict.fromkeys(r["perturb"] for r in rows))
    g = defaultdict(list)
    for r in rows:
        g[(r["method"], r["data"], r["perturb"])].append(r)
    L = ["# Multi-method comparison: real photos and third-party synthetic data", ""]
    for ds in sorted(d for d in present if d.startswith("fsynth_")):
        L += report_pose(rows, ds, meths, pert)
    if not {"celeba_pos", "celeba_neg", "fsynth"} <= present:
        open(os.path.join(out, "report.md"), "w").write("\n".join(L) + "\n")
        print("\n".join(L))
        return
    L += ["## CelebAMask-HQ real photos with glasses (F = harmonic mean of precision and outline recall, 6 px tolerance)", "",
          "| Method | " + " | ".join(f"{p} F (P/R)" for p in PERTURB) + " | Detection rate (orig) |",
          "|---|" + "---|" * (len(PERTURB) + 1)]
    for m in meths:
        cells = []
        for p in PERTURB:
            rr = g[(m, "celeba_pos", p)]
            cells.append(f"{np.mean([r['F'] for r in rr]):.3f} ({np.mean([r['P'] for r in rr]):.2f}/{np.mean([r['R'] for r in rr]):.2f})")
        det = np.mean([r["detected"] for r in g[(m, "celeba_pos", "orig")]])
        L.append(f"| {m} | " + " | ".join(cells) + f" | {det:.0%} |")
    L += ["", "By pose (orig, HeadPose estimate):", "", "| Method | |yaw|<15 | 15-30 | 30+ | |pitch|<10 | 10-20 | 20+ |", "|---|---|---|---|---|---|---|"]
    for m in meths:
        rr = g[(m, "celeba_pos", "orig")]
        def fm(sel):
            s = [r["F"] for r in rr if sel(r)]
            return f"{np.mean(s):.3f} (n={len(s)})" if s else "-"
        L.append(f"| {m} | {fm(lambda r: abs(r['yaw']) < 15)} | {fm(lambda r: 15 <= abs(r['yaw']) < 30)} | {fm(lambda r: abs(r['yaw']) >= 30)} | "
                 f"{fm(lambda r: abs(r['pitch']) < 10)} | {fm(lambda r: 10 <= abs(r['pitch']) < 20)} | {fm(lambda r: abs(r['pitch']) >= 20)} |")
    L += ["", "## CelebAMask-HQ photos without glasses: false-positive area (lower is better)", "", "| Method | orig | small |", "|---|---|---|"]
    for m in meths:
        L.append(f"| {m} | " + " | ".join(f"{100 * np.mean([r['fp'] for r in g[(m, 'celeba_neg', p)]]):.3f}%" for p in ("orig", "small")) + " |")
    L += ["", "## Microsoft Face Synthetics (clear lenses: the GLASSES class is the frame; boundary F1 2 px)", "",
          "| Method | " + " | ".join(f"{p}" for p in PERTURB) + " | Opaque lenses (orig) |", "|---|" + "---|" * (len(PERTURB) + 1)]
    for m in meths:
        cells = [f"{np.mean([r['bf1'] for r in g[(m, 'fsynth', p)] if r['kind'] == 'clear']):.3f}" for p in PERTURB]
        op = [r["bf1"] for r in g[(m, "fsynth", "orig")] if r["kind"] == "opaque"]
        L.append(f"| {m} | " + " | ".join(cells) + f" | {np.mean(op):.3f} |" if op else " | - |")
    open(os.path.join(out, "report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def save_examples(meth, data, out, n=6):
    D = data["celeba_pos"]
    rng = np.random.default_rng(3)
    idx = rng.choice(len(D["crops"]), n, replace=False)
    tiles = []
    for pk in PERTURB:
        pairs = [perturb(D["crops"][i], D["masks"][i], pk, int(i)) for i in idx]
        probs = meth([p[0] for p in pairs])
        row = []
        for (im, mk), pr in zip(pairs, probs):
            o = im.astype(np.float32) * 0.55
            o[..., 1] = np.maximum(o[..., 1], (pr > 0.5) * 255)
            cnt, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            o = o.astype(np.uint8)
            cv2.drawContours(o, cnt, -1, (0, 0, 255), 1)
            row.append(cv2.resize(o, (200, 200), interpolation=cv2.INTER_AREA))
        tiles.append(np.hstack(row))
    cv2.imwrite(os.path.join(out, f"examples_{meth.name}.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--n_neg", type=int, default=500)
    p.add_argument("--max_pos", type=int, default=2000)
    p = sub.add_parser("prepare_fsynth")
    p.add_argument("--src", required=True, help="folder with extracted Face Synthetics files (NNNNNN.png, _seg.png, _ldmks.txt)")
    p.add_argument("--name", required=True, help="cache name, must start with fsynth_")
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--cache", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--max_n", type=int, default=100000)
    p.add_argument("--examples", action="store_true")
    p.add_argument("--data", nargs="+", default=["celeba_pos", "celeba_neg", "fsynth"])
    p.add_argument("--perturb", nargs="+", default=PERTURB, choices=PERTURB)
    a = ap.parse_args()
    dict(prepare=cmd_prepare, prepare_fsynth=cmd_prepare_fsynth, run=cmd_run)[a.cmd](a)


if __name__ == "__main__":
    main()
