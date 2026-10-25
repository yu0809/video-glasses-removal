#!/usr/bin/env python3
"""
Qualitative figures for the README / blog post (writes JPEGs into docs/images/).

  TOG_DIR=/path/to/take-off-eyeglasses python docs/scripts/make_examples.py

  qualitative_e2e.jpg     end-to-end removal on synthetic test clips (input / rule / glasses-detector / ours / ground truth),
                          frames picked automatically where the methods differ most
  masks_compare.jpg       frame masks of each method on the same crop
  synthetic_samples.jpg   training samples rendered by the 3D glasses synthesizer (with labels)
  real_example.jpg        end-to-end on input/example.mp4 (a real video)

Faces in the synthetic figures come from FFHQ (NVIDIA, CC BY-NC-SA 4.0; individual images under their Flickr licences).
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "images")
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, ROOT)
os.makedirs(OUT, exist_ok=True)

FONT = cv2.FONT_HERSHEY_SIMPLEX
INK = (11, 11, 11)
BG = (251, 252, 252)


def read_video(p):
    cap = cv2.VideoCapture(p)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def header(labels, w, h=34):
    bar = np.full((h, w * len(labels), 3), BG, np.uint8)
    for i, t in enumerate(labels):
        (tw, _), _ = cv2.getTextSize(t, FONT, 0.55, 1)
        cv2.putText(bar, t, (i * w + (w - tw) // 2, 23), FONT, 0.55, INK, 1, cv2.LINE_AA)
    return bar


def side_label(img, text, w=150):
    pad = np.full((img.shape[0], w, 3), BG, np.uint8)
    y = img.shape[0] // 2
    for k, line in enumerate(text.split("\n")):
        cv2.putText(pad, line, (8, y - 10 + 22 * k), FONT, 0.5, INK, 1, cv2.LINE_AA)
    return np.hstack([pad, img])


def gap(img, px=4, axis=1):
    shape = list(img.shape)
    shape[axis] = px
    return np.full(shape, BG, np.uint8)


def hjoin(ims, px=4):
    out = [ims[0]]
    for im in ims[1:]:
        out += [gap(im, px, 1), im]
    return np.hstack(out)


def vjoin(ims, px=4):
    out = [ims[0]]
    for im in ims[1:]:
        out += [gap(im, px, 0), im]
    return np.vstack(out)


# ----------------------------------------------------------------------------
def residue_frac(out, clean, m):
    g = m > 127
    if g.sum() < 20:
        return 0.0
    lo, lc = (cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32) for x in (out, clean))
    return float((np.linalg.norm(lo - lc, axis=2)[g] > 20).mean())


def qualitative_e2e(clips=(("02_0", "Look up / down\n+-40 deg"), ("03_0", "Roll 30 deg\n+ pitch 30 deg"),
                           ("04_0", "Yaw +-60 deg\n+ pitch 15 deg"), ("08_0", "Fast head turn\n(motion blur)")), tile=230):
    rows = []
    for clip, name in clips:
        d = os.path.join(ROOT, "data", "test_videos", clip)
        g, c = read_video(os.path.join(d, "glasses.mp4")), read_video(os.path.join(d, "clean.mp4"))
        outs = {m: read_video(os.path.join(ROOT, "reports", "e2e_r1", clip, m, "glasses_no_glasses.mp4"))
                for m in ("rule", "gd", "net_k05")}
        n = min(len(g), len(c), *[len(v) for v in outs.values()])
        best, bi = -1, 0
        for i in range(0, n, 2):
            m = cv2.imread(os.path.join(d, "mask", f"{i:05d}.png"), 0)
            if m is None or (m > 127).sum() < 50:
                continue
            r = {k: residue_frac(outs[k][i], c[i], m) for k in outs}
            score = (r["rule"] + r["gd"]) / 2 - 2 * r["net_k05"]
            if score > best:
                best, bi = score, i
        m = cv2.imread(os.path.join(d, "mask", f"{bi:05d}.png"), 0)
        ys, xs = np.nonzero(m > 60)
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2 + 0.15 * (ys.max() - ys.min())
        s = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.35)
        H, W = m.shape
        x0, y0 = int(np.clip(cx - s / 2, 0, W - s)), int(np.clip(cy - s / 2, 0, H - s))
        crop = lambda im: cv2.resize(im[y0:y0 + s, x0:x0 + s], (tile, tile), interpolation=cv2.INTER_AREA)  # noqa: E731
        ims = [crop(g[bi]), crop(outs["rule"][bi]), crop(outs["gd"][bi]), crop(outs["net_k05"][bi]), crop(c[bi])]
        rows.append(side_label(hjoin(ims), name))
    head = np.hstack([np.full((34, 150, 3), BG, np.uint8),
                      header(["Input", "Rule-based", "glasses-detector", "Ours", "Ground truth"], tile + 4)[:, :5 * tile + 16]])
    img = vjoin([head] + rows)
    cv2.imwrite(os.path.join(OUT, "qualitative_e2e.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote qualitative_e2e.jpg")


# ----------------------------------------------------------------------------
def masks_compare(clip="03_0", tile=230):
    import baselines
    from common import biggest, crop_face
    from uniface import RetinaFace
    d = os.path.join(ROOT, "data", "test_videos", clip)
    frames = read_video(os.path.join(d, "glasses.mp4"))
    i = len(frames) // 2
    f = frames[i]
    det = RetinaFace()
    c, M, _ = crop_face(f, biggest(det.detect(f)).bbox)
    gt = cv2.warpAffine(cv2.imread(os.path.join(d, "mask", f"{i:05d}.png"), 0), M, (512, 512))
    names = [("ours", "Ours"), ("gd_medium", "glasses-detector"), ("tog", "take-off-eyeglasses"), ("rule", "Rule-based")]
    tiles = []

    def ov(mask):
        o = c.astype(np.float32) * 0.6
        o[..., 1] = np.maximum(o[..., 1], mask * 255)
        o = o.astype(np.uint8)[90:400, 60:452]
        return cv2.resize(o, (tile, int(tile * 310 / 392)), interpolation=cv2.INTER_AREA)
    tiles.append(ov((gt > 127).astype(np.float32)))
    for key, _ in names:
        p = baselines.build([key], os.path.join(ROOT, "weights", "frame_seg_r1.pt"))[0]([c])[0]
        tiles.append(ov((p > 0.5).astype(np.float32)))
    img = vjoin([header(["Ground truth"] + [n for _, n in names], tile + 4)[:, :5 * tile + 16], hjoin(tiles)])
    cv2.imwrite(os.path.join(OUT, "masks_compare.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote masks_compare.jpg")


# ----------------------------------------------------------------------------
def synthetic_samples(tile=200, seed=11):
    from synth.glasses import sample_params
    from synth.pipeline import load_index, make_sample
    pool = os.path.join(ROOT, "data", "faces_pool")
    recs = [r for r in load_index(pool) if r.get("kps_err", 1.0) <= 0.10]
    by_id = {r["id"]: r for r in recs}

    def root(r):
        while r.get("base_id") in by_id and r["base_id"] != r["id"]:
            r = by_id[r["base_id"]]
        return r.get("dataset")
    # real FFHQ photos and LivePortrait re-posed FFHQ faces only (mesh re-projection can leave hair artefacts)
    ffhq = [r for r in recs if root(r) == "ffhq512" and r.get("method", "real") in ("real", "liveportrait")]
    rng = np.random.default_rng(seed)
    targets = [(35, -35, 0), (-35, 25, 12), (5, 50, -8), (20, 5, -25), (-30, -15, 0), (10, -50, 15)]
    styles = ["full", "acetate", "half", "full", "rimless", "acetate"]
    top, bot = [], []
    for (tp, ty, tr), st in zip(targets, styles):
        cand = sorted(ffhq, key=lambda r: (r["pitch"] - tp) ** 2 + (r["yaw"] - ty) ** 2)[:15]
        rec = cand[int(rng.integers(len(cand)))]
        s = make_sample(pool, rec, rng, size=384, roll_target=tr, flip=False, negative=False,
                        params=sample_params(rng, st), quality=False, distort=False, cam_shift=0.0)
        im = cv2.resize(s["image"], (tile, tile), interpolation=cv2.INTER_AREA)
        o = s["image"].astype(np.float32) * 0.55
        o[..., 1] = np.maximum(o[..., 1], s["frame"] * 255)
        o[..., 2] = np.maximum(o[..., 2], s["lens"] * 140)
        top.append(im)
        bot.append(cv2.resize(o.astype(np.uint8), (tile, tile), interpolation=cv2.INTER_AREA))
    img = vjoin([side_label(hjoin(top), "Rendered\nsample", 110), side_label(hjoin(bot), "Labels\ngreen=frame\nred=lens", 110)])
    cv2.imwrite(os.path.join(OUT, "synthetic_samples.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote synthetic_samples.jpg")


# ----------------------------------------------------------------------------
def real_example(times=(1.0, 4.0), tile=260):
    src = read_video(os.path.join(ROOT, "input", "example.mp4"))
    outs = {m: read_video(os.path.join(ROOT, "reports", "real", m, "example_no_glasses.mp4"))
            for m in ("rule", "gd", "net_k0.5")}
    rows = []
    for t in times:
        i = int(t * 30)
        ims = [src[i], outs["rule"][i], outs["gd"][i], outs["net_k0.5"][i]]
        h, w = ims[0].shape[:2]
        crop = lambda im: cv2.resize(im[int(h * 0.18):int(h * 0.62), int(w * 0.12):int(w * 0.88)],  # noqa: E731
                                     (tile, int(tile * 0.44 * h / (0.76 * w))), interpolation=cv2.INTER_AREA)
        rows.append(hjoin([crop(x) for x in ims]))
    img = vjoin([header(["Input (real video)", "Rule-based", "glasses-detector", "Ours"], tile + 4)[:, :4 * tile + 12]] + rows)
    cv2.imwrite(os.path.join(OUT, "real_example.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote real_example.jpg")


if __name__ == "__main__":
    which = sys.argv[1:] or ["e2e", "masks", "synth", "real"]
    if "e2e" in which:
        qualitative_e2e()
    if "masks" in which:
        masks_compare()
    if "synth" in which:
        synthetic_samples()
    if "real" in which:
        real_example()
