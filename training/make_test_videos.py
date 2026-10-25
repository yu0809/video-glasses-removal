#!/usr/bin/env python3
"""
Synthetic test-video benchmark: "videos with glasses" that have a ground truth
=============================================================================
Each clip has one fixed pair of glasses (same parameters throughout), rendered frame by frame at the current pose; outputs:
  data/test_videos/<scene>_<k>/glasses.mp4   input (with glasses)
                             clean.mp4     ground truth (without glasses)
                             mask/00000.png per-frame frame labels (soft, 0-255); lens/ lenses
                             meta.json     fps, per-frame target pose, glasses parameters, scene description
Base footage: LivePortrait turns the heads of validation-set people (unseen in training) continuously; frontal talking-head clips from input/smoke_faces.
Fixed random seed; generated once and stored so every evaluation uses the same set.

  python training/make_test_videos.py --pool data/faces_pool --out data/test_videos            # all scenes
  python training/make_test_videos.py --pool data/faces_pool --out data/test_videos --scenes 3 5 --clips 1

Scenes (--clips clips per scene, default 3):
  01 frontal talking (thin metal / thick acetate / rimless / half rim)   02 slowly look up to +40 deg, then down to -40 deg
  03 roll 30 deg while looking down 30 deg / up 30 deg                 04 turn to +-60 deg with 15 deg pitch
  05 all three axes changing slowly ("figure 8")                      06 distant tiny face 30-60 px, moving
  07 dolly zoom: face 60 px -> 400 px                                  08 fast head turn (motion blur from temporal supersampling)
  09 multi-camera cuts: same person from 3 angles + one empty shot      10 lighting: back light / low light with heavy noise / strong lens reflections
  11 partial occlusion: hand passing in front of the glasses / hair over the temples
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from common import biggest, crop_face, load_models, mesh_pose  # noqa: E402
from dataset import load_recs, split_ids  # noqa: E402
from repose.mesh3d import kps5_from_mesh, transfer_landmarks  # noqa: E402
from synth import augment as aug  # noqa: E402
from synth.glasses import sample_params  # noqa: E402
from synth.render import render_sample  # noqa: E402

FPS = 25
SCENES = {
    1: "Frontal talking", 2: "Look up / down", 3: "Roll + pitch", 4: "Yaw + pitch", 5: "Three axes (figure 8)", 6: "Tiny face",
    7: "Dolly zoom", 8: "Fast head turn", 9: "Multi-camera", 10: "Lighting", 11: "Occlusion",
}
STYLES4 = ["full", "acetate", "rimless", "half"]


# ----------------------------------------------------------------------------
# per-frame landmarks (re-posed footage: transfer the base-image landmarks when FaceMesh is untrustworthy)
# ----------------------------------------------------------------------------
class Landmarker:
    def __init__(self):
        self.m = load_models(["CPUExecutionProvider"], parsing=True)

    def __call__(self, frame, base=None):
        """Returns (landmarks (468,3) in frame coordinates, parsing (frame size), source); (None, None, None) if nothing is detected."""
        m = self.m
        fs = m["det"].detect(frame)
        if not fs:
            return None, None, None
        crop, M, _ = crop_face(frame, biggest(fs).bbox)
        cf = m["det"].detect(crop)
        if not cf:
            return None, None, None
        cf = biggest(cf)
        kps = np.asarray(cf.landmarks, np.float32)
        iod = max(float(np.linalg.norm(kps[1] - kps[0])), 0.25 * float(cf.bbox[2] - cf.bbox[0]), 1.0)
        lm = m["mesh"].predict(crop, [cf])[0].landmarks.astype(np.float32)
        x1, y1, x2, y2 = [int(max(0, v)) for v in cf.bbox[:4]]
        hp = m["hp"].estimate(crop[y1:y2, x1:x2])
        pose = np.array([hp.pitch, hp.yaw, hp.roll], np.float32)
        kerr = float(np.linalg.norm(kps5_from_mesh(lm) - kps, axis=1).mean() / iod)
        src = "facemesh"
        if (kerr > 0.10 or np.abs(pose - mesh_pose(lm)).max() > 20) and base is not None:
            lm_t, e = transfer_landmarks(base["landmarks"], pose - base["pose"], kps)
            if e / iod <= 0.10:
                lm, src = lm_t, "transfer"
        lab = m["par"].parse(crop)
        Minv = cv2.invertAffineTransform(M)
        lm_f = lm.copy()
        lm_f[:, :2] = lm[:, :2] @ Minv[:, :2].T + Minv[:, 2]
        lm_f[:, 2] = lm[:, 2] / M[0, 0]
        par = cv2.warpAffine(lab.astype(np.uint8), Minv, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_NEAREST,
                             borderValue=0)
        return lm_f, par, src


def smooth_landmarks(lms, sigma=1.0):
    """Mild temporal smoothing of per-frame landmarks (no jittering glasses); frames where detection failed are interpolated."""
    ok = np.array([l is not None for l in lms])
    if not ok.any():
        return lms
    idx = np.arange(len(lms))
    A = np.stack([l if l is not None else np.zeros((468, 3), np.float32) for l in lms])
    for j in range(468):
        for c in range(3):
            A[~ok, j, c] = np.interp(idx[~ok], idx[ok], A[ok, j, c])
    return list(gaussian_filter1d(A, sigma, axis=0, mode="nearest")) if sigma > 0 else list(A)


# ----------------------------------------------------------------------------
# base footage
# ----------------------------------------------------------------------------
def lp_frames(lp, base_img, base_rec, poses, margin=40, sub=1):
    """LivePortrait generates consecutive frames for a target (pitch, yaw) sequence in the HeadPose convention. With sub > 1, sub-1 sub-frames are inserted between frames (for motion blur)."""
    st = lp.prepare(base_img)
    p0 = np.array([base_rec["pitch"], base_rec["yaw"], 0.0])
    out = []
    for k in range(len(poses)):
        a = np.array(poses[k], float)
        b = np.array(poses[min(k + 1, len(poses) - 1)], float)
        subs = []
        for j in range(sub):
            t = a + (b - a) * j / sub
            subs.append(lp.render(st, (np.array([t[0], t[1], 0.0]) - p0), margin=margin))
        out.append(subs)
    return out


def ease_path(keys, n):
    """Ease-in / ease-out interpolation between key poses, n frames in total."""
    keys = np.asarray(keys, float)
    seg = len(keys) - 1
    out = []
    for i in range(n):
        u = i / max(1, n - 1) * seg
        k = min(int(u), seg - 1)
        t = u - k
        t = 0.5 - 0.5 * np.cos(np.pi * t)
        out.append(keys[k] * (1 - t) + keys[k + 1] * t)
    return np.array(out)


def video_frames(path, n, start=0):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    fr = []
    while len(fr) < n:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    return fr


# ----------------------------------------------------------------------------
# frame post-processing (applied to the frames with glasses, the clean frames and the labels alike)
# ----------------------------------------------------------------------------
def rotate_all(arrs, center, ang):
    """Rotate clockwise by ang degrees (roll increases by ang)."""
    R = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), -ang, 1.0)
    h, w = arrs[0].shape[:2]
    return [cv2.warpAffine(a, R, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) for a in arrs]


def place_on_canvas(arrs, scale, center_xy, canvas=(960, 540), bg=None):
    """Scale the whole frame by scale and place it at center_xy on the canvas (distant / dolly shots). The background is a heavily blurred upscaled copy of the same frame,
    the foreground edge is feathered (avoids mirrored fake faces). Label layers have a 0 background."""
    W, H = canvas
    h, w = arrs[0].shape[:2]
    A = np.float32([[scale, 0, center_xy[0] - scale * w / 2], [0, scale, center_xy[1] - scale * h / 2]])
    feather = np.zeros((h, w), np.float32)
    cv2.ellipse(feather, (w // 2, h // 2), (int(w * 0.46), int(h * 0.48)), 0, 0, 360, 1.0, -1)
    feather = cv2.GaussianBlur(feather, (0, 0), w * 0.04)
    fa = cv2.warpAffine(feather, A, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
    out = []
    for k, a in enumerate(arrs):
        fg = cv2.warpAffine(a, A, (W, H), flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR, borderValue=0)
        if a.ndim == 3:
            b = bg if bg is not None else cv2.GaussianBlur(cv2.resize(arrs[0], (W, H)), (0, 0), 25)
            out.append((fg.astype(np.float32) * fa + b.astype(np.float32) * (1 - fa)).astype(np.uint8))
        else:
            out.append(fg * fa[..., 0])
    return out


def draw_hand(shape, t, skin_bgr, rng_seed, face_box):
    """A simplified hand (palm + four fingers) sweeping from left to right across the glasses; t in [0,1]. Returns (colour, alpha)."""
    h, w = shape
    x0, y0, x1, y1 = face_box
    fw = x1 - x0
    cx = x0 - 0.6 * fw + t * 2.2 * fw
    cy = y0 + 0.35 * (y1 - y0)
    a = np.zeros((h, w), np.float32)
    cv2.ellipse(a, (int(cx), int(cy + 0.25 * fw)), (int(0.28 * fw), int(0.33 * fw)), 15, 0, 360, 1.0, -1)
    for k in range(4):
        fx = cx - 0.18 * fw + k * 0.12 * fw
        cv2.line(a, (int(fx), int(cy + 0.1 * fw)), (int(fx + 0.05 * fw), int(cy - 0.35 * fw)), 1.0, max(2, int(0.09 * fw)))
    a = cv2.GaussianBlur(a, (0, 0), max(1.0, 0.01 * fw))
    rng = np.random.default_rng(rng_seed)
    shade = cv2.resize(rng.random((6, 6)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    col = np.clip(skin_bgr[None, None, :] * (0.8 + 0.3 * shade[..., None]), 0, 255)
    return col, a


# ----------------------------------------------------------------------------
# one clip
# ----------------------------------------------------------------------------
class ClipWriter:
    def __init__(self, out_dir, size):
        os.makedirs(os.path.join(out_dir, "mask"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "lens"), exist_ok=True)
        self.dir, self.n = out_dir, 0
        w, h = size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.g = cv2.VideoWriter(os.path.join(out_dir, "glasses_raw.mp4"), fourcc, FPS, (w, h))
        self.c = cv2.VideoWriter(os.path.join(out_dir, "clean_raw.mp4"), fourcc, FPS, (w, h))

    def write(self, g, c, mask, lens):
        self.g.write(g)
        self.c.write(c)
        cv2.imwrite(os.path.join(self.dir, "mask", f"{self.n:05d}.png"), np.round(np.clip(mask, 0, 1) * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(self.dir, "lens", f"{self.n:05d}.png"), np.round(np.clip(lens, 0, 1) * 255).astype(np.uint8))
        self.n += 1

    def close(self):
        self.g.release()
        self.c.release()
        for k in ("glasses", "clean"):    # convert to H.264 (ffmpeg), near lossless
            src, dst = os.path.join(self.dir, f"{k}_raw.mp4"), os.path.join(self.dir, f"{k}.mp4")
            os.system(f'ffmpeg -y -loglevel error -i "{src}" -c:v libx264 -crf 14 -pix_fmt yuv420p "{dst}" && rm "{src}"')


def render_clip(frames, lms, pars, params, seed):
    """Render the same glasses every frame (random parts use a fixed seed for frame-to-frame consistency)."""
    outs = []
    for f, lm, par in zip(frames, lms, pars):
        if lm is None:
            z = np.zeros(f.shape[:2], np.float32)
            outs.append(dict(image=f.copy(), clean=f, frame_mask=z, lens_mask=z))
            continue
        outs.append(render_sample(f, lm, par, np.random.default_rng(seed), params=params))
    return outs


def make_clip(scene, k, a, ctx):
    rng = np.random.default_rng(1000 * scene + k)
    seed = 1000 * scene + k
    name = f"{scene:02d}_{k}"
    out_dir = os.path.join(a.out, name)
    if os.path.exists(os.path.join(out_dir, "meta.json")):
        print(f"  {name} exists, skipping")
        return
    lmk, lp, bases, pool = ctx["lmk"], ctx["lp"], ctx["bases"], a.pool
    base = bases[(scene * 7 + k) % len(bases)]
    base_img = cv2.imread(os.path.join(pool, "faces", base["id"] + ".jpg"))
    base_npz = dict(np.load(os.path.join(pool, "faces", base["id"] + ".npz")))
    style = STYLES4[k % 4] if scene == 1 else None
    params = sample_params(rng, style)
    dur = rng.uniform(3, 6)
    n = int(dur * FPS)
    meta = dict(scene=scene, scene_name=SCENES[scene], clip=k, fps=FPS, base_id=base["id"], style=params["style"],
                color=params["color_name"])
    post = dict(roll=np.zeros(n), scale=None, center=None, light=None, hand=False)
    sub = 1
    if scene == 1:
        vids = sorted(os.listdir(os.path.join(ROOT, "input", "smoke_faces")))
        v = os.path.join(ROOT, "input", "smoke_faces", vids[k % len(vids)])
        frames = [[f] for f in video_frames(v, n, start=int(rng.integers(0, 30)))]
        poses = None
        meta["source"] = os.path.basename(v)
    else:
        if scene == 2:
            keys = [(0, 0), (40, 0), (-40, 0), (0, 0)]
        elif scene == 3:
            sg = 1 if k % 2 == 0 else -1          # look down / look up
            keys = [(0, 0), (-30 * sg, 0), (-30 * sg, 0), (0, 0)]
            post["roll"] = ease_path([[0], [30 * (1 if k < 2 else -1)], [30 * (1 if k < 2 else -1)], [0]], n)[:, 0]
        elif scene == 4:
            sy = 1 if k % 2 == 0 else -1
            keys = [(15, 0), (15, 60 * sy), (-15, -60 * sy), (15, 0)]
        elif scene == 5:
            tt = np.linspace(0, 2 * np.pi, n)
            poses = np.stack([25 * np.sin(2 * tt), 40 * np.sin(tt)], 1)
            post["roll"] = 25 * np.sin(tt + 0.7)
            keys = None
        elif scene in (6, 7, 10, 11):
            keys = [(0, -20), (10, 20), (-10, 0)]
        elif scene == 8:
            keys = [(0, -50), (0, -50), (5, 50), (5, 50)]
            n = int(2.5 * FPS)
            sub = 4
        elif scene == 9:
            keys = None
        if scene == 9:
            seg = [(20, -40), (-25, 10), (5, 55)]
            poses = np.concatenate([ease_path([np.array(s) + [0, -5], np.array(s) + [0, 5]], n // 3) for s in seg])
        elif keys is not None:
            poses = ease_path(keys, n)
        frames = lp_frames(lp, base_img, base, poses, sub=sub)
        n = len(frames)
        meta["source"] = "liveportrait"
    # per-frame landmarks (main frames only would misalign the sub-frames that reuse the nearest main frame, so sub-frames are computed too)
    flat = [f for subs in frames for f in subs]
    lms, pars, srcs = [], [], []
    for f in flat:
        lm, par, src = lmk(f, base_npz if meta["source"] == "liveportrait" else None)
        lms.append(lm)
        pars.append(par if par is not None else np.zeros(f.shape[:2], np.uint8))
        srcs.append(src)
    lms = smooth_landmarks(lms, sigma=0.0 if sub > 1 else 1.0)
    if scene == 11:
        params["hair_cover"] = [True, True]
    if scene == 10:
        params["reflect"] = 0.4 if k == 2 else params["reflect"]
        params["reflect_window"] = k == 2
    rend = render_clip(flat, lms, pars, params, seed)
    meta["landmark_source"] = {s: srcs.count(s) for s in set(srcs)}
    meta["target_poses"] = None if poses is None else np.round(poses, 1).tolist()

    # merge sub-frames (motion blur)
    G, C, Mk, Ln = [], [], [], []
    for i in range(n):
        rr = rend[i * sub:(i + 1) * sub]
        G.append(np.mean([r["image"].astype(np.float32) for r in rr], 0).astype(np.uint8))
        C.append(np.mean([r["clean"].astype(np.float32) for r in rr], 0).astype(np.uint8))
        Mk.append(np.mean([r["frame_mask"] for r in rr], 0))
        Ln.append(np.mean([r["lens_mask"] for r in rr], 0))
    h, w = G[0].shape[:2]
    # scene-specific post-processing
    canvas = (w, h)
    if scene in (6, 7, 9):
        canvas = (960, 540)
    writer = ClipWriter(out_dir, canvas)
    face_px0 = None
    skin = None
    for i in range(n):
        g, c, m, ln = G[i], C[i], Mk[i], Ln[i]
        lm = lms[i * sub]
        if abs(post["roll"][i]) > 0.01 and lm is not None:
            g, c, m, ln = rotate_all([g, c, m, ln], lm[:, :2].mean(0), post["roll"][i])
        if lm is not None and face_px0 is None:
            face_px0 = float(np.ptp(lm[:, 0]) * aug.DET_SIDE)
        if scene == 6:          # distant tiny face 30-60 px, moving slowly
            target = rng.uniform(30, 60) if i == 0 else post["target"]
            post["target"] = target
            s = target / face_px0
            cx = 480 + 300 * np.sin(2 * np.pi * i / n)
            cy = 270 + 80 * np.sin(4 * np.pi * i / n)
            g, c, m, ln = place_on_canvas([g, c, m, ln], s, (cx, cy))
        elif scene == 7:        # dolly: 60 px -> 400 px
            fpx = 60 * (400 / 60) ** (0.5 - 0.5 * np.cos(np.pi * i / max(1, n - 1)))
            g, c, m, ln = place_on_canvas([g, c, m, ln], fpx / face_px0, (480, 290))
        elif scene == 9:        # 3 cameras: close, medium, far, with a 1 s empty shot in between
            cam = min(2, i // (n // 3))
            s = [1.0, 0.45, 0.2][cam] * 300 / face_px0
            pos = [(480, 300), (300, 260), (700, 250)][cam]
            g, c, m, ln = place_on_canvas([g, c, m, ln], s, pos)
        elif scene == 10:
            r2 = np.random.default_rng(seed + i)
            if k == 0:
                g, c = aug.backlight(g, np.random.default_rng(seed)), aug.backlight(c, np.random.default_rng(seed))
            elif k == 1:
                g, c = aug.low_light(g, r2), aug.low_light(c, np.random.default_rng(seed + i))
        elif scene == 11 and k % 2 == 0 and lm is not None:
            if skin is None:
                skin = np.median(c[np.isin(pars[0], [1])], 0) if (pars[0] == 1).any() else np.array([150, 170, 210.0])
            x0_, y0_ = lm[:, :2].min(0)
            x1_, y1_ = lm[:, :2].max(0)
            col, ha = draw_hand(g.shape[:2], i / max(1, n - 1), skin, seed, (x0_, y0_, x1_, y1_))
            ha3 = ha[..., None]
            g = (g * (1 - ha3) + col * ha3).astype(np.uint8)
            c = (c * (1 - ha3) + col * ha3).astype(np.uint8)
            m = m * (1 - ha)
            ln = ln * (1 - ha)
        if scene == 9 and i == n // 3:      # 1 s empty shot (no face) before the first cut
            empty = cv2.GaussianBlur(cv2.resize(C[0], canvas), (0, 0), 30)
            for _ in range(FPS):
                writer.write(empty, empty, np.zeros(canvas[::-1], np.float32), np.zeros(canvas[::-1], np.float32))
        writer.write(g, c, m, ln)
    writer.close()
    meta["frames"] = writer.n
    meta["size"] = list(canvas)
    meta["params"] = {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in params.items()
                      if kk not in ("tint",)}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, default=float)
    print(f"  {name} {SCENES[scene]}: {writer.n} frames, {meta['style']} {meta['color']}, landmarks {meta['landmark_source']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", type=int, nargs="*", default=list(SCENES))
    ap.add_argument("--clips", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tr, va = split_ids(load_recs(a.pool))
    # base footage: validation-set people (unseen in training), near-frontal real photos with accurate landmarks
    bases = [r for r in va if r.get("method", "real") == "real" and abs(r["pitch"]) < 15 and abs(r["yaw"]) < 15
             and r.get("dataset") != "smoke"]
    if not bases:
        bases = [r for r in va if r.get("method", "real") == "real"]
    sys.path.insert(0, os.path.join(HERE, "repose"))
    from repose.liveportrait_repose import LPRepose
    ctx = dict(lmk=Landmarker(), lp=LPRepose(), bases=bases)
    print(f"{len(bases)} base-image candidates (validation set, unseen in training)")
    for sc in a.scenes:
        for k in range(a.clips):
            make_clip(sc, k, a, ctx)


if __name__ == "__main__":
    main()
