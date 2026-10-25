#!/usr/bin/env python3
"""
Video glasses-removal pipeline
==============================
0. scan the whole video and split it into clips where "a face is continuously visible" (shot cuts and face disappearance are detected);
   parts without a face are passed through unchanged
1. RetinaFace face detection, smoothed into a stable face crop
2. BiSeNet face parsing for the glasses region (the eye_g class of CelebAMask-HQ)
3. mask: the frame segmentation network (seg_model.py + learned_mask.py) or, without weights, the rule-based adaptive mask (adaptive.py);
   only the rims, bridge, hinges / temples are covered,
   the eyes behind the lenses stay untouched
4. ProPainter performs temporally consistent inpainting of the cropped face video
5. per-frame quality check: frames with frame residue above a threshold get a wider mask and are reworked together with their neighbours; kept only if better
6. feathered paste-back into the original video, keeping the original audio; writes a per-frame report and a quality plot

Usage:
    python remove_glasses.py -i input/video.mp4
    python remove_glasses.py -i input/video.mp4 --start 5 --duration 4   # try a short section first
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

import adaptive as ad

HERE = os.path.dirname(os.path.abspath(__file__))
PROPAINTER_DIR = os.path.join(HERE, "third_party", "ProPainter")

GLASSES = 6          # eye_g
L_BROW, R_BROW, L_EYE, R_EYE = 2, 3, 4, 5
HAIR = 17


# ----------------------------------------------------------------------------
# video I/O (streaming; the whole video is never held in memory)
# ----------------------------------------------------------------------------
def probe(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, W, H


def iter_frames(path, start_frame=0, limit=None):
    """Read frame by frame. The start position is reached by skipping frames one by one so every read returns exactly the same frames."""
    cap = cv2.VideoCapture(path)
    for _ in range(start_frame):
        if not cap.grab():
            break
    i = 0
    while limit is None or i < limit:
        ok, f = cap.read()
        if not ok:
            break
        yield i, f
        i += 1
    cap.release()


class VideoPipe:
    """Write video frame by frame through an ffmpeg pipe."""

    def __init__(self, path, w, h, fps, audio_src=None, start=0.0, duration=None, crf=16):
        self.path = path
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-"]
        if audio_src:
            # trim the audio to the video duration (not -shortest, which may drop the last frame)
            cmd += ["-ss", f"{start}"] + (["-t", f"{duration}"] if duration else []) + \
                   ["-i", audio_src, "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-b:a", "192k"]
        cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
                "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", path]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            sys.exit(f"ffmpeg failed to write: {self.path}")


# ----------------------------------------------------------------------------
# face detection, shot cuts, clips
# ----------------------------------------------------------------------------
def frame_signature(f):
    small = cv2.resize(f, (96, 54), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def scan_video(path, start_frame, limit, detector, min_face):
    """First pass: face candidates per frame + image difference to the previous frame (to detect shot cuts)."""
    cands, cut_dist = [], []
    prev_sig = None
    for i, f in iter_frames(path, start_frame, limit):
        faces = detector.detect(f)
        c = []
        for fc in faces or []:
            x1, y1, x2, y2 = [float(v) for v in fc.bbox[:4]]
            if min(x2 - x1, y2 - y1) >= min_face:
                c.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
        cands.append(c)
        sig = frame_signature(f)
        cut_dist.append(0.0 if prev_sig is None else
                        cv2.compareHist(prev_sig, sig, cv2.HISTCMP_BHATTACHARYYA))
        prev_sig = sig
        if i % 100 == 0:
            print(f"  scanned {i} frames", end="\r", flush=True)
    print(" " * 30, end="\r")
    return cands, np.array(cut_dist)


def build_segments(cands, cut_dist, fps, args):
    """Split the video into clips in which "the same face is continuously visible".
    A clip ends at a shot cut, when the face disappears for longer than --max_gap seconds, or when the face position / size jumps.
    Frames with brief missed detections inside a clip are interpolated."""
    max_gap = int(round(args.max_gap * fps))
    segments = []
    cur = None

    def close():
        nonlocal cur
        if cur is None:
            return
        idx = np.array(sorted(cur["boxes"]))
        s, e = int(idx[0]), int(idx[-1])
        if e - s + 1 >= args.min_segment:
            arr = np.array([cur["boxes"][k] for k in idx])
            full = np.arange(s, e + 1)
            boxes = np.stack([np.interp(full, idx, arr[:, k]) for k in range(4)], axis=1)
            segments.append({"start": s, "end": e, "boxes": boxes, "detected": len(idx)})
        cur = None

    for i, c in enumerate(cands):
        if i > 0 and cut_dist[i] > args.cut_threshold:
            close()
        if cur is not None:
            if c:
                lb = cur["last"]
                pick = min(c, key=lambda b: (b[0] - lb[0]) ** 2 + (b[1] - lb[1]) ** 2)
                move = np.hypot(pick[0] - lb[0], pick[1] - lb[1]) / lb[2]
                ratio = max(pick[2] / lb[2], lb[2] / pick[2])
                gap = i - cur["last_i"] - 1
                if move < 0.6 and ratio < 1.5 and gap <= max_gap:
                    cur["boxes"][i] = pick
                    cur["last"], cur["last_i"] = pick, i
                    continue
                close()
            elif i - cur["last_i"] > max_gap:
                close()
                continue
            else:
                continue
        if c:
            pick = max(c, key=lambda b: b[2] * b[3])
            cur = {"boxes": {i: pick}, "last": pick, "last_i": i}
    close()
    return segments


def crop_transforms(boxes, fps, crop_scale, proc_size, y_shift):
    """Smooth the face boxes; returns one 2x3 affine matrix per frame (image -> crop). The crop side is fixed within a clip."""
    sigma = max(1.0, fps / 8.0)
    cx = gaussian_filter1d(boxes[:, 0], sigma, mode="nearest")
    cy = gaussian_filter1d(boxes[:, 1], sigma, mode="nearest")
    side = float(np.median(np.maximum(boxes[:, 2], boxes[:, 3]))) * crop_scale
    cy = cy - y_shift * side      # the glasses are in the upper half of the face, so the crop moves up slightly
    s = proc_size / side
    mats = []
    for x, y in zip(cx, cy):
        mats.append(np.array([[s, 0, proc_size / 2 - s * x],
                              [0, s, proc_size / 2 - s * y]], dtype=np.float64))
    return mats, side


# ----------------------------------------------------------------------------
# glasses mask
# ----------------------------------------------------------------------------
def ellipse(r):
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def glasses_mask(parsing, face_w, args):
    """parsing: BiSeNet class map. Returns a uint8 0/255 mask (the region to inpaint)."""
    blob = (parsing == GLASSES).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(blob)
    min_area = (0.01 * face_w) ** 2 * 4
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not keep:
        return np.zeros_like(blob)
    blob = np.isin(lab, keep).astype(np.uint8)

    if args.mask_mode == "full":
        # inpaint the whole glasses (for sunglasses / dark lenses)
        return cv2.dilate(blob, ellipse(args.dilate * face_w)) * 255

    rim = args.rim_width * face_w
    outer = cv2.dilate(blob, ellipse(rim * 0.5))
    inner = cv2.erode(blob, ellipse(rim))
    band = outer & (1 - inner)

    # bridge: upper part between the two lenses
    big = sorted(keep, key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)[:2]
    if len(big) == 2:
        a, b = sorted(big, key=lambda i: stats[i, cv2.CC_STAT_LEFT])
        ax2 = stats[a, cv2.CC_STAT_LEFT] + stats[a, cv2.CC_STAT_WIDTH]
        bx1 = stats[b, cv2.CC_STAT_LEFT]
        if bx1 > ax2:
            y1 = min(stats[a, cv2.CC_STAT_TOP], stats[b, cv2.CC_STAT_TOP]) - rim
            y2 = (cent[a][1] + cent[b][1]) / 2
            x1, x2 = ax2 - rim, bx1 + rim
            cv2.rectangle(band, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)
        # temples: look for thin horizontal lines outside the lenses (optional)
        if args.temples == "on":
            band |= temple_mask(parsing, stats, a, b, rim, face_w)

    # never touch the eyes
    eyes = np.isin(parsing, [L_EYE, R_EYE]).astype(np.uint8)
    band &= 1 - cv2.dilate(eyes, ellipse(rim * 0.3))
    return band * 255


def temple_mask(parsing, stats, a, b, rim, face_w):
    """Draw a narrow horizontal band outside each lens (temples usually start at 1/3 of the lens height and run to the ear / hair)."""
    m = np.zeros(parsing.shape, np.uint8)
    face = np.isin(parsing, [1, 2, 3, 7, 8, 9, 10, HAIR]).astype(np.uint8)  # skin / brows / ears / hair
    for comp, side in ((a, -1), (b, 1)):
        x, y, w, h = [stats[comp, k] for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                  cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
        yc = y + h * 0.3
        th = max(2, rim * 0.9)
        length = 0.25 * face_w
        if side < 0:
            x0, x1 = x - length, x + rim
        else:
            x0, x1 = x + w - rim, x + w + length
        cv2.rectangle(m, (int(x0), int(yc - th)), (int(x1), int(yc + th)), 1, -1)
    return m & face


def build_masks(crops, parser, face_w, args):
    masks = []
    for c in crops:
        p = parser.parse(c)
        masks.append(glasses_mask(p, face_w, args))
    # union over +-k frames to remove mask flicker
    k = args.temporal_union
    out = []
    for i in range(len(masks)):
        acc = np.zeros_like(masks[i])
        for j in range(max(0, i - k), min(len(masks), i + k + 1)):
            acc = np.maximum(acc, masks[j])
        out.append(acc)
    return out


# ----------------------------------------------------------------------------
# ProPainter
# ----------------------------------------------------------------------------
def run_propainter(crops, masks, fps, proc_size, args, workdir):
    vdir = os.path.join(workdir, "clip")
    mdir = os.path.join(workdir, "clip_mask")
    odir = os.path.join(workdir, "out")
    for d in (vdir, mdir, odir):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    for i, (c, m) in enumerate(zip(crops, masks)):
        cv2.imwrite(os.path.join(vdir, f"{i:05d}.png"), c)
        cv2.imwrite(os.path.join(mdir, f"{i:05d}.png"), m)
    cmd = [sys.executable, "inference_propainter.py",
           "-i", vdir, "-m", mdir, "-o", odir,
           "--height", str(proc_size), "--width", str(proc_size),
           "--mask_dilation", str(args.pp_dilation),
           "--neighbor_length", str(args.neighbor_length),
           "--ref_stride", str(args.ref_stride),
           "--subvideo_length", str(args.subvideo_length),
           "--raft_iter", str(args.raft_iter),
           "--save_fps", str(int(round(fps))), "--save_frames"]
    if args.fp16:
        cmd.append("--fp16")
    def launch():
        env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
        if args.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["PROPAINTER_FORCE_CPU"] = "1"
        return subprocess.run(cmd, cwd=PROPAINTER_DIR, env=env).returncode

    rc = launch()
    killed = rc < 0 or rc == 137
    if rc != 0 and not killed and args.device != "cpu":
        # Apple GPU (mps) occasionally hits unsupported operators: retry once on CPU, and use CPU for the remaining chunks
        print("  GPU run failed, retrying on CPU (slower)")
        args.device = "cpu"
        rc = launch()
        killed = rc < 0 or rc == 137
    if rc != 0:
        hint = " (the process was killed by the system, most likely out of memory: lower --chunk or --proc_size)" if killed else ""
        sys.exit(f"ProPainter failed with exit code {rc}{hint}")
    fdir = os.path.join(odir, "clip", "frames")
    res = [cv2.imread(os.path.join(fdir, f"{i:04d}.png")) for i in range(len(crops))]
    if any(x is None for x in res):
        sys.exit("ProPainter output frames are missing")
    return res


def inpaint_chunks(crops, masks, fps, proc_size, args, workdir):
    n = len(crops)
    if n <= args.chunk:
        return run_propainter(crops, masks, fps, proc_size, args, workdir)
    out = [None] * n
    weight = np.zeros(n)
    step = args.chunk - args.overlap
    starts = list(range(0, n - args.overlap, step))
    for ci, s in enumerate(starts):
        e = min(n, s + args.chunk)
        print(f"  chunk {ci + 1}/{len(starts)}: frames {s}-{e - 1}")
        res = run_propainter(crops[s:e], masks[s:e], fps, proc_size, args, workdir)
        for j, img in enumerate(res):
            idx = s + j
            w = 1.0
            if ci > 0 and j < args.overlap:
                w = (j + 1) / (args.overlap + 1)
            if e < n and j >= len(res) - args.overlap:
                w = min(w, (len(res) - j) / (args.overlap + 1))
            acc = img.astype(np.float32) * w
            out[idx] = acc if out[idx] is None else out[idx] + acc
            weight[idx] += w
        if e >= n:
            break
    return [np.clip(o / w, 0, 255).astype(np.uint8) for o, w in zip(out, weight)]


# ----------------------------------------------------------------------------
# adaptive: build masks -> inpaint -> per-frame quality check -> rework failed frames with wider masks
# ----------------------------------------------------------------------------
def score_all(infos, outs, crops, masks, parser, idx=None):
    idx = range(len(outs)) if idx is None else idx
    res = {}
    for j in idx:
        sc, line, parse, rmap = ad.residue_score(infos[j], outs[j], parser, masks[j] > 0, crops[j])
        res[j] = (sc, rmap)
    return res


def group_windows(bad, n, ctx):
    """Merge the frames to rework, plus ctx frames on each side, into windows."""
    wins = []
    for j in sorted(bad):
        a, b = max(0, j - ctx), min(n, j + ctx + 1)
        if wins and a <= wins[-1][1]:
            wins[-1][1] = max(wins[-1][1], b)
        else:
            wins.append([a, b])
    return wins


def process_segment_adaptive(crops, parser, fps, P, args, workdir, seg_label, preview):
    cfg = dict(ad.DEFAULT_CFG)
    n = len(crops)
    grow = np.full(n, float(args.mask_grow))
    masks, infos, labs, stats = ad.adaptive_masks(crops, parser, cfg, temples=args.temples,
                                                  union=args.temporal_union, grow=grow)
    for c, m in zip(crops, masks):
        preview.write(overlay_mask(c, m))
    n_glasses = sum(1 for m in masks if m.max() > 0)
    thick = [st["thick"] for st in stats if st.get("glasses")]
    temples = sum(1 for inf in infos if inf is not None and inf["temple_px"].any())
    print(f"    glasses found in {n_glasses}/{n} frames; measured frame thickness {np.median(thick) if thick else 0:.1f}px, "
          f"temples detected in {temples} frames, mean mask {np.mean([m.mean() / 2.55 for m in masks]):.2f}%")
    rows = [dict(st, score0=None, score=None, rounds=0) for st in stats]
    if n_glasses == 0 or args.mask_only:
        return None, masks, rows, n_glasses

    fixed = inpaint_chunks(crops, masks, fps, P, args, workdir)
    sc = score_all(infos, fixed, crops, masks, parser)
    for j in range(n):
        rows[j]["score0"] = rows[j]["score"] = round(sc[j][0], 3)

    for it in range(args.refine_iters):
        bad = [j for j in range(n) if sc[j][0] > args.refine_threshold]
        if not bad:
            break
        print(f"    quality check round {it + 1}: {len(bad)} frames with residue (score > {args.refine_threshold}), reworking with wider masks")
        # widen: by a fraction of this frame's frame width, and merge the residue locations into the mask directly
        for j in bad:
            st = stats[j]
            step = max(2.0, 0.35 * (st.get("hi", 0) - st.get("lo", 0))) if st.get("glasses") else 2.0
            grow[j] += step
        new_masks, _, _, _ = ad.adaptive_masks(crops, parser, cfg, temples=args.temples,
                                               union=args.temporal_union, grow=grow, infos=infos, labs=labs)
        for j in bad:
            rmap = sc[j][1]
            if rmap is not None and rmap.any():
                new_masks[j] = np.maximum(new_masks[j], cv2.dilate(rmap, ad.ellipse(3)) * 255)
        improved = 0
        for a, b in group_windows(bad, n, args.refine_context):
            cand = inpaint_chunks(crops[a:b], new_masks[a:b], fps, P, args, workdir)
            in_win = [j for j in bad if a <= j < b]
            new_sc = score_all(infos[a:b], cand, crops[a:b], new_masks[a:b], parser,
                               [j - a for j in range(a, b)])
            old_mean = np.mean([sc[j][0] for j in in_win])
            new_mean = np.mean([new_sc[j - a][0] for j in in_win])
            if new_mean >= old_mean:
                continue
            improved += len(in_win)
            ramp = max(1, min(4, (b - a) // 4))
            for j in range(a, b):
                # cross-fade with the previous result at both window ends to avoid jumps; use the reworked result in the middle
                w = 1.0
                if a > 0:
                    w = min(w, (j - a + 1) / (ramp + 1))
                if b < n:
                    w = min(w, (b - j) / (ramp + 1))
                if j in in_win:
                    w = 1.0
                fixed[j] = (cand[j - a].astype(np.float32) * w + fixed[j].astype(np.float32) * (1 - w)).astype(np.uint8)
                masks[j] = np.maximum(masks[j], new_masks[j])
                sc[j] = new_sc[j - a] if w == 1.0 else sc[j]
                rows[j]["score"] = round(sc[j][0], 3)
                rows[j]["rounds"] = it + 1 if j in in_win else rows[j]["rounds"]
                rows[j]["grow"] = float(grow[j])
        print(f"      {improved} frames improved after rework")
    left = sum(1 for j in range(n) if sc[j][0] > args.refine_threshold)
    print(f"    final: {n - left}/{n} frames passed the quality check" + (f", {left} frames may still have residue (see the report)" if left else ""))
    return fixed, masks, rows, n_glasses


def process_segment_learned(crops, seg, fps, P, args, workdir, preview):
    """Network version: frame segmentation probabilities -> temporal smoothing + hysteresis -> inpaint -> network re-checks residue -> rework failed frames with wider masks."""
    import learned_mask as lm
    n = len(crops)
    frame_p, lens_p = seg.predict(crops)
    grow = np.full(n, float(args.mask_grow))
    mcfg = dict(margin=args.seg_margin, margin_k=args.seg_margin_k)
    masks, sm, stats = lm.build_masks(frame_p, lens_p, grow, mcfg)
    for c, m in zip(crops, masks):
        preview.write(overlay_mask(c, m))
    n_glasses = sum(1 for m in masks if m.max() > 0)
    print(f"    glasses found in {n_glasses}/{n} frames (network), mean mask {np.mean([m.mean() / 2.55 for m in masks]):.2f}%")
    rows = [dict(st, score0=None, score=None, rounds=0) for st in stats]
    if n_glasses == 0 or args.mask_only:
        return None, masks, rows, n_glasses
    thr = args.seg_refine_threshold
    fixed = inpaint_chunks(crops, masks, fps, P, args, workdir)
    sc = lm.residue_scores(seg, fixed, masks)
    for j in range(n):
        rows[j]["score0"] = rows[j]["score"] = rows[j]["score_model"] = round(sc[j][0], 3)
    for it in range(args.refine_iters):
        bad = [j for j in range(n) if sc[j][0] > thr]
        if not bad:
            break
        print(f"    quality check round {it + 1}: {len(bad)} frames with residue (network score > {thr}), reworking with wider masks")
        for j in bad:
            grow[j] += 2.0 + 0.5 * it
        new_masks, _, _ = lm.build_masks(frame_p, lens_p, grow, mcfg)
        for j in bad:
            if sc[j][1] is not None and sc[j][1].any():
                new_masks[j] = np.maximum(new_masks[j], cv2.dilate(sc[j][1], ellipse(3)) * 255)
        improved = 0
        for a, b in group_windows(bad, n, args.refine_context):
            cand = inpaint_chunks(crops[a:b], new_masks[a:b], fps, P, args, workdir)
            new_sc = lm.residue_scores(seg, cand, new_masks[a:b])
            in_win = [j for j in bad if a <= j < b]
            if np.mean([new_sc[j - a][0] for j in in_win]) >= np.mean([sc[j][0] for j in in_win]):
                continue
            improved += len(in_win)
            ramp = max(1, min(4, (b - a) // 4))
            for j in range(a, b):
                w = 1.0
                if a > 0:
                    w = min(w, (j - a + 1) / (ramp + 1))
                if b < n:
                    w = min(w, (b - j) / (ramp + 1))
                if j in in_win:
                    w = 1.0
                fixed[j] = (cand[j - a].astype(np.float32) * w + fixed[j].astype(np.float32) * (1 - w)).astype(np.uint8)
                masks[j] = np.maximum(masks[j], new_masks[j])
                if w == 1.0:
                    sc[j] = new_sc[j - a]
                rows[j]["score"] = rows[j]["score_model"] = round(sc[j][0], 3)
                rows[j]["rounds"] = it + 1 if j in in_win else rows[j]["rounds"]
                rows[j]["grow"] = float(grow[j])
                rows[j]["mask_pct"] = round(masks[j].mean() / 2.55, 2)
        print(f"      {improved} frames improved after rework")
    left = sum(1 for j in range(n) if sc[j][0] > thr)
    print(f"    final: {n - left}/{n} frames passed the quality check" + (f", {left} frames may still have residue (see the report)" if left else ""))
    return fixed, masks, rows, n_glasses


def write_report(path_csv, path_png, all_rows, fps, offset, threshold):
    import csv
    keys = ["frame", "time", "segment", "glasses", "gw", "conf", "thick", "lo", "hi",
            "grow", "mask_pct", "prob_mean", "score0", "score", "score_model", "rounds"]
    with open(path_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = [r["time_s"] for r in all_rows]
        s0 = [np.nan if r.get("score0") is None else r["score0"] for r in all_rows]
        s1 = [np.nan if r.get("score") is None else r["score"] for r in all_rows]
        mk = [np.nan if not r.get("glasses") else r.get("mask_pct", np.nan) for r in all_rows]
        fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
        ax[0].plot(t, s0, color="#bbb", lw=1, label="first pass")
        ax[0].plot(t, s1, color="#d33", lw=1.5, label="final")
        ax[0].axhline(threshold, color="#888", ls="--", lw=1, label="threshold")
        ax[0].set_ylabel("residue score (lower=better)")
        ax[0].legend(loc="upper right", fontsize=8)
        ax[1].plot(t, mk, color="#27a", lw=1.2)
        ax[1].set_ylabel("mask area %")
        ax[1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(path_png, dpi=110)
        plt.close(fig)
    except Exception as e:  # a failed plot does not affect the main pipeline
        print(f"  (quality plot could not be drawn: {e})")


# ----------------------------------------------------------------------------
# main pipeline
# ----------------------------------------------------------------------------
def fmt_t(frame_idx, fps, offset):
    t = offset + frame_idx / fps
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def overlay_mask(c, m):
    o = c.copy()
    sel = m > 0
    o[sel] = (0.35 * o[sel] + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
    return np.hstack([c, o])


def main():
    ap = argparse.ArgumentParser(description="Video glasses removal (ProPainter + face parsing)")
    ap.add_argument("-i", "--input", required=True, help="input video")
    ap.add_argument("-o", "--output", default=os.path.join(HERE, "output"), help="output folder")
    ap.add_argument("--start", type=float, default=0.0, help="start time (seconds)")
    ap.add_argument("--duration", type=float, default=None, help="how many seconds to process (default: to the end)")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--proc_size", type=int, default=512, help="processing resolution of the face crop (multiple of 8)")
    ap.add_argument("--crop_scale", type=float, default=1.3, help="crop box = face box x this factor")
    ap.add_argument("--y_shift", type=float, default=0.08, help="upward shift of the crop box (fraction)")
    ap.add_argument("--min_face", type=int, default=48, help="ignore faces narrower than this (px)")
    ap.add_argument("--max_gap", type=float, default=0.3,
                    help="longest brief missed detection allowed inside a clip (seconds); longer gaps split the clip")
    ap.add_argument("--min_segment", type=int, default=5, help="skip clips shorter than this many frames")
    ap.add_argument("--cut_threshold", type=float, default=0.45,
                    help="shot-cut detection threshold (0-1, lower is more sensitive)")
    ap.add_argument("--fixed", action="store_true",
                    help="disable the adaptive mask and use the old fixed-parameter mask (with --rim_width / --mask_mode)")
    ap.add_argument("--temples", nargs="?", const="on", default="auto", choices=["auto", "on", "off"],
                    help="temples: auto = decide from the image (default); on = always erase; off = never erase")
    ap.add_argument("--mask_grow", type=float, default=0.0, help="extra overall widening of the adaptive mask (px)")
    ap.add_argument("--refine_iters", type=int, default=2, help="maximum rework rounds when the quality check fails (0 = no rework)")
    ap.add_argument("--refine_threshold", type=float, default=0.15,
                    help="rework when the residue score exceeds this (0-1, lower is stricter)")
    ap.add_argument("--refine_context", type=int, default=8, help="number of neighbouring frames reworked together on each side")
    ap.add_argument("--mask_mode", choices=["rim", "full"], default="rim",
                    help="[--fixed mode] rim = remove the frame only; full = the whole glasses (sunglasses / tinted lenses)")
    ap.add_argument("--rim_width", type=float, default=0.03, help="[--fixed mode] frame width (relative to face width)")
    ap.add_argument("--dilate", type=float, default=0.015, help="[--fixed mode] dilation for full (relative to face width)")
    ap.add_argument("--temporal_union", type=int, default=1, help="temporal union radius of the mask (frames)")
    ap.add_argument("--feather", type=float, default=3.0, help="feathering for the paste-back (px)")
    ap.add_argument("--chunk", type=int, default=50, help="frames per chunk; larger is more coherent but needs more memory (lower it if memory runs out)")
    ap.add_argument("--overlap", type=int, default=8, help="overlapping frames between chunks")
    ap.add_argument("--pp_dilation", type=int, default=2, help="ProPainter internal mask dilation")
    ap.add_argument("--neighbor_length", type=int, default=10)
    ap.add_argument("--ref_stride", type=int, default=10)
    ap.add_argument("--subvideo_length", type=int, default=50)
    ap.add_argument("--raft_iter", type=int, default=20)
    ap.add_argument("--fp16", action="store_true", help="NVIDIA GPUs only")
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto")
    ap.add_argument("--mask_only", action="store_true", help="only write a mask preview, no inpainting (for tuning)")
    ap.add_argument("--seg_model", default="auto",
                    help="frame segmentation weights (.pt / .onnx). auto: use weights/frame_seg.pt if it exists, otherwise the rule-based mask; none: force the rule-based mask")
    ap.add_argument("--seg_margin", type=float, default=2.0, help="[network] base mask dilation outside the frame (px)")
    ap.add_argument("--seg_margin_k", type=float, default=0.5,
                    help="[network] additional dilation of k x the estimated frame thickness (0.5 chosen by a sweep: synthetic residue 20%%->16%%, faint outlines on real close-ups mostly gone; use 1.0 for videos with many close-ups)")
    ap.add_argument("--seg_refine_threshold", type=float, default=0.2,
                    help="[network] rework when the mean frame probability inside the original mask exceeds this after inpainting")
    ap.add_argument("--keep_temp", action="store_true")
    args = ap.parse_args()
    assert args.proc_size % 8 == 0, "--proc_size must be a multiple of 8"

    from uniface import RetinaFace, BiSeNet
    from uniface.constants import ParsingWeights

    t0 = time.time()
    name = os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(args.output, exist_ok=True)
    P = args.proc_size
    face_w = P / args.crop_scale

    fps, W, H = probe(args.input)
    start_frame = int(round(args.start * fps))
    limit = None
    if args.duration:
        limit = int(round(args.duration * fps))
    if args.max_frames:
        limit = min(limit or args.max_frames, args.max_frames)

    print("[1/4] scanning the video: face detection + shot-cut detection")
    detector = RetinaFace()
    cands, cut_dist = scan_video(args.input, start_frame, limit, detector, args.min_face)
    N = len(cands)
    if N == 0:
        sys.exit("no frames were read; check --start / --duration")
    segments = build_segments(cands, cut_dist, fps, args)
    covered = sum(s["end"] - s["start"] + 1 for s in segments)
    print(f"  {N} frames, {W}x{H}, {fps:.2f} fps; {int((cut_dist > args.cut_threshold).sum())} shot cuts")
    print(f"  found {len(segments)} clips with a face, {covered} frames in total; the other {N - covered} frames are passed through")
    for k, s in enumerate(segments):
        print(f"    clip {k + 1}: {fmt_t(s['start'], fps, args.start)} - {fmt_t(s['end'], fps, args.start)}"
              f" ({s['end'] - s['start'] + 1} frames)")
    if not segments:
        sys.exit("no sufficiently large face was detected anywhere in the video; nothing to process")

    print("[2/4] building glasses masks per clip" + ("" if args.mask_only else " and inpainting"))
    parser = BiSeNet(model_name=ParsingWeights.RESNET34)
    seg = None
    seg_path = args.seg_model
    if seg_path == "auto":
        seg_path = os.path.join(HERE, "weights", "frame_seg.pt")
        seg_path = seg_path if os.path.exists(seg_path) else None
    elif seg_path == "none":
        seg_path = None
    if seg_path and not args.fixed and args.mask_mode != "full":
        from seg_model import FrameSegmenter
        seg = FrameSegmenter(seg_path)
        print(f"  mask: frame segmentation network {seg_path}")
    else:
        print("  mask: " + ("fixed parameters" if args.fixed or args.mask_mode == "full" else "rule-based adaptive"))
    workdir = tempfile.mkdtemp(prefix="deglasses_")
    resdir = os.path.join(workdir, "results")
    os.makedirs(resdir)
    preview_path = os.path.join(args.output, f"{name}_mask_preview.mp4")
    preview = VideoPipe(preview_path, 2 * P, P, fps)
    seg_of = np.full(N, -1)
    for k, s in enumerate(segments):
        s["mats"], s["side"] = crop_transforms(s["boxes"], fps, args.crop_scale, P, args.y_shift)
        s["has_glasses"] = False
        seg_of[s["start"]:s["end"] + 1] = k

    report_rows = {}
    try:
        crops = []
        for i, f in iter_frames(args.input, start_frame, N):
            k = seg_of[i]
            if k < 0:
                continue
            s = segments[k]
            M = s["mats"][i - s["start"]]
            crops.append(cv2.warpAffine(f, M, (P, P), flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REFLECT))
            if i != s["end"]:
                continue
            # -- a clip has been collected; process it --
            print(f"  clip {k + 1}/{len(segments)}: {len(crops)} frames, crop {s['side']:.0f}px -> {P}px")
            if args.fixed or args.mask_mode == "full":
                masks = build_masks(crops, parser, face_w, args)
                for c, m in zip(crops, masks):
                    preview.write(overlay_mask(c, m))
                n_glasses = sum(1 for m in masks if m.max() > 0)
                print(f"    glasses found in {n_glasses}/{len(masks)} frames")
                fixed = None
                if n_glasses and not args.mask_only:
                    fixed = inpaint_chunks(crops, masks, fps, P, args, os.path.join(workdir, "pp"))
                rows = [dict(glasses=int(m.max() > 0), mask_pct=round(m.mean() / 2.55, 2)) for m in masks]
            elif seg is not None:
                fixed, masks, rows, n_glasses = process_segment_learned(
                    crops, seg, fps, P, args, os.path.join(workdir, "pp"), preview)
            else:
                fixed, masks, rows, n_glasses = process_segment_adaptive(
                    crops, parser, fps, P, args, os.path.join(workdir, "pp"), k + 1, preview)
            for j, r in enumerate(rows):
                fi = s["start"] + j
                r.update(frame=fi + start_frame, time_s=round(args.start + fi / fps, 3),
                         time=fmt_t(fi, fps, args.start), segment=k + 1)
                report_rows[fi] = r
            if n_glasses == 0:
                print("    no glasses in this clip, skipping")
            elif fixed is not None:
                s["has_glasses"] = True
                for j, (fx, m) in enumerate(zip(fixed, masks)):
                    cv2.imwrite(os.path.join(resdir, f"{s['start'] + j:06d}_img.png"), fx)
                    cv2.imwrite(os.path.join(resdir, f"{s['start'] + j:06d}_mask.png"), m)
            crops = []
        preview.close()
        print(f"  mask preview: {preview_path}")
        rows = [report_rows.get(i, dict(frame=i + start_frame, time_s=round(args.start + i / fps, 3),
                                         time=fmt_t(i, fps, args.start), segment=0, glasses=0))
                for i in range(N)]
        csv_path = os.path.join(args.output, f"{name}_report.csv")
        png_path = os.path.join(args.output, f"{name}_quality.png")
        thr_used = args.seg_refine_threshold if seg is not None else args.refine_threshold
        write_report(csv_path, png_path, rows, fps, args.start, thr_used)
        print(f"  per-frame report: {csv_path}")
        if not args.mask_only and not args.fixed:
            print(f"  quality plot: {png_path}")
            bad = [r for r in rows if r.get("score") is not None and r["score"] > thr_used]
            if bad:
                spans, cur = [], [bad[0], bad[0]]
                for r in bad[1:]:
                    if r["frame"] - cur[1]["frame"] <= 3:
                        cur[1] = r
                    else:
                        spans.append(cur); cur = [r, r]
                spans.append(cur)
                print("  these time ranges may still have residue, check them first:")
                for a_, b_ in spans[:15]:
                    print(f"    {a_['time']} - {b_['time']}")
        if args.mask_only:
            return

        print("[3/4] pasting back into the original video")
        out_path = os.path.join(args.output, f"{name}_no_glasses.mp4")
        cmp_path = os.path.join(args.output, f"{name}_compare.mp4")
        out = VideoPipe(out_path, W, H, fps, audio_src=args.input, start=args.start, duration=N / fps)
        cmp_ = VideoPipe(cmp_path, 2 * W, H, fps)
        for i, f in iter_frames(args.input, start_frame, N):
            r = f
            k = seg_of[i]
            if k >= 0 and segments[k]["has_glasses"]:
                s = segments[k]
                fx = cv2.imread(os.path.join(resdir, f"{i:06d}_img.png"))
                m = cv2.imread(os.path.join(resdir, f"{i:06d}_mask.png"), cv2.IMREAD_GRAYSCALE)
                Minv = cv2.invertAffineTransform(s["mats"][i - s["start"]])
                back = cv2.warpAffine(fx, Minv, (W, H), flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)
                a = cv2.dilate(m, ellipse(1)).astype(np.float32) / 255
                if args.feather > 0:
                    a = cv2.GaussianBlur(a, (0, 0), args.feather)
                    a = np.clip(a * 1.5, 0, 1)
                a = cv2.warpAffine(a, Minv, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
                r = (f.astype(np.float32) * (1 - a) + back.astype(np.float32) * a).astype(np.uint8)
            out.write(r)
            cmp_.write(np.hstack([f, r]))
        out.close()
        cmp_.close()
    finally:
        if not args.keep_temp:
            shutil.rmtree(workdir, ignore_errors=True)

    print(f"[4/4] done in {time.time() - t0:.0f} s")
    print(f"  result:     {out_path}")
    print(f"  comparison: {cmp_path}")


if __name__ == "__main__":
    main()
