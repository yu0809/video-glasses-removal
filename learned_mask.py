"""
Network mask (frame segmentation network -> inpainting mask)
=============================================================
- temporal smoothing: frame probabilities of the neighbouring +-k frames, aligned by the glasses' centre
  (centroid of the lens probability), are averaged with weights
- hysteresis: pixels above the high threshold 0.5 are seeds; regions connected to them above 0.25 count as frame
- dilation by margin = base + margin_k x frame width + grow (grow follows --mask_grow and the rework widening)
- the lens interior (lens probability > 0.5 and farther than the margin from the frame) is not inpainted
- residue score: run the network again after inpainting; mean frame probability inside the original mask
All lengths are pixels of the face crop (--proc_size).
"""
import cv2
import numpy as np

DEFAULT_CFG = dict(
    smooth_k=2,          # temporal smoothing radius (frames)
    smooth_w=(1, 2, 4, 2, 1),
    hi=0.5, lo=0.25,     # hysteresis thresholds
    margin=2.0,          # base frame margin (px)
    margin_k=0.0,        # plus margin_k x this frame's estimated frame width (close-ups have thick rims with wider edges and reflections)
    lens_thr=0.5,
    min_frac=0.0003,     # a frame with a smaller share of frame pixels is treated as having no glasses
)


def ellipse(r):
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def lens_center(lens_prob, thr=0.5):
    ys, xs = np.nonzero(lens_prob > thr)
    if len(xs) < 20:
        return None
    return np.array([xs.mean(), ys.mean()], np.float32)


def shift(img, d):
    M = np.float32([[1, 0, d[0]], [0, 1, d[1]]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)


def temporal_smooth(probs, centers, k, weights):
    """Weighted average of the +-k neighbouring probabilities aligned by the glasses' centre; frames without a centre are used unaligned."""
    if k <= 0:
        return probs
    w = np.asarray(weights[len(weights) // 2 - k: len(weights) // 2 + k + 1], np.float32)
    out = []
    n = len(probs)
    for i in range(n):
        acc = np.zeros_like(probs[i])
        ws = 0.0
        for o, wt in zip(range(-k, k + 1), w):
            j = i + o
            if not 0 <= j < n:
                continue
            p = probs[j]
            if o != 0 and centers[i] is not None and centers[j] is not None:
                p = shift(p, centers[i] - centers[j])
            acc += wt * p
            ws += wt
        out.append(acc / ws)
    return out


def hysteresis(prob, hi, lo, min_rel=0.02, min_px=30):
    """Seeds above the high threshold grown into the low threshold; then drop isolated small blobs (area < min_rel of the largest and < min_px, usually background false positives)."""
    low = (prob > lo).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(low)
    if n <= 1:
        return np.zeros_like(low)
    seeds = np.unique(lab[prob > hi])
    seeds = seeds[seeds > 0]
    if len(seeds) == 0:
        return np.zeros_like(low)
    area = st[seeds, cv2.CC_STAT_AREA]
    keep = seeds[(area >= min_rel * area.max()) | (area >= min_px)]
    return np.isin(lab, keep).astype(np.uint8)


def build_masks(frame_probs, lens_probs, grow, cfg=None):
    """Return (masks 0/255 uint8, smoothed frame probs, list of stats). grow: extra dilation per frame in pixels (array)."""
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    centers = [lens_center(l, cfg["lens_thr"]) for l in lens_probs]
    sm = temporal_smooth(frame_probs, centers, cfg["smooth_k"], cfg["smooth_w"])
    masks, stats = [], []
    for i, (p, lp) in enumerate(zip(sm, lens_probs)):
        core = hysteresis(p, cfg["hi"], cfg["lo"])
        if core.mean() < cfg["min_frac"]:
            masks.append(np.zeros(p.shape, np.uint8))
            stats.append(dict(glasses=0, prob_mean=round(float(p.max()), 3)))
            continue
        thick = 0.0
        if cfg["margin_k"] > 0:
            dt = cv2.distanceTransform(core, cv2.DIST_L2, 3)
            thick = 2.0 * float(np.percentile(dt[core > 0], 90))
        m = cfg["margin"] + cfg["margin_k"] * thick + float(grow[i])
        mask = cv2.dilate(core, ellipse(m))
        # keep the lens interior farther than the margin from the frame untouched (do not repaint the eyes)
        far = cv2.distanceTransform(1 - core, cv2.DIST_L2, 3) > m
        mask[(lp > cfg["lens_thr"]) & far] = 0
        masks.append(mask * 255)
        stats.append(dict(glasses=1, prob_mean=round(float(p[core > 0].mean()), 3),
                          mask_pct=round(100 * float(mask.mean()), 2), grow=float(grow[i]), thick=round(thick, 1)))
    return masks, sm, stats


def residue_scores(seg, outs, masks):
    """Run the network again after inpainting: mean frame probability inside the original mask (lower is better) and the residue location (probability > 0.5)."""
    probs, _ = seg.predict(outs)
    res = []
    for p, m in zip(probs, masks):
        sel = m > 0
        if not sel.any():
            res.append((0.0, None))
            continue
        res.append((float(p[sel].mean()), ((p > 0.5) & sel).astype(np.uint8)))
    return res
