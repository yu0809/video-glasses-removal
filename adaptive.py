"""
Rule-based adaptive mask
========================
- every frame "measures" the actual frame in the image (morphological line detection) and sets that frame's mask width accordingly
- temples are erased only when the image shows evidence of them
- the temporal union compensates for the glasses' motion, so fast head movement does not smear the mask wider
- after inpainting every frame is scored (amount of frame residue) for the rework loop

All lengths are in units of "this frame's glasses width gw", so they scale automatically with distance to the camera and head turns.
"""
import cv2
import numpy as np

GLASSES = 6
L_BROW, R_BROW, L_EYE, R_EYE = 2, 3, 4, 5
HAIR = 17
SKIN_LIKE = [1, 2, 3, 7, 8, 9, 10, HAIR]   # skin / brows / ears / hair: where temples can appear


def ellipse(r):
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


# ----------------------------------------------------------------------------
# face parsing (argmax and glasses probability)
# ----------------------------------------------------------------------------
def parse_with_prob(parser, img):
    """Returns (class map, glasses probability map), both the size of the input image."""
    x = parser.preprocess(img)
    logits = parser.session.run(parser.output_names, {parser.input_name: x})[0][0]  # (19,h,w)
    lab = logits.argmax(0).astype(np.uint8)
    e = np.exp(logits - logits.max(0, keepdims=True))
    prob = (e[GLASSES] / e.sum(0)).astype(np.float32)
    h, w = img.shape[:2]
    lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return lab, prob


# ----------------------------------------------------------------------------
# line response: thin structures whose colour differs from their surroundings (rims, temples)
# ----------------------------------------------------------------------------
def line_response(img, radius, polarity=0):
    """Morphological top-hat / black-hat: bright and dark lines thinner than radius respond strongly.
    Computed on each of the three Lab channels and maxed, so gold / silver / black frames are all detected.
    polarity: +1 only lines brighter than their surroundings (silver frames), -1 only darker (black frames), 0 both."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    k = ellipse(radius)
    resp = np.zeros(img.shape[:2], np.float32)
    for c, w in ((0, 1.0), (1, 2.0), (2, 2.0)):
        ch = lab[..., c]
        if c > 0 or polarity <= 0:
            resp = np.maximum(resp, w * cv2.morphologyEx(ch, cv2.MORPH_BLACKHAT, k))
        if c > 0 or polarity >= 0:
            resp = np.maximum(resp, w * cv2.morphologyEx(ch, cv2.MORPH_TOPHAT, k))
    return resp


def otsu_threshold(values, floor):
    if values.size < 50:
        return floor
    v = np.clip(values, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return max(float(t), floor)


# ----------------------------------------------------------------------------
# per-frame analysis
# ----------------------------------------------------------------------------
def clean_blob(lab, min_frac):
    blob = (lab == GLASSES).astype(np.uint8)
    n, cc, stats, cent = cv2.connectedComponentsWithStats(blob)
    if n <= 1:
        return None
    biggest = stats[1:, cv2.CC_STAT_AREA].max()
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= max(min_frac * blob.size, 0.15 * biggest)]
    if not keep:
        return None
    keep = sorted(keep, key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)[:2]
    keep = sorted(keep, key=lambda i: stats[i, cv2.CC_STAT_LEFT])
    blob = np.isin(cc, keep).astype(np.uint8)
    lenses = [dict(x=int(stats[i, 0]), y=int(stats[i, 1]), w=int(stats[i, 2]), h=int(stats[i, 3]),
                   cx=float(cent[i][0]), cy=float(cent[i][1])) for i in keep]
    if len(lenses) == 1 and lenses[0]["w"] > 1.8 * lenses[0]["h"]:
        # the two lenses merged into one blob through the bridge: split at the narrowest column in the middle
        l = lenses[0]
        cols = blob[l["y"]:l["y"] + l["h"], l["x"]:l["x"] + l["w"]].sum(0)
        a, b = int(0.3 * l["w"]), int(0.7 * l["w"])
        cut = l["x"] + a + int(np.argmin(cols[a:b]))
        lenses = []
        for x0, x1 in ((l["x"], cut), (cut, l["x"] + l["w"])):
            ys, xs = np.nonzero(blob[:, x0:x1])
            if len(xs) == 0:
                return blob, [l]
            xs = xs + x0
            lenses.append(dict(x=int(xs.min()), y=int(ys.min()), w=int(xs.max() - xs.min() + 1),
                               h=int(ys.max() - ys.min() + 1), cx=float(xs.mean()), cy=float(ys.mean())))
    return blob, lenses


def analyze_frame(img, lab, cfg):
    """Measure this frame's glasses. Returns a dict (None when no glasses are detected):
    blob glasses region, lenses lens boxes, gw glasses width (px), frame_px detected frame pixels,
    thick estimated frame thickness (px), thr line threshold, centroid glasses centre"""
    r = clean_blob(lab, 0.0008)
    if r is None:
        return None
    blob, lenses = r
    xs = [l["x"] for l in lenses] + [l["x"] + l["w"] for l in lenses]
    gw = float(max(xs) - min(xs))
    if gw < 10:
        return None
    resp = line_response(img, max(2, cfg["kernel"] * gw))

    # look for the frame only near the glasses outline: a band slightly outside and inside, excluding the eyes
    ring = cv2.dilate(blob, ellipse(cfg["search_out"] * gw)) & \
        (1 - cv2.erode(blob, ellipse(cfg["search_in"] * gw)))
    eyes = np.isin(lab, [L_EYE, R_EYE]).astype(np.uint8)
    ring &= 1 - cv2.dilate(eyes, ellipse(0.01 * gw))
    ring &= 1 - np.isin(lab, [L_BROW, R_BROW]).astype(np.uint8)   # brows are not frame
    vals = resp[ring > 0]
    thr = otsu_threshold(vals, cfg["resp_floor"])
    frame_px = ((resp > thr) & (ring > 0)).astype(np.uint8)

    prof = contour_profiles(blob, frame_px, cfg["search_in"] * gw, cfg["search_out"] * gw)

    info = dict(img=img, lab_img=cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32), blob=blob, lenses=lenses, gw=gw, resp=resp, thr=thr, ring=ring,
                frame_px=frame_px, eyes=eyes, **prof,
                centroid=np.array([np.mean([l["cx"] for l in lenses]),
                                   np.mean([l["cy"] for l in lenses])]))
    info["temple_sides"] = temple_evidence(info, lab, cfg)
    # small blobs labelled "glasses" by the face parsing that belong to neither lens (usually hinges, temple roots) are added directly
    extra = ((lab == GLASSES) & (blob == 0)).astype(np.uint8)
    if extra.any():
        near = cv2.dilate(blob, ellipse(0.2 * gw))
        n, cc, st, cen = cv2.connectedComponentsWithStats(extra & near)
        cx = info["centroid"][0]
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < 3:
                continue
            si = 0 if cen[i][0] < cx else 1
            info["temple_sides"][si] |= cv2.dilate((cc == i).astype(np.uint8), ellipse(max(1.5, 0.01 * gw)))
    info["temple_px"] = info["temple_sides"][0] | info["temple_sides"][1]
    return info


def contour_profiles(blob, frame_px, depth_in, depth_out, step=3):
    """Every few pixels along the glasses outline, look along the normal for where the frame line starts and ends.
    Summarised with quantiles, so occasional distractors such as brows or reflections are ignored as outliers.
    Returns lo/hi (signed distance from the outline, positive = outwards), thick (frame thickness), conf (share of outline points that found a frame)"""
    H, W = blob.shape
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    t = np.arange(-int(np.ceil(depth_in)), int(np.ceil(depth_out)) + 1)
    starts, ends = [], []
    rim_px = np.zeros_like(blob)
    total = 0
    for c in contours:
        pts = c[:, 0, :].astype(np.float32)
        n = len(pts)
        if n < 20:
            continue
        for i in range(0, n, step):
            p = pts[i]
            tg = pts[(i + 4) % n] - pts[(i - 4) % n]
            nrm = np.array([tg[1], -tg[0]], np.float32)
            ln = np.linalg.norm(nrm)
            if ln < 1e-3:
                continue
            nrm /= ln
            q = p + 3 * nrm      # make sure the normal points outwards
            qx, qy = int(round(q[0])), int(round(q[1]))
            if 0 <= qx < W and 0 <= qy < H and blob[qy, qx]:
                nrm = -nrm
            total += 1
            xs = np.clip(np.round(p[0] + t * nrm[0]).astype(int), 0, W - 1)
            ys = np.clip(np.round(p[1] + t * nrm[1]).astype(int), 0, H - 1)
            v = frame_px[ys, xs] > 0
            if not v.any():
                continue
            # take the continuous line segment closest to the outline
            idx = np.where(v)[0]
            runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            zero = int(np.where(t == 0)[0][0])
            run = min(runs, key=lambda r: 0 if r[0] <= zero <= r[-1] else min(abs(r[0] - zero), abs(r[-1] - zero)))
            starts.append(t[run[0]])
            ends.append(t[run[-1]])
            rim_px[ys[run], xs[run]] = 1
    conf = len(starts) / max(total, 1)
    rim_px = cv2.morphologyEx(rim_px, cv2.MORPH_CLOSE, ellipse(1))
    if len(starts) < 10:
        return dict(lo=None, hi=None, thick=0.0, conf=conf, rim_px=rim_px)
    starts, ends = np.array(starts, float), np.array(ends, float)
    thick = float(np.median(ends - starts + 1))
    return dict(lo=float(np.percentile(starts, 20)), hi=float(np.percentile(ends, 80)),
                thick=thick, conf=conf, rim_px=rim_px)


def temple_evidence(info, lab, cfg):
    """Find hinges and temples outside the two lenses (left and right masks returned separately): thin-line detection + line fitting (Hough),
    requiring roughly horizontal segments of sufficient length with one end touching the upper half of the outer lens edge. Erased only when found."""
    ms = [np.zeros(lab.shape, np.uint8), np.zeros(lab.shape, np.uint8)]   # left / right
    if len(info["lenses"]) < 2:
        return ms
    gw = info["gw"]
    H, W = lab.shape
    thick = max(1.5, info["thick"])
    allow = np.isin(lab, SKIN_LIKE + [0, GLASSES]).astype(np.uint8)   # temples may also extend in front of the background
    info["allow"] = allow
    fcol, ctol = None, None
    if info["rim_px"].sum() > 20:
        px = info["lab_img"][info["rim_px"] > 0]
        fcol = np.median(px, axis=0)
        ctol = max(12.0, 1.3 * float(np.percentile(np.linalg.norm((px - fcol) * [0.6, 1.0, 1.0], axis=1), 70)))
    for si, (lens, side) in enumerate(((info["lenses"][0], -1), (info["lenses"][1], 1))):
        m = ms[si]
        y0 = int(lens["y"] - 0.05 * gw)
        y1 = int(lens["y"] + lens["h"] * 0.6)
        if side < 0:
            x0, x1 = int(lens["x"] - cfg["temple_len"] * gw), int(lens["x"] + 0.03 * gw)
            edge = lens["x"]
        else:
            x0, x1 = int(lens["x"] + lens["w"] - 0.03 * gw), int(lens["x"] + lens["w"] + cfg["temple_len"] * gw)
            edge = lens["x"] + lens["w"]
        x0, x1, y0, y1 = max(x0, 0), min(x1, W), max(y0, 0), min(y1, H)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        patch = info["img"][y0:y1, x0:x1]
        resp = line_response(patch, max(2, 1.5 * thick + 1))
        sel = allow[y0:y1, x0:x1] > 0
        vals = resp[sel]
        if vals.size < 30:
            continue
        thr = max(otsu_threshold(vals, cfg["resp_floor"]), info["thr"] * 0.8)
        hits = ((resp > thr) & sel).astype(np.uint8) * 255
        # hinge: a short piece on the outer lens edge at about 1/3 height (too short for Hough).
        # its position is almost fixed, so draw a short horizontal bar from this frame's lens position and size, with a width following the frame thickness
        hc = lens["y"] + cfg["hinge_y"] * lens["h"]
        hh = max(cfg["hinge_h"] * gw, 1.2 * thick + 1)
        hx_in, hx_out = (edge + 0.03 * gw, edge - cfg["hinge_len"] * gw) if side < 0 else \
            (edge - 0.03 * gw, edge + cfg["hinge_len"] * gw)
        cv2.rectangle(m, (int(min(hx_in, hx_out)), int(hc - hh)), (int(max(hx_in, hx_out)), int(hc + hh)), 1, -1)
        m &= allow
        lines = cv2.HoughLinesP(hits, 1, np.pi / 180, threshold=max(8, int(0.04 * gw)),
                                minLineLength=max(8, int(0.07 * gw)), maxLineGap=max(2, int(0.015 * gw)))
        if lines is None:
            continue
        best, best_len = None, 0
        for xa, ya, xb, yb in np.asarray(lines).reshape(-1, 4):
            ang = abs(np.degrees(np.arctan2(yb - ya, xb - xa)))
            ang = min(ang, 180 - ang)
            if ang > 25:
                continue
            # the end point closer to the lens
            ex, ey = (xb, yb) if (side < 0) == (xb > xa) else (xa, ya)
            ex, ey = ex + x0, ey + y0
            if abs(ex - edge) > 0.08 * gw or not (lens["y"] - 0.05 * gw <= ey <= lens["y"] + 0.55 * lens["h"]):
                continue
            ln = np.hypot(xb - xa, yb - ya)
            if fcol is not None:
                k = np.linspace(0, 1, 15)
                px = info["lab_img"][(ya + k * (yb - ya)).astype(int) + y0, (xa + k * (xb - xa)).astype(int) + x0]
                if np.median(np.linalg.norm((px - fcol) * [0.6, 1.0, 1.0], axis=1)) > 1.3 * ctol:
                    continue   # colour too different from the frame (e.g. a strand of hair): not a temple
            if ln > best_len:
                best, best_len = (xa + x0, ya + y0, xb + x0, yb + y0), ln
        if best is not None:
            cv2.line(m, best[:2], best[2:], 1, max(1, int(round(thick))))
    return ms


def morph(blob, r):
    """r > 0 dilates by r px, r < 0 erodes by |r| px."""
    if abs(r) < 0.5:
        return blob.copy()
    return cv2.dilate(blob, ellipse(r)) if r > 0 else cv2.erode(blob, ellipse(-r))


def build_mask(info, lab, lo, hi, cfg, temples="auto", grow=0.0):
    """Build the mask from this frame's measured frame position [lo, hi] (signed distance from the glasses outline, px).
    grow: extra widening in px during rework."""
    blob, lenses, gw = info["blob"], info["lenses"], info["gw"]
    m = cfg["rim_margin"] + grow
    lo, hi = lo - m, hi + m
    wmin, wmax = cfg["rim_min"] * gw + 2 * grow, cfg["rim_max"] * gw + 2 * grow
    if hi - lo < wmin:
        c = (hi + lo) / 2; lo, hi = c - wmin / 2, c + wmin / 2
    if hi - lo > wmax:
        c = (hi + lo) / 2; lo, hi = c - wmax / 2, c + wmax / 2
    rim = hi - lo
    band = morph(blob, hi) & (1 - morph(blob, lo))
    # local deviations: frame pixels detected near the main band are added too
    near = cv2.dilate(band, ellipse(0.5 * rim + 2))
    band |= info["rim_px"] & near
    # bridge
    if len(lenses) == 2:
        a, b = lenses
        ax2, bx1 = a["x"] + a["w"], b["x"]
        if bx1 > ax2:
            y1 = min(a["y"], b["y"]) - rim * 0.5
            y2 = (a["cy"] + b["cy"]) / 2
            cv2.rectangle(band, (int(ax2 - rim), int(y1)), (int(bx1 + rim), int(y2)), 1, -1)
    # temples
    if temples == "auto":
        if info["temple_px"].any():
            band |= cv2.dilate(info["temple_px"], ellipse(max(1, 0.25 * info["thick"] + 1 + grow)))
    elif temples == "on":
        band |= forced_temples(lenses, lab, rim, gw, cfg)
    band &= 1 - cv2.dilate(info["eyes"], ellipse(max(1, rim * 0.2)))
    return band


def forced_temples(lenses, lab, rim, gw, cfg):
    m = np.zeros(lab.shape, np.uint8)
    if len(lenses) < 2:
        return m
    allow = np.isin(lab, SKIN_LIKE).astype(np.uint8)
    for lens, side in ((lenses[0], -1), (lenses[1], 1)):
        yc = lens["y"] + lens["h"] * 0.3
        th = max(2, rim * 0.9)
        if side < 0:
            x0, x1 = lens["x"] - cfg["temple_len"] * gw, lens["x"] + rim
        else:
            x0, x1 = lens["x"] + lens["w"] - rim, lens["x"] + lens["w"] + cfg["temple_len"] * gw
        cv2.rectangle(m, (int(x0), int(yc - th)), (int(x1), int(yc + th)), 1, -1)
    return m & allow


# ----------------------------------------------------------------------------
# clip level: per-frame masks
# ----------------------------------------------------------------------------
DEFAULT_CFG = dict(
    kernel=0.035,       # line-detection kernel radius (relative to glasses width): only thinner lines count as frame
    search_out=0.035,   # how far outside the glasses outline to look for the frame
    search_in=0.06,     # how deep inside the glasses outline to look for the frame
    resp_floor=12.0,    # minimum line-response threshold (avoids false positives on smooth skin)
    rim_min=0.02,       # lower bound of the mask width (relative to glasses width)
    rim_max=0.09,       # upper bound of the mask width
    rim_margin=2.0,     # margin on each side of the measured frame (px)
    min_conf=0.25,      # below this share of outline points with a frame, the frame is unreliable and values from other frames of the clip are used
    fallback_lo=-0.035, # default position when nothing can be measured (relative to glasses width)
    fallback_hi=0.02,
    temple_len=0.4,     # temple search length (relative to glasses width)
    temple_min_rate=0.15,  # a temple counts as present (and is filled into the other frames) only if detected in at least this share of the clip
    hinge_y=0.3,        # approximate hinge position along the lens height (from the top)
    hinge_h=0.03,       # half height of the hinge bar (relative to glasses width)
    hinge_len=0.15,     # outward length of the hinge bar (relative to glasses width)
    smooth=3,           # temporal median filter radius (frames)
)


def smooth_series(vals, k):
    out = vals.copy()
    for i in range(len(vals)):
        w = vals[max(0, i - k): i + k + 1]
        w = w[~np.isnan(w)]
        if w.size:
            out[i] = float(np.median(w))
    return out


def frame_bounds(infos, cfg):
    """Per-frame frame position -> normalised by glasses width -> temporal median filter -> unreliable frames filled by interpolation / clip median."""
    lo = np.array([np.nan if (f is None or f["lo"] is None or f["conf"] < cfg["min_conf"]) else f["lo"] / f["gw"]
                   for f in infos])
    hi = np.array([np.nan if (f is None or f["hi"] is None or f["conf"] < cfg["min_conf"]) else f["hi"] / f["gw"]
                   for f in infos])
    if np.all(np.isnan(lo)):
        lo[:] = cfg["fallback_lo"]
        hi[:] = cfg["fallback_hi"]
    else:
        idx = np.arange(len(lo))
        ok = ~np.isnan(lo)
        lo = np.interp(idx, idx[ok], lo[ok])
        hi = np.interp(idx, idx[ok], hi[ok])
    lo, hi = smooth_series(lo, cfg["smooth"]), smooth_series(hi, cfg["smooth"])
    return lo, hi


def lens_anchor(info, si):
    lens = info["lenses"][si]
    x = lens["x"] if si == 0 else lens["x"] + lens["w"]
    return np.array([x, lens["y"]], np.float32)


def propagate_temples(infos, cfg):
    """Hinge / temple detections come and go. If one side was detected in enough frames of the clip,
    it really exists: frames without a detection get the most recent detection moved by the outer lens-edge displacement.
    Otherwise ProPainter would "copy" the hinge back from the frames where it is not covered."""
    for si in (0, 1):
        have = [i for i, f in enumerate(infos)
                if f is not None and len(f["lenses"]) == 2 and f["temple_sides"][si].any()]
        valid = [i for i, f in enumerate(infos) if f is not None and len(f["lenses"]) == 2]
        if not valid or len(have) < cfg["temple_min_rate"] * len(valid):
            continue
        have_arr = np.array(have)
        for i in valid:
            f = infos[i]
            if f["temple_sides"][si].any() or f.get("_propagated_" + str(si)):
                continue
            j = int(have_arr[np.argmin(np.abs(have_arr - i))])
            d = lens_anchor(f, si) - lens_anchor(infos[j], si)
            f["temple_sides"][si] = shift(infos[j]["temple_sides"][si], d)
            f["_propagated_" + str(si)] = True
    for f in infos:
        if f is not None and "temple_sides" in f:
            f["temple_px"] = f["temple_sides"][0] | f["temple_sides"][1]


def adaptive_masks(crops, parser, cfg, temples="auto", union=1, grow=None, infos=None, labs=None):
    """Adaptive masks for all crops of one clip.
    grow: extra widening in px per frame (for rework). infos/labs may be passed as a cache to avoid re-parsing.
    Returns (masks (0/255), infos, labs, stats)"""
    if infos is None:
        infos, labs = [], []
        for c in crops:
            lab, _ = parse_with_prob(parser, c)
            labs.append(lab)
            infos.append(analyze_frame(c, lab, cfg))
    lo, hi = frame_bounds(infos, cfg)
    propagate_temples(infos, cfg)
    if grow is None:
        grow = np.zeros(len(crops))
    masks, stats = [], []
    for i, (inf, lab) in enumerate(zip(infos, labs)):
        if inf is None:
            masks.append(np.zeros(lab.shape, np.uint8))
            stats.append(dict(glasses=0))
            continue
        gw = inf["gw"]
        m = build_mask(inf, lab, lo[i] * gw, hi[i] * gw, cfg, temples, grow[i])
        masks.append(m)
        stats.append(dict(glasses=1, gw=round(gw, 1), conf=round(inf["conf"], 2),
                          thick=round(inf["thick"], 1), lo=round(lo[i] * gw, 1),
                          hi=round(hi[i] * gw, 1), grow=float(grow[i]),
                          mask_pct=round(100 * m.mean(), 2)))
    masks = motion_union(masks, infos, union)
    return [m * 255 for m in masks], infos, labs, stats


def shift(mask, d):
    M = np.float32([[1, 0, d[0]], [0, 1, d[1]]])
    return cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)


def motion_union(masks, infos, k):
    """Union of the masks of +-k frames; neighbouring masks are first aligned by the glasses-centre shift."""
    if k <= 0:
        return masks
    out = []
    for i, inf in enumerate(infos):
        acc = masks[i].copy()
        for j in range(max(0, i - k), min(len(masks), i + k + 1)):
            if j == i or not masks[j].any():
                continue
            if inf is not None and infos[j] is not None:
                acc |= shift(masks[j], inf["centroid"] - infos[j]["centroid"])
            elif inf is None:
                acc |= masks[j]
        out.append(acc)
    return out


# ----------------------------------------------------------------------------
# scoring after inpainting
# ----------------------------------------------------------------------------
def residue_score(orig_info, out_img, parser, mask, orig_img=None, tau=10.0):
    """How much frame is left in this frame after inpainting.
    - line: share of the original frame pixels whose colour barely changed after inpainting and that still look like thin lines
      (a colour change means they were replaced; inpainted brows and eyelids also look like lines but differ in colour from the original frame, so they are not misjudged)
    - parse: area the face parsing still considers glasses after inpainting / original glasses area
    Returns (score, line, parse, residue_map); lower score is better."""
    if orig_info is None:
        return 0.0, 0.0, 0.0, None
    gw = orig_info["gw"]
    lab_out, _ = parse_with_prob(parser, out_img)
    resp = line_response(out_img, max(2, DEFAULT_CFG["kernel"] * gw))
    # only lines inside "the band where the frame was" count (reflections inside the lenses and brows do not)
    if orig_info["lo"] is not None and orig_info["conf"] >= DEFAULT_CFG["min_conf"]:
        lo, hi = orig_info["lo"], orig_info["hi"]
    else:
        lo, hi = DEFAULT_CFG["fallback_lo"] * gw, DEFAULT_CFG["fallback_hi"] * gw
    zone = morph(orig_info["blob"], hi + 4) & (1 - morph(orig_info["blob"], lo - 4))
    fp = ((orig_info["rim_px"] > 0) & (zone > 0)) | (orig_info["temple_px"] > 0)
    hit = (resp > orig_info["thr"] * 0.7) & fp
    if orig_img is not None:
        d = np.linalg.norm(cv2.cvtColor(orig_img, cv2.COLOR_BGR2LAB).astype(np.float32) -
                           cv2.cvtColor(out_img, cv2.COLOR_BGR2LAB).astype(np.float32), axis=2)
        hit &= d < tau
    rim_fp = (orig_info["rim_px"] > 0) & (zone > 0)
    tp = orig_info["temple_px"] > 0
    line = float((hit & rim_fp).sum()) / max(1, rim_fp.sum())
    if tp.sum() >= 15:   # hinges / temples scored separately: they are small and would be diluted inside the frame
        line = max(line, float((hit & tp).sum()) / tp.sum())
    parse = float((lab_out == GLASSES).sum()) / max(1, orig_info["blob"].sum())
    return max(line, parse), line, parse, hit.astype(np.uint8)
