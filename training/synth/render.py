"""
Render the parametric 3D glasses onto a face crop
=================================================
render_sample(face_img, landmarks, parsing, rng) -> dict(image, frame_mask, lens_mask, clean, shadow_mask, params)

- anti-aliased polylines drawn at SS x supersampling, downscaled with INTER_AREA into a soft alpha
- occlusion: the FaceMesh triangles are rasterised into a depth map; every small glasses segment (<= 2 px) compares depth at its pixel and is hidden when the face is closer.
  In profile views the far temple, nose pads hidden by the nose and the far lens edge disappear naturally.
  Outside the mesh: temples crossing hair are covered with 50% probability; the far temple is hidden where it crosses the head (skin / hair / ears).
- materials: metal (shading + highlight line + dark edge + 1-3 sharp specular spots), acetate (black / brown / tortoiseshell / clear / coloured, side thickness, highlight on the upper inner edge)
- shadow: blurred frame alpha shifted along the light direction darkens the skin; output separately as shadow_mask, not part of the frame label
- lenses: refraction scaling, tint, reflections (only pixels inside the lenses change, not part of the frame label)
Labels: frame_mask = soft coverage of rims, bridge, nose pads, hinges, visible temples (and the bright lens-edge line of rimless styles); lens_mask = lens interior.
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from repose.mesh_reproject import extend_points, tris  # noqa: E402
from synth.glasses import FaceFrame, build_fitting, sample_params  # noqa: E402

SS = 3                      # supersampling factor (3 and 4 look identical, 3 is twice as fast)
HEAD_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18]
SKIN_CLASSES = [1, 2, 3, 10]
HAIR = 17


def zbuffer(lm, shape):
    """Face mesh depth map (image z, smaller is closer; +inf outside the mesh). Painter's algorithm: far triangles first, planar depth."""
    H, W = shape
    P = extend_points(lm.astype(np.float64))
    T = tris()
    zt = P[T][:, :, 2].mean(1)
    zb = np.full((H, W), np.inf, np.float32)
    pts = np.round(P[:, :2] * 16).astype(np.int32)
    for k in np.argsort(-zt):
        cv2.fillConvexPoly(zb, pts[T[k]], float(zt[k]), lineType=cv2.LINE_8, shift=4)
    return zb


def _rgb2bgr(c):
    return np.asarray(c, np.float32)[::-1]


def _densify(P, closed, max_len):
    """Subdivide polylines (already projected to image coordinates, extra columns allowed) by 2D length, at most max_len px per segment."""
    if closed:
        P = np.vstack([P, P[:1]])
    seg = P[1:] - P[:-1]
    n = np.maximum(1, np.ceil(np.linalg.norm(seg[:, :2], axis=1) / max_len).astype(int))
    k = np.repeat(np.arange(len(seg)), n)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 1) / np.repeat(n, n)
    return np.vstack([P[:1], P[k] + t[:, None] * seg[k]])


class Canvas:
    """Supersampled canvas over a local region: coverage alpha, premultiplied colour, label coverage."""

    def __init__(self, x0, y0, w, h):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.a = np.zeros((h * SS, w * SS), np.float32)
        self.c = np.zeros((h * SS, w * SS, 3), np.float32)
        self.lab = np.zeros((h * SS, w * SS), np.float32)

    def pt(self, x, y):
        return (int(round((x - self.x0) * SS * 16)), int(round((y - self.y0) * SS * 16)))

    def line(self, p0, p1, width_px, color, alpha=1.0, label=1.0):
        th = max(1, int(round(width_px * SS)))
        a, b = self.pt(*p0), self.pt(*p1)
        if alpha > 0:
            cv2.line(self.a, a, b, float(alpha), th, cv2.LINE_8, 4)
            cv2.line(self.c, a, b, tuple(float(v) * alpha for v in color), th, cv2.LINE_8, 4)
        if label > 0:
            cv2.line(self.lab, a, b, float(label), th, cv2.LINE_8, 4)

    def polys(self, lines, width_px, color, alpha=1.0, label=1.0):
        """Draw several polylines at once (same width and colour)."""
        th = max(1, int(round(width_px * SS)))
        q = [np.round((np.asarray(l)[:, :2] - [self.x0, self.y0]) * SS * 16).astype(np.int32) for l in lines]
        if alpha > 0:
            cv2.polylines(self.a, q, False, float(alpha), th, cv2.LINE_8, 4)
            cv2.polylines(self.c, q, False, tuple(float(v) * alpha for v in color), th, cv2.LINE_8, 4)
        if label > 0:
            cv2.polylines(self.lab, q, False, float(label), th, cv2.LINE_8, 4)

    def poly(self, pts, color, alpha=1.0, label=1.0):
        q = np.array([self.pt(x, y) for x, y in pts], np.int32)
        cv2.fillPoly(self.a, [q], float(alpha), cv2.LINE_8, 4)
        cv2.fillPoly(self.c, [q], tuple(float(v) * alpha for v in color), cv2.LINE_8, 4)
        if label > 0:
            cv2.fillPoly(self.lab, [q], float(label), cv2.LINE_8, 4)

    def dot(self, x, y, r_px, color, alpha=1.0):
        cv2.circle(self.a, self.pt(x, y), max(1, int(r_px * SS * 16)), float(alpha), -1, cv2.LINE_8, 4)
        cv2.circle(self.c, self.pt(x, y), max(1, int(r_px * SS * 16)), tuple(float(v) * alpha for v in color),
                   -1, cv2.LINE_8, 4)

    def resolve(self):
        sz = (self.w, self.h)
        a = cv2.resize(self.a, sz, interpolation=cv2.INTER_AREA)
        c = cv2.resize(self.c, sz, interpolation=cv2.INTER_AREA)
        lab = cv2.resize(self.lab, sz, interpolation=cv2.INTER_AREA)
        return np.clip(a, 0, 1), c, np.clip(lab, 0, 1)


def _noise(rng, h, w, scales=((6, 1.0), (20, 0.5))):
    out = np.zeros((h, w), np.float32)
    for s, amp in scales:
        out += amp * cv2.resize(rng.random((s, s)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    out -= out.min()
    return out / (out.max() + 1e-6)


def render_sample(face_img, landmarks, parsing, rng, params=None, draw_glasses=True):
    H, W = face_img.shape[:2]
    clean = face_img
    img = face_img.astype(np.float32)
    frame = FaceFrame(landmarks)
    E = frame.E
    p = params if params is not None else sample_params(rng)
    out = dict(clean=clean, params=p)
    if not draw_glasses:
        z = np.zeros((H, W), np.float32)
        out.update(image=clean.copy(), frame_mask=z, lens_mask=z.copy(), shadow_mask=z.copy())
        return out
    g = build_fitting(frame, p)
    zb = zbuffer(landmarks, (H, W))
    tol = 0.012 * E                 # temples, nose pads: strict occlusion
    tol_front = 0.06 * E            # front frame (rims, bridge, hinges, lenses): slight "penetration" into cheeks or nose wings counts as visible;
                                    # in profile the far rim behind the nose (depth gap ~0.3E) is still hidden
    X = frame.A[0]
    light = p["light"]

    # project all parts to determine the canvas extent
    proj = []
    for s in g["strokes"]:
        P = frame.to_img(s["pts"])
        proj.append(P)
    lens_img = [frame.to_img(Lp) for Lp in g["lenses"]]
    allp = np.vstack(proj + lens_img)
    pad = int(0.1 * E) + 4
    x0 = int(max(0, np.floor(allp[:, 0].min()) - pad))
    y0 = int(max(0, np.floor(allp[:, 1].min()) - pad))
    x1 = int(min(W, np.ceil(allp[:, 0].max()) + pad))
    y1 = int(min(H, np.ceil(allp[:, 1].max()) + pad))
    if x1 - x0 < 4 or y1 - y0 < 4:
        z = np.zeros((H, W), np.float32)
        out.update(image=clean.copy(), frame_mask=z, lens_mask=z.copy(), shadow_mask=z.copy())
        return out
    cv_ = Canvas(x0, y0, x1 - x0, y1 - y0)

    color = _rgb2bgr(p["color"])
    metal = p["material"] == "metal"
    white = np.array([255, 255, 255], np.float32)
    hl_col = 0.45 * color + 0.55 * white
    dark_col = 0.45 * color

    head_cls = np.zeros(256, bool)
    head_cls[HEAD_CLASSES] = True

    def visible(m, kind, side):
        """Whether the midpoint of each small segment is visible (vectorised)."""
        xi, yi = m[:, 0].astype(np.int64), m[:, 1].astype(np.int64)
        inb = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        xi, yi = np.clip(xi, 0, W - 1), np.clip(yi, 0, H - 1)
        zf = zb[yi, xi]
        fin = np.isfinite(zf)
        t_ = tol if kind in ("temple", "pad", "pad_arm") else tol_front
        v = inb & (~fin | (m[:, 2] <= zf + t_))
        if kind == "temple":
            cls = parsing[yi, xi]
            if p["hair_cover"][side]:
                v &= ~(~fin & (cls == HAIR))
            sgn = -1 if side == 0 else 1
            if sgn * X[2] > 0.05:                       # the far temple is behind the head where it crosses the head region
                v &= ~(~fin & head_cls[cls])
        return v

    def runs(D, keep, key):
        """Merge consecutive visible segments with the same key into polylines: returns {key: [polyline points (n,2)]}."""
        groups = {}
        i, n = 0, len(keep)
        while i < n:
            if not keep[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and keep[j + 1] and key[j + 1] == key[i]:
                j += 1
            groups.setdefault(key[i], []).append(D[i:j + 2])
            i = j + 1
        return groups

    def offset_poly(q, dist):
        """Offset every polyline point by dist px along its own normal."""
        t = np.gradient(q[:, :2], axis=0)
        nn = np.stack([-t[:, 1], t[:, 0]], 1)
        nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-9
        return q[:, :2] + nn * dist

    op = p["opacity"]
    spec_cands = []
    for s, P in zip(g["strokes"], proj):
        kind, side = s["kind"], s["side"]
        wpx = np.broadcast_to(np.asarray(s["w"], np.float64) * E, (len(P),))
        if s.get("fill"):
            if visible(P.mean(0, keepdims=True), kind, side)[0]:
                cv_.poly(P[:, :2], _rgb2bgr(p["pad_color"]), alpha=0.75, label=1.0)
            continue
        D = _densify(np.hstack([P, wpx[:, None]]), s["closed"], 1.5)
        if len(D) < 2:
            continue
        mids = (D[:-1] + D[1:]) / 2
        vis = visible(mids, kind, side)
        if not vis.any():
            continue
        d = D[1:, :2] - D[:-1, :2]
        perp = np.stack([-d[:, 1], d[:, 0]], 1) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        lit = np.abs(perp @ light)
        wq = np.maximum(1, np.round(mids[:, 3] * SS)).astype(int)
        nrm2 = None
        if s.get("nrm") is not None and kind in ("rim", "edge"):
            nrm_l = np.vstack([s["nrm"], s["nrm"][:1]]) if s["closed"] else s["nrm"]
            sgn = -1 if side == 0 else 1
            n3 = (sgn * nrm_l[:, :1]) * frame.A[0] + nrm_l[:, 1:2] * frame.A[1]
            idx = np.linspace(0, len(n3) - 1, len(mids)).round().astype(int)
            nrm2 = n3[idx, :2]
            nrm2 /= np.linalg.norm(nrm2, axis=1, keepdims=True) + 1e-9
        if kind == "nylon":
            for wk, pl in runs(D, vis, wq).items():
                # very faint nylon line: label strength matches its visibility (0.35), so it does not count as frame after binarisation (> 0.5)
                cv_.polys(pl, max(wk / SS, 0.5), (205, 205, 205), alpha=0.35, label=0.35)
            continue
        if kind == "edge":
            lev = np.round(4 * (0.12 + 0.5 * np.maximum(0, nrm2 @ light))).astype(int) if nrm2 is not None \
                else np.full(len(mids), 2)
            for k, pl in runs(D, vis, list(zip(wq, lev))).items():
                # bright lens-edge line of rimless styles: the network should not be asked to find invisible parts, so the label follows visibility (bright segments ~1, dark ones ~0.2)
                cv_.polys(pl, max(k[0] / SS, 0.6), white * 0.95, alpha=k[1] / 4, label=min(1.0, 1.6 * k[1] / 4))
            continue
        if kind == "rim_side":
            for wk, pl in runs(D, vis, wq).items():
                cv_.polys(pl, wk / SS, color * 0.6, alpha=op, label=1.0)
            continue
        lev = np.digitize(lit, [0.33, 0.66]) if metal else np.zeros(len(mids), int)
        for (wk, lv), pl in runs(D, vis, list(zip(wq, lev))).items():
            w = wk / SS
            lt = (0.2, 0.5, 0.85)[lv]
            cv_.polys(pl, w, color * (0.7 + 0.25 * lt) if metal else color, alpha=op, label=1.0)
            if metal:
                off = light * 0.22 * w            # highlight towards the light, dark edge on the other side
                cv_.polys([q[:, :2] + off for q in pl], 0.4 * w, hl_col * (0.8 + 0.3 * lt), alpha=1.0, label=0)
                cv_.polys([q[:, :2] - off for q in pl], 0.3 * w, dark_col, alpha=1.0, label=0)
            elif kind in ("rim", "bridge", "temple", "hinge"):
                # acetate: rounded shading (bright towards the light, dark away from it) + one soft reflection
                off = light * 0.25 * w
                cv_.polys([q[:, :2] - off for q in pl], 0.35 * w, color * 0.7, alpha=op, label=0)
                cv_.polys([q[:, :2] + off for q in pl], 0.3 * w, color * 0.75 + white * 0.25, alpha=op, label=0)
                cv_.polys([q[:, :2] + off * 1.4 for q in pl], max(0.08 * w, 0.4), color * 0.4 + white * 0.6,
                          alpha=0.6 * op, label=0)
            if not metal and p["color_name"] == "transparent" and kind in ("rim", "bridge"):
                for sg in (-1, 1):
                    cv_.polys([offset_poly(q, sg * 0.45 * w) for q in pl], max(0.15 * w, 0.5), white,
                              alpha=0.6, label=0)
        if not metal and kind == "rim" and nrm2 is not None:
            top = vis & (nrm2[:, 1] < -0.5)            # thin highlight on the upper inner edge of acetate
            for wk, pl in runs(D, top, wq).items():
                cv_.polys([offset_poly(q, 0.3 * wk / SS) for q in pl], 0.2 * wk / SS,
                          color * 0.6 + white * 0.4, alpha=op, label=0)
        if metal and kind == "rim" and nrm2 is not None:
            ii = np.where(vis & (nrm2 @ light > 0.6))[0]
            spec_cands += [(mids[i, 0], mids[i, 1], mids[i, 3]) for i in ii]
    if metal and spec_cands:
        for k in rng.choice(len(spec_cands), min(p["n_spec"], len(spec_cands)), replace=False):
            x, y, w = spec_cands[k]
            cv_.dot(x, y, max(0.5 * w, 0.6), white, alpha=1.0)

    alpha, col_pm, lab = cv_.resolve()
    if p["material"] != "metal":           # subtle colour variation on acetate surfaces
        col_pm = col_pm * (0.92 + 0.16 * _noise(rng, cv_.h, cv_.w, ((12, 1.0), (40, 0.5))))[..., None]
    if p["material"] != "metal" and p["color_name"] == "tortoise":
        n = _noise(rng, cv_.h, cv_.w)
        t = np.clip((n - 0.35) * 2.5, 0, 1)[..., None]
        tex = _rgb2bgr((45, 22, 10)) * (1 - t) + _rgb2bgr((185, 115, 45)) * t
        col_pm = tex * alpha[..., None] * 0.85 + col_pm * 0.15

    # lens region (visible part)
    lens = np.zeros((cv_.h * SS, cv_.w * SS), np.float32)
    for Pl in lens_img:
        q = np.array([cv_.pt(x, y) for x, y in Pl[:, :2]], np.int32)
        cv2.fillPoly(lens, [q], 1.0, cv2.LINE_8, 4)
    lens = cv2.resize(lens, (cv_.w, cv_.h), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    lens_z = np.full(lens.shape, np.inf, np.float32)
    for Pl in lens_img:               # fit a planar depth per lens
        Aq = np.c_[Pl[:, 0], Pl[:, 1], np.ones(len(Pl))]
        coef = np.linalg.lstsq(Aq, Pl[:, 2], rcond=None)[0]
        poly = np.array([(x - x0, y - y0) for x, y in Pl[:, :2]], np.float32)
        inside = np.zeros(lens.shape, np.uint8)
        cv2.fillPoly(inside, [np.round(poly).astype(np.int32)], 1)
        inside = cv2.dilate(inside, np.ones((3, 3), np.uint8)).astype(bool)
        lens_z[inside] = np.minimum(lens_z[inside], (coef[0] * xx + coef[1] * yy + coef[2])[inside])
    zr = zb[y0:y1, x0:x1]
    lens_vis = ~(np.isfinite(zr) & (lens_z > zr + tol_front))
    lens = lens * lens_vis

    region = img[y0:y1, x0:x1]
    region = _lens_effects(region, lens, lens_img, x0, y0, p, rng)

    # shadow (falls on skin only, not part of the frame label)
    sh = cv2.GaussianBlur(alpha, (0, 0), max(0.8, 0.012 * E))
    off = -light * p["shadow_off"] * E      # light comes from the light direction, the shadow falls the opposite way
    M = np.float32([[1, 0, off[0]], [0, 1, off[1]]])
    sh = cv2.warpAffine(sh, M, (cv_.w, cv_.h))
    skin = np.isin(parsing[y0:y1, x0:x1], SKIN_CLASSES).astype(np.float32)
    sh = sh * skin * (1 - alpha)
    region = region * (1 - p["shadow"] * sh[..., None])

    # composite the frame
    a3 = alpha[..., None]
    col = col_pm / np.maximum(a3, 1e-4)
    region = region * (1 - a3) + col * a3
    img[y0:y1, x0:x1] = region

    def full(m):
        f = np.zeros((H, W), np.float32)
        f[y0:y1, x0:x1] = m
        return f

    out.update(image=np.clip(img, 0, 255).astype(np.uint8), frame_mask=full(lab), lens_mask=full(lens),
               shadow_mask=full(sh))
    return out


def _lens_effects(region, lens, lens_img, x0, y0, p, rng):
    """Refraction scaling, tint, reflections: only inside the lenses."""
    if lens.max() <= 0:
        return region
    ys, xs = np.nonzero(lens > 0)
    by0, by1 = max(0, ys.min() - 3), min(lens.shape[0], ys.max() + 4)
    bx0, bx1 = max(0, xs.min() - 3), min(lens.shape[1], xs.max() + 4)
    sub = _lens_effects_box(region[by0:by1, bx0:bx1], lens[by0:by1, bx0:bx1], lens_img, x0 + bx0, y0 + by0, p, rng)
    region = region.copy()
    region[by0:by1, bx0:bx1] = sub
    return region


def _lens_effects_box(region, lens, lens_img, x0, y0, p, rng):
    h, w = lens.shape
    m = cv2.GaussianBlur(lens, (0, 0), 1.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mx, my = xx.copy(), yy.copy()
    s = p["refract"]
    for Pl in lens_img:
        c = Pl[:, :2].mean(0) - [x0, y0]
        r = np.ptp(Pl[:, :2], axis=0).max() * 0.75       # each lens only scales its own neighbourhood
        near = m * (((xx - c[0]) ** 2 + (yy - c[1]) ** 2) < r * r)
        mx += near * ((c[0] + (xx - c[0]) / s) - xx)
        my += near * ((c[1] + (yy - c[1]) / s) - yy)
    region = cv2.remap(region, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m3 = m[..., None]
    if p["tint"] is not None:
        tc, k = p["tint"]
        region = region * (1 - k * m3) + _rgb2bgr(tc) * k * m3
    if p["reflect"] > 0:
        R = np.zeros((h, w), np.float32)
        for Pl in lens_img:
            c = Pl[:, :2].mean(0) - [x0, y0]
            ext = np.ptp(Pl[:, :2], axis=0)
            if p["reflect_window"]:
                cx, cy = c + rng.uniform(-0.3, 0.3, 2) * ext
                ww, hh = ext * rng.uniform(0.15, 0.35, 2)
                for dx in (-0.55, 0.55):
                    for dy in (-0.55, 0.55):
                        cv2.rectangle(R, (int(cx + dx * ww - ww / 2.2), int(cy + dy * hh - hh / 2.2)),
                                      (int(cx + dx * ww + ww / 2.2), int(cy + dy * hh + hh / 2.2)), 1.0, -1)
                R = cv2.GaussianBlur(R, (0, 0), max(1.5, 0.08 * float(ext.max()))) * rng.uniform(0.4, 0.8)
            else:
                ang = rng.uniform(0, np.pi)
                d = np.array([np.cos(ang), np.sin(ang)])
                t = ((xx - c[0]) * d[0] + (yy - c[1]) * d[1]) / (ext.max() / 2 + 1e-6)
                R = np.maximum(R, np.clip(1 - np.abs(t - rng.uniform(-0.5, 0.5)) / rng.uniform(0.2, 0.6), 0, 1))
        R = R * m * p["reflect"]
        region = 255 - (255 - region) * (1 - R[..., None])
    return region
