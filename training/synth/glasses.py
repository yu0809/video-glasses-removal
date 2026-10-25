"""
Parametric 3D glasses: random parameters + geometry generated in a face-local coordinate frame
===============================================================================================
Local frame:
  origin O = point 168 (between the eyes); X = normalize(p263 - p33) (in a frontal face it points to image right = the subject's left);
  Y = (p152 - p10) with the X component removed (pointing down); Z = X x Y, oriented towards the camera; length unit E = |p263 - p33|.
  The local point (u, v, w) maps to the image point O + E(uX + vY + wZ); larger w is closer to the camera.
The right lens (subject's right eye) is on the u < 0 side. "Outwards" = away from the nose bridge: -u for the right lens, +u for the left.

The geometry is a set of "strokes" (3D polylines + width + part type) and the 3D outlines of the two lenses; rendering is in render.py.
"""
import numpy as np
from scipy.interpolate import LinearNDInterpolator

# ----------------------------------------------------------------------------
# local coordinate frame
# ----------------------------------------------------------------------------


class FaceFrame:
    def __init__(self, lm):
        lm = lm.astype(np.float64)
        self.O = lm[168].copy()
        X = lm[263] - lm[33]
        self.E = float(np.linalg.norm(X))
        X /= self.E
        Y = lm[152] - lm[10]
        Y -= X * (Y @ X)
        Y /= np.linalg.norm(Y)
        Z = np.cross(X, Y)
        if Z[2] > 0:          # smaller image z is closer, so "towards the camera" means a negative z component
            Z = -Z
        self.A = np.stack([X, Y, Z])      # row vectors
        self.L = self.to_local(lm)        # local coordinates of the 468 landmarks

    def to_local(self, P):
        return (np.asarray(P, np.float64) - self.O) @ self.A.T / self.E

    def to_img(self, L):
        """Local (u,v,w) -> image (x, y, z); z has the same scale as x, smaller is closer."""
        return self.O + (np.asarray(L, np.float64) * self.E) @ self.A

    def face_w(self):
        """Interpolator local (u,v) -> face surface w (nan outside the mesh)."""
        if not hasattr(self, "_fw"):
            self._fw = LinearNDInterpolator(self.L[:, :2], self.L[:, 2])
        return self._fw


# ----------------------------------------------------------------------------
# random parameters
# ----------------------------------------------------------------------------
METALS = {  # RGB
    "gold": (212, 175, 55), "silver": (196, 198, 204), "gunmetal": (82, 86, 94),
    "black_metal": (28, 28, 30), "rose_gold": (183, 110, 110),
}
ACETATES = ("black", "brown", "tortoise", "transparent", "color")


def sample_params(rng, style=None):
    p = {}
    p["style"] = style or rng.choice(["full", "half", "rimless", "acetate"], p=[0.55, 0.15, 0.10, 0.20])
    p["material"] = "acetate" if p["style"] == "acetate" else "metal"
    if p["style"] == "full" and rng.random() < 0.25:
        p["material"] = "acetate_thin"                   # thin acetate full rim (between metal and thick acetate)
    # lens shape
    p["a"] = rng.uniform(0.19, 0.30)
    p["b"] = p["a"] * rng.uniform(0.55, 1.0)
    p["n"] = rng.uniform(2.0, 6.0)
    p["rot"] = np.radians(rng.uniform(-6, 6))
    p["top_scale"] = rng.uniform(0.85, 1.0)
    p["bot_scale"] = rng.uniform(0.9, 1.1)
    p["cat"] = rng.uniform(0.1, 0.35) if rng.random() < 0.18 else 0.0
    p["aviator"] = rng.uniform(0.15, 0.4) if rng.random() < 0.12 else 0.0
    # position
    p["off_x"] = rng.uniform(-0.01, 0.04)
    p["off_y"] = rng.uniform(-0.06, 0.08)
    p["gap"] = rng.uniform(0.15, 0.26)                     # distance between the inner lens edges at the bridge (DBL)
    p["w0"] = rng.uniform(0.0, 0.06)
    p["tilt"] = np.radians(rng.uniform(4, 12))
    p["wrap"] = rng.uniform(0.0, 0.06)
    # thickness
    if p["material"] == "metal":
        p["t"] = rng.uniform(0.006, 0.02)
    elif p["material"] == "acetate_thin":
        p["t"] = rng.uniform(0.015, 0.03)
    else:
        p["t"] = rng.uniform(0.025, 0.07)
    p["depth"] = rng.uniform(0.02, 0.05) if p["material"] != "metal" else 0.0   # acetate front thickness (along -Z)
    # bridge, nose pads, hinges, temples
    p["bridge_arch"] = rng.uniform(0.0, 0.06) if rng.random() < 0.6 else 0.0
    p["double_bridge"] = bool(rng.random() < (0.6 if p["aviator"] else 0.1)) and p["material"] == "metal"
    p["pads"] = p["material"] == "metal" and rng.random() < 0.7
    p["hinge_h"] = rng.uniform(0.25, 0.33)
    p["hinge_len"] = rng.uniform(0.03, 0.08)
    p["temple_drop"] = rng.uniform(-0.02, 0.10)          # how much lower the temple ends than the hinge (top of the ear)
    p["temple_back"] = rng.uniform(0.05, 0.2)    # 234/454 lie in front of the tragus, the top of the ear is a bit further back; 0.3-0.6 drew temples that were too long
    p["temple_droop"] = rng.uniform(0.0, 0.03)
    p["temple_w"] = p["t"] * rng.uniform(0.9, 1.3) if p["material"] == "metal" else p["t"] * rng.uniform(0.5, 0.8)
    p["temple_w"] = max(p["temple_w"], 0.008)
    p["half_cut"] = rng.uniform(-0.1, 0.25)               # half rim: where the upper arc ends (relative to b)
    # colour and material
    if p["material"] == "metal":
        name = rng.choice(list(METALS))
        p["color_name"] = name
        p["color"] = np.clip(np.array(METALS[name], float) * rng.uniform(0.85, 1.15) + rng.normal(0, 8, 3), 0, 255)
    else:
        name = rng.choice(ACETATES, p=[0.35, 0.2, 0.2, 0.1, 0.15])
        p["color_name"] = name
        if name == "black":
            c = np.array([22, 22, 24.0]) + rng.uniform(0, 15)
        elif name == "brown":
            c = np.array([95, 60, 35.0]) * rng.uniform(0.6, 1.3)
        elif name == "tortoise":
            c = np.array([110, 70, 35.0])
        elif name == "transparent":
            c = np.array([225, 215, 200.0]) * rng.uniform(0.8, 1.0)
        else:
            import colorsys
            c = np.array(colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.4, 0.9), rng.uniform(0.25, 0.8))) * 255
        p["color"] = np.clip(c, 0, 255)
    p["opacity"] = rng.uniform(0.3, 0.6) if p.get("color_name") == "transparent" else 1.0
    p["pad_color"] = np.array([235, 235, 230.0]) if rng.random() < 0.7 else p["color"]
    p["hair_cover"] = [bool(rng.random() < 0.5), bool(rng.random() < 0.5)]   # whether the left / right temple is covered by hair
    # lens effects
    p["refract"] = rng.uniform(0.88, 1.03)
    p["tint"] = None
    if rng.random() < 0.3:
        p["tint"] = (np.array([(160, 120, 70), (110, 110, 110), (90, 120, 170)][rng.integers(3)], float),
                     rng.uniform(0.05, 0.25))
    p["reflect"] = rng.uniform(0.1, 0.4) if rng.random() < 0.5 else 0.0
    p["reflect_window"] = bool(rng.random() < 0.3)
    # lighting
    ang = rng.uniform(-150, -30) if rng.random() < 0.8 else rng.uniform(-180, 180)
    p["light"] = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
    p["n_spec"] = int(rng.integers(1, 4))
    p["shadow"] = rng.uniform(0.1, 0.3)
    p["shadow_off"] = rng.uniform(0.01, 0.04)
    return p


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------
def build_fitting(frame, p, tries=6):
    """build; when a lens cuts into the face, shrink the lenses by 10% and move them up by 0.02E, then retry (modifies p)."""
    for _ in range(tries):
        g = build(frame, p)
        if g is not None:
            return g
        p["a"] *= 0.9
        p["b"] *= 0.9
        p["off_y"] -= 0.02
    return build(frame, p, force=True)


def lens_outline(p, K=96):
    """Canonical lens outline (x' outwards, y' down, origin at the lens centre, unit E); returns (K,2)."""
    a, b, n = p["a"], p["b"], p["n"]
    th = np.linspace(0, 2 * np.pi, K, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    x = a * np.sign(c) * np.abs(c) ** (2 / n)
    y = b * np.sign(s) * np.abs(s) ** (2 / n)
    y = np.where(y < 0, y * p["top_scale"], y * p["bot_scale"])
    if p["cat"]:          # cat-eye: raised outer upper corner
        y -= p["cat"] * b * np.clip(x / a, 0, 1) ** 2 * (y < 0)
    if p["aviator"]:      # aviator: lower half stretched inwards and down
        k = np.clip(y / b, 0, None)
        inward = np.clip(0.6 - x / a, 0, 1.6) / 1.6
        y = y + p["aviator"] * b * k * inward
        x = x - 0.3 * p["aviator"] * a * k
    cr, sr = np.cos(p["rot"]), np.sin(p["rot"])
    return np.stack([cr * x - sr * y, sr * x + cr * y], 1)


def outline_normals(xy):
    """2D outward normal at every point of a closed polyline."""
    t = np.roll(xy, -1, 0) - np.roll(xy, 1, 0)
    nrm = np.stack([t[:, 1], -t[:, 0]], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    c = xy.mean(0)
    if ((xy - c) * nrm).sum() < 0:
        nrm = -nrm
    return nrm


def build(frame: FaceFrame, p, force=False):
    """Returns dict(strokes=[...], lenses=[(K,3) local coordinates x2], lens_centers, ...).
    Stroke: dict(pts=(N,3) local coordinates, w=width (E) or (N,) array, kind, closed, side (0 right / 1 left / -1 middle))."""
    L = frame.L
    fw = frame.face_w()
    eye_r = L[[33, 133]].mean(0)
    eye_l = L[[263, 362]].mean(0)
    half = (eye_l[0] - eye_r[0]) / 2 + p["off_x"]
    half = float(np.clip(half, p["gap"] / 2 + p["a"] * 0.95, 0.30 / 2 + p["a"]))
    u_mid = (eye_l[0] + eye_r[0]) / 2
    v_c = (eye_l[1] + eye_r[1]) / 2 + p["off_y"]
    base = lens_outline(p)
    nb = outline_normals(base)
    t = p["t"]

    def plane_w(u, v, w0):
        return w0 - np.tan(p["tilt"]) * (v - v_c) - p["wrap"] * (u - u_mid) ** 2

    def to3d(xy, side, w0, dw=0.0):
        sgn = -1 if side == 0 else 1                  # outwards is -u for the right lens
        u = u_mid + sgn * half + sgn * xy[:, 0]
        v = v_c + xy[:, 1]
        return np.stack([u, v, plane_w(u, v, w0) + dw], 1)

    # front plane: keep the frame at least 0.02E in front of the face surface (brow ridge, cheekbones)
    w0 = L[168, 2] + p["w0"]
    rim_xy = base + nb * (t / 2 if p["style"] != "rimless" else 0)
    # frame points near the bridge (|u| < 0.2E) are ignored: the nose sits between the lenses anyway, and including it would push the glasses 1-2 cm forward,
    # invisible frontally, but in profile the glasses would float in front of the nose tip. The forward push is at most 0.05E.
    push, pen = 0.0, 0.0
    for side in (0, 1):
        P = to3d(base + nb * t, side, w0)
        wf = fw(P[:, 0], P[:, 1])
        ok = ~np.isnan(wf)
        if ok.any():
            gap = wf + 0.02 - P[:, 2]
            far = ok & (np.abs(P[:, 0] - u_mid) > 0.2)
            if far.any():
                push = max(push, float(np.max(gap[far])))
            pen = max(pen, float(np.max(gap[ok])))
    push = float(np.clip(push, 0.0, 0.05))
    w0 += push
    # still cutting into the face after moving forward (usually lenses too big or too low, the lower rim pressing into the nose wings / cheekbones): such glasses cannot be worn, so let the caller shrink and retry
    if pen - push > 0.06 and not force:
        return None
    strokes = []
    lenses = [to3d(base, s, w0) for s in (0, 1)]

    def idx_near(cond, key):
        ii = np.where(cond)[0]
        return int(ii[np.argmin(key[ii])]) if len(ii) else int(np.argmin(key))

    b = p["b"]
    ys = base[:, 1]
    # positions of the bridge, hinges and nose pads on the canonical outline
    i_bridge = idx_near((ys > -0.6 * b) & (ys < -0.15 * b), base[:, 0])            # upper third of the inner edge
    hy = -b + p["hinge_h"] * 2 * b
    i_hinge = idx_near(np.abs(ys - hy) < 0.15 * b, -base[:, 0])                     # outer edge
    i_pad = idx_near((ys > 0.05 * b) & (ys < 0.6 * b), base[:, 0])                   # inner lower part
    i_top = idx_near(np.abs(base[:, 0] + 0.15 * p["a"]) < 0.3 * p["a"], ys)          # top, slightly inwards

    style = p["style"]
    for side in (0, 1):
        sgn = -1 if side == 0 else 1
        if style in ("full", "acetate"):
            layers = [0.0] if p["depth"] <= 0 else list(np.linspace(0, -p["depth"], 3))
            for k, dw in enumerate(layers):
                strokes.append(dict(pts=to3d(rim_xy, side, w0, dw), w=t, kind="rim" if k == 0 else "rim_side",
                                    closed=True, side=side, nrm=nb))
        elif style == "half":
            cut = p["half_cut"] * b
            top = rim_xy[:, 1] < cut
            order = np.roll(np.arange(len(base)), -int(np.argmax(~top & np.roll(top, 1))))
            seg = [i for i in order if top[i]]
            strokes.append(dict(pts=to3d(rim_xy[seg], side, w0), w=max(t, 0.012), kind="rim", closed=False,
                                side=side, nrm=nb[seg]))
            low = [i for i in order if not top[i]]
            low = [seg[-1]] + low + [seg[0]] if seg else low
            strokes.append(dict(pts=to3d(base[low], side, w0), w=0.003, kind="nylon", closed=False, side=side,
                                nrm=nb[low]))
        else:  # rimless: a thin bright line around the lens edge
            strokes.append(dict(pts=to3d(base, side, w0), w=0.004, kind="edge", closed=True, side=side, nrm=nb))

        # hinge: on the outer edge at 1/4-1/3 height, sticking outwards and backwards
        h0 = to3d(rim_xy[[i_hinge]] + nb[[i_hinge]] * t / 2, side, w0)[0]
        h1 = h0 + np.array([sgn * p["hinge_len"], 0, -p["hinge_len"] * 0.7])
        hw = max(1.6 * t, 0.018) if p["material"] == "metal" else max(t, 0.02)
        strokes.append(dict(pts=np.stack([h0, h1]), w=hw, kind="hinge", closed=False, side=side))

        # temple: hinge -> above the ear (234/454 shifted towards 127/356), bending slightly down, kept outside the head
        # ending towards 127/356 (temples) made the temples tilt upwards into vertical bars; real temples run roughly horizontally back from the hinge
        # to the top of the ear (about the height of the outer eye corner), so u, w come from the face-side points 234/454 and the height from the hinge, slightly lower
        ear = L[234] if side == 0 else L[454]
        end = np.array([ear[0], h1[1] + p["temple_drop"], ear[2] - p["temple_back"]])
        mid = (h1 + end) / 2 + np.array([0, p["temple_droop"], 0])
        s_ = np.linspace(0, 1, 28)[:, None]
        path = (1 - s_) ** 2 * h1 + 2 * s_ * (1 - s_) * mid + s_ ** 2 * end
        path = push_outside(path, L, side, clearance=0.015)
        tw = np.linspace(p["temple_w"] * (1.3 if p["material"] != "metal" else 1.0), p["temple_w"] * 0.8, len(path))
        strokes.append(dict(pts=path, w=tw, kind="temple", closed=False, side=side))

        # nose pads (metal frames): small ellipse + thin arm on the inner lower part of the lens, resting on the side of the nose
        if p["pads"]:
            r0 = to3d(rim_xy[[i_pad]], side, w0)[0]
            c = r0 + np.array([-sgn * 0.03, 0.02, -0.06])
            wfc = fw(c[0], c[1])
            if not np.isnan(wfc):
                c[2] = float(wfc) + 0.01
            d1 = np.array([0, 1.0, 0])
            d2 = np.array([-sgn * 0.5, 0, 0.85])
            d2 /= np.linalg.norm(d2)
            th = np.linspace(0, 2 * np.pi, 16, endpoint=False)[:, None]
            pad = c + 0.045 * np.cos(th) * d1 + 0.018 * np.sin(th) * d2
            strokes.append(dict(pts=np.stack([r0, (r0 + c) / 2 + np.array([0, 0.01, 0]), c]), w=0.006,
                                kind="pad_arm", closed=False, side=side))
            strokes.append(dict(pts=pad, w=0.012, kind="pad", closed=True, side=side, fill=True))

    # bridge (single arched / straight; double bridge adds a top bar)
    br = [to3d(rim_xy[[i_bridge]], s, w0)[0] for s in (0, 1)]
    s_ = np.linspace(0, 1, 12)[:, None]
    mid = (br[0] + br[1]) / 2 + np.array([0, -p["bridge_arch"], 0.01])
    bridge = (1 - s_) ** 2 * br[0] + 2 * s_ * (1 - s_) * mid + s_ ** 2 * br[1]
    bw = t * 1.1 if p["material"] == "metal" else t
    strokes.append(dict(pts=bridge, w=max(bw, 0.008), kind="bridge", closed=False, side=-1))
    if p["double_bridge"]:
        tp = [to3d(rim_xy[[i_top]], s, w0)[0] for s in (0, 1)]
        mid = (tp[0] + tp[1]) / 2 + np.array([0, -0.02, 0.01])
        bar = (1 - s_) ** 2 * tp[0] + 2 * s_ * (1 - s_) * mid + s_ ** 2 * tp[1]
        strokes.append(dict(pts=bar, w=max(t, 0.008), kind="bridge", closed=False, side=-1))
    return dict(strokes=strokes, lenses=lenses, w0=w0, v_c=v_c, half=half, u_mid=u_mid)


def push_outside(path, L, side, clearance=0.015):
    """Temples must not cut into the head and should be straight bars: the start (hinge) stays fixed and only the end moves outwards,
    so that |u| at every path point is at least the maximum |u| of nearby face points (same side, similar v and w) plus a margin.
    Moving the end out by delta moves a point at fraction s along the path by s*delta, so delta = max(gap / s)."""
    sgn = -1 if side == 0 else 1
    pts = L[(L[:, 0] * sgn) > 0.1]
    n = len(path)
    s_ = np.linspace(0, 1, n)
    delta = 0.0
    for i in range(1, n):
        q = path[i]
        near = pts[(np.abs(pts[:, 1] - q[1]) < 0.12) & (np.abs(pts[:, 2] - q[2]) < 0.15)]
        if len(near) and s_[i] > 0.35:      # points near the hinge have tiny gaps; dividing by a small s would push the end very far
            gap = float(np.abs(near[:, 0]).max()) + clearance - abs(q[0])
            delta = max(delta, gap / s_[i])
    delta = min(delta, 0.2)
    out = path.copy()
    out[:, 0] += sgn * delta * s_
    return out
