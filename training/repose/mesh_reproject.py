#!/usr/bin/env python3
"""
Method B: 3D re-projection based on the face mesh ("face profiling")
====================================================================
Rotates the face (and the 3D glasses placed on it later) by d_pitch, d_yaw around the head centre and projects it back into the image:
  - inside the mesh: per-triangle affine warping of the FaceMesh triangles (painter's algorithm, far first, back-face culling)
  - outside the mesh (hair, ears, neck, background): smooth thin-plate-spline (TPS) extrapolation of the face-contour displacement,
    with the image border fixed
  - triangles newly exposed by the rotation and invisible in the source (facing away or heavily foreshortened): filled with OpenCV inpaint,
    recorded in uncertain_mask and down-weighted during training
Landmarks are obtained by rotating the source landmarks directly (small source angle, accurate landmarks), so the glasses labels are exact.
Keep a single rotation within +-30 deg.

  python training/repose/mesh_reproject.py batch --pool data/faces_pool --n 400 --max_delta 30
  python training/repose/mesh_reproject.py demo --pool data/faces_pool --id 000026 --out x.jpg

batch writes the results straight into the source pool (method=reproject, with an extra uncertain array in the npz) and re-checks the pose with HeadPose.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import face_triangles, mesh_pose  # noqa: E402
from repose.mesh3d import head_center, kps5_from_mesh, rot_matrix, rotate  # noqa: E402
from repose.targets import pick_base, sparse_cells  # noqa: E402

# vertices of MediaPipe FACEMESH_FACE_OVAL (outer face contour, clockwise)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
# the FaceMesh mesh is empty inside the eyes and mouth; these three rings plus a centre point each are closed with fan triangles
HOLES = [
    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],          # right eye
    [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],       # left eye
    [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191],  # inner lips
]
_TRIS = None


def tris():
    """Triangles of the 468 points + hole-filling triangles (the hole centre points are 468, 469, 470)."""
    global _TRIS
    if _TRIS is None:
        t = [face_triangles()]
        for h, loop in enumerate(HOLES):
            c = 468 + h
            t.append(np.array([[loop[i], loop[(i + 1) % len(loop)], c] for i in range(len(loop))], np.int32))
        _TRIS = np.vstack(t)
    return _TRIS


def extend_points(P):
    """Append the centre points of the eyes and mouth after the 468 points."""
    return np.vstack([P[:468]] + [P[loop].mean(0, keepdims=True) for loop in HOLES])


# face-parsing class -> how much the outer virtual points follow the head rotation: hair, ears, hats, earrings fully; neck, necklace, clothes slightly; background not at all
FOLLOW = np.zeros(19)
FOLLOW[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18]] = 1.0
FOLLOW[[14, 15, 16]] = 0.2


def skull_rings(P, c, parsing=None, scales=(1.15, 1.35, 1.6), decay=(1.0, 0.65, 0.3)):
    """Rings of virtual points outside the face contour (around hair and ears) that rotate rigidly with the head so the TPS extrapolation looks more like a turning head.
    Returns (points, follow weights). With face parsing the weights follow the class; without it, points below the chin follow only 0.25."""
    oval = P[FACE_OVAL]
    chin_y = P[152, 1]
    pts, w = [], []
    for s, dk in zip(scales, decay):
        q = oval.copy()
        q[:, :2] = c[:2] + s * (oval[:, :2] - c[:2])
        q[:, 2] = oval[:, 2] + (s - 1) * 0.8 * (c[2] - oval[:, 2].mean())
        pts.append(q)
        if parsing is None:
            wk = np.where(q[:, 1] > chin_y, 0.25, 1.0)
        else:
            H, W = parsing.shape
            xi = np.clip(np.round(q[:, 0]).astype(int), 0, W - 1)
            yi = np.clip(np.round(q[:, 1]).astype(int), 0, H - 1)
            wk = FOLLOW[np.clip(parsing[yi, xi], 0, 18)]
        # smooth along each ring (one point following fully next to one fixed point folds the TPS); outer rings follow less
        for _ in range(2):
            wk = (np.roll(wk, 1) + 2 * wk + np.roll(wk, -1)) / 4
        w.append(dk * wk)
    return np.vstack(pts), np.concatenate(w)


def tri_normals_z(P, T, center):
    """z component of each triangle's outward normal (< 0 faces the camera)."""
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    n = np.cross(b - a, c - a)
    out = (a + b + c) / 3 - center
    n *= np.sign((n * out).sum(1, keepdims=True) + 1e-9)
    return n[:, 2] / (np.linalg.norm(n, axis=1) + 1e-9)


def tps_map(src_pts, dst_pts, shape, grid=48):
    """Backward mapping: every target pixel -> source coordinates. Control points sit at dst_pts in the target and src_pts in the source.
    The thin-plate spline is evaluated on a coarse grid and upsampled (fast)."""
    from scipy.interpolate import RBFInterpolator
    H, W = shape
    rbf = RBFInterpolator(dst_pts, src_pts - dst_pts, kernel="thin_plate_spline", smoothing=5.0)
    gy, gx = np.meshgrid(np.linspace(0, H - 1, grid), np.linspace(0, W - 1, grid), indexing="ij")
    d = rbf(np.stack([gx.ravel(), gy.ravel()], 1)).reshape(grid, grid, 2).astype(np.float32)
    d = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return xx + d[..., 0], yy + d[..., 1]


def border_points(H, W, n=6):
    t = np.linspace(0, 1, n)
    pts = [(x * (W - 1), 0) for x in t] + [(x * (W - 1), H - 1) for x in t] + \
          [(0, y * (H - 1)) for y in t[1:-1]] + [(W - 1, y * (H - 1)) for y in t[1:-1]]
    return np.array(pts, np.float64)


def reproject(img, lm, dpitch, dyaw, parsing=None, inpaint=True):
    """Returns dict(image, landmarks, parsing, uncertain (0/1 uint8), map_x, map_y).
    map_x/map_y is the target -> source backward mapping, usable to warp other layers aligned with the source image the same way."""
    H, W = img.shape[:2]
    T = tris()
    P0 = extend_points(lm.astype(np.float64))
    c = head_center(P0)
    R = rot_matrix(dpitch, dyaw, 0.0)
    P1 = rotate(P0, R, c)

    # outside the mesh: TPS (face contour + two rings of virtual skull points + fixed image border)
    border = border_points(H, W)
    ring0, wr = skull_rings(P0, c, parsing)
    ring1 = rotate(ring0, R, c)
    ring1[:, :2] = ring0[:, :2] + wr[:, None] * (ring1[:, :2] - ring0[:, :2])
    keep = (ring0[:, 0] > -0.1 * W) & (ring0[:, 0] < 1.1 * W) & (ring0[:, 1] > -0.1 * H) & (ring0[:, 1] < 1.1 * H)
    dst = np.vstack([P1[FACE_OVAL, :2], ring1[keep, :2], border])
    src = np.vstack([P0[FACE_OVAL, :2], ring0[keep, :2], border])
    map_x, map_y = tps_map(src, dst, (H, W))

    # inside the mesh: per-triangle affine (painter's algorithm, far first)
    nz1 = tri_normals_z(P1, T, c)
    nz0 = tri_normals_z(P0, T, c)
    vis = np.where(nz1 < -0.02)[0]
    order = vis[np.argsort(-P1[T[vis]][:, :, 2].mean(1))]
    tid = np.full((H, W), -1, np.int32)
    for k in order:
        cv2.fillConvexPoly(tid, np.round(P1[T[k], :2] * 16).astype(np.int32), int(k), lineType=cv2.LINE_8, shift=4)
    # target -> source affine matrix of every triangle (vectorised solve)
    A1 = np.concatenate([P1[T][:, :, :2], np.ones((len(T), 3, 1))], 2)       # (n,3,3)
    B0 = P0[T][:, :, :2]                                                      # (n,3,2)
    ok = np.abs(np.linalg.det(A1)) > 1e-6
    Aff = np.zeros((len(T), 3, 2))
    Aff[ok] = np.linalg.solve(A1[ok], B0[ok])
    ys, xs = np.nonzero(tid >= 0)
    k = tid[ys, xs]
    good = ok[k]
    ys, xs, k = ys[good], xs[good], k[good]
    pts = np.stack([xs, ys, np.ones_like(xs)], 1).astype(np.float64)
    sxy = np.einsum("ni,nij->nj", pts, Aff[k])
    map_x[ys, xs] = sxy[:, 0]
    map_y[ys, xs] = sxy[:, 1]

    # triangles invisible in the source (facing away or heavily foreshortened) -> untrustworthy
    def area(P):
        a, b, cc = P[T[:, 0], :2], P[T[:, 1], :2], P[T[:, 2], :2]
        return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (cc[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (cc[:, 0] - a[:, 0]))
    stretch = area(P1) / (area(P0) + 1e-6)
    bad_tri = (nz0 > -0.05) | (stretch > 3.0)
    uncertain = np.zeros((H, W), np.uint8)
    sel = (tid >= 0)
    uncertain[sel] = bad_tri[tid[sel]].astype(np.uint8)

    out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if inpaint and uncertain.any():
        uncertain = cv2.dilate(uncertain, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        out = cv2.inpaint(out, uncertain, 5, cv2.INPAINT_TELEA)
    res = dict(image=out, landmarks=P1[:468].astype(np.float32), uncertain=uncertain, map_x=map_x, map_y=map_y)
    if parsing is not None:
        res["parsing"] = cv2.remap(parsing, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    return res


# ----------------------------------------------------------------------------
def load_pool(pool):
    with open(os.path.join(pool, "index.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]


def pick_delta(rng, p0, max_delta, lim_p=50, lim_y=75, grid_counts=None):
    """Pick a delta so the result lands in a sparsely populated pitch x yaw cell."""
    best, best_c = None, None
    for _ in range(12):
        d = rng.uniform(-max_delta, max_delta, 2)
        if rng.random() < 0.5:                       # half the time rotate both axes by a fair amount
            d = np.sign(d) * rng.uniform(0.5 * max_delta, max_delta, 2)
        t = p0[:2] + d
        if abs(t[0]) > lim_p or abs(t[1]) > lim_y:
            continue
        c = 0 if grid_counts is None else grid_counts.get((int((t[0] + 50) // 10), int((t[1] + 80) // 10)), 0)
        if best is None or c < best_c:
            best, best_c = d, c
    return best


def cmd_batch(a):
    sys.path.insert(0, os.path.dirname(HERE))
    from common import load_models, biggest
    m = load_models(["CPUExecutionProvider"], parsing=False)
    rng = np.random.default_rng(a.seed)
    recs = load_pool(a.pool)
    grid = {}
    for r in recs:
        key = (int((r["pitch"] + 50) // 10), int((r["yaw"] + 80) // 10))
        grid[key] = grid.get(key, 0) + 1
    if a.sparse_below:     # targeted mode: any accurate-landmark non-reprojected sample can be a base (including LivePortrait outputs, i.e. method C + B combined)
        bases = [r for r in recs if r.get("method") != "reproject" and r.get("kps_err", 1.0) <= 0.08]
    else:
        bases = [r for r in recs if r.get("lm_src", "facemesh") == "facemesh" and abs(r["yaw"]) < 45
                 and abs(r["pitch"]) < 35 and r.get("kps_err", 1.0) <= 0.08]          # base-image landmarks must be accurate
    base_pose = np.array([[r["pitch"], r["yaw"]] for r in bases], np.float32)
    fdir = os.path.join(a.pool, "faces")
    idx = len(recs)
    n_ok = n_bad = 0
    with open(os.path.join(a.pool, "index.jsonl"), "a") as index:
        for i in range(a.n):
            if a.sparse_below:
                # targeted: pick a target from a sparse cell, then find a base that can reach it
                if i % 200 == 0:
                    targets = sparse_cells(load_pool(a.pool) if i else recs, a.sparse_below)
                    if not targets:
                        print("\nall cells are filled")
                        break
                t = np.array(targets[int(rng.integers(len(targets)))]) + rng.uniform(-5, 5, 2)
                j = pick_base(rng, base_pose, t, a.max_delta)
                if j is None:
                    continue
                b = bases[j]
                p0 = np.array([b["pitch"], b["yaw"], b["roll"]])
                delta = t - p0[:2]
            else:
                b = bases[int(rng.integers(len(bases)))]
                p0 = np.array([b["pitch"], b["yaw"], b["roll"]])
                delta = pick_delta(rng, p0, a.max_delta, grid_counts=grid)
                if delta is None:
                    continue
            img = cv2.imread(os.path.join(fdir, b["id"] + ".jpg"))
            d = np.load(os.path.join(fdir, b["id"] + ".npz"))
            res = reproject(img, d["landmarks"], delta[0], delta[1], parsing=d["parsing"])
            target = p0 + np.array([delta[0], delta[1], 0.0])
            # re-check: a large gap between measured HeadPose and the expected pose means the warp went wrong
            fs = m["det"].detect(res["image"])
            if not fs:
                n_bad += 1
                continue
            bf = biggest(fs)
            x1, y1, x2, y2 = [int(max(0, v)) for v in bf.bbox[:4]]
            hp = m["hp"].estimate(res["image"][y1:y2, x1:x2])
            meas = np.array([hp.pitch, hp.yaw, hp.roll])
            if np.abs(meas[:2] - target[:2]).max() > a.max_err:
                n_bad += 1
                continue
            # are landmarks and image aligned: 5 corresponding rotated landmarks vs the RetinaFace 5 points on the result
            kps = np.asarray(bf.landmarks, np.float32)
            iod = max(float(np.linalg.norm(kps[1] - kps[0])), 0.25 * float(bf.bbox[2] - bf.bbox[0]), 1.0)
            kerr = float(np.linalg.norm(kps5_from_mesh(res["landmarks"]) - kps, axis=1).mean() / iod)
            if kerr > a.max_kps_err:
                n_bad += 1
                continue
            name = f"{idx:06d}"
            cv2.imwrite(os.path.join(fdir, name + ".jpg"), res["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])
            np.savez_compressed(os.path.join(fdir, name + ".npz"), landmarks=res["landmarks"], parsing=res["parsing"],
                                pose=target.astype(np.float32), mesh_pose=mesh_pose(res["landmarks"]),
                                uncertain=np.packbits(res["uncertain"]), uncertain_shape=np.array(res["uncertain"].shape))
            index.write(json.dumps(dict(id=name, dataset=b["dataset"], src=b["src"], frame=b.get("frame", 0),
                                        method="reproject", base_id=b["id"], face_px=b["face_px"],
                                        pitch=round(float(target[0]), 2), yaw=round(float(target[1]), 2),
                                        roll=round(float(target[2]), 2), measured=np.round(meas, 1).tolist(),
                                        delta=np.round(delta, 1).tolist(), lm_src="reproject", kps_err=round(kerr, 3),
                                        uncertain_frac=round(float(res["uncertain"].mean()), 4))) + "\n")
            key = (int((target[0] + 50) // 10), int((target[1] + 80) // 10))
            grid[key] = grid.get(key, 0) + 1
            idx += 1
            n_ok += 1
            print(f"  {i + 1}/{a.n}, kept {n_ok}, failed HeadPose re-check {n_bad}", end="\r", flush=True)
    print(f"\ndone: generated {n_ok} (dropped by re-check {n_bad}), pool now has {idx} images")


def cmd_demo(a):
    fdir = os.path.join(a.pool, "faces")
    img = cv2.imread(os.path.join(fdir, a.id + ".jpg"))
    d = np.load(os.path.join(fdir, a.id + ".npz"))
    rows = []
    for dp in (-30, 0, 30):
        line = []
        for dy in (-30, -15, 0, 15, 30):
            r = reproject(img, d["landmarks"], dp, dy, parsing=d["parsing"])
            t = r["image"].copy()
            for q in r["landmarks"][::4]:
                cv2.circle(t, (int(q[0]), int(q[1])), 1, (0, 255, 0), -1)
            t[r["uncertain"] > 0] = (0.5 * t[r["uncertain"] > 0] + [0, 0, 127]).astype(np.uint8)
            cv2.putText(t, f"dp{dp:+d} dy{dy:+d}", (8, 30), 0, 0.9, (0, 255, 255), 2)
            line.append(cv2.resize(t, (256, 256), interpolation=cv2.INTER_AREA))
        rows.append(np.hstack(line))
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("batch")
    p.add_argument("--pool", required=True)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--max_delta", type=float, default=20, help="maximum single rotation (degrees); beyond 20 deg the hair region starts to show folding artefacts")
    p.add_argument("--max_err", type=float, default=15, help="allowed HeadPose re-check error (degrees)")
    p.add_argument("--max_kps_err", type=float, default=0.10, help="upper bound of the 5-point error (relative to eye distance)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sparse_below", type=int, default=0, help=">0: only fill pitch x yaw cells with fewer samples than this")
    p = sub.add_parser("demo")
    p.add_argument("--pool", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(batch=cmd_batch, demo=cmd_demo)[a.cmd](a)


if __name__ == "__main__":
    main()
