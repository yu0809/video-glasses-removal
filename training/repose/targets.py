"""
Targeted filling of rare poses: find sparse cells of the pitch x yaw grid (10 deg cells) to use as targets for pose augmentation.
"""
import numpy as np

P_CENTERS = np.arange(-45, 46, 10)      # pitch ±50
Y_CENTERS = np.arange(-75, 76, 10)      # yaw ±80 (target ±75)


def usable(r, max_kps_err=0.10):
    """A face usable for training: the 5-point landmark error is recorded and below the threshold (early samples without this check have no such field and are excluded)."""
    return r.get("kps_err", 1.0) <= max_kps_err


def cell_counts(recs):
    c = {}
    for r in recs:
        if not usable(r):
            continue
        k = (int(np.floor((r["pitch"] + 50) / 10)), int(np.floor((r["yaw"] + 80) / 10)))
        c[k] = c.get(k, 0) + 1
    return c


def sparse_cells(recs, below):
    """Centres [(pitch, yaw), ...] of target cells with fewer than `below` samples."""
    c = cell_counts(recs)
    out = []
    for p in P_CENTERS:
        for y in Y_CENTERS:
            k = (int(np.floor((p + 50) / 10)), int(np.floor((y + 80) / 10)))
            if c.get(k, 0) < below:
                out.append((float(p), float(y)))
    return out


def pick_base(rng, bases_pose, t, reach):
    """Randomly pick a base face that can reach target t within `reach` degrees in both pitch and yaw; None if there is none."""
    d = np.abs(bases_pose - np.asarray(t)[None, :2]).max(1)
    ok = np.where(d <= reach)[0]
    if len(ok) == 0:
        return None
    # prefer bases that need less rotation (less distortion)
    w = 1.0 / (5.0 + d[ok])
    return int(rng.choice(ok, p=w / w.sum()))
