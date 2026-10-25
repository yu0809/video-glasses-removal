"""
Rigid rotation and 2D alignment of FaceMesh 3D points (shared by mesh re-projection and landmark transfer)
=========================================================================================================
Coordinates: x, y are crop pixels, z has the same scale as x and grows away from the camera (near-orthographic).
Angles always follow the HeadPose convention (training/common.py): pitch > 0 looks up, yaw > 0 turns to the
subject's right, roll > 0 is clockwise.
"""
import numpy as np

# FaceMesh points matching the 5 RetinaFace keypoints (image-left eye, image-right eye, nose tip, image-left / right mouth corner)
KPS5_MESH = [(33, 133), (263, 362), (1,), (61,), (291,)]


def head_center(lm):
    """Head rotation centre: about one eye distance behind the midpoint of the eyes, close to the top of the spine."""
    eye = (lm[33] + lm[263]) / 2
    iod = np.linalg.norm(lm[263, :2] - lm[33, :2]) + 1e-6
    c = eye.copy()
    c[2] = lm[:, 2].mean() + 0.9 * iod
    return c


def rot_matrix(dpitch, dyaw, droll=0.0):
    """Rotation in image coordinates (x right, y down, z into the image) producing (dpitch, dyaw, droll) in the
    HeadPose convention. Signs verified by mesh_pose(rotate(lm)) ~ mesh_pose(lm) + delta (see _self_test)."""
    p, y, r = np.radians([dpitch, dyaw, droll])
    # looking up: chin moves forward (smaller z) and up (smaller y): about the x axis
    Rx = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])
    # turning to the subject's right (face towards image left): nose tip x decreases: about the y axis
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    # clockwise (in the image, y down): about the z axis
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotate(pts, R, center):
    return (pts - center) @ R.T + center


def kps5_from_mesh(lm):
    return np.array([lm[list(ix), :2].mean(0) for ix in KPS5_MESH])


def fit_similarity(src, dst):
    """Least-squares 2D similarity transform src -> dst (Umeyama); returns (s, R2, t)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a)
    D = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        D[1, 1] = -1
    R2 = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (a ** 2).sum()
    return s, R2, md - s * R2 @ ms


def apply_similarity(lm, s, R2, t):
    """Apply a 2D similarity transform to 3D points (z is only scaled by s)."""
    out = lm.copy()
    out[:, :2] = s * lm[:, :2] @ R2.T + t
    out[:, 2] = s * lm[:, 2]
    return out


def transfer_landmarks(base_lm, dpose, kps5):
    """Rotate the base face's 3D landmarks by the pose change dpose, then align them to the target image's 5 RetinaFace
    keypoints with a similarity transform. Returns (new landmarks, mean 5-point error in pixels)."""
    R = rot_matrix(*dpose)
    lm = rotate(base_lm.astype(np.float64), R, head_center(base_lm.astype(np.float64)))
    src = kps5_from_mesh(lm)
    s, R2, t = fit_similarity(src, np.asarray(kps5, np.float64))
    out = apply_similarity(lm, s, R2, t)
    err = np.linalg.norm(kps5_from_mesh(out) - kps5, axis=1).mean()
    return out.astype(np.float32), float(err)


def _self_test(lm):
    """Check the signs of rot_matrix: the geometric pose change after rotation must match the input."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common import mesh_pose
    p0 = mesh_pose(lm)
    for d in ((20, 0, 0), (0, 20, 0), (0, 0, 20), (15, -25, 10)):
        p1 = mesh_pose(rotate(lm.astype(np.float64), rot_matrix(*d), head_center(lm.astype(np.float64))))
        print(d, "->", np.round(p1 - p0, 1))
