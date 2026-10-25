"""
Constants and small helpers shared by the training code
=======================================================

Head-pose convention (follows what the uniface HeadPose model actually outputs, measured in 2026-10;
note that the pitch sign in the uniface documentation is the opposite of the measured behaviour):
  pitch > 0  looking up (nostrils visible)                 pitch < 0  looking down
  yaw   > 0  subject turns to their own right (face towards image left)   yaw < 0  to their left
  roll  > 0  clockwise tilt in the image (top of the head towards image right)
How it was measured: rotating an image 20 deg counter-clockwise changes roll from 1.2 to -18.8; pitch / yaw correlate
with the geometric angles computed from FaceMesh 3D points with coefficients -0.91 / -0.98 (see the signs in mesh_pose).

FaceMesh landmarks (subject's own left/right): right eye outer/inner corner 33/133, left eye outer/inner 263/362,
between the eyes 168, nose bridge 6, nose tip 1, top of forehead 10, chin 152, face sides 234/454, temples 127/356.
In a frontal face 33 is on the image left and 263 on the image right. Smaller z is closer to the camera.
"""
import cv2
import numpy as np

CROP_SCALE, Y_SHIFT, SIZE = 1.3, 0.08, 512   # same as the default crop in remove_glasses.py
GLASSES, HAIR = 6, 17                        # BiSeNet (CelebAMask-HQ) classes

LM = dict(r_eye_out=33, r_eye_in=133, l_eye_out=263, l_eye_in=362, mid_eyes=168, bridge=6,
          nose_tip=1, forehead=10, chin=152, r_face=234, l_face=454, r_temple=127, l_temple=356)


def crop_matrix(bbox, size=SIZE, crop_scale=CROP_SCALE, y_shift=Y_SHIFT):
    """Face box -> 2x3 affine matrix of the pipeline crop (image -> crop), identical to remove_glasses.crop_transforms."""
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * crop_scale
    cy -= y_shift * side
    s = size / side
    return np.array([[s, 0, size / 2 - s * cx], [0, s, size / 2 - s * cy]], np.float64), side


def crop_face(img, bbox, size=SIZE):
    M, side = crop_matrix(bbox, size)
    crop = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return crop, M, side


def mesh_pose(lm):
    """Rough (pitch, yaw, roll) from FaceMesh 3D points, with the same signs as HeadPose.
    Only used as a consistency check: a large disagreement with HeadPose means either the landmarks or the pose is wrong."""
    lm = lm.astype(np.float64)
    X = lm[LM["l_eye_out"]] - lm[LM["r_eye_out"]]
    X /= np.linalg.norm(X) + 1e-9
    Y = lm[LM["chin"]] - lm[LM["forehead"]]
    Y -= X * (Y @ X)
    Y /= np.linalg.norm(Y) + 1e-9
    down = np.degrees(np.arcsin(np.clip(Y[2], -1, 1)))         # looking down moves the chin away from the camera
    turn_left = np.degrees(np.arcsin(np.clip(X[2], -1, 1)))    # turning to the subject's left brings the right-eye side closer
    roll = np.degrees(np.arctan2(X[1], X[0]))
    return np.array([-down, -turn_left, roll], np.float32)


def face_triangles():
    """FaceMesh triangles (852, 3). Every 3 edges of FACEMESH_TESSELATION_FULL form one triangle (all verified valid)."""
    from uniface.landmark._tessellation import FACEMESH_TESSELATION_FULL
    tris = []
    for e in np.asarray(FACEMESH_TESSELATION_FULL).reshape(-1, 3, 2):
        v = list(dict.fromkeys(e.ravel().tolist()))
        if len(v) == 3:
            tris.append(v)
    return np.array(tris, np.int32)


def limit_ort_threads(n=1):
    """Limit onnxruntime to n threads per process when running several worker processes. By default every session
    uses all cores; with 6 processes that is 60 competing threads (load average 65, almost no progress).
    uniface creates its own SessionOptions internally, so the class itself is replaced here."""
    import onnxruntime as ort
    if getattr(ort.SessionOptions, "_limited", None) == n:
        return
    base = ort.SessionOptions if not hasattr(ort.SessionOptions, "_base") else ort.SessionOptions._base

    class Limited(base):
        _limited, _base = n, base

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.intra_op_num_threads = n
            self.inter_op_num_threads = 1

    ort.SessionOptions = Limited


def load_models(providers=None, parsing=True):
    """uniface models. providers=None picks automatically (CoreML on a Mac); ['CPUExecutionProvider'] forces CPU."""
    from uniface import RetinaFace, BiSeNet, FaceMesh, HeadPose
    from uniface.constants import ParsingWeights
    m = dict(det=RetinaFace(providers=providers), mesh=FaceMesh(providers=providers),
             hp=HeadPose(providers=providers))
    if parsing:
        m["par"] = BiSeNet(model_name=ParsingWeights.RESNET34, providers=providers)
    return m


def biggest(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
