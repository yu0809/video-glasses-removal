"""
Geometric and quality changes applied after synthesis
=====================================================
- method A: roll is applied exactly by a 2D rotation, then the image is re-cropped the pipeline way (detection box x 1.3, shifted up by 0.08)
- method E: close-range wide-angle perspective / barrel distortion, high/low camera (crop centre offset), motion blur, interlacing, compression blocking, low-light noise, back light
- small affine jitter, horizontal flip, tiny-face simulation (downscale by 1/1.5-1/7 and upscale again, labels stay at full resolution), blur, noise, JPEG, colour
Images and labels (frame, lens, weight) share the geometric changes; quality changes apply to the image only.
"""
import cv2
import numpy as np

from common import CROP_SCALE, Y_SHIFT

# measured on real samples of the source pool (431 images): RetinaFace box side ~ 1.10 x FaceMesh landmark bounding-box side; the box centre is 0.076 x side above the landmark centre
DET_SIDE, DET_DY = 1.10, -0.076


def det_box_from_landmarks(lm2):
    x0, y0 = lm2.min(0)
    x1, y1 = lm2.max(0)
    s = max(x1 - x0, y1 - y0) * DET_SIDE
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + DET_DY * max(x1 - x0, y1 - y0)
    return cx, cy, s


def geometry_matrix(lm2, size, roll_delta, rng, jitter=True, cam_shift=0.0):
    """2x3 affine from crop coordinates to new crop coordinates: rotate by roll_delta degrees around the face centre (clockwise positive, i.e. roll increases),
    re-crop the pipeline way from the rotated "detection box", then add small jitter (scale, translation)."""
    c = lm2.mean(0)
    R = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), -roll_delta, 1.0)   # positive angles are counter-clockwise in OpenCV
    lm_r = lm2 @ R[:, :2].T + R[:, 2]
    cx, cy, s = det_box_from_landmarks(lm_r)
    side = s * CROP_SCALE
    cy -= Y_SHIFT * side
    if jitter:
        side *= 1 / rng.uniform(0.85, 1.15)
        cx += rng.uniform(-0.06, 0.06) * side
        cy += rng.uniform(-0.06, 0.06) * side
    cy += cam_shift * side                       # high / low camera: the face sits higher or lower in the crop
    k = size / side
    C = np.array([[k, 0, size / 2 - k * cx], [0, k, size / 2 - k * cy]])
    return np.vstack([C, [0, 0, 1]]) @ np.vstack([R, [0, 0, 1]])


def warp_all(M, arrs, size, interp_img=cv2.INTER_LINEAR):
    out = []
    for k, a in enumerate(arrs):
        if a is None:
            out.append(None)
            continue
        out.append(cv2.warpAffine(a, M[:2], (size, size), flags=interp_img if k == 0 else cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT))
    return out


def lens_distort(arrs, rng):
    """Close-range wide angle: barrel distortion + mild perspective (image and labels together)."""
    h, w = arrs[0].shape[:2]
    k1 = rng.uniform(-0.25, 0.05)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)
    r2 = nx * nx + ny * ny
    f = 1 + k1 * r2
    mx, my = nx * f * (w / 2) + w / 2, ny * f * (h / 2) + h / 2
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-0.04, 0.04, (4, 2)).astype(np.float32) * w
    P = cv2.getPerspectiveTransform(dst, src)
    pts = np.stack([mx.ravel(), my.ravel(), np.ones(mx.size, np.float32)])
    q = P @ pts
    mx, my = (q[0] / q[2]).reshape(h, w).astype(np.float32), (q[1] / q[2]).reshape(h, w).astype(np.float32)
    return [None if a is None else cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            for a in arrs]


def motion_blur(img, rng, length=None):
    L = int(length or rng.integers(3, 15))
    k = np.zeros((L, L), np.float32)
    k[L // 2, :] = 1
    ang = rng.uniform(0, 180)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D(((L - 1) / 2, (L - 1) / 2), ang, 1.0), (L, L))
    k /= k.sum() + 1e-6
    return cv2.filter2D(img, -1, k)


def small_face(img, rng):
    """Distant tiny face: downscale by 1/1.5-1/7 and upscale again (random interpolation)."""
    h, w = img.shape[:2]
    f = rng.uniform(1.5, 7.0)
    small = cv2.resize(img, (max(8, int(w / f)), max(8, int(h / f))), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.3:     # small images are often compressed too
        q = int(rng.integers(40, 90))
        small = cv2.imdecode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    up = rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    return cv2.resize(small, (w, h), interpolation=up), f


def color_jitter(img, rng):
    x = img.astype(np.float32) / 255
    x = x * rng.uniform(0.7, 1.3) + rng.uniform(-0.1, 0.1)                     # brightness
    m = x.mean()
    x = (x - m) * rng.uniform(0.7, 1.3) + m                                    # contrast
    g = x.mean(2, keepdims=True)
    x = g + (x - g) * rng.uniform(0.6, 1.4)                                    # saturation
    t = rng.uniform(-0.08, 0.08)
    x = x * np.array([1 - t, 1, 1 + t], np.float32)                            # colour temperature (BGR)
    x = np.clip(x, 0, 1) ** rng.uniform(0.75, 1.35)                            # gamma
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def low_light(img, rng):
    x = img.astype(np.float32) * rng.uniform(0.15, 0.45)
    x += rng.normal(0, rng.uniform(4, 12), x.shape) + rng.normal(0, rng.uniform(2, 6), x.shape[:2])[..., None]
    return np.clip(x * rng.uniform(1.5, 2.5), 0, 255).astype(np.uint8)        # brightening makes the noise more visible


def backlight(img, rng):
    """Back light: the face gets darker with lower contrast, the background (image border) brighter."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2)
    glow = np.clip((r - 0.25) * 2.5, 0, 1)[..., None]
    x = img.astype(np.float32) * rng.uniform(0.35, 0.7)
    x = x * (1 - glow) + (255 * rng.uniform(0.8, 1.0)) * glow * rng.uniform(0.5, 1.0) + x * glow * 0.3
    return np.clip(x, 0, 255).astype(np.uint8)


def interlace(img, rng):
    out = img.copy()
    s = int(rng.integers(1, 4))
    out[1::2] = np.roll(img[1::2], s, axis=1)
    return out


def block_artifacts(img, rng):
    """H.264-like: mild blur followed by low-quality JPEG, for stronger blocking."""
    x = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.2))
    q = int(rng.integers(10, 30))
    return cv2.imdecode(cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def quality_aug(img, rng, info):
    """Image-only quality changes; info records which were applied."""
    if rng.random() < 0.5:
        img, f = small_face(img, rng)
        info["small_face"] = round(float(f), 2)
    r = rng.random()
    if r < 0.2:
        img = motion_blur(img, rng)
        info["motion_blur"] = True
    elif r < 0.4:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.8))
        info["blur"] = True
    if rng.random() < 0.6:
        img = color_jitter(img, rng)
    r = rng.random()
    if r < 0.08:
        img = low_light(img, rng)
        info["low_light"] = True
    elif r < 0.14:
        img = backlight(img, rng)
        info["backlight"] = True
    if rng.random() < 0.3:
        img = np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(2, 10), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.04:
        img = interlace(img, rng)
        info["interlace"] = True
    r = rng.random()
    if r < 0.1:
        img = block_artifacts(img, rng)
        info["blocky"] = True
    elif r < 0.7:
        q = int(rng.integers(25, 96))
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    return img
