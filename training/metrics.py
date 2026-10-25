"""
Segmentation metrics. IoU is naturally low for thin frames, so boundary F1 (2 px tolerance) is the main metric.
Boundary F1: prediction and label are binarised (> 0.5); precision = share of predicted pixels within tol of the label,
recall = share of label pixels within tol of the prediction. Frames are only 1-3 px wide, so the frame pixels are their
own boundary and the mask pixels are used directly.
"""
import cv2
import numpy as np

_K = {}


def _disk(r):
    if r not in _K:
        _K[r] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return _K[r]


def seg_metrics(prob, gt, thr=0.5, tol=2):
    p = (prob > thr).astype(np.uint8)
    g = (gt > 0.5).astype(np.uint8)
    np_, ng = int(p.sum()), int(g.sum())
    out = dict(has_gt=ng > 0, fp_frac=float(np_) / p.size)
    if ng == 0:
        out.update(bf1=1.0 if np_ == 0 else 0.0, iou=1.0 if np_ == 0 else 0.0, dice=1.0 if np_ == 0 else 0.0,
                   precision=1.0 if np_ == 0 else 0.0, recall=1.0)
        return out
    inter = int((p & g).sum())
    out["iou"] = inter / max(1, np_ + ng - inter)
    out["dice"] = 2 * inter / max(1, np_ + ng)
    if np_ == 0:
        out.update(bf1=0.0, precision=0.0, recall=0.0)
        return out
    gd = cv2.dilate(g, _disk(tol))
    pd = cv2.dilate(p, _disk(tol))
    prec = float((p & gd).sum()) / np_
    rec = float((g & pd).sum()) / ng
    out.update(precision=prec, recall=rec, bf1=2 * prec * rec / max(1e-9, prec + rec))
    return out
