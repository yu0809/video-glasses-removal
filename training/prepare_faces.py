#!/usr/bin/env python3
"""
Step 1: prepare the "no glasses" face source pool
=================================================
Finds faces in videos / image folders, crops them exactly like the glasses-removal pipeline (so the training images
match what the model sees in use), automatically drops people already wearing glasses, and saves:

  out/faces/000123.jpg         512x512 face crop
  out/faces/000123.npz         468 3D face landmarks (crop coordinates), face parsing, head pose etc.
  out/index.jsonl              one line per sample: source, method, pose, face size
  out/done_tasks.txt           sources already processed (skipped when re-run after an interruption)

Usage:
  python training/prepare_faces.py --src input/smoke_faces --out data/faces_pool --every 2
  python training/prepare_faces.py --src ~/datasets/ffhq --out data/faces_pool --workers 6

Videos take one frame every 5 by default and skip frames whose pose barely differs from the previous one (avoids heavy duplication).
If an image has a .json with the same name next to it (e.g. {"method": "liveportrait", "target_pose": [p, y, r]} written by a re-posing script),
it is read: method goes into the index, and samples whose measured pose is too far from target_pose are dropped.

Filters:
  - face too small (< --min_face in the original image)
  - wearing glasses (glasses area share in the face parsing > --max_glasses_frac)
  - no face re-detected in the crop
  - untrustworthy landmarks: little overlap between their bounding box and the detection box, the 5 corresponding points too far from the RetinaFace
    keypoints (> --max_kps_err eye distances),
    or FaceMesh geometric pose and HeadPose differing by > --max_pose_diff degrees (at least one of them is wrong).
    Re-posed samples (base_npz in the json) then fall back to "base-image landmarks rotated in 3D by the pose change + 5-point similarity alignment"
    and are kept if the alignment error is small enough (lm_src=transfer in the index), see repose/mesh3d.py
  - pose change from the previous frame of the same video < --min_pose_change degrees (deduplication)
"""
import argparse
import io
import json
import multiprocessing as mp
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import GLASSES, biggest, crop_face, limit_ort_threads, load_models, mesh_pose  # noqa: E402
from repose.mesh3d import kps5_from_mesh, transfer_landmarks  # noqa: E402

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
ARROW_EXT = (".arrow",)  # Hugging Face datasets Arrow shards (image column is {bytes, path}); one task per shard
IMG_CHUNK = 200       # images are handed to worker processes in groups of this size


# ----------------------------------------------------------------------------
# task split: one task per video; one task per IMG_CHUNK images
# ----------------------------------------------------------------------------
def list_tasks(paths, max_per_source):
    tasks = []
    for p in paths:
        p = os.path.abspath(os.path.expanduser(p))
        if os.path.isdir(p):
            imgs, vids, arrows = [], [], []
            for root, dirs, names in os.walk(p):
                dirs.sort()
                for n in sorted(names):
                    f = os.path.join(root, n)
                    if n.lower().endswith(IMAGE_EXT):
                        imgs.append(f)
                    elif n.lower().endswith(VIDEO_EXT):
                        vids.append(f)
                    elif n.lower().endswith(ARROW_EXT):
                        arrows.append(f)
            tasks += [dict(key=f, kind="arrow", files=[f], root=p) for f in arrows]
            imgs = imgs[:max_per_source]
            for k in range(0, len(imgs), IMG_CHUNK):
                tasks.append(dict(key=f"{p}#{k}", kind="images", files=imgs[k:k + IMG_CHUNK], root=p))
            tasks += [dict(key=v, kind="video", files=[v], root=p) for v in vids]
        elif p.lower().endswith(VIDEO_EXT):
            tasks.append(dict(key=p, kind="video", files=[p], root=os.path.dirname(p)))
        elif p.lower().endswith(ARROW_EXT):
            tasks.append(dict(key=p, kind="arrow", files=[p], root=os.path.dirname(p)))
        elif p.lower().endswith(IMAGE_EXT):
            tasks.append(dict(key=p, kind="images", files=[p], root=os.path.dirname(p)))
    return tasks


def iter_task(task, every, max_n):
    """Yield (source file, frame index, BGR image, attached json metadata)."""
    if task["kind"] == "arrow":
        import pyarrow as pa
        with pa.memory_map(task["files"][0]) as src:
            try:
                tab = pa.ipc.open_stream(src).read_all()
            except pa.ArrowInvalid:
                tab = pa.ipc.open_file(src).read_all()
        col = tab.column("image")
        for k in range(min(len(col), max_n)):
            v = col[k].as_py()
            im = cv2.imdecode(np.frombuffer(v["bytes"], np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                yield task["files"][0] + "#" + (v.get("path") or str(k)), k, im, {}
        return
    if task["kind"] == "images":
        for f in task["files"]:
            im = cv2.imread(f)
            if im is None:
                continue
            side = os.path.splitext(f)[0] + ".json"
            meta = json.load(open(side)) if os.path.exists(side) else {}
            yield f, 0, im, meta
        return
    path = task["files"][0]
    cap = cv2.VideoCapture(path)
    i = n = 0
    while n < max_n:
        if not cap.grab():
            break
        if i % every == 0:
            ok, f = cap.retrieve()
            if ok:
                yield path, i, f, {}
                n += 1
        i += 1
    cap.release()


# ----------------------------------------------------------------------------
# worker processes
# ----------------------------------------------------------------------------
_M = None
_A = None


def _init_worker(args):
    global _M, _A
    cv2.setNumThreads(1)
    _A = args
    providers = None if args.providers == "auto" else ["CPUExecutionProvider"]
    if args.workers > 1:
        limit_ort_threads(max(1, (os.cpu_count() or 2) // args.workers))
    _M = load_models(providers)


def box_overlap(a, b):
    """Intersection area / area of the smaller box."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))


def process_face(img, bbox, meta, stats):
    """One face -> sample dict, or None (recording the rejection reason in stats)."""
    a, m = _A, _M
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    face_px = max(x2 - x1, y2 - y1)
    if min(x2 - x1, y2 - y1) < a.min_face:
        stats["small"] += 1
        return None
    crop, M, side = crop_face(img, (x1, y1, x2, y2))
    lab = m["par"].parse(crop)
    gfrac = float((lab == GLASSES).mean())
    if gfrac > a.max_glasses_frac:
        stats["glasses"] += 1
        return None
    # re-locate the face inside the crop for FaceMesh and HeadPose
    cf = m["det"].detect(crop)
    if not cf:
        stats["nodet"] += 1
        return None
    cf = biggest(cf)
    kps = np.asarray(cf.landmarks, np.float32)
    # normalising length: eye distance; it shrinks in profile views, so it is at least 1/4 of the face box width
    iod = max(float(np.linalg.norm(kps[1] - kps[0])), 0.25 * float(cf.bbox[2] - cf.bbox[0]), 1.0)
    lm = m["mesh"].predict(crop, [cf])[0].landmarks.astype(np.float32)
    bx1, by1, bx2, by2 = [int(max(0, v)) for v in cf.bbox[:4]]
    hp = m["hp"].estimate(crop[by1:by2, bx1:bx2])
    pose = np.array([hp.pitch, hp.yaw, hp.roll], np.float32)
    tp = meta.get("target_pose")
    if tp is not None and float(np.abs(pose[:2] - np.asarray(tp, np.float32)[:2]).max()) > a.max_target_diff:
        stats["off_target"] += 1
        return None
    # is FaceMesh trustworthy: landmark bounding box overlaps the detection box, 5 corresponding points close to the RetinaFace keypoints, geometric pose consistent with HeadPose
    lb = [lm[:, 0].min(), lm[:, 1].min(), lm[:, 0].max(), lm[:, 1].max()]
    kerr = float(np.linalg.norm(kps5_from_mesh(lm) - kps, axis=1).mean() / iod)
    mp_ = mesh_pose(lm)
    diff = float(np.abs(pose - mp_).max())
    mesh_ok = box_overlap(lb, cf.bbox[:4]) >= 0.6 and kerr <= a.max_kps_err and diff <= a.max_pose_diff
    lm_src = "facemesh"
    if not mesh_ok and meta.get("base_npz") and os.path.exists(meta["base_npz"]):
        # re-posed samples: FaceMesh is inaccurate at large angles, so use "base-image landmarks rotated by the pose change + 5-point alignment"
        base = np.load(meta["base_npz"])
        lm_t, kerr_t = transfer_landmarks(base["landmarks"], pose - base["pose"], kps)
        kerr_t /= iod
        if kerr_t <= a.max_kps_err:
            lm, kerr, lm_src, mp_ = lm_t, kerr_t, "transfer", mesh_pose(lm_t)
            diff = float(np.abs(pose - mp_).max())
            mesh_ok = True
    if not mesh_ok:
        stats["badmesh" if diff <= a.max_pose_diff else "inconsistent"] += 1
        return None
    stats["lm_" + lm_src] = stats.get("lm_" + lm_src, 0) + 1
    return dict(crop=crop, lab=lab.astype(np.uint8), lm=lm, pose=pose, mesh_pose=mp_,
                det_bbox=np.asarray(cf.bbox[:4], np.float32),
                det_kps=np.asarray(cf.landmarks, np.float32), M=M.astype(np.float32),
                face_px=face_px, glasses_frac=gfrac, pose_diff=diff, kps_err=kerr, lm_src=lm_src, meta=meta)


def run_task(task):
    a = _A
    stats = dict(seen=0, kept=0, glasses=0, small=0, dup=0, nodet=0, badmesh=0,
                 inconsistent=0, off_target=0, noface=0)
    out = []
    last_pose = None
    for src, fi, img, meta in iter_task(task, a.every, a.max_per_source):
        stats["seen"] += 1
        faces = _M["det"].detect(img) or []
        if not faces:
            stats["noface"] += 1
        # an image may contain several people; try every face (videos only take the largest one so deduplication is meaningful)
        if task["kind"] == "video" and faces:
            faces = [biggest(faces)]
        for face in faces:
            s = process_face(img, face.bbox, meta, stats)
            if s is None:
                continue
            if task["kind"] == "video":
                if last_pose is not None and np.abs(s["pose"] - last_pose).max() < a.min_pose_change:
                    stats["dup"] += 1
                    continue
                last_pose = s["pose"]
            ok, jpg = cv2.imencode(".jpg", s.pop("crop"), [cv2.IMWRITE_JPEG_QUALITY, 95])
            s["jpg"] = jpg.tobytes()
            s["src"] = os.path.relpath(src, task["root"]) if task["kind"] != "arrow" else \
                os.path.basename(src)
            s["dataset"] = a.dataset or os.path.basename(task["root"].rstrip("/"))
            s["frame"] = int(fi)
            out.append(s)
            stats["kept"] += 1
    return task["key"], out, stats


# ----------------------------------------------------------------------------
# main process: dispatch tasks, number and write files in order
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="video files / images / folders, several allowed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=None, help="dataset name (written to the index; default: name of the source folder)")
    ap.add_argument("--method", default="real",
                    help="sample source method: real = real images / video frames; re-posed images carry their method in the json")
    ap.add_argument("--every", type=int, default=5, help="take one video frame every N frames")
    ap.add_argument("--max_per_source", type=int, default=3000)
    ap.add_argument("--min_face", type=int, default=90, help="minimum face size in the original image (px)")
    ap.add_argument("--max_glasses_frac", type=float, default=0.002,
                    help="drop the face as wearing glasses when the glasses area share in the face parsing exceeds this")
    ap.add_argument("--min_pose_change", type=float, default=4.0,
                    help="skip a frame whose pose differs from the previous one of the same video by less than this many degrees (deduplication)")
    ap.add_argument("--max_pose_diff", type=float, default=20.0,
                    help="maximum allowed difference between FaceMesh geometric pose and HeadPose (degrees)")
    ap.add_argument("--max_target_diff", type=float, default=20.0,
                    help="re-posed samples: maximum allowed difference between measured and target pitch/yaw (degrees)")
    ap.add_argument("--max_kps_err", type=float, default=0.10,
                    help="upper bound of the mean error between the 5 landmark points and the RetinaFace keypoints (relative to eye distance)")
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--providers", choices=["cpu", "auto"], default="cpu",
                    help="onnxruntime on CPU or automatic (CoreML). CPU is usually faster with several processes")
    a = ap.parse_args()

    fdir = os.path.join(a.out, "faces")
    os.makedirs(fdir, exist_ok=True)
    index_path = os.path.join(a.out, "index.jsonl")
    done_path = os.path.join(a.out, "done_tasks.txt")
    idx = 0
    if os.path.exists(index_path):
        with open(index_path) as f:
            idx = sum(1 for _ in f)
    done = set(open(done_path).read().splitlines()) if os.path.exists(done_path) else set()
    tasks = [t for t in list_tasks(a.src, a.max_per_source) if t["key"] not in done]
    print(f"{len(tasks)} tasks to process ({len(done)} already done), {a.workers} processes, {idx} images in the pool")
    if not tasks:
        return

    total = {}
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init_worker, initargs=(a,)) as pool, \
            open(index_path, "a") as index, open(done_path, "a") as donef:
        for k, (key, samples, st) in enumerate(pool.imap_unordered(run_task, tasks)):
            for s in samples:
                name = f"{idx:06d}"
                with open(os.path.join(fdir, name + ".jpg"), "wb") as f:
                    f.write(s["jpg"])
                np.savez_compressed(os.path.join(fdir, name + ".npz"), landmarks=s["lm"], parsing=s["lab"],
                                    pose=s["pose"], mesh_pose=s["mesh_pose"], det_bbox=s["det_bbox"],
                                    det_kps=s["det_kps"], crop_M=s["M"])
                meta = s["meta"]
                rec = dict(id=name, dataset=s["dataset"], src=s["src"], frame=s["frame"],
                           method=meta.get("method", a.method), face_px=round(float(s["face_px"]), 1),
                           pitch=round(float(s["pose"][0]), 2), yaw=round(float(s["pose"][1]), 2),
                           roll=round(float(s["pose"][2]), 2), pose_diff=round(s["pose_diff"], 1),
                           kps_err=round(s["kps_err"], 3), lm_src=s["lm_src"],
                           glasses_frac=round(s["glasses_frac"], 5))
                for kk in ("target_pose", "base_id"):
                    if kk in meta:
                        rec[kk] = meta[kk]
                index.write(json.dumps(rec) + "\n")
                idx += 1
            index.flush()
            donef.write(key + "\n")
            donef.flush()
            for kk, v in st.items():
                total[kk] = total.get(kk, 0) + v
            el = time.time() - t0
            print(f"  task {k + 1}/{len(tasks)}, seen {total['seen']}, kept {total['kept']}, "
                  f"{total['seen'] / max(el, 1e-6):.1f} images/s", end="\r", flush=True)
    rej = ", ".join(f"{k} {v}" for k, v in total.items() if k not in ("seen", "kept") and not k.startswith("lm_") and v)
    lsrc = ", ".join(f"{k[3:]} {v}" for k, v in total.items() if k.startswith("lm_"))
    print(f"\ndone: saw {total['seen']} images/frames, kept {total['kept']} faces (rejected: {rej}). "
          f"Landmark sources: {lsrc}. Pool now has {idx} images: {a.out}, took {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
