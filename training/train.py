#!/usr/bin/env python3
"""
Train the frame segmentation network
====================================
  # smoke test: 256 input, 500 steps
  python training/train.py --out runs/smoke --size 256 --steps 500 --val_every 250 --batch 12
  # full training (in the background)
  python training/train.py --out runs/r34_384 --steps 30000
  # resume from a checkpoint
  python training/train.py --out runs/r34_384 --steps 60000 --resume

Every --val_every steps: boundary F1 (2 px tolerance) / IoU / Dice on the fixed validation set, save last.pt; best.pt on the best boundary F1;
write log.csv, curves.png, preview_XXXXXX.jpg (validation samples + real test-video frames).
Before training starts, a plot of the "pose distribution actually sampled" is written to pose_sampling.png.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from dataset import FixedSet, SynthDataset, worker_init  # noqa: E402
from metrics import seg_metrics  # noqa: E402
from model import build_model, normalize_batch  # noqa: E402


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------------
# loss
# ----------------------------------------------------------------------------
def soft_dice(p, t, eps=1.0):
    num = 2 * (p * t).flatten(1).sum(1) + eps
    den = p.flatten(1).sum(1) + t.flatten(1).sum(1) + eps
    return 1 - num / den


def seg_loss(logits, target, weight, pos_weight=6.0, lens_w=0.3):
    logits = logits.float()
    w = weight.unsqueeze(1)
    pw = torch.tensor([pos_weight, 2.0], device=logits.device).view(1, 2, 1, 1)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none") * w
    p = torch.sigmoid(logits)
    lf = bce[:, 0].mean() + soft_dice(p[:, 0] * weight, target[:, 0] * weight).mean()
    ll = bce[:, 1].mean() + soft_dice(p[:, 1], target[:, 1]).mean()
    return lf + lens_w * ll, lf.detach(), ll.detach()


# ----------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, device, amp):
    model.eval()
    rows = []
    for b in loader:
        x = normalize_batch(b["image"].to(device))
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            prob = torch.sigmoid(model(x).float()).cpu().numpy()
        tgt = b["target"].numpy()
        for k in range(len(prob)):
            m = seg_metrics(prob[k, 0], tgt[k, 0])
            m["lens_iou"] = seg_metrics(prob[k, 1], tgt[k, 1])["iou"]
            m["idx"] = int(b["idx"][k])
            rows.append(m)
    model.train()
    pos = [r for r in rows if r["has_gt"]]
    neg = [r for r in rows if not r["has_gt"]]
    out = dict(bf1=float(np.mean([r["bf1"] for r in pos])) if pos else 0.0,
               iou=float(np.mean([r["iou"] for r in pos])) if pos else 0.0,
               dice=float(np.mean([r["dice"] for r in pos])) if pos else 0.0,
               lens_iou=float(np.mean([r["lens_iou"] for r in pos])) if pos else 0.0,
               neg_fp=float(np.mean([r["fp_frac"] for r in neg])) if neg else 0.0)
    return out, rows


# ----------------------------------------------------------------------------
# preview: validation samples + real video frames
# ----------------------------------------------------------------------------
def real_preview_crops(size, cache):
    """Crop a few face frames the pipeline way from the real sample video input/example.mp4 (done once, cached to cache)."""
    if os.path.exists(cache):
        d = np.load(cache)
        return list(d["crops"])
    from common import biggest, crop_face, load_models
    m = load_models(["CPUExecutionProvider"], parsing=False)
    crops = []
    for v, frames in (("example.mp4", (30, 100, 200, 300)),):
        cap = cv2.VideoCapture(os.path.join(ROOT, "input", v))
        for fi in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, f = cap.read()
            if not ok:
                continue
            fs = m["det"].detect(f)
            if not fs:
                continue
            c, _, _ = crop_face(f, biggest(fs).bbox)
            crops.append(cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA))
    np.savez_compressed(cache, crops=np.stack(crops))
    return crops


@torch.no_grad()
def make_preview(model, val_set, real, device, amp, path, n_val=8):
    model.eval()
    tiles = []
    idx = np.linspace(0, len(val_set) - 1, n_val).astype(int)
    imgs = [val_set[int(i)]["image"].numpy() for i in idx] + list(real)
    gts = [val_set[int(i)]["target"].numpy() for i in idx] + [None] * len(real)
    x = normalize_batch(torch.from_numpy(np.stack(imgs)).to(device))
    probs = []
    for k in range(0, len(x), 8):
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            probs.append(torch.sigmoid(model(x[k:k + 8]).float()).cpu().numpy())
    probs = np.concatenate(probs)
    for im, gt, pr in zip(imgs, gts, probs):
        def ov(m):
            o = im.astype(np.float32) * 0.5
            o[..., 1] = np.maximum(o[..., 1], m[0] * 255)
            o[..., 2] = np.maximum(o[..., 2], m[1] * 120)
            return o.astype(np.uint8)
        g = ov(gt) if gt is not None else np.zeros_like(im)
        tiles.append(np.hstack([im, g, ov(pr)]))
    rows = [np.hstack(tiles[i:i + 2]) if i + 1 < len(tiles) else np.hstack([tiles[i], np.zeros_like(tiles[i])])
            for i in range(0, len(tiles), 2)]
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    model.train()


def plot_curves(log_path, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(log_path)))
    if not rows:
        return
    st = [int(r["step"]) for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for k in ("loss", "loss_frame", "loss_lens"):
        ax[0].plot(st, [float(r[k]) for r in rows], label=k)
    ax[0].set_yscale("log")
    ax[0].legend()
    v = [r for r in rows if r.get("val_bf1")]
    for k in ("val_bf1", "val_iou", "val_dice", "val_lens_iou"):
        ax[1].plot([int(r["step"]) for r in v], [float(r[k]) for r in v], marker="o", label=k)
    ax[1].axhline(0.85, color="gray", ls="--", lw=1)
    ax[1].legend()
    ax[2].plot(st, [float(r["lr"]) for r in rows])
    ax[2].set_title("lr")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def plot_pose_sampling(pool, out, n=6000):
    """Before training: the pose distribution the sampling strategy will actually produce (no rendering, only the sampling logic)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dataset import load_recs, split_ids
    from synth.pipeline import PoseSampler
    tr, _ = split_ids(load_recs(pool))
    sp = PoseSampler(tr)
    rng = np.random.default_rng(0)
    P, cats, meth = [], [], []
    for _ in range(n):
        t, cat = sp.target(rng)
        flip = rng.random() < 0.5
        tt = t.copy()
        if flip:
            tt[1:] *= -1
        r = sp.recs[sp.pick(tt, rng)]
        pose = np.array([r["pitch"], r["yaw"], tt[2]]) * np.array([1, -1 if flip else 1, -1 if flip else 1])
        P.append(pose)
        cats.append(cat)
        meth.append(r.get("method", "real"))
    P = np.array(P)
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    for k, nm in enumerate(("pitch", "yaw", "roll")):
        ax[k].hist(P[:, k], bins=np.arange(-80, 81, 5))
        ax[k].set_title(nm)
    h, _, _ = np.histogram2d(P[:, 0], P[:, 1], bins=[np.arange(-50, 51, 10), np.arange(-80, 81, 10)])
    ax[3].imshow(h, origin="lower", extent=[-80, 80, -50, 50], aspect="auto")
    ax[3].set_title("pitch x yaw sampled")
    big = (np.abs(P) > 25).sum(1)
    from collections import Counter
    fig.suptitle(f"n={n}  categories {dict(Counter(cats))}  methods {dict(Counter(meth))}  "
                 f"2+ axes >25°: {(big >= 2).mean():.0%}  3 axes: {(big >= 3).mean():.1%}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.path.join(ROOT, "data", "faces_pool"))
    ap.add_argument("--val", default=os.path.join(ROOT, "data", "val_synth"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--enc_lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--pos_weight", type=float, default=6.0)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--no_amp", action="store_true", help="disable fp16 autocast")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default=None, help="initialise the weights from another checkpoint (for fine-tuning)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    device = pick_device()
    torch.manual_seed(a.seed)
    cfg = dict(encoder=a.encoder, size=a.size, crop_scale=1.3, y_shift=0.08, channels=["frame", "lens"],
               mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], input="BGR uint8 pipeline crop -> RGB")
    model = build_model(cfg).to(device)
    enc_params = list(model.enc.parameters())
    enc_ids = set(id(p) for p in enc_params)
    dec_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([dict(params=enc_params, lr=a.enc_lr), dict(params=dec_params, lr=a.lr)], weight_decay=a.wd)
    base_lrs = [a.enc_lr, a.lr]

    def lr_factor(step):
        if step < a.warmup:
            return (step + 1) / a.warmup
        t = (step - a.warmup) / max(1, a.steps - a.warmup)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    step, best = 0, -1.0
    last = os.path.join(a.out, "last.pt")
    if a.resume and os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck.get("best", -1.0)
        print(f"resuming from step {step} (best bf1 {best:.4f})")
    elif a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"initialised from {a.init}")
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)

    if step == 0:
        plot_pose_sampling(a.pool, os.path.join(a.out, "pose_sampling.png"))
    val_set = FixedSet(a.val)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=2)
    real = real_preview_crops(a.size, os.path.join(ROOT, "data", f"real_preview_{a.size}.npz"))

    train_set = SynthDataset(a.pool, "train", size=a.size, epoch_len=a.batch * 1000, seed=a.seed)
    amp = (not a.no_amp) and device.type in ("mps", "cuda")
    log_path = os.path.join(a.out, "log.csv")
    new_log = not (a.resume and os.path.exists(log_path))
    logf = open(log_path, "w" if new_log else "a", newline="")
    fields = ["step", "lr", "loss", "loss_frame", "loss_lens", "sec_per_step", "data_wait", "val_bf1", "val_iou",
              "val_dice", "val_lens_iou", "val_neg_fp", "mem_gb"]
    writer = csv.DictWriter(logf, fieldnames=fields)
    if new_log:
        writer.writeheader()

    model.train()
    epoch = step // 1000
    acc = dict(loss=0.0, lf=0.0, ll=0.0, n=0, wait=0.0)
    t_last = time.time()
    nan_count = 0
    print(f"device {device}, amp={amp}, batch {a.batch}, input {a.size}, training to step {a.steps}", flush=True)
    while step < a.steps:
        train_set.set_epoch(epoch)
        loader = DataLoader(train_set, batch_size=a.batch, shuffle=False, num_workers=a.workers,
                            worker_init_fn=worker_init, persistent_workers=False, drop_last=True,
                            prefetch_factor=4 if a.workers else None)
        t_wait = time.time()
        for b in loader:
            acc["wait"] += time.time() - t_wait
            f = lr_factor(step)
            for g, lr0 in zip(opt.param_groups, base_lrs):
                g["lr"] = lr0 * f
            x = normalize_batch(b["image"].to(device, non_blocking=True))
            tgt = b["target"].to(device)
            w = b["weight"].to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss, lf, ll = seg_loss(logits, tgt, w, a.pos_weight)
            if not torch.isfinite(loss):
                nan_count += 1
                opt.zero_grad(set_to_none=True)
                if amp and nan_count >= 3:
                    print("repeated NaN under fp16, falling back to fp32", flush=True)
                    amp = False
                t_wait = time.time()
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step += 1
            acc["loss"] += float(loss.detach())
            acc["lf"] += float(lf)
            acc["ll"] += float(ll)
            acc["n"] += 1
            row = None
            if step % a.log_every == 0:
                dt = (time.time() - t_last) / acc["n"]
                mem = torch.mps.driver_allocated_memory() / 1e9 if device.type == "mps" else 0.0
                row = dict(step=step, lr=f"{a.lr * f:.2e}", loss=f"{acc['loss'] / acc['n']:.4f}",
                           loss_frame=f"{acc['lf'] / acc['n']:.4f}", loss_lens=f"{acc['ll'] / acc['n']:.4f}",
                           sec_per_step=f"{dt:.3f}", data_wait=f"{acc['wait'] / acc['n']:.3f}", mem_gb=f"{mem:.1f}")
                print(f"step {step}  loss {acc['loss'] / acc['n']:.4f}  {dt:.2f}s/step (data wait {acc['wait'] / acc['n']:.2f}s)"
                      f"  mps {mem:.1f}GB", flush=True)
                acc = dict(loss=0.0, lf=0.0, ll=0.0, n=0, wait=0.0)
                t_last = time.time()
            if step % a.val_every == 0 or step == a.steps:
                res, _ = validate(model, val_loader, device, amp)
                row = row or dict(step=step, lr=f"{a.lr * f:.2e}", loss="", loss_frame="", loss_lens="")
                row.update(val_bf1=f"{res['bf1']:.4f}", val_iou=f"{res['iou']:.4f}", val_dice=f"{res['dice']:.4f}",
                           val_lens_iou=f"{res['lens_iou']:.4f}", val_neg_fp=f"{res['neg_fp']:.5f}")
                ck = dict(model=model.state_dict(), opt=opt.state_dict(), step=step, cfg=cfg, best=max(best, res["bf1"]),
                          val=res)
                torch.save(ck, last)
                if res["bf1"] > best:
                    best = res["bf1"]
                    torch.save(dict(model=model.state_dict(), cfg=cfg, step=step, val=res), os.path.join(a.out, "best.pt"))
                make_preview(model, val_set, real, device, amp, os.path.join(a.out, f"preview_{step:06d}.jpg"))
                print(f"[val] step {step}  boundary F1 {res['bf1']:.4f}  IoU {res['iou']:.4f}  Dice {res['dice']:.4f}  "
                      f"lens IoU {res['lens_iou']:.4f}  negative FP {res['neg_fp']:.5f}  (best {best:.4f})", flush=True)
                t_last = time.time()
            if row:
                writer.writerow({k: row.get(k, "") for k in fields})
                logf.flush()
                if "val_bf1" in row:
                    plot_curves(log_path, os.path.join(a.out, "curves.png"))
            if step >= a.steps:
                break
            t_wait = time.time()
        epoch += 1
    logf.close()
    print(f"training finished: {step} steps, best boundary F1 {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
