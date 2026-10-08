"""Iron stage 3c: fix bad pose frames (shape frozen) + clean merged-blob masks.

1. Mask cleaning: a traced mask that splits into >1 sizeable blob under a 9x9 opening
   (e.g. headcam f42-45, where a phone on the bed merged into the iron mask) keeps only
   the blob that best overlaps the current fitted projection. Raw masks kept in masks_raw/.
2. Pose re-fit: per-frame 6-DoF on the frozen refined shape against the silhouettes
   (soft IoU, gsplat opaque surface Gaussians), exocam1 x2, exocam2 x0.25. Frames whose
   IoU is poor are re-initialised from a multi-start search seeded by their good
   neighbours (interpolated pose x lift x pitch x yaw offsets). Stronger temporal
   smoothing than 20_iron_refine; the soleplate may lift off the cloth (one-sided plane
   prior) and pitch, but stays smooth.

Writes poses.npz (old one kept as poses_refine.npz), refit_metrics.json.
"""
import importlib
import json
import os
import shutil
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn
from gsplat import rasterization

from orbifold.geometry import CONVENTIONS, transform

sys.path.insert(0, "scripts")
C = importlib.import_module("20_iron_common")
dev = "cuda"
torch.manual_seed(0)
NF, CAMS = 60, C.CAMS
WCAM = {"headcam": 1.0, "exocam1": 2.0, "exocam2": 0.25}
STEPS = int(os.environ.get("IRON_REFIT_STEPS", 3000))

cams = np.load("out/clip/cameras.npz")
M4 = np.eye(4); M4[:3, :3] = CONVENTIONS["ros_body"]
VM = {c: torch.tensor(np.stack([np.linalg.inv(T @ M4) for T in cams[f"{c}_T"]]), dtype=torch.float32, device=dev) for c in CAMS}
KK = {c: torch.tensor(cams[f"{c}_K"], dtype=torch.float32, device=dev) for c in CAMS}

src = f"{C.OUT}/poses_refine.npz"
if not os.path.exists(src):
    shutil.copy(f"{C.OUT}/poses.npz", src)
P0 = dict(np.load(src))
T_init = P0["T"]
S = np.load(f"{C.OUT}/shape_refined.npz")
V, F = S["vertices"], S["faces"]


def ignore(cam, f, m):
    ig = cv2.imread(f"{C.OUT}/hand/{cam}/{f:04d}.png", 0) > 0
    p = f"{C.OUT}/occluder/{cam}/{f:04d}.png"
    if os.path.exists(p):
        ig |= cv2.imread(p, 0) > 0
    return ig & ~m


def project(cam, f, T):
    uv, d = C.cam_project(transform(T, V), cams[f"{cam}_K"], cams[f"{cam}_T"][f])
    return C.raster(uv, d, F, 1080, 1920) >= 0


# --- 1. mask cleaning ----------------------------------------------------------------------
raw = f"{C.OUT}/masks_raw"
if not os.path.exists(raw):
    shutil.copytree(f"{C.OUT}/masks", raw)
cleaned = []
for cam in CAMS:
    for f in range(NF):
        m = cv2.imread(f"{raw}/{cam}/{f:04d}.png", 0) > 0
        if m.sum() < 50:
            continue
        op = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(op)
        big = [i for i in range(1, n) if st[i, 4] > max(1500, 0.1 * st[1:, 4].max())]
        out = m
        if len(big) > 1:
            cov = cv2.dilate(project(cam, f, T_init[f]).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            best = max(big, key=lambda i: ((lab == i) & cov).sum() / (lab == i).sum())
            keep = cv2.dilate((lab == best).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
            out = m & keep
            cleaned.append((cam, f, int(m.sum()), int(out.sum())))
        cv2.imwrite(f"{C.OUT}/masks/{cam}/{f:04d}.png", out.astype(np.uint8) * 255)
print("cleaned masks", cleaned, flush=True)


def ious(T):
    r = {}
    for cam in CAMS:
        for f in range(NF):
            m, _ = C.load_masks(cam, f)
            if m.sum() < 50:
                continue
            r[cam, f] = C.iou(project(cam, f, T[f]), m, ignore(cam, f, m))
    return r


iou0 = ious(T_init)

# --- 2. views ------------------------------------------------------------------------------
views = []
for cam in CAMS:
    for f in range(NF):
        m, _ = C.load_masks(cam, f)
        if m.sum() < 50:
            continue
        ig = ignore(cam, f, m)
        ys, xs = np.nonzero(m)
        pad = 160 if cam != "exocam2" else 60
        x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
        x1, y1 = min(1920, xs.max() + pad), min(1080, ys.max() + pad)
        sl = (slice(y0, y1), slice(x0, x1))
        views.append(dict(cam=cam, f=f, bb=(x0, y0, x1, y1),
                          m=torch.tensor(m[sl], dtype=torch.float32, device=dev),
                          w=torch.tensor(~ig[sl], dtype=torch.float32, device=dev)))
by_frame = {f: [v for v in views if v["f"] == f] for f in range(NF)}

rng = np.random.default_rng(0)
area0 = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
k = np.clip(np.round(area0 / area0.mean() * 4), 1, 16).astype(int)
tri = np.repeat(np.arange(len(F)), k)
r1, r2 = rng.random(len(tri)), rng.random(len(tri))
s_ = np.sqrt(r1)
bary = np.stack([1 - s_, s_ * (1 - r2), s_ * r2], 1)
PTS = torch.tensor(np.einsum("nk,nkc->nc", bary, V[F[tri]]), dtype=torch.float32, device=dev)
SIG = torch.tensor((0.75 * np.sqrt(area0[tri] / k[tri])).clip(6e-4, 6e-3), dtype=torch.float32, device=dev)
NG = len(PTS)
Q1 = torch.tensor([[1.0, 0, 0, 0]], device=dev).expand(NG, 4)


def rotm(p):
    yaw, roll, pitch = p[..., 3], p[..., 4], p[..., 5]
    cy, sy, cr, sr, cp, sp = torch.cos(yaw), torch.sin(yaw), torch.cos(roll), torch.sin(roll), torch.cos(pitch), torch.sin(pitch)
    o, z = torch.ones_like(cy), torch.zeros_like(cy)
    Rz = torch.stack([torch.stack([cy, -sy, z], -1), torch.stack([sy, cy, z], -1), torch.stack([z, z, o], -1)], -2)
    Ry = torch.stack([torch.stack([cp, z, sp], -1), torch.stack([z, o, z], -1), torch.stack([-sp, z, cp], -1)], -2)
    Rx = torch.stack([torch.stack([o, z, z], -1), torch.stack([z, cr, -sr], -1), torch.stack([z, sr, cr], -1)], -2)
    return Rz @ Ry @ Rx


def sil(view, p):
    R = rotm(p)
    means = PTS @ R.T + p[:3]
    x0, y0, x1, y1 = view["bb"]
    K = KK[view["cam"]].clone(); K[0, 2] -= x0; K[1, 2] -= y0
    _, alpha, _ = rasterization(means, Q1, SIG[:, None].expand(NG, 3), torch.full((NG,), 0.97, device=dev),
                                torch.ones(NG, 1, device=dev), VM[view["cam"]][view["f"]][None], K[None],
                                x1 - x0, y1 - y0, packed=False, near_plane=0.02)
    return alpha[0, ..., 0]


def soft_iou(a, v):
    inter = (a * v["m"] * v["w"]).sum()
    uni = ((a + v["m"] - a * v["m"]) * v["w"]).sum()
    return 1 - inter / uni.clamp(min=1)


def frame_loss(f, p):
    vs = by_frame[f]
    if not vs:
        return torch.tensor(0.0, device=dev)
    return sum(WCAM[v["cam"]] * soft_iou(sil(v, p), v) for v in vs) / sum(WCAM[v["cam"]] for v in vs)


def yaw_of(T):
    return np.arctan2(T[1, 0], T[0, 0])


def pitch_roll(T):
    R = T[:3, :3]
    return np.arctan2(R[2, 1], R[2, 2]), np.arcsin(-np.clip(R[2, 0], -1, 1))


x0 = np.stack([T_init[:, 0, 3], T_init[:, 1, 3], T_init[:, 2, 3], np.unwrap([yaw_of(T) for T in T_init]),
               *np.array([pitch_roll(T) for T in T_init]).T], 1)
bed = float(P0["bed_z"])

# bad frames: weighted IoU score
score = {}
for f in range(NF):
    w = [(WCAM[c], iou0[c, f]) for c in CAMS if (c, f) in iou0 and c != "exocam2"]
    score[f] = sum(a * b for a, b in w) / sum(a for a, _ in w)
bad = [f for f in range(NF) if score[f] < 0.72 or iou0.get(("exocam1", f), 1) < 0.6 or iou0.get(("headcam", f), 1) < 0.7]
print("bad frames", bad, flush=True)

# multi-start search for bad frames, seeded from nearest good neighbours
good = [f for f in range(NF) if f not in bad]
for f in bad:
    lo = max([g for g in good if g < f], default=None)
    hi = min([g for g in good if g > f], default=None)
    seeds = [x0[f]]
    if lo is not None and hi is not None:
        a = (f - lo) / (hi - lo)
        seeds.append((1 - a) * x0[lo] + a * x0[hi])
    else:
        seeds.append(x0[lo if lo is not None else hi])
    cands = []
    for sd in seeds:
        for dz in (0.0, 0.015, 0.03, 0.05):
            for dpitch in (0.0, -0.12, 0.12):
                for dyaw in (0.0, -0.15, 0.15):
                    c_ = sd.copy(); c_[2] = max(c_[2], bed) + dz; c_[5] += dpitch; c_[3] += dyaw
                    cands.append(c_)
    with torch.no_grad():
        ls = [frame_loss(f, torch.tensor(c_, dtype=torch.float32, device=dev)).item() for c_ in cands]
    order = np.argsort(ls)[:4]
    best, bl = None, 9
    for i in order:   # short local polish of the top candidates
        p = torch.nn.Parameter(torch.tensor(cands[i], dtype=torch.float32, device=dev))
        o = torch.optim.Adam([p], lr=2e-3)
        for _ in range(150):
            l = frame_loss(f, p) + 0.05 * (Fn.relu(bed - p[2]) / 0.002).pow(2)
            o.zero_grad(); l.backward(); o.step()
        l = frame_loss(f, p).item()
        if l < bl:
            bl, best = l, p.detach().cpu().numpy()
    print(f"  f{f}: loss {frame_loss(f, torch.tensor(x0[f], dtype=torch.float32, device=dev)).item():.3f} -> {bl:.3f} "
          f"lift {100 * (best[2] - bed):.1f}cm pitch {np.degrees(best[5]):.1f} roll {np.degrees(best[4]):.1f}", flush=True)
    x0[f] = best

# --- 3. joint smooth re-fit of all frames -------------------------------------------------------
pose = torch.nn.Parameter(torch.tensor(x0, dtype=torch.float32, device=dev))
opt = torch.optim.Adam([pose], lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, 1e-4)
acc_s = torch.tensor([0.002, 0.002, 0.002, 0.015, 0.01, 0.01], device=dev)
vel_s = torch.tensor([0.02, 0.02, 0.01, 0.1, 0.03, 0.03], device=dev)
B = 12
for it in range(STEPS + 1):
    batch = rng.choice(len(views), B, replace=False)
    l_sil = sum(WCAM[views[i]["cam"]] * soft_iou(sil(views[i], pose[views[i]["f"]]), views[i]) for i in batch) \
        / sum(WCAM[views[i]["cam"]] for i in batch)
    lift = pose[:, 2] - bed
    l_plane = (Fn.relu(-lift) / 0.002).pow(2).mean() + (lift / 0.04).pow(2).mean() \
        + (pose[:, 4] / 0.05).pow(2).mean() + (pose[:, 5] / 0.1).pow(2).mean()
    acc = pose[2:] - 2 * pose[1:-1] + pose[:-2]
    vel = pose[1:] - pose[:-1]
    l_t = (acc / acc_s).pow(2).mean() + 0.2 * (vel / vel_s).pow(2).mean()
    loss = l_sil + 0.01 * l_plane + 0.03 * l_t
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step(); sched.step()
    if it % 500 == 0:
        print(f"it {it} sil {l_sil.item():.4f} plane {l_plane.item():.3f} t {l_t.item():.3f}", flush=True)

Pf = pose.detach().cpu().numpy()
with torch.no_grad():
    Rf = rotm(pose).cpu().numpy()
T = np.stack([np.eye(4)] * NF)
T[:, :3, :3] = Rf; T[:, :3, 3] = Pf[:, :3]
iou1 = ious(T)
P0.update(T=T, params=Pf)
np.savez(f"{C.OUT}/poses.npz", **P0)
res = dict(cleaned_masks=cleaned, bad_frames=bad,
           lift_cm=dict(max=float(100 * (Pf[:, 2] - bed).max()), min=float(100 * (Pf[:, 2] - bed).min())),
           pitch_deg_max=float(np.degrees(np.abs(Pf[:, 5]).max())), roll_deg_max=float(np.degrees(np.abs(Pf[:, 4]).max())))
for cam in CAMS:
    ks = [k_ for k_ in iou0 if k_[0] == cam]
    res[cam] = dict(before=float(np.mean([iou0[k_] for k_ in ks])), after=float(np.mean([iou1[k_] for k_ in ks])),
                    min_before=float(np.min([iou0[k_] for k_ in ks])), min_after=float(np.min([iou1[k_] for k_ in ks])),
                    per_frame={int(k_[1]): round(float(iou1[k_]), 3) for k_ in ks})
    print(cam, " ".join(f"{k_[1]}:{iou0[k_]:.2f}->{iou1[k_]:.2f}" for k_ in ks), flush=True)
json.dump(res, open(f"{C.OUT}/refit_metrics.json", "w"), indent=1)
print(json.dumps({k_: v for k_, v in res.items() if k_ not in CAMS} | {c: {k2: res[c][k2] for k2 in ("before", "after", "min_before", "min_after")} for c in CAMS}, indent=1))
