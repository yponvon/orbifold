"""Iron stage 3b: joint shape + pose refinement against the traced masks (differentiable).

Silhouettes are rendered with gsplat from opaque Gaussians sampled on the mesh surface
(alpha only), compared to the traced SAM masks by soft IoU with hand / body-occluder pixels
ignored. Free: anisotropic scale, per-vertex offsets (Laplacian-smoothed, soleplate kept
flat at z=0), bed height, a constant grasp transform (palm-horizontal frame: dx, dy, dyaw)
and per-frame 6-DoF poses tied to it (prior), to the cloth plane (sole on the bed,
roll/pitch ~ 0) and to their neighbours. exocam1 weighted x2 (it pins height and tilt).

Init: poses.npz from 20_iron_fit (rigid silhouette fit). Writes shape_refined.npz,
poses.npz (refined; old one kept as poses_silfit.npz), refine_metrics.json, preview_refine.jpg.
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

from orbifold.geometry import CONVENTIONS

sys.path.insert(0, "scripts")
C = importlib.import_module("20_iron_common")
dev = "cuda"
torch.manual_seed(0)
NF, CAMS = 60, C.CAMS
WCAM = {"headcam": 1.0, "exocam1": 2.0, "exocam2": 0.5}
STEPS = int(os.environ.get("IRON_REFINE_STEPS", 2500))

cams = np.load("out/clip/cameras.npz")
M4 = np.eye(4); M4[:3, :3] = CONVENTIONS["ros_body"]
VM = {c: torch.tensor(np.stack([np.linalg.inv(T @ M4) for T in cams[f"{c}_T"]]), dtype=torch.float32, device=dev) for c in CAMS}
KK = {c: torch.tensor(cams[f"{c}_K"], dtype=torch.float32, device=dev) for c in CAMS}
J = np.load("out/clip/joints.npz")
idx = {n: i for i, n in enumerate(J["names"])}
xyz = J["xyz"]
kn = xyz[:, [idx["right_index_mcp"], idx["right_middle_mcp"], idx["right_pinky_mcp"]]].mean(1)
palm = (xyz[:, idx["right_wrist"]] + kn) / 2
hd = kn[:, :2] - xyz[:, idx["right_wrist"], :2]
yaw_hand = np.arctan2(hd[:, 1], hd[:, 0])

src = f"{C.OUT}/poses_silfit.npz"
if not os.path.exists(src):
    shutil.copy(f"{C.OUT}/poses.npz", src)
P0 = np.load(src)
T_init = P0["T"]

V0, F, part = C.build_mesh_v2()
V0[:, :2] *= float(P0["sxy"])

# --- targets ---------------------------------------------------------------------------
views = []
for cam in CAMS:
    for f in range(NF):
        m, h = C.load_masks(cam, f)
        if m.sum() < 50:
            continue
        ig = h.copy()
        p = f"{C.OUT}/occluder/{cam}/{f:04d}.png"
        if os.path.exists(p):
            ig |= cv2.imread(p, 0) > 0
        ig &= ~m
        ys, xs = np.nonzero(m)
        pad = 140 if cam != "exocam2" else 60
        x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
        x1, y1 = min(1920, xs.max() + pad), min(1080, ys.max() + pad)
        sl = (slice(y0, y1), slice(x0, x1))
        views.append(dict(cam=cam, f=f, bb=(x0, y0, x1, y1),
                          m=torch.tensor(m[sl], dtype=torch.float32, device=dev),
                          w=torch.tensor(~ig[sl], dtype=torch.float32, device=dev)))
print("views", len(views), {c: sum(v["cam"] == c for v in views) for c in CAMS}, flush=True)

# --- parameters --------------------------------------------------------------------------
rng = np.random.default_rng(0)
Vt = torch.tensor(V0, dtype=torch.float32, device=dev)
Ft = torch.tensor(F, dtype=torch.long, device=dev)
area0 = 0.5 * np.linalg.norm(np.cross(V0[F[:, 1]] - V0[F[:, 0]], V0[F[:, 2]] - V0[F[:, 0]]), axis=1)
k = np.clip(np.round(area0 / area0.mean() * 4), 1, 16).astype(int)
tri = torch.tensor(np.repeat(np.arange(len(F)), k), device=dev)
r1, r2 = rng.random(len(tri)), rng.random(len(tri))
s_ = np.sqrt(r1)
bary = torch.tensor(np.stack([1 - s_, s_ * (1 - r2), s_ * r2], 1), dtype=torch.float32, device=dev)
kt = torch.tensor(k, dtype=torch.float32, device=dev)[tri]
print("silhouette gaussians", len(tri), flush=True)

# uniform Laplacian
nbr = [set() for _ in range(len(V0))]
for a, b, c in F:
    nbr[a] |= {b, c}; nbr[b] |= {a, c}; nbr[c] |= {a, b}
rows = np.concatenate([[i] * len(n) for i, n in enumerate(nbr)])
cols = np.concatenate([list(n) for n in nbr])
vals = np.concatenate([[1.0 / len(n)] * len(n) for n in nbr])
Lap = torch.sparse_coo_tensor(np.stack([rows, cols]), vals, (len(V0), len(V0))).float().to(dev)
lap0 = Vt - torch.sparse.mm(Lap, Vt)
sole = torch.tensor(V0[:, 2] < 1e-6, device=dev)
# left-right mirror partner of every vertex (the procedural mesh is symmetric in y)
_mir = V0 * [1, -1, 1]
_d = ((V0[:, None, :] - _mir[None, :, :]) ** 2).sum(-1)
mirror = torch.tensor(_d.argmin(1), device=dev)
print("mirror match max err mm", 1000 * np.sqrt(_d.min(1).max()), flush=True)


def yaw_of(T):
    return np.arctan2(T[1, 0], T[0, 0])


x_init = np.stack([T_init[:, 0, 3], T_init[:, 1, 3], T_init[:, 2, 3],
                   np.unwrap([yaw_of(T) for T in T_init]), np.zeros(NF), np.zeros(NF)], 1)
rel = []
for f in range(NF):
    c, s = np.cos(yaw_hand[f]), np.sin(yaw_hand[f])
    d = x_init[f, :2] - palm[f, :2]
    rel.append([c * d[0] + s * d[1], -s * d[0] + c * d[1], np.angle(np.exp(1j * (x_init[f, 3] - yaw_hand[f])))])
G_init = np.median(np.array(rel), 0)
P = torch.nn.Parameter
pose = P(torch.tensor(x_init, dtype=torch.float32, device=dev))      # x y z yaw roll pitch
G = P(torch.tensor(G_init, dtype=torch.float32, device=dev))
bed = P(torch.tensor(float(P0["bed_z"]), device=dev))
log_s = P(torch.zeros(3, device=dev))
dV = P(torch.zeros_like(Vt))
palm_t = torch.tensor(palm, dtype=torch.float32, device=dev)
yh = torch.tensor(np.unwrap(yaw_hand), dtype=torch.float32, device=dev)
# make yaw_hand + G[2] continuous with pose yaw
off = np.round((x_init[:, 3] - np.unwrap(yaw_hand) - G_init[2]) / (2 * np.pi))
yh = yh + torch.tensor(off * 2 * np.pi, dtype=torch.float32, device=dev)


def rotm(p):
    yaw, roll, pitch = p[3], p[4], p[5]
    cy, sy, cr, sr, cp, sp = torch.cos(yaw), torch.sin(yaw), torch.cos(roll), torch.sin(roll), torch.cos(pitch), torch.sin(pitch)
    o, z = torch.ones_like(cy), torch.zeros_like(cy)
    Rz = torch.stack([torch.stack([cy, -sy, z]), torch.stack([sy, cy, z]), torch.stack([z, z, o])])
    Ry = torch.stack([torch.stack([cp, z, sp]), torch.stack([z, o, z]), torch.stack([-sp, z, cp])])
    Rx = torch.stack([torch.stack([o, z, z]), torch.stack([z, cr, -sr]), torch.stack([z, sr, cr])])
    return Rz @ Ry @ Rx


def shape():
    d = dV * torch.stack([torch.ones_like(sole, dtype=torch.float32)] * 2 + [(~sole).float()], 1)
    return (Vt + d) * torch.exp(log_s)


def sil(view, Vs):
    f = view["f"]
    R = rotm(pose[f]); t = pose[f, :3]
    tv = Vs[Ft[tri]]
    pts = (bary[:, :, None] * tv).sum(1)
    means = pts @ R.T + t
    e1, e2 = tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0]
    ar = 0.5 * torch.linalg.norm(torch.cross(e1, e2, dim=-1), dim=-1)
    sig = (0.75 * torch.sqrt(ar / kt)).clamp(6e-4, 6e-3)
    N = len(tri)
    x0, y0, x1, y1 = view["bb"]
    K = KK[view["cam"]].clone(); K[0, 2] -= x0; K[1, 2] -= y0
    _, alpha, _ = rasterization(means, torch.tensor([[1.0, 0, 0, 0]], device=dev).expand(N, 4), sig[:, None].expand(N, 3),
                                torch.full((N,), 0.97, device=dev), torch.ones(N, 1, device=dev),
                                VM[view["cam"]][f][None], K[None], x1 - x0, y1 - y0, packed=False, near_plane=0.02)
    return alpha[0, ..., 0]


def soft_iou(a, v):
    m, w = v["m"], v["w"]
    inter = (a * m * w).sum()
    uni = ((a + m - a * m) * w).sum()
    return 1 - inter / uni.clamp(min=1)


opt = torch.optim.Adam([dict(params=[pose], lr=1e-3), dict(params=[G], lr=1e-3), dict(params=[bed], lr=5e-4),
                        dict(params=[log_s], lr=2e-3), dict(params=[dV], lr=1.5e-4)])
sig_prior = torch.tensor([0.015, 0.015, 0.087], device=dev)
B = 8
for it in range(STEPS + 1):
    shape_on = it >= STEPS // 5      # pose-only warm-up
    Vs = shape()
    batch = rng.choice(len(views), B, replace=False)
    ls, wsum = 0, 0
    for bi in batch:
        v = views[bi]
        ls = ls + WCAM[v["cam"]] * soft_iou(sil(v, Vs), v)
        wsum += WCAM[v["cam"]]
    l_sil = ls / wsum
    # grasp prior (palm-horizontal frame)
    c, s = torch.cos(yh), torch.sin(yh)
    gxy = palm_t[:, :2] + torch.stack([c * G[0] - s * G[1], s * G[0] + c * G[1]], 1)
    r = torch.cat([pose[:, :2] - gxy, (pose[:, 3] - yh - G[2])[:, None]], 1) / sig_prior
    l_grasp = r.pow(2).mean()
    # soleplate on the cloth plane: z ~ bed (one-sided softer above), roll/pitch ~ 0
    lift = pose[:, 2] - bed
    l_plane = (Fn.relu(-lift) / 0.002).pow(2).mean() + (lift / 0.01).pow(2).mean() + (pose[:, 4:] / 0.026).pow(2).mean()
    # temporal smoothness (second differences)
    acc = pose[2:] - 2 * pose[1:-1] + pose[:-2]
    l_t = (acc / torch.tensor([0.004, 0.004, 0.003, 0.03, 0.02, 0.02], device=dev)).pow(2).mean()
    dVm = dV * torch.tensor([1.0, -1.0, 1.0], device=dev)
    l_shape = 1e5 * (Vs / torch.exp(log_s) - torch.sparse.mm(Lap, Vs / torch.exp(log_s)) - lap0).pow(2).sum(1).mean() \
        + 1e3 * dV.pow(2).sum(1).mean() + 1e3 * Fn.relu(dV.norm(dim=1) - 0.006).pow(2).sum() \
        + 1e3 * (dV - dVm[mirror]).pow(2).sum(1).mean() + (log_s / 0.1).pow(2).sum() * 0.05
    loss = l_sil + 0.02 * l_grasp + 0.02 * l_plane + 0.02 * l_t + l_shape
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if not shape_on:
        dV.grad = None; log_s.grad = None
    opt.step()
    if it % 250 == 0:
        print(f"it {it} sil {l_sil.item():.4f} grasp {l_grasp.item():.3f} plane {l_plane.item():.3f} t {l_t.item():.3f} "
              f"shape {l_shape.item():.4f} scale {torch.exp(log_s).detach().cpu().numpy().round(3)} bed {bed.item():.4f} "
              f"|dV|max {dV.abs().max().item() * 1000:.1f}mm mem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB", flush=True)

# --- save -----------------------------------------------------------------------------------
with torch.no_grad():
    Vf = shape().cpu().numpy().astype(np.float64)
    Pf = pose.detach().cpu().numpy()
    T = np.stack([np.eye(4)] * NF)
    for f in range(NF):
        T[f, :3, :3] = rotm(pose[f]).cpu().numpy(); T[f, :3, 3] = Pf[f, :3]
np.savez(f"{C.OUT}/shape_refined.npz", vertices=Vf, faces=F, part=part, scale=torch.exp(log_s).detach().cpu().numpy())
bed_f = float(bed.item())
top = Vf[:, 2].max()
clear = palm[:, 2] - (Pf[:, 2] + top)     # flat iron: handle top height = z + top
np.savez(f"{C.OUT}/poses.npz", T=T, params=Pf, bed_z=bed_f, sxy=float(P0["sxy"]), grasp=G.detach().cpu().numpy(),
         palm_clearance=clear, handle_top=top)
# grasp consistency: iron pose in the palm-horizontal frame
relf = []
for f in range(NF):
    c, s = np.cos(yaw_hand[f]), np.sin(yaw_hand[f])
    d = Pf[f, :2] - palm[f, :2]
    relf.append([c * d[0] + s * d[1], -s * d[0] + c * d[1], np.degrees(np.angle(np.exp(1j * (Pf[f, 3] - yaw_hand[f]))))])
relf = np.array(relf)
res = dict(bed_z=bed_f, scale=torch.exp(log_s).detach().cpu().numpy().tolist(),
           size_m=dict(length=float(np.ptp(Vf[:, 0])), width=float(np.ptp(Vf[:, 1])), height=float(top)),
           dV_max_mm=float(dV.abs().max().item() * 1000),
           grasp_std=dict(dx_cm=float(relf[:, 0].std() * 100), dy_cm=float(relf[:, 1].std() * 100), yaw_deg=float(relf[:, 2].std())),
           lift_cm=dict(max=float(100 * (Pf[:, 2] - bed_f).max()), min=float(100 * (Pf[:, 2] - bed_f).min())),
           tilt_deg_max=float(np.degrees(np.abs(Pf[:, 4:]).max())),
           palm_clearance_cm=dict(min=float(100 * clear.min()), median=float(100 * np.median(clear)), frames_below_1p5=int((clear < 0.015).sum())))
# hard IoU with the cv2 rasteriser at full resolution
from orbifold.geometry import transform  # noqa: E402
for cam in CAMS:
    vals = []
    for f in range(NF):
        m, h = C.load_masks(cam, f)
        if m.sum() < 50:
            continue
        ig = h.copy()
        p = f"{C.OUT}/occluder/{cam}/{f:04d}.png"
        if os.path.exists(p):
            ig |= cv2.imread(p, 0) > 0
        uv, d = C.cam_project(transform(T[f], Vf), cams[f"{cam}_K"], cams[f"{cam}_T"][f])
        cov = C.raster(uv, d, F, 1080, 1920) >= 0
        vals.append(C.iou(cov, m, ig & ~m))
    res[f"iou_{cam}"] = float(np.mean(vals)); res[f"iou_{cam}_min"] = float(np.min(vals)); res[f"n_{cam}"] = len(vals)
print(json.dumps(res, indent=1))
json.dump(res, open(f"{C.OUT}/refine_metrics.json", "w"), indent=1)
