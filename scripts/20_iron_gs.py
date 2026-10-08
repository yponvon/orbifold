"""Iron stage 6: mesh-anchored Gaussians on the rigid iron, trained with gsplat.

Gaussians live on the iron mesh triangles (K per triangle by area, fixed barycentric
coords, learned offset in the triangle frame) and move rigidly with the per-frame pose
from 20_iron_fit (plus a small learned per-frame (dx, dy, dz, dyaw) refinement).
Learned per Gaussian: SH deg 0-1 colour, opacity, scale, rotation, offset; per camera a
3x3 + bias colour affine. Loss on the ORIGINAL 1920x1080 frames: L1 + SSIM inside the
traced iron mask, alpha vs mask outside it (not on exocam2, whose SAM masks are partial);
hand / body-occluder pixels excluded. Every 10th frame (0, 10, ..., 50) is held out.
Antialiased (mip-splatting style) rasterisation; scales capped (in-plane <= 8 mm,
anisotropy <= 4:1, normal axis <= smallest in-plane axis) so there are no needles.

Outputs (out/assets/iron/):
  gs.pt, render/<cam>/<f>.png (RGBA, straight alpha, sRGB), depth/<cam>/<f>.npy (float16,
  camera-z metres, inf where alpha < 0.05), preview_gs.jpg, gs_metrics.json.
"""
import importlib
import json
import os
import sys

import cv2
import lpips
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
C0 = 0.28209479177387814
STEPS = int(os.environ.get("IRON_GS_STEPS", 15000))
RMODE = "antialiased"


def heldout(f):
    return f % 10 == 0


# --- data ------------------------------------------------------------------------------
cams = np.load("out/clip/cameras.npz")
M = np.eye(4)
M[:3, :3] = CONVENTIONS["ros_body"]
VM = {c: torch.tensor(np.stack([np.linalg.inv(T @ M) for T in cams[f"{c}_T"]]), dtype=torch.float32, device=dev) for c in CAMS}
KK = {c: torch.tensor(cams[f"{c}_K"], dtype=torch.float32, device=dev) for c in CAMS}
mesh = np.load(f"{C.OUT}/iron_mesh.npz")
V = mesh["vertices"].astype(np.float32)
Fc = mesh["faces"].astype(np.int64)
face_rgb = mesh["face_srgb"].astype(np.float32)
poses = np.load(f"{C.OUT}/poses.npz")
T0 = torch.tensor(poses["T"], dtype=torch.float32, device=dev)


def load_frame(cam, f):
    img = cv2.imread(f"out/clip/frames/{cam}/{f:04d}.jpg")[..., ::-1].astype(np.float32) / 255
    m, h = C.load_masks(cam, f)
    occ = np.zeros_like(m)
    p = f"{C.OUT}/occluder/{cam}/{f:04d}.png"
    if os.path.exists(p):
        occ = cv2.imread(p, 0) > 0
    return img, m, (h | occ) & ~m


data = {}
for cam in CAMS:
    for f in range(NF):
        img, m, ig = load_frame(cam, f)
        if m.sum() < 50:
            continue
        ys, xs = np.nonzero(m)
        pad = 60
        bb = (max(0, xs.min() - pad), max(0, ys.min() - pad), min(1920, xs.max() + pad), min(1080, ys.max() + pad))
        x0, y0, x1, y1 = bb
        er = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        di = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        sl = (slice(y0, y1), slice(x0, x1))
        data[cam, f] = dict(
            bb=bb,
            img=torch.tensor(img[sl], device=dev).permute(2, 0, 1),
            inside=torch.tensor(er[sl] & ~ig[sl], device=dev, dtype=torch.float32),
            outside=torch.tensor(~di[sl] & ~ig[sl], device=dev, dtype=torch.float32),
            valid=torch.tensor(~ig[sl], device=dev, dtype=torch.float32))
train_keys = [k for k in data if not heldout(k[1])]
print("views: train", len(train_keys), "held-out", len(data) - len(train_keys),
      {c: sum(1 for k in train_keys if k[0] == c) for c in CAMS}, flush=True)


# --- model -----------------------------------------------------------------------------
def quat_to_mat(q):
    w, x, y, z = Fn.normalize(q, dim=-1).unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


class IronGS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        rng = np.random.default_rng(0)
        v = V[Fc]
        area = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
        k = np.clip(np.round(area / area.mean() * 12), 1, 48).astype(int)
        tri = np.repeat(np.arange(len(Fc)), k)
        r1, r2 = rng.random(len(tri)), rng.random(len(tri))
        s = np.sqrt(r1)
        bary = np.stack([1 - s, s * (1 - r2), s * r2], 1).astype(np.float32)
        e1, e2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
        n = np.cross(e1, e2); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        t = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12)
        b = np.cross(n, t)
        R = np.stack([t, b, n], -1).astype(np.float32)
        center = np.einsum("nk,nkc->nc", bary, v[tri]).astype(np.float32)
        self.register_buffer("center", torch.tensor(center))
        self.register_buffer("Rtri", torch.tensor(R[tri]))
        N = len(tri)
        st = np.sqrt(area[tri] / k[tri]).clip(1e-3, 0.012) * 0.8
        P = torch.nn.Parameter
        self.offset = P(torch.zeros(N, 3))
        self.quat = P(torch.tensor([1.0, 0, 0, 0]).repeat(N, 1))
        self.log_scale = P(torch.tensor(np.log(np.stack([st, st, np.full_like(st, 1e-3)], 1)), dtype=torch.float32))
        self.opacity = P(torch.logit(torch.full((N,), 0.9)))
        self.sh0 = P(torch.tensor((face_rgb[tri] - 0.5) / C0, dtype=torch.float32)[:, None])
        self.shN = P(torch.zeros(N, 3, 3))
        self.dpose = P(torch.zeros(NF, 4))          # dx, dy, dz (m), dyaw (rad)
        self.ccM = P(torch.eye(3).repeat(len(CAMS), 1, 1))
        self.ccB = P(torch.zeros(len(CAMS), 3))
        self.N = N

    def pose(self, f):
        d = self.dpose[f]
        if heldout(f):   # no photometric signal: interpolate the neighbours' refinement
            d = 0.5 * (self.dpose[max(f - 1, 0)] + self.dpose[min(f + 1, NF - 1)]).detach()
        c, s = torch.cos(d[3]), torch.sin(d[3])
        Rz = torch.stack([torch.stack([c, -s, 0 * c]), torch.stack([s, c, 0 * c]),
                          torch.tensor([0.0, 0, 1], device=dev)])
        R = Rz @ T0[f, :3, :3]
        t = T0[f, :3, 3] + d[:3]
        return R, t

    def scales(self):
        s = torch.exp(self.log_scale)
        sxy = s[:, :2].clamp(3e-4, 8e-3)
        sxy = torch.maximum(sxy, sxy.max(1, keepdim=True).values / 4)      # anisotropy <= 4:1
        sz = torch.minimum(s[:, 2:].clamp(min=2e-4), sxy.min(1, keepdim=True).values)
        return torch.cat([sxy, sz], 1)

    def gaussians(self, f):
        R, t = self.pose(f)
        loc = self.center + torch.einsum("nij,nj->ni", self.Rtri, self.offset)
        means = loc @ R.T + t
        rot = R[None] @ self.Rtri @ quat_to_mat(self.quat)
        S = self.scales()
        Mx = rot * S[:, None, :]
        return means, Mx @ Mx.transpose(1, 2), torch.sigmoid(self.opacity), torch.cat([self.sh0, self.shN], 1)


def render(model, cam, f, bb=None, mode="RGB"):
    x0, y0, x1, y1 = bb if bb is not None else (0, 0, 1920, 1080)
    K = KK[cam].clone(); K[0, 2] -= x0; K[1, 2] -= y0
    means, cov, op, sh = model.gaussians(f)
    out, alpha, _ = rasterization(means, None, None, op, sh, VM[cam][f][None], K[None], x1 - x0, y1 - y0,
                                  covars=cov, sh_degree=1, packed=False, near_plane=0.02, render_mode=mode,
                                  rasterize_mode=RMODE)
    ci = CAMS.index(cam)
    rgb = out[0, ..., :3]
    a = alpha[0]
    rgb = rgb @ model.ccM[ci].T + model.ccB[ci] * a   # premultiplied colour affine
    return rgb, a, (out[0, ..., 3:4] if mode == "RGB+ED" else None)


def ssim_map(a, b):
    g = torch.exp(-((torch.arange(11, device=a.device) - 5) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None]
    w = g.expand(3, 1, 11, 11)
    blur = lambda x: Fn.conv2d(x, w, padding=5, groups=3)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    s_a, s_b, s_ab = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_a + s_b + c2))


model = IronGS().to(dev)
print("gaussians", model.N, flush=True)
opt = torch.optim.Adam([
    dict(params=[model.sh0], lr=4e-3), dict(params=[model.shN], lr=2e-4),
    dict(params=[model.opacity], lr=3e-2), dict(params=[model.log_scale], lr=4e-3),
    dict(params=[model.quat], lr=1e-3), dict(params=[model.offset], lr=5e-5),
    dict(params=[model.dpose], lr=5e-5), dict(params=[model.ccM, model.ccB], lr=5e-4)])
rng = np.random.default_rng(1)
for it in range(STEPS + 1):
    if it == STEPS // 3:   # start pose refinement once colours have settled
        pass
    cam, f = train_keys[rng.integers(len(train_keys))]
    d = data[cam, f]
    rgb, a, _ = render(model, cam, f, d["bb"])
    rgb, a = rgb.permute(2, 0, 1), a.permute(2, 0, 1)
    gt = d["img"]
    comp = rgb + (1 - a) * gt
    ins, out_, val = d["inside"], d["outside"], d["valid"]
    l1 = ((comp - gt).abs().mean(0) * ins).sum() / ins.sum().clamp(min=1)
    ss = 1 - (ssim_map(comp[None], gt[None])[0].mean(0) * ins).sum() / ins.sum().clamp(min=1)
    ac = a[0].clamp(1e-4, 1 - 1e-4)
    la = (-(torch.log(ac) * ins).sum() / ins.sum().clamp(min=1) - (torch.log(1 - ac) * out_).sum() / out_.sum().clamp(min=1))
    off = model.offset.norm(dim=-1)
    reg = 10 * Fn.relu(off - 0.004).mean() + 1e-3 * (model.dpose[:, :3] / 0.01).pow(2).mean() + 1e-3 * (model.dpose[:, 3] / 0.03).pow(2).mean() \
        + 1e-2 * ((model.dpose[1:] - model.dpose[:-1]) / torch.tensor([0.003, 0.003, 0.003, 0.01], device=dev)).pow(2).mean()
    wa = 0.0 if cam == "exocam2" else 0.1
    loss = 0.8 * l1 + 0.2 * ss + wa * la + reg
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if it < STEPS // 3:
        model.dpose.grad = None
    opt.step()
    if it % 1000 == 0:
        print(f"it {it} loss {loss.item():.4f} l1 {l1.item():.4f} ssim {1 - ss.item():.3f} alpha {la.item():.3f} "
              f"|dpose| {model.dpose[:, :3].abs().max().item() * 100:.2f}cm {np.degrees(model.dpose[:, 3].abs().max().item()):.1f}deg",
              f"mem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB", flush=True)
torch.save(model.state_dict(), f"{C.OUT}/gs.pt")

# --- evaluation, outputs ----------------------------------------------------------------
lp = lpips.LPIPS(net="alex", verbose=False).to(dev)
res = {c: {"train": {"iou": [], "lpips": [], "psnr": []}, "heldout": {"iou": [], "lpips": [], "psnr": []}} for c in CAMS}
prev_rows = []
for cam in CAMS:
    os.makedirs(f"{C.OUT}/render/{cam}", exist_ok=True)
    os.makedirs(f"{C.OUT}/depth/{cam}", exist_ok=True)
    for f in range(NF):
        with torch.no_grad():
            rgb, a, dep = render(model, cam, f, mode="RGB+ED")
        a_np = a[..., 0].clamp(0, 1).cpu().numpy()
        pm = torch.minimum(rgb.clamp(0, 1), a.clamp(0, 1)).cpu().numpy()
        straight = np.where(a_np[..., None] > 1e-4, pm / np.maximum(a_np[..., None], 1e-4), 0).clip(0, 1)
        rgba = np.dstack([straight[..., ::-1], a_np[..., None]])
        cv2.imwrite(f"{C.OUT}/render/{cam}/{f:04d}.png", (rgba * 255 + 0.5).astype(np.uint8))
        dz = dep[..., 0].cpu().numpy()
        dz = np.where(a_np > 0.05, dz, np.inf).astype(np.float16)
        np.save(f"{C.OUT}/depth/{cam}/{f:04d}.npy", dz)
        img, m, ig = load_frame(cam, f)
        if m.sum() < 50:
            continue
        split = "heldout" if heldout(f) else "train"
        iou = C.iou(a_np > 0.5, m, ig)
        ys, xs = np.nonzero(m)
        x0, y0, x1, y1 = max(0, xs.min() - 16), max(0, ys.min() - 16), min(1920, xs.max() + 16), min(1080, ys.max() + 16)
        mc = m[y0:y1, x0:x1, None].astype(np.float32)
        g_ = img[y0:y1, x0:x1] * mc
        p_ = pm[y0:y1, x0:x1] * mc
        mse = ((g_ - p_) ** 2).sum() / max(mc.sum() * 3, 1)
        with torch.no_grad():
            L = lp(torch.tensor(p_, device=dev).permute(2, 0, 1)[None] * 2 - 1,
                   torch.tensor(g_, device=dev).permute(2, 0, 1)[None] * 2 - 1).item()
        for k_, v_ in (("iou", iou), ("lpips", L), ("psnr", -10 * np.log10(max(mse, 1e-10)))):
            res[cam][split][k_].append(float(v_))
        if f == 30:   # preview row: [ORIGINAL crop + traced outline | OURS], tight crop on the iron
            both = m | (a_np > 0.5)
            ys, xs = np.nonzero(both)
            pad = 20
            x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
            x1, y1 = min(1920, xs.max() + pad), min(1080, ys.max() + pad)
            A = (img[..., ::-1] * 255).astype(np.uint8).copy()
            B = ((pm + (1 - a_np[..., None]) * 0.5)[..., ::-1] * 255).astype(np.uint8)
            A, B = A[y0:y1, x0:x1], B[y0:y1, x0:x1]
            sc = 360 / A.shape[0]
            A, B = cv2.resize(A, None, fx=sc, fy=sc), cv2.resize(B, None, fx=sc, fy=sc)
            for P_, txt in ((A, f"ORIGINAL {cam} f30 (held-out)"), (B, f"OURS IoU {iou:.3f} LPIPS {L:.3f}")):
                cv2.rectangle(P_, (0, 0), (P_.shape[1], 30), (0, 0, 0), -1)
                cv2.putText(P_, txt, (6, 21), 0, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            prev_rows.append(np.hstack([A, np.full((A.shape[0], 8, 3), 255, np.uint8), B]))
W_ = max(r.shape[1] for r in prev_rows)
prev_rows = [np.pad(r, ((0, 8), (0, W_ - r.shape[1]), (0, 0)), constant_values=255) for r in prev_rows]
cv2.imwrite(f"{C.OUT}/preview_gs.jpg", np.vstack(prev_rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
summary = {c: {s: {k: (float(np.mean(v)) if v else None) for k, v in d.items()} | {"n": len(d["iou"])}
               for s, d in res[c].items()} for c in CAMS}
summary["pose_refinement"] = dict(max_cm=float(model.dpose[:, :3].abs().max().item() * 100),
                                  max_deg=float(np.degrees(model.dpose[:, 3].abs().max().item())))
summary["n_gaussians"] = model.N
# refined poses: what the RGBA layer was rendered with
Tr = np.stack([np.eye(4) for _ in range(NF)])
with torch.no_grad():
    for f in range(NF):
        R, t = model.pose(f)
        Tr[f, :3, :3], Tr[f, :3, 3] = R.cpu().numpy(), t.cpu().numpy()
np.savez(f"{C.OUT}/poses_refined.npz", T=Tr)
json.dump(summary, open(f"{C.OUT}/gs_metrics.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
