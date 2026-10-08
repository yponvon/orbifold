"""Train mesh-anchored Gaussians of the person against the real footage (all 3 cameras).

Gaussians live on the triangles of body + tunic + trousers (barycentric anchor), with a
learnable static offset in the triangle frame, a time-smooth per-frame residual (DCT basis,
so held-out frames are interpolated), colour SH degree 1, opacity, scale and rotation.
Targets are the original full-resolution frames (out/clip/frames, never downscaled). The
photometric loss (L1 + SSIM) and an alpha->1 coverage term are computed only inside the
traced person mask (YOLO, minus the dilated iron mask, eroded by ERODE_PX). An alpha->0
penalty applies outside the dilated person mask, except where the iron is, and except where
the room splats (out/room/splats.pt) are in front of the person (e.g. the bed hiding the
legs in exocam1). Regularisers: capped static/per-frame offsets (tanh caps in the model)
with L2, opacity penalty on Gaussians pushing past the caps, scale <= SCALE_CAP and a
needle penalty (largest/second-largest scale > --max-aniso). Per-camera colour affine; ego
camera weighted 0.5. Every 10th frame (0, 10, ..., 50) is held
out. A preview_<step>.jpg (frame 30, held out) is written at each milestone.

    UV_NO_SYNC=1 uv run scripts/30_human_gs_train.py [--steps 10000]
Writes out/assets/human_gs/model.pt and preview_<step>.jpg.
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
hgs = importlib.import_module("30_human_gs_common")

CAM_WEIGHT = {"headcam": 0.5, "exocam1": 1.0, "exocam2": 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--w-off", type=float, default=5.0)
    ap.add_argument("--w-res", type=float, default=100.0)
    ap.add_argument("--w-cover", type=float, default=0.1)
    ap.add_argument("--w-outside", type=float, default=0.3)
    ap.add_argument("--outside-dilate", type=int, default=8)
    ap.add_argument("--w-aniso", type=float, default=0.1)
    ap.add_argument("--max-aniso", type=float, default=4.0)
    ap.add_argument("--w-capopac", type=float, default=0.1)
    ap.add_argument("--preview-every", type=int, default=2500)
    ap.add_argument("--tag", default="v2", help="preview_<tag>_<step>.jpg")
    ap.add_argument("--n-basis", type=int, default=10)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    t_start = time.time()

    cams = hgs.load_cameras()
    verts, faces, part = hgs.load_mesh()
    model = hgs.HumanGS(verts, faces, part, hgs.load_trim(), n_basis=args.n_basis).to(dev)
    cc = hgs.CamColor(len(hgs.CAMS), hgs.N_FRAMES).to(dev)
    N = model.tri.shape[0]
    print(f"{N:,} gaussians on {len(faces):,} triangles")

    # Pre-load images, masks and per-image person crops.
    room = hgs.Room(dev)
    imgs, masks, outs, rdepth, boxes = {}, {}, {}, {}, {}
    for ci, cam in enumerate(hgs.CAMS):
        K, vms, (W, H) = cams[cam]
        for f in range(hgs.N_FRAMES):
            im = cv2.imread(str(hgs.CLIP / "frames" / cam / f"{f:04d}.jpg"))[..., ::-1].copy()
            m = hgs.load_mask(cam, f, erode=hgs.ERODE_PX)
            full = hgs.load_mask(cam, f, iron=False)
            d = args.outside_dilate
            outside = ~(cv2.dilate(full.astype(np.uint8), hgs.disk(d)) > 0) & ~hgs.load_iron(cam, f, d)
            uv, z = hgs.project_cv(verts[f], K, vms[f])
            uv = uv[(z > 0.05) & (uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)]
            ys, xs = np.nonzero(m)
            xs = np.concatenate([xs, uv[:, 0]])
            ys = np.concatenate([ys, uv[:, 1]])
            pad = 96
            x0, x1 = int(max(xs.min() - pad, 0)), int(min(xs.max() + pad, W))
            y0, y1 = int(max(ys.min() - pad, 0)), int(min(ys.max() + pad, H))
            boxes[ci, f] = (x0, y0, x1, y1)
            imgs[ci, f] = torch.from_numpy(im[y0:y1, x0:x1]).to(dev)
            masks[ci, f] = torch.from_numpy(m[y0:y1, x0:x1]).to(dev)
            outs[ci, f] = torch.from_numpy(outside[y0:y1, x0:x1]).to(dev)
            Kc = hgs.crop_K(torch.tensor(K, dtype=torch.float32, device=dev), x0, y0)
            vmt = torch.tensor(vms[f], dtype=torch.float32, device=dev)
            rdepth[ci, f] = room.depth(cam, f, vmt, Kc, x1 - x0, y1 - y0).half()
    del room
    torch.cuda.empty_cache()
    Ks = {
        ci: torch.tensor(cams[c][0], dtype=torch.float32, device=dev)
        for ci, c in enumerate(hgs.CAMS)
    }
    VMs = {
        ci: torch.tensor(cams[c][1], dtype=torch.float32, device=dev)
        for ci, c in enumerate(hgs.CAMS)
    }
    train = [(ci, f) for ci in range(3) for f in range(hgs.N_FRAMES) if not hgs.is_heldout(f)]
    print(f"{len(train)} training images; loaded in {time.time() - t_start:.0f}s")

    lr = {"offset": 3e-4, "resid": 3e-4, "global_t": 1e-4, "quat": 1e-3, "log_scale": 5e-3,
          "opacity": 3e-2, "sh0": 5e-3, "shN": 5e-4}  # fmt: skip
    opt = torch.optim.Adam(
        [{"params": [getattr(model, k)], "lr": v, "name": k} for k, v in lr.items()], eps=1e-15
    )
    opt_c = torch.optim.Adam(cc.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.1 ** (s / args.steps))
    eye = torch.eye(3, device=dev)

    lp = lpips.LPIPS(net="alex", verbose=False).to(dev)
    t0 = time.time()
    for step in range(args.steps):
        loss = 0.0
        stats = []
        for j in torch.randint(0, len(train), (args.batch,)).tolist():
            ci, f = train[j]
            x0, y0, x1, y1 = boxes[ci, f]
            out, alpha, _ = hgs.render(
                model, f, VMs[ci][f], hgs.crop_K(Ks[ci], x0, y0), x1 - x0, y1 - y0,
                sh_degree=0 if step < 1000 else 1, render_mode="RGB+ED",
            )  # fmt: skip
            pred = cc(out[..., :3], alpha, ci, f)
            m = masks[ci, f].float()[..., None]
            msum = m.sum().clamp_min(1)
            gt = imgs[ci, f].float() / 255.0 * m
            pm = pred * m  # only pixels inside the traced mask are compared
            l1 = ((pm - gt).abs()).sum() / (msum * 3)
            smap = hgs.ssim_map(pm.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]).mean(1)
            ss = (smap[0] * m[..., 0]).sum() / msum
            a = alpha.clamp(1e-4, 1 - 1e-4)
            cover = (-torch.log(a) * m).sum() / msum
            w = CAM_WEIGHT[hgs.CAMS[ci]]
            term = 0.8 * l1 + 0.2 * (1 - ss) + args.w_cover * cover
            if args.w_outside > 0:
                # Outside the person, unless the room is in front of where we render.
                hidden = rdepth[ci, f].float() < out[..., 3].detach() - 0.05
                o = (outs[ci, f] & ~hidden).float()
                term = term + args.w_outside * (-torch.log(1 - a[..., 0]) * o).sum() / o.sum().clamp_min(1)
            loss = loss + w * term / args.batch
            res = model.residual_raw(f)
            loss = loss + args.w_res * (res**2).mean() / args.batch
            over = F.relu(res.norm(dim=-1) - hgs.RES_CAP)
            loss = loss + args.w_capopac * (torch.sigmoid(model.opacity) * over).mean() / hgs.RES_CAP / args.batch
            with torch.no_grad():
                mse = ((pm - gt) ** 2).sum() / (msum * 3)
                stats.append(-10 * torch.log10(mse).item())
        loss = loss + args.w_off * (model.offset**2).mean()
        over = F.relu(model.offset.norm(dim=-1) - hgs.OFF_CAP)
        loss = loss + args.w_capopac * (torch.sigmoid(model.opacity) * over).mean() / hgs.OFF_CAP
        s_sorted = torch.sort(model.scales(), dim=-1, descending=True).values
        loss = loss + args.w_aniso * F.relu(s_sorted[:, 0] / s_sorted[:, 1] - args.max_aniso).mean()
        loss = loss + 1e-2 * (((cc.color - eye) ** 2).sum() + (cc.bias**2).sum())
        loss.backward()
        opt.step()
        opt_c.step()
        opt.zero_grad(set_to_none=True)
        opt_c.zero_grad(set_to_none=True)
        sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            off = model.offset.detach().norm(dim=-1)
            print(
                f"step {step:5d} loss {loss.item():.4f} masked-PSNR {np.mean(stats):5.2f} "
                f"offset mean {off.mean() * 100:.2f}cm p99 {off.quantile(0.99) * 100:.2f}cm "
                f"{time.time() - t0:5.0f}s  mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB",
                flush=True,
            )  # fmt: skip
        if step > 0 and (step % args.preview_every == 0 or step == args.steps - 1):
            lps = hgs.write_preview(model, cc, cams, f"{args.tag}_{step + 1 if step == args.steps - 1 else step:05d}", lp, dev=dev)
            print("  preview f30 LPIPS " + " ".join(f"{k} {v:.3f}" for k, v in lps.items()), flush=True)
            torch.cuda.empty_cache()

    train_s = time.time() - t0
    torch.save(
        {"model": {k: v for k, v in model.state_dict().items() if k != "verts"},
         "camcolor": cc.state_dict(), "n_basis": args.n_basis, "args": vars(args),
         "train_seconds": train_s, "capped": True},
        hgs.OUT / "model.pt",
    )  # fmt: skip
    print(f"saved {hgs.OUT / 'model.pt'}; training {train_s:.0f}s")


if __name__ == "__main__":
    main()
