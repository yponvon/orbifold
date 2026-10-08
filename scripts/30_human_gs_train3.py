"""v3 training of the mesh-anchored human Gaussians: aimed at removing blur and shards.

Changes from v2 (scripts/30_human_gs_train.py), following out/research/blur_avatar.md:
  - learnable 6-DoF camera delta per headcam frame and per exo camera (L2 1e-3, lr 1e-4,
    switched on after --pose-start); untrained headcam frames interpolate their neighbours
  - temporal residual with --n-basis 30 (was 10)
  - hard anti-shard scale rules (HumanGS3.scales): s1 <= min(1 cm, 1.5 x edge), in-plane
    ratio <= 3, s3 >= max(1 mm, s1/8); opacity L1; prune opacity < 0.02
  - coverage term 0.02 and only where the mesh projects; outside-mask alpha penalty 1.0,
    3 px dilation, full frame (except iron pixels and where the room is in front)
  - 6 Gaussians per skin triangle, 12 per garment triangle; absgrad densification
    (split into 2 on the same triangle, scale / 1.6) from 1k to 15k steps, cap 1.5M
  - headcam weight 1.5; 2 x 512^2 patches per rendered view (one centred on a hand),
    VGG-LPIPS 0.05 after 3k steps
  - rasterize_mode="antialiased"; blurred headcam frames (Laplacian variance < 0.5 x median)
    are dropped from training
Targets: original full-resolution frames; photometric loss only inside the traced person
mask minus the dilated iron mask. Every 10th frame is held out.

    UV_NO_SYNC=1 uv run scripts/30_human_gs_train3.py [--steps 20000]
Writes out/assets/human_gs/<--out> (default model_v3.pt), model_<tag>_<step>.pt and
preview_<tag>_<step>.jpg every --preview-every steps.
"""

import argparse
import importlib
import math
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
CAM_WEIGHT = {"headcam": 1.5, "exocam1": 1.0, "exocam2": 1.0}
HAND_PREFIX = ("wrist", "finger", "metacarpal")


def reindex(model, opt, keep: torch.Tensor, parents: torch.Tensor | None, rng):
    """Keep rows `keep` and append one child per entry of `parents` (copy of the parent with
    a fresh barycentric position on the same triangle and scale / 1.6). Adam state follows."""
    for name in hgs.GAUSS_KEYS:
        old = getattr(model, name)
        data = old.data[keep]
        if parents is not None and len(parents):
            child = old.data[parents].clone()
            if name == "log_scale":
                child -= math.log(1.6)
            data = torch.cat([data, child])
        newp = torch.nn.Parameter(data)
        group = next(g for g in opt.param_groups if g["name"] == name)
        st = opt.state.pop(old, None)
        group["params"][0] = newp
        if st:
            for k in ("exp_avg", "exp_avg_sq"):
                v = st[k][keep]
                if parents is not None and len(parents):
                    v = torch.cat([v, torch.zeros_like(st[k][parents])])
                st[k] = v
            opt.state[newp] = st
        setattr(model, name, newp)
    tri, bary = model.tri[keep], model.bary[keep]
    if parents is not None and len(parents):
        nb = torch.from_numpy(hgs.sample_bary(len(parents), rng)).to(bary.device)
        tri, bary = torch.cat([tri, model.tri[parents]]), torch.cat([bary, nb])
    model.tri, model.bary = tri, bary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--views", type=int, default=4, help="full-frame renders per step")
    ap.add_argument("--patch", type=int, default=512)
    ap.add_argument("--n-basis", type=int, default=30)
    ap.add_argument("--w-off", type=float, default=5.0)
    ap.add_argument("--w-res", type=float, default=100.0)
    ap.add_argument("--w-capopac", type=float, default=0.1)
    ap.add_argument("--w-cover", type=float, default=0.02)
    ap.add_argument("--w-outside", type=float, default=1.0)
    ap.add_argument("--outside-dilate", type=int, default=3)
    ap.add_argument("--w-opac", type=float, default=0.01)
    ap.add_argument("--w-lpips", type=float, default=0.05)
    ap.add_argument("--lpips-start", type=int, default=3000)
    ap.add_argument("--pose-start", type=int, default=500)
    ap.add_argument("--densify-from", type=int, default=1000)
    ap.add_argument("--densify-until", type=int, default=15000)
    ap.add_argument("--densify-every", type=int, default=500)
    ap.add_argument("--grad-thr", type=float, default=8e-4)
    ap.add_argument("--max-gauss", type=int, default=1_500_000)
    ap.add_argument("--prune-every", type=int, default=1000)
    ap.add_argument("--prune-opac", type=float, default=0.02)
    ap.add_argument("--preview-every", type=int, default=5000)
    ap.add_argument("--tag", default="v3")
    ap.add_argument("--raster", default="antialiased", choices=["antialiased", "classic"])
    ap.add_argument("--head-weight", type=float, default=1.5)
    ap.add_argument("--out", default="model_v3.pt", help="file name under out/assets/human_gs")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    t_start = time.time()

    cams = hgs.load_cameras()
    verts, faces, part = hgs.load_mesh()
    model = hgs.HumanGS3(verts, faces, part, hgs.load_trim(), n_basis=args.n_basis).to(dev)
    model.pose = hgs.CamPose().to(dev)
    model.raster_mode = args.raster
    CAM_WEIGHT["headcam"] = args.head_weight
    cc = hgs.CamColor(len(hgs.CAMS), hgs.N_FRAMES).to(dev)
    print(f"{model.tri.shape[0]:,} gaussians on {len(faces):,} triangles", flush=True)

    B = np.load("out/body/fit.npz")
    lab = B["bone_labels"][B["vertex_bone"]]
    is_hand = np.array([str(x).startswith(HAND_PREFIX) for x in lab])
    hands = [np.flatnonzero(is_hand & np.char.endswith(lab.astype(str), s)) for s in (".L", ".R")]

    room = hgs.Room(dev)
    V = {}  # per training view: tensors on the GPU
    blur = {}
    for ci, cam in enumerate(hgs.CAMS):
        K, vms, (W, H) = cams[cam]
        Kt = torch.tensor(K, dtype=torch.float32, device=dev)
        for f in range(hgs.N_FRAMES):
            if hgs.is_heldout(f):
                continue
            im = cv2.imread(str(hgs.CLIP / "frames" / cam / f"{f:04d}.jpg"))[..., ::-1].copy()
            m = hgs.load_mask(cam, f, erode=hgs.ERODE_PX)
            full = hgs.load_mask(cam, f, iron=False)
            d = args.outside_dilate
            outside = ~(cv2.dilate(full.astype(np.uint8), hgs.disk(d)) > 0) & ~hgs.load_iron(cam, f, d)
            sil = hgs.mesh_silhouette(verts[f], faces, K, vms[f], (W, H)) > 0
            vmt = torch.tensor(vms[f], dtype=torch.float32, device=dev)
            centres = []
            for hv in hands:
                uv, z = hgs.project_cv(verts[f][hv], K, vms[f])
                ok = z > 0.05
                if ok.sum() > 10:
                    c = uv[ok].mean(0)
                    if 0 <= c[0] < W and 0 <= c[1] < H:
                        centres.append(c)
            ys, xs = np.nonzero(m)
            if cam == "headcam":
                g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
                blur[f] = float(cv2.Laplacian(g, cv2.CV_32F)[m].var())
            V[ci, f] = {
                "img": torch.from_numpy(im).to(dev),
                "m": torch.from_numpy(m).to(dev),
                "out": torch.from_numpy(outside).to(dev),
                "cover": torch.from_numpy(m & sil).to(dev),
                "rdepth": room.depth(cam, f, vmt, Kt, W, H).half(),
                "hands": centres,
                "bbox": (xs.min(), ys.min(), xs.max(), ys.max()),
                "K": Kt, "vm": vmt, "W": W, "H": H,
            }  # fmt: skip
    del room
    torch.cuda.empty_cache()
    med = np.median(list(blur.values()))
    dropped = [f for f, v in blur.items() if v < 0.5 * med]
    for f in dropped:
        del V[0, f]
    for f in blur:
        if f not in dropped:
            model.pose.trained[f] = True
    train = list(V.keys())
    print(f"{len(train)} training views (dropped blurred headcam frames {dropped}); "
          f"loaded in {time.time() - t_start:.0f}s, mem {torch.cuda.memory_allocated() / 2**30:.1f}GB", flush=True)  # fmt: skip

    lr = {"offset": 3e-4, "resid": 3e-4, "quat": 1e-3, "log_scale": 5e-3,
          "opacity": 3e-2, "sh0": 5e-3, "shN": 5e-4}  # fmt: skip
    decay = {"offset": 0.01, "resid": 0.01, "quat": 0.1, "log_scale": 0.1,
             "opacity": 1.0, "sh0": 1.0, "shN": 1.0, "global_t": 0.01}  # fmt: skip
    groups = [{"params": [getattr(model, k)], "lr": v, "name": k} for k, v in lr.items()]
    groups.append({"params": [model.global_t], "lr": 1e-4, "name": "global_t"})
    opt = torch.optim.Adam(groups, eps=1e-15)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, [lambda s, d=decay[g["name"]]: d ** (s / args.steps) for g in groups]
    )
    opt_c = torch.optim.Adam(cc.parameters(), lr=1e-3)
    opt_p = torch.optim.Adam([model.pose.rot, model.pose.trans], lr=1e-4)
    lp_vgg = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    lp_alex = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in [*lp_vgg.parameters(), *lp_alex.parameters()]:
        p.requires_grad_(False)
    eye = torch.eye(3, device=dev)
    N = model.tri.shape[0]
    acc, cnt = torch.zeros(N, device=dev), torch.zeros(N, device=dev)
    ps = args.patch

    def save(path, train_s):
        sd = {k: v for k, v in model.state_dict().items() if k != "verts" and not k.startswith("pose.")}
        torch.save(
            {"model": sd, "pose": model.pose.state_dict(), "camcolor": cc.state_dict(),
             "n_basis": args.n_basis, "args": vars(args), "train_seconds": train_s,
             "capped": True, "version": 3, "dropped_headcam": dropped, "raster": args.raster},
            path,
        )  # fmt: skip

    t0 = time.time()
    for step in range(args.steps):
        loss = 0.0
        preds, gts, ms, wts, stats = [], [], [], [], []
        infos = []
        for j in rng.choice(len(train), args.views, replace=False):
            ci, f = train[j]
            v = V[ci, f]
            W, H = v["W"], v["H"]
            vm = model.pose.viewmat(ci, f, v["vm"]) if step >= args.pose_start else v["vm"]
            out, alpha, info = hgs.render(
                model, f, vm, v["K"], W, H, sh_degree=0 if step < 1000 else 1,
                render_mode="RGB+ED", absgrad=True,
            )  # fmt: skip
            info["means2d"].retain_grad()
            infos.append((info, W, H))
            pred = cc(out[..., :3], alpha, ci, f)
            a = alpha[..., 0].clamp(1e-4, 1 - 1e-4)
            w = CAM_WEIGHT[hgs.CAMS[ci]]
            hidden = v["rdepth"].float() < out[..., 3].detach() - 0.05
            o = v["out"] & ~hidden
            l_out = (-torch.log(1 - a) * o).sum() / o.sum().clamp_min(1)
            cov = v["cover"]
            l_cov = (-torch.log(a) * cov).sum() / cov.sum().clamp_min(1)
            loss = loss + w * (args.w_outside * l_out + args.w_cover * l_cov) / args.views
            # Patches: one centred on a hand (if visible), one on a random person pixel.
            x0b, y0b, x1b, y1b = v["bbox"]
            for kind in ("hand", "rand"):
                if kind == "hand" and v["hands"]:
                    cx, cy = v["hands"][rng.integers(len(v["hands"]))] + rng.normal(0, 48, 2)
                else:
                    cx, cy = rng.uniform(x0b, x1b + 1), rng.uniform(y0b, y1b + 1)
                px = int(np.clip(cx - ps / 2, 0, W - ps))
                py = int(np.clip(cy - ps / 2, 0, H - ps))
                m = v["m"][py : py + ps, px : px + ps].float()[..., None]
                gts.append(v["img"][py : py + ps, px : px + ps].float() / 255 * m)
                preds.append(pred[py : py + ps, px : px + ps] * m)
                ms.append(m)
                wts.append(w)
            loss = loss + args.w_res * (model.residual_raw(f) ** 2).mean() / args.views
        P = torch.stack(preds).permute(0, 3, 1, 2)
        G = torch.stack(gts).permute(0, 3, 1, 2)
        M = torch.stack(ms).permute(0, 3, 1, 2)
        wt = torch.tensor(wts, device=dev)
        msum = M.flatten(1).sum(1).clamp_min(1)
        valid = (M.flatten(1).sum(1) > 0).float()
        l1 = (P - G).abs().flatten(1).sum(1) / (3 * msum)
        ss = (hgs.ssim_map(P, G).mean(1, keepdim=True) * M).flatten(1).sum(1) / msum
        per = 0.8 * l1 + 0.2 * (1 - ss)
        if step >= args.lpips_start and args.w_lpips > 0:
            per = per + args.w_lpips * lp_vgg(P * 2 - 1, G * 2 - 1).flatten()
        loss = loss + (per * wt * valid).sum() / valid.sum().clamp_min(1)
        with torch.no_grad():
            mse = ((P - G) ** 2).flatten(1).sum(1) / (3 * msum)
            psnr = (-10 * torch.log10(mse.clamp_min(1e-10)))[valid > 0].mean().item()

        over = F.relu(model.offset.norm(dim=-1) - hgs.OFF_CAP)
        loss = loss + args.w_capopac * (torch.sigmoid(model.opacity) * over).mean() / hgs.OFF_CAP
        loss = loss + args.w_off * (model.offset**2).mean()
        loss = loss + args.w_opac * torch.sigmoid(model.opacity).mean()
        loss = loss + 1e-2 * (((cc.color - eye) ** 2).sum() + (cc.bias**2).sum())
        if step >= args.pose_start:
            loss = loss + 1e-3 * ((model.pose.rot**2).sum() + (model.pose.trans**2).sum())
        loss.backward()
        with torch.no_grad():
            for info, W, H in infos:
                g = info["means2d"].absgrad[0].clone()
                g[:, 0] *= W / 2
                g[:, 1] *= H / 2
                vis = (info["radii"][0] > 0).all(-1)
                acc[vis] += g.norm(dim=-1)[vis]
                cnt[vis] += 1
        opt.step()
        opt_c.step()
        if step >= args.pose_start:
            opt_p.step()
        for o_ in (opt, opt_c, opt_p):
            o_.zero_grad(set_to_none=True)
        sched.step()

        # Densify (split on the same triangle) and prune.
        changed = False
        with torch.no_grad():
            if args.densify_from <= step < args.densify_until and step % args.densify_every == 0:
                avg = acc / cnt.clamp_min(1)
                smax = model.scales().max(-1).values
                sel = torch.nonzero((avg > args.grad_thr) & (smax > 0.002))[:, 0]
                room_left = (args.max_gauss - model.tri.shape[0]) // 1  # each split adds 1 net
                if len(sel) > room_left:
                    sel = sel[torch.argsort(avg[sel], descending=True)[: max(room_left, 0)]]
                if len(sel):
                    keep = torch.ones(model.tri.shape[0], dtype=torch.bool, device=dev)
                    keep[sel] = False
                    reindex(model, opt, torch.nonzero(keep)[:, 0], torch.cat([sel, sel]), rng)
                    changed = True
            if step > 0 and step % args.prune_every == 0:
                keep = torch.sigmoid(model.opacity) >= args.prune_opac
                if not keep.all():
                    reindex(model, opt, torch.nonzero(keep)[:, 0], None, rng)
                    changed = True
            if changed or (step % args.densify_every == 0):
                N = model.tri.shape[0]
                acc, cnt = torch.zeros(N, device=dev), torch.zeros(N, device=dev)

        if step % 250 == 0 or step == args.steps - 1:
            print(
                f"step {step:5d} loss {loss.item():.4f} patch-PSNR {psnr:5.2f} "
                f"gaussians {model.tri.shape[0]:,} pose|t| head {model.pose.trans[0].norm(dim=-1).max().item() * 100:.1f}cm "
                f"{time.time() - t0:5.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB",
                flush=True,
            )  # fmt: skip
        if step > 0 and (step % args.preview_every == 0 or step == args.steps - 1):
            tag = f"{args.tag}_{step + 1 if step == args.steps - 1 else step:05d}"
            lps = hgs.write_preview(model, cc, cams, tag, lp_alex, dev=dev)
            print("  preview f30 LPIPS " + " ".join(f"{k} {v:.3f}" for k, v in lps.items()), flush=True)
            if step != args.steps - 1:
                save(hgs.OUT / f"model_{tag}.pt", time.time() - t0)
            torch.cuda.empty_cache()

    train_s = time.time() - t0
    save(hgs.OUT / args.out, train_s)
    print(f"saved {hgs.OUT / args.out}; training {train_s:.0f}s", flush=True)


if __name__ == "__main__":
    main()
