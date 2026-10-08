"""Stage 08b: a sharper room splat model for the head camera.

Differences from stage 08 (see out/research/blur_egocentric.md):
  - head-camera pose corrections learn from step 0 (lr 1e-4, L2 1e-6) and the first
    --half-res-steps steps run at half resolution so poses settle first (BARF-style);
  - tripod poses frozen (at stage 08's learned correction); tripods sampled every 3rd
    frame as anchors (out/room_sharp/data_tripod3.npz from 08b_data_tripods.py);
  - MCMCStrategy (cap 2M) with opacity/scale regularisers, initialised from
    out/room/splats.pt (and its colour/pose/exposure corrections);
  - the blurriest 25% of head-camera frames (Laplacian variance on room pixels) are
    dropped, except the clip frames, which are all kept and oversampled;
  - rasterize_mode="antialiased".

    uv run scripts/08b_train_room_sharp.py [--steps 60000]
Writes out/room_sharp/splats.pt (+ splats_<step>.pt milestones) and progress_<step>.jpg.
"""

import argparse
import importlib.util
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import MCMCStrategy, rasterization

_spec = importlib.util.spec_from_file_location("t08", Path(__file__).with_name("08_train_room.py"))
t08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t08)

ROOM, OUT, CLIP = Path("out/room"), Path("out/room_sharp"), Path("out/clip")
SH_DEGREE = 3
MODE = "antialiased"
STOP = OUT / "STOP"


def npz_mmap(path: Path, key: str) -> np.ndarray:
    """Memory-map one array of an uncompressed .npz (no 10 GB up-front read)."""
    import zipfile

    with zipfile.ZipFile(path) as z:
        info = z.getinfo(key + ".npy")
        assert info.compress_type == zipfile.ZIP_STORED
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        h = f.read(30)
        start = info.header_offset + 30 + int.from_bytes(h[26:28], "little") + int.from_bytes(
            h[28:30], "little"
        )
        f.seek(start)
        ver = np.lib.format.read_magic(f)
        rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = rd(f)
        off = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=off)


class Data:
    """Images stay in host RAM (15 GB); a batch is copied to the GPU per step."""

    def __init__(self, paths: list[Path]):
        self.src, meta = [], {k: [] for k in ("K", "T_world_cam", "cam_id", "frame_idx")}
        self.loc = []
        for s, p in enumerate(paths):
            if not p.exists():
                print(f"(skipping missing {p})")
                continue
            D = np.load(p)
            imgs, msks = npz_mmap(p, "images"), npz_mmap(p, "masks")
            print(f"loaded {p}: {imgs.shape}", flush=True)
            self.src.append((imgs, msks))
            for k in meta:
                meta[k].append(D[k])
            self.loc += [(len(self.src) - 1, i) for i in range(len(imgs))]
            self.cam_names = [str(c) for c in D["cam_names"]]
            self.n_old = self.n_old if s else len(imgs)
        for k, v in meta.items():
            setattr(self, k, np.concatenate(v))
        self.H, self.W = self.src[0][0].shape[1:3]

    def __len__(self):
        return len(self.loc)

    def image(self, i):
        s, j = self.loc[i]
        return self.src[s][0][j], self.src[s][1][j]

    def batch(self, idx, dev):
        ims, ms = zip(*(self.image(int(i)) for i in idx), strict=True)
        im = torch.from_numpy(np.stack(ims)).to(dev)
        m = torch.from_numpy(np.stack(ms)).to(dev)
        return im.float() / 255.0, m.float()


def room_sharpness(img: np.ndarray, mask: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32)
    m = cv2.erode(cv2.resize(mask, g.shape[::-1], interpolation=cv2.INTER_NEAREST),
                  np.ones((9, 9), np.uint8)) > 0  # fmt: skip
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return float(lap[m].var()) if m.sum() > 1000 else 0.0


def clip_slots(data: Data) -> np.ndarray:
    """Training index of every clip frame, matched by exact recorded pose (not by index)."""
    C = np.load(CLIP / "cameras.npz")
    head = data.cam_names.index("headcam")
    hi = np.flatnonzero(data.cam_id == head)
    slots = []
    for f, T in enumerate(C["headcam_T"]):
        cand = hi[np.all(np.abs(data.T_world_cam[hi] - T) < 1e-9, axis=(1, 2))]
        assert len(cand), f"clip frame {f} is not a training image"
        if len(cand) > 1:  # identical poses: prefer the same stamp index
            cand = cand[np.argsort(np.abs(data.frame_idx[cand] - f))]
        slots.append(int(cand[0]))
    return np.array(slots)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--cap-max", type=int, default=2_000_000)
    ap.add_argument("--half-res-steps", type=int, default=5000)
    ap.add_argument("--drop-blur", type=float, default=0.25)
    ap.add_argument("--head-share", type=float, default=0.6, help="fraction of samples")
    ap.add_argument("--clip-boost", type=float, default=3.0)
    ap.add_argument("--init", default=str(ROOM / "splats.pt"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-extra", action="store_true")
    ap.add_argument("--no-blur-filter", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    np.random.seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    STOP.unlink(missing_ok=True)

    data = Data([ROOM / "data.npz"] + ([] if args.no_extra else [OUT / "data_tripod3.npz"]))
    n_img, H, W = len(data), data.H, data.W
    head = data.cam_names.index("headcam")
    is_head = data.cam_id == head
    clip = clip_slots(data)
    print(f"{n_img} images {W}x{H}; headcam {is_head.sum()}, tripod {(~is_head).sum()}; "
          f"clip slots {clip[:3]}..{clip[-1]}")  # fmt: skip

    # Motion blur: drop the blurriest head-camera frames (never clip frames).
    t0 = time.time()
    sharp = np.full(n_img, np.inf)
    for i in [] if args.no_blur_filter else np.flatnonzero(is_head):
        sharp[i] = room_sharpness(*data.image(i))
    cand = np.setdiff1d(np.flatnonzero(is_head), clip)
    thr = np.quantile(sharp[np.flatnonzero(is_head)], args.drop_blur) if not args.no_blur_filter else -1
    dropped = cand[sharp[cand] < thr]
    keep = np.ones(n_img, bool)
    keep[dropped] = False
    print(f"sharpness: dropped {len(dropped)} blurry headcam frames (lapvar < {thr:.1f}; "
          f"median {np.median(sharp[is_head]):.1f}) in {time.time() - t0:.0f}s")  # fmt: skip
    np.save(OUT / "headcam_sharpness.npy", np.stack([np.arange(n_img), sharp, keep]))

    # Sampling weights: head share split across kept head frames, clip frames boosted.
    w = np.zeros(n_img)
    hk = np.flatnonzero(is_head & keep)
    w[hk] = 1.0
    w[clip] *= args.clip_boost
    w[hk] *= args.head_share / w[hk].sum()
    tk = np.flatnonzero(~is_head)
    w[tk] = (1 - args.head_share) / len(tk)
    weights = torch.from_numpy(w).float().to(dev)

    Ks = torch.from_numpy(data.K).float().to(dev)
    viewmats = torch.from_numpy(np.stack([t08.ros_to_cv(T) for T in data.T_world_cam]))
    viewmats = viewmats.float().to(dev)
    cam_id = torch.from_numpy(data.cam_id).long().to(dev)
    is_head_t = torch.from_numpy(is_head).to(dev)

    # ---- init from the stage-08 model ----
    ck = torch.load(args.init, map_location=dev, weights_only=False)
    p0 = ck["params"]
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in p0.items()})
    centers = data.T_world_cam[:, :3, 3]
    lo, hi = centers.min(0) - 2.0, centers.max(0) + 2.0
    lo[2], hi[2] = centers[:, 2].min() - 1.6, centers[:, 2].max() + 1.5
    scene_scale = float(np.linalg.norm(hi - lo) / 2)
    print(f"init {len(params['means']):,} gaussians from {args.init}; scene_scale {scene_scale:.2f}")

    # Corrections: one pose slot per image; tripod rows frozen at stage 08's per-camera value.
    corr = t08.Corrections(n_img, len(data.cam_names)).to(dev)
    oc = ck["corr"]
    n_old = data.n_old
    with torch.no_grad():
        corr.color.copy_(oc["color"])
        corr.bias.copy_(oc["bias"])
        corr.rot[:n_old] = oc["rot"][:n_old]
        corr.trans[:n_old] = oc["trans"][:n_old]
        corr.log_gain[:n_old] = oc["log_gain"][:n_old]
        for c in range(len(data.cam_names)):
            if c == head:
                continue
            rows = torch.from_numpy(np.flatnonzero(data.cam_id == c)).to(dev)
            corr.rot[rows] = oc["rot"][c]  # stage 08 used slot = cam id for tripods
            corr.trans[rows] = oc["trans"][c]
            corr.log_gain[rows[rows >= n_old]] = oc["log_gain"][:n_old][
                torch.from_numpy(data.cam_id[:n_old] == c).to(dev)
            ].mean()
    freeze = (~is_head_t).float()[:, None]
    corr.rot.register_hook(lambda g: g * (1 - freeze))
    corr.trans.register_hook(lambda g: g * (1 - freeze))

    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}  # fmt: skip
    optimizers = {k: torch.optim.Adam([params[k]], lr=lr, eps=1e-15) for k, lr in lrs.items()}
    means_sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / args.steps)
    )
    pose_opt = torch.optim.Adam([corr.rot, corr.trans], lr=1e-4)
    pose_sched = torch.optim.lr_scheduler.ExponentialLR(pose_opt, gamma=0.1 ** (1.0 / args.steps))
    app_opt = torch.optim.Adam([corr.color, corr.bias, corr.log_gain], lr=1e-3)
    strategy = MCMCStrategy(
        cap_max=args.cap_max, noise_lr=5e5, refine_every=100,
        refine_stop_iter=int(args.steps * 0.75), refine_start_iter=min(500, args.steps // 10), min_opacity=0.005, verbose=False,
    )  # fmt: skip
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()

    eval_slots = clip[[0, 30, 59]]
    t0 = time.time()
    for step in range(args.steps):
        idx = torch.multinomial(weights, args.batch, replacement=True)
        gt, m = data.batch(idx.cpu().numpy(), dev)
        K = Ks[idx]
        w_, h_ = W, H
        if step < args.half_res_steps:
            w_, h_ = W // 2, H // 2
            K = K.clone()
            K[:, :2] *= 0.5
            gt = F.interpolate(gt.permute(0, 3, 1, 2), (h_, w_), mode="area").permute(0, 2, 3, 1)
            m = (F.interpolate(m[:, None], (h_, w_), mode="area")[:, 0] > 0.99).float()
        vm = corr.viewmat(viewmats[idx], idx)
        sh = torch.cat([params["sh0"], params["shN"]], 1)
        render, _, info = rasterization(
            params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), sh, vm, K, w_, h_,
            sh_degree=SH_DEGREE, packed=False, rasterize_mode=MODE,
        )  # fmt: skip
        render = corr.colorize(render, cam_id[idx], idx)
        m = m[..., None]
        comp = render * m + gt * (1 - m)
        l1 = (render - gt).abs().mul(m).sum() / (m.sum() * 3).clamp_min(1)
        loss = 0.8 * l1 + 0.2 * (1 - t08.ssim(comp.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)))
        hrow = is_head_t[idx]
        loss = loss + 1e-6 * ((corr.trans[idx][hrow] ** 2).sum() + (corr.rot[idx][hrow] ** 2).sum())
        loss = loss + 0.01 * torch.sigmoid(params["opacities"]).mean()
        loss = loss + 0.01 * torch.exp(params["scales"]).mean()
        loss.backward()
        for opt in [*optimizers.values(), pose_opt, app_opt]:
            opt.step()
            opt.zero_grad(set_to_none=True)
        lr_now = means_sched.get_last_lr()[0]
        means_sched.step()
        pose_sched.step()
        strategy.step_post_backward(params, optimizers, state, step, info, lr=lr_now)

        if step % 1000 == 0 or step == args.steps - 1:
            with torch.no_grad():
                mse = ((render - gt) ** 2).mul(m).sum() / (m.sum() * 3).clamp_min(1)
                hr = corr.rot[is_head_t].norm(dim=1).median() * 57.3
                ht = corr.trans[is_head_t].norm(dim=1).median() * 100
            print(
                f"step {step:6d}  loss {loss.item():.4f}  batch PSNR {-10 * math.log10(mse.item()):5.2f}"
                f"  gaussians {len(params['means']):,}  pose med {hr:.2f}deg {ht:.2f}cm"
                f"  mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G  {time.time() - t0:5.0f}s",
                flush=True,
            )
        if step % 2000 == 0 or step == args.steps - 1:
            with torch.no_grad():
                ps = []
                sh = torch.cat([params["sh0"], params["shN"]], 1)
                for s in eval_slots:
                    s_t = torch.tensor([int(s)], device=dev)
                    g, mm = data.batch([int(s)], dev)
                    r, _, _ = rasterization(
                        params["means"], F.normalize(params["quats"], dim=-1),
                        torch.exp(params["scales"]), torch.sigmoid(params["opacities"]), sh,
                        corr.viewmat(viewmats[s_t], s_t), Ks[s_t], W, H, sh_degree=SH_DEGREE,
                        rasterize_mode=MODE,
                    )  # fmt: skip
                    r = corr.colorize(r, cam_id[s_t], s_t).clamp(0, 1)
                    mm = mm[..., None]
                    e = ((r - g) ** 2).mul(mm).sum() / (mm.sum() * 3)
                    ps.append(-10 * math.log10(e.item()))
                print(f"  eval clip frames 0/30/59 masked PSNR: "
                      + " ".join(f"{v:.2f}" for v in ps), flush=True)  # fmt: skip
                vis = np.hstack([g[0].cpu().numpy(), r[0].cpu().numpy()])
                cv2.imwrite(str(OUT / f"progress_{step:06d}.jpg"),
                            (vis[..., ::-1] * 255).astype(np.uint8))  # fmt: skip
        if (step > 0 and step % 10000 == 0) or step == args.steps - 1 or STOP.exists():
            ckpt = {
                "params": {k: v.detach() for k, v in params.items()}, "corr": corr.state_dict(),
                "sh_degree": SH_DEGREE, "cam_names": data.cam_names, "rasterize_mode": MODE,
                "clip_slots": clip, "cam_id": data.cam_id, "frame_idx": data.frame_idx,
                "data_files": [str(ROOM / "data.npz"), str(OUT / "data_tripod3.npz")],
                "step": step,
            }  # fmt: skip
            name = "splats.pt" if step == args.steps - 1 else f"splats_{step:06d}.pt"
            torch.save(ckpt, OUT / name)
            print(f"saved {OUT / name}", flush=True)
            if STOP.exists():
                torch.save(ckpt, OUT / "splats.pt")
                print("STOP file: saved splats.pt and exiting", flush=True)
                return


if __name__ == "__main__":
    main()
