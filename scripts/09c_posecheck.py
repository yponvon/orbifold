"""Stage 09c (pose check): is the head-camera pose correction converged?

For a few clip frames: freeze the scene of a 08b checkpoint and fit that frame's 6-DoF pose
+ gain (Adam lr 1e-3, room pixels, full res). Reports how far the pose moves and the masked
PSNR gain. Large moves (> ~0.1 deg / 3 mm) with a > 1 dB gain mean training poses lag.

    uv run scripts/09c_posecheck.py [--ckpt out/room_sharp/splats.pt] [--frames 0 30 59]
"""

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t08 = _load("t08", "08_train_room.py")
t08b = _load("t08b", "08b_train_room_sharp.py")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/room_sharp/splats.pt")
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 15, 30, 45, 59])
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    p = ck["params"]
    corr = t08.Corrections(len(ck["cam_id"]), len(ck["cam_names"])).to(dev)
    corr.load_state_dict(ck["corr"])
    head = ck["cam_names"].index("headcam")
    data = t08b.Data([Path(f) for f in ck["data_files"]])
    sh = torch.cat([p["sh0"], p["shN"]], 1)
    geo = (p["means"], F.normalize(p["quats"], dim=-1), torch.exp(p["scales"]),
           torch.sigmoid(p["opacities"]), sh)  # fmt: skip
    H, W = data.H, data.W
    ci = torch.tensor([head], device=dev)
    for f in args.frames:
        s = int(ck["clip_slots"][f])
        st = torch.tensor([s], device=dev)
        gt, m = data.batch([s], dev)
        m = m[..., None]
        vm0 = torch.from_numpy(t08.ros_to_cv(data.T_world_cam[s])).float().to(dev)[None]
        K = torch.from_numpy(data.K[s]).float().to(dev)[None]
        with torch.no_grad():
            vm = corr.viewmat(vm0, st)
        dr = torch.zeros(1, 3, device=dev, requires_grad=True)
        dt = torch.zeros(1, 3, device=dev, requires_grad=True)
        dg = torch.zeros(1, device=dev, requires_grad=True)
        opt = torch.optim.Adam([dr, dt, dg], lr=1e-3)
        psnr = []
        for it in range(args.steps + 1):
            D4 = torch.eye(4, device=dev)[None].clone()
            D4[:, :3, :3] = t08.axis_angle_to_matrix(dr)
            D4[:, :3, 3] = dt
            r, _, _ = rasterization(*geo, D4 @ vm, K, W, H, sh_degree=ck["sh_degree"],
                                    rasterize_mode=ck["rasterize_mode"])  # fmt: skip
            r = corr.colorize(r, ci, st) * torch.exp(dg)
            if it in (0, args.steps):
                with torch.no_grad():
                    e = ((r.clamp(0, 1) - gt) ** 2).mul(m).sum() / (m.sum() * 3)
                    psnr.append(-10 * math.log10(e.item()))
            if it == args.steps:
                break
            loss = (r - gt).abs().mul(m).sum() / (m.sum() * 3)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"frame {f:2d} slot {s:4d}: refine moves rot {dr.norm().item() * 57.3:.3f} deg, "
              f"trans {dt.norm().item() * 1000:.1f} mm, gain {dg.item():+.3f}; "
              f"masked PSNR {psnr[0]:.2f} -> {psnr[1]:.2f}  "
              f"(learned corr: {corr.rot[s].norm().item() * 57.3:.2f} deg, "
              f"{corr.trans[s].norm().item() * 100:.2f} cm)", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
