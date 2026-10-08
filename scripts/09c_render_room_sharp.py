"""Stage 09c: render the sharp room model (out/room_sharp/splats.pt) for every clip frame.

Head camera: each clip frame uses its own learned pose correction and exposure (clip
frames are training images, matched by exact recorded pose in 08b). Optionally a
post-hoc fit (--refine N Adam steps, lr 1e-3) of that frame's pose and gain with the
scene frozen, on room pixels. Tripods: their frozen correction and mean exposure.
rasterize_mode="antialiased", as in training.

    uv run scripts/09c_render_room_sharp.py [--refine 0]
Writes out/room_render_sharp/<cam>/0000.png ...
"""

import argparse
import importlib.util
from pathlib import Path

import cv2
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
CLIP, OUT = Path("out/clip"), Path("out/room_render_sharp")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/room_sharp/splats.pt")
    ap.add_argument("--refine", type=int, default=0, help="post-hoc pose/gain steps per frame")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--cams", nargs="+", default=None, help="subset of cameras (default all)")
    args = ap.parse_args()
    out = Path(args.out)
    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    p = {k: v for k, v in ck["params"].items()}
    cam_names, cam_id, clip = ck["cam_names"], ck["cam_id"], ck["clip_slots"]
    corr = t08.Corrections(len(cam_id), len(cam_names)).to(dev)
    corr.load_state_dict(ck["corr"])
    mode = ck.get("rasterize_mode", "antialiased")
    C = np.load(CLIP / "cameras.npz")
    head = cam_names.index("headcam")
    sh = torch.cat([p["sh0"], p["shN"]], 1)
    geo = (p["means"], F.normalize(p["quats"], dim=-1), torch.exp(p["scales"]),
           torch.sigmoid(p["opacities"]), sh)  # fmt: skip
    head_masks = t08b.npz_mmap(Path(ck["data_files"][0]), "masks") if args.refine else None

    def render(vm, K, w, h, ci, log_gain):
        rgb, _, _ = rasterization(*geo, vm, K, w, h, sh_degree=ck["sh_degree"],
                                  rasterize_mode=mode)  # fmt: skip
        rgb = torch.einsum("hwc,dc->hwd", rgb[0], corr.color[ci]) + corr.bias[ci]
        return rgb * torch.exp(log_gain)

    for cam in [str(c) for c in C["names"]]:
        if args.cams and cam not in args.cams:
            continue
        ci = cam_names.index(cam)
        K = torch.from_numpy(C[f"{cam}_K"]).float().to(dev)[None]
        w, h = (int(v) for v in C[f"{cam}_size"])
        rows = torch.from_numpy(np.flatnonzero(cam_id == ci)).to(dev)
        mean_gain = corr.log_gain[rows].mean().detach()
        d = out / cam
        d.mkdir(parents=True, exist_ok=True)
        for f, T in enumerate(C[f"{cam}_T"]):
            vm0 = torch.from_numpy(t08.ros_to_cv(T)).float().to(dev)[None]
            if ci == head:
                slot = torch.tensor([int(clip[f])], device=dev)
                with torch.no_grad():
                    vm, gain = corr.viewmat(vm0, slot), corr.log_gain[slot[0]]
                if args.refine:
                    gt = cv2.imread(str(CLIP / "frames" / cam / f"{f:04d}.jpg"))[..., ::-1]
                    gt = torch.from_numpy(gt.copy()).float().to(dev) / 255
                    m = torch.from_numpy(np.asarray(head_masks[int(clip[f])])).float().to(dev)
                    m = m[..., None]
                    dr = torch.zeros(1, 3, device=dev, requires_grad=True)
                    dt = torch.zeros(1, 3, device=dev, requires_grad=True)
                    dg = torch.zeros((), device=dev, requires_grad=True)
                    opt = torch.optim.Adam([dr, dt, dg], lr=1e-3)
                    for _ in range(args.refine):
                        D4 = torch.eye(4, device=dev)[None].clone()
                        D4[:, :3, :3] = t08.axis_angle_to_matrix(dr)
                        D4[:, :3, 3] = dt
                        r = render(D4 @ vm, K, w, h, ci, gain + dg)
                        loss = ((r - gt).abs() * m).sum() / (m.sum() * 3)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    with torch.no_grad():
                        D4 = torch.eye(4, device=dev)[None].clone()
                        D4[:, :3, :3] = t08.axis_angle_to_matrix(dr)
                        D4[:, :3, 3] = dt
                        vm, gain = D4 @ vm, gain + dg
            else:
                with torch.no_grad():
                    vm, gain = corr.viewmat(vm0, rows[:1]), mean_gain
            with torch.no_grad():
                rgb = render(vm, K, w, h, ci, gain).clamp(0, 1).cpu().numpy()
            cv2.imwrite(str(d / f"{f:04d}.png"), (rgb[..., ::-1] * 255).astype(np.uint8))
        print(f"rendered {cam}: {len(C[f'{cam}_T'])} frames -> {d}", flush=True)


if __name__ == "__main__":
    main()
