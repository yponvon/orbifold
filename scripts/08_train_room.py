"""Stage 08: train a Gaussian-splat model of the static room (person masked out).

Learns, alongside the splats:
  - a small pose correction per head-camera image and per tripod camera
    (the recorded head poses are a little noisy),
  - a colour response per camera (3x3 + bias) and an exposure gain per image
    (the cameras differ in white balance; the head camera auto-exposes).

    uv run --extra recon scripts/08_train_room.py [--steps 30000]
Reads out/room/data.npz, writes out/room/splats.pt and out/room/progress_<step>.jpg.
"""

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import DefaultStrategy, rasterization

from orbifold.geometry import CONVENTIONS

ROOM = Path("out/room")
SH_DEGREE = 3


def ros_to_cv(T_world_cam: np.ndarray) -> np.ndarray:
    """ros_body-axes camera pose -> OpenCV world-to-camera view matrix."""
    M = np.eye(4)
    M[:3, :3] = CONVENTIONS["ros_body"]
    return np.linalg.inv(T_world_cam @ M)


def axis_angle_to_matrix(w: torch.Tensor) -> torch.Tensor:
    """(N,3) axis-angle -> (N,3,3) rotation (Rodrigues)."""
    theta = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = w / theta
    K = torch.zeros(w.shape[0], 3, 3, device=w.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K = K - K.transpose(1, 2)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return torch.eye(3, device=w.device) + s * K + (1 - c) * K @ K


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean SSIM of (B,3,H,W) images with an 11x11 Gaussian window."""
    g = torch.exp(-((torch.arange(11, device=a.device) - 5) ** 2) / (2 * 1.5**2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None]
    w = g.expand(3, 1, 11, 11)

    def blur(x):
        return F.conv2d(x, w, padding=5, groups=3)

    mu_a, mu_b = blur(a), blur(b)
    s_a, s_b = blur(a * a) - mu_a**2, blur(b * b) - mu_b**2
    s_ab = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    return (
        ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (s_a + s_b + c2))
    ).mean()


class Corrections(torch.nn.Module):
    """Per-image pose deltas, per-camera colour transform, per-image exposure."""

    def __init__(self, n_images: int, n_cams: int):
        super().__init__()
        self.rot = torch.nn.Parameter(torch.zeros(n_images, 3))
        self.trans = torch.nn.Parameter(torch.zeros(n_images, 3))
        self.color = torch.nn.Parameter(torch.eye(3).repeat(n_cams, 1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(n_cams, 3))
        self.log_gain = torch.nn.Parameter(torch.zeros(n_images))

    def viewmat(self, base: torch.Tensor, pose_slot: torch.Tensor) -> torch.Tensor:
        d = torch.eye(4, device=base.device).repeat(len(pose_slot), 1, 1)
        d[:, :3, :3] = axis_angle_to_matrix(self.rot[pose_slot])
        d[:, :3, 3] = self.trans[pose_slot]
        return d @ base

    def colorize(self, rgb: torch.Tensor, cam: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("bhwc,bdc->bhwd", rgb, self.color[cam]) + self.bias[cam][:, None, None]
        return out * torch.exp(self.log_gain[img])[:, None, None, None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--n-init", type=int, default=300_000)
    # Pushes opacity of unsupported Gaussians (only ever behind the person) to zero so they
    # get pruned instead of keeping their initial colour as floaters.
    ap.add_argument("--opacity-reg", type=float, default=0.01)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)

    D = np.load(ROOM / "data.npz")
    images = torch.from_numpy(D["images"]).to(dev)  # uint8 (N,H,W,3)
    masks = torch.from_numpy(D["masks"]).to(dev)
    Ks = torch.from_numpy(D["K"]).float().to(dev)
    viewmats = torch.from_numpy(np.stack([ros_to_cv(T) for T in D["T_world_cam"]])).float().to(dev)
    cam_id = torch.from_numpy(D["cam_id"]).long().to(dev)
    cam_names = [str(c) for c in D["cam_names"]]
    n_img, H, W = images.shape[:3]
    # Tripod cameras share one pose slot each; head-camera images get their own.
    head = cam_names.index("headcam")
    pose_slot = torch.where(cam_id == head, torch.arange(n_img, device=dev), cam_id)
    print(f"{n_img} images {W}x{H}; cameras {cam_names}")

    # Init: random points in the box spanned by the cameras, padded by 2 m.
    centers = np.stack([T[:3, 3] for T in D["T_world_cam"]])
    lo, hi = centers.min(0) - 2.0, centers.max(0) + 2.0
    lo[2], hi[2] = centers[:, 2].min() - 1.6, centers[:, 2].max() + 1.5
    means = torch.rand(args.n_init, 3, device=dev) * torch.tensor(hi - lo, device=dev).float()
    means += torch.tensor(lo, device=dev).float()
    scene_scale = float(np.linalg.norm(hi - lo) / 2)
    print(f"init {args.n_init} gaussians in box {lo.round(2)} .. {hi.round(2)}")

    n_sh = (SH_DEGREE + 1) ** 2
    params = torch.nn.ParameterDict(
        {
            "means": means,
            "scales": torch.full((args.n_init, 3), math.log(0.04), device=dev),
            "quats": F.normalize(torch.randn(args.n_init, 4, device=dev), dim=-1),
            "opacities": torch.logit(torch.full((args.n_init,), 0.1, device=dev)),
            "sh0": torch.zeros(args.n_init, 1, 3, device=dev),  # neutral grey
            "shN": torch.zeros(args.n_init, n_sh - 1, 3, device=dev),
        }
    )
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in params.items()})
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}  # fmt: skip
    optimizers = {k: torch.optim.Adam([params[k]], lr=lr, eps=1e-15) for k, lr in lrs.items()}
    means_sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / args.steps)
    )
    corr = Corrections(n_img, len(cam_names)).to(dev)
    corr_opt = torch.optim.Adam(
        [
            {"params": [corr.rot, corr.trans], "lr": 1e-4},
            {"params": [corr.color, corr.bias, corr.log_gain], "lr": 1e-3},
        ]
    )
    strategy = DefaultStrategy(refine_stop_iter=int(args.steps * 0.6), verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    batch, t0 = 4, time.time()
    for step in range(args.steps):
        idx = torch.randint(0, n_img, (batch,), device=dev)
        vm = corr.viewmat(viewmats[idx], pose_slot[idx]) if step > 1000 else viewmats[idx]
        sh = torch.cat([params["sh0"], params["shN"]], 1)
        render, alpha, info = rasterization(
            params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), sh, vm, Ks[idx], W, H,
            sh_degree=min(step // 1000, SH_DEGREE), packed=False,
        )  # fmt: skip
        render = corr.colorize(render, cam_id[idx], idx)
        gt = images[idx].float() / 255.0
        m = masks[idx].float()[..., None]
        comp = render * m + gt * (1 - m)  # masked pixels contribute nothing
        l1 = (render - gt).abs().mul(m).sum() / (m.sum() * 3).clamp_min(1)
        loss = 0.8 * l1 + 0.2 * (1 - ssim(comp.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)))
        loss = loss + 1e-3 * (corr.trans**2).sum() + 1e-3 * (corr.rot**2).sum()
        loss = loss + args.opacity_reg * torch.sigmoid(params["opacities"]).mean()

        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        for opt in [*optimizers.values(), corr_opt]:
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_sched.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)

        if step % 1000 == 0 or step == args.steps - 1:
            with torch.no_grad():
                mse = ((render - gt) ** 2).mul(m).sum() / (m.sum() * 3).clamp_min(1)
                psnr = -10 * torch.log10(mse)
            print(
                f"step {step:6d}  loss {loss.item():.4f}  masked PSNR {psnr.item():5.2f} dB  "
                f"gaussians {len(params['means']):,}  {time.time() - t0:5.0f}s"
            )
        if step % 5000 == 0 or step == args.steps - 1:
            vis = np.hstack([gt[0].cpu().numpy(), render[0].detach().clamp(0, 1).cpu().numpy()])
            cv2.imwrite(
                str(ROOM / f"progress_{step:06d}.jpg"), (vis[..., ::-1] * 255).astype(np.uint8)
            )

    torch.save(
        {"params": {k: v.detach() for k, v in params.items()}, "corr": corr.state_dict(),
         "sh_degree": SH_DEGREE, "cam_names": cam_names},
        ROOM / "splats.pt",
    )  # fmt: skip
    print(f"saved {ROOM / 'splats.pt'}")


if __name__ == "__main__":
    main()
