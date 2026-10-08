"""Stage 09: render the room splat model from each camera for every clip frame.

Uses the learned per-camera colour response, and for head-camera frames that were in
training, that frame's refined pose and exposure.

    uv run --extra recon scripts/09_render_room.py
Writes out/room_render/<cam>/0000.png ... at full resolution.
"""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization

CLIP, ROOM, OUT = Path("out/clip"), Path("out/room"), Path("out/room_render")

# Reuse the helpers from the training script (scripts/ is not a package).
_spec = importlib.util.spec_from_file_location(
    "train", Path(__file__).with_name("08_train_room.py")
)
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)


@torch.no_grad()
def main() -> None:
    dev = "cuda"
    # Our own checkpoint (trusted); older ones hold numpy strings, so not weights_only.
    ckpt = torch.load(ROOM / "splats.pt", map_location=dev, weights_only=False)
    p = ckpt["params"]
    D, C = np.load(ROOM / "data.npz"), np.load(CLIP / "cameras.npz")
    cam_names = ckpt["cam_names"]
    corr = train.Corrections(len(D["cam_id"]), len(cam_names)).to(dev)
    corr.load_state_dict(ckpt["corr"])
    head = cam_names.index("headcam")
    sh = torch.cat([p["sh0"], p["shN"]], 1)

    for cam in C["names"]:
        ci = cam_names.index(cam)
        K = torch.from_numpy(C[f"{cam}_K"]).float().to(dev)[None]
        w, h = (int(v) for v in C[f"{cam}_size"])
        cam_imgs = np.flatnonzero(D["cam_id"] == ci)
        mean_gain = corr.log_gain[torch.from_numpy(cam_imgs).to(dev)].mean()
        out_dir = OUT / cam
        out_dir.mkdir(parents=True, exist_ok=True)
        for f, T in enumerate(C[f"{cam}_T"]):
            vm = torch.from_numpy(train.ros_to_cv(T)).float().to(dev)[None]
            gain = mean_gain
            if ci == head:
                match = cam_imgs[D["frame_idx"][cam_imgs] == f]
                if len(match):  # refined pose + exposure learned for this exact frame
                    slot = torch.tensor([int(match[0])], device=dev)
                    vm, gain = corr.viewmat(vm, slot), corr.log_gain[slot[0]]
            else:
                vm = corr.viewmat(vm, torch.tensor([ci], device=dev))
            rgb, _, _ = rasterization(
                p["means"], F.normalize(p["quats"], dim=-1), torch.exp(p["scales"]),
                torch.sigmoid(p["opacities"]), sh, vm, K, w, h, sh_degree=ckpt["sh_degree"],
            )  # fmt: skip
            rgb = torch.einsum("hwc,dc->hwd", rgb[0], corr.color[ci]) + corr.bias[ci]
            rgb = (rgb * torch.exp(gain)).clamp(0, 1).cpu().numpy()
            cv2.imwrite(str(out_dir / f"{f:04d}.png"), (rgb[..., ::-1] * 255).astype(np.uint8))
        print(f"rendered room for {cam}: {len(C[f'{cam}_T'])} frames -> {out_dir}")


if __name__ == "__main__":
    main()
