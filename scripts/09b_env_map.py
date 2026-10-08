"""Stage 09b: 360-degree lighting environment rendered from the room model.

Renders a cube map from the splat room at the person's chest height (the person is not in
the room model), converts it to an equirectangular HDR in Blender's convention, and saves
it for 04_build_scene to use as world lighting, so the body is lit by the room's real
window/ceiling light (round-4 review fix 5).

    uv run --extra recon scripts/09b_env_map.py
Writes out/room/env.hdr (linear radiance) and out/room/env_preview.jpg.
"""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization

CLIP, ROOM = Path("out/clip"), Path("out/room")
CKPT = Path(__import__("os").environ.get("SPLATS", "out/room/splats.pt"))
FACE, W, H = 512, 2048, 1024
_spec = importlib.util.spec_from_file_location(
    "train", Path(__file__).with_name("08_train_room.py")
)
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)


def face_basis(fwd: np.ndarray) -> np.ndarray:
    """World->camera rotation (rows: right, down, forward) for a cube face, OpenCV axes."""
    down = np.array([0, 0, -1.0]) if abs(fwd[2]) < 0.9 else np.array([0, 1.0, 0])
    right = np.cross(down, fwd)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd])


@torch.no_grad()
def main() -> None:
    dev = "cuda"
    ckpt = torch.load(CKPT, map_location=dev, weights_only=False)
    p = ckpt["params"]
    n_img = ckpt["corr"]["log_gain"].shape[0]
    corr = train.Corrections(n_img, len(ckpt["cam_names"])).to(dev)
    corr.load_state_dict(ckpt["corr"])
    ci = ckpt["cam_names"].index("exocam1")  # use exocam1's colour response
    cam_id = np.load(ROOM / "data.npz")["cam_id"]
    if len(cam_id) == n_img:
        idx = torch.from_numpy(np.flatnonzero(cam_id == ci)).to(dev)
        gain = torch.exp(corr.log_gain[idx].mean())
    else:  # checkpoint trained on an older data set: use the overall exposure
        gain = torch.exp(corr.log_gain.mean())

    J = np.load(CLIP / "joints.npz")
    names = list(J["names"])
    centre = J["xyz"][:, names.index("spine3")].mean(0)
    print(f"environment probe at {centre.round(2)} (mean spine3 over the clip)")

    sh = torch.cat([p["sh0"], p["shN"]], 1)
    K = torch.tensor([[FACE / 2, 0, FACE / 2], [0, FACE / 2, FACE / 2], [0, 0, 1.0]], device=dev)
    dirs = [
        np.array(d, float)
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    ]
    faces, bases = [], []
    for fwd in dirs:
        R = face_basis(fwd)
        vm = np.eye(4)
        vm[:3, :3], vm[:3, 3] = R, -R @ centre
        rgb, _, _ = rasterization(
            p["means"], F.normalize(p["quats"], dim=-1), torch.exp(p["scales"]),
            torch.sigmoid(p["opacities"]), sh, torch.from_numpy(vm).float().to(dev)[None],
            K[None], FACE, FACE, sh_degree=ckpt["sh_degree"], near_plane=0.05,
        )  # fmt: skip
        rgb = torch.einsum("hwc,dc->hwd", rgb[0], corr.color[ci]) + corr.bias[ci]
        faces.append((rgb * gain).clamp(0, 1).cpu().numpy())
        bases.append(R)

    # Equirect pixel -> direction, in Blender's convention (verified by a probe render):
    # u = 0.5 - azimuth / 2pi (+x at the centre), v = 0.5 + elevation / pi (from the bottom).
    u = (np.arange(W) + 0.5) / W
    v = 1 - (np.arange(H) + 0.5) / H
    az, el = (0.5 - u)[None, :] * 2 * np.pi, (v - 0.5)[:, None] * np.pi
    d = np.stack(
        np.broadcast_arrays(np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)), -1
    )
    out = np.zeros((H, W, 3), np.float32)
    best = np.argmax(np.stack([d @ f for f in dirs], -1), -1)
    for k, R in enumerate(bases):
        m = best == k
        c = d[m] @ R.T  # camera coords (right, down, forward)
        x = (FACE / 2 + FACE / 2 * c[:, 0] / c[:, 2]).clip(0, FACE - 1).astype(int)
        y = (FACE / 2 + FACE / 2 * c[:, 1] / c[:, 2]).clip(0, FACE - 1).astype(int)
        out[m] = faces[k][y, x]
    linear = out**2.2  # display values -> linear radiance
    cv2.imwrite(str(ROOM / "env.hdr"), linear[..., ::-1])
    cv2.imwrite(str(ROOM / "env_preview.jpg"), (out[..., ::-1] * 255).astype(np.uint8))
    up = linear[: H // 2].mean()
    # Key light direction: mean direction of the brightest 0.5% of the sphere (the window),
    # weighted by solid angle (cos elevation).
    lum = linear.mean(-1) * np.cos(el)
    top = lum >= np.quantile(lum, 0.995)
    key = (d[top] * lum[top, None]).sum(0)
    key /= np.linalg.norm(key)
    print(f"key light (window) direction {key.round(2)}")
    stats = {"upper_mean": float(up), "mean": float(linear.mean()), "key_dir": key.tolist()}
    json.dump(stats, open(ROOM / "env_stats.json", "w"))
    print(
        f"wrote {ROOM / 'env.hdr'}  mean linear radiance: upper {up:.3f}, all {linear.mean():.3f}"
    )


if __name__ == "__main__":
    main()
