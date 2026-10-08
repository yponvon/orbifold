"""Stage 03c: measure each garment region's colour from the real footage.

Projects the fitted body (stage 03b) into the tripod cameras, keeps faces that are visible
(z-buffer over face centroids), and takes the median real pixel colour per region.
Only colours are measured; no footage pixels go into the render.

    uv run scripts/03c_sample_colors.py            # real footage -> out/body/colors.json
    uv run scripts/03c_sample_colors.py --sim      # our render   -> out/body/colors_sim.json
Writes {region: [r, g, b]} in sRGB 0-1, plus sample counts. Shoes are skipped: the feet
are hidden behind the bed in both tripod views, so their samples would be the sheet.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from orbifold.geometry import project

CLIP, BODY = Path("out/clip"), Path("out/body")
CAMS, FRAMES, DOWN = ["exocam1", "exocam2"], range(0, 60, 5), 4
SKIP = {"shoes"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="sample our foreground render instead")
    sim = ap.parse_args().sim
    fit, C = np.load(BODY / "fit.npz"), np.load(CLIP / "cameras.npz")
    faces, region = fit["faces"], fit["face_region"]
    regions = [str(r) for r in fit["regions"]]
    samples: dict[str, list] = {r: [] for r in regions}
    for cam in CAMS:
        K, (w, h) = C[f"{cam}_K"], C[f"{cam}_size"]
        for f in FRAMES:
            T = C[f"{cam}_T"][f]
            cen = fit["vertices"][f][faces].mean(1)
            uv, front = project(cen, K, T, "ros_body")
            depth = np.linalg.norm(cen - T[:3, 3], axis=1)
            px = (uv / DOWN).astype(int)
            ok = (
                front
                & (px[:, 0] >= 0)
                & (px[:, 0] < w // DOWN)
                & (px[:, 1] >= 0)
                & (px[:, 1] < h // DOWN)
            )
            zbuf = np.full((h // DOWN, w // DOWN), np.inf)
            np.minimum.at(zbuf, (px[ok, 1], px[ok, 0]), depth[ok])
            vis = ok.copy()
            vis[ok] = depth[ok] <= zbuf[px[ok, 1], px[ok, 0]] + 0.02
            if sim:
                rgba = cv2.imread(f"out/render/{cam}/{f:04d}.png", cv2.IMREAD_UNCHANGED)
                if rgba is None:  # previews render every Nth frame only
                    continue
                img = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2RGB)
                vis &= np.r_[rgba[..., 3][uv[:, 1].clip(0, h - 1).astype(int),
                                          uv[:, 0].clip(0, w - 1).astype(int)] > 240]  # fmt: skip
            else:
                img = cv2.cvtColor(
                    cv2.imread(str(CLIP / "frames" / cam / f"{f:04d}.jpg")), cv2.COLOR_BGR2RGB
                )
            # Sample a 3x3 neighbourhood median to avoid single noisy pixels.
            blur = cv2.medianBlur(img, 3)
            for i in np.flatnonzero(vis):
                x, y = uv[i].astype(int)
                samples[regions[region[i]]].append(blur[y, x])
    colors, counts = {}, {}
    for r, vals in samples.items():
        if vals and r not in SKIP:
            colors[r] = (np.median(np.array(vals), 0) / 255.0).round(3).tolist()
            counts[r] = len(vals)
    name = "colors_sim.json" if sim else "colors.json"
    json.dump({"colors_srgb": colors, "samples": counts}, open(BODY / name, "w"), indent=1)
    for r in colors:
        print(f"{r:9s} sRGB {colors[r]}  ({counts[r]} samples)")


if __name__ == "__main__":
    main()
