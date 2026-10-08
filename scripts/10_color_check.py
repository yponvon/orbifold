"""Stage 10: compare the rendered body's colour with the real person's, per garment band.

Real person pixels come from YOLO segmentation of the real frames; sim body pixels are
the opaque pixels of the Blender foreground. Compares the dark (trousers/hair), mid and
bright (shirt) parts separately via percentiles, and suggests a light-level multiplier.

    uv run --extra recon scripts/10_color_check.py [--frames 0 20 40]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CLIP, FG = Path("out/clip/frames"), Path("out/render")
CAMS = ["exocam1", "exocam2"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 20, 40])
    args = ap.parse_args()
    seg = YOLO("yolo11x-seg.pt")
    ratios = []
    for cam in CAMS:
        real_px, sim_px = [], []
        for f in args.frames:
            real = cv2.imread(str(CLIP / cam / f"{f:04d}.jpg"))
            res = seg.predict(real, classes=[0], verbose=False, retina_masks=True)[0]
            if res.masks is not None:
                real_px.append(real[res.masks.data.any(0).cpu().numpy()])
            fg = cv2.imread(str(FG / cam / f"{f:04d}.png"), cv2.IMREAD_UNCHANGED)
            sim_px.append(fg[..., :3][fg[..., 3] > 240])
        r = cv2.cvtColor(np.concatenate(real_px)[None], cv2.COLOR_BGR2LAB)[0].astype(float)
        s = cv2.cvtColor(np.concatenate(sim_px)[None], cv2.COLOR_BGR2LAB)[0].astype(float)
        print(f"{cam}: lightness (0-255) percentiles 10/50/90")
        print(
            f"  real {np.percentile(r[:, 0], [10, 50, 90]).round()}  ab {r[:, 1:].mean(0).round()}"
        )
        print(
            f"  sim  {np.percentile(s[:, 0], [10, 50, 90]).round()}  ab {s[:, 1:].mean(0).round()}"
        )
        # Brightness ratio on the bright band (shirt), converted from L* to linear light.
        lin = lambda L: ((L / 255 * 100 + 16) / 116) ** 3  # noqa: E731
        ratios.append(lin(np.percentile(r[:, 0], 90)) / lin(np.percentile(s[:, 0], 90)))
    print(f"suggested light multiplier (bright band): {np.mean(ratios):.2f}")


if __name__ == "__main__":
    main()
