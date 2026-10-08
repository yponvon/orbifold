"""Stage 16: score Track A candidate frame folders against the original footage.

For each camera, frames 0,10..50: mean absolute error (0-255) in the ring around the
person's outline (where the blur is) and over the whole frame. Lower is better.

    UV_NO_SYNC=1 uv run scripts/16_eval_trackA.py out/final_trackA_clean out/cand_x ...
"""

import sys

import cv2
import numpy as np

CAMS = ("headcam", "exocam1", "exocam2")


def score(d: str, cam: str) -> tuple[float, float]:
    ring_e, all_e = [], []
    for f in range(0, 60, 10):
        real = cv2.imread(f"out/clip/frames/{cam}/{f:04d}.jpg").astype(np.float32)
        a = cv2.imread(f"out/assets/human_gs_clean/render/{cam}/{f:04d}.png", cv2.IMREAD_UNCHANGED)[..., 3]
        m = (a > 128).astype(np.uint8)
        ring = (cv2.dilate(m, np.ones((25, 25))) - cv2.erode(m, np.ones((9, 9)))) > 0
        o = cv2.imread(f"{d}/{cam}/{f:04d}.png").astype(np.float32)
        ring_e.append(np.abs(o - real)[ring].mean())
        all_e.append(np.abs(o - real).mean())
    return float(np.mean(ring_e)), float(np.mean(all_e))


for d in sys.argv[1:]:
    print(d, " | ".join(f"{c}: ring {r:.2f} frame {a:.2f}" for c in CAMS for r, a in [score(d, c)]))
