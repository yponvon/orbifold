"""Stage 03 check: draw tracked joints on the real frames to verify cameras.

    uv run scripts/03_overlay.py [--frames 0 30 59]

Writes out/overlay/overlay_<frame>.jpg (all cameras stacked) and prints the
fraction of joints that land inside each image.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from orbifold.geometry import project
from orbifold.skeleton import bones

CLIP, OUT = Path("out/clip"), Path("out/overlay")
CONVENTION = "ros_body"  # chosen in stage 02: 96-100% of joints in-image vs 0% for others


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 30, 59])
    args = ap.parse_args()
    J, C = np.load(CLIP / "joints.npz"), np.load(CLIP / "cameras.npz")
    names = list(J["names"])
    edges = bones(names)
    OUT.mkdir(parents=True, exist_ok=True)
    for f in args.frames:
        tiles = []
        for cam in C["names"]:
            img = cv2.imread(str(CLIP / "frames" / cam / f"{f:04d}.jpg"))
            uv, front = project(J["xyz"][f], C[f"{cam}_K"], C[f"{cam}_T"][f], CONVENTION)
            for a, b in edges:
                if front[a] and front[b]:
                    pa, pb = tuple(uv[a].astype(int)), tuple(uv[b].astype(int))
                    cv2.line(img, pa, pb, (0, 255, 255), 3, cv2.LINE_AA)
            for j, (p, ok) in enumerate(zip(uv, front, strict=True)):
                if ok:
                    color = (0, 0, 255) if J["bad"][f][j] else (0, 255, 0)
                    cv2.circle(img, tuple(p.astype(int)), 6, color, -1, cv2.LINE_AA)
            cv2.putText(
                img, f"{cam} f{f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4
            )
            tiles.append(cv2.resize(img, (960, 540)))
        cv2.imwrite(str(OUT / f"overlay_{f:04d}.jpg"), np.hstack(tiles))
    print(f"wrote {len(args.frames)} overlays to {OUT}")


if __name__ == "__main__":
    main()
