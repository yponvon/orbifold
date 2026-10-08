"""Stage 13: full-quality progress video — ORIGINAL | OURS, one row per camera.

Uses the original frames (decoded from the MCAP in data/) and our composited frames,
at full 1920x1080 per panel (no downscaling of the content), encoded with H.264 at high
quality via ffmpeg. Output is 3840 x 3240 (2 panels wide, 3 cameras tall).

    uv run scripts/13_progress_video.py --sim out/assets/human_gs/composite \
        --out out/progress/progress_human_gs.mp4
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

CLIP = Path("out/clip/frames")
CAMS = ["headcam", "exocam1", "exocam2"]


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.putText(img, text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 10, cv2.LINE_AA)
    cv2.putText(img, text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 4, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="dir with <cam>/<f>.png composites")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crf", type=int, default=14)
    ap.add_argument("--fps", default="30", help="30 / frame step keeps real-time speed")
    args = ap.parse_args()
    sim = Path(args.sim)
    frames = sorted(int(p.stem) for p in (sim / CAMS[0]).glob("*.png"))
    with tempfile.TemporaryDirectory() as tmp:
        for k, f in enumerate(frames):
            rows = []
            for cam in CAMS:
                real = cv2.imread(str(CLIP / cam / f"{f:04d}.jpg"))
                ours = cv2.imread(str(sim / cam / f"{f:04d}.png"))[..., :3]
                rows.append(np.hstack([label(real, f"ORIGINAL {cam}"), label(ours, f"OURS {cam}")]))
            cv2.imwrite(f"{tmp}/{k:04d}.png", np.vstack(rows))
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-framerate", args.fps, "-i", f"{tmp}/%04d.png",
             "-c:v", "libx264", "-crf", str(args.crf), "-preset", "slow", "-pix_fmt", "yuv420p",
             args.out],
            check=True,
        )  # fmt: skip
    print(f"wrote {args.out} ({len(frames)} frames, ORIGINAL | OURS, full resolution)")


if __name__ == "__main__":
    main()
