"""Stage 06: composite and compare — ORIGINAL | OURS video, one row per camera, full res.

The sim frame is the Blender foreground (body, iron, shadows; RGBA, straight alpha) over
the room render from the splat model (stage 09). Without a room render, the foreground is
shown on grey. Everything stays at the native 1920x1080: no resizing anywhere (a room
render of a different size is an error, not something to silently upscale).

    uv run scripts/06_compare.py [--out out/compare.mp4] [--crf 12]
Writes out/sim/<cam>/0000.png (full-res composites), the comparison video (each panel at
full 1920x1080, 3840x3240 total, H.264 CRF <= 14 via ffmpeg), and prints the mean absolute
pixel error per camera plus the worst frame (the one to fix first).
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

CLIP, FG, BG, OUT = (
    Path("out/clip/frames"),
    Path("out/render"),
    Path("out/room_render"),
    Path("out"),
)
CAMS = ["headcam", "exocam1", "exocam2"]
FPS = 30


def composite(cam: str, f: int, size: tuple[int, int], fg_dir: Path = FG) -> np.ndarray:
    fg = cv2.imread(str(fg_dir / cam / f"{f:04d}.png"), cv2.IMREAD_UNCHANGED)
    if fg.shape[1::-1] != size:
        raise SystemExit(f"{cam} {f}: render is {fg.shape[1::-1]}, footage {size}: render at 100%")
    fg = fg.astype(np.float32)
    bg_path = BG / cam / f"{f:04d}.png"
    if bg_path.exists():
        bg = cv2.imread(str(bg_path)).astype(np.float32)
        if bg.shape[1::-1] != size:
            raise SystemExit(f"{bg_path} is {bg.shape[1::-1]}, not {size}: re-render the room")
    else:
        bg = np.full(fg.shape[:2] + (3,), 128, np.float32)
    if fg.shape[2] == 3:
        return fg.astype(np.uint8)
    a = fg[..., 3:] / 255.0
    return np.rint(fg[..., :3] * a + bg * (1 - a)).clip(0, 255).astype(np.uint8)


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.putText(img, text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 10, cv2.LINE_AA)
    cv2.putText(img, text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 4, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "compare.mp4"))
    ap.add_argument("--crf", type=int, default=12, help="H.264 CRF (<= 14; lower = better)")
    ap.add_argument("--pix-fmt", default="yuv420p", help="yuv444p keeps full chroma res")
    ap.add_argument("--fg", default=str(FG), help="RGBA renders <fg>/<cam>/<f>.png")
    ap.add_argument("--sim", default=str(OUT / "sim"), help="where composites are written")
    ap.add_argument("--no-video", action="store_true", help="composites + errors only")
    args = ap.parse_args()
    fg_dir, sim_dir = Path(args.fg), Path(args.sim)
    frames = sorted(
        set.intersection(*({int(p.stem) for p in (fg_dir / c).glob("*.png")} for c in CAMS))
    )
    n = len(frames)
    step = int(np.median(np.diff(frames))) if n > 1 else 1
    h, w = cv2.imread(str(CLIP / CAMS[0] / f"{frames[0]:04d}.jpg")).shape[:2]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Raw frames straight into ffmpeg: no intermediate lossy step.
    ff = None if args.no_video else subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{2 * w}x{3 * h}", "-framerate", f"{FPS / step:g}", "-i", "-",
         "-c:v", "libx264", "-crf", str(args.crf), "-preset", "slow", "-tune", "film",
         "-pix_fmt", args.pix_fmt, "-movflags", "+faststart", args.out],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    err = np.zeros((len(CAMS), n))
    for k, f in enumerate(frames):
        rows = []
        for i, cam in enumerate(CAMS):
            real = cv2.imread(str(CLIP / cam / f"{f:04d}.jpg"))
            sim = composite(cam, f, (real.shape[1], real.shape[0]), fg_dir)
            (sim_dir / cam).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(sim_dir / cam / f"{f:04d}.png"), sim)
            err[i, k] = np.abs(real.astype(np.float32) - sim.astype(np.float32)).mean()
            rows.append(np.hstack([label(real, f"ORIGINAL {cam}"), label(sim, f"OURS {cam}")]))
        if ff:
            ff.stdin.write(np.ascontiguousarray(np.vstack(rows)).tobytes())
    if ff:
        ff.stdin.close()
        if ff.wait():
            raise SystemExit("ffmpeg failed")
    for i, cam in enumerate(CAMS):
        worst = frames[err[i].argmax()]
        print(f"{cam:8s} mean abs error {err[i].mean():5.1f} / 255   worst frame {worst}")
    if ff:
        print(
            f"wrote {args.out} ({n} frames @ {FPS / step:g} fps, ORIGINAL | OURS, "
            f"{2 * w}x{3 * h}, H.264 CRF {args.crf})"
        )


if __name__ == "__main__":
    main()
