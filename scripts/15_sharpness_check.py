"""Stage 15: sharpness check — is our composite as sharp as the original footage?

For every camera, compares OURS (out/sim/<cam>/<f>.png) with the ORIGINAL frame
(out/clip/frames/<cam>/<f>.jpg) on the luma channel, in two regions:
  * person: our rendered foreground (alpha from out/render), dilated so both the real
    and the rendered person's edges fall inside it,
  * background: everything well away from the person.
Metrics: variance of the Laplacian and mean gradient magnitude (Sobel). Prints the
ours/original ratios; the target is >= 0.9 everywhere.

    uv run scripts/15_sharpness_check.py [--sim out/sim] [--fg out/render]
    uv run scripts/15_sharpness_check.py --tile 640x360   # what a downscaled preview costs
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

CLIP = Path("out/clip/frames")
CAMS = ["headcam", "exocam1", "exocam2"]
TARGET = 0.9


def luma(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32)


def metrics(g: np.ndarray, m: np.ndarray) -> tuple[float, float]:
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return float(lap[m].var()), float(np.sqrt(gx * gx + gy * gy)[m].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="out/sim", help="dir with <cam>/<f>.png composites")
    ap.add_argument("--fg", default="out/render", help="dir with <cam>/<f>.png RGBA renders")
    ap.add_argument("--tile", help="WxH: simulate a downscale-then-upscale preview of ours")
    ap.add_argument("--cams", nargs="+", default=CAMS)
    args = ap.parse_args()
    sim, fg = Path(args.sim), Path(args.fg)
    tile = tuple(int(v) for v in args.tile.split("x")) if args.tile else None
    k_person = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121))
    print(f"{'camera':8s} {'region':10s} {'lapvar ours/orig':>22s} {'ratio':>6s}"
          f" {'grad ours/orig':>16s} {'ratio':>6s}")  # fmt: skip
    worst = np.inf
    for cam in args.cams:
        frames = sorted(int(p.stem) for p in (sim / cam).glob("*.png"))
        acc = {r: np.zeros(4) for r in ("person", "background")}
        cnt = {r: 0 for r in acc}
        for f in frames:
            if not (fg / cam / f"{f:04d}.png").exists():
                continue  # no alpha for this frame (e.g. a STEP>1 render): can't mask it
            ours = cv2.imread(str(sim / cam / f"{f:04d}.png"))
            orig = cv2.imread(str(CLIP / cam / f"{f:04d}.jpg"))
            if tile:
                ours = cv2.resize(cv2.resize(ours, tile, interpolation=cv2.INTER_AREA),
                                  ours.shape[1::-1], interpolation=cv2.INTER_LINEAR)  # fmt: skip
            alpha = cv2.imread(str(fg / cam / f"{f:04d}.png"), cv2.IMREAD_UNCHANGED)
            a = alpha[..., 3] if alpha.ndim == 3 and alpha.shape[2] == 4 else alpha[..., 0] * 0
            person = cv2.dilate((a > 127).astype(np.uint8), k_person) > 0
            bg = cv2.dilate((a > 64).astype(np.uint8), k_bg) == 0
            # Ignore a thin border (filter edge effects).
            border = np.zeros_like(bg)
            border[8:-8, 8:-8] = True
            go, gr = luma(ours), luma(orig)
            for name, m in (("person", person & border), ("background", bg & border)):
                if m.sum() < 500:
                    continue
                acc[name] += np.array([*metrics(go, m), *metrics(gr, m)])
                cnt[name] += 1
        for name, (lo, gro, lr, grr) in acc.items():
            if lr == 0:
                print(f"{cam:8s} {name:10s}  (no pixels)")
                continue
            r1, r2 = lo / lr, gro / grr
            worst = min(worst, r1, r2)
            n = cnt[name]
            flag = ("" if min(r1, r2) >= TARGET else "  <-- SOFT") + f"  ({n} frames)"
            print(f"{cam:8s} {name:10s} {lo / n:10.1f}/{lr / n:<10.1f} {r1:6.2f}"
                  f" {gro / n:7.2f}/{grr / n:<7.2f} {r2:6.2f}{flag}")  # fmt: skip
    print(f"worst ratio {worst:.2f} (target >= {TARGET}) over {args.sim}"
          + (f" via {args.tile} tiles" if tile else ""))  # fmt: skip


if __name__ == "__main__":
    main()
