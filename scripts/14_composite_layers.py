"""Stage 14: composite the separate assets — room, person, iron — into final frames.

Layers per camera and frame (all rendered from our 3D scene):
  room   out/room_render/<cam>/<f>.png                     (cached, static room)
  person out/assets/human_gs/render/<cam>/<f>.png (RGBA) + depth/<cam>/<f>.npy
  iron   out/assets/iron/render/<cam>/<f>.png (RGBA)     + depth/<cam>/<f>.npy
Person and iron are depth-tested per pixel (fingers in front of the handle, iron in
front of the body), then alpha-composited over the room. Missing layers are skipped.

    uv run scripts/14_composite_layers.py
Writes out/final/<cam>/<f>.png.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOM, HUMAN, IRON, OUT = (
    Path("out/room_render"),
    Path("out/assets/human_gs"),
    Path("out/assets/iron"),
    Path("out/final"),
)
CAMS = ["headcam", "exocam1", "exocam2"]


def layer(root: Path, cam: str, f: int):
    rgba = cv2.imread(str(root / "render" / cam / f"{f:04d}.png"), cv2.IMREAD_UNCHANGED)
    if rgba is None:
        return None
    d_path = root / "depth" / cam / f"{f:04d}.npy"
    depth = np.load(d_path).astype(np.float32) if d_path.exists() else None
    if depth is None:  # no depth: treat as far, so it sits behind a layer that has depth
        depth = np.full(rgba.shape[:2], 1e3, np.float32)
    return rgba[..., :3].astype(np.float32), rgba[..., 3:].astype(np.float32) / 255, depth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default=str(ROOM), help="room render dir (e.g. the sharp one)")
    ap.add_argument("--iron", default=str(IRON), help="iron layer dir (render/ + depth/)")
    ap.add_argument("--human", default=str(HUMAN), help="person layer dir (render/ + depth/)")
    ap.add_argument(
        "--edge",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        help="crisp person outline: remap the person layer's alpha from [LO,HI] to [0,1]",
    )
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument(
        "--grow",
        type=int,
        default=0,
        help="grow the person layer's alpha by N px (covers the room's smeared ring "
        "where the room model never saw behind the person)",
    )
    ap.add_argument(
        "--match-color",
        metavar="MASKS",
        help="per-camera colour response for the person layer, fitted against the real "
        "person (masks dir with <cam>/<f>.png); like the room's per-camera colour",
    )
    args = ap.parse_args()
    room, iron, human, out_dir = Path(args.room), Path(args.iron), Path(args.human), Path(args.out)
    n = 0
    gains = {}
    if args.match_color:  # per-channel gain + bias per camera, least squares on person pixels
        for cam in CAMS:
            xs, ys = [], []
            for f in range(0, 60, 6):
                lay = layer(human, cam, f)
                m_p = Path(args.match_color) / cam / f"{f:04d}.png"
                real_p = Path("out/clip/frames") / cam / f"{f:04d}.jpg"
                if lay is None or not m_p.exists():
                    continue
                c, a, _ = lay
                real = cv2.imread(str(real_p)).astype(np.float32)
                both = (a[..., 0] > 0.95) & (cv2.imread(str(m_p), 0) > 127)
                xs.append(c[both]), ys.append(real[both])
            if xs:
                # Shirt-only match: pixels light in BOTH (the shirt; trousers/skin/hair are
                # dark), per-channel gain = ratio of medians, no bias (a full affine over the
                # whole person collapsed colours toward the mean).
                x, y = np.concatenate(xs), np.concatenate(ys)
                light = (x.mean(1) > 110) & (y.mean(1) > 110)
                if light.sum() > 500:
                    gain = np.median(y[light], 0) / np.maximum(np.median(x[light], 0), 1)
                    gains[cam] = np.stack([gain, np.zeros(3)], 1)
                    print(
                        f"{cam}: shirt colour gain (BGR) {gain.round(2)} from {int(light.sum())} px"
                    )
    for cam in CAMS:
        (out_dir / cam).mkdir(parents=True, exist_ok=True)
        for room_path in sorted((room / cam).glob("*.png")):
            f = int(room_path.stem)
            out = cv2.imread(str(room_path)).astype(np.float32)
            person = layer(human, cam, f)
            if person is not None and cam in gains:
                c, a, d = person
                person = ((c * gains[cam][:, 0] + gains[cam][:, 1]).clip(0, 255), a, d)
            if person is not None and args.grow:
                c, a, d = person
                k = np.ones((2 * args.grow + 1,) * 2, np.uint8)
                a_g = cv2.dilate(a[..., 0], k)[..., None]
                # colours for the grown ring: nearest person colour (dilated premultiplied / alpha)
                c_g = cv2.dilate(c * a, k) / np.maximum(a_g, 1e-3)
                c = np.where(a > 0.5, c, c_g)
                person = (c, np.maximum(a, a_g * 0.999), d)
            if person is not None and args.edge:  # tighten our own rendered outline
                lo, hi = args.edge
                c, a, d = person
                person = (c, ((a - lo) / (hi - lo)).clip(0, 1), d)
            layers = [x for x in (person, layer(iron, cam, f)) if x is not None]
            if len(layers) == 2:  # per pixel, draw the farther layer first
                (c1, a1, d1), (c2, a2, d2) = layers
                near_is_2 = (d2 <= d1)[..., None]
                far_c, far_a = np.where(near_is_2, c1, c2), np.where(near_is_2, a1, a2)
                near_c, near_a = np.where(near_is_2, c2, c1), np.where(near_is_2, a2, a1)
                layers = [(far_c, far_a, None), (near_c, near_a, None)]
            for c, a, _ in layers:
                out = c * a + out * (1 - a)
            cv2.imwrite(str(out_dir / cam / f"{f:04d}.png"), out.clip(0, 255).astype(np.uint8))
            n += 1
    print(f"wrote {n} composited frames to {out_dir}")


if __name__ == "__main__":
    main()
