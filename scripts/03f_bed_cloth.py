"""Stage 03f: the navy track trousers being ironed, as a cloth mesh on the bed.

Measures their footprint from the footage: navy pixels in the head-camera frames (union over
the clip, so the hand/iron don't leave holes), back-projected onto the bed surface plane,
rasterised on a 1 cm grid and meshed. Red stripe pixels are measured the same way.
Only the shape and two colours are measured; no footage pixels are used in the render.

    uv run scripts/03f_bed_cloth.py
Writes out/body/bed_cloth.npz: vertices (N,3), faces (Q,4), face_stripe (Q,) bool,
navy_srgb (3,), stripe_srgb (3,), bed_top.
"""

from pathlib import Path

import cv2
import numpy as np

from orbifold.geometry import CONVENTIONS, project

CLIP, BODY = Path("out/clip"), Path("out/body")
# Measured from the head camera: closest view, and in the same frame as the body.
CAM, FRAMES, CELL = "headcam", (0, 10, 20, 30, 40, 50, 59), 0.01
OTHERS = ("exocam1", "exocam2")


def pixel_rays(uv: np.ndarray, K: np.ndarray, T: np.ndarray) -> np.ndarray:
    """World-space ray directions for pixels (ros_body camera axes)."""
    d_cv = np.c_[(uv - K[:2, 2]) / np.diag(K)[:2], np.ones(len(uv))]
    d_cam = d_cv @ CONVENTIONS["ros_body"].T  # OpenCV -> ros_body camera frame
    return d_cam @ T[:3, :3].T


def navy_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    return (h > 100) & (h < 130) & (s > 60) & (v > 25) & (v < 120)


def solve_bed_height(pix: np.ndarray, K, T, C, guess: float) -> float:
    """Bed-top height that makes the exocam1 navy footprint land on navy pixels in the
    other cameras (exocam1 sees the bed at a grazing angle, so height errors slide it)."""
    others = []
    for cam in OTHERS:
        img = cv2.imread(str(CLIP / "frames" / cam / "0000.jpg"))
        m = cv2.dilate(navy_mask(img).astype(np.uint8), np.ones((7, 7))) > 0
        others.append((m, C[f"{cam}_K"], C[f"{cam}_T"][0]))
    d = pixel_rays(pix, K, T)
    best = (-1.0, guess)
    for h in np.arange(guess - 0.15, guess + 0.15, 0.005):
        t = (h - T[2, 3]) / d[:, 2]
        pts = T[:3, 3] + d[t > 0] * t[t > 0, None]
        score = 0.0
        for m, Ko, To in others:
            uv, fr = project(pts, Ko, To, "ros_body")
            ok = (
                fr
                & (uv[:, 0] >= 0)
                & (uv[:, 0] < m.shape[1])
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < m.shape[0])
            )
            score += m[uv[ok, 1].astype(int), uv[ok, 0].astype(int)].sum() / max(len(pts), 1)
        best = max(best, (score, h))
    print(f"bed top: hand-based guess {guess:.3f} -> solved {best[1]:.3f} (score {best[0]:.2f})")
    return float(best[1])


def main() -> None:
    J, C = np.load(CLIP / "joints.npz"), np.load(CLIP / "cameras.npz")
    names = list(J["names"])
    bed_top = float(J["xyz"][:, names.index("left_middle_mcp"), 2].min()) - 0.03  # as 04
    K, T = C[f"{CAM}_K"], C[f"{CAM}_T"][0]
    first = cv2.imread(str(CLIP / "frames" / CAM / "0000.jpg"))
    ys, xs = np.nonzero(navy_mask(first))
    bed_top = solve_bed_height(np.c_[xs, ys][::20].astype(float), K, T, C, bed_top)

    navy_pts, red_pts, navy_px, red_px = [], [], [], []
    for f in FRAMES:
        img = cv2.imread(str(CLIP / "frames" / CAM / f"{f:04d}.jpg"))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
        navy = navy_mask(img)
        red = ((h < 8) | (h > 170)) & (s > 120) & (v > 60)
        # Keep the navy blob nearest the right hand (the trousers being ironed).
        hand = project(
            J["xyz"][f, [names.index("right_middle_mcp")]], K, C[f"{CAM}_T"][f], "ros_body"
        )[0][0]
        n, lab = cv2.connectedComponents(
            cv2.morphologyEx(navy.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9)))
        )
        best, best_d = 0, np.inf
        for k in range(1, n):
            ys, xs = np.nonzero(lab == k)
            if len(xs) < 2000:
                continue
            dist = np.min(np.hypot(xs - hand[0], ys - hand[1]))
            if dist < best_d:
                best, best_d = k, dist
        blob = lab == best
        blob &= np.arange(img.shape[0])[:, None] < img.shape[0] * 0.75  # not the wearer
        near = cv2.dilate(blob.astype(np.uint8), np.ones((31, 31))) > 0
        for mask, pts, px in ((blob, navy_pts, navy_px), (red & near, red_pts, red_px)):
            ys, xs = np.nonzero(mask)
            uv = np.c_[xs, ys].astype(float)
            Tf = C[f"{CAM}_T"][f]
            d = pixel_rays(uv, K, Tf)
            t = (bed_top - Tf[2, 3]) / d[:, 2]
            ok = t > 0
            pts.append(Tf[:3, 3] + d[ok] * t[ok, None])
            px.append(img[ys[ok], xs[ok]])
    navy_xy, red_xy = np.concatenate(navy_pts)[:, :2], np.concatenate(red_pts)[:, :2]

    lo = navy_xy.min(0) - 0.05
    shape = np.ceil((navy_xy.max(0) + 0.05 - lo) / CELL).astype(int)
    grid = np.zeros(shape[::-1], np.uint8)
    ij = ((navy_xy - lo) / CELL).astype(int)
    grid[ij[:, 1], ij[:, 0]] = 1
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((5, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grid)
    grid = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    stripe = np.zeros_like(grid)
    rj = ((red_xy - lo) / CELL).astype(int)
    ok = (rj >= 0).all(1) & (rj[:, 0] < shape[0]) & (rj[:, 1] < shape[1])
    stripe[rj[ok, 1], rj[ok, 0]] = 1
    stripe = cv2.morphologyEx(stripe, cv2.MORPH_CLOSE, np.ones((3, 3))) & cv2.dilate(
        grid, np.ones((3, 3))
    )
    grid |= stripe

    # Mesh: one quad per occupied cell, shared corner vertices, 6 mm above the bed.
    cells = np.argwhere(grid)  # (row=y, col=x)
    corner_id: dict[tuple, int] = {}
    verts, quads = [], []
    for r, c in cells:
        q = []
        for dr, dc in ((0, 0), (0, 1), (1, 1), (1, 0)):
            key = (r + dr, c + dc)
            if key not in corner_id:
                corner_id[key] = len(verts)
                verts.append((lo[0] + key[1] * CELL, lo[1] + key[0] * CELL, bed_top + 0.006))
            q.append(corner_id[key])
        quads.append(q)
    face_stripe = stripe[cells[:, 0], cells[:, 1]].astype(bool)
    navy_srgb = np.median(np.concatenate(navy_px), 0)[::-1] / 255
    red_srgb = (
        np.median(np.concatenate(red_px), 0)[::-1] / 255
        if len(red_xy)
        else np.array([0.6, 0.1, 0.1])
    )
    BODY.mkdir(parents=True, exist_ok=True)
    np.savez(
        BODY / "bed_cloth.npz",
        vertices=np.array(verts),
        faces=np.array(quads),
        face_stripe=face_stripe,
        navy_srgb=navy_srgb,
        stripe_srgb=red_srgb,
        bed_top=bed_top,
    )
    area = len(cells) * CELL**2
    print(
        f"trousers footprint {area:.2f} m^2 ({len(cells)} cells), stripe cells {face_stripe.sum()}"
    )
    print(
        f"navy sRGB {navy_srgb.round(3)}  stripe sRGB {red_srgb.round(3)}  bed top z {bed_top:.3f}"
    )
    cv2.imwrite(str(BODY / "bed_cloth_footprint.png"), (grid * 128 + stripe * 127)[::-1])


if __name__ == "__main__":
    main()
