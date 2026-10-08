"""Iron preview: frame 30, one row per camera: [original crop + traced outline | our render].

Usage: uv run scripts/20_iron_preview.py <tag>   (uses iron_mesh.npz colours if present)
Our panel is a pure 3D render of the mesh (unlit baked colours) on a grey background.
"""

import importlib
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "scripts")
C = importlib.import_module("20_iron_common")
cams = np.load("out/clip/cameras.npz")
PART_RGB = (
    np.array(
        [
            [180, 180, 180],
            [120, 205, 195],
            [120, 205, 195],
            [235, 235, 235],
            [90, 90, 90],
            [245, 245, 245],
        ]
    )
    / 255
)


def load_mesh():
    p = f"{C.OUT}/iron_mesh.npz"
    if os.path.exists(p):
        d = np.load(p)
        return d["vertices"], d["faces"], d["corner_srgb"]
    if os.path.exists(f"{C.OUT}/shape_refined.npz"):
        d = np.load(f"{C.OUT}/shape_refined.npz")
        rgb = np.array([[90, 90, 90], [120, 205, 195], [245, 245, 245]]) / 255
        return d["vertices"], d["faces"], np.repeat(rgb[d["part"]][:, None], 3, 1)
    poses = np.load(f"{C.OUT}/poses.npz")
    V, F, part = C.build_mesh()
    return C.scaled(V, float(poses["sxy"])), F, np.repeat(PART_RGB[part][:, None], 3, 1)


def render(V, F, col, cam, f, T, bary=True):
    """Full-res RGB render (float, sRGB 0..1), coverage mask, face-id buffer."""
    from orbifold.geometry import transform

    Pw = transform(T, V)
    uv, d = C.cam_project(Pw, cams[f"{cam}_K"], cams[f"{cam}_T"][f])
    buf = C.raster(uv, d, F, 1080, 1920)
    cov = buf >= 0
    img = np.full((1080, 1920, 3), 0.5)
    ys, xs = np.nonzero(cov)
    fi = buf[ys, xs]
    if bary:  # barycentric interpolation of per-corner colours (screen-space)
        a, b, c = uv[F[fi, 0]], uv[F[fi, 1]], uv[F[fi, 2]]
        p = np.stack([xs, ys], 1) + 0.5
        v0, v1, v2 = b - a, c - a, p - a
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        den = np.where(np.abs(den) < 1e-9, 1e-9, den)
        w1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        w2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        w = np.clip(np.stack([1 - w1 - w2, w1, w2], 1), 0, 1)
        w /= w.sum(1, keepdims=True) + 1e-9
        img[ys, xs] = (col[fi] * w[:, :, None]).sum(1)
    else:
        img[ys, xs] = col[fi].mean(1)
    return img, cov, buf


def ignore_mask(cam, f):
    m, h = C.load_masks(cam, f)
    ig = h.copy()
    p = f"{C.OUT}/occluder/{cam}/{f:04d}.png"
    if os.path.exists(p):
        ig |= cv2.imread(p, 0) > 0
    return m, ig & ~m


def preview(tag, frame=30):
    V, F, col = load_mesh()
    T = np.load(f"{C.OUT}/poses.npz")["T"][frame]
    rows = []
    for cam in C.CAMS:
        real = cv2.imread(f"out/clip/frames/{cam}/{frame:04d}.jpg")
        img, cov, _ = render(V, F, col, cam, frame, T)
        m, ig = ignore_mask(cam, frame)
        v = C.iou(cov, m, ig)
        both = m | cov
        ys, xs = np.nonzero(both)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        pad = int(0.15 * max(x1 - x0, y1 - y0)) + 8
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(1920, x1 + pad), min(1080, y1 + pad)
        a = real.copy()
        cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(a, cnt, -1, (0, 0, 255), 2)
        b = (img[..., ::-1] * 255).astype(np.uint8)
        hh = 320
        sc = hh / (y1 - y0)
        A = cv2.resize(a[y0:y1, x0:x1], None, fx=sc, fy=sc)
        B = cv2.resize(b[y0:y1, x0:x1], None, fx=sc, fy=sc)
        if A.shape[1] < 240:
            padw = 240 - A.shape[1]
            A = cv2.copyMakeBorder(A, 0, 0, 0, padw, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            B = cv2.copyMakeBorder(B, 0, 0, 0, padw, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        cv2.putText(A, f"{cam} f{frame} ORIGINAL", (6, 22), 0, 0.6, (0, 255, 255), 2)
        cv2.putText(
            B,
            f"OURS  IoU {v:.3f}" if not np.isnan(v) else "OURS  IoU n/a",
            (6, 22),
            0,
            0.6,
            (0, 255, 255),
            2,
        )
        rows.append(np.hstack([A, np.full((hh, 6, 3), 255, np.uint8), B]))
        print(f"preview {tag} {cam} IoU {v:.3f}")
    W = max(r.shape[1] for r in rows)
    rows = [
        cv2.copyMakeBorder(r, 0, 6, 0, W - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        for r in rows
    ]
    cv2.imwrite(f"{C.OUT}/preview_{tag}.jpg", np.vstack(rows))


if __name__ == "__main__":
    preview(sys.argv[1] if len(sys.argv) > 1 else "latest")
