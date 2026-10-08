"""Stage 03h: bake the person's real appearance onto the 3D body and garments.

For every face of the fitted body and garment meshes, sample points on the face, project
them into all three cameras for every clip frame, keep samples that are visible (z-buffer
over the whole person) and inside the real person mask (YOLO), and take the median colour.
The result is a per-face-corner colour attribute (sRGB), rendered from 3D in Blender as the
surface's colour — appearance learned from the footage, rendered from our 3D scene (same
principle as the splat room).

    uv run --extra recon scripts/03h_bake_texture.py
Writes out/body/baked.npz: body_colors (T,3,3), tunic_colors, trousers_colors (sRGB 0-1),
and *_coverage (T,) = number of samples behind each face.
"""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from orbifold.geometry import project

CLIP, BODY = Path("out/clip"), Path("out/body")
CAMS = ("headcam", "exocam1", "exocam2")
FRAMES = range(0, 60, 3)
BARY = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1 / 3, 1 / 3, 1 / 3]], float) * 0.85 + 0.05


def main() -> None:
    C = np.load(CLIP / "cameras.npz")
    fit, gar = np.load(BODY / "fit.npz"), np.load(BODY / "garments.npz")
    meshes = {
        "body": (fit["vertices"], fit["faces"]),
        "tunic": (gar["tunic_vertices"], gar["tunic_faces"]),
        "trousers": (gar["trousers_vertices"], gar["trousers_faces"]),
    }
    seg = YOLO("yolo11x-seg.pt")
    samples = {k: [[[] for _ in range(4)] for _ in range(len(f))] for k, (_, f) in meshes.items()}
    for cam in CAMS:
        K = C[f"{cam}_K"]
        for fr in FRAMES:
            T = C[f"{cam}_T"][fr]
            img = cv2.imread(str(CLIP / "frames" / cam / f"{fr:04d}.jpg"))
            h, w = img.shape[:2]
            r = seg.predict(img, classes=[0], verbose=False, retina_masks=True)[0]
            if r.masks is None:
                continue
            person = (
                cv2.erode(r.masks.data.any(0).cpu().numpy().astype(np.uint8), np.ones((5, 5))) > 0
            )
            # z-buffer over all meshes together (garments occlude the body under them)
            pts, owner = [], []
            for name, (V, F) in meshes.items():
                tri = V[fr][F]  # (T,3,3)
                p = np.einsum("bk,tkc->tbc", BARY, tri)  # (T,4,3) sample points
                pts.append(p.reshape(-1, 3))
                owner.append((name, len(F)))
            P = np.concatenate(pts)
            uv, front = project(P, K, T, "ros_body")
            depth = np.linalg.norm(P - T[:3, 3], axis=1)
            px = np.round(uv).astype(int)
            ok = front & (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
            zbuf = np.full((h // 2 + 1, w // 2 + 1), np.inf)
            np.minimum.at(zbuf, (px[ok, 1] // 2, px[ok, 0] // 2), depth[ok])
            vis = ok.copy()
            vis[ok] = depth[ok] <= zbuf[px[ok, 1] // 2, px[ok, 0] // 2] + 0.015
            vis[ok] &= person[px[ok, 1], px[ok, 0]]
            start = 0
            for name, n in owner:
                sl = slice(start, start + n * 4)
                v = vis[sl].reshape(n, 4)
                c = img[px[sl, 1].clip(0, h - 1), px[sl, 0].clip(0, w - 1)].reshape(n, 4, 3)
                for t, b in zip(*np.nonzero(v), strict=True):
                    samples[name][t][b].append(c[t, b])
                start += n * 4
        print(f"sampled {cam}")

    # Faces never seen (e.g. under the arms) fall back to their region's measured colour.
    import json

    region_srgb = json.load(open(BODY / "colors.json"))["colors_srgb"]
    regions = [str(r) for r in fit["regions"]]
    fallback = {
        "body": np.array(
            [region_srgb.get(regions[r], [0.4, 0.33, 0.27]) for r in fit["face_region"]]
        ),
        "tunic": np.tile(region_srgb["shirt"], (len(gar["tunic_faces"]), 1)),
        "trousers": np.tile(region_srgb["trousers"], (len(gar["trousers_faces"]), 1)),
    }
    out = {}
    for name, (_, F) in meshes.items():
        cols = np.zeros((len(F), 4, 3))
        cov = np.zeros(len(F), int)
        for t in range(len(F)):
            for b in range(4):
                if samples[name][t][b]:
                    cols[t, b] = np.median(np.array(samples[name][t][b]), 0)[::-1] / 255
            cov[t] = sum(len(s) for s in samples[name][t])
        # corners: blend each corner sample with the centre sample; fill unseen faces later
        corner = 0.7 * cols[:, :3] + 0.3 * cols[:, 3:4]
        corner[cov == 0] = fallback[name][cov == 0, None, :]
        out[f"{name}_colors"], out[f"{name}_coverage"] = corner, cov
        print(f"{name}: {np.mean(cov > 0):.0%} of faces seen in the footage")
    np.savez(BODY / "baked.npz", **out)
    print(f"wrote {BODY / 'baked.npz'}")


if __name__ == "__main__":
    main()
