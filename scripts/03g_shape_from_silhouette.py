"""Stage 03g: choose the body's girth from the real silhouettes.

Joint positions say nothing about how broad someone is. For each candidate fit (03b with
--weight W, garments from 03e), rasterise body + garments into the tripod views and
compare with the real person mask (YOLO segmentation). The best mean IoU wins and is
copied to out/body/fit.npz and garments.npz.

    uv run --extra recon scripts/03g_shape_from_silhouette.py 0.45 0.7 0.9
(expects out/body/fit_w<W>.npz and garments_w<W>.npz for each W)
"""

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from orbifold.geometry import project

CLIP, BODY = Path("out/clip"), Path("out/body")
CAMS, FRAMES = ("exocam1", "exocam2"), (0, 15, 30, 45, 59)


def raster(verts: np.ndarray, faces: np.ndarray, K, T, shape) -> np.ndarray:
    uv, front = project(verts, K, T, "ros_body")
    mask = np.zeros(shape, np.uint8)
    ok = front[faces].all(1)
    cv2.fillPoly(mask, list(uv[faces[ok]].astype(np.int32)), 1)
    return mask > 0


def main() -> None:
    weights = sys.argv[1:]
    C = np.load(CLIP / "cameras.npz")
    seg = YOLO("yolo11x-seg.pt")
    real = {}
    for cam in CAMS:
        for f in FRAMES:
            img = cv2.imread(str(CLIP / "frames" / cam / f"{f:04d}.jpg"))
            r = seg.predict(img, classes=[0], verbose=False, retina_masks=True)[0]
            real[cam, f] = r.masks.data.any(0).cpu().numpy() if r.masks is not None else None
    scores = {}
    for w in weights:
        fit = np.load(BODY / f"fit_w{w}.npz")
        gar = np.load(BODY / f"garments_w{w}.npz")
        ious = []
        for cam in CAMS:
            K, T = C[f"{cam}_K"], C[f"{cam}_T"]
            for f in FRAMES:
                if real[cam, f] is None:
                    continue
                shape = real[cam, f].shape
                m = raster(fit["vertices"][f], fit["faces"], K, T[f], shape)
                for g in ("tunic", "trousers"):
                    m |= raster(gar[f"{g}_vertices"][f], gar[f"{g}_faces"], K, T[f], shape)
                ious.append((m & real[cam, f]).sum() / max((m | real[cam, f]).sum(), 1))
        scores[w] = float(np.mean(ious))
        print(f"weight {w}: mean silhouette IoU {scores[w]:.3f}")
    best = max(scores, key=scores.get)
    shutil.copy(BODY / f"fit_w{best}.npz", BODY / "fit.npz")
    shutil.copy(BODY / f"garments_w{best}.npz", BODY / "garments.npz")
    print(f"best weight {best} -> out/body/fit.npz, garments.npz")


if __name__ == "__main__":
    main()
