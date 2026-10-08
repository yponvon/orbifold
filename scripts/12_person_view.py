"""Stage 12: cut-out human comparison for quick review.

For each camera and a few frames: crop around the real person, draw the real clothed
person's mask outline (green, YOLO) and our body + garments outline (red), and report the
silhouette IoU. If a foreground render exists, also show the cut-out sim person next to
the cut-out real person.

    uv run --extra recon scripts/12_person_view.py --tag H1a [--fit F --garments G]
Writes out/person/<tag>.jpg and prints IoU per camera.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from orbifold.geometry import project

CLIP, OUT = Path("out/clip"), Path("out/person")
CAMS, FRAMES = ("headcam", "exocam1", "exocam2"), (30,)


def raster(V, F, K, T, shape):
    uv, front = project(V, K, T, "ros_body")
    m = np.zeros(shape, np.uint8)
    ok = front[F].all(1)
    cv2.fillPoly(m, list(uv[F[ok]].astype(np.int32)), 1)
    return m > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fit", default="out/body/fit.npz")
    ap.add_argument("--garments", default="out/body/garments.npz")
    ap.add_argument("--use-render", action="store_true", help="show our render, not our outline")
    args = ap.parse_args()
    fit, gar, C = np.load(args.fit), np.load(args.garments), np.load(CLIP / "cameras.npz")
    seg = YOLO("yolo11x-seg.pt")
    rows, ious = [], {c: [] for c in CAMS}
    for f in FRAMES:
        tiles = []
        for cam in CAMS:
            img = cv2.imread(str(CLIP / "frames" / cam / f"{f:04d}.jpg"))
            K, T = C[f"{cam}_K"], C[f"{cam}_T"][f]
            r = seg.predict(img, classes=[0], verbose=False, retina_masks=True)[0]
            real = (
                r.masks.data.any(0).cpu().numpy()
                if r.masks is not None
                else np.zeros(img.shape[:2], bool)
            )
            ours = raster(fit["vertices"][f], fit["faces"], K, T, img.shape[:2])
            for g in ("tunic", "trousers"):
                ours |= raster(gar[f"{g}_vertices"][f], gar[f"{g}_faces"], K, T, img.shape[:2])
            if cam == "headcam":  # the wearer's own body; real mask covers arms/legs only
                pass
            iou = (real & ours).sum() / max((real | ours).sum(), 1)
            ious[cam].append(iou)
            vis = img.copy()
            for m, col in ((real, (0, 255, 0)), (ours, (0, 0, 255))):
                cnts, _ = cv2.findContours(
                    m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                cv2.drawContours(vis, cnts, -1, col, 4)
            # crop around the real person (fallback: ours), padded
            ys, xs = np.nonzero(real | ours)
            pad = 40
            y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
            x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
            # Two panels only: ORIGINAL (real frame from data/, traced outline in green)
            # | OURS (our body + garments outline in red over the same frame, or our
            # rendered cut-out if a render exists).
            orig = img.copy()
            cnts, _ = cv2.findContours(
                real.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(orig, cnts, -1, (0, 255, 0), 4)
            ours_panel = vis
            sim_p = Path(f"out/render/{cam}/{f:04d}.png")
            if args.use_render and sim_p.exists():
                fg = cv2.imread(str(sim_p), cv2.IMREAD_UNCHANGED)
                a = fg[..., 3:] / 255.0
                ours_panel = (fg[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
            crop = np.hstack([orig[y0:y1, x0:x1], ours_panel[y0:y1, x0:x1]])
            h = 420
            crop = cv2.resize(crop, (int(crop.shape[1] * h / crop.shape[0]), h))
            cv2.putText(
                crop,
                f"{cam} f{f} IoU {iou:.2f}",
                (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                4,
            )
            cv2.putText(
                crop,
                f"{cam} f{f} IoU {iou:.2f}",
                (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            tiles.append(crop)
        rows.extend(tiles)  # one row per camera
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"{args.tag}.jpg"), np.vstack(rows))
    print(" | ".join(f"{c} IoU {np.mean(v):.3f}" for c, v in ious.items()))
    print(f"wrote {OUT / args.tag}.jpg")


if __name__ == "__main__":
    main()
