"""Person masks for every clip frame and camera (YOLO11x-seg, class 0).

Keeps the YOLO person instances that overlap the projected (dilated) body+garment mesh, so
bystanders/reflections are dropped. The head camera is mounted upside down, so it is
segmented both as-is and rotated 180 degrees and the matching instances are unioned
(the wearer's own arms/torso/legs are the person).

    UV_NO_SYNC=1 uv run scripts/30_human_gs_masks.py
Writes out/assets/human_gs/masks/<cam>/<f>.png (0/255) and masks/stats.json.
"""

import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
hgs = importlib.import_module("30_human_gs_common")


def instances(model, img):
    r = model(img, classes=[0], conf=0.15, retina_masks=True, verbose=False)[0]
    if r.masks is None:
        return []
    return [m.astype(bool) for m in r.masks.data.cpu().numpy()]


def main():
    model = YOLO("yolo11x-seg.pt")
    cams = hgs.load_cameras()
    verts, faces, _ = hgs.load_mesh()
    stats = {}
    for cam in hgs.CAMS:
        K, vms, size = cams[cam]
        d = hgs.OUT / "masks" / cam
        d.mkdir(parents=True, exist_ok=True)
        ious = []
        for f in range(hgs.N_FRAMES):
            img = cv2.imread(str(hgs.CLIP / "frames" / cam / f"{f:04d}.jpg"))
            sil = hgs.mesh_silhouette(verts[f], faces, K, vms[f], size) > 0
            k = 61 if cam == "headcam" else 31
            sil_d = cv2.dilate(sil.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
            cands = instances(model, img)
            if cam == "headcam":
                cands += [m[::-1, ::-1] for m in instances(model, img[::-1, ::-1].copy())]
            mask = np.zeros(sil.shape, bool)
            for m in cands:
                if m.sum() and (m & sil_d).sum() / m.sum() > 0.3:
                    mask |= m
            cv2.imwrite(str(d / f"{f:04d}.png"), mask.astype(np.uint8) * 255)
            iou = (mask & sil).sum() / max((mask | sil).sum(), 1)
            ious.append(float(iou))
            print(f"{cam} {f:02d}: {len(cands)} cands, mask {mask.mean() * 100:.1f}% "
                  f"mesh-IoU {iou:.2f}", flush=True)  # fmt: skip
        stats[cam] = {"mesh_iou_mean": float(np.mean(ious)), "mesh_iou": ious}
    (hgs.OUT / "masks" / "stats.json").write_text(json.dumps(stats, indent=1))
    print({c: round(s["mesh_iou_mean"], 3) for c, s in stats.items()})


if __name__ == "__main__":
    main()
