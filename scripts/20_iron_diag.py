"""Per-frame IoU table + montage of the worst frames per camera (mask red, ours cyan)."""
import importlib
import sys

import cv2
import numpy as np

from orbifold.geometry import transform

sys.path.insert(0, "scripts")
C = importlib.import_module("20_iron_common")
PV = importlib.import_module("20_iron_preview")
cams = np.load("out/clip/cameras.npz")
V, F, _ = PV.load_mesh()
T = np.load(f"{C.OUT}/poses.npz")["T"]
tag = sys.argv[1] if len(sys.argv) > 1 else "diag"
rows = []
for cam in C.CAMS:
    ious, covs = {}, {}
    for f in range(60):
        m, ig = PV.ignore_mask(cam, f)
        if m.sum() < 50:
            continue
        uv, d = C.cam_project(transform(T[f], V), cams[f"{cam}_K"], cams[f"{cam}_T"][f])
        cov = C.raster(uv, d, F, 1080, 1920) >= 0
        ious[f] = C.iou(cov, m, ig)
        covs[f] = cov
    print(cam, " ".join(f"{f}:{v:.2f}" for f, v in ious.items()))
    worst = sorted(ious, key=ious.get)[:4]
    row = []
    for f in worst:
        img = cv2.imread(f"out/clip/frames/{cam}/{f:04d}.jpg")
        m, ig = PV.ignore_mask(cam, f)
        img[ig] = (img[ig] * 0.5).astype(np.uint8)
        for mm, col in ((m, (0, 0, 255)), (covs[f], (255, 255, 0))):
            cnt, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cnt, -1, col, 2)
        ys, xs = np.nonzero(m | covs[f])
        pad = 30
        c = img[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]
        c = cv2.resize(c, (360, 300))
        cv2.putText(c, f"{cam} f{f} IoU {ious[f]:.2f}", (5, 20), 0, 0.6, (0, 255, 255), 2)
        row.append(c)
    rows.append(np.hstack(row))
cv2.imwrite(f"{C.OUT}/{tag}.jpg", np.vstack(rows))
