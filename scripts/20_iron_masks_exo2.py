"""Iron stage 1b: exocam2 iron masks, prompted by the fitted iron's projection.

The aqua-colour seed of 20_iron_masks finds nothing in exocam2 (the iron shows its white
shell there), so: SAM 2.1 with the projected iron box + a bright pixel inside it; keep the
mask only if mostly inside the (dilated) projection. YOLO11x-seg person mask = occluder.
Writes masks/exocam2/*.png, occluder/exocam2/*.png, masks_check_exo2.jpg.
"""
import importlib
import os
import sys

import cv2
import numpy as np
from ultralytics import SAM, YOLO

sys.path.insert(0, "scripts")
C = importlib.import_module("20_iron_common")
PV = importlib.import_module("20_iron_preview")
poses = np.load(f"{C.OUT}/poses.npz")
Ts, sxy = poses["T"], float(poses["sxy"])
V0, F, part = C.build_mesh()
V = C.scaled(V0, sxy)
NF = 60

# --- 1. exocam2 masks + occluders -----------------------------------------------------
sam = SAM(f"{C.OUT}/weights/sam2.1_b.pt")
yolo = YOLO("yolo11x-seg.pt")
os.makedirs(f"{C.OUT}/occluder/exocam2", exist_ok=True)
n_ok = 0
for f in range(NF):
    img = cv2.imread(f"out/clip/frames/exocam2/{f:04d}.jpg")
    _, cov, _ = PV.render(V, F, np.zeros((len(F), 3, 3)), "exocam2", f, Ts[f], bary=False)
    r = yolo(img, classes=[0], verbose=False)[0]
    person = np.zeros(cov.shape, bool)
    if r.masks is not None:
        for mm in r.masks.data.cpu().numpy():
            person |= cv2.resize(mm, (1920, 1080)) > 0.5
    ys, xs = np.nonzero(cov)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = int(0.3 * max(x1 - x0, y1 - y0))
    cx0, cy0, cx1, cy1 = max(0, x0 - 3 * pad), max(0, y0 - 3 * pad), min(1920, x1 + 3 * pad), min(1080, y1 + 3 * pad)
    crop = img[cy0:cy1, cx0:cx1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    covc = cv2.dilate(cov.astype(np.uint8), np.ones((15, 15), np.uint8))[cy0:cy1, cx0:cx1] > 0
    bright = covc & (hsv[..., 2] > 150) & ((hsv[..., 1] < 70) | ((hsv[..., 0] >= 76) & (hsv[..., 0] <= 100))) & ~person[cy0:cy1, cx0:cx1]
    bright = cv2.morphologyEx(bright.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    iron = np.zeros(cov.shape, bool)
    if bright.sum() >= 20:
        n, lab, st, cen = cv2.connectedComponentsWithStats(bright.astype(np.uint8))
        k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
        pyx = np.argwhere(lab == k)
        pt = pyx[np.argmin(((pyx - cen[k][::-1]) ** 2).sum(1))][::-1].astype(float)
        box = [x0 - pad - cx0, y0 - pad - cy0, x1 + pad - cx0, y1 + pad - cy0]
        res = sam(crop, bboxes=[box], points=[[list(pt)]], labels=[[1]], verbose=False)
        if res[0].masks is not None:
            m = res[0].masks.data.cpu().numpy()[0].astype(bool)
            inside = (m & covc).sum() / max(m.sum(), 1)
            if inside > 0.6 and m.sum() < 4 * covc.sum():
                # colour gate: the iron is white / pale aqua here (drop green sheet, navy cloth)
                h_, s_, v_ = hsv[..., 0], hsv[..., 1], hsv[..., 2]
                pale = (v_ > 120) & ((s_ < 80) | ((h_ >= 76) & (h_ <= 100)))
                g = m & pale & ~person[cy0:cy1, cx0:cx1]
                g = cv2.morphologyEx(g.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                n_, lab_, st_, _ = cv2.connectedComponentsWithStats(g)
                if n_ > 1:
                    g = lab_ == 1 + np.argmax(st_[1:, cv2.CC_STAT_AREA])
                    iron[cy0:cy1, cx0:cx1] = g
                n_ok += iron.any()
    cv2.imwrite(f"{C.OUT}/masks/exocam2/{f:04d}.png", iron.astype(np.uint8) * 255)
    cv2.imwrite(f"{C.OUT}/occluder/exocam2/{f:04d}.png", (person & ~iron).astype(np.uint8) * 255)
print("exocam2 frames with a traced visible iron part:", n_ok, flush=True)
del sam, yolo


rows = []
for f in range(0, 60, 5):
    img = cv2.imread(f"out/clip/frames/exocam2/{f:04d}.jpg")
    m = cv2.imread(f"{C.OUT}/masks/exocam2/{f:04d}.png", 0) > 0
    o = cv2.imread(f"{C.OUT}/occluder/exocam2/{f:04d}.png", 0) > 0
    _, cov, _ = PV.render(V, F, np.zeros((len(F), 3, 3)), "exocam2", f, Ts[f], bary=False)
    img[m] = (0.5 * img[m] + [0, 0, 127]).astype(np.uint8)
    img[o] = (0.6 * img[o]).astype(np.uint8)
    cnt, _ = cv2.findContours(cov.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnt, -1, (255, 255, 0), 1)
    ys, xs = np.nonzero(cov)
    cx, cy = int(xs.mean()), int(ys.mean())
    c = img[max(0, cy - 120):cy + 120, max(0, cx - 160):cx + 160]
    c = cv2.resize(c, (320, 240))
    cv2.putText(c, f"exo2 {f} px={m.sum()}", (5, 20), 0, 0.6, (0, 255, 255), 2)
    rows.append(c)
cv2.imwrite(f"{C.OUT}/masks_check_exo2.jpg", np.vstack([np.hstack(rows[i:i + 4]) for i in range(0, 12, 4)]))
