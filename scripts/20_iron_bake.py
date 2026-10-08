"""Iron stage 2b/4/5: exocam2 masks, colour bake, final metrics, preview.

1. exocam2 sees the iron only around the body. Trace it with SAM 2.1 prompted by the
   box of the fitted iron's projection plus bright (white-shell) pixels inside it; reject
   masks mostly outside the projection. Person mask (YOLO11x-seg) = occluder/ignore region.
2. Bake: per face, mean colour of its visible pixels inside the (eroded) traced mask and
   outside the hand, per view; median across all views/frames. Unseen faces filled by
   diffusion over the mesh graph. Corner colours = vertex averages.
3. Metrics (full-res rasterised silhouette vs traced masks) + preview_baked.jpg.
"""
import importlib
import json
import os
import sys

import cv2
import numpy as np
from ultralytics import SAM, YOLO

from orbifold.geometry import transform

sys.path.insert(0, "scripts")
C = importlib.import_module("20_iron_common")
PV = importlib.import_module("20_iron_preview")
cams = np.load("out/clip/cameras.npz")
J = np.load("out/clip/joints.npz")
idx = {n: i for i, n in enumerate(J["names"])}
xyz = J["xyz"]
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
                iron[cy0:cy1, cx0:cx1] = m & ~person[cy0:cy1, cx0:cx1]
                n_ok += iron.any()
    cv2.imwrite(f"{C.OUT}/masks/exocam2/{f:04d}.png", iron.astype(np.uint8) * 255)
    cv2.imwrite(f"{C.OUT}/occluder/exocam2/{f:04d}.png", (person & ~iron).astype(np.uint8) * 255)
print("exocam2 frames with a traced visible iron part:", n_ok, flush=True)
del sam, yolo

# --- 2. bake -----------------------------------------------------------------------------
nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
samples = [[] for _ in range(len(F))]
for cam in C.CAMS:
    for f in range(NF):
        m, h = C.load_masks(cam, f)
        if not m.any():
            continue
        img = cv2.imread(f"out/clip/frames/{cam}/{f:04d}.jpg")[..., ::-1].astype(np.float32) / 255
        _, cov, buf = PV.render(V, F, np.zeros((len(F), 3, 3)), cam, f, Ts[f], bary=False)
        good = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        good &= ~cv2.dilate(h.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        sel = good & cov
        fi = buf[sel]
        if not len(fi):
            continue
        # drop grazing faces
        Tc = cams[f"{cam}_T"][f]
        cen = transform(Ts[f], V[F].mean(1))
        nw = nrm @ Ts[f][:3, :3].T
        vd = Tc[:3, 3] - cen
        vd /= np.linalg.norm(vd, axis=1, keepdims=True)
        facing = (nw * vd).sum(1)
        px = img[sel]
        cnt = np.bincount(fi, minlength=len(F))
        sm = np.stack([np.bincount(fi, px[:, c], minlength=len(F)) for c in range(3)], 1)
        for k in np.nonzero((cnt >= 2) & (facing > 0.25))[0]:
            samples[k].append(sm[k] / cnt[k])
seen = np.array([len(s) > 0 for s in samples])
fc = np.zeros((len(F), 3))
for k in np.nonzero(seen)[0]:
    fc[k] = np.median(np.array(samples[k]), 0)
print("faces seen %d / %d" % (seen.sum(), len(F)), flush=True)
# diffuse into unseen faces via shared vertices
vf = [[] for _ in range(len(V))]
for k, tri in enumerate(F):
    for v in tri:
        vf[v].append(k)
known = seen.copy()
for _ in range(200):
    if known.all():
        break
    vc = np.zeros((len(V), 3)); vn = np.zeros(len(V))
    for k in np.nonzero(known)[0]:
        vc[F[k]] += fc[k]; vn[F[k]] += 1
    new = known.copy()
    for k in np.nonzero(~known)[0]:
        w = vn[F[k]]
        if w.sum() > 0:
            fc[k] = (vc[F[k]]).sum(0) / w.sum(); new[k] = True
    known = new
# soleplate bottom is never seen: give it the median of the sole-edge band
fc[part == 4] = np.median(fc[(part == 0) & seen], 0) if ((part == 0) & seen).any() else 0.7
# corner colours = vertex average of adjacent faces within the same part (keeps part edges crisp)
corner = np.zeros((len(F), 3, 3))
for k, tri in enumerate(F):
    for j, v in enumerate(tri):
        nb = [q for q in vf[v] if part[q] == part[k]]
        corner[k, j] = fc[nb].mean(0)
corner = np.clip(corner, 0, 1).astype(np.float32)
np.savez(f"{C.OUT}/iron_mesh.npz", vertices=V.astype(np.float32), faces=F.astype(np.int32),
         corner_srgb=corner, face_srgb=fc.astype(np.float32), part=part, seen=seen,
         part_names=np.array(["sole_edge", "base", "shell", "top", "sole_bottom", "handle"]))

# --- 3. metrics ------------------------------------------------------------------------
kn = xyz[:, [idx["right_index_mcp"], idx["right_middle_mcp"], idx["right_pinky_mcp"]]].mean(1)
palm = (xyz[:, idx["right_wrist"]] + kn) / 2
res = {}
for cam in C.CAMS:
    vals, vals_raw = [], []
    for f in range(NF):
        _, cov, _ = PV.render(V, F, corner, cam, f, Ts[f], bary=False)
        m, ig = PV.ignore_mask(cam, f)
        if m.any():
            vals.append(C.iou(cov, m, ig))
            vals_raw.append(C.iou(cov, m, np.zeros_like(m)))
    res[cam] = dict(iou_occlusion_aware=float(np.mean(vals)) if vals else None,
                    iou_raw=float(np.mean(vals_raw)) if vals_raw else None, n_frames=len(vals))
    print(cam, res[cam], flush=True)
# penetration: right-hand joints inside the iron volume, and palm clearance over the handle
hand_j = [i for n, i in idx.items() if n.startswith("right_") and any(k in n for k in ("thumb", "index", "middle", "ring", "pinky", "wrist"))]
inside_cnt = 0
path = C._catmull(C.HANDLE_PATH, 20) * [sxy, 1]
for f in range(NF):
    loc = transform(np.linalg.inv(Ts[f]), xyz[f, hand_j])
    m_ = len(C.outline(80))
    rings = [(V[k * m_:(k + 1) * m_, :2].astype(np.float32), V[k * m_:(k + 1) * m_, 2].min()) for k in range(13)]
    for p in loc:
        # handle tube (ellipse radii) test
        dx = np.hypot(path[:, 0] - p[0], path[:, 1] - p[2]).min()
        in_handle = (dx / C.HANDLE_R[1]) ** 2 + (p[1] / (C.HANDLE_R[0] * sxy)) ** 2 < 1
        in_body = p[2] > 0 and any(p[2] < zk and cv2.pointPolygonTest(rk, (float(p[0]), float(p[1])), False) > 0
                                   for rk, zk in rings)
        if in_body or in_handle:
            inside_cnt += 1
clear = palm[:, 2] - (Ts[:, 2, 3] + V[:, 2].max())
res["penetration"] = dict(hand_joints_inside_iron=int(inside_cnt), joints_checked=len(hand_j) * NF,
                          palm_clearance_cm_min=float(100 * clear.min()), palm_clearance_cm_median=float(100 * np.median(clear)),
                          frames_clearance_below_1p5cm=int((clear < 0.0149).sum()))
res["globals"] = dict(sxy=sxy, bed_z=float(poses["bed_z"]), length_m=float(np.ptp(V[:, 0])), width_m=float(np.ptp(V[:, 1])),
                      height_m=float(V[:, 2].max()), lift_cm_max=float(100 * (Ts[:, 2, 3] - poses["bed_z"]).max()))
print(json.dumps(res, indent=1))
json.dump(res, open(f"{C.OUT}/metrics.json", "w"), indent=1)
PV.preview("baked")
