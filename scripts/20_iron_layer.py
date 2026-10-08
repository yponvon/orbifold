"""Iron layer (deadline path): bake colours on the refined mesh, render RGBA + depth.

Bake: per face, mean of its visible pixels inside the eroded traced mask and outside the
dilated hand/occluder, per view (facing > 0.25); median across all 3 cams x 60 frames.
Writes iron_mesh.npz, render/<cam>/<f>.png (RGBA straight alpha), depth/<cam>/<f>.npy
(float16 camera-z metres, inf where empty), layer_metrics.json, preview_layer.jpg, RENDER_DONE.
"""
import importlib
import json
import os
import sys

import cv2
import numpy as np

from orbifold.geometry import transform

sys.path.insert(0, "scripts")
C = importlib.import_module("20_iron_common")
PV = importlib.import_module("20_iron_preview")
cams = np.load("out/clip/cameras.npz")
S = np.load(f"{C.OUT}/shape_refined.npz")
V, F, part = S["vertices"], S["faces"], S["part"]
Ts = np.load(f"{C.OUT}/poses.npz")["T"]
NF = 60
nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
samples = [[] for _ in range(len(F))]
zero = np.zeros((len(F), 3, 3))
for cam in C.CAMS:
    for f in range(NF):
        m, ig = PV.ignore_mask(cam, f)
        if m.sum() < 50:
            continue
        img = cv2.imread(f"out/clip/frames/{cam}/{f:04d}.jpg")[..., ::-1].astype(np.float32) / 255
        _, cov, buf = PV.render(V, F, zero, cam, f, Ts[f], bary=False)
        good = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        good &= ~(cv2.dilate(ig.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0)
        sel = good & cov
        fi = buf[sel]
        if not len(fi):
            continue
        cen = transform(Ts[f], V[F].mean(1))
        vd = cams[f"{cam}_T"][f][:3, 3] - cen
        vd /= np.linalg.norm(vd, axis=1, keepdims=True)
        facing = ((nrm @ Ts[f][:3, :3].T) * vd).sum(1)
        cnt = np.bincount(fi, minlength=len(F))
        sm = np.stack([np.bincount(fi, img[sel][:, c], minlength=len(F)) for c in range(3)], 1)
        for k in np.nonzero((cnt >= 2) & (facing > 0.25))[0]:
            samples[k].append(sm[k] / cnt[k])
seen = np.array([len(s) > 0 for s in samples])
fc = np.zeros((len(F), 3))
for k in np.nonzero(seen)[0]:
    fc[k] = np.median(np.array(samples[k]), 0)
print("faces seen", seen.sum(), "/", len(F), flush=True)
vf = [[] for _ in range(len(V))]
for k, tri in enumerate(F):
    for v in tri:
        vf[v].append(k)
known = seen.copy()
for _ in range(300):
    if known.all():
        break
    vc = np.zeros((len(V), 3)); vn = np.zeros(len(V))
    for k in np.nonzero(known)[0]:
        vc[F[k]] += fc[k]; vn[F[k]] += 1
    new = known.copy()
    for k in np.nonzero(~known)[0]:
        if vn[F[k]].sum() > 0:
            fc[k] = vc[F[k]].sum(0) / vn[F[k]].sum(); new[k] = True
    known = new
if (part == 0).any():
    fc[(part == 0) & ~seen] = 0.35   # soleplate underside: never seen; dark metal
corner = np.zeros((len(F), 3, 3))
for k, tri in enumerate(F):
    for j, v in enumerate(tri):
        nb = [q for q in vf[v] if part[q] == part[k]]
        corner[k, j] = fc[nb].mean(0)
corner = np.clip(corner, 0, 1).astype(np.float32)
np.savez(f"{C.OUT}/iron_mesh.npz", vertices=V.astype(np.float32), faces=F.astype(np.int32), corner_srgb=corner,
         face_srgb=fc.astype(np.float32), part=part, seen=seen, part_names=np.array(["soleplate_bottom", "body", "handle"]))

res = {}
for cam in C.CAMS:
    os.makedirs(f"{C.OUT}/render/{cam}", exist_ok=True)
    os.makedirs(f"{C.OUT}/depth/{cam}", exist_ok=True)
    vals = []
    for f in range(NF):
        img, cov, buf = PV.render(V, F, corner, cam, f, Ts[f])
        # depth: camera z interpolated over the face (painter's order already resolved visibility)
        uv, d = C.cam_project(transform(Ts[f], V), cams[f"{cam}_K"], cams[f"{cam}_T"][f])
        dep = np.full(cov.shape, np.inf, np.float32)
        ys, xs = np.nonzero(cov)
        fi = buf[ys, xs]
        a, b, c = uv[F[fi, 0]], uv[F[fi, 1]], uv[F[fi, 2]]
        p = np.stack([xs, ys], 1) + 0.5
        v0, v1, v2 = b - a, c - a, p - a
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        den = np.where(np.abs(den) < 1e-9, 1e-9, den)
        w1 = np.clip((v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den, 0, 1)
        w2 = np.clip((v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den, 0, 1)
        w0 = np.clip(1 - w1 - w2, 0, 1)
        dz = d[F[fi]]
        dep[ys, xs] = (w0 * dz[:, 0] + w1 * dz[:, 1] + w2 * dz[:, 2]) / np.maximum(w0 + w1 + w2, 1e-6)
        np.save(f"{C.OUT}/depth/{cam}/{f:04d}.npy", dep.astype(np.float16))
        rgba = np.dstack([(img[..., ::-1] * 255 + 0.5).astype(np.uint8), cov.astype(np.uint8) * 255])
        rgba[~cov, :3] = 0
        cv2.imwrite(f"{C.OUT}/render/{cam}/{f:04d}.png", rgba)
        m, ig = PV.ignore_mask(cam, f)
        if m.sum() >= 50:
            vals.append(C.iou(cov, m, ig))
    res[cam] = dict(iou_mean=float(np.mean(vals)), iou_min=float(np.min(vals)), n=len(vals))
    print(cam, res[cam], flush=True)
json.dump(res, open(f"{C.OUT}/layer_metrics.json", "w"), indent=1)
PV.preview("layer")
open(f"{C.OUT}/RENDER_DONE", "w").write("iron layer v1 (baked mesh raster)\n")
