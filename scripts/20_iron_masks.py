"""Iron stage 1: SAM 2.1 iron (and occluding right-hand) masks in all cameras.

Prompt: a box from the projected rough 3D iron volume under the right palm, positive
points on aqua pixels inside it, negative point on the projected palm. The hand mask is
kept as an "ignore" region for silhouette fitting (the hand occludes the handle).
Writes out/assets/iron/masks/<cam>/<f>.png (255 iron), hand/<cam>/<f>.png, masks_check.jpg.
"""

import json
import os

import cv2
import numpy as np
from ultralytics import SAM

from orbifold.geometry import project

OUT = "out/assets/iron"
CAMS = ["headcam", "exocam1", "exocam2"]
cams = np.load("out/clip/cameras.npz")
J = np.load("out/clip/joints.npz")
idx = {n: i for i, n in enumerate(J["names"])}
xyz = J["xyz"]
sam = SAM(f"{OUT}/weights/sam2.1_b.pt")


def palm_of(f):
    k = xyz[f, [idx["right_index_mcp"], idx["right_middle_mcp"], idx["right_pinky_mcp"]]].mean(0)
    return (xyz[f, idx["right_wrist"]] + k) / 2, k


def aqua(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 76) & (h <= 100) & (s >= 45) & (v >= 90)


def best(res, prefer=None):
    if res[0].masks is None:
        return None
    m = res[0].masks.data.cpu().numpy().astype(bool)
    if m.ndim == 3 and len(m) > 1 and prefer is not None:
        return m[np.argmax([(x & prefer).sum() / (x.sum() + 1) for x in m])]
    return m[0]


def clean(m):
    # largest connected component, holes filled (window / accents are part of the iron)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1:
        return m
    m = lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    ff = m.astype(np.uint8).copy()
    pad = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 1) if not m[0, 0] else None
    return m | (ff == 0)


stats = {}
for cam in CAMS:
    os.makedirs(f"{OUT}/masks/{cam}", exist_ok=True)
    os.makedirs(f"{OUT}/hand/{cam}", exist_ok=True)
    K, Ts = cams[f"{cam}_K"], cams[f"{cam}_T"]
    for f in range(60):
        img = cv2.imread(f"out/clip/frames/{cam}/{f:04d}.jpg")
        H, W = img.shape[:2]
        palm, knuck = palm_of(f)
        g = np.mgrid[-1:2:2, -1:2:2, 0:2].reshape(3, -1).T.astype(float)
        box3 = np.stack(
            [
                palm[0] + 0.17 * g[:, 0],
                palm[1] + 0.17 * g[:, 1],
                np.where(g[:, 2] > 0, palm[2] + 0.03, 0.46),
            ],
            1,
        )
        uv, _ = project(box3, K, Ts[f], "ros_body")
        x0, y0 = np.clip(uv.min(0), 0, [W - 1, H - 1]).astype(int)
        x1, y1 = np.clip(uv.max(0), 0, [W - 1, H - 1]).astype(int)
        # crop for SAM resolution: square, >= 400 px, centred on the box
        cx, cy, r = (x0 + x1) // 2, (y0 + y1) // 2, max(200, int(0.75 * max(x1 - x0, y1 - y0)))
        cx0, cy0 = max(0, cx - r), max(0, cy - r)
        cx1, cy1 = min(W, cx + r), min(H, cy + r)
        crop = img[cy0:cy1, cx0:cx1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        aq = aqua(hsv)
        boxmask = np.zeros(crop.shape[:2], bool)
        boxmask[max(0, y0 - cy0) : y1 - cy0, max(0, x0 - cx0) : x1 - cx0] = True
        aq &= boxmask
        aq = cv2.morphologyEx(aq.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        puv, _ = project(np.array([palm, knuck]), K, Ts[f], "ros_body")
        puv = puv - [cx0, cy0]
        iron = np.zeros((H, W), bool)
        hand = np.zeros((H, W), bool)
        n_aq = int(aq.sum())
        if n_aq >= 30:
            ys, xs = np.nonzero(aq)
            # positives: aqua pixels nearest to the aqua centroid and the 2 extremes along PCA
            c = np.array([xs.mean(), ys.mean()])
            P = np.stack([xs, ys], 1).astype(float)
            ev = np.linalg.svd(P - c, full_matrices=False)[2][0]
            t = (P - c) @ ev
            pos = [
                P[np.argmin(((P - c) ** 2).sum(1))],
                P[np.argmin(t)] * 0.8 + c * 0.2,
                P[np.argmax(t)] * 0.8 + c * 0.2,
            ]
            pos = [P[np.argmin(((P - p) ** 2).sum(1))] for p in pos]
            neg = [p for p in puv if 0 <= p[0] < crop.shape[1] and 0 <= p[1] < crop.shape[0]]
            # tight bbox around aqua pixels, grown to cover white top
            ax0, ay0, ax1, ay1 = xs.min(), ys.min(), xs.max(), ys.max()
            pad = 0.35 * max(ax1 - ax0, ay1 - ay0)
            bb = [
                max(0, ax0 - pad),
                max(0, ay0 - pad),
                min(crop.shape[1], ax1 + pad),
                min(crop.shape[0], ay1 + pad),
            ]
            pts = [list(map(float, p)) for p in pos + neg]
            lab = [1] * len(pos) + [0] * len(neg)
            m = best(sam(crop, points=[pts], labels=[lab], verbose=False), aq)
            if m is not None:
                m = clean(m & boxmask)
                iron[cy0:cy1, cx0:cx1] = m
            # hand: positive at palm & knuckles, negative at iron centroid
            hp = [list(map(float, p)) for p in neg] + [list(map(float, pos[0]))]
            if neg:
                hm = best(sam(crop, points=[hp], labels=[[1] * len(neg) + [0]], verbose=False))
                if hm is not None:
                    hand[cy0:cy1, cx0:cx1] = hm & ~iron[cy0:cy1, cx0:cx1]
        if iron.any():  # holes of the iron that open onto the occluding hand are iron too
            iron = clean(iron | hand) & ~hand
        cv2.imwrite(f"{OUT}/masks/{cam}/{f:04d}.png", iron.astype(np.uint8) * 255)
        cv2.imwrite(f"{OUT}/hand/{cam}/{f:04d}.png", hand.astype(np.uint8) * 255)
        stats[f"{cam}/{f}"] = dict(
            aqua=n_aq,
            iron=int(iron.sum()),
            hand=int(hand.sum()),
            crop=[int(cx0), int(cy0), int(cx1), int(cy1)],
        )
    print(
        cam,
        "iron px median",
        np.median([stats[f"{cam}/{f}"]["iron"] for f in range(60)]),
        "empty",
        sum(stats[f"{cam}/{f}"]["iron"] == 0 for f in range(60)),
        flush=True,
    )
json.dump(stats, open(f"{OUT}/masks/stats.json", "w"))

# montage
rows = []
for f in [0, 10, 20, 30, 40, 50, 59]:
    row = []
    for cam in CAMS:
        img = cv2.imread(f"out/clip/frames/{cam}/{f:04d}.jpg")
        m = cv2.imread(f"{OUT}/masks/{cam}/{f:04d}.png", 0) > 0
        h = cv2.imread(f"{OUT}/hand/{cam}/{f:04d}.png", 0) > 0
        o = img.copy()
        o[m] = (0.5 * o[m] + [0, 0, 127]).astype(np.uint8)
        o[h] = (0.5 * o[h] + [127, 0, 0]).astype(np.uint8)
        x0, y0, x1, y1 = stats[f"{cam}/{f}"]["crop"]
        c = cv2.resize(o[y0:y1, x0:x1], (400, 400))
        cv2.putText(c, f"{cam} {f}", (8, 30), 0, 0.9, (0, 255, 255), 2)
        row.append(c)
    rows.append(np.hstack(row))
cv2.imwrite(f"{OUT}/masks_check.jpg", np.vstack(rows))
