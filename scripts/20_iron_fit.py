"""Iron stage 3: 6-DoF pose per frame from multi-view silhouettes.

Soleplate is kept parallel to the bed (roll = pitch = 0), so each frame has (x, y, yaw,
lift) with z = bed_z + lift, lift >= 0. Globals (bed_z, planform scale sxy) are grid-searched.
Fitted against convex-hull-filled SAM masks with the hand as an ignore region, in headcam
and exocam1 (exocam2 sees the iron only through gaps around the body; weight 0).
Writes out/assets/iron/poses.npz.
"""

import importlib
import json
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

sys.path.insert(0, "scripts")
C = importlib.import_module("20_iron_common")

S = 1.0  # full resolution (original frames)
WCAM = {"headcam": 1.0, "exocam1": 1.0, "exocam2": 0.0}
cams = np.load("out/clip/cameras.npz")
J = np.load("out/clip/joints.npz")
idx = {n: i for i, n in enumerate(J["names"])}
xyz = J["xyz"]
NF = 60
V0, F, part = C.build_mesh()
Vs = V0[::3]
stats = json.load(open(f"{C.OUT}/masks/stats.json"))

kn = xyz[:, [idx["right_index_mcp"], idx["right_middle_mcp"], idx["right_pinky_mcp"]]].mean(1)
palm = (xyz[:, idx["right_wrist"]] + kn) / 2
hand_dir = kn[:, :2] - xyz[:, idx["right_wrist"], :2]
yaw_hand = np.arctan2(hand_dir[:, 1], hand_dir[:, 0])

# targets at half resolution, cropped
tgt = {}
for cam in C.CAMS:
    for f in range(NF):
        m, h = C.load_masks(cam, f)
        x0, y0, x1, y1 = stats[f"{cam}/{f}"]["crop"]
        pad = 150
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(1920, x1 + pad), min(1080, y1 + pad)
        t = m & ~h  # the traced SAM mask itself is the ground truth
        sl = (slice(y0, y1), slice(x0, x1))
        tt = (
            cv2.resize(t[sl].astype(np.uint8), None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST)
            > 0
        )
        ig = (
            cv2.resize(h[sl].astype(np.uint8), None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST)
            > 0
        )
        tgt[cam, f] = (tt, ig, np.array([x0, y0]), m.sum() > 0)


def sil_iou(cam, f, T, Vs_=None):
    tt, ig, o, valid = tgt[cam, f]
    if not valid:
        return np.nan
    uv, d = C.cam_project(
        C.transform(T, Vs if Vs_ is None else Vs_), cams[f"{cam}_K"], cams[f"{cam}_T"][f]
    )
    uv = ((uv - o) * S).astype(np.int32)
    pred = np.zeros_like(tt, np.uint8)
    if (d > 0.05).all():
        cv2.fillConvexPoly(pred, cv2.convexHull(uv), 1)
    return C.iou(pred > 0, tt, ig)


C.transform = importlib.import_module("orbifold.geometry").transform


def loss(p, f, bed, Vsc, prev=None, prior_xy=None):
    x, y, yaw, lift = p
    T = C.pose_T(x, y, bed + max(lift, 0), yaw)
    s, w = 0.0, 0.0
    for cam, wc in WCAM.items():
        if wc > 0:
            v = sil_iou(cam, f, T, Vsc)
            if not np.isnan(v):
                s += wc * v
                w += wc
    L = -(s / w if w else 0)
    L += 0.5 * abs(min(lift, 0)) / 0.01 + 0.02 * max(lift, 0) / 0.01
    # palm clearance is NOT enforced: the traced footage is ground truth (see report)
    if prior_xy is not None:
        L += 0.02 * (np.linalg.norm(np.array([x, y]) - prior_xy) / 0.05) ** 2
    if prev is not None:
        L += 0.02 * (np.linalg.norm(np.array([x, y]) - prev[:2]) / 0.02) ** 2
        L += 0.02 * (np.angle(np.exp(1j * (yaw - prev[2]))) / 0.1) ** 2
    return L


def fit_frame(f, bed, Vsc, starts, prev=None, prior_xy=None):
    best = None
    for p0 in starts:
        r = minimize(
            loss,
            p0,
            args=(f, bed, Vsc, prev, prior_xy),
            method="Nelder-Mead",
            options=dict(
                initial_simplex=p0 + np.vstack([np.zeros(4), np.diag([0.04, 0.04, 0.4, 0.01])]),
                maxiter=400,
                xatol=1e-4,
                fatol=1e-4,
            ),
        )
        if best is None or r.fun < best.fun:
            best = r
    return best


def starts_for(f, extra=()):
    base = [
        np.array([palm[f, 0], palm[f, 1], yaw_hand[f] + dy, 0.0]) for dy in (0, np.pi, 0.7, -0.7)
    ]
    return base + list(extra)


TOP = V0[:, 2].max()
LIFT_MAX = np.zeros(NF)


def set_lift_max(bed):
    LIFT_MAX[:] = np.maximum(0.0, palm[:, 2] - 0.015 - TOP - bed)


# --- stage A: globals on a frame subset ---------------------------------------------
sub = list(range(0, NF, 6))
best_g, gscore = None, -1
for sxy in (0.95, 1.01, 1.07):
    Vsc = C.scaled(Vs, sxy)
    for bed in (0.49, 0.505, 0.52, 0.535):  # footage decides; coordinator estimate 0.48-0.50
        set_lift_max(bed)
        sc = np.mean([-fit_frame(f, bed, Vsc, starts_for(f)).fun for f in sub])
        print(f"sxy {sxy} bed {bed}: {sc:.3f}", flush=True)
        if sc > gscore:
            gscore, best_g = sc, (sxy, bed)
sxy, bed = best_g
# refine
for sxy2 in (sxy - 0.03, sxy + 0.03):
    for bed2 in (bed - 0.005, bed, bed + 0.005):
        set_lift_max(bed2)
        Vsc = C.scaled(Vs, sxy2)
        sc = np.mean([-fit_frame(f, bed2, Vsc, starts_for(f)).fun for f in sub])
        if sc > gscore:
            gscore, best_g = sc, (sxy2, bed2)
sxy, bed = best_g
set_lift_max(bed)
print("globals sxy", sxy, "bed", bed, "score", gscore, flush=True)
Vsc = C.scaled(Vs, sxy)

# --- stage B: per-frame fit, then grasp offset, then refit with priors ---------------
P = np.zeros((NF, 4))
prev = None
for f in range(NF):
    extra = [prev.copy()] if prev is not None else []
    r = fit_frame(f, bed, Vsc, starts_for(f, extra) if prev is None else extra + starts_for(f)[:1])
    P[f] = r.x
    P[f, 3] = max(P[f, 3], 0)
    prev = P[f]
# constant grasp offset: iron pose relative to the horizontal palm frame
rel = []
for f in range(NF):
    c, s = np.cos(yaw_hand[f]), np.sin(yaw_hand[f])
    d = P[f, :2] - palm[f, :2]
    rel.append(
        [c * d[0] + s * d[1], -s * d[0] + c * d[1], np.angle(np.exp(1j * (P[f, 2] - yaw_hand[f])))]
    )
rel = np.array(rel)
G = np.array([np.median(rel[:, 0]), np.median(rel[:, 1]), np.angle(np.exp(1j * rel[:, 2]).mean())])
print(
    "grasp offset (palm frame dx, dy, dyaw):", G.round(4), "spread", rel.std(0).round(4), flush=True
)
for it in range(2):
    for f in range(NF):
        c, s = np.cos(yaw_hand[f]), np.sin(yaw_hand[f])
        pxy = palm[f, :2] + [c * G[0] - s * G[1], s * G[0] + c * G[1]]
        nb = P[max(f - 1, 0)] if f > 0 else None
        r = fit_frame(
            f, bed, Vsc, [P[f].copy(), np.r_[pxy, yaw_hand[f] + G[2], 0.0]], prev=nb, prior_xy=pxy
        )
        P[f] = r.x
        P[f, 3] = max(P[f, 3], 0)
# --- stage C: temporal smoothing ------------------------------------------------------
yaw_u = np.unwrap(P[:, 2])
Ps = P.copy()
for k in (0, 1, 3):
    Ps[:, k] = gaussian_filter1d(P[:, k], 1.0, mode="nearest")
Ps[:, 2] = gaussian_filter1d(yaw_u, 1.0, mode="nearest")
Ps[:, 3] = np.maximum(Ps[:, 3], 0)

# penetration: handle top + 1.5 cm must stay below the palm centre
Vfull = C.scaled(V0, sxy)
top = Vfull[:, 2].max()
T = np.stack([C.pose_T(Ps[f, 0], Ps[f, 1], bed + Ps[f, 3], Ps[f, 2]) for f in range(NF)])
clear = palm[:, 2] - (bed + Ps[:, 3] + top)
print(
    "handle top above soleplate",
    round(top, 4),
    "palm clearance cm: min %.2f median %.2f; frames <1.5cm: %d"
    % (100 * clear.min(), 100 * np.median(clear), (clear < 0.015).sum()),
)
for cam in C.CAMS:
    v = [sil_iou(cam, f, T[f]) for f in range(NF)]
    print(cam, "hull IoU (fit res) mean %.3f" % np.nanmean(v))
np.savez(
    f"{C.OUT}/poses.npz",
    T=T,
    params_raw=P,
    params=Ps,
    bed_z=bed,
    sxy=sxy,
    grasp=G,
    palm_clearance=clear,
    handle_top=top,
)
print(
    "lift cm max %.2f, yaw range deg %.1f..%.1f"
    % (100 * Ps[:, 3].max(), np.degrees(Ps[:, 2].min()), np.degrees(Ps[:, 2].max()))
)
