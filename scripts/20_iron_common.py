"""Shared iron helpers: procedural mesh, painter's-algorithm rasteriser, mask loading.

Iron local frame (metres): origin = centre of the soleplate's bottom face, +x towards the
tip, +y left, +z up (soleplate is the z=0 plane). Import with
importlib.import_module("20_iron_common") after adding scripts/ to sys.path.
"""

import cv2
import numpy as np

from orbifold.geometry import CONVENTIONS, transform

OUT = "out/assets/iron"
CAMS = ["headcam", "exocam1", "exocam2"]
L, WMAX = 0.25, 0.115  # length, max width of the soleplate
X_REAR = -0.105  # rear edge x (tip at X_REAR + L)
SHELL_TOP = 0.075
HANDLE_TOP = 0.107  # top of the handle above the soleplate bottom


def outline(n=80, s=1.0, cx=-0.01):
    """Closed teardrop ring (2n-2 points, CCW) of the soleplate, scaled by s about (cx, 0)."""
    t = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
    r = 0.08
    f = np.where(
        t < r,
        np.sqrt(np.clip(1 - ((r - t) / r) ** 2, 0, 1)),
        np.clip(1 - ((t - r) / (1 - r)) ** 2, 0, 1) ** 0.55,
    )
    x = X_REAR + L * t
    hw = WMAX / 2 * f
    ring = np.concatenate([np.stack([x, -hw], 1), np.stack([x[::-1], hw[::-1]], 1)[1:-1]])
    ring[:, 0] = cx + (ring[:, 0] - cx) * s
    ring[:, 1] *= s
    return ring


def _catmull(P, n):
    P = np.asarray(P, float)
    Pp = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(Pp) - 2):
        for u in np.linspace(0, 1, n, endpoint=False):
            p0, p1, p2, p3 = Pp[i - 1 : i + 3]
            out.append(
                0.5
                * (
                    2 * p1
                    + (-p0 + p2) * u
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * u**3
                )
            )
    out.append(P[-1])
    return np.array(out)


HANDLE_PATH = [
    (-0.088, 0.045),
    (-0.088, 0.073),
    (-0.074, 0.090),
    (-0.035, 0.095),
    (0.025, 0.094),
    (0.065, 0.086),
    (0.095, 0.068),
    (0.115, 0.045),
]
HANDLE_R = (0.016, 0.010)  # half-width (y), half-thickness (in the x-z plane)


def build_mesh(n=80):
    """Returns V (N,3), F (M,3), part (M,) 0 sole edge, 1 base, 2 shell, 3 top cap, 4 bottom, 5 handle."""
    V, F, part = [], [], []
    layers = [(0.0, 1.0, 0), (0.006, 1.0, 1), (0.03, 0.985, 2)]
    for i in range(1, 11):
        ph = i / 10 * np.pi / 2
        layers.append((0.03 + (SHELL_TOP - 0.03) * np.sin(ph), 0.985 - 0.40 * (1 - np.cos(ph)), 2))
    rings = []
    for z, s, _ in layers:
        r = outline(n, s)
        zz = np.full(len(r), z)
        if z > 0.03:  # shell slopes down towards the tip
            g = 1.12 - 0.42 * np.clip((r[:, 0] - X_REAR) / L, 0, 1)
            zz = 0.03 + (z - 0.03) * g
        V.append(np.c_[r, zz])
    m = len(V[0])
    off = np.cumsum([0] + [m] * len(layers))
    for k in range(len(layers) - 1):
        a, b = off[k], off[k + 1]
        for j in range(m):
            j2 = (j + 1) % m
            F += [(a + j, a + j2, b + j2), (a + j, b + j2, b + j)]
            part += [layers[k + 1][2]] * 2
    nv = off[len(layers)]
    # caps: top (fan to centre) and bottom
    top_c, bot_c = nv, nv + 1
    V.append(np.array([[-0.01, 0, V[-1][:, 2].mean()], [-0.01, 0, 0.0]]))
    a = off[len(layers) - 1]
    for j in range(m):
        F.append((a + j, a + (j + 1) % m, top_c))
        part.append(3)
        F.append((j, bot_c, (j + 1) % m))
        part.append(4)
    nv += 2
    # handle tube
    path = _catmull(HANDLE_PATH, 8)
    tang = np.gradient(path, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)  # in-plane normal (x,z)
    k = 16
    th = np.linspace(0, 2 * np.pi, k, endpoint=False)
    ring_pts = []
    for p, q in zip(path, nrm):
        for a_ in th:
            ring_pts.append(
                [
                    p[0] + HANDLE_R[1] * np.cos(a_) * q[0],
                    HANDLE_R[0] * np.sin(a_),
                    p[1] + HANDLE_R[1] * np.cos(a_) * q[1],
                ]
            )
    V.append(np.array(ring_pts))
    for i in range(len(path) - 1):
        for j in range(k):
            a, b = nv + i * k, nv + (i + 1) * k
            j2 = (j + 1) % k
            F += [(a + j, b + j, b + j2), (a + j, b + j2, a + j2)]
            part += [5, 5]
    for i, c in [(0, 0), (len(path) - 1, 1)]:  # end caps
        base = nv + i * k
        for j in range(1, k - 1):
            F.append((base, base + j + 1, base + j) if c == 0 else (base, base + j, base + j + 1))
            part.append(5)
    V = np.concatenate(V).astype(np.float64)
    return V, np.array(F, np.int32), np.array(part, np.int32)


def scaled(V, sxy):
    V = V.copy()
    V[:, :2] *= sxy
    return V


def pose_T(x, y, z, yaw):
    T = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = (x, y, z)
    return T


def cam_project(Pw, K, Tc):
    """World points -> (uv (N,2), depth (N,)) in OpenCV axes."""
    pc = transform(np.linalg.inv(Tc), Pw) @ CONVENTIONS["ros_body"]
    d = pc[:, 2]
    uv = (pc @ K.T)[:, :2] / np.maximum(d, 1e-3)[:, None]
    return uv, d


def raster(uv, depth, F, H, W, scale=1.0):
    """Face-id buffer (-1 = empty) by painter's algorithm (far faces first)."""
    buf = np.full((H, W), -1, np.int32)
    fd = depth[F].mean(1)
    ok = (depth[F] > 0.05).all(1)
    pts = np.round(uv[F] * scale * 4).astype(np.int32)  # 2 fractional bits
    for fi in np.argsort(-fd):
        if ok[fi]:
            cv2.fillConvexPoly(buf, pts[fi], int(fi), lineType=cv2.LINE_8, shift=2)
    return buf


def load_masks(cam, f):
    m = cv2.imread(f"{OUT}/masks/{cam}/{f:04d}.png", 0) > 0
    h = cv2.imread(f"{OUT}/hand/{cam}/{f:04d}.png", 0) > 0
    return m, h


def hull_fill(m):
    out = np.zeros_like(m, np.uint8)
    if m.any():
        ys, xs = np.nonzero(m)
        hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
        cv2.fillConvexPoly(out, hull, 1)
    return out > 0


def iou(pred, target, ignore):
    v = ~ignore
    inter = (pred & target & v).sum()
    uni = ((pred | target) & v).sum()
    return inter / uni if uni else np.nan


# ---------------------------------------------------------------------------------------
# v2 shape: a low, long wedge (loft of tapered cross-sections) + a closed bridge handle
# joined to the heel at the back and to the shell at the front (coordinator research pass).
# ---------------------------------------------------------------------------------------
def _interp(t, pts):
    pts = np.asarray(pts, float)
    return np.interp(t, pts[:, 0], pts[:, 1])


H_BODY = [(0, 0.088), (0.10, 0.086), (0.22, 0.068), (0.45, 0.056), (0.72, 0.042), (0.92, 0.028), (1, 0.018)]
HANDLE2 = [(-0.098, 0.070), (-0.090, 0.093), (-0.060, 0.104), (-0.010, 0.104), (0.035, 0.098),
           (0.068, 0.082), (0.090, 0.060), (0.104, 0.044)]
HANDLE2_R = (0.017, 0.012)


def _halfwidth(t):
    rear = 0.80 + 0.20 * np.sqrt(np.clip(t / 0.07, 0, 1))
    front = np.clip(1 - ((t - 0.07) / 0.93) ** 2, 0, 1) ** 0.55
    return WMAX / 2 * np.where(t < 0.07, rear, front * 1.0)


def build_mesh_v2(nx=56, nside=10, ntop=8):
    """Returns V, F, part (0 soleplate bottom, 1 body, 2 handle). Local frame as above."""
    t = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, nx))
    t[-1] = 0.995
    secs = []
    for ti in t:
        hw = max(_halfwidth(ti), 0.004)
        H = _interp(ti, H_BODY)
        tw = min(0.6 * hw, 0.016 + 0.012 * (1 - ti))      # top half-width
        r = min(tw, 0.4 * (H - 0.012))                   # top rounding radius
        x = X_REAR + L * ti
        pts = [(hw, 0.0), (hw, 0.010)]
        # slanted side from the soleplate edge to the shoulder of the top
        for k in range(1, nside + 1):
            u = k / nside
            ease = np.sin(u * np.pi / 2)                  # slightly convex side wall
            pts.append((hw + (tw - hw) * u ** 1.3, 0.010 + (H - r - 0.010) * ease))
        for k in range(1, ntop):
            a = np.pi * k / ntop
            pts.append((tw * np.cos(a), H - r + r * np.sin(a)))
        right = pts
        left = [(-y, z) for y, z in reversed(right)]
        ring = right + left
        secs.append([(x, y, z) for y, z in ring])
    secs = np.array(secs)                               # (nx, M, 3)
    nxs, M, _ = secs.shape
    V = secs.reshape(-1, 3)
    F, part = [], []
    for i in range(nxs - 1):
        for j in range(M):
            a, b = i * M + j, i * M + (j + 1) % M
            c, d = a + M, b + M
            seg_bottom = j == M - 1                      # ring closes along the bottom (z=0)
            F += [(a, c, d), (a, d, b)]
            part += [0, 0] if seg_bottom else [1, 1]
    # end caps (heel and tip): fan to the section centroid
    V = list(V)
    for i, flip in ((0, True), (nxs - 1, False)):
        c = secs[i].mean(0)
        ci = len(V); V.append(c)
        for j in range(M):
            a, b = i * M + j, i * M + (j + 1) % M
            F.append((a, b, ci) if flip else (b, a, ci)); part.append(1)
    V = np.array(V)
    nv = len(V)
    # handle tube
    path = _catmull(HANDLE2, 8)
    tang = np.gradient(path, axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    k = 16
    th = np.linspace(0, 2 * np.pi, k, endpoint=False)
    hp = []
    for p, q in zip(path, nrm):
        for a_ in th:
            hp.append([p[0] + HANDLE2_R[1] * np.cos(a_) * q[0], HANDLE2_R[0] * np.sin(a_), p[1] + HANDLE2_R[1] * np.cos(a_) * q[1]])
    V = np.vstack([V, hp])
    for i in range(len(path) - 1):
        for j in range(k):
            a, b = nv + i * k, nv + (i + 1) * k
            j2 = (j + 1) % k
            F += [(a + j, b + j, b + j2), (a + j, b + j2, a + j2)]; part += [2, 2]
    for i, c in [(0, 0), (len(path) - 1, 1)]:
        base = nv + i * k
        for j in range(1, k - 1):
            F.append((base, base + j + 1, base + j) if c == 0 else (base, base + j, base + j + 1)); part.append(2)
    F = np.array(F, np.int32)
    # make every triangle face outward (body is star-shaped about its axis; handle about its path)
    Vt = V[F]
    cen = Vt.mean(1)
    n = np.cross(Vt[:, 1] - Vt[:, 0], Vt[:, 2] - Vt[:, 0])
    part = np.array(part, np.int32)
    ref = np.where(part[:, None] == 2, np.c_[cen[:, 0], np.zeros(len(F)), np.full(len(F), 0.085)],
                   np.c_[cen[:, 0], np.zeros(len(F)), np.full(len(F), 0.03)])
    flip = (n * (cen - ref)).sum(1) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return V.astype(np.float64), F, part
