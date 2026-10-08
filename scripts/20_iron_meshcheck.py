"""Render the uncoloured procedural iron from 3 viewpoints (shape sanity check)."""

import sys
import importlib
import cv2
import numpy as np

sys.path.insert(0, "scripts")
C = importlib.import_module("20_iron_common")
V, F, part = C.build_mesh_v2()
print("V", V.shape, "F", F.shape, "bbox", V.min(0).round(3), V.max(0).round(3))
cols = np.array(
    [
        [180, 180, 180],
        [190, 200, 90],
        [200, 210, 120],
        [235, 235, 235],
        [90, 90, 90],
        [250, 250, 250],
    ]
)
imgs = []
for name, Tc in [("top", C.pose_T(0, 0, 0, 0)), ("side", None), ("3q", None)]:
    # simple look-at camera in ros_body convention (x fwd, y left, z up)
    eye = {
        "top": np.array([0, 0, 0.6]),
        "side": np.array([0, -0.6, 0.05]),
        "3q": np.array([-0.35, -0.4, 0.35]),
    }[name]
    tgt = np.array([0.0, 0, 0.05])
    fwd = (tgt - eye) / np.linalg.norm(tgt - eye)
    up0 = np.array([1.0, 0, 0]) if name == "top" else np.array([0, 0, 1.0])
    left = np.cross(up0, fwd)
    left /= np.linalg.norm(left)
    up = np.cross(fwd, left)
    Tc = np.eye(4)
    Tc[:3, :3] = np.stack([fwd, left, up], 1)
    Tc[:3, 3] = eye
    K = np.array([[900, 0, 300], [0, 900, 300], [0, 0, 1.0]])
    uv, d = C.cam_project(V, K, Tc)
    buf = C.raster(uv, d, F, 600, 600)
    # lambert shade
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    sh = 0.45 + 0.55 * np.abs(n @ (-fwd))
    img = np.full((600, 600, 3), 40, np.uint8)
    ok = buf >= 0
    img[ok] = (cols[part[buf[ok]]] * sh[buf[ok], None]).astype(np.uint8)
    cv2.putText(img, name, (10, 30), 0, 1, (0, 255, 255), 2)
    imgs.append(img)
cv2.imwrite(f"{C.OUT}/mesh_shape.jpg", np.hstack(imgs))
