"""Rigid transforms and pinhole projection."""

import numpy as np

# Rotation taking points from an OpenCV camera frame (x right, y down, z forward)
# into another camera convention. Selected by the reprojection check in stage 02.
CONVENTIONS = {
    "opencv": np.eye(3),
    "opengl": np.diag([1.0, -1.0, -1.0]),  # x right, y up, z back (Blender)
    "ros_body": np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float),  # x fwd, y left, z up
}


def quat_to_mat(xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(xyzw, float) / np.linalg.norm(xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def make_T(xyz, xyzw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(xyzw)
    T[:3, 3] = xyz
    return T


def transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def project(
    pts_world: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray, convention: str
) -> tuple[np.ndarray, np.ndarray]:
    """World points → pixel coords (N,2) and a mask of points in front of the camera."""
    p_cam = transform(np.linalg.inv(T_world_cam), pts_world)
    p_cv = p_cam @ CONVENTIONS[convention]  # into OpenCV axes
    front = p_cv[:, 2] > 1e-3
    uv = (p_cv @ K.T)[:, :2] / np.maximum(p_cv[:, 2:3], 1e-3)
    return uv, front
