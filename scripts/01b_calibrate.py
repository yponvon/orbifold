"""Stage 01b: one fixed pose per tripod camera, solved over the whole recording.

Both exo cameras are physically static (0 px image shift over 57 s), but the recorded
exocam2 extrinsics wander ~0.23 m / 14 deg. Per frame, each extrinsic still projects the
joints onto the person, so we use those projections as 2D observations and solve a single
PnP pose per camera against the world-space joints from all frames.

    uv run scripts/01b_calibrate.py
Writes out/calib/static_cams.npz: <cam>_T (4,4) world pose (ros_body axes), <cam>_median_px.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from orbifold.geometry import CONVENTIONS, make_T, project, transform
from orbifold.mcap_io import iter_messages

OUT = Path("out/calib")
STATIC_CAMS = ["exocam1", "exocam2"]
EVERY = 5  # use every 5th frame


def median_err(T: np.ndarray, P: np.ndarray, uv: np.ndarray, K: np.ndarray) -> float:
    """Median pixel distance between projections through T and the observed uv."""
    uv_hat, front = project(P, K, T, "ros_body")
    err = np.linalg.norm(uv_hat - uv, axis=1)
    return float(np.median(np.where(front & np.isfinite(err), err, 1e6)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/clip.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    path, cams, trk = cfg["mcap"], cfg["cameras"], cfg["trackers"]

    K = {}
    for name in STATIC_CAMS:
        for _, _, m in iter_messages(path, topics=[cams[name]["intrinsics"]]):
            pc = m.sensor_intrinsics[0].pinhole_camera
            K[name] = np.array([[pc.fx, 0, pc.cx], [0, pc.fy, pc.cy], [0, 0, 1]])

    topics = list(trk.values()) + [cams[n]["extrinsics"] for n in STATIC_CAMS]
    by_t: dict[int, dict] = {}
    for topic, t, m in iter_messages(path, topics=topics):
        by_t.setdefault(t, {})[topic] = m
    stamps = sorted(t for t, d in by_t.items() if len(d) == len(topics))[::EVERY]

    obj = {n: [] for n in STATIC_CAMS}
    img = {n: [] for n in STATIC_CAMS}
    per_frame_T = {n: [] for n in STATIC_CAMS}
    for t in stamps:
        d = by_t[t]
        pelvis = d[trk["upperbody"]].trackers[0].pose
        T_wp = make_T(pelvis.position_meters_xyz, pelvis.orientation_xyzw)
        pts = [T_wp[:3, 3]] + [
            transform(T_wp, np.asarray(tr.pose.position_meters_xyz)[None])[0]
            for topic in (trk["upperbody"], trk["lowerbody"])
            for tr in d[topic].trackers
            if tr.name != "pelvis"
        ]
        pts = np.asarray(pts)
        for n in STATIC_CAMS:
            e = d[cams[n]["extrinsics"]].sensor_extrinsics[0]
            T = T_wp @ make_T(e.translation_meters_xyz, e.rotation_xyzw)
            uv, front = project(pts, K[n], T, "ros_body")
            obj[n].append(pts[front])
            img[n].append(uv[front])
            per_frame_T[n].append(T)

    M = np.eye(4)
    M[:3, :3] = CONVENTIONS["ros_body"]  # columns: OpenCV axes in the ros_body camera frame
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for n in STATIC_CAMS:
        P, uv = np.concatenate(obj[n]), np.concatenate(img[n])
        # Initial guess: the median-position per-frame pose.
        Ts = np.array(per_frame_T[n])
        T0 = Ts[np.argmin(np.linalg.norm(Ts[:, :3, 3] - np.median(Ts[:, :3, 3], 0), axis=1))]
        # Candidate A: average of the per-frame poses (chordal mean rotation).
        U, _, Vt = np.linalg.svd(Ts[:, :3, :3].mean(0))
        T_avg = np.eye(4)
        T_avg[:3, :3], T_avg[:3, 3] = U @ Vt, np.median(Ts[:, :3, 3], 0)
        # Candidate B: PnP against all observations, robust to the wandering frames.
        V0 = np.linalg.inv(T0 @ M)  # world -> OpenCV camera
        rvec, _ = cv2.Rodrigues(V0[:3, :3])
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            P, uv, K[n], None, rvec, V0[:3, 3].copy(), useExtrinsicGuess=True,
            reprojectionError=25.0, iterationsCount=500,
        )  # fmt: skip
        candidates = {"average": T_avg}
        if ok:
            rvec, tvec = cv2.solvePnPRefineLM(P[inl[:, 0]], uv[inl[:, 0]], K[n], None, rvec, tvec)
            V = np.eye(4)
            V[:3, :3], V[:3, 3] = cv2.Rodrigues(rvec)[0], tvec[:, 0]
            candidates["pnp"] = np.linalg.inv(V) @ M.T

        errs = {k: median_err(T, P, uv, K[n]) for k, T in candidates.items()}
        best = min(errs, key=errs.get)
        T_world_cam = candidates[best]
        moved = np.linalg.norm(T_world_cam[:3, 3] - Ts[:, :3, 3], axis=1)
        print(
            f"{n}: {len(P)} obs | median reproj vs per-frame: "
            + ", ".join(f"{k} {v:.1f}px" for k, v in errs.items())
            + f" -> {best} | per-frame position deviates median {np.median(moved) * 100:.0f} cm"
        )
        result[f"{n}_T"] = T_world_cam
        result[f"{n}_median_px"] = errs[best]
    np.savez(OUT / "static_cams.npz", **result)
    print(f"wrote {OUT / 'static_cams.npz'}")


if __name__ == "__main__":
    main()
