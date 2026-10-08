"""Stage 02: extract a clip — frames, cameras and world-space joints — into out/clip/.

    uv run scripts/02_extract.py [--config configs/clip.yaml] [--duration 2]

Writes:
  out/clip/frames/<cam>/0000.jpg ...   real frames
  out/clip/cameras.npz                 per cam: K (3,3), T_world_cam (F,4,4), width, height
  out/clip/joints.npz                  names (J,), xyz (F,J,3) world, bad (F,J) quality flags,
                                       right_wrist_T (F,4,4) world pose of the right wrist
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from orbifold.geometry import make_T, transform
from orbifold.mcap_io import decode_image, iter_messages, summary

OUT = Path("out/clip")
STATIC = Path("out/calib/static_cams.npz")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/clip.yaml")
    ap.add_argument("--duration", type=float, help="override clip.duration_s")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    path, cams = cfg["mcap"], cfg["cameras"]
    t0 = summary(path)["start_ns"] + int(cfg["clip"]["start_s"] * 1e9)
    t1 = t0 + int((args.duration or cfg["clip"]["duration_s"]) * 1e9)

    # Intrinsics are published once at the start of the recording.
    K, size = {}, {}
    intr = {c["intrinsics"]: name for name, c in cams.items()}
    for topic, _, m in iter_messages(path, topics=list(intr)):
        pc = m.sensor_intrinsics[0].pinhole_camera
        K[intr[topic]] = np.array([[pc.fx, 0, pc.cx], [0, pc.fy, pc.cy], [0, 0, 1]])
        size[intr[topic]] = (pc.image_width, pc.image_height)

    # Per-frame messages, grouped by timestamp (all streams share log_time).
    tracker_topics = list(cfg["trackers"].values())
    topics = tracker_topics + [c[k] for c in cams.values() for k in ("image", "extrinsics")]
    by_t: dict[int, dict] = {}
    for topic, t, m in iter_messages(path, topics=topics, start_ns=t0, end_ns=t1):
        by_t.setdefault(t, {})[topic] = m
    stamps = sorted(t for t, d in by_t.items() if len(d) == len(topics))
    print(f"{len(stamps)} complete frames in clip")

    names: list[str] = []
    xyz, bad, T_cam, wrist_T = [], [], {n: [] for n in cams}, []
    for i, t in enumerate(stamps):
        d = by_t[t]
        pelvis = d[cfg["trackers"]["upperbody"]].trackers[0]
        assert pelvis.name == "pelvis" and pelvis.pose.source_frame_id == "world"
        T_wp = make_T(pelvis.pose.position_meters_xyz, pelvis.pose.orientation_xyzw)

        # Joints: pelvis in world, everything else pelvis-relative. Hand topics repeat the
        # wrists from upperbody, so keep the first occurrence of each name.
        frame: dict[str, tuple] = {}
        for topic in tracker_topics:
            for tr in d[topic].trackers:
                if tr.name in frame:
                    continue
                p = np.asarray(tr.pose.position_meters_xyz)
                world = T_wp[:3, 3] if tr.name == "pelvis" else transform(T_wp, p[None])[0]
                is_bad = any(
                    k == "is_quality_bad" and v.bool_value for k, v in tr.metadata.fields.items()
                )
                frame[tr.name] = (world, is_bad)
                if tr.name == "right_wrist" and topic == cfg["trackers"]["hand_right"]:
                    wrist_T.append(T_wp @ make_T(p, tr.pose.orientation_xyzw))
        if not names:
            names = list(frame)
        xyz.append([frame[n][0] for n in names])
        bad.append([frame[n][1] for n in names])

        for name, c in cams.items():
            e = d[c["extrinsics"]].sensor_extrinsics[0]
            T_cam[name].append(T_wp @ make_T(e.translation_meters_xyz, e.rotation_xyzw))
            out_dir = OUT / "frames" / name
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"{i:04d}.jpg"), decode_image(d[c["image"]]))

    np.savez(
        OUT / "joints.npz",
        names=np.array(names),
        xyz=np.array(xyz),
        bad=np.array(bad),
        right_wrist_T=np.array(wrist_T),
        stamps_ns=np.array(stamps),
    )
    # Tripod cameras: replace the wandering per-frame extrinsics with the fixed pose
    # solved over the whole recording by 01b_calibrate.py (if it has been run).
    if STATIC.exists():
        static = np.load(STATIC)
        for n in cams:
            if f"{n}_T" in static:
                T_cam[n] = [static[f"{n}_T"]] * len(stamps)
                print(f"{n}: using fixed pose from {STATIC}")

    np.savez(
        OUT / "cameras.npz",
        **{f"{n}_K": K[n] for n in cams},
        **{f"{n}_T": np.array(T_cam[n]) for n in cams},
        **{f"{n}_size": np.array(size[n]) for n in cams},
        names=np.array(list(cams)),
    )
    print(f"{len(names)} joints, {len(cams)} cameras -> {OUT}")


if __name__ == "__main__":
    main()
