"""Stage 07: training data for the room reconstruction (person removed).

Samples frames from the whole recording, masks out the person (YOLO segmentation union
a generous mask around the projected tracker skeleton, which also covers the iron in
hand and the wearer's own body in the head camera), and stores half-resolution images,
masks and camera poses.

    uv run scripts/07_room_data.py [--clip-seconds 2]

Writes out/room/data.npz:
  images (N,H,W,3) uint8 RGB, masks (N,H,W) uint8 (1 = room pixel to learn),
  K (N,3,3) at half res, T_world_cam (N,4,4) ros_body axes, cam_id (N,), frame_idx (N,)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from orbifold.geometry import make_T, project, transform
from orbifold.mcap_io import decode_image, iter_messages, summary
from orbifold.skeleton import bones

OUT = Path("out/room")
STATIC = Path("out/calib/static_cams.npz")
CAMS = ["exocam1", "exocam2", "headcam"]
EVERY = {"exocam1": 10, "exocam2": 10, "headcam": 2}  # sampling stride across the recording
SCALE = 1.0  # full resolution: half-res training made the headcam blurry (round 5)
LIMB_R, HAND_R, HEAD_R = 0.12, 0.18, 0.2  # mask radii in metres (hand covers the iron)


def skeleton_mask(xyz, names, K, T, shape) -> np.ndarray:
    """Draw thick 3D capsules around the bones, sized by perspective."""
    mask = np.zeros(shape, np.uint8)
    uv, front = project(xyz, K, T, "ros_body")
    depth = np.linalg.norm(xyz - T[:3, 3], axis=1)
    fx = K[0, 0]
    for a, b in bones(names):
        if front[a] and front[b]:
            r = fx * LIMB_R / max(min(depth[a], depth[b]), 0.1)
            pa, pb = uv[a].astype(int), uv[b].astype(int)
            cv2.line(mask, tuple(pa), tuple(pb), 1, max(int(2 * r), 1))
    for j, n in enumerate(names):
        if front[j]:
            rad = HEAD_R if n == "head" else HAND_R if n.endswith("middle_mcp") else LIMB_R
            r = int(fx * rad / max(depth[j], 0.1))
            cv2.circle(mask, tuple(uv[j].astype(int)), r, 1, -1)
    return mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/clip.yaml")
    ap.add_argument("--clip-seconds", type=float, default=2.0, help="headcam: keep every frame")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    path, cams, trk = cfg["mcap"], cfg["cameras"], cfg["trackers"]
    static = np.load(STATIC)
    t_start = summary(path)["start_ns"]
    clip_end = t_start + int((cfg["clip"]["start_s"] + args.clip_seconds) * 1e9)

    K = {}
    for n in CAMS:
        for _, _, m in iter_messages(path, topics=[cams[n]["intrinsics"]]):
            pc = m.sensor_intrinsics[0].pinhole_camera
            K[n] = np.array([[pc.fx, 0, pc.cx], [0, pc.fy, pc.cy], [0, 0, 1]])

    seg = YOLO("yolo11x-seg.pt")
    topics = list(trk.values()) + [cams[n][k] for n in CAMS for k in ("image", "extrinsics")]
    frames: dict[int, dict] = {}
    for topic, t, m in iter_messages(path, topics=topics):
        frames.setdefault(t, {})[topic] = m
    stamps = sorted(t for t, d in frames.items() if len(d) == len(topics))

    images, masks, Ks, Ts, cam_id, frame_idx = [], [], [], [], [], []
    names = None
    for i, t in enumerate(stamps):
        d = frames.pop(t)
        wanted = [n for n in CAMS if i % EVERY[n] == 0 or (n == "headcam" and t <= clip_end)]
        if not wanted:
            continue
        pelvis = d[trk["upperbody"]].trackers[0].pose
        T_wp = make_T(pelvis.position_meters_xyz, pelvis.orientation_xyzw)
        pts, nm = [], []
        for topic in trk.values():
            for tr in d[topic].trackers:
                if tr.name in nm:
                    continue
                p = np.asarray(tr.pose.position_meters_xyz)
                pts.append(T_wp[:3, 3] if tr.name == "pelvis" else transform(T_wp, p[None])[0])
                nm.append(tr.name)
        names = names or nm
        xyz = np.asarray(pts)
        for n in wanted:
            if f"{n}_T" in static:
                T = static[f"{n}_T"]
            else:
                e = d[cams[n]["extrinsics"]].sensor_extrinsics[0]
                T = T_wp @ make_T(e.translation_meters_xyz, e.rotation_xyzw)
            bgr = decode_image(d[cams[n]["image"]])
            h, w = bgr.shape[:2]
            person = np.zeros((h, w), np.uint8)
            res = seg.predict(bgr, classes=[0], conf=0.25, verbose=False, retina_masks=True)[0]
            if res.masks is not None:
                person |= (res.masks.data.any(0).cpu().numpy() > 0).astype(np.uint8)
            person |= skeleton_mask(xyz, names, K[n], T, (h, w))
            person = cv2.dilate(person, np.ones((25, 25), np.uint8))
            small = cv2.resize(bgr, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
            keep = cv2.resize(1 - person, small.shape[1::-1], interpolation=cv2.INTER_NEAREST)
            Ks_ = K[n].copy()
            Ks_[:2] *= SCALE
            images.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            masks.append(keep)
            Ks.append(Ks_)
            Ts.append(T)
            cam_id.append(CAMS.index(n))
            frame_idx.append(i)
        if i % 300 == 0:
            print(f"frame {i}/{len(stamps)}: {len(images)} training images so far")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / "data.npz",
        images=np.array(images), masks=np.array(masks), K=np.array(Ks),
        T_world_cam=np.array(Ts), cam_id=np.array(cam_id), frame_idx=np.array(frame_idx),
        cam_names=np.array(CAMS),
    )  # fmt: skip
    # A few mask previews to eyeball.
    for k in np.linspace(0, len(images) - 1, 6).astype(int):
        vis = images[k].copy()
        vis[masks[k] == 0] = (vis[masks[k] == 0] * 0.3).astype(np.uint8)
        cv2.imwrite(str(OUT / f"mask_preview_{k:04d}.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    counts = np.bincount(cam_id, minlength=len(CAMS))
    print(
        f"wrote {OUT / 'data.npz'}: {len(images)} images "
        + str(dict(zip(CAMS, counts, strict=True)))
    )
    print(f"room pixels kept: {np.mean([m.mean() for m in masks]):.0%}")


if __name__ == "__main__":
    main()
