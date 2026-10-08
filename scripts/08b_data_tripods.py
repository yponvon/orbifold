"""Stage 08b (data): extra tripod anchor frames for the sharp room model.

out/room/data.npz samples the tripods every 10th frame. The sharp trainer wants them every
3rd frame (they are well calibrated and pin the geometry the head-camera poses align to).
This writes only the *missing* tripod frames (i % 3 == 0 and i % 10 != 0), with the same
person masking as stage 07, to out/room_sharp/data_tripod3.npz.

    uv run scripts/08b_data_tripods.py
"""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from orbifold.geometry import make_T, transform
from orbifold.mcap_io import decode_image, iter_messages

_spec = importlib.util.spec_from_file_location("d07", Path(__file__).with_name("07_room_data.py"))
d07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d07)

OUT = Path("out/room_sharp")
TRIPODS = ["exocam1", "exocam2"]
STRIDE, OLD_STRIDE = 3, 10


def main() -> None:
    cfg = yaml.safe_load(open("configs/clip.yaml"))
    path, cams, trk = cfg["mcap"], cfg["cameras"], cfg["trackers"]
    static = np.load(d07.STATIC)
    K = {}
    for n in TRIPODS:
        for _, _, m in iter_messages(path, topics=[cams[n]["intrinsics"]]):
            pc = m.sensor_intrinsics[0].pinhole_camera
            K[n] = np.array([[pc.fx, 0, pc.cx], [0, pc.fy, pc.cy], [0, 0, 1]])
    seg = YOLO("yolo11x-seg.pt")
    # Same topic set as stage 07 so the "complete frame" stamp indices agree.
    topics = list(trk.values()) + [cams[n][k] for n in d07.CAMS for k in ("image", "extrinsics")]
    frames: dict[int, dict] = {}
    for topic, t, m in iter_messages(path, topics=topics):
        frames.setdefault(t, {})[topic] = m
    stamps = sorted(t for t, d in frames.items() if len(d) == len(topics))

    images, masks, Ks, Ts, cam_id, frame_idx, st = [], [], [], [], [], [], []
    names = None
    for i, t in enumerate(stamps):
        d = frames.pop(t)
        if i % STRIDE or i % OLD_STRIDE == 0:
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
        for n in TRIPODS:
            T = static[f"{n}_T"]
            bgr = decode_image(d[cams[n]["image"]])
            h, w = bgr.shape[:2]
            person = np.zeros((h, w), np.uint8)
            res = seg.predict(bgr, classes=[0], conf=0.25, verbose=False, retina_masks=True)[0]
            if res.masks is not None:
                person |= (res.masks.data.any(0).cpu().numpy() > 0).astype(np.uint8)
            person |= d07.skeleton_mask(xyz, names, K[n], T, (h, w))
            person = cv2.dilate(person, np.ones((25, 25), np.uint8))
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            masks.append(1 - person)
            Ks.append(K[n].copy())
            Ts.append(T)
            cam_id.append(d07.CAMS.index(n))
            frame_idx.append(i)
            st.append(t)
        if i % 300 == 0:
            print(f"frame {i}/{len(stamps)}: {len(images)} extra tripod images", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / "data_tripod3.npz",
        images=np.array(images), masks=np.array(masks), K=np.array(Ks),
        T_world_cam=np.array(Ts), cam_id=np.array(cam_id), frame_idx=np.array(frame_idx),
        stamps_ns=np.array(st), cam_names=np.array(d07.CAMS),
    )  # fmt: skip
    print(f"wrote {OUT / 'data_tripod3.npz'}: {len(images)} images")


if __name__ == "__main__":
    main()
