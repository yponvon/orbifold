"""Stage 03a: 2D body keypoints of the real person in the tripod views (for H1 alignment).

The lower-body tracker is unreliable (it bends a knee forward while the real legs are
straight), so the body fit also matches the image: COCO-17 keypoints from YOLO pose on
every clip frame of exocam1/exocam2.

    uv run --extra recon scripts/03a_detect_2d.py
Writes out/body/kp2d.npz: <cam>_xy (F,17,2), <cam>_conf (F,17).
"""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CLIP, BODY = Path("out/clip"), Path("out/body")
CAMS = ("exocam1", "exocam2")
COCO = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow",
        "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle",
        "r_ankle"]  # fmt: skip


def main() -> None:
    pose = YOLO("yolo11x-pose.pt")
    n = len(list((CLIP / "frames" / CAMS[0]).glob("*.jpg")))
    out = {}
    for cam in CAMS:
        xy, conf = np.zeros((n, 17, 2)), np.zeros((n, 17))
        for f in range(n):
            r = pose.predict(
                cv2.imread(str(CLIP / "frames" / cam / f"{f:04d}.jpg")), verbose=False
            )[0]
            if r.keypoints is not None and len(r.boxes):
                i = int(r.boxes.conf.argmax())
                xy[f], conf[f] = r.keypoints.xy[i].cpu().numpy(), r.keypoints.conf[i].cpu().numpy()
        out[f"{cam}_xy"], out[f"{cam}_conf"] = xy, conf
        legs = conf[:, 11:].mean()
        print(f"{cam}: {n} frames, mean leg keypoint confidence {legs:.2f}")
    BODY.mkdir(parents=True, exist_ok=True)
    np.savez(BODY / "kp2d.npz", names=np.array(COCO), **out)
    print(f"wrote {BODY / 'kp2d.npz'}")


if __name__ == "__main__":
    main()
