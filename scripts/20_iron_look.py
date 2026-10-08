"""Iron stage 0: crops around the projected right palm in every camera (inspection only)."""

import cv2
import numpy as np
from orbifold.geometry import project

OUT = "out/assets/iron"
cams = np.load("out/clip/cameras.npz")
J = np.load("out/clip/joints.npz")
names = list(J["names"])
xyz = J["xyz"]
idx = {n: i for i, n in enumerate(names)}
rows = []
for f in [0, 15, 30, 45, 59]:
    row = []
    for cam in ["headcam", "exocam1", "exocam2"]:
        img = cv2.imread(f"out/clip/frames/{cam}/{f:04d}.jpg")
        palm = (
            xyz[f, idx["right_wrist"]]
            + xyz[
                f, [idx["right_index_mcp"], idx["right_middle_mcp"], idx["right_pinky_mcp"]]
            ].mean(0)
        ) / 2
        pts = np.array([palm, palm - [0, 0, 0.15]])
        uv, fr = project(pts, cams[f"{cam}_K"], cams[f"{cam}_T"][f], "ros_body")
        u, v = uv[0].astype(int)
        for p in uv.astype(int):
            cv2.circle(img, tuple(p), 8, (0, 0, 255), -1)
        H = 500
        x0, y0 = np.clip(u - H, 0, 1920 - 2 * H), np.clip(v - H // 2 - 100, 0, 1080 - H - 100)
        crop = img[y0 : y0 + H + 100, x0 : x0 + 2 * H]
        cv2.putText(
            crop, f"{cam} f{f} uv={u},{v} front={fr[0]}", (10, 40), 0, 1.2, (0, 255, 255), 3
        )
        row.append(cv2.resize(crop, (500, 300)))
    rows.append(np.hstack(row))
cv2.imwrite(f"{OUT}/look.jpg", np.vstack(rows))
print("palm f0", palm)
