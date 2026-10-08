"""Stage 11: automatic scoring of a round (replaces eyeballing videos).

Per camera and frame, on the composited sim frame (out/sim) vs the real frame:
  - silhouette IoU: real person mask (YOLO segmentation) vs our rendered alpha,
  - PSNR / SSIM / LPIPS inside the person region (union of both masks, dilated),
  - 2D keypoint error: YOLO pose on real and sim, mean pixel distance of the COCO
    keypoints both detect confidently (Fidelity),
and physics from the fitted body / iron: palm-into-iron penetration, foot-floor gap,
stance-foot sliding. Prints a table, the 10 worst frames per camera (by LPIPS), and
writes out/metrics.json.

    uv run --extra recon scripts/11_metrics.py [--round 6]
"""

import argparse
import json
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
from ultralytics import YOLO

CLIP, SIM, FG, OUT = Path("out/clip/frames"), Path("out/sim"), Path("out/render"), Path("out")
CAMS = ["headcam", "exocam1", "exocam2"]


def ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    s_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a**2
    s_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b**2
    s_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (s_a + s_b + c2))
    return float(m.mean(-1)[mask].mean())


def best_person(res):
    if res.keypoints is None or len(res.boxes) == 0:
        return None, None
    i = int(res.boxes.conf.argmax())
    return res.keypoints.xy[i].cpu().numpy(), res.keypoints.conf[i].cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="latest")
    args = ap.parse_args()
    seg, pose = YOLO("yolo11x-seg.pt"), YOLO("yolo11x-pose.pt")
    lp = lpips.LPIPS(net="alex", verbose=False).cuda()
    frames = sorted(int(p.stem) for p in (SIM / "exocam1").glob("*.png"))
    per_cam, worst = {}, {}
    for cam in CAMS:
        rows = []
        for f in frames:
            real = cv2.imread(str(CLIP / cam / f"{f:04d}.jpg"))
            sim = cv2.imread(str(SIM / cam / f"{f:04d}.png"))
            fg = cv2.imread(str(FG / cam / f"{f:04d}.png"), cv2.IMREAD_UNCHANGED)
            if real is None or sim is None or fg is None:
                continue
            r = seg.predict(real, classes=[0], verbose=False, retina_masks=True)[0]
            m_real = (
                r.masks.data.any(0).cpu().numpy()
                if r.masks is not None
                else np.zeros(real.shape[:2], bool)
            )
            m_sim = fg[..., 3] > 128
            union = m_real | m_sim
            iou = float((m_real & m_sim).sum() / max(union.sum(), 1))
            region = cv2.dilate(union.astype(np.uint8), np.ones((15, 15))) > 0
            if region.sum() < 100:
                continue
            mse = ((real.astype(float) - sim.astype(float)) ** 2)[region].mean()
            psnr = 10 * np.log10(255**2 / max(mse, 1e-6))
            ys, xs = np.nonzero(region)
            crop = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))

            def t(img, crop=crop):
                x = cv2.resize(img[crop], (256, 256))[..., ::-1].copy()
                return torch.from_numpy(x).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1

            with torch.no_grad():
                lp_val = float(lp(t(real), t(sim)))
            kr, cr = best_person(pose.predict(real, verbose=False)[0])
            ks, cs = best_person(pose.predict(sim, verbose=False)[0])
            kp = np.nan
            if kr is not None and ks is not None:
                both = (cr > 0.5) & (cs > 0.5)
                if both.any():
                    kp = float(np.linalg.norm(kr[both] - ks[both], axis=1).mean())
            rows.append({"frame": f, "iou": iou, "psnr": psnr, "ssim": ssim(real, sim, region),
                         "lpips": lp_val, "kp_px": kp})  # fmt: skip
        mean = {
            k: float(np.nanmean([r[k] for r in rows]))
            for k in ("iou", "psnr", "ssim", "lpips", "kp_px")
        }
        per_cam[cam] = mean
        worst[cam] = [r["frame"] for r in sorted(rows, key=lambda r: -r["lpips"])[:10]]

    # Physics from the fitted body and the iron placement rule of 04_build_scene.
    J = np.load("out/clip/joints.npz")
    names = list(J["names"])
    xyz = J["xyz"]
    fit = np.load("out/body/fit.npz")
    bed_top = float(xyz[:, names.index("left_middle_mcp"), 2].min()) - 0.03
    knuck = xyz[
        :, [names.index(n) for n in ("right_index_mcp", "right_middle_mcp", "right_pinky_mcp")]
    ]
    palm_z = ((xyz[:, names.index("right_wrist")] + knuck.mean(1)) / 2)[:, 2]
    handle_top = np.maximum(palm_z - 0.15, bed_top + 0.001) + 0.15
    penetration = np.clip(handle_top - palm_z, 0, None) * 100
    feet = xyz[:, [names.index("left_foot"), names.index("right_foot")]]
    floor_z = float(feet[..., 2].min()) - 0.03
    body_min_z = fit["vertices"][..., 2].min(1)
    foot_gap = (body_min_z - floor_z) * 100
    vel = np.linalg.norm(np.diff(feet[..., :2], axis=0), axis=-1) * 30  # m/s
    stance = feet[1:, :, 2] < floor_z + 0.08
    slide = float(vel[stance].mean() * 100) if stance.any() else 0.0
    physics = {"palm_into_iron_cm_max": float(penetration.max()),
               "lowest_body_point_vs_floor_cm": [float(foot_gap.min()), float(foot_gap.max())],
               "stance_foot_slide_cm_per_s": slide}  # fmt: skip

    hdr = ["camera", "IoU", "PSNR", "SSIM", "LPIPS", "kp px"]
    print(f"{hdr[0]:9s}" + "".join(f"{h:>7s}" for h in hdr[1:]) + "   10 worst frames (LPIPS)")
    for cam, m in per_cam.items():
        vals = "".join(f"{m[k]:7.3f}" for k in ("iou", "psnr", "ssim", "lpips", "kp_px"))
        print(f"{cam:9s}{vals}   {worst[cam]}")
    print("physics:", json.dumps(physics))
    out = {"round": args.round, "per_camera": per_cam, "worst_frames": worst, "physics": physics}
    json.dump(out, open(OUT / "metrics.json", "w"), indent=1)
    hist = OUT / "metrics_history.jsonl"
    with open(hist, "a") as fh:
        fh.write(json.dumps(out) + "\n")
    print(f"wrote {OUT / 'metrics.json'} (appended to {hist})")


if __name__ == "__main__":
    main()
