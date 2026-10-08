"""Stage 09c (measure): headcam room sharpness and masked PSNR, before vs after.

Room-only region = the stage-07 room mask of that clip frame, eroded by 60 px (away from
the person), plus the two background patches from out/research/blur_diagnosis.md.
Writes a 2-panel ORIGINAL | OURS preview of frame 30's corners (1:1 crops).

    uv run scripts/09c_measure.py [--tag t1]
"""

import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch

_spec = importlib.util.spec_from_file_location(
    "t08b", Path(__file__).with_name("08b_train_room_sharp.py")
)
t08b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t08b)

CLIP = Path("out/clip/frames/headcam")
PATCHES = {"right_floor": (slice(300, 900), slice(1650, 1900)),
           "left_floor": (slice(620, 1000), slice(0, 250))}  # fmt: skip


def metrics(g, m):
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return np.array([lap[m].var(), np.sqrt(gx * gx + gy * gy)[m].mean()])


def luma(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="out/room_render/headcam")
    ap.add_argument("--after", default="out/room_render_sharp/headcam")
    ap.add_argument("--ckpt", default="out/room_sharp/splats.pt")
    ap.add_argument("--tag", default="final")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    masks = t08b.npz_mmap(Path(ck["data_files"][0]), "masks")
    clip = ck["clip_slots"]
    border = np.zeros((1080, 1920), bool)
    border[8:-8, 8:-8] = True
    print(f"{'set':7s} {'region':12s} {'LapVar ratio':>12s} {'Grad ratio':>10s}")
    for name, d in (("before", Path(args.before)), ("after", Path(args.after))):
        acc = {}
        for f in (0, 30, 59):
            room = cv2.erode(np.asarray(masks[int(clip[f])]), np.ones((121, 121), np.uint8)) > 0
            o, r = luma(CLIP / f"{f:04d}.jpg"), luma(d / f"{f:04d}.png")
            regs = {"room_only": room & border}
            for k, (ys, xs) in PATCHES.items():
                mm = np.zeros_like(room)
                mm[ys, xs] = True
                regs[k] = mm
            for k, mm in regs.items():
                a = acc.setdefault(k, np.zeros(4))
                a += np.r_[metrics(r, mm), metrics(o, mm)]
        for k, (lr, gr, lo, go) in acc.items():
            print(f"{name:7s} {k:12s} {lr / lo:12.2f} {gr / go:10.2f}")
        ps = []
        for f in range(len(clip)):
            m = np.asarray(masks[int(clip[f])]).astype(bool)
            o = cv2.imread(str(CLIP / f"{f:04d}.jpg")).astype(np.float32) / 255
            r = cv2.imread(str(d / f"{f:04d}.png")).astype(np.float32) / 255
            ps.append(-10 * np.log10(((o - r) ** 2)[m].mean()))
        print(f"{name:7s} masked PSNR headcam (60 frames): mean {np.mean(ps):.2f} dB "
              f"(f0 {ps[0]:.2f}, f30 {ps[30]:.2f}, f59 {ps[59]:.2f})")  # fmt: skip

    # Preview: 2x2 corner crops at 1:1, ORIGINAL | OURS.
    o, r = cv2.imread(str(CLIP / "0030.jpg")), cv2.imread(str(Path(args.after) / "0030.png"))
    cw, chh = 640, 360

    def corners(im):
        return np.vstack([np.hstack([im[:chh, :cw], im[:chh, -cw:]]),
                          np.hstack([im[-chh:, :cw], im[-chh:, -cw:]])])  # fmt: skip

    pa, pb = corners(o), corners(r)
    for img, lab in ((pa, "ORIGINAL"), (pb, "OURS (sharp)")):
        cv2.putText(img, lab, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    sep = np.full((pa.shape[0], 8, 3), 255, np.uint8)
    out = Path("out/room_sharp") / f"preview_{args.tag}.jpg"
    cv2.imwrite(str(out), np.hstack([pa, sep, pb]), [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
