"""Render the trained human Gaussians for every clip frame and camera, score them, preview.

    UV_NO_SYNC=1 uv run scripts/30_human_gs_render.py
Writes, under out/assets/human_gs/:
  render/<cam>/<f>.png     RGBA 1920x1080 (straight alpha), the person alone
  composite/<cam>/<f>.png  person over out/room_render/<cam>/<f>.png
  depth/<cam>/<f>.npy      expected depth (gsplat "RGB+ED"), float16 metres from the camera,
                           inf where alpha < 0.5
  metrics.json             per camera, train vs held-out: LPIPS (alex), PSNR, SSIM inside
                           the traced person mask, plus silhouette IoU (alpha>0.5 vs mask)
  preview_<tag>.jpg        frame 30, one row per camera: ORIGINAL cut-out | OURS cut-out
  RENDER_DONE              touched when everything is written
The composite rasterises the human and room Gaussians together (one gsplat call), so room
geometry in front of the person (the bed) hides it; the room colour comes from room_render.
Metrics use the traced person mask minus the dilated iron mask (the iron is its own layer).
    --score-only --model X --tag Y   re-score another checkpoint (writes metrics_<tag>.json
                                     and preview_<tag>.jpg only)
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
hgs = importlib.import_module("30_human_gs_common")
ROOM = Path("out/room_render")
KEYS = ["lpips", "psnr", "ssim", "sharp", "sil_iou"]


def write_outputs(model, cc, room, cams, ci, cam, f, p, an, depth):
    straight = np.where(an > 1e-4, p / np.maximum(an, 1e-4), 0)
    rgba = (np.concatenate([straight, an], -1) * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(str(hgs.OUT / "render" / cam / f"{f:04d}.png"), rgba[..., [2, 1, 0, 3]])
    np.save(hgs.OUT / "depth" / cam / f"{f:04d}.npy", depth.astype(np.float16))
    room_p = ROOM / cam / f"{f:04d}.png"
    if room_p.exists():
        bg = cv2.imread(str(room_p))[..., ::-1].astype(np.float32) / 255
        if bg.shape[:2] != p.shape[:2]:
            bg = cv2.resize(bg, p.shape[1::-1])
        pv, av = hgs.render_joint(model, cc, room, cams, ci, f)  # visible part of the person
        comp = pv + (1 - av) * bg
        cv2.imwrite(
            str(hgs.OUT / "composite" / cam / f"{f:04d}.png"),
            (comp[..., ::-1] * 255 + 0.5).astype(np.uint8),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(hgs.OUT / "model.pt"))
    ap.add_argument("--tag", default="final")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    model, cc = hgs.load_model(args.model, dev)
    room = None if args.score_only else hgs.Room(dev)
    cams = hgs.load_cameras()
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev)
    if not args.score_only:
        (hgs.OUT / "RENDER_DONE").unlink(missing_ok=True)
    rows = {}
    for ci, cam in enumerate(hgs.CAMS):
        for sub in ("render", "composite", "depth"):
            (hgs.OUT / sub / cam).mkdir(parents=True, exist_ok=True)
        for f in range(hgs.N_FRAMES):
            p, an, depth = hgs.render_full(model, cc, cams, ci, f, dev, with_depth=True)
            if not args.score_only:
                write_outputs(model, cc, room, cams, ci, cam, f, p, an, depth)
            real = hgs.load_real(cam, f)
            m = hgs.load_mask(cam, f)
            s = hgs.score(p, real, m, lp, dev)
            sil = an[..., 0] > 0.5
            s["sil_iou"] = float((sil & m).sum() / max((sil | m).sum(), 1))
            split = "heldout" if hgs.is_heldout(f) else "train"
            rows.setdefault(cam, {}).setdefault(split, []).append([s[k] for k in KEYS])
            print(f"{cam} {f:02d} [{split}] " + " ".join(f"{k} {s[k]:.3f}" for k in KEYS), flush=True)

    out = {}
    for cam, d in rows.items():
        out[cam] = {}
        for split, v in d.items():
            out[cam][split] = dict(zip(KEYS, np.array(v).mean(0).round(4).tolist()))
            out[cam][split]["n_frames"] = len(v)
    for split in ("train", "heldout"):
        v = np.concatenate([np.array(rows[c][split]) for c in rows])
        out.setdefault("all", {})[split] = dict(zip(KEYS, v.mean(0).round(4).tolist()))
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    out["model"] = args.model
    out["train_seconds"] = ck.get("train_seconds")
    out["train_args"] = ck.get("args")
    out["n_gaussians"] = int(model.tri.shape[0])
    out["mask"] = "YOLO person mask minus iron mask dilated 5 px"
    name = f"metrics_{args.tag}.json" if args.score_only else "metrics.json"
    (hgs.OUT / name).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    lps = hgs.write_preview(model, cc, cams, args.tag, lp, dev=dev)
    print(f"wrote {hgs.OUT / f'preview_{args.tag}.jpg'}: f30 LPIPS {lps}")
    if not args.score_only:
        (hgs.OUT / "RENDER_DONE").touch()
        print("touched RENDER_DONE")


if __name__ == "__main__":
    main()
