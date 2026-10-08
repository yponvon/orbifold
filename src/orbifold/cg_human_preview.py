"""CG-human milestone preview: ORIGINAL | OURS per camera, cropped tight on the person.

    uv run python -m orbifold.cg_human_preview --tag skin1 [--render] [--blend out/scene.blend]
        [--frame 30] [--samples 128]

--render renders the frame from the .blend (all three cameras, full 1920x1080, no motion
blur) into out/assets/cg_human/render/<cam>.png; without it, the existing pipeline
composite out/sim/<cam>/<frame>.png is used. The person crop is the projected bounding box
of the fitted body (out/body/fit.npz) plus a margin. Writes
out/assets/cg_human/preview_<tag>.jpg (and the per-camera full-res crops next to it).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

CLIP, OUT = Path("out/clip"), Path("out/assets/cg_human")
CAMS = ["headcam", "exocam1", "exocam2"]


def person_box(cam: str, f: int, size: tuple[int, int]) -> tuple[int, int, int, int]:
    C = np.load(CLIP / "cameras.npz")
    K, T = C[f"{cam}_K"], C[f"{cam}_T"][f]
    pts = [np.load("out/body/fit.npz")["vertices"][f]]
    g = Path("out/body/garments.npz")
    if g.exists():
        G = np.load(g)
        pts += [G["tunic_vertices"][f], G["trousers_vertices"][f]]
    P = np.concatenate(pts).astype(float)
    pc = (np.linalg.inv(T) @ np.c_[P, np.ones(len(P))].T)[:3].T  # ros: x fwd, y left, z up
    ok = pc[:, 0] > 0.15
    x, y, z = -pc[ok, 1], -pc[ok, 2], pc[ok, 0]
    u, v = K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2]
    w, h = size
    u0, u1 = np.clip(np.percentile(u, [0.5, 99.5]), 0, w)
    v0, v1 = np.clip(np.percentile(v, [0.5, 99.5]), 0, h)
    m = 0.06 * max(u1 - u0, v1 - v0)
    return (int(max(u0 - m, 0)), int(max(v0 - m, 0)), int(min(u1 + m, w)), int(min(v1 + m, h)))


def render(blend: str, frame: int, samples: int) -> None:
    import bpy

    bpy.ops.wm.open_mainfile(filepath=str(Path(blend).resolve()))
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    scene.cycles.device = "GPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.denoising_use_gpu = True
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.render.resolution_percentage = 100
    scene.render.use_motion_blur = False  # sharp, per the brief
    scene.cycles.filter_width = 1.5
    scene.frame_set(frame)
    (OUT / "render").mkdir(parents=True, exist_ok=True)
    for cam in CAMS:
        scene.camera = bpy.data.objects[cam]
        body = bpy.data.objects.get("body")
        if body and "hide_head" in body.modifiers:
            body.modifiers["hide_head"].show_render = cam == "headcam"
        scene.camera.data.clip_start = 0.14 if cam == "headcam" else 0.01
        for ob in bpy.data.objects:
            if ob.get("cg_head_only") or ob.name in ("hair_bun", "ponytail", "iron_cord"):
                ob.hide_render = cam == "headcam"
        scene.render.filepath = str((OUT / "render" / f"{cam}.png").resolve())
        bpy.ops.render.render(write_still=True)
        print(f"rendered {cam}")


def composite(cam: str, frame: int, rendered: bool) -> np.ndarray:
    if not rendered:
        return cv2.imread(f"out/sim/{cam}/{frame:04d}.png")
    fg = cv2.imread(str(OUT / "render" / f"{cam}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    bg = cv2.imread(f"out/room_render/{cam}/{frame:04d}.png")
    bg = (np.full(fg.shape[:2] + (3,), 128.0) if bg is None
          else cv2.resize(bg, fg.shape[1::-1]).astype(np.float32))  # fmt: skip
    a = fg[..., 3:] / 255.0
    return (fg[..., :3] * a + bg * (1 - a)).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frame", type=int, default=30)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--blend", default="out/scene.blend")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--height", type=int, default=900, help="row height of the sheet")
    a = ap.parse_args()
    if a.render:
        render(a.blend, a.frame, a.samples)
    rows = []
    for cam in CAMS:
        real = cv2.imread(str(CLIP / "frames" / cam / f"{a.frame:04d}.jpg"))
        ours = composite(cam, a.frame, a.render)
        x0, y0, x1, y1 = person_box(cam, a.frame, real.shape[1::-1])
        pair = []
        for img, label in ((real, f"ORIGINAL {cam}"), (ours, f"OURS {cam}")):
            c = img[y0:y1, x0:x1].copy()
            cv2.imwrite(str(OUT / f"crop_{a.tag}_{cam}_{label.split()[0].lower()}.png"), c)
            s = a.height / c.shape[0]
            c = cv2.resize(c, (int(c.shape[1] * s), a.height), interpolation=cv2.INTER_LANCZOS4)
            for col, th in (((0, 0, 0), 5), ((255, 255, 255), 2)):
                cv2.putText(c, label, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, th)
            pair.append(c)
        rows.append(np.hstack(pair))
    wmax = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in rows])
    out = OUT / f"preview_{a.tag}.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {out} {sheet.shape[1]}x{sheet.shape[0]}")


if __name__ == "__main__":
    main()
