"""Stage 22: render the CG iron (src/orbifold/cg_iron.py) as an RGBA layer for every frame.

Same scene as 21_cg_iron_preview.py (iron + cameras + room light + bed shadow catcher,
Cycles, no motion blur, sharp 1.0 px filter), rendered over a transparent background for
all clip frames. Only a box around the iron is rendered (fast); the PNG is full 1920x1080.
Depth for occlusion with the person comes from the iron fit (same poses.npz).

    UV_NO_SYNC=1 uv run scripts/22_cg_iron_layer.py --cams exocam1 [--samples 64]
Writes out/assets/cg_iron/render/<cam>/<f>.png; links out/assets/cg_iron/depth -> iron/depth.
"""

import argparse
import importlib
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
prev = importlib.import_module("21_cg_iron_preview")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--cams", nargs="+", default=["headcam", "exocam1", "exocam2"])
    args = ap.parse_args()
    out = prev.OUT
    J, C = np.load(prev.CLIP / "joints.npz"), np.load(prev.CLIP / "cameras.npz")
    names, xyz = list(J["names"]), J["xyz"]
    n_frames = len(xyz)
    bed_top = float(xyz[:, names.index("left_middle_mcp"), 2].min()) - 0.03

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 0, n_frames - 1
    iron = prev.build_iron(scene, n_frames, xyz, names, bed_top)
    plane_z = float(np.load(prev.IRON_MASKS / "poses.npz")["bed_z"])
    bpy.ops.mesh.primitive_plane_add(size=6, location=(*xyz[0, names.index("pelvis"), :2], plane_z))
    bpy.context.object.is_shadow_catcher = True
    prev.cameras(scene, C, n_frames)
    prev.lighting(scene)
    scene.render.engine = "CYCLES"
    print("device:", prev.use_gpu())
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    scene.cycles.denoising_prefilter = "ACCURATE"
    scene.render.film_transparent = True
    scene.render.image_settings.file_format, scene.render.image_settings.color_mode = "PNG", "RGBA"
    scene.view_settings.view_transform, scene.view_settings.look = "Standard", "None"
    scene.cycles.pixel_filter_type, scene.cycles.filter_width = "BLACKMAN_HARRIS", 1.0
    scene.render.use_motion_blur = False
    parts = [o for o in iron.children_recursive if o.type == "MESH"]

    depth_link = out / "depth"
    if not depth_link.exists():
        depth_link.symlink_to(Path("../iron/depth"))
    for cam in args.cams:
        W, H = (int(v) for v in C[f"{cam}_size"])
        scene.render.resolution_x, scene.render.resolution_y = W, H
        scene.render.resolution_percentage = 100
        cam_ob = bpy.data.objects[cam]
        scene.camera = cam_ob
        cam_ob.data.clip_start = 0.14 if cam == "headcam" else 0.01
        bpy.data.objects["iron_cord"].hide_render = cam == "headcam"
        (out / "render" / cam).mkdir(parents=True, exist_ok=True)
        for f in range(n_frames):
            scene.frame_set(f)
            _, (x0, x1, y0, y1) = prev.silhouette(scene, cam_ob, parts, W, H)
            pad = 0.6 * max(x1 - x0, y1 - y0) + 40
            rx0, rx1 = max(0, x0 - pad), min(W, x1 + pad)
            ry0, ry1 = max(0, y0 - pad), min(H, y1 + pad)
            scene.render.use_border = rx1 > rx0 and ry1 > ry0
            scene.render.use_crop_to_border = False
            if scene.render.use_border:
                scene.render.border_min_x, scene.render.border_max_x = rx0 / W, rx1 / W
                scene.render.border_min_y, scene.render.border_max_y = 1 - ry1 / H, 1 - ry0 / H
            scene.render.filepath = str((out / "render" / cam / f"{f:04d}.png").resolve())
            bpy.ops.render.render(write_still=True)
        print(f"rendered CG iron for {cam}: {n_frames} frames")


if __name__ == "__main__":
    main()
