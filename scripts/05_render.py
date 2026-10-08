"""Stage 05: render out/scene.blend from every camera with Cycles.

    uv run --extra render scripts/05_render.py [--samples 64] [--frames 0 59] [--step 3]

Uses the GPU when present (OptiX/CUDA on the 5090, Metal on a Mac).
Writes out/render/<cam>/0000.png ... (or --out DIR)

Sharpness: the output must be as sharp as the 1920x1080 footage, so by default
  * 100% resolution (never render small and upscale),
  * a narrow 1.0 px Blackman-Harris pixel filter (Blender's 1.5 px default, and the 2.0 px
    we used before, visibly soften edges and fabric detail),
  * no motion blur and no depth of field,
  * NO denoiser, 256 samples. Measured (15_sharpness_check, frames 15/30/45): even OIDN
    with ACCURATE albedo+normal prefiltering at HIGH quality smears the fabric/skin bump
    detail (person mean-gradient ratio 0.81-0.90 vs 0.96-1.01 without). --denoise accurate
    remains available for fast low-sample previews.
"""

import argparse
from pathlib import Path

import bpy

OUT = Path("out")


def use_gpu() -> str:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA", "METAL", "HIP"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        devices = [d for d in prefs.devices if d.type == backend]
        if devices:
            for d in devices:
                d.use = True
            bpy.context.scene.cycles.device = "GPU"
            return f"{backend}: {', '.join(d.name for d in devices)}"
    return "CPU"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=100, help="resolution percent (keep 100)")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--frames", type=int, nargs=2, help="first last (inclusive)")
    ap.add_argument("--cams", nargs="+", default=["headcam", "exocam1", "exocam2"])
    ap.add_argument("--step", type=int, default=1, help="render every Nth frame (previews)")
    ap.add_argument("--out", default=str(OUT / "render"), help="output dir (<out>/<cam>/)")
    ap.add_argument("--cpu", type=int, default=0, help="render on N CPU threads (GPU busy)")
    ap.add_argument("--blend", default=str(OUT / "scene.blend"), help="scene to render")
    ap.add_argument(
        "--filter", default="BLACKMAN_HARRIS", choices=["BLACKMAN_HARRIS", "GAUSSIAN", "BOX"]
    )
    ap.add_argument("--filter-width", type=float, default=1.0, help="pixel filter width (px)")
    ap.add_argument("--denoise", default="none", choices=["accurate", "fast", "none"])
    ap.add_argument("--motion-blur", type=float, default=0.0, help="shutter (frames); 0 = off")
    args = ap.parse_args()
    if args.scale != 100:
        print(f"WARNING: --scale {args.scale}: output is NOT full resolution (soft)")

    bpy.ops.wm.open_mainfile(filepath=str(Path(args.blend).resolve()))
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    if args.cpu:
        scene.cycles.device = "CPU"
        scene.render.threads_mode, scene.render.threads = "FIXED", args.cpu
        print("device: CPU x", args.cpu)
    else:
        print("device:", use_gpu())
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = args.denoise != "none"
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_use_gpu = not args.cpu  # was the bottleneck on the 5090 (CPU denoise)
    scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    if args.denoise == "accurate":
        scene.cycles.denoising_prefilter = "ACCURATE"
        scene.cycles.denoising_quality = "HIGH"
    else:
        scene.cycles.denoising_prefilter = "FAST"
    scene.cycles.texture_limit_render = "OFF"  # never downsample image textures
    scene.render.use_persistent_data = True
    scene.render.film_transparent = True  # foreground + shadows only, over the splat room
    scene.render.image_settings.color_mode = "RGBA"
    # Plain sRGB like a camera, not Blender's filmic AgX look (which lifts blacks).
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.render.resolution_percentage = args.scale
    scene.render.image_settings.file_format = "PNG"
    if args.frames:
        scene.frame_start, scene.frame_end = args.frames
    scene.frame_step = args.step
    scene.cycles.pixel_filter_type = args.filter
    scene.cycles.filter_width = args.filter_width
    scene.render.use_motion_blur = args.motion_blur > 0
    if args.motion_blur > 0:
        scene.render.motion_blur_shutter = args.motion_blur
    scene.render.use_sequencer = False
    comp = getattr(scene, "compositing_node_group", None) or getattr(scene, "node_tree", None)
    if comp is not None and len(comp.nodes) > 2:
        print("WARNING: compositor nodes present:", [n.bl_idname for n in comp.nodes])
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15  # PNG is lossless; just faster to write
    print(
        f"sharpness: {scene.render.resolution_x}x{scene.render.resolution_y} @ "
        f"{scene.render.resolution_percentage}%, filter {args.filter} {args.filter_width}px, "
        f"denoise {args.denoise}, motion blur {args.motion_blur}"
    )

    for cam in args.cams:
        scene.camera = bpy.data.objects[cam]
        # The head camera sits inside the head: don't render the head from it.
        body = bpy.data.objects.get("body")
        if body and "hide_head" in body.modifiers:
            body.modifiers["hide_head"].show_render = cam == "headcam"
        tunic = bpy.data.objects.get("tunic")
        if tunic and "hide_neck" in tunic.modifiers:
            tunic.modifiers["hide_neck"].show_render = cam == "headcam"
            if "neck_smooth" in tunic.modifiers:
                tunic.modifiers["neck_smooth"].show_render = cam == "headcam"
        for ob in bpy.data.objects:  # placket buttons: white dots right under the head camera
            if ob.name.startswith("button_"):
                ob.hide_render = cam == "headcam"
        # The brown placket trim reads as a jagged stripe from above: plain fabric there.
        if tunic and len(tunic.material_slots) > 1:
            trim = bpy.data.materials.get("trim_brown")
            tunic.material_slots[1].material = tunic.material_slots[0].material if cam == "headcam" else trim
        # Cut away the collar and neck right next to the head camera.
        scene.camera.data.clip_start = 0.22 if cam == "headcam" else 0.01
        scene.camera.data.dof.use_dof = False  # the footage is in focus everywhere
        # Eyebrows/eyelashes sit inside the head camera's view of the chest: hide them too.
        for name in ("hair_bun", "ponytail", "hair_strands", "iron_cord", "cg_brows", "cg_lashes", "collar"):
            if name in bpy.data.objects:
                bpy.data.objects[name].hide_render = cam == "headcam"
        out_dir = (Path(args.out) / cam).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(out_dir) + "/####"
        bpy.ops.render.render(animation=True)
        print(f"rendered {cam} frames {scene.frame_start}-{scene.frame_end} -> {out_dir}")


if __name__ == "__main__":
    main()
