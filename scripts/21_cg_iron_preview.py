"""CG iron preview: minimal scene (iron + cameras + room light), Cycles, one frame per camera.

    UV_NO_SYNC=1 uv run scripts/21_cg_iron_preview.py [--frame 30] [--tag r1] [--snap]

Writes out/assets/cg_iron/preview_<cam>[_<tag>].jpg: ORIGINAL crop | OURS crop, where OURS
is the full-resolution (1920x1080, no blur) Cycles render alpha-composited over the original
frame with the real iron inpainted away. The label gives the silhouette IoU against the
traced iron mask (hand pixels ignored). Also saves out/assets/cg_iron/cg_iron.blend.
"""

import argparse
import json
import sys
from pathlib import Path

import bpy
import cv2
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, "src")
from orbifold.cg_iron import build_iron  # noqa: E402

CLIP, OUT = Path("out/clip"), Path("out/assets/cg_iron")
ENV, ENV_STATS = Path("out/room/env.hdr"), Path("out/room/env_stats.json")
IRON_MASKS = Path("out/assets/iron")
KEY_STRENGTH = 0.6
ROS_TO_BLENDER_CAM = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)


def use_gpu() -> str:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        devs = [d for d in prefs.devices if d.type == backend]
        if devs:
            for d in devs:
                d.use = True
            bpy.context.scene.cycles.device = "GPU"
            return backend
    return "CPU"


def cameras(scene, C, n_frames):
    """Copied from 04_build_scene.py."""
    conv = np.eye(4)
    conv[:3, :3] = ROS_TO_BLENDER_CAM
    for cam in C["names"]:
        K, T, (w, h_px) = C[f"{cam}_K"], C[f"{cam}_T"], C[f"{cam}_size"]
        data = bpy.data.cameras.new(cam)
        data.sensor_fit, data.sensor_width = "HORIZONTAL", 36.0
        data.lens = float(K[0, 0]) * 36.0 / float(w)
        data.shift_x = (w / 2 - K[0, 2]) / w
        data.shift_y = (K[1, 2] - h_px / 2) / w
        data.clip_start = 0.02
        ob = bpy.data.objects.new(cam, data)
        scene.collection.objects.link(ob)
        for f in range(n_frames):
            ob.matrix_world = Matrix((T[f] @ conv).tolist())
            ob.keyframe_insert("location", frame=f)
            ob.keyframe_insert("rotation_euler", frame=f)


def lighting(scene):
    """Copied from 04_build_scene.py: room env map + soft window key."""
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    scene.world = world
    if ENV.exists():
        env = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(str(ENV.resolve()))
        env.image.colorspace_settings.name = "Linear Rec.709"
        world.node_tree.links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = 1.0
        key_dir = json.load(open(ENV_STATS)).get("key_dir")
        if key_dir:
            sun = bpy.data.lights.new("window_key", "SUN")
            sun.energy, sun.angle = KEY_STRENGTH, np.radians(20)
            sob = bpy.data.objects.new("window_key", sun)
            sob.rotation_mode = "QUATERNION"
            sob.rotation_quaternion = Vector(key_dir).to_track_quat("Z", "Y")
            scene.collection.objects.link(sob)
    else:
        bg.inputs["Color"].default_value = (0.85, 0.8, 0.72, 1)
        bg.inputs["Strength"].default_value = 0.8


def silhouette(scene, cam_ob, objs, W, H):
    """Iron coverage mask (H,W) by filling every projected triangle, and its pixel bbox."""
    from bpy_extras.object_utils import world_to_camera_view

    dg = bpy.context.evaluated_depsgraph_get()
    cov = np.zeros((H, W), np.uint8)
    allp = []
    for ob in objs:
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        me.calc_loop_triangles()
        mw = ev.matrix_world
        P = np.array([world_to_camera_view(scene, cam_ob, mw @ v.co) for v in me.vertices])
        uv = np.c_[P[:, 0] * W, (1 - P[:, 1]) * H]
        tris = np.array([t.vertices[:] for t in me.loop_triangles])
        ok = (P[tris, 2] > 0).all(1)
        cv2.fillPoly(cov, list(np.round(uv[tris[ok]] * 4).astype(np.int32)), 1, cv2.LINE_8, 2)
        allp.append(uv[P[:, 2] > 0])
        ev.to_mesh_clear()
    p = np.concatenate(allp)
    return cov > 0, (p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=30)
    ap.add_argument("--tag", default="")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--snap", action="store_true", help="snap the pose track to 04's bed_top")
    ap.add_argument("--cams", nargs="+", default=["headcam", "exocam1", "exocam2"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    J, C = np.load(CLIP / "joints.npz"), np.load(CLIP / "cameras.npz")
    names, xyz = list(J["names"]), J["xyz"]
    n_frames = len(xyz)
    bed_top = float(xyz[:, names.index("left_middle_mcp"), 2].min()) - 0.03  # as in 04

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 0, n_frames - 1
    scene.render.fps = 30
    iron = build_iron(scene, n_frames, xyz, names, bed_top, snap_to_bed=args.snap)
    plane_z = bed_top
    if not args.snap and (IRON_MASKS / "poses.npz").exists():
        plane_z = float(np.load(IRON_MASKS / "poses.npz")["bed_z"])
    bpy.ops.mesh.primitive_plane_add(size=6, location=(*xyz[0, names.index("pelvis"), :2], plane_z))
    catcher = bpy.context.object
    catcher.name, catcher.is_shadow_catcher = "bed_catcher", True
    cameras(scene, C, n_frames)
    lighting(scene)

    scene.render.engine = "CYCLES"
    print("device:", use_gpu())
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    scene.cycles.denoising_prefilter = "ACCURATE"
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.cycles.pixel_filter_type, scene.cycles.filter_width = "BLACKMAN_HARRIS", 1.0
    scene.render.use_motion_blur = False
    scene.frame_set(args.frame)
    bpy.ops.wm.save_as_mainfile(filepath=str((OUT / "cg_iron.blend").resolve()))

    parts = [o for o in iron.children_recursive if o.type == "MESH"]
    suffix = f"_{args.tag}" if args.tag else ""
    for cam in args.cams:
        W, H = (int(v) for v in C[f"{cam}_size"])
        scene.render.resolution_x, scene.render.resolution_y = W, H
        scene.render.resolution_percentage = 100
        cam_ob = bpy.data.objects[cam]
        scene.camera = cam_ob
        cam_ob.data.clip_start = 0.14 if cam == "headcam" else 0.01
        bpy.data.objects["iron_cord"].hide_render = cam == "headcam"  # as in 05_render
        cov, (x0, x1, y0, y1) = silhouette(scene, cam_ob, parts, W, H)
        pad = 0.6 * max(x1 - x0, y1 - y0) + 20
        rx0, rx1 = max(0, x0 - pad), min(W, x1 + pad)
        ry0, ry1 = max(0, y0 - pad), min(H, y1 + pad)
        if rx1 <= rx0 or ry1 <= ry0:
            print(f"{cam}: iron off-screen")
            continue
        scene.render.use_border, scene.render.use_crop_to_border = True, False
        scene.render.border_min_x, scene.render.border_max_x = rx0 / W, rx1 / W
        scene.render.border_min_y, scene.render.border_max_y = 1 - ry1 / H, 1 - ry0 / H
        png = OUT / f"render_{cam}{suffix}.png"
        scene.render.filepath = str(png.resolve())
        bpy.ops.render.render(write_still=True)
        compose(cam, args.frame, png, cov, (x0, x1, y0, y1), OUT / f"preview_{cam}{suffix}.jpg")


def compose(cam, f, png, cov, box, dst):
    real = cv2.imread(str(CLIP / f"frames/{cam}/{f:04d}.jpg"))
    rgba = cv2.imread(str(png), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
    H, W = real.shape[:2]
    mp, hp = IRON_MASKS / f"masks/{cam}/{f:04d}.png", IRON_MASKS / f"hand/{cam}/{f:04d}.png"
    m = cv2.imread(str(mp), 0) > 0 if mp.exists() else np.zeros((H, W), bool)
    hand = cv2.imread(str(hp), 0) > 0 if hp.exists() else np.zeros((H, W), bool)
    # Background for OURS: the real frame with the real iron inpainted away.
    bg = real
    if m.any():
        hole = cv2.dilate(m.astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
        bg = cv2.inpaint(real, hole, 7, cv2.INPAINT_TELEA)
    a = rgba[..., 3:4]
    ours = (rgba[..., :3] + bg.astype(np.float32) / 255 * (1 - a)) * 255  # premultiplied
    ours = ours.clip(0, 255).astype(np.uint8)
    v = ~(hand & ~m)
    inter, uni = (cov & m & v).sum(), ((cov | m) & v).sum()
    iou = inter / uni if (uni and m.any()) else float("nan")
    x0, x1, y0, y1 = box
    if m.any():
        ys, xs = np.nonzero(m)
        x0, x1 = min(x0, xs.min()), max(x1, xs.max())
        y0, y1 = min(y0, ys.min()), max(y1, ys.max())
    pad = 0.25 * max(x1 - x0, y1 - y0) + 10
    cx0, cx1 = int(max(0, x0 - pad)), int(min(W, x1 + pad))
    cy0, cy1 = int(max(0, y0 - pad)), int(min(H, y1 + pad))
    A, B = real[cy0:cy1, cx0:cx1], ours[cy0:cy1, cx0:cx1]
    s = max(1, int(np.ceil(420 / (cy1 - cy0))))  # integer nearest-neighbour zoom: no blur
    A = cv2.resize(A, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    B = cv2.resize(B, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    cv2.putText(A, f"{cam} f{f} ORIGINAL", (8, 26), 0, 0.7, (0, 255, 255), 2)
    cv2.putText(B, f"OURS (CG iron)  IoU {iou:.3f}", (8, 26), 0, 0.7, (0, 255, 255), 2)
    out = np.hstack([A, np.full((A.shape[0], 6, 3), 255, np.uint8), B])
    cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    # Debug: traced real outline (red) over OURS, and colour statistics of both.
    dbg = ours.copy()
    cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(dbg, cnt, -1, (0, 0, 255), 1)
    D = cv2.resize(dbg[cy0:cy1, cx0:cx1], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(dst).replace(".jpg", "_dbg.jpg"), D, [cv2.IMWRITE_JPEG_QUALITY, 90])
    real_px, our_px = real[m & ~hand], ours[cov & (rgba[..., 3] > 0.99)]
    for name, px in (("real", real_px), ("ours", our_px)):
        if len(px):
            q = np.percentile(px[:, ::-1], [25, 50, 75, 95], axis=0).astype(int)
            print(f"  {name} RGB p25/50/75/95: {q.tolist()}")
    print(f"preview {cam}: IoU {iou:.3f} -> {dst}")


if __name__ == "__main__":
    main()
