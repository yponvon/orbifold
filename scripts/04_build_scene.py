"""Stage 04: build the Blender scene from out/clip/ and save out/scene.blend.

Foreground only: the Anny body mesh fitted to the trackers (stage 03b), dressed by
region (shirt, trousers, skin, hair, wrist bands), and an iron attached to the right
hand. The bed, floor and wall are invisible shadow catchers (the modelled room replaces
them later).

    uv run --extra render scripts/04_build_scene.py
Open out/scene.blend in desktop Blender 5.2 to look around.
"""

import json
import os
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

from orbifold import cg_human

CLIP, OUT, BODY = (
    Path("out/clip"),
    Path("out"),
    Path(os.environ.get("BODY_FIT", "out/body/fit.npz")),
)
COLORS, ENV = Path("out/body/colors.json"), Path("out/room/env.hdr")
ENV_STATS = Path("out/room/env_stats.json")
ALBEDO = Path("out/body/albedo.json")
GARMENTS = Path(os.environ.get("GARMENTS", "out/body/garments.npz"))
BAKED = Path("out/body/baked.npz")
KEY_STRENGTH = 0.6  # sun strength (W/m^2) on top of the room light map
# Blender camera axes (x right, y up, z back) expressed in the tracker camera frame
# (ros_body: x forward, y left, z up), verified by the stage 03 overlay.
ROS_TO_BLENDER_CAM = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
# Light levels; tuned so the rendered body matches the real person's brightness.
LIGHT = {"world": 0.8, "ceiling_w": 120.0}


def material(name: str, rgb: tuple, roughness: float = 0.6) -> bpy.types.Material:
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return m


def trim_mask(gm, fab, G, trim_bsdf) -> None:
    """Brown tunic trim as a smooth shader mask (stage 03e: rest coords + cuff distance).

    Cuff band CUFF_W from the sleeve opening, placket stripe PLACKET_W wide down the front
    centre, pocket-top band POCKET_W; edges are anti-aliased iso-lines of interpolated
    per-vertex fields, so they no longer follow the triangles.
    """
    CUFF_W, PLACKET_W, POCKET_W, SOFT = 0.025, 0.015, 0.012, 0.002
    for name, data in (("rest", G["tunic_rest"]), ("cuffd", G["tunic_cuffd"])):
        a = gm.attributes.new(name, "FLOAT_VECTOR" if data.ndim == 2 else "FLOAT", "POINT")
        a.data.foreach_set("vector" if data.ndim == 2 else "value", data.reshape(-1))
    front_y, plk_z0, pocket_z = (float(v) for v in G["tunic_trim_params"])
    nt = fab.node_tree
    N = nt.nodes.new

    def math(op, a, b=None):
        m = N("ShaderNodeMath")
        m.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = v
            else:
                nt.links.new(v, m.inputs[i])
        return m.outputs[0]

    def band(d, half):  # 1 inside |d| < half, smooth SOFT-wide edge
        s = N("ShaderNodeMapRange")
        s.interpolation_type = "SMOOTHSTEP"
        nt.links.new(d, s.inputs["Value"])
        s.inputs["From Min"].default_value = half + SOFT
        s.inputs["From Max"].default_value = half - SOFT
        return s.outputs["Result"]

    rest = N("ShaderNodeAttribute")
    rest.attribute_name = "rest"
    xyz = N("ShaderNodeSeparateXYZ")
    nt.links.new(rest.outputs["Vector"], xyz.inputs[0])
    x, y, z = xyz.outputs
    cd = N("ShaderNodeAttribute")
    cd.attribute_name = "cuffd"
    cuff = band(cd.outputs["Fac"], CUFF_W)
    front = band(math("SUBTRACT", y, front_y), 0.0)  # signed: 1 where y < front_y
    plk = math("MULTIPLY", band(math("ABSOLUTE", x), PLACKET_W / 2),
               band(math("SUBTRACT", plk_z0, z), 0.0))  # fmt: skip
    pocket = math("MULTIPLY", band(math("ABSOLUTE", math("SUBTRACT", z, pocket_z)), POCKET_W / 2),
                  band(math("ABSOLUTE", math("SUBTRACT", x, 0.10)), 0.05))  # fmt: skip
    w = math("MAXIMUM", cuff, math("MULTIPLY", front, math("MAXIMUM", plk, pocket)))
    bsdf = nt.nodes["Principled BSDF"]
    for inp, kind in (("Base Color", "RGBA"), ("Roughness", "FLOAT")):
        mix = N("ShaderNodeMix")
        mix.data_type = kind
        nt.links.new(w, mix.inputs["Factor"])
        src = bsdf.inputs[inp].default_value
        trg = trim_bsdf.inputs[inp].default_value
        if kind == "RGBA":
            mix.inputs[6].default_value, mix.inputs[7].default_value = tuple(src), tuple(trg)
        else:
            mix.inputs[2].default_value, mix.inputs[3].default_value = src, trg
        nt.links.new(mix.outputs[2] if kind == "RGBA" else mix.outputs[0], bsdf.inputs[inp])


def box(name: str, size: tuple, loc: tuple, mat: bpy.types.Material, rot_z: float = 0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=(0, 0, rot_z))
    ob = bpy.context.object
    ob.name, ob.scale = name, size
    ob.data.materials.append(mat)
    return ob


def main() -> None:
    J, C = np.load(CLIP / "joints.npz"), np.load(CLIP / "cameras.npz")
    names, xyz = list(J["names"]), J["xyz"]
    n_frames = len(xyz)
    idx = {n: i for i, n in enumerate(names)}

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 0, n_frames - 1
    scene.render.fps = 30

    # --- Room, estimated from the joints ---------------------------------------------
    feet = xyz[:, [idx["left_foot"], idx["right_foot"]], 2]
    floor_z = float(feet.min()) - 0.03
    pelvis_xy = xyz[:, idx["pelvis"], :2].mean(0)
    hands = xyz[:, [idx["left_middle_mcp"], idx["right_middle_mcp"]]]
    hand_xy = hands[..., :2].reshape(-1, 2).mean(0)
    fwd = (hand_xy - pelvis_xy) / np.linalg.norm(hand_xy - pelvis_xy)
    yaw = float(np.arctan2(fwd[1], fwd[0]))

    # Bed: 2.0 m long (along fwd) x 1.8 m wide; top surface just under the lowest palm.
    # The left hand rests flat on the cloth; the right one holds the iron (round-4 review).
    bed_top = float(xyz[:, idx["left_middle_mcp"], 2].min()) - 0.03
    bed_len, bed_w = 2.0, 1.8
    # Near edge just past the furthest-forward foot, so the legs are never inside the bed.
    feet_xy = xyz[:, [idx["left_foot"], idx["right_foot"]], :2].reshape(-1, 2)
    near_edge = pelvis_xy + fwd * (float(((feet_xy - pelvis_xy) @ fwd).max()) + 0.08)
    bed_c = near_edge + fwd * bed_len / 2
    # The bed shadow catcher's top follows the iron's fitted bed plane (poses.npz bed_z,
    # ~5 cm above the palm estimate) so the iron and hands get correct contact shadows.
    poses = OUT / "assets/iron/poses.npz"
    catch_top = float(np.load(poses)["bed_z"]) if poses.exists() else bed_top
    h = catch_top - floor_z
    # Proxies only: invisible shadow catchers so the body casts shadows onto the splat room.
    catch = material("proxy", (0.8, 0.8, 0.8))
    # The bed is only its TOP sheet: a solid box hid her legs behind the bed edge (exocam1).
    proxies = [
        box("bed", (bed_len, bed_w, 0.004), (*bed_c, catch_top - 0.002), catch, yaw),
        box("wall_back", (0.1, 8.0, 3.0), (*(bed_c + fwd * (bed_len / 2 + 0.3)), floor_z + 1.5),
            catch, yaw),
    ]  # fmt: skip
    bpy.ops.mesh.primitive_plane_add(size=30, location=(*pelvis_xy, floor_z))
    bpy.context.object.name = "floor"
    bpy.context.object.data.materials.append(catch)
    proxies.append(bpy.context.object)
    for ob in proxies:
        ob.is_shadow_catcher = True

    # --- Body: Anny mesh fitted to the trackers (stage 03b), one shape key per frame ------
    def srgb(c: tuple) -> tuple:  # colours were picked from the footage in sRGB
        return tuple(v**2.2 for v in c)

    fit = np.load(BODY)
    regions = [str(r) for r in fit["regions"]]
    looks = {  # fallback base colour (sRGB) and roughness
        "skin": ((0.66, 0.48, 0.38), 0.45),
        "shirt": ((0.74, 0.70, 0.62), 0.85),
        "trousers": ((0.07, 0.07, 0.08), 0.8),
        "hair": ((0.05, 0.04, 0.035), 0.35),
        "shoes": ((0.85, 0.85, 0.82), 0.5),  # white clogs (exocam2)
        "strap": ((0.03, 0.03, 0.03), 0.7),
        "eye": ((0.12, 0.08, 0.06), 0.1),
    }
    albedo = {r: srgb(c) for r, (c, _) in looks.items()}
    if COLORS.exists() and ENV_STATS.exists():
        # Measured colour = albedo x room light, so divide by the room's mean radiance.
        sampled = json.load(open(COLORS))["colors_srgb"]
        light = json.load(open(ENV_STATS))["mean"]
        for r, c in sampled.items():
            if r == "shoes":
                continue
            albedo[r] = tuple(min(v / light, 0.9) for v in srgb(c))
    if ALBEDO.exists():  # closed-loop calibrated (03d) takes precedence
        albedo |= {r: tuple(v) for r, v in json.load(open(ALBEDO)).items()}
    # Track B user feedback: the shirt is a light greige (measured ~sRGB 0.63/0.59/0.50 under
    # the room light); the calibrated albedo rendered it too dark and brown.
    if os.environ.get("SHIRT_SRGB"):
        albedo["shirt"] = srgb(tuple(float(v) for v in os.environ["SHIRT_SRGB"].split(",")))
    # The closed loop saturated the shirt to near-white (0.95); pin it and the skin by hand
    # to the footage's greige tunic and a warmer skin tone.
    albedo["shirt"] = (0.36, 0.31, 0.22)
    albedo["skin"] = (0.48, 0.30, 0.21)
    print("albedo:", {r: tuple(round(v, 3) for v in a) for r, a in albedo.items()})
    json.dump(albedo, open(BODY.parent / "albedo_used.json", "w"), indent=1)
    mesh = bpy.data.meshes.new("body")
    verts = fit["vertices"]
    mesh.from_pydata(verts[0].tolist(), [], fit["faces"].tolist())
    uv_layer = mesh.uv_layers.new(name="UVMap")
    uv_layer.data.foreach_set("uv", fit["uv"][fit["face_uv"]].reshape(-1))
    body = bpy.data.objects.new("body", mesh)
    scene.collection.objects.link(body)
    for r in regions:
        mat = material(f"body_{r}", albedo[r], looks[r][1])
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        if r == "skin":  # skin: subsurface scattering, pores, blotchiness, oily/matte areas
            nt = mat.node_tree
            bsdf.inputs["Subsurface Weight"].default_value = 0.3
            bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2)
            bsdf.inputs["Subsurface Scale"].default_value = 0.008
            pores, bump = nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeBump")
            pores.inputs["Scale"].default_value = 2500.0
            pores.inputs["Detail"].default_value = 4.0
            bump.inputs["Strength"].default_value = 0.06
            nt.links.new(pores.outputs["Fac"], bump.inputs["Height"])
            nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
            blotch = nt.nodes.new("ShaderNodeTexNoise")
            blotch.inputs["Scale"].default_value = 60.0
            tint = nt.nodes.new("ShaderNodeMix")
            tint.data_type = "RGBA"
            tint.inputs["Factor"].default_value = 0.12
            tint.inputs["A"].default_value = (*albedo[r], 1)
            tint.inputs["B"].default_value = (*(v * 0.8 for v in albedo[r]), 1)
            nt.links.new(blotch.outputs["Fac"], tint.inputs["Factor"])
            nt.links.new(tint.outputs["Result"], bsdf.inputs["Base Color"])
            rough = nt.nodes.new("ShaderNodeMapRange")
            rough.inputs["To Min"].default_value, rough.inputs["To Max"].default_value = 0.38, 0.6
            nt.links.new(blotch.outputs["Fac"], rough.inputs["Value"])
            nt.links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
        if r in ("shirt", "trousers"):  # woven-fabric micro bump
            nt = mat.node_tree
            noise, bump = nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeBump")
            noise.inputs["Scale"].default_value = 400.0
            bump.inputs["Strength"].default_value = 0.15
            nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
            nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        if r == "skin":  # CG human: MakeHuman CC0 skin texture + normal/roughness + SSS
            cg_human.skin_material(mat, albedo[r])
        if r == "hair":
            cg_human.hair_look(mat)
        mesh.materials.append(mat)
    cg_human.brows_lashes(scene, n_frames)
    face_region = fit["face_region"]
    mesh.polygons.foreach_set("material_index", face_region.astype(np.int32))
    mesh.polygons.foreach_set("use_smooth", np.ones(len(face_region), bool))

    # The head camera sits inside the head: 05_render enables this mask for that camera.
    head_group = body.vertex_groups.new(name="head")
    head_group.add(np.unique(fit["faces"][fit["face_is_head"]]).tolist(), 1.0, "REPLACE")
    mask = body.modifiers.new("hide_head", "MASK")
    mask.vertex_group, mask.invert_vertex_group, mask.show_render = "head", True, False
    sub = body.modifiers.new("smooth", "SUBSURF")
    sub.levels, sub.render_levels = 0, 1

    body.shape_key_add(name="basis")
    for f in range(n_frames):
        key = body.shape_key_add(name=f"f{f:04d}", from_mix=False)
        key.data.foreach_set("co", verts[f].reshape(-1))
        for g, val in ((f - 1, 0.0), (f, 1.0), (f + 1, 0.0)):
            if 0 <= g < n_frames:
                key.value = val
                key.keyframe_insert("value", frame=g)

    # Loose garments (stage 03e): follow the body via shape keys, draped by cloth sim.
    garment_objs = []
    if GARMENTS.exists():
        G = np.load(GARMENTS)
        body.modifiers.new("collision", "COLLISION")
        body.collision.thickness_outer = 0.004
        trim = material("trim_brown", srgb((0.27, 0.19, 0.14)), 0.8)
        for g, look in (("tunic", "shirt"), ("trousers", "trousers")):
            V, Fc = G[f"{g}_vertices"], G[f"{g}_faces"]
            gm = bpy.data.meshes.new(g)
            gm.from_pydata(V[0].tolist(), [], Fc.tolist())
            gm.uv_layers.new(name="UVMap").data.foreach_set("uv", G[f"{g}_uv"].reshape(-1))
            ob = bpy.data.objects.new(g, gm)
            scene.collection.objects.link(ob)
            fab = material(f"cloth_{g}", albedo[look], 0.9)
            nt = fab.node_tree
            # Matte cotton: little specular, so black trousers stay black under bright light.
            nt.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = 0.15
            nt.nodes["Principled BSDF"].inputs["Sheen Weight"].default_value = 0.2
            noise, bump = nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeBump")
            noise.inputs["Scale"].default_value = 500.0
            bump.inputs["Strength"].default_value = 0.1
            nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
            nt.links.new(bump.outputs["Normal"], nt.nodes["Principled BSDF"].inputs["Normal"])
            gm.materials.append(fab)
            if g == "tunic" and "tunic_rest" in G:  # smooth shader trim, no per-face split
                trim_mask(gm, fab, G, trim.node_tree.nodes["Principled BSDF"])
            else:
                gm.materials.append(trim)
                gm.polygons.foreach_set("material_index", G[f"{g}_trim"].astype(np.int32))
            gm.polygons.foreach_set("use_smooth", np.ones(len(Fc), bool))
            ob.shape_key_add(name="basis")
            for f in range(n_frames):
                key = ob.shape_key_add(name=f"f{f:04d}", from_mix=False)
                key.data.foreach_set("co", V[f].reshape(-1))
                for gf, val in ((f - 1, 0.0), (f, 1.0), (f + 1, 0.0)):
                    if 0 <= gf < n_frames:
                        key.value = val
                        key.keyframe_insert("value", frame=gf)
            pin = ob.vertex_groups.new(name="pin")
            # Shoulders/waist follow the body; the hem is held loosely (drapes, no balloon).
            for i, wgt in enumerate(0.4 + 0.6 * G[f"{g}_pin"]):
                pin.add([i], float(wgt), "REPLACE")
            cloth = ob.modifiers.new("cloth", "CLOTH")
            cs = cloth.settings
            cs.quality, cs.mass = 6, 0.2
            cs.tension_stiffness = cs.compression_stiffness = 12.0
            cs.shear_stiffness, cs.bending_stiffness = 8.0, 2.0
            cs.vertex_group_mass, cs.pin_stiffness = "pin", 1.0
            cloth.collision_settings.distance_min = 0.004
            cloth.point_cache.frame_start, cloth.point_cache.frame_end = 0, n_frames - 1
            ob.modifiers.new("thickness", "SOLIDIFY").thickness = 0.003
            if g == "tunic":  # 05_render enables this for the head camera (neck opening)
                labels = [str(b) for b in fit["bone_labels"]]
                neck_rest = fit["rest_bone_heads"][labels.index("neck01")]
                near = np.argmin(np.linalg.norm(fit["rest_vertices"] - neck_rest, axis=1))
                neck0 = verts[0][near]  # neck joint carried to frame 0 by the nearest skin vertex
                ids = np.flatnonzero(np.linalg.norm(V[0] - neck0, axis=1) < 0.12)
                ob.vertex_groups.new(name="neck").add(ids.tolist(), 1.0, "REPLACE")
                nm = ob.modifiers.new("hide_neck", "MASK")
                nm.vertex_group, nm.invert_vertex_group, nm.show_render = "neck", True, False
            garment_objs.append(ob)
            if g == "tunic" and "tunic_buttons" in G:  # buttons pinned to the placket
                btn_mat = material("button", srgb((0.85, 0.83, 0.78)), 0.3)
                for bi, vi in enumerate(G["tunic_buttons"]):
                    bpy.ops.mesh.primitive_cylinder_add(radius=0.006, depth=0.003, vertices=16)
                    btn = bpy.context.object
                    btn.name = f"button_{bi}"
                    btn.data.materials.append(btn_mat)
                    btn.parent, btn.parent_type = ob, "VERTEX"
                    btn.parent_vertices = [int(vi)] * 3
                    btn.location = (0, 0, 0)
            if g == "tunic" and "tunic_collar" in G:  # collar points: flat flaps on the chest
                cm = bpy.data.meshes.new("collar")
                tri_ids = G["tunic_collar"]
                cm.from_pydata([tuple(V[0][i]) for t in tri_ids for i in t], [],
                               [(0, 1, 2), (3, 4, 5)])  # fmt: skip
                collar = bpy.data.objects.new("collar", cm)
                scene.collection.objects.link(collar)
                cm.materials.append(fab)
                collar.shape_key_add(name="basis")
                for f in range(n_frames):
                    key = collar.shape_key_add(name=f"f{f:04d}", from_mix=False)
                    key.data.foreach_set("co", V[f][tri_ids.ravel()].ravel())
                    for gf, val in ((f - 1, 0.0), (f, 1.0), (f + 1, 0.0)):
                        if 0 <= gf < n_frames:
                            key.value = val
                            key.keyframe_insert("value", frame=gf)
                sol = collar.modifiers.new("thick", "SOLIDIFY")
                sol.thickness, sol.offset = 0.004, 1.0
                collar.modifiers.new("round", "SUBSURF").render_levels = 1
        print(f"garments: {[o.name for o in garment_objs]} (cloth sim)")

    # Appearance learned from the footage (03h): per-face-corner colours, shown unlit
    # (the footage already contains the room's lighting), like the splat room.
    if BAKED.exists():
        baked = np.load(BAKED)
        for name in ("body", "tunic", "trousers"):
            ob = bpy.data.objects.get(name)
            if ob is None or f"{name}_colors" not in baked:
                continue
            lin = (baked[f"{name}_colors"].reshape(-1, 3).clip(0, 1) ** 2.2).astype(np.float32)
            attr = ob.data.color_attributes.new("footage", "FLOAT_COLOR", "CORNER")
            attr.data.foreach_set("color", np.c_[lin, np.ones(len(lin), np.float32)].ravel())
            mat = bpy.data.materials.new(f"{name}_footage")
            mat.use_nodes = True
            nt = mat.node_tree
            nt.nodes.remove(nt.nodes["Principled BSDF"])
            col = nt.nodes.new("ShaderNodeAttribute")
            col.attribute_name = "footage"
            emit = nt.nodes.new("ShaderNodeEmission")
            nt.links.new(col.outputs["Color"], emit.inputs["Color"])
            nt.links.new(emit.outputs["Emission"], nt.nodes["Material Output"].inputs["Surface"])
            ob.data.materials.clear()
            ob.data.materials.append(mat)
        print("appearance: baked from footage (unlit)")

    # Black hair tied up in a bun at the back of the head, with a clip.
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, segments=24, ring_count=16)
    tail = bpy.context.object
    tail.name = "hair_bun"
    tail.data.materials.append(bpy.data.materials["body_hair"])
    for poly in tail.data.polygons:
        poly.use_smooth = True
    tail.rotation_mode = "QUATERNION"
    R0 = fit["head_rest_pose"][:3, :3]
    # In Anny's rest frame (z up, -y forward): behind and above the head bone's head.
    off_local = R0.T @ np.array([0.0, 0.075, 0.12])
    for f in range(n_frames):
        Hf = fit["head_pose"][f]
        Rw = Hf[:3, :3] @ R0.T  # head rotation relative to rest
        tail.location = Hf[:3, 3] + Hf[:3, :3] @ off_local
        tail.rotation_quaternion = (
            Matrix(Rw.tolist()).to_quaternion() @ Matrix.Rotation(0.6, 4, "X").to_quaternion()
        )
        tail.scale = (0.035, 0.03, 0.05)
        tail.keyframe_insert("location", frame=f)
        tail.keyframe_insert("rotation_quaternion", frame=f)

    # --- Iron: the CG model of src/orbifold/cg_iron.py (pose track out/assets/iron/poses.npz)
    import sys

    sys.path.insert(0, "src")
    from orbifold.cg_iron import build_iron

    iron = build_iron(scene, n_frames, xyz, names, bed_top)

    # --- Cameras ------------------------------------------------------------------------
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
    scene.camera = bpy.data.objects["exocam1"]

    # --- Lighting -----------------------------------------------------------------------
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    scene.world = world
    if ENV.exists():
        # The room's own light: a 360-degree map rendered from the room model (stage 09b).
        env = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(str(ENV.resolve()))
        env.image.colorspace_settings.name = "Linear Rec.709"
        world.node_tree.links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = 1.0
        print(f"world lighting from {ENV}")
        key_dir = json.load(open(ENV_STATS)).get("key_dir")
        if key_dir:  # soft key light from the window direction, for shading and shadows
            sun = bpy.data.lights.new("window_key", "SUN")
            sun.energy, sun.angle = KEY_STRENGTH, np.radians(20)
            sob = bpy.data.objects.new("window_key", sun)
            sob.rotation_mode = "QUATERNION"
            sob.rotation_quaternion = Vector(key_dir).to_track_quat("Z", "Y")
            scene.collection.objects.link(sob)
            print(f"key light from {np.round(key_dir, 2)}")
    else:
        bg.inputs["Color"].default_value = (0.85, 0.8, 0.72, 1)
        bg.inputs["Strength"].default_value = LIGHT["world"]
        light = bpy.data.lights.new("ceiling", "AREA")
        light.energy, light.size = LIGHT["ceiling_w"], 3.0
        lob = bpy.data.objects.new("ceiling", light)
        lob.location = (*bed_c, floor_z + 2.7)
        scene.collection.objects.link(lob)

    w, h_px = C["exocam1_size"]
    scene.render.resolution_x, scene.render.resolution_y = int(w), int(h_px)
    blend = str((OUT / "scene.blend").resolve())
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    if garment_objs:
        for ob in garment_objs:
            ob.modifiers["cloth"].point_cache.use_disk_cache = True
        with bpy.context.temp_override(scene=scene):
            bpy.ops.ptcache.bake_all(bake=True)
        print("cloth simulation baked")
        bpy.ops.wm.save_as_mainfile(filepath=blend)
    print(f"saved {OUT / 'scene.blend'}: Anny body {len(verts[0])} vertices, {n_frames} frames")
    print(f"floor z={floor_z:.3f}  bed top z={bed_top:.3f}  facing yaw={np.degrees(yaw):.0f} deg")


if __name__ == "__main__":
    main()
