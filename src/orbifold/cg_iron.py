"""CG steam iron for the Blender scene: modelled, shaded and posed (no Gaussians).

Iron local frame (same as scripts/20_iron_common.py and out/assets/iron/poses.npz):
metres, origin = centre of the soleplate's bottom face, +x towards the tip, +y left,
+z up (soleplate bottom is the z=0 plane).

    from orbifold.cg_iron import build_iron
    iron = build_iron(scene, n_frames, xyz, names, bed_top)

The returned object is an empty named "iron" carrying the per-frame pose; the parts are its
children. The cord is a separate world-space curve named "iron_cord" (05_render hides
objects with that name in the head camera).
"""

from __future__ import annotations

from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

POSES = Path("out/assets/iron/poses.npz")

# --- Dimensions (unscaled; poses.npz "sxy" scales x and y) ---------------------------------
L, WMAX = 0.25, 0.115  # soleplate length, max width
X_REAR = -0.105  # heel x; tip at X_REAR + L
SOLE_T = 0.006  # soleplate thickness
BODY_TOP = 0.030  # aqua lower body -> white shell seam
H_REAR, H_TIP = 0.075, 0.032  # shell crown height at the heel and near the tip
CORD_R = 0.0035
FILL = 0.35  # emission fill strength for plastics (matches footage exposure)


def srgb(c) -> tuple:
    return tuple(float(v) ** 2.2 for v in c)


# --- Geometry helpers ---------------------------------------------------------------------
def outline(n: int = 64, s: float = 1.0, cx: float = -0.01) -> np.ndarray:
    """Closed teardrop ring (2n-2 points, CCW seen from +z) of the soleplate, scaled by s."""
    t = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
    r = 0.08
    f = np.where(
        t < r,
        np.sqrt(np.clip(1 - ((r - t) / r) ** 2, 0, 1)),
        np.clip(1 - ((t - r) / (1 - r)) ** 2, 0, 1) ** 0.55,
    )
    x = X_REAR + L * t
    hw = WMAX / 2 * f
    ring = np.concatenate([np.stack([x, -hw], 1), np.stack([x[::-1], hw[::-1]], 1)[1:-1]])
    ring[:, 0] = cx + (ring[:, 0] - cx) * s
    ring[:, 1] *= s
    return ring


def crown(x: np.ndarray) -> np.ndarray:
    """Shell crown height as a function of x: highest at the heel, sloping to the tip."""
    t = np.clip((x - X_REAR) / L, 0, 1)
    return H_TIP + (H_REAR - H_TIP) * (1 - t**1.6)


def loft(rings, cap_bottom=True, cap_top=True):
    """Rings of equal length -> (verts, faces) with quad sides and n-gon caps."""
    m = len(rings[0])
    V = np.concatenate(rings)
    F = []
    for k in range(len(rings) - 1):
        a, b = k * m, (k + 1) * m
        for j in range(m):
            j2 = (j + 1) % m
            F.append((a + j, a + j2, b + j2, b + j))
    if cap_bottom:
        F.append(tuple(range(m - 1, -1, -1)))
    if cap_top:
        a = (len(rings) - 1) * m
        F.append(tuple(range(a, a + m)))
    return V, F


def tube(path, radii, k: int = 16):
    """Tube along a polyline in the x-z plane; radii (N,2) = (half-width y, half-thickness)."""
    path = np.asarray(path, float)
    tang = np.gradient(path, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    th = np.linspace(0, 2 * np.pi, k, endpoint=False)
    rings = []
    for p, q, (ry, rt) in zip(path, nrm, radii):
        c, s = np.cos(th), np.sin(th)
        # Rounded-rectangle-ish cross-section (superellipse) reads like moulded plastic.
        c2 = np.sign(c) * np.abs(c) ** 0.7
        s2 = np.sign(s) * np.abs(s) ** 0.7
        rings.append(np.stack([p[0] + rt * c2 * q[0], ry * s2, p[1] + rt * c2 * q[1]], 1))
    return loft(rings)


def catmull(P, n: int) -> np.ndarray:
    P = np.asarray(P, float)
    Pp = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(Pp) - 2):
        p0, p1, p2, p3 = Pp[i - 1 : i + 3]
        for u in np.linspace(0, 1, n, endpoint=False):
            out.append(
                0.5
                * (
                    2 * p1
                    + (-p0 + p2) * u
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * u**3
                )
            )
    out.append(P[-1])
    return np.array(out)


def sole_geom(sxy):
    rings = []
    for z, s in ((0.0, 0.975), (0.0012, 0.993), (0.0035, 1.0), (SOLE_T, 0.996), (SOLE_T + 0.001, 0.985)):
        r = outline(s=s)
        rings.append(np.c_[r * sxy, np.full(len(r), z)])
    return loft(rings)


def body_geom(sxy):
    """Aqua lower body: a skirt from the soleplate up to the shell seam, slightly tucked in."""
    rings = []
    for z, s in ((SOLE_T - 0.001, 0.985), (SOLE_T + 0.004, 0.99), (0.018, 0.988),
                 (BODY_TOP - 0.004, 0.982), (BODY_TOP + 0.004, 0.96)):  # fmt: skip
        r = outline(s=s)
        rings.append(np.c_[r * sxy, np.full(len(r), z)])
    return loft(rings)


def shell_geom(sxy, k: int = 12):
    """White shell: a tent-like dome (straight sloping flanks, rounded crown) over the
    teardrop, whose crown height follows crown(x). Seen from behind it reads as a triangle."""
    rings = []
    z0 = BODY_TOP - 0.003
    for i in range(k + 1):
        u = i / k
        s = 1.005 - 0.60 * u**1.1
        r = outline(s=s)
        h = crown(r[:, 0])
        lift = np.sin(np.pi / 2 * u) ** 0.6 if i else 0.0  # near-vertical lip, then slope
        z = z0 + (h - z0) * (0.35 * lift + 0.65 * u * (1.3 - 0.3 * u))
        rings.append(np.c_[r * sxy, z])
    return loft(rings)


HANDLE_PATH = [  # closed bridge handle (x, z): slanted heel strut -> grip -> front junction
    (-0.101, 0.030), (-0.091, 0.060), (-0.075, 0.087), (-0.050, 0.100), (-0.015, 0.103),
    (0.015, 0.100), (0.038, 0.088), (0.052, 0.070), (0.060, 0.050),
]  # fmt: skip
HANDLE_RADII = [  # (half-width y, half-thickness) at each path point
    (0.030, 0.012), (0.026, 0.012), (0.020, 0.012), (0.016, 0.011), (0.015, 0.011),
    (0.015, 0.011), (0.017, 0.012), (0.021, 0.014), (0.024, 0.015),
]  # fmt: skip


def handle_geom(sxy):
    path = catmull(HANDLE_PATH, 6)
    t = np.linspace(0, 1, len(path))
    tk = np.linspace(0, 1, len(HANDLE_RADII))
    R = np.array(HANDLE_RADII)
    radii = np.stack([np.interp(t, tk, R[:, 0]), np.interp(t, tk, R[:, 1])], 1)
    V, F = tube(path, radii, k=20)
    V[:, 0] *= sxy
    V[:, 1] *= sxy
    return V, F


def window_geom(sxy, n: int = 32, k: int = 8):
    """Teal translucent water-tank window: a flattened ellipsoid under the handle bridge that
    shows on both flanks of the shell beside the grip."""
    cx, ax, ay, az = -0.008, 0.056, 0.036, 0.016
    cz = crown(np.array([cx]))[0] - 0.008
    rings = []
    for i in range(1, k):
        ph = -np.pi / 2 + np.pi * i / k
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rings.append(np.stack([cx + ax * np.cos(ph) * np.cos(th), ay * np.cos(ph) * np.sin(th),
                               np.full(n, cz + az * np.sin(ph))], 1))  # fmt: skip
    V, F = loft(rings)
    V[:, :2] *= sxy
    return V, F


def make_mesh(name, V, F, mat, parent, subsurf=2, smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(V).tolist(), [], [list(f) for f in F])
    me.validate()
    me.polygons.foreach_set("use_smooth", np.full(len(me.polygons), smooth))
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if subsurf:
        sub = ob.modifiers.new("smooth", "SUBSURF")
        sub.levels, sub.render_levels = 1, subsurf
    ob.parent = parent
    return ob


# --- Materials ----------------------------------------------------------------------------
def principled(name, rgb, rough, fill=FILL, **inputs) -> bpy.types.Material:
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    for k, v in inputs.items():
        b.inputs[k].default_value = v
    if fill:  # ambient fill: the room env map alone under-exposes plastics vs the footage
        b.inputs["Emission Color"].default_value = (*rgb, 1.0)
        b.inputs["Emission Strength"].default_value = fill
    return m


LOOK = {  # sRGB base colours picked from the footage (headcam f0, exocam1 f30)
    "white": (0.92, 0.97, 0.96),  # mint-tinted white
    "aqua": (0.70, 0.92, 0.91),
    "teal": (0.15, 0.58, 0.53),
    "window": (0.10, 0.52, 0.48),
    "steel": (0.80, 0.80, 0.82),
    "cord": (0.90, 0.90, 0.88),
}


def shell_material(sxy) -> bpy.types.Material:
    """Glossy white plastic with clearcoat and teal accents (circle, wedge) placed procedurally."""
    m = principled("iron_shell_white", srgb(LOOK["white"]), 0.22,
                   **{"Coat Weight": 0.6, "Coat Roughness": 0.04})  # fmt: skip
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])

    def ellipse(cx, cy, ax, ay, rot=0.0):
        """1 inside the ellipse (soft 0.4 mm edge), 0 outside; also needs z > 3.5 cm."""
        c, s = np.cos(rot), np.sin(rot)
        dx = nt.nodes.new("ShaderNodeMath")
        dx.operation, dx.inputs[1].default_value = "SUBTRACT", cx * sxy
        nt.links.new(sep.outputs["X"], dx.inputs[0])
        dy = nt.nodes.new("ShaderNodeMath")
        dy.operation, dy.inputs[1].default_value = "SUBTRACT", cy * sxy
        nt.links.new(sep.outputs["Y"], dy.inputs[0])
        comb = nt.nodes.new("ShaderNodeCombineXYZ")
        nt.links.new(dx.outputs[0], comb.inputs["X"])
        nt.links.new(dy.outputs[0], comb.inputs["Y"])
        # rotate then scale into the unit circle
        rotn = nt.nodes.new("ShaderNodeVectorRotate")
        rotn.rotation_type = "Z_AXIS"
        rotn.inputs["Angle"].default_value = -rot
        nt.links.new(comb.outputs[0], rotn.inputs["Vector"])
        sc = nt.nodes.new("ShaderNodeVectorMath")
        sc.operation = "MULTIPLY"
        sc.inputs[1].default_value = (1 / (ax * sxy), 1 / (ay * sxy), 0)
        nt.links.new(rotn.outputs[0], sc.inputs[0])
        ln = nt.nodes.new("ShaderNodeVectorMath")
        ln.operation = "LENGTH"
        nt.links.new(sc.outputs[0], ln.inputs[0])
        mr = nt.nodes.new("ShaderNodeMapRange")
        edge = 0.0004 / (min(ax, ay) * sxy)
        mr.inputs["From Min"].default_value = 1 + edge
        mr.inputs["From Max"].default_value = 1 - edge
        nt.links.new(ln.outputs["Value"], mr.inputs["Value"])
        return mr.outputs["Result"]

    def add(a, b_):
        n = nt.nodes.new("ShaderNodeMath")
        n.operation, n.use_clamp = "ADD", True
        nt.links.new(a, n.inputs[0])
        nt.links.new(b_, n.inputs[1])
        return n.outputs[0]

    # Left shoulder: round teal spot; right shoulder: teal wedge (two overlapping ellipses).
    mask = ellipse(0.075, 0.030, 0.010, 0.010)
    mask = add(mask, ellipse(0.086, 0.0, 0.012, 0.009))
    mask = add(mask, ellipse(0.072, -0.030, 0.016, 0.008, rot=-0.45))
    mask = add(mask, ellipse(0.060, -0.034, 0.010, 0.007, rot=0.3))
    # Teal trim stripe around the heel (seen behind the hand in exocam1).
    zgate = nt.nodes.new("ShaderNodeMapRange")
    zgate.inputs["From Min"].default_value, zgate.inputs["From Max"].default_value = 0.036, 0.040
    nt.links.new(sep.outputs["Z"], zgate.inputs["Value"])
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    nt.links.new(mask, mul.inputs[0])
    nt.links.new(zgate.outputs["Result"], mul.inputs[1])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (*srgb(LOOK["white"]), 1)
    mix.inputs["B"].default_value = (*srgb(LOOK["teal"]), 1)
    nt.links.new(mul.outputs[0], mix.inputs["Factor"])
    nt.links.new(mix.outputs["Result"], b.inputs["Base Color"])
    nt.links.new(mix.outputs["Result"], b.inputs["Emission Color"])
    return m


def steel_material() -> bpy.types.Material:
    """Brushed steel: anisotropic metal with streaky roughness along the iron's length."""
    m = principled("iron_soleplate_steel", srgb(LOOK["steel"]), 0.22, fill=0.0,
                   **{"Metallic": 1.0, "Anisotropic": 0.6})  # fmt: skip
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (8.0, 1200.0, 1200.0)  # streaks along x
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1.0
    noise.inputs["Detail"].default_value = 6.0
    nt.links.new(mp.outputs["Vector"], noise.inputs["Vector"])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 0.15, 0.32
    nt.links.new(noise.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], b.inputs["Roughness"])
    tan = nt.nodes.new("ShaderNodeTangent")
    tan.direction_type, tan.axis = "RADIAL", "Z"
    nt.links.new(tan.outputs["Tangent"], b.inputs["Tangent"])
    return m


def materials(sxy) -> dict:
    return {
        "shell": shell_material(sxy),
        "aqua": principled("iron_aqua", srgb(LOOK["aqua"]), 0.25,
                           **{"Coat Weight": 0.4, "Coat Roughness": 0.08}),  # fmt: skip
        "handle": principled("iron_handle_white", srgb(LOOK["white"]), 0.28,
                             **{"Coat Weight": 0.4, "Coat Roughness": 0.08}),  # fmt: skip
        "teal": principled("iron_teal", srgb(LOOK["teal"]), 0.25, **{"Coat Weight": 0.5}),
        "window": principled("iron_window", srgb(LOOK["window"]), 0.08, fill=0.1,
                             **{"Transmission Weight": 0.65, "IOR": 1.49,
                                "Coat Weight": 0.8, "Coat Roughness": 0.02}),  # fmt: skip
        "steel": steel_material(),
        "cord": principled("iron_cord_white", srgb(LOOK["cord"]), 0.55),
        "dark": principled("iron_dial_mark", srgb((0.08, 0.30, 0.28)), 0.4),
    }


# --- Pose ---------------------------------------------------------------------------------
def palm_poses(xyz, names, bed_top) -> np.ndarray:
    """Fallback (04_build_scene.py): yaw-only pose under the right palm, sole on the bed."""
    idx = {n: i for i, n in enumerate(names)}
    T = np.repeat(np.eye(4)[None], len(xyz), 0)
    for f in range(len(xyz)):
        kn = xyz[f, [idx["right_index_mcp"], idx["right_middle_mcp"], idx["right_pinky_mcp"]]]
        palm = (xyz[f, idx["right_wrist"]] + kn.mean(0)) / 2
        fwd = kn.mean(0)[:2] - xyz[f, idx["right_wrist"], :2]
        yaw = np.arctan2(fwd[1], fwd[0])
        T[f, :3, :3] = Matrix.Rotation(yaw, 3, "Z")
        T[f, :3, 3] = (*palm[:2], max(palm[2] - 0.155, bed_top + 0.001))
    return T


def load_poses(n_frames, xyz, names, bed_top, snap_to_bed):
    """World poses (n,4,4), xy scale, and the source used."""
    if POSES.exists():
        P = np.load(POSES)
        T = P["T"][:n_frames].copy()
        if len(T) == n_frames:
            if snap_to_bed:  # move the whole track so the fitted bed plane == bed_top
                T[:, 2, 3] += bed_top - float(P["bed_z"])
            # Never sink below the bed (soleplate flat on it while ironing).
            T[:, 2, 3] = np.maximum(T[:, 2, 3], (bed_top if snap_to_bed else float(P["bed_z"])))
            return T, float(P["sxy"]), "poses.npz"
    return palm_poses(xyz, names, bed_top), 1.0, "palm"


# --- Cord ---------------------------------------------------------------------------------
def bed_frame(xyz, names):
    """Bed near edge and forward direction, as in 04_build_scene.py."""
    idx = {n: i for i, n in enumerate(names)}
    pelvis_xy = xyz[:, idx["pelvis"], :2].mean(0)
    hands = xyz[:, [idx["left_middle_mcp"], idx["right_middle_mcp"]]]
    fwd = hands[..., :2].reshape(-1, 2).mean(0) - pelvis_xy
    fwd /= np.linalg.norm(fwd)
    feet_xy = xyz[:, [idx["left_foot"], idx["right_foot"]], :2].reshape(-1, 2)
    near = pelvis_xy + fwd * (float(((feet_xy - pelvis_xy) @ fwd).max()) + 0.08)
    feet_z = float(xyz[:, [idx["left_foot"], idx["right_foot"]], 2].min()) - 0.03
    return near, fwd, feet_z


def cord_points(T, sxy, near, fwd, bed_top, floor_z, n: int = 40) -> np.ndarray:
    """Hand-shaped drape: out of the heel, across the bed to its near edge, down to the floor."""
    a_loc = np.array([(X_REAR - 0.018) * sxy, 0.0, 0.040, 1])
    d_loc = np.array([-1.0, 0.0, 0.3])
    A = (T @ a_loc)[:3]
    d = T[:3, :3] @ (d_loc / np.linalg.norm(d_loc))
    right = np.array([fwd[1], -fwd[0]])  # person's right, in the bed plane
    # Where the cord goes over the bed's near edge: beside the iron, towards the right.
    lat = float((A[:2] - near) @ right)
    E2 = near + right * (lat + 0.05)
    E = np.array([*E2, bed_top + CORD_R])
    # On the bed: quadratic Bezier from the heel to the edge, then clamp onto the cloth.
    Q = A + d * 0.12 + np.array([0, 0, 0.03])  # stiff cord arcs up out of the heel
    u = np.linspace(0, 1, n)[:, None]
    bed = (1 - u) ** 2 * A + 2 * u * (1 - u) * Q + u**2 * E
    sag = np.sin(np.pi * u[:, 0]) * 0.03
    bed[:, 2] = np.maximum(bed[:, 2] - sag, bed_top + CORD_R)
    bed[0] = A
    # Over the edge and down the side of the bed.
    m = n // 2
    v = np.linspace(0, 1, m + 1)[1:, None]
    out = np.r_[-fwd, 0.0]
    drop = E + out * (0.03 * np.sin(np.pi / 2 * v)) + np.array([0, 0, -1]) * (
        (bed_top - floor_z - 0.05) * v**1.3
    )
    drop[:, :2] += right * 0.06 * v
    return np.vstack([bed, drop])


def build_cord(scene, Ts, sxy, xyz, names, bed_top, mat) -> bpy.types.Object:
    near, fwd, floor_z = bed_frame(xyz, names)
    pts = [cord_points(T, sxy, near, fwd, bed_top, floor_z) for T in Ts]
    cu = bpy.data.curves.new("iron_cord", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth, cu.bevel_resolution = CORD_R, 4
    cu.resolution_u = 6
    cu.use_fill_caps = True
    sp = cu.splines.new("NURBS")
    sp.points.add(len(pts[0]) - 1)
    for p, c in zip(sp.points, pts[0]):
        p.co = (*c, 1.0)
    sp.order_u, sp.use_endpoint_u = 4, True
    cu.materials.append(mat)
    ob = bpy.data.objects.new("iron_cord", cu)
    scene.collection.objects.link(ob)
    ob.shape_key_add(name="basis")
    n = len(Ts)
    for f in range(n):
        key = ob.shape_key_add(name=f"f{f:04d}", from_mix=False)
        for kp, c in zip(key.data, pts[f]):
            kp.co = c
        for g, val in ((f - 1, 0.0), (f, 1.0), (f + 1, 0.0)):
            if 0 <= g < n:
                key.value = val
                key.keyframe_insert("value", frame=g)
    return ob


# --- Main entry ---------------------------------------------------------------------------
def build_iron(scene, n_frames, xyz, names, bed_top, snap_to_bed: bool = False) -> bpy.types.Object:
    """Build the CG iron (+ cord) in `scene`, keyed per frame; returns the root object "iron".

    Pose: out/assets/iron/poses.npz when present (fitted to the traced masks), else the
    palm-based placement of 04. snap_to_bed=True shifts the fitted track so its bed plane
    matches `bed_top`; the default keeps the image-fitted heights (poses' own bed_z).
    """
    Ts, sxy, src = load_poses(n_frames, xyz, list(names), bed_top, snap_to_bed)
    mats = materials(sxy)
    root = bpy.data.objects.new("iron", None)
    scene.collection.objects.link(root)
    root.empty_display_size = 0.1
    root.rotation_mode = "QUATERNION"

    make_mesh("iron_sole", *sole_geom(sxy), mats["steel"], root, subsurf=1)
    make_mesh("iron_body", *body_geom(sxy), mats["aqua"], root)
    make_mesh("iron_shell", *shell_geom(sxy), mats["shell"], root)
    make_mesh("iron_handle", *handle_geom(sxy), mats["handle"], root)
    make_mesh("iron_window", *window_geom(sxy), mats["window"], root)
    # Temperature dial at the front of the handle, tilted forward with the bridge.
    bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=0.014, depth=0.008)
    dial = bpy.context.object
    dial.name = "iron_dial"
    dial.location = (0.0545 * sxy, 0.0, 0.0873)
    dial.rotation_euler = (0.0, np.radians(52), 0.0)
    dial.data.materials.append(mats["handle"])
    bev = dial.modifiers.new("round", "BEVEL")
    bev.width, bev.segments = 0.002, 3
    for p in dial.data.polygons:
        p.use_smooth = True
    dial.parent = root
    bpy.ops.mesh.primitive_cube_add(size=1)
    mark = bpy.context.object
    mark.name = "iron_dial_mark"
    mark.scale = (0.010, 0.0025, 0.002)
    mark.location = (0.003, 0.0, 0.0045)
    mark.data.materials.append(mats["teal"])
    mark.parent = dial
    # Strain relief where the cord leaves the heel.
    bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.008, radius2=0.0045, depth=0.03)
    boot = bpy.context.object
    boot.name = "iron_cord_boot"
    boot.location = ((X_REAR - 0.004) * sxy, 0.0, 0.036)
    boot.rotation_euler = (0.0, np.radians(-90 + 17), 0.0)
    boot.data.materials.append(mats["handle"])
    for p in boot.data.polygons:
        p.use_smooth = True
    boot.parent = root
    for ob in root.children_recursive:
        ob.matrix_parent_inverse = Matrix.Identity(4)

    for f in range(n_frames):
        root.matrix_world = Matrix(Ts[f].tolist())
        root.keyframe_insert("location", frame=f)
        root.keyframe_insert("rotation_quaternion", frame=f)

    cord_bed = bed_top if snap_to_bed or src != "poses.npz" else float(np.load(POSES)["bed_z"])
    build_cord(scene, Ts, sxy, xyz, list(names), cord_bed, mats["cord"])
    print(f"cg_iron: {n_frames} frames, pose from {src}, sxy {sxy:.3f}")
    return root
