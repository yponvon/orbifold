"""CG human look for scripts/04_build_scene.py (bpy): textured SSS skin, eyebrows,
eyelashes, glossy dark hair. Assets come from orbifold.cg_human_assets (run it first);
every step is skipped gracefully if its asset is missing.
"""

import json
from pathlib import Path

import bpy
import numpy as np

A = Path("out/assets/cg_human")


def _img(path: Path, colour: bool):
    im = bpy.data.images.load(str(path.resolve()), check_existing=True)
    im.colorspace_settings.name = "sRGB" if colour else "Non-Color"
    return im


def skin_material(mat: bpy.types.Material, target_linear: tuple) -> bool:
    """Rebuild `mat` as MakeHuman CC0 skin texture (gained to the measured skin albedo),
    derived detail normal + roughness maps, random-walk subsurface scattering."""
    sk = A / "skin"
    if not (sk / "albedo.png").exists():
        return False
    mean = json.load(open(sk / "stats.json"))["mean_linear"]
    gain = [min(t / m, 2.0) for t, m in zip(target_linear, mean)]
    nt = mat.node_tree
    for n in list(nt.nodes):
        if n.type not in ("BSDF_PRINCIPLED", "OUTPUT_MATERIAL"):
            nt.nodes.remove(n)
    bsdf = nt.nodes["Principled BSDF"]
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = _img(sk / "albedo.png", True)
    tex.interpolation = "Cubic"
    mul = nt.nodes.new("ShaderNodeMix")
    mul.data_type, mul.blend_type = "RGBA", "MULTIPLY"
    mul.inputs["Factor"].default_value = 1.0
    mul.inputs["B"].default_value = (*gain, 1.0)
    nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], mul.inputs["A"])
    nt.links.new(mul.outputs["Result"], bsdf.inputs["Base Color"])
    rough = nt.nodes.new("ShaderNodeTexImage")
    rough.image = _img(sk / "rough.png", False)
    nt.links.new(uv.outputs["UV"], rough.inputs["Vector"])
    nt.links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])
    ntex = nt.nodes.new("ShaderNodeTexImage")
    ntex.image = _img(sk / "normal.png", False)
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nmap.uv_map = "UVMap"
    nmap.inputs["Strength"].default_value = 0.25
    nt.links.new(uv.outputs["UV"], ntex.inputs["Vector"])
    nt.links.new(ntex.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.subsurface_method = "RANDOM_WALK_SKIN"
    bsdf.inputs["Subsurface Weight"].default_value = 0.35
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.4, 0.22)
    bsdf.inputs["Subsurface Scale"].default_value = 0.006
    bsdf.inputs["Specular IOR Level"].default_value = 0.45
    bsdf.inputs["Coat Weight"].default_value = 0.08  # thin sebum layer
    bsdf.inputs["Coat Roughness"].default_value = 0.35
    print(f"cg_human: textured skin, gain {np.round(gain, 3)}")
    return True


def _keyed_mesh(scene, name, V, faces, uv, mat, n_frames):
    me = bpy.data.meshes.new(name)
    polys = [[int(i) for i in f if i >= 0] for f in faces]
    me.from_pydata(V[0].tolist(), [], polys)
    loops_uv = np.concatenate([uv[i, : len(p)] for i, p in enumerate(polys)])
    me.uv_layers.new(name="UVMap").data.foreach_set("uv", loops_uv.ravel())
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    scene.collection.objects.link(ob)
    ob["cg_head_only"] = True  # hidden from the head camera (sits inside the head)
    ob.shape_key_add(name="basis")
    for f in range(n_frames):
        key = ob.shape_key_add(name=f"f{f:04d}", from_mix=False)
        key.data.foreach_set("co", V[f].reshape(-1))
        for g, val in ((f - 1, 0.0), (f, 1.0), (f + 1, 0.0)):
            if 0 <= g < n_frames:
                key.value = val
                key.keyframe_insert("value", frame=g)
    return ob


def hair_strand_material(name: str, melanin: float = 0.95) -> bpy.types.Material:
    """Dark, slightly glossy hair: Principled BSDF with anisotropic-ish low roughness
    (works on mesh cards and on curves)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.012, 0.010, 0.009, 1)
    b.inputs["Roughness"].default_value = 0.38
    b.inputs["Specular IOR Level"].default_value = 0.6
    b.inputs["Coat Weight"].default_value = 0.25
    b.inputs["Coat Roughness"].default_value = 0.25
    return m


def brows_lashes(scene, n_frames: int) -> None:
    p = A / "proxies.npz"
    if not p.exists():
        return
    P = np.load(p)
    mat = hair_strand_material("cg_brow_lash")
    for name in ("brows", "lashes"):
        _keyed_mesh(scene, f"cg_{name}", P[f"{name}_vertices"], P[f"{name}_faces"],
                    P[f"{name}_uv"], mat, n_frames)  # fmt: skip
    print("cg_human: eyebrows + eyelashes (MakeHuman CC0 Mindfront proxies)")


def hair_look(mat: bpy.types.Material) -> None:
    """Scalp/bun hair: near-black, slightly glossy, fine strand bump along the head."""
    nt = mat.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.014, 0.012, 0.011, 1)
    b.inputs["Roughness"].default_value = 0.32
    b.inputs["Specular IOR Level"].default_value = 0.7
    b.inputs["Coat Weight"].default_value = 0.3
    b.inputs["Coat Roughness"].default_value = 0.2
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type, wave.bands_direction = "BANDS", "Z"
    wave.inputs["Scale"].default_value = 400.0
    wave.inputs["Distortion"].default_value = 6.0
    wave.inputs["Detail"].default_value = 4.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    bump.inputs["Distance"].default_value = 0.0005
    nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
