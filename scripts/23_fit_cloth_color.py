"""Stage 23: global colour fit of the CG garments (Track B).

Fits ONE linear base colour per garment (shared by all cameras) so the median rendered
garment colour matches the median real one. Only global statistics are taken from the
footage; no footage pixels go into the render.

    # 1. a flat-ID copy of the scene (tunic red, trousers green, rest black; no shadows)
    uv run --extra render scripts/23_fit_cloth_color.py mask --blend out/scene_color.blend
    uv run --extra render scripts/05_render.py --blend out/color_check/mask.blend \
        --out out/color_check/mask --samples 1 --frames 30 30 --cams headcam exocam1 exocam2
    # 2. medians (real: footage inside our garment mask & the person mask), and the fit
    uv run scripts/23_fit_cloth_color.py measure --render out/color_check/render [--write]
--write saves out/body/cloth_color_fit.json (read by 04_build_scene.py).
"""

import argparse
import json
import sys
from pathlib import Path

OUT = Path("out")
CAMS = ["headcam", "exocam1", "exocam2"]
GARMENTS = {"shirt": ("cloth_tunic", (1, 0, 0)), "trousers": ("cloth_trousers", (0, 1, 0))}
FIT = OUT / "body" / "cloth_color_fit.json"


def make_mask_blend(blend: str, out: str) -> None:
    import bpy

    bpy.ops.wm.open_mainfile(filepath=str(Path(blend).resolve()))
    ids = {mat: col for mat, col in GARMENTS.values()}
    for m in bpy.data.materials:
        if not m.use_nodes:
            m.use_nodes = True
        nt = m.node_tree
        for n in list(nt.nodes):
            if n.bl_idname == "ShaderNodeOutputMaterial":
                nt.nodes.remove(n)
        em, o = nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
        em.inputs["Color"].default_value = (*ids.get(m.name, (0, 0, 0)), 1)
        nt.links.new(em.outputs[0], o.inputs["Surface"])
    for ob in bpy.data.objects:
        if getattr(ob, "is_shadow_catcher", False):
            ob.hide_render = True
    if bpy.context.scene.world:
        bpy.context.scene.world.use_nodes = False
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(out).resolve()))
    print("wrote", out)


def lin(c):
    return [v**2.2 for v in c]


def measure(render: Path, mask_dir: Path, frames: list[int], need_render: bool = True) -> dict:
    import cv2
    import numpy as np

    k = np.ones((7, 7), np.uint8)
    res = {g: {} for g in GARMENTS}
    for cam in CAMS:
        for g, (_, col) in GARMENTS.items():
            real_px, ours_px = [], []
            for f in frames:
                idm = cv2.imread(str(mask_dir / cam / f"{f:04d}.png"), cv2.IMREAD_UNCHANGED)
                if idm is None:
                    continue
                ch = {0: 2, 1: 1}[col.index(1)]  # BGR channel of the ID colour
                gm = (idm[..., ch] > 200) & (idm[..., 3] > 250)
                gm &= idm[..., 3 - ch if ch != 1 else 2] < 50
                gm = cv2.erode(gm.astype(np.uint8), k) > 0
                real = cv2.imread(str(OUT / "clip/frames" / cam / f"{f:04d}.jpg"))
                pm = cv2.imread(str(OUT / "assets/human_gs/masks" / cam / f"{f:04d}.png"), 0)
                if pm is not None:
                    pm = cv2.erode((pm > 127).astype(np.uint8), k) > 0
                    real_px.append(real[gm & pm])
                p = render / cam / f"{f:04d}.png"
                rgba = cv2.imread(str(p), cv2.IMREAD_UNCHANGED) if need_render and p.exists() else None
                if rgba is not None:
                    ours_px.append(rgba[..., :3][gm & (rgba[..., 3] > 250)])
            med = lambda px: (np.median(np.vstack(px), 0)[::-1] / 255).round(3).tolist()  # noqa: E731
            if real_px and sum(len(p) for p in real_px) > 200:
                res[g].setdefault(cam, {})["real"] = med(real_px)
            if ours_px and sum(len(p) for p in ours_px) > 200:
                res[g].setdefault(cam, {})["ours"] = med(ours_px)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["mask", "measure"])
    ap.add_argument("--blend", default=str(OUT / "scene_color.blend"))
    ap.add_argument("--render", default=str(OUT / "color_check/render"))
    ap.add_argument("--mask", default=str(OUT / "color_check/mask"))
    ap.add_argument("--real-frames", type=int, nargs="+", default=[10, 30, 50])
    ap.add_argument("--frames", type=int, nargs="+", default=[30])
    ap.add_argument("--garments", nargs="+", default=["shirt"], help="which fits to save")
    ap.add_argument("--write", action="store_true", help="save the fitted albedo for 04")
    args = ap.parse_args(sys.argv[1:] if "--" not in sys.argv else sys.argv[sys.argv.index("--") + 1 :])
    if args.mode == "mask":
        make_mask_blend(args.blend, str(OUT / "color_check/mask.blend"))
        return
    import numpy as np

    real = measure(Path(args.render), Path(args.mask), args.real_frames, need_render=False)
    ours = measure(Path(args.render), Path(args.mask), args.frames)
    used = json.load(open(OUT / "body/albedo_used.json"))
    fit = {}
    report = {}
    for g in GARMENTS:
        cams = [c for c in CAMS if "real" in real[g].get(c, {}) and "ours" in ours[g].get(c, {})]
        if not cams:
            continue
        R = np.array([lin(real[g][c]["real"]) for c in cams])
        O = np.array([lin(ours[g][c]["ours"]) for c in cams])
        # Least squares in log space: each camera's ratio counts equally (exocam2's warm,
        # bright exposure must not dominate), then a hue-preserving cap at albedo 0.9.
        scale = np.exp(np.log(np.maximum(R, 1e-5) / np.maximum(O, 1e-5)).mean(0))
        new = np.array(used[g]) * scale
        new = np.clip(new * min(1.0, 0.9 / new.max()), 0.003, 0.9)
        fit[g] = new.round(4).tolist()
        report[g] = {c: {"real": real[g][c]["real"], "ours": ours[g][c]["ours"]} for c in cams}
        print(f"{g}: albedo {np.round(used[g], 4).tolist()} -> {fit[g]}  (scale {scale.round(3)})")
        for c in cams:
            print(f"   {c:8s} real sRGB {real[g][c]['real']}   ours {ours[g][c]['ours']}")
    json.dump(report, open(Path(args.render).parent / "medians.json", "w"), indent=1)
    fit = {g: v for g, v in fit.items() if g in args.garments}
    if args.write:
        json.dump({"albedo_linear": fit, "medians": report}, open(FIT, "w"), indent=1)
        print("wrote", FIT)


if __name__ == "__main__":
    main()
