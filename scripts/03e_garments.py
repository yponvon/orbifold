"""Stage 03e: build loose garments (tunic + wide trousers) around the fitted body.

The real person wears a loose, short-sleeved greige tunic (brown trim on the cuffs,
placket and pocket edge) hanging to the upper thigh, and wide black trousers. Each garment
is cut from the Anny body surface, pushed out along the per-frame normals with a loose
fit (flaring towards the hem / ankles) so it follows the body every frame, and given pin
weights for Blender's cloth simulation (tight at shoulders/waist, free at the hems).

    uv run scripts/03e_garments.py
Writes out/body/garments.npz with, per garment: <g>_vertices (F,V,3), <g>_faces (T,3),
<g>_uv (T,3,2), <g>_pin (V,), <g>_trim (T,) bool.
"""

import argparse
from pathlib import Path

import numpy as np

BODY = Path("out/body")


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-9)


def smooth(values: np.ndarray, faces: np.ndarray, iters: int) -> np.ndarray:
    rows = np.concatenate([faces[:, [0, 0, 1, 1, 2, 2]].ravel()])
    cols = np.concatenate([faces[:, [1, 2, 0, 2, 0, 1]].ravel()])
    deg = np.bincount(rows, minlength=len(values)).clip(1)
    for _ in range(iters):
        values = 0.5 * values + 0.5 * np.bincount(rows, values[cols], len(values)) / deg
    return values


def ring(face_set: np.ndarray, faces: np.ndarray, seed_verts: np.ndarray, n: int) -> np.ndarray:
    """Faces in face_set within n rings of the seed vertices."""
    verts = set(seed_verts.tolist())
    hit = np.zeros(len(faces), bool)
    for _ in range(n):
        new = face_set & np.isin(faces, list(verts)).any(1)
        hit |= new
        verts |= set(faces[new].ravel().tolist())
    return hit


def build(fit, keep: np.ndarray, offset: np.ndarray, pin: np.ndarray, trim: np.ndarray) -> dict:
    """Cut faces `keep` out of the body, offset per vertex, re-index."""
    faces = fit["faces"][keep]
    used = np.unique(faces)
    remap = np.full(len(fit["rest_vertices"]), -1)
    remap[used] = np.arange(len(used))
    all_faces = fit["faces"]
    verts = []
    for f in range(len(fit["vertices"])):
        vn = vertex_normals(fit["vertices"][f], all_faces)
        verts.append(fit["vertices"][f][used] + vn[used] * offset[used, None])
    uv = fit["uv"][fit["face_uv"][keep]]
    return {"vertices": np.array(verts, np.float32), "faces": remap[faces], "uv": uv,
            "pin": pin[used], "trim": trim[keep]}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=str(BODY / "fit.npz"))
    ap.add_argument("--out", default=str(BODY / "garments.npz"))
    args = ap.parse_args()
    fit = np.load(args.fit)
    regions = [str(r) for r in fit["regions"]]
    labels = [str(b) for b in fit["bone_labels"]]
    region, faces, rest = fit["face_region"], fit["faces"], fit["rest_vertices"]
    heads = {n: fit["rest_bone_heads"][i] for i, n in enumerate(labels)}
    centre = rest[faces].mean(1)  # Anny rest frame: z up, -y forward, +x = person's left
    vb = np.array(labels)[fit["vertex_bone"]]
    face_bone = vb[faces[:, 0]]
    hip_z = heads["upperleg01.L"][2]
    hem_z = hip_z - 0.12  # tunic hangs to the upper thigh
    chest_z = heads["spine02"][2]
    knee_z, ankle_z = heads["lowerleg01.L"][2], heads["foot.L"][2]

    shirt, trousers = region == regions.index("shirt"), region == regions.index("trousers")
    arm_skin = (region == regions.index("skin")) & (np.char.find(face_bone, "arm") >= 0)

    # --- Tunic: shirt region + body down to the hem; flares below the chest ---------------
    tunic = shirt | (trousers & (centre[:, 2] > hem_z))
    # The real tunic's front ends in two triangular flaps (headcam): cut an inverted-V
    # opening at the front centre, OPEN_H tall at the middle, OPEN_W half-wide at the hem.
    OPEN_H, OPEN_W = 0.13, 0.13
    front_half = centre[:, 1] < heads["spine01"][1]
    v_top = hem_z + OPEN_H * np.clip(1 - np.abs(centre[:, 0]) / OPEN_W, 0, 1)
    tunic &= ~(front_half & (centre[:, 2] < v_top) & (np.abs(centre[:, 0]) < OPEN_W))
    z = rest[:, 2]
    # Close-fitting over the shoulders/chest (the head camera sees them from ~20 cm),
    # loosening towards the hem.
    off_t = 0.006 + 0.012 * np.clip((chest_z - z) / (chest_z - hem_z), 0, 1) ** 1.5
    sleeve = np.char.find(vb, "upperarm") >= 0
    off_t = np.where(sleeve, 0.012, off_t)
    # Loose drape: the real tunic hangs straight from the shoulder blades (and chest) instead
    # of following the lumbar hollow / buttocks. Per 2 cm x-column of the rest pose, build the
    # upper hull of the back profile y(z) below the blades (blade -> buttock peak line, then a
    # vertical drop) and push back vertices out to it; same at the front, at half strength.
    rn = vertex_normals(rest, faces)
    xb = np.round(rest[:, 0] / 0.02).astype(int)
    zs = np.arange(hem_z - 0.03, chest_z + 0.25, 0.01)
    zi = np.clip(np.round((z - zs[0]) / 0.01).astype(int), 0, len(zs) - 1)
    for sgn, gain, nmin in ((1.0, 1.0, 0.3), (-1.0, 0.5, 0.5)):  # back (+y), front (-y)
        side = (sgn * rn[:, 1] > nmin) & (np.abs(rest[:, 0]) < 0.2) & ~sleeve
        side &= (z > zs[0]) & (z < zs[-1])
        target = np.full(len(rest), -np.inf)
        for c in np.unique(xb[side]):
            m = side & (np.abs(xb - c) <= 1)
            prof = np.full(len(zs), -np.inf)
            np.maximum.at(prof, zi[m], sgn * rest[m, 1])
            up = zs >= chest_z
            if not np.isfinite(prof[up]).any():
                continue
            ib = np.flatnonzero(up)[np.argmax(prof[up])]  # shoulder blade / chest peak
            low = np.arange(ib + 1)
            ip = low[np.argmax(np.where(np.isfinite(prof[low]), prof[low], -np.inf))]
            hull = np.full(len(zs), prof[ib])
            if prof[ip] > prof[ib] and ip < ib:
                t = np.clip((zs - zs[ip]) / (zs[ib] - zs[ip]), 0, 1)
                hull = prof[ip] + (prof[ib] - prof[ip]) * t
            hull[ib:] = -np.inf  # above the blades: follow the body
            mc = side & (xb == c)
            target[mc] = hull[zi[mc]]
        gap = (target + 0.006 - sgn * rest[:, 1]) / np.abs(rn[:, 1]).clip(0.3)
        w = np.clip((sgn * rn[:, 1] - nmin) / 0.25, 0, 1)  # fade in towards the sides
        drape = np.clip(gain * w * np.nan_to_num(gap, neginf=0.0), 0, 0.08)
        off_t = np.where(side, np.maximum(off_t, drape), off_t)
    off_t = smooth(off_t, faces, 10)
    pin_t = np.clip((z - hem_z) / (chest_z - hem_z), 0, 1) ** 0.7  # free hem, pinned shoulders
    pin_t = np.where(sleeve, 0.5, pin_t)
    # Brown trim: sleeve cuffs (next to bare arm), front placket, pocket top edge.
    cuff = ring(tunic, faces, np.unique(faces[arm_skin]), 1)
    tri = rest[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    front = (fn / np.linalg.norm(fn, axis=1, keepdims=True).clip(1e-9))[:, 1] < -0.7  # faces -y
    placket = tunic & front & (np.abs(centre[:, 0]) < 0.015) & (centre[:, 2] > hip_z + 0.05)
    pocket = (tunic & front & (centre[:, 0] > 0.05) & (centre[:, 0] < 0.15)
              & (np.abs(centre[:, 2] - (chest_z + 0.06)) < 0.008))  # fmt: skip
    trim_t = cuff | placket | pocket
    print(f"trim faces: cuff {cuff.sum()}, placket {placket.sum()}, pocket {pocket.sum()}")
    # Smooth trim (shader mask, not per-face): per tunic vertex the rest position and the
    # distance to the sleeve opening. 04 draws the cuff band (CUFF_W) from `cuffd` and the
    # placket/pocket bands analytically from the interpolated rest coords, so the edges are
    # straight lines instead of triangle saw-teeth.
    tv_all = np.unique(faces[tunic])
    opening = np.intersect1d(tv_all, np.unique(faces[arm_skin]))
    from scipy.spatial import cKDTree

    cuffd = cKDTree(rest[opening]).query(rest[tv_all])[0] if len(opening) else np.full(len(tv_all), 9.0)
    trim_params = np.array([heads["spine01"][1], hip_z + 0.05, chest_z + 0.06], np.float32)

    # --- Wide trousers: legs below the tunic hem, widening towards the ankle -------------
    legs = trousers & (centre[:, 2] <= hem_z + 0.04)
    off_l = 0.012 + 0.025 * np.clip((knee_z - z) / (knee_z - ankle_z), 0, 1)
    off_l = smooth(off_l, faces, 10)
    pin_l = np.clip((z - knee_z) / (hip_z - knee_z), 0, 1) * 0.7 + 0.3
    no_trim = np.zeros(len(faces), bool)

    out = {}
    for name, g in (("tunic", build(fit, tunic, off_t, pin_t, trim_t)),
                    ("trousers", build(fit, legs, off_l, pin_l, no_trim))):  # fmt: skip
        out |= {f"{name}_{k}": v for k, v in g.items()}
        if name == "tunic":  # same vertex order as build(): np.unique of the kept faces
            out["tunic_rest"] = rest[tv_all].astype(np.float32)
            out["tunic_cuffd"] = cuffd.astype(np.float32)
            out["tunic_trim_params"] = trim_params  # front y, placket z min, pocket z
        print(f"{name}: {g['vertices'].shape[1]} vertices, {len(g['faces'])} faces, "
              f"{int(g['trim'].sum())} trim faces")  # fmt: skip
    # Buttons: 5 placket vertices from above the V opening up to the chest.
    tv = np.unique(fit["faces"][tunic])
    rest_t = rest[tv]
    vn_rest = vertex_normals(rest, faces)[tv]
    cand = (np.abs(rest_t[:, 0]) < 0.012) & (vn_rest[:, 1] < -0.5)
    levels = np.linspace(hem_z + 0.15, chest_z + 0.12, 5)
    buttons = [int(np.flatnonzero(cand)[np.argmin(np.abs(rest_t[cand, 2] - zl))]) for zl in levels]
    out["tunic_buttons"] = np.array(buttons)  # indices into the tunic's vertex list
    # Collar points: per side, three tunic vertices (neck side, shoulder-front, chest tip)
    # spanning a flat triangle that lies on the upper chest like a folded shirt collar.
    neck = heads["neck01"]
    collar = []
    for sgn in (1, -1):  # +x = her left
        targets = (
            neck + np.array([sgn * 0.06, 0.0, -0.01]),  # side of the neck
            neck + np.array([sgn * 0.10, -0.06, -0.06]),  # out over the collarbone, front
            neck + np.array([sgn * 0.015, -0.11, -0.12]),  # collar point, near the centre
        )
        ids = []
        for tgt in targets:
            d = np.linalg.norm(rest_t - tgt, axis=1)
            ids.append(int(np.argmin(np.where(vn_rest[:, 1] < 0.3, d, 9))))
        collar.append(ids)
    out["tunic_collar"] = np.array(collar)  # (2,3) tunic vertex indices
    np.savez(args.out, **out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
