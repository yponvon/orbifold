"""Offline asset preparation for the CG human (no bpy): skin maps, eyebrows, eyelashes.

    uv run python -m orbifold.cg_human_assets

Sources (CC0, MakeHuman community asset packs; see out/assets/cg_human/SOURCES.md),
unzipped under out/assets/cg_human/dl/:
  skins01     onlytheghosts_middle_aged_eurasian_female  (diffuse, hm08 UV layout)
  eyebrows01  mindfront_eyebrows_01                      (strand-card mesh + .mhclo)
  eyelashes01 mindfront_eyelashes_01                     (strand-card mesh + .mhclo)

Anny's body is MakeHuman's hm08 base mesh (same 21334 UVs, verified to 3e-8), so the
skins drop straight onto fit.npz's UVs, and the .mhclo proxies (which reference hm08 vertex
indices) are re-targeted through the UV-index correspondence fit vertex <-> hm08 vertex.

Writes out/assets/cg_human/:
  skin/albedo.png, skin/normal.png, skin/rough.png, skin/stats.json
  proxies.npz   brows_/lashes_ vertices (F,V,3), faces (quads, -1 padded), uv
"""

import json
from pathlib import Path

import cv2
import numpy as np

A = Path("out/assets/cg_human")
DL = A / "dl"
FIT = Path("out/body/fit.npz")
SKIN = DL / "skins01/skins/onlytheghosts_middle_aged_eurasian_female"
PROXIES = {
    "brows": DL / "eyebrows01/eyebrows/mindfront_eyebrows_01/mindfront_eyebrows_01.mhclo",
    "lashes": DL / "eyelashes01/eyelashes/mindfront_eyelashes_01/mindfront_eyelashes_01.mhclo",
}


def base_obj_path() -> Path:
    import anny

    return Path(anny.__file__).parent / "data/mpfb2/3dobjs/base.obj"


def read_obj(path: Path):
    """Vertices, UVs, faces as lists of (v, vt) index tuples (0-based), groups per face."""
    V, VT, F, G, g = [], [], [], [], ""
    for line in open(path):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            VT.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("g "):
            g = line.split()[1]
        elif line.startswith("f "):
            corners = []
            for c in line.split()[1:]:
                p = c.split("/")
                corners.append((int(p[0]) - 1, int(p[1]) - 1 if len(p) > 1 and p[1] else -1))
            F.append(corners)
            G.append(g)
    return np.array(V), np.array(VT), F, G


def fit_to_hm08(fit) -> np.ndarray:
    """hm08 vertex index -> fit vertex index (-1 where the fit mesh has no such vertex)."""
    _, _, F, _ = read_obj(base_obj_path())
    vt_to_hm = {}
    for face in F:
        for v, t in face:
            vt_to_hm[t] = v
    hm_to_fit = np.full(19158, -1)
    fv, fuv = fit["faces"].ravel(), fit["face_uv"].ravel()
    for v, t in zip(fv, fuv):
        hm_to_fit[vt_to_hm[int(t)]] = v
    return hm_to_fit


def tri_frame(p0, p1, p2):
    """Orthonormal frame (..., 3, 3) of triangles; rows are e1, e2, n."""
    e1 = p1 - p0
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    n = np.cross(p1 - p0, p2 - p0)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return np.stack([e1, np.cross(n, e1), n], -2)


def fit_proxy(mhclo: Path, fit, hm_to_fit) -> dict:
    """Place a .mhclo proxy on the rest body, then carry it to every frame through the
    local frames of its three reference triangles' vertices."""
    lines = open(mhclo).read().splitlines()
    scale, refs, obj = {}, [], None
    in_verts = False
    for ln in lines:
        s = ln.split()
        if not s:
            continue
        if s[0] in ("x_scale", "y_scale", "z_scale"):
            scale[s[0][0]] = (int(s[1]), int(s[2]), float(s[3]))
        elif s[0] == "obj_file":
            obj = mhclo.parent / s[1]
        elif s[0] == "verts":
            in_verts = True
        elif in_verts:
            if len(s) == 9:
                refs.append([float(x) for x in s])
            elif len(s) == 1 and s[0].isdigit():  # single-vertex reference
                refs.append([float(s[0]), float(s[0]), float(s[0]), 1, 0, 0, 0, 0, 0])
            else:
                in_verts = False
    refs = np.array(refs)
    idx = hm_to_fit[refs[:, :3].astype(int)]
    assert (idx >= 0).all(), "proxy references vertices missing from the fit mesh"
    w, off_mh = refs[:, 3:6], refs[:, 6:9]
    rest = fit["rest_vertices"].astype(float)
    # MakeHuman axes (x left, y up, z forward) -> Anny rest axes (x left, z up, -y forward)
    ax = {"x": 0, "y": 2, "z": 1}
    s = np.ones(3)
    for k, (a, b, ref) in scale.items():
        i = ax[k]
        pa, pb = rest[hm_to_fit[a]], rest[hm_to_fit[b]]
        s[i] = abs(pa[i] - pb[i]) / ref
    off = np.c_[off_mh[:, 0], -off_mh[:, 2], off_mh[:, 1]] * s
    tri_r = rest[idx]  # (P,3,3)
    p_rest = (w[:, :, None] * tri_r).sum(1) + off
    fr = tri_frame(tri_r[:, 0], tri_r[:, 1], tri_r[:, 2])
    loc = np.einsum("pij,pj->pi", fr, p_rest - tri_r[:, 0])  # in the triangle's frame
    out = []
    for f in range(len(fit["vertices"])):
        t = fit["vertices"][f].astype(float)[idx]
        frf = tri_frame(t[:, 0], t[:, 1], t[:, 2])
        out.append(t[:, 0] + np.einsum("pji,pj->pi", frf, loc))
    V, VT, F, _ = read_obj(obj)
    assert len(V) == len(refs)
    n = max(len(f) for f in F)
    faces = np.full((len(F), n), -1)
    fuv = np.zeros((len(F), n, 2), np.float32)
    for i, face in enumerate(F):
        faces[i, : len(face)] = [v for v, _ in face]
        fuv[i, : len(face)] = [VT[t] if t >= 0 else (0, 0) for _, t in face]
    return {"vertices": np.array(out, np.float32), "faces": faces, "uv": fuv}


def skin_maps(fit) -> None:
    """Albedo (as is, gain applied in the shader), a detail normal map derived from the
    albedo's high frequencies plus fine pores, and a roughness map (oilier T-zone)."""
    out = A / "skin"
    out.mkdir(parents=True, exist_ok=True)
    src = next(SKIN.glob("*diffuse.png"))
    img = cv2.imread(str(src))
    N = 2048
    img = cv2.resize(img, (N, N), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(out / "albedo.png"), img)
    # Mean linear colour over the texels the body actually uses (for the colour gain).
    uv = fit["uv"][fit["face_uv"][fit["face_region"] == 0]]
    mask = np.zeros((N, N), np.uint8)
    pts = np.c_[uv[..., 0] * N, (1 - uv[..., 1]) * N].reshape(-1, 3, 2).astype(np.int32)
    for p in pts: cv2.fillConvexPoly(mask, p, 255)
    lin = (img[..., ::-1].astype(np.float32) / 255) ** 2.2
    mean = lin[mask > 0].mean(0)
    # Height: albedo high-pass (darker = deeper: pores, creases) + fine pore noise.
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    hp = lum - cv2.GaussianBlur(lum, (0, 0), 6)
    rng = np.random.default_rng(0)
    pores = cv2.GaussianBlur(rng.standard_normal((N, N)).astype(np.float32), (0, 0), 1.2)
    pores = np.minimum(pores, 0) * 0.5  # pits
    h = hp * 4.0 + pores * 0.6
    h = cv2.GaussianBlur(h, (0, 0), 0.8)
    gx, gy = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3)
    nrm = np.dstack([-gx, gy, np.ones_like(h) * 0.5])  # tangent space, OpenGL (+Y up)
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    cv2.imwrite(str(out / "normal.png"), ((nrm * 0.5 + 0.5)[..., ::-1] * 255).astype(np.uint8))
    # Roughness: 0.5 base; slightly glossier where the albedo is brighter (forehead,
    # nose, cheeks in this texture) and rougher in high-frequency detail.
    r = 0.52 - 0.10 * (cv2.GaussianBlur(lum, (0, 0), 25) - lum.mean()) / max(lum.std(), 1e-3)
    r = np.clip(r + 2.0 * np.abs(hp), 0.35, 0.7)
    cv2.imwrite(str(out / "rough.png"), (r * 255).astype(np.uint8))
    json.dump({"source": str(src), "mean_linear": mean.tolist()}, open(out / "stats.json", "w"),
              indent=1)  # fmt: skip
    print(f"skin maps {N}px; mean linear albedo {np.round(mean, 3)}")


def main() -> None:
    fit = np.load(FIT)
    hm_to_fit = fit_to_hm08(fit)
    print(f"fit<->hm08: {int((hm_to_fit >= 0).sum())} of 19158 hm08 vertices mapped")
    skin_maps(fit)
    out = {}
    for name, path in PROXIES.items():
        p = fit_proxy(path, fit, hm_to_fit)
        out |= {f"{name}_{k}": v for k, v in p.items()}
        print(f"{name}: {p['vertices'].shape[1]} vertices, {len(p['faces'])} faces")
    np.savez_compressed(A / "proxies.npz", **out)
    print(f"wrote {A / 'proxies.npz'}")


if __name__ == "__main__":
    main()
