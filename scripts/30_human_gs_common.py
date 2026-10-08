"""Shared helpers for the mesh-anchored human Gaussians (scripts/30_human_gs_*.py)."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orbifold.geometry import CONVENTIONS  # noqa: E402

CLIP = Path("out/clip")
OUT = Path("out/assets/human_gs")
CAMS = ["headcam", "exocam1", "exocam2"]
N_FRAMES = 60


def ros_to_cv(T_world_cam: np.ndarray) -> np.ndarray:
    """ros_body-axes camera pose -> OpenCV world-to-camera view matrix (as in stage 08)."""
    M = np.eye(4)
    M[:3, :3] = CONVENTIONS["ros_body"]
    return np.linalg.inv(T_world_cam @ M)


def load_cameras():
    """Returns {cam: (K (3,3), viewmats (60,4,4), (W,H))}."""
    C = np.load(CLIP / "cameras.npz")
    out = {}
    for c in CAMS:
        out[c] = (C[f"{c}_K"], np.stack([ros_to_cv(T) for T in C[f"{c}_T"]]), tuple(C[f"{c}_size"]))
    return out


def load_mesh():
    """Union of body + tunic + trousers. Returns verts (60,V,3), faces (F,3), part (F,) with
    0..6 = body regions (skin, shirt, trousers, hair, shoes, strap, eye), 7 = tunic, 8 = trousers
    garment."""
    B = np.load("out/body/fit.npz")
    G = np.load("out/body/garments.npz")
    vs = [B["vertices"], G["tunic_vertices"], G["trousers_vertices"]]
    fs, parts, off = [], [], 0
    for v, f, p in [
        (B["vertices"], B["faces"], B["face_region"]),
        (G["tunic_vertices"], G["tunic_faces"], np.full(len(G["tunic_faces"]), 7)),
        (G["trousers_vertices"], G["trousers_faces"], np.full(len(G["trousers_faces"]), 8)),
    ]:
        fs.append(f + off)
        parts.append(p)
        off += v.shape[1]
    return np.concatenate(vs, 1).astype(np.float32), np.concatenate(fs), np.concatenate(parts)


def project_cv(pts: np.ndarray, K: np.ndarray, viewmat: np.ndarray):
    p = pts @ viewmat[:3, :3].T + viewmat[:3, 3]
    z = p[:, 2]
    uv = (p @ K.T)[:, :2] / np.maximum(z[:, None], 1e-4)
    return uv, z


def mesh_silhouette(verts, faces, K, viewmat, size, near=0.05):
    W, H = size
    uv, z = project_cv(verts, K, viewmat)
    ok = (z[faces] > near).all(1)
    tri = uv[faces[ok]]
    keep = np.isfinite(tri).all((1, 2)) & (np.abs(tri) < 1e5).all((1, 2))
    tri = np.round(tri[keep] * 4).astype(np.int32)  # 2 bits of sub-pixel precision
    m = np.zeros((H, W), np.uint8)
    for t in tri:  # per triangle: fillPoly on many polygons uses even-odd filling
        cv2.fillConvexPoly(m, t, 255, lineType=cv2.LINE_8, shift=2)
    return m


# ----------------------------------------------------------------------------------------------
# Mesh-anchored Gaussian model
# ----------------------------------------------------------------------------------------------
import math  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

C0 = 0.28209479177387814
# Initial colours (sRGB) per part: skin, shirt, trousers, hair, shoes, strap, eye, tunic, trousers.
PART_RGB = np.array(
    [[0.62, 0.47, 0.40], [0.74, 0.71, 0.65], [0.07, 0.07, 0.08], [0.05, 0.05, 0.05],
     [0.10, 0.10, 0.10], [0.05, 0.05, 0.05], [0.10, 0.08, 0.08], [0.74, 0.71, 0.65],
     [0.07, 0.07, 0.08]], np.float32)  # fmt: skip
TRIM_RGB = np.array([0.30, 0.22, 0.16], np.float32)
# Gaussians per triangle: dense on the coarse garment meshes, sparser on the fine body mesh.
K_PER_PART = [2, 1, 1, 2, 1, 2, 2, 6, 6]
OFF_CAP, RES_CAP, SCALE_CAP = 0.04, 0.02, 0.025  # metres


def dct_basis(n_frames: int, n_basis: int, device) -> torch.Tensor:
    """(n_frames, n_basis) low-frequency cosine basis; smooth in time so held-out frames
    get interpolated residuals rather than zeros."""
    t = (torch.arange(n_frames, device=device).float() + 0.5) / n_frames
    b = torch.arange(n_basis, device=device).float()
    return torch.cos(math.pi * t[:, None] * b[None])


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
         2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
         2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1,
    ).reshape(*q.shape[:-1], 3, 3)  # fmt: skip


class HumanGS(torch.nn.Module):
    def __init__(self, verts: np.ndarray, faces: np.ndarray, part: np.ndarray,
                 trim: np.ndarray | None = None, n_basis: int = 10, seed: int = 0,
                 capped: bool = True):  # fmt: skip
        super().__init__()
        # v2 ("capped"): static offset <= OFF_CAP, per-frame residual <= RES_CAP (soft tanh caps),
        # every scale <= SCALE_CAP. v1 checkpoints were trained without caps.
        self.capped = capped
        rng = np.random.default_rng(seed)
        k = np.array(K_PER_PART)[part]
        tri = np.repeat(np.arange(len(faces)), k)
        r1, r2 = rng.random(len(tri)), rng.random(len(tri))
        s = np.sqrt(r1)
        bary = np.stack([1 - s, s * (1 - r2), s * r2], 1).astype(np.float32)
        n_frames = verts.shape[0]
        self.register_buffer("verts", torch.from_numpy(verts))
        self.register_buffer("faces", torch.from_numpy(faces).long())
        self.register_buffer("tri", torch.from_numpy(tri).long())
        self.register_buffer("bary", torch.from_numpy(bary))
        self.register_buffer("basis", dct_basis(n_frames, n_basis, "cpu"))
        N = len(tri)
        # Initial in-plane scale from the triangle size (frame 0).
        v = verts[0][faces]
        area = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
        s_t = np.sqrt(area[tri] / k[tri]).clip(2e-3, 0.03) * 0.7
        log_s = np.log(np.stack([s_t, s_t, np.full_like(s_t, 2e-3)], 1)).astype(np.float32)
        rgb = PART_RGB[part[tri]]
        if trim is not None:
            rgb[trim[tri]] = TRIM_RGB
        P = torch.nn.Parameter
        self.offset = P(torch.zeros(N, 3))  # static offset in the triangle frame (m)
        self.resid = P(torch.zeros(n_basis, N, 3))  # per-frame residual, DCT coefficients
        self.global_t = P(torch.zeros(n_basis, 3))  # smooth whole-body translation per frame
        self.quat = P(torch.tensor([1.0, 0, 0, 0]).repeat(N, 1))  # in the triangle frame
        self.log_scale = P(torch.from_numpy(log_s))
        self.opacity = P(torch.logit(torch.full((N,), 0.7)))
        self.sh0 = P(torch.from_numpy((rgb - 0.5) / C0)[:, None])
        self.shN = P(torch.zeros(N, 3, 3))  # SH degree 1

    def tri_frames(self, f: int):
        v = self.verts[f][self.faces]  # (F,3,3)
        e1, e2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
        n = F.normalize(torch.cross(e1, e2, dim=-1), dim=-1)
        t = F.normalize(e1, dim=-1)
        b = torch.cross(n, t, dim=-1)
        R = torch.stack([t, b, n], -1)  # columns = local axes
        return v, R

    def residual_raw(self, f: int) -> torch.Tensor:
        return torch.einsum("b,bnc->nc", self.basis[f], self.resid)

    def residual(self, f: int) -> torch.Tensor:
        r = self.residual_raw(f)
        return RES_CAP * torch.tanh(r / RES_CAP) if self.capped else r

    def static_offset(self) -> torch.Tensor:
        o = self.offset
        return OFF_CAP * torch.tanh(o / OFF_CAP) if self.capped else o

    def scales(self) -> torch.Tensor:
        S = torch.exp(self.log_scale)
        return S.clamp(max=SCALE_CAP) if self.capped else S

    def gaussians(self, f: int):
        v, R = self.tri_frames(f)
        R = R[self.tri]
        center = torch.einsum("nk,nkc->nc", self.bary, v[self.tri])
        local = self.static_offset() + self.residual(f)
        means = center + torch.einsum("nij,nj->ni", R, local) + self.basis[f] @ self.global_t
        rot = R @ quat_to_mat(self.quat)
        M = rot * self.scales()[:, None, :]
        covars = M @ M.transpose(1, 2)
        sh = torch.cat([self.sh0, self.shN], 1)
        return means, covars, torch.sigmoid(self.opacity), sh


class CamColor(torch.nn.Module):
    """Per-camera 3x3 + bias colour response and a smooth per-frame exposure gain."""

    def __init__(self, n_cams: int, n_frames: int, n_basis: int = 6):
        super().__init__()
        self.color = torch.nn.Parameter(torch.eye(3).repeat(n_cams, 1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(n_cams, 3))
        self.gain = torch.nn.Parameter(torch.zeros(n_cams, n_basis))
        self.register_buffer("basis", dct_basis(n_frames, n_basis, "cpu"))

    def forward(self, rgb: torch.Tensor, alpha: torch.Tensor, cam: int, f: int):
        """rgb is premultiplied by alpha (black background); keep it premultiplied."""
        out = rgb @ self.color[cam].T + self.bias[cam] * alpha
        return out * torch.exp(self.basis[f] @ self.gain[cam])


def render(model: HumanGS, f: int, viewmat: torch.Tensor, K: torch.Tensor, W: int, H: int,
           sh_degree: int = 1, render_mode: str = "RGB", absgrad: bool = False):  # fmt: skip
    from gsplat import rasterization

    means, covars, opac, sh = model.gaussians(f)
    rgb, alpha, info = rasterization(
        means, None, None, opac, sh, viewmat[None], K[None], W, H, covars=covars,
        sh_degree=sh_degree, packed=False, near_plane=0.02, render_mode=render_mode,
        rasterize_mode=getattr(model, "raster_mode", "classic"), absgrad=absgrad,
    )  # fmt: skip
    return rgb[0], alpha[0], info


def crop_K(K: torch.Tensor, x0: int, y0: int) -> torch.Tensor:
    K = K.clone()
    K[0, 2] -= x0
    K[1, 2] -= y0
    return K


def ssim_map(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SSIM map of (B,3,H,W) images, 11x11 Gaussian window (same as stage 08)."""
    g = torch.exp(-((torch.arange(11, device=a.device) - 5) ** 2) / (2 * 1.5**2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None]
    w = g.expand(3, 1, 11, 11)

    def blur(x):
        return F.conv2d(x, w, padding=5, groups=3)

    mu_a, mu_b = blur(a), blur(b)
    s_a, s_b = blur(a * a) - mu_a**2, blur(b * b) - mu_b**2
    s_ab = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (s_a + s_b + c2)
    )


def load_model(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    verts, faces, part = load_mesh()
    trim = load_trim()
    if ck.get("version", 2) >= 3:
        n = ck["model"]["tri"].shape[0]
        model = HumanGS3(verts, faces, part, trim, n_basis=ck["n_basis"], n_gauss=n).to(device)
        model.load_state_dict(ck["model"], strict=False)
        model.pose = CamPose().to(device)
        model.pose.load_state_dict(ck["pose"])
        model.raster_mode = ck.get("raster", "antialiased")
    else:
        model = HumanGS(verts, faces, part, trim, n_basis=ck["n_basis"],
                        capped=ck.get("capped", False)).to(device)  # fmt: skip
        model.load_state_dict(ck["model"], strict=False)
    # Always take the latest mesh from disk (fit.npz is re-aligned in place).
    model.verts = torch.from_numpy(verts).to(device)
    cc = CamColor(len(CAMS), N_FRAMES).to(device)
    cc.load_state_dict(ck["camcolor"])
    return model, cc


def load_trim() -> np.ndarray:
    """(F,) bool: brown trim faces of the garments (False on the body)."""
    B = np.load("out/body/fit.npz")
    G = np.load("out/body/garments.npz")
    return np.concatenate([np.zeros(len(B["faces"]), bool), G["tunic_trim"], G["trousers_trim"]])


def is_heldout(f: int) -> bool:
    return f % 10 == 0


ERODE_PX = 3  # the loss uses the traced mask eroded by this many pixels


IRON = Path("out/assets/iron/masks")
IRON_DILATE_PX = 5


def disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)


def load_iron(cam: str, f: int, dilate: int = IRON_DILATE_PX) -> np.ndarray:
    """The iron agent's iron mask (dilated); all False if missing."""
    p = IRON / cam / f"{f:04d}.png"
    if not p.exists():
        return np.zeros((1080, 1920), bool)
    m = cv2.imread(str(p), 0)
    return (cv2.dilate(m, disk(dilate)) if dilate else m) > 0


def load_mask(cam: str, f: int, erode: int = 0, iron: bool = True) -> np.ndarray:
    """Traced person mask (YOLO); minus the dilated iron (its own layer) when iron=True."""
    m = cv2.imread(str(OUT / "masks" / cam / f"{f:04d}.png"), 0)
    if iron:
        m = m * ~load_iron(cam, f)
    if erode:
        m = cv2.erode(m, disk(erode))
    return m > 0


def bbox(m: np.ndarray, pad: int = 24):
    ys, xs = np.nonzero(m)
    H, W = m.shape
    if len(xs) == 0:
        return 0, 0, W, H
    return (max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0),
            min(int(xs.max()) + pad, W), min(int(ys.max()) + pad, H))  # fmt: skip


@torch.no_grad()
def score(pred: np.ndarray, real: np.ndarray, m: np.ndarray, lp, dev="cuda") -> dict:
    """Person-region scores. pred: premultiplied render over black (H,W,3) in [0,1];
    real: original frame in [0,1]; m: traced person mask. Both images are cut out by the
    real mask and compared on its bounding box."""
    x0, y0, x1, y1 = bbox(m)
    mc = m[y0:y1, x0:x1, None].astype(np.float32)
    gt_c, pr_c = real[y0:y1, x0:x1] * mc, pred[y0:y1, x0:x1] * mc
    mse = ((gt_c - pr_c) ** 2).sum() / max(mc.sum() * 3, 1)
    g = torch.from_numpy(np.ascontiguousarray(gt_c)).permute(2, 0, 1)[None].to(dev)
    q = torch.from_numpy(np.ascontiguousarray(pr_c)).permute(2, 0, 1)[None].to(dev)
    smap = ssim_map(q, g).mean(1)[0].cpu().numpy()
    inner = cv2.erode(m[y0:y1, x0:x1].astype(np.uint8), disk(5)) > 0
    return {"lpips": float(lp(q * 2 - 1, g * 2 - 1).item()),
            "psnr": float(-10 * np.log10(max(mse, 1e-10))),
            "ssim": float((smap * mc[..., 0]).sum() / max(mc.sum(), 1)),
            "sharp": lapvar(pr_c, inner) / max(lapvar(gt_c, inner), 1e-12)}  # fmt: skip


def lapvar(img: np.ndarray, m: np.ndarray) -> float:
    """Variance of the Laplacian of the grey image inside m (sharpness)."""
    g = cv2.cvtColor(np.ascontiguousarray(img, dtype=np.float32), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return float(lap[m].var()) if m.any() else 0.0


def render_full(model, cc, cams, ci: int, f: int, dev="cuda", with_depth: bool = False):
    """Full-resolution render of camera ci at frame f -> (premultiplied rgb, alpha) numpy."""
    K, vms, (W, H) = cams[CAMS[ci]]
    Kt = torch.tensor(K, dtype=torch.float32, device=dev)
    vm = torch.tensor(vms[f], dtype=torch.float32, device=dev)
    with torch.no_grad():
        vm = model_viewmat(model, ci, f, vm)
        out, alpha, _ = render(model, f, vm, Kt, W, H, render_mode="RGB+ED")
        a = alpha.clamp(0, 1)
        pred = torch.minimum(cc(out[..., :3], alpha, ci, f).clamp(0, 1), a)
        depth = torch.where(a[..., 0] >= 0.5, out[..., 3], torch.full_like(out[..., 3], torch.inf))
    if with_depth:
        return pred.cpu().numpy(), a.cpu().numpy(), depth.cpu().numpy()
    return pred.cpu().numpy(), a.cpu().numpy()


def load_real(cam: str, f: int) -> np.ndarray:
    im = cv2.imread(str(CLIP / "frames" / cam / f"{f:04d}.jpg"))[..., ::-1]
    return im.astype(np.float32) / 255


def write_preview(model, cc, cams, tag: str, lp, frame: int = 30, dev="cuda") -> dict:
    """preview_<tag>.jpg: one row per camera at `frame`, full resolution, tight crop:
    [ORIGINAL cut-out] | [OURS cut-out (own alpha)], LPIPS in the person region on ours."""
    rows, lps = [], {}
    font = cv2.FONT_HERSHEY_SIMPLEX
    for ci, cam in enumerate(CAMS):
        real = load_real(cam, frame)
        m = load_mask(cam, frame)
        pred, a = render_full(model, cc, cams, ci, frame, dev)
        s = score(pred, real, m, lp, dev)
        lps[cam] = s["lpips"]
        x0, y0, x1, y1 = bbox(m | (a[..., 0] > 0.5), 16)
        left = (real * m[..., None])[y0:y1, x0:x1]
        right = pred[y0:y1, x0:x1]
        panels = []
        for img, label in [(left, f"ORIGINAL {cam} f{frame}"),
                           (right, f"OURS  LPIPS {s['lpips']:.3f}  PSNR {s['psnr']:.1f}")]:  # fmt: skip
            p = (img[..., ::-1] * 255 + 0.5).astype(np.uint8).copy()
            (tw, _), _ = cv2.getTextSize(label, font, 0.8, 2)
            p = np.pad(p, ((40, 0), (0, max(tw + 16 - p.shape[1], 0)), (0, 0)))
            cv2.putText(p, label, (8, 28), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(p)
        h = panels[0].shape[0]
        rows.append(np.hstack([panels[0], np.full((h, 8, 3), 128, np.uint8), panels[1]]))
    wmax = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 8), (0, wmax - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    cv2.imwrite(str(OUT / f"preview_{tag}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return lps


# ----------------------------------------------------------------------------------------------
# Room splats (stage 08/09) for occlusion: carving visibility and the joint composite
# ----------------------------------------------------------------------------------------------
class Room:
    """The trained room Gaussians plus the per-camera pose corrections that stage 09 renders
    with. Room Gaussians are expressed in the human camera frame on demand, so one
    rasterisation call can sort human and room Gaussians together."""

    def __init__(self, dev="cuda"):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "room_train", Path(__file__).with_name("08_train_room.py")
        )
        rt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rt)
        ck = torch.load("out/room/splats.pt", map_location=dev, weights_only=False)
        p = ck["params"]
        D = np.load("out/room/data.npz")
        self.cam_id, self.frame_idx = D["cam_id"], D["frame_idx"]
        self.cam_names = list(ck["cam_names"])
        self.corr = rt.Corrections(len(self.cam_id), len(self.cam_names)).to(dev)
        self.corr.load_state_dict(ck["corr"])
        self.means = p["means"]
        R = quat_to_mat(p["quats"])
        M = R * torch.exp(p["scales"])[:, None, :]
        self.covars = M @ M.transpose(1, 2)
        self.opac = torch.sigmoid(p["opacities"])
        self.dev = dev

    @torch.no_grad()
    def viewmat(self, cam: str, f: int, vm_raw: torch.Tensor) -> torch.Tensor:
        """Same pose logic as scripts/09_render_room.py."""
        ci = self.cam_names.index(cam)
        if cam == "headcam":
            imgs = np.flatnonzero((self.cam_id == ci) & (self.frame_idx == f))
            if len(imgs) == 0:
                return vm_raw
            slot = torch.tensor([int(imgs[0])], device=self.dev)
        else:
            slot = torch.tensor([ci], device=self.dev)
        return self.corr.viewmat(vm_raw[None], slot)[0]

    @torch.no_grad()
    def in_human_frame(self, cam: str, f: int, vm_raw: torch.Tensor, vm_render=None):
        """Room means/covars moved so that rendering with the human's viewmat (vm_render,
        default the raw one) reproduces what stage 09 renders with its corrected viewmat."""
        vm_render = vm_raw if vm_render is None else vm_render
        A = torch.linalg.inv(vm_render) @ self.viewmat(cam, f, vm_raw)
        R, t = A[:3, :3], A[:3, 3]
        return self.means @ R.T + t, R @ self.covars @ R.T

    @torch.no_grad()
    def depth(self, cam: str, f: int, vm_raw, K, W, H) -> torch.Tensor:
        """Expected depth of the room along each pixel (H,W); inf where the room is empty."""
        from gsplat import rasterization

        means, covars = self.in_human_frame(cam, f, vm_raw)
        d, a, _ = rasterization(
            means, None, None, self.opac, torch.zeros(len(means), 1, device=self.dev),
            vm_raw[None], K[None], W, H, covars=covars, render_mode="ED", packed=False,
        )  # fmt: skip
        d = d[0, ..., 0]
        return torch.where(a[0, ..., 0] > 0.5, d, torch.full_like(d, float("inf")))


def sh_to_rgb(sh: torch.Tensor, means: torch.Tensor, viewmat: torch.Tensor) -> torch.Tensor:
    """Evaluate SH degree <= 1 like gsplat (view direction from the camera centre)."""
    campos = torch.linalg.inv(viewmat)[:3, 3]
    d = F.normalize(means - campos, dim=-1)
    x, y, z = d.unbind(-1)
    C1 = 0.4886025119029199
    c = C0 * sh[:, 0]
    if sh.shape[1] >= 4:
        c = c + C1 * (-y[:, None] * sh[:, 1] + z[:, None] * sh[:, 2] - x[:, None] * sh[:, 3])
    return (c + 0.5).clamp_min(0)


@torch.no_grad()
def render_joint(model, cc, room: Room, cams, ci: int, f: int, dev="cuda"):
    """Human + room Gaussians in ONE rasterisation (correct depth order). Returns the human's
    visible premultiplied colour and visible alpha (occluded by room Gaussians), numpy."""
    from gsplat import rasterization

    cam = CAMS[ci]
    K, vms, (W, H) = cams[cam]
    Kt = torch.tensor(K, dtype=torch.float32, device=dev)
    vm_raw = torch.tensor(vms[f], dtype=torch.float32, device=dev)
    vm = model_viewmat(model, ci, f, vm_raw)
    hm, hc, ho, hsh = model.gaussians(f)
    rgb = sh_to_rgb(hsh, hm, vm)
    feat_h = torch.cat([rgb, torch.ones_like(rgb[:, :1])], 1)
    rm, rc = room.in_human_frame(cam, f, vm_raw, vm)
    feat = torch.cat([feat_h, torch.zeros(len(rm), 4, device=dev)])
    out, _, _ = rasterization(
        torch.cat([hm, rm]), None, None, torch.cat([ho, room.opac]), feat, vm[None], Kt[None],
        W, H, covars=torch.cat([hc, rc]), packed=False, near_plane=0.02,
        rasterize_mode=getattr(model, "raster_mode", "classic"),
    )  # fmt: skip
    a = out[0, ..., 3:].clamp(0, 1)
    pred = torch.minimum(cc(out[0, ..., :3], a, ci, f).clamp(0, 1), a)
    return pred.cpu().numpy(), a.cpu().numpy()


# ----------------------------------------------------------------------------------------------
# v3: denser, densifiable, hard anti-shard scale rules, per-frame camera pose correction
# ----------------------------------------------------------------------------------------------
K3_PER_PART = [6, 1, 1, 4, 2, 2, 1, 12, 12]  # skin (incl. hands/forearms) 6, garments 12
S_MAX, S_MIN = 0.01, 0.001  # metres
PLANE_RATIO, MIN_RATIO = 3.0, 8.0
GAUSS_KEYS = ["offset", "resid", "quat", "log_scale", "opacity", "sh0", "shN"]  # dim 0 = N


def sample_bary(n: int, rng) -> np.ndarray:
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    return np.stack([1 - s, s * (1 - r2), s * r2], 1).astype(np.float32)


def hand_faces() -> tuple[np.ndarray, np.ndarray]:
    """(F,) bool hand faces and forearm faces of the union mesh (body bone labels)."""
    B = np.load("out/body/fit.npz")
    lab = B["bone_labels"][B["vertex_bone"]][B["faces"]]  # (Fb,3)
    is_hand = np.vectorize(lambda s: s.startswith(("wrist", "finger", "metacarpal")))(lab)
    is_fore = np.vectorize(lambda s: s.startswith("lowerarm"))(lab)
    n_extra = len(np.load("out/body/garments.npz")["tunic_faces"]) + len(
        np.load("out/body/garments.npz")["trousers_faces"]
    )
    pad = np.zeros(n_extra, bool)
    return (np.concatenate([is_hand.sum(1) >= 2, pad]), np.concatenate([is_fore.sum(1) >= 2, pad]))


class HumanGS3(torch.nn.Module):
    version = 3

    def __init__(self, verts, faces, part, trim=None, n_basis=30, seed=0, n_gauss=None):
        super().__init__()
        self.capped = True
        self.raster_mode = "antialiased"
        self.pose = None
        rng = np.random.default_rng(seed)
        if n_gauss is None:  # fresh init
            k = np.array(K3_PER_PART)[part]
            tri = np.repeat(np.arange(len(faces)), k)
        else:  # placeholder sizes, filled by load_state_dict
            k = np.ones(len(faces), int)
            tri = np.zeros(n_gauss, int)
        N = len(tri)
        bary = sample_bary(N, rng)
        self.register_buffer("verts", torch.from_numpy(verts))
        self.register_buffer("faces", torch.from_numpy(faces).long())
        self.register_buffer("tri", torch.from_numpy(tri).long())
        self.register_buffer("bary", torch.from_numpy(bary))
        self.register_buffer("basis", dct_basis(verts.shape[0], n_basis, "cpu"))
        v = verts[0][faces]
        edge = np.linalg.norm(v - v[:, [1, 2, 0]], axis=-1).mean(1)
        self.register_buffer("face_cap", torch.from_numpy(np.minimum(S_MAX, 1.5 * edge).astype(np.float32)))
        area = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
        s_t = np.minimum(np.sqrt(area[tri] / k[tri]) * 0.7, self.face_cap.numpy()[tri])
        s_t = np.maximum(s_t, S_MIN)
        log_s = np.log(np.stack([s_t, s_t, np.maximum(S_MIN, s_t / MIN_RATIO)], 1)).astype(np.float32)
        rgb = PART_RGB[part[tri]]
        if trim is not None:
            rgb[trim[tri]] = TRIM_RGB
        P = torch.nn.Parameter
        self.offset = P(torch.zeros(N, 3))
        self.resid = P(torch.zeros(N, n_basis, 3))
        self.global_t = P(torch.zeros(n_basis, 3))
        self.quat = P(torch.tensor([1.0, 0, 0, 0]).repeat(N, 1))
        self.log_scale = P(torch.from_numpy(log_s))
        self.opacity = P(torch.logit(torch.full((N,), 0.7)))
        self.sh0 = P(torch.from_numpy((rgb - 0.5) / C0)[:, None].float())
        self.shN = P(torch.zeros(N, 3, 3))

    tri_frames = HumanGS.tri_frames

    def residual_raw(self, f: int) -> torch.Tensor:
        return torch.einsum("b,nbc->nc", self.basis[f], self.resid)

    def residual(self, f: int) -> torch.Tensor:
        r = self.residual_raw(f)
        return RES_CAP * torch.tanh(r / RES_CAP)

    def static_offset(self) -> torch.Tensor:
        return OFF_CAP * torch.tanh(self.offset / OFF_CAP)

    def scales(self) -> torch.Tensor:
        """Hard anti-shard rules: s1 <= min(1 cm, 1.5 x edge); s2 >= s1/3 (in-plane ratio
        <= 3); s3 >= max(1 mm, s1/8) so edge-on discs cannot become needles."""
        S = torch.exp(self.log_scale)
        srt, idx = torch.sort(S, dim=-1, descending=True)
        s1 = torch.minimum(srt[:, 0], self.face_cap[self.tri])
        s2 = torch.maximum(torch.minimum(srt[:, 1], s1), s1 / PLANE_RATIO)
        lo = torch.clamp(s1 / MIN_RATIO, min=S_MIN)
        s3 = torch.maximum(torch.minimum(srt[:, 2], s2), lo)
        return torch.empty_like(S).scatter(1, idx, torch.stack([s1, s2, s3], 1))

    gaussians = HumanGS.gaussians


class CamPose(torch.nn.Module):
    """6-DoF viewmat delta: one per headcam frame, one per (static) exo camera."""

    def __init__(self, n_frames: int = N_FRAMES):
        super().__init__()
        self.rot = torch.nn.Parameter(torch.zeros(len(CAMS), n_frames, 3))
        self.trans = torch.nn.Parameter(torch.zeros(len(CAMS), n_frames, 3))
        # headcam frames that received gradients; the rest are interpolated from neighbours
        self.register_buffer("trained", torch.zeros(n_frames, dtype=torch.bool))

    def slot(self, ci: int, f: int) -> int:
        return f if CAMS[ci] == "headcam" else 0

    def delta(self, ci: int, f: int):
        if CAMS[ci] == "headcam" and not bool(self.trained[f]):
            tr = torch.nonzero(self.trained)[:, 0]
            if len(tr) == 0:
                return self.rot[ci, f] * 0, self.trans[ci, f] * 0
            lo, hi = tr[tr < f], tr[tr > f]
            if len(lo) and len(hi):
                a, b = int(lo[-1]), int(hi[0])
                w = (f - a) / (b - a)
                return ((1 - w) * self.rot[ci, a] + w * self.rot[ci, b],
                        (1 - w) * self.trans[ci, a] + w * self.trans[ci, b])  # fmt: skip
            n = int(lo[-1]) if len(lo) else int(hi[0])
            return self.rot[ci, n], self.trans[ci, n]
        s = self.slot(ci, f)
        return self.rot[ci, s], self.trans[ci, s]

    def viewmat(self, ci: int, f: int, vm: torch.Tensor) -> torch.Tensor:
        w, t = self.delta(ci, f)
        theta = w.norm().clamp_min(1e-8)
        k = w / theta
        Kx = torch.zeros(3, 3, device=vm.device)
        Kx[0, 1], Kx[0, 2], Kx[1, 2] = -k[2], k[1], -k[0]
        Kx = Kx - Kx.T
        R = torch.eye(3, device=vm.device) + torch.sin(theta) * Kx + (1 - torch.cos(theta)) * Kx @ Kx
        D = torch.eye(4, device=vm.device)
        D = torch.cat([torch.cat([R, t[:, None]], 1), D[3:]], 0)
        return D @ vm


def model_viewmat(model, ci: int, f: int, vm: torch.Tensor) -> torch.Tensor:
    pose = getattr(model, "pose", None)
    return pose.viewmat(ci, f, vm) if pose is not None else vm
