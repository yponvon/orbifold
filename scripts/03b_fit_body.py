"""Stage 03b: fit the Anny body model (Apache-2.0, MakeHuman-based) to the tracked joints.

Per frame: a rotation for every bone + root translation. Shared over the clip: body shape
(gender, age, muscle, weight, height, proportions). Loss = weighted joint distance between
tracker joints and the matching Anny bone heads, plus temporal smoothness and a small pose
prior (joint positions alone don't pin down bone twist).

    uv run --extra body scripts/03b_fit_body.py [--iters 2000]
Writes out/body/fit.npz: vertices (F,V,3) world, faces (T,3), uv (U,2), face_uv (T,3),
vertex_bone (V,) dominant bone index, bone_labels, joint_err_cm (F,J).
"""

import argparse
from pathlib import Path

import anny
import numpy as np
import roma
import torch
import yaml
from PIL import Image

CLIP, OUT = Path("out/clip"), Path("out/body")
REGIONS = ["skin", "shirt", "trousers", "hair", "shoes", "strap", "eye"]

# tracker joint -> (Anny bone whose head sits at that joint, weight)
MAP = {
    "pelvis": ("root", 1.0),
    "spine1": ("spine04", 0.3), "spine2": ("spine02", 0.3), "spine3": ("spine01", 0.3),
    "neck": ("neck01", 0.7), "head": ("head", 0.5),
}  # fmt: skip
for s, a in (("left", "L"), ("right", "R")):
    MAP |= {
        # Lower-body tracker is unreliable (H1 diagnosis): weak prior, images decide.
        f"{s}_hip": (f"upperleg01.{a}", 0.3), f"{s}_knee": (f"lowerleg01.{a}", 0.1),
        f"{s}_ankle": (f"foot.{a}", 0.1), f"{s}_foot": (f"toe3-1.{a}", 0.05),
        f"{s}_collar": (f"clavicle.{a}", 0.3), f"{s}_shoulder": (f"upperarm01.{a}", 1.0),
        f"{s}_elbow": (f"lowerarm01.{a}", 1.0), f"{s}_wrist": (f"wrist.{a}", 2.0),
        f"{s}_thumb_cmc": (f"finger1-1.{a}", 1.0), f"{s}_thumb_mcp": (f"finger1-2.{a}", 1.0),
        f"{s}_thumb_ip": (f"finger1-3.{a}", 1.0),
    }  # fmt: skip
    for k, finger in enumerate(("index", "middle", "ring", "pinky"), start=2):
        MAP |= {
            f"{s}_{finger}_mcp": (f"finger{k}-1.{a}", 1.0),
            f"{s}_{finger}_pip": (f"finger{k}-2.{a}", 1.0),
            f"{s}_{finger}_dip": (f"finger{k}-3.{a}", 1.0),
        }


def kabsch(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Rotation R (3,3) minimising |R @ (src - mean) - (dst - mean)|."""
    a, b = src - src.mean(0), dst - dst.mean(0)
    U, _, Vt = torch.linalg.svd(a.T @ b)
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d.item()], device=src.device))
    return Vt.T @ D @ U.T


def face_regions(model, rest_verts: np.ndarray, rest_heads: dict) -> np.ndarray:
    """Garment region per face, from Anny's UV body-part map, dominant bone and position."""
    seg_dir = Path(anny.__file__).parent / "data" / "segmentation"
    seg = np.array(Image.open(seg_dir / "body_parts_segmentation.png"))[..., :3].astype(int)
    colors = yaml.safe_load(open(seg_dir / "body_parts_segmentation.yaml"))["colors"]
    faces = model.faces.cpu().numpy()
    uv = model.texture_coordinates.cpu().numpy()[
        model.face_texture_coordinate_indices.cpu().numpy()
    ]
    c = uv.mean(1)
    px = seg[
        ((1 - c[:, 1]) * (seg.shape[0] - 1)).astype(int), (c[:, 0] * (seg.shape[1] - 1)).astype(int)
    ]
    names = list(colors)
    pal = np.array([colors[n] for n in names])
    part = np.array(names)[np.abs(px[:, None] - pal[None]).sum(-1).argmin(1)]

    labels = list(model.bone_labels)
    vb = model.vertex_bone_indices[torch.arange(model.vertex_bone_indices.shape[0]),
                                   model.vertex_bone_weights.argmax(-1)].cpu().numpy()  # fmt: skip
    bone = np.array(labels)[[np.bincount(vb[f]).argmax() for f in faces]]
    centre = rest_verts[faces].mean(1)

    region = np.full(len(faces), REGIONS.index("shirt"))
    legs = np.char.find(bone, "leg") >= 0
    legs |= np.isin(bone, ["pelvis.L", "pelvis.R", "root"]) & (centre[:, 2] < -0.03)
    region[legs] = REGIONS.index("trousers")
    bare = (np.char.find(bone, "lowerarm") >= 0) | (np.char.find(bone, "upperarm02") >= 0)
    bare |= (np.char.find(bone, "neck") >= 0) | np.isin(part, ["hand.L", "hand.R", "head"])
    region[bare] = REGIONS.index("skin")
    region[np.isin(part, ["foot.L", "foot.R"]) | (np.char.find(bone, "toe") >= 0)] = REGIONS.index(
        "shoes"
    )
    for side in ("L", "R"):  # black wrist bands, ~4 cm wide, just above the wrist
        w = rest_heads[f"wrist.{side}"]
        elbow = rest_heads[f"lowerarm01.{side}"]
        axis = (elbow - w) / np.linalg.norm(elbow - w)
        t = (centre - w) @ axis
        radial = np.linalg.norm(centre - w - t[:, None] * axis, axis=1)
        region[(t > 0.0) & (t < 0.045) & (radial < 0.06)] = REGIONS.index("strap")
    head = rest_heads["head"]
    on_head = part == "head"
    eye_z = rest_heads["eye.L"][2]
    hair = on_head & ((centre[:, 2] > eye_z + 0.035) | (centre[:, 1] > head[1] + 0.02))
    region[hair] = REGIONS.index("hair")
    region[np.char.startswith(part, "eye")] = REGIONS.index("eye")
    is_head = (part == "head") | np.char.startswith(part, "eye") | (part == "mouth_cavity")
    return region, is_head


# Fabric push-out along the normal (m), smoothly blended across region borders.
# Kept small: large offsets made shoulder "humps" and a ragged hem (round-4 review).
OFFSET = {"shirt": 0.006, "trousers": 0.012, "hair": 0.008, "shoes": 0.004, "strap": 0.004}


def dress(verts: np.ndarray, faces: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Offset garment vertices along per-frame normals with a Laplacian-smoothed amount."""
    push = np.zeros(verts.shape[1])
    for i, r in enumerate(REGIONS):
        if r in OFFSET:
            v = np.unique(faces[region == i])
            push[v] = np.maximum(push[v], OFFSET[r])
    nbr = [[] for _ in range(verts.shape[1])]
    for a, b, c in faces:
        nbr[a] += [b, c]
        nbr[b] += [a, c]
        nbr[c] += [a, b]
    rows = np.repeat(np.arange(len(nbr)), [len(n) for n in nbr])
    cols = np.concatenate([np.asarray(n, int) for n in nbr])
    deg = np.bincount(rows, minlength=len(nbr))
    for _ in range(15):
        push = 0.5 * push + 0.5 * np.bincount(rows, push[cols], len(nbr)) / np.maximum(deg, 1)
    out = verts.copy()
    for f in range(len(verts)):
        tri = verts[f][faces]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        vn = np.zeros_like(verts[f])
        for k in range(3):
            np.add.at(vn, faces[:, k], fn)
        vn /= np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-9)
        out[f] += vn * push[:, None]
    return out


# COCO keypoint (YOLO pose, stage 03a) -> Anny bone head
COCO_TO_BONE = {5: "upperarm01.L", 6: "upperarm01.R", 7: "lowerarm01.L", 8: "lowerarm01.R",
                9: "wrist.L", 10: "wrist.R", 11: "upperleg01.L", 12: "upperleg01.R",
                13: "lowerleg01.L", 14: "lowerleg01.R", 15: "foot.L", 16: "foot.R"}  # fmt: skip
# Columns are the OpenCV camera axes expressed in ros_body axes (geometry.CONVENTIONS).
# Index permutations over the 12 COCO_TO_BONE keypoints: identity, and left<->right swap.
ID_PERM = list(range(12))
SWAP_PERM = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
ROS_TO_CV = torch.tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    # Girth is invisible to joint positions: fix "weight" (0-1) from the silhouettes (03g).
    ap.add_argument("--weight", type=float, help="fix the Anny weight phenotype")
    ap.add_argument("--out", default=str(OUT / "fit.npz"))
    ap.add_argument("--w2d", type=float, default=0.2, help="weight of the 2D keypoint term")
    ap.add_argument("--w-leg", type=float, default=0.02, help="straight-leg prior weight")
    args = ap.parse_args()
    dev = "cuda"
    J = np.load(CLIP / "joints.npz")
    names = list(J["names"])
    target = torch.from_numpy(J["xyz"]).float().to(dev)  # (F,62,3)
    F_ = target.shape[0]

    model = anny.Anny(local_changes="default").to(device=dev, dtype=torch.float32)
    labels = list(model.bone_labels)
    B = model.bone_count
    used = [n for n in names if n in MAP]
    t_idx = torch.tensor([names.index(n) for n in used], device=dev)
    b_idx = torch.tensor([labels.index(MAP[n][0]) for n in used], device=dev)
    w = torch.tensor([MAP[n][1] for n in used], device=dev)
    # Tracker quality flags: down-weight joints marked bad on that frame.
    good = torch.from_numpy(~J["bad"]).float().to(dev)[:, t_idx] * 0.9 + 0.1
    tgt = target[:, t_idx]

    # Fingertips have no bone head: predict them as distal head + bone axis * length,
    # with the length taken from the middle phalanx (round-4 review: fingers splayed).
    tips = []  # (tracker idx, distal bone, middle bone, length ratio)
    for s_, a in (("left", "L"), ("right", "R")):
        for k, finger in enumerate(("thumb", "index", "middle", "ring", "pinky"), start=1):
            name = f"{s_}_{finger}_tip"
            if name in names:
                tips.append((names.index(name), labels.index(f"finger{k}-3.{a}"),
                             labels.index(f"finger{k}-2.{a}"), 0.9 if k == 1 else 0.8))  # fmt: skip
    tip_t = torch.tensor([t[0] for t in tips], device=dev)
    tip_d = torch.tensor([t[1] for t in tips], device=dev)
    tip_m = torch.tensor([t[2] for t in tips], device=dev)
    tip_ratio = torch.tensor([t[3] for t in tips], device=dev)
    # 2D keypoints of the real person in the tripod views (stage 03a), if available.
    views = []
    kp_path = OUT / "kp2d.npz"
    if kp_path.exists() and args.w2d > 0:
        kp, C = np.load(kp_path), np.load(CLIP / "cameras.npz")
        coco_idx = torch.tensor(list(COCO_TO_BONE), device=dev)
        coco_bone = torch.tensor([labels.index(b) for b in COCO_TO_BONE.values()], device=dev)
        for cam in ("exocam1", "exocam2"):
            T = torch.from_numpy(C[f"{cam}_T"]).float().to(dev)  # (F,4,4)
            K = torch.from_numpy(C[f"{cam}_K"]).float().to(dev)
            xy = torch.from_numpy(kp[f"{cam}_xy"]).float().to(dev)[:, coco_idx]
            cf = torch.from_numpy(kp[f"{cam}_conf"]).float().to(dev)[:, coco_idx]
            views.append((T, K, xy, cf * (cf > 0.4)))
        print(f"2D keypoint term on {len(views)} views (weight {args.w2d})")
    rot_cv = ROS_TO_CV.to(dev)

    # Spine/neck twist is unconstrained by joint positions alone: stronger prior there.
    stiff = torch.tensor([1.0 if any(k in n for k in ("spine", "neck")) else 0.0
                          for n in labels], device=dev)  # fmt: skip
    # She stands straight-legged the whole clip (the footage), but the bed hides her lower
    # legs in exocam1 and the tracker bends a knee: discourage knee and ankle bend.
    straight_leg = torch.tensor([1.0 if any(k in n for k in ("lowerleg", "foot.")) else 0.0
                                 for n in labels], device=dev)  # fmt: skip

    def run(rotvec, trans, pheno_logits):
        R = roma.rotvec_to_rotmat(rotvec)  # (F,B,3,3)
        pose = torch.eye(4, device=dev).repeat(F_, B, 1, 1)
        pose[..., :3, :3] = R
        pose[:, 0, :3, 3] = trans
        pheno = {k: torch.sigmoid(pheno_logits[i]).expand(F_) for i, k in
                 enumerate(model.phenotype_labels)}  # fmt: skip
        if args.weight is not None:
            pheno["weight"] = torch.full((F_,), args.weight, device=dev)
        return model(pose_parameters=pose, phenotype_kwargs=pheno)

    # Init: rest pose, root rotation/translation from a Kabsch fit of the torso per frame.
    rotvec = torch.zeros(F_, B, 3, device=dev)
    pheno_logits = torch.zeros(len(model.phenotype_labels), device=dev)
    with torch.no_grad():
        rest = run(rotvec, torch.zeros(F_, 3, device=dev), pheno_logits)["bone_poses"][0, :, :3, 3]
        core = [n for n in ("pelvis", "left_hip", "right_hip", "left_shoulder",
                            "right_shoulder", "neck") if n in used]  # fmt: skip
        ci = torch.tensor([used.index(n) for n in core], device=dev)
        trans = torch.zeros(F_, 3, device=dev)
        for f in range(F_):
            Rf = kabsch(rest[b_idx[ci]], tgt[f, ci])
            rotvec[f, 0] = roma.rotmat_to_rotvec(Rf)
            trans[f] = tgt[f, used.index("pelvis")]
        rest_full = run(rotvec, trans, pheno_logits)["bone_poses"][0]
        # Which local axis points along each distal bone (towards the fingertip)?
        seg = rest_full[tip_d, :3, 3] - rest_full[tip_m, :3, 3]
        cos = torch.einsum(
            "nc,ncj->nj", seg / seg.norm(dim=-1, keepdim=True), rest_full[tip_d, :3, :3]
        )
        axis = cos.abs().mean(0).argmax().item()
        sign = torch.sign(cos[:, axis])
        print(f"fingertip axis: local {'xyz'[axis]} (signs {sign.tolist()})")
    rotvec.requires_grad_(True)
    trans.requires_grad_(True)
    pheno_logits.requires_grad_(True)
    opt = torch.optim.Adam([{"params": [rotvec, trans], "lr": 0.02},
                            {"params": [pheno_logits], "lr": 0.01}])  # fmt: skip

    for it in range(args.iters):
        out = run(rotvec, trans, pheno_logits)
        heads = out["bone_poses"][:, :, :3, 3][:, b_idx]
        d2 = ((heads - tgt) ** 2).sum(-1)
        data = (d2 * w * good).sum() / (w * good).sum()
        bp = out["bone_poses"]
        length = (bp[:, tip_d, :3, 3] - bp[:, tip_m, :3, 3]).norm(dim=-1, keepdim=True) * tip_ratio[
            :, None
        ]
        tip_pred = bp[:, tip_d, :3, 3] + bp[:, tip_d, :3, axis] * sign[:, None] * length
        tip_d2 = ((tip_pred - target[:, tip_t]) ** 2).sum(-1)
        data = data + tip_d2.mean()
        loss2d = torch.zeros((), device=dev)
        for T, K, xy, cf in views:
            p = bp[:, coco_bone, :3, 3]  # (F,12,3) world
            pc = torch.einsum("fij,fnj->fni", T[:, :3, :3].transpose(1, 2), p - T[:, None, :3, 3])
            pc = pc @ rot_cv  # ros_body -> OpenCV axes (same as geometry.project: p @ M)
            uv = (pc @ K.T)[..., :2] / pc[..., 2:3].clamp_min(0.05)
            # The detector can swap left/right (e.g. seen from behind): per frame, use the
            # assignment that fits better.
            errs = []
            for perm in (ID_PERM, SWAP_PERM):
                e = torch.nn.functional.huber_loss(
                    uv / 100, xy[:, perm] / 100, reduction="none", delta=0.3
                )
                errs.append((e.sum(-1) * cf[:, perm]).sum(-1))
            per_frame = torch.minimum(*errs)
            loss2d = loss2d + per_frame.sum() / cf.sum().clamp_min(1)
        data = data + args.w2d * loss2d
        smooth = ((rotvec[1:] - rotvec[:-1]) ** 2).mean() + ((trans[1:] - trans[:-1]) ** 2).mean()
        prior = (rotvec[:, 1:] ** 2).mean()
        prior_stiff = ((rotvec**2).sum(-1) * stiff).mean()
        prior_leg = ((rotvec**2).sum(-1) * straight_leg).mean()
        loss = data + 0.05 * smooth + 1e-4 * prior + 2e-3 * prior_stiff + args.w_leg * prior_leg
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 250 == 0 or it == args.iters - 1:
            rms = d2.sqrt().mean().item() * 100
            tip_cm = tip_d2.sqrt().mean().item() * 100
            print(
                f"iter {it:5d}  loss {loss.item():.6f}  joints {rms:.2f} cm  tips {tip_cm:.2f} cm"
            )

    with torch.no_grad():
        out = run(rotvec, trans, pheno_logits)
        heads = out["bone_poses"][:, :, :3, 3][:, b_idx]
        err = ((heads - tgt) ** 2).sum(-1).sqrt() * 100
        pheno = {k: round(torch.sigmoid(pheno_logits[i]).item(), 3)
                 for i, k in enumerate(model.phenotype_labels)}  # fmt: skip
        if args.weight is not None:
            pheno["weight"] = args.weight
    print(f"phenotype {pheno}")
    # 2D check: mean pixel error per view for legs / arms (confident detections only).
    with torch.no_grad():
        bp = out["bone_poses"]
        for (T, K, xy, cf), cam in zip(views, ("exocam1", "exocam2"), strict=False):
            p = bp[:, coco_bone, :3, 3]
            pc = torch.einsum("fij,fnj->fni", T[:, :3, :3].transpose(1, 2), p - T[:, None, :3, 3])
            pc = pc @ rot_cv
            uv = (pc @ K.T)[..., :2] / pc[..., 2:3].clamp_min(0.05)
            d = (uv - xy).norm(dim=-1)
            ok = cf > 0
            legs, arms = slice(6, 12), slice(0, 6)
            print(f"2D error {cam}: legs {d[:, legs][ok[:, legs]].mean():.1f}px "
                  f"arms {d[:, arms][ok[:, arms]].mean():.1f}px")  # fmt: skip
    worst = err.mean(0).argsort(descending=True)[:5]
    print("worst joints (cm):", {used[i]: round(err[:, i].mean().item(), 1) for i in worst})

    rest_heads_all = (
        out["rest_bone_poses"][0, :, :3, 3].cpu().numpy() if "rest_bone_poses" in out else None
    )
    rest_heads = {n: rest_heads_all[labels.index(n)] for n in labels}
    region, is_head = face_regions(model, out["rest_vertices"][0].cpu().numpy(), rest_heads)
    print("faces per region:", {r: int((region == i).sum()) for i, r in enumerate(REGIONS)})

    faces_np = model.faces.cpu().numpy()
    dressed = dress(out["vertices"].cpu().numpy(), faces_np, region)
    head_pose = out["bone_poses"][:, labels.index("head")].cpu().numpy()
    head_rest_pose = out["rest_bone_poses"][0, labels.index("head")].cpu().numpy()

    OUT.mkdir(parents=True, exist_ok=True)
    vb = model.vertex_bone_indices[torch.arange(model.vertex_bone_indices.shape[0]),
                                   model.vertex_bone_weights.argmax(-1)]  # fmt: skip
    np.savez(
        args.out,
        vertices=dressed,
        head_pose=head_pose,
        head_rest_pose=head_rest_pose,
        rest_vertices=out["rest_vertices"][0].cpu().numpy(),
        rest_bone_heads=rest_heads_all,
        faces=faces_np,
        uv=model.texture_coordinates.cpu().numpy(),
        face_uv=model.face_texture_coordinate_indices.cpu().numpy(),
        vertex_bone=vb.cpu().numpy(),
        bone_labels=np.array(labels),
        joint_names=np.array(used),
        joint_err_cm=err.cpu().numpy(),
        phenotype=np.array([pheno[k] for k in model.phenotype_labels]),
        face_region=region,
        face_is_head=is_head,
        regions=np.array(REGIONS),
    )
    print(f"wrote {args.out}: {out['vertices'].shape[1]} vertices x {F_} frames")


if __name__ == "__main__":
    main()
