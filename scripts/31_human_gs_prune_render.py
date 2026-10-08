"""Stage 31: prune bad person Gaussians, then render (fast cleanup, no retraining).

Removes, before rasterising:
  - globally: Gaussians with opacity < OPAC_MIN or largest scale > SCALE_MAX (halo, shards),
  - per frame: Gaussians farther than OFF_MAX from their anchor point on the body surface
    (floaters), and Gaussians inside a box around the iron at its fitted pose (the iron is
    its own layer). The box stops below the handle top, so the gripping fingers stay.
Pruning is geometry removal, so every pixel still comes from the 3D scene.

    UV_NO_SYNC=1 uv run scripts/31_human_gs_prune_render.py
Writes the same outputs as 30_human_gs_render.py under out/assets/human_gs_pruned/.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
hgs = importlib.import_module("30_human_gs_common")
render_mod = importlib.import_module("30_human_gs_render")

OPAC_MIN, SCALE_MAX, OFF_MAX = 0.08, 0.012, 0.03
IRON_BOX = ((-0.13, 0.13), (-0.06, 0.06), (-0.01, 0.09))  # iron local frame, metres
OUT = Path("out/assets/human_gs_pruned")


def pruned(model):
    orig = model.gaussians
    poses_p = Path("out/assets/iron/poses.npz")
    iron_T = torch.from_numpy(np.load(poses_p)["T"]).float().cuda() if poses_p.exists() else None
    with torch.no_grad():
        keep_static = (torch.sigmoid(model.opacity) >= OPAC_MIN) & (
            model.scales().max(-1).values <= SCALE_MAX
        )
    print(f"static prune keeps {int(keep_static.sum())}/{len(keep_static)} Gaussians")

    def gaussians(f: int):
        means, covars, opac, sh = orig(f)
        with torch.no_grad():
            local = model.static_offset() + model.residual(f)
            keep = keep_static & (local.norm(dim=-1) <= OFF_MAX)
            if iron_T is not None:
                Ti = torch.linalg.inv(iron_T[f])
                p = means @ Ti[:3, :3].T + Ti[:3, 3]
                inside = torch.ones_like(keep)
                for k, (lo, hi) in enumerate(IRON_BOX):
                    inside &= (p[:, k] > lo) & (p[:, k] < hi)
                keep &= ~inside
        return means, covars, opac * keep.float(), sh

    model.gaussians = gaussians
    return model


def main() -> None:
    load = hgs.load_model
    hgs.load_model = lambda path, device="cuda": (lambda m, cc: (pruned(m), cc))(
        *load(path, device)
    )
    hgs.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], "--model", "out/assets/human_gs/model.pt", "--tag", "pruned"]
    render_mod.main()


if __name__ == "__main__":
    main()
