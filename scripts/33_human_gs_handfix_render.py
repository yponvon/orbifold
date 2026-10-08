"""Stage 33: render the person with hand/forearm/trousers Gaussians cleaned (no retraining).
Only Gaussians on hand, forearm and trousers triangles are touched:
offset (static+residual) > 1.5 cm -> opacity 0 for that frame; sigmoid(opacity) < 0.15 -> 0;
largest scale capped at 6 mm."""
import importlib, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
hgs = importlib.import_module("30_human_gs_common")
render_mod = importlib.import_module("30_human_gs_render")
OFF_MAX, OP_MIN, S_CAP = 0.015, 0.15, 0.006
OUT = Path("out/assets/human_gs_handfix")

def wrap(model):
    _, _, part = hgs.load_mesh()
    hand, fore = hgs.hand_faces()
    sel_f = hand | fore | np.isin(part, [2, 8])
    sel = torch.from_numpy(sel_f).to(model.tri.device)[model.tri]
    orig_g, orig_s = model.gaussians, model.scales
    def scales():
        s = orig_s()
        return torch.where(sel[:, None], s.clamp(max=S_CAP), s)
    model.scales = scales
    def gaussians(f):
        means, cov, op, sh = orig_g(f)
        off = (model.static_offset() + model.residual(f)).norm(dim=-1)
        kill = sel & ((off > OFF_MAX) | (op < OP_MIN))
        if f in (0, 30):
            print(f"frame {f}: selected {int(sel.sum())}/{len(sel)}, killed {int(kill.sum())} "
                  f"(offset {int((sel & (off > OFF_MAX)).sum())}, low-op {int((sel & (op < OP_MIN)).sum())})", flush=True)
        return means, cov, torch.where(kill, torch.zeros_like(op), op), sh
    model.gaussians = gaussians
    with torch.no_grad():
        s = orig_s()
        print("scale-capped:", int((sel & (s.max(-1).values > S_CAP)).sum()), flush=True)
    return model

def main():
    load = hgs.load_model
    hgs.load_model = lambda path, device="cuda": (lambda m, cc: (wrap(m), cc))(*load(path, device))
    hgs.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    masks = OUT / "masks"
    if not masks.exists():
        masks.symlink_to(Path("../human_gs/masks"))
    sys.argv = [sys.argv[0], "--model", "out/assets/human_gs/model.pt", "--tag", "handfix"]
    render_mod.main()

if __name__ == "__main__":
    main()
