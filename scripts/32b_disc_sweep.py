"""Stage 32: render the learned person with needle Gaussians reshaped (no retraining).

Spikes in the person render are needle-shaped Gaussians (one axis much longer than the
others). Deleting them removed real colour (stage 31 lost ~8 dB) and clamping every
Gaussian also hurt (1-6 dB), so only true needles are reshaped at render time; colours,
opacities and all other Gaussians are unchanged.

    UV_NO_SYNC=1 uv run scripts/32_human_gs_clean_render.py
Writes the 30_human_gs_render.py outputs under out/assets/human_gs_clean/.
"""

import importlib
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
hgs = importlib.import_module("30_human_gs_common")
render_mod = importlib.import_module("30_human_gs_render")

# Only true needles are reshaped (everything else renders exactly as trained): a Gaussian
# whose longest axis is > NEEDLE x its middle axis gets that axis cut to RATIO x middle.
NEEDLE, RATIO = 5.0, 3.0
# Flat discs seen edge-on render as streaks: give every disc a minimum thickness.
DISC = float(os.environ.get("DISC", "6.0"))  # smallest axis >= middle axis / DISC
OUT = Path(os.environ.get("OUT", f"out/assets/human_gs_disc{os.environ.get('DISC', '6')}"))


def clamp_scales(model):
    orig = model.scales

    def scales():
        s = orig()
        srt, idx = torch.sort(s, dim=-1)  # ascending: small, middle, large
        mid, big = srt[:, 1:2], srt[:, 2:3]
        needle = big > NEEDLE * mid
        big = torch.where(needle, mid * RATIO, big)
        small = torch.maximum(srt[:, 0:1], mid / DISC)
        return torch.empty_like(s).scatter_(-1, idx, torch.cat([small, mid, big], -1))

    with torch.no_grad():
        s = orig()
        ratio = s.max(-1).values / s.sort(-1).values[:, 1].clamp_min(1e-6)
        srt = s.sort(-1).values
        thin = int((srt[:, 0] < srt[:, 1] / DISC).sum())
        print(
            f"needles reshaped: {int((ratio > NEEDLE).sum())}, thin discs thickened: {thin}/{len(s)}"
        )
    model.scales = scales
    return model


def main() -> None:
    load = hgs.load_model
    hgs.load_model = lambda path, device="cuda": (lambda m, cc: (clamp_scales(m), cc))(
        *load(path, device)
    )
    hgs.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    masks = OUT / "masks"
    if not masks.exists():
        masks.symlink_to(Path("../human_gs/masks"))
    sys.argv = [sys.argv[0], "--model", "out/assets/human_gs/model.pt", "--tag", f"disc{os.environ.get('DISC', '6')}"]
    render_mod.main()


if __name__ == "__main__":
    main()
