"""Stage 03d: closed-loop colour calibration of the body materials.

After a render, compares each garment region's rendered colour (03c --sim) with the real
footage's (03c) and scales the albedo used for that render by real/rendered, in linear
light. Run 04 -> 05 again afterwards; repeat if needed.

    uv run scripts/03d_calibrate_albedo.py
Writes out/body/albedo.json (read by 04_build_scene.py).
"""

import json
from pathlib import Path

BODY = Path("out/body")


def lin(c: list) -> list:
    return [v**2.2 for v in c]


def main() -> None:
    real = json.load(open(BODY / "colors.json"))["colors_srgb"]
    sim = json.load(open(BODY / "colors_sim.json"))["colors_srgb"]
    used = json.load(open(BODY / "albedo_used.json"))
    new = dict(used)
    for r in real.keys() & sim.keys():
        ratio = [a / max(b, 1e-4) for a, b in zip(lin(real[r]), lin(sim[r]), strict=True)]
        new[r] = [min(max(u * k, 0.003), 0.95) for u, k in zip(used[r], ratio, strict=True)]
        print(f"{r:9s} real {real[r]}  sim {sim[r]}  albedo {[round(v, 3) for v in used[r]]}"
              f" -> {[round(v, 3) for v in new[r]]}")  # fmt: skip
    json.dump(new, open(BODY / "albedo.json", "w"), indent=1)
    print(f"wrote {BODY / 'albedo.json'}")


if __name__ == "__main__":
    main()
