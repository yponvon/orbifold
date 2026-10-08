"""Bone list for the tracker joints (parent, child), used for overlays and the stick-figure body."""

_FINGERS = ["thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"], *(
    [f"{f}_mcp", f"{f}_pip", f"{f}_dip", f"{f}_tip"] for f in ("index", "middle", "ring", "pinky")
)


def bones(names: list[str]) -> list[tuple[int, int]]:
    pairs = [
        ("pelvis", "spine1"), ("spine1", "spine2"), ("spine2", "spine3"),
        ("spine3", "neck"), ("neck", "head"),
    ]  # fmt: skip
    for s in ("left", "right"):
        pairs += [
            ("spine3", f"{s}_collar"), (f"{s}_collar", f"{s}_shoulder"),
            (f"{s}_shoulder", f"{s}_elbow"), (f"{s}_elbow", f"{s}_wrist"),
            ("pelvis", f"{s}_hip"), (f"{s}_hip", f"{s}_knee"),
            (f"{s}_knee", f"{s}_ankle"), (f"{s}_ankle", f"{s}_foot"),
        ]  # fmt: skip
        for chain in _FINGERS:
            prev = f"{s}_wrist"
            for j in chain:
                pairs.append((prev, f"{s}_{j}"))
                prev = f"{s}_{j}"
    idx = {n: i for i, n in enumerate(names)}
    return [(idx[a], idx[b]) for a, b in pairs if a in idx and b in idx]
