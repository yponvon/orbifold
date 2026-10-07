"""Stage 01: inspect the MCAP so later stages can be written against real data.

Prints the first message of every topic (image bytes elided), tracker joint names,
frame ids, whether exo extrinsics are static, and the /session labels.
Writes one frame per camera and a contact sheet to out/inspect/ for picking the
ironing window.

    uv run scripts/01_inspect.py [--config configs/clip.yaml]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from google.protobuf import text_format

from orbifold.mcap_io import decode_image, iter_messages, summary

OUT = Path("out/inspect")
SHEET_EVERY_S = 3.0
MAX_LINES = 60


def show(topic: str, proto) -> None:
    msg = type(proto)()
    msg.CopyFrom(proto)
    if hasattr(msg, "data") and isinstance(msg.data, bytes):
        n = len(msg.data)
        msg.ClearField("data")
        print(f"  [data: {n} bytes, starts {proto.data[:4]!r}]")
    lines = text_format.MessageToString(msg).splitlines()
    for line in lines[:MAX_LINES]:
        print("  " + line)
    if len(lines) > MAX_LINES:
        print(f"  ... ({len(lines) - MAX_LINES} more lines)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/clip.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    path = cfg["mcap"]
    OUT.mkdir(parents=True, exist_ok=True)

    s = summary(path)
    t0 = s["start_ns"]
    print(f"== {path}: {(s['end_ns'] - t0) / 1e9:.2f} s")
    for topic, n in s["counts"].items():
        print(f"  {topic:45s} {n}")

    # First message of every topic.
    seen: set[str] = set()
    for topic, t, proto in iter_messages(path):
        if topic in seen:
            if len(seen) == len(s["counts"]):
                break
            continue
        seen.add(topic)
        print(f"\n== {topic}  (t={(t - t0) / 1e9:.3f} s, {type(proto).__name__})")
        show(topic, proto)
        if topic.endswith("/image"):
            cam = topic.strip("/").split("/")[0]
            img = decode_image(proto)
            cv2.imwrite(str(OUT / f"{cam}.jpg"), img)
            print(f"  -> decoded {img.shape} {img.dtype}, saved {OUT / cam}.jpg")
        if type(proto).__name__ == "Trackers":
            names = [(tr.name, tr.pose.source_frame_id) for tr in proto.trackers]
            print(f"  -> {len(names)} trackers (name, frame): {names}")

    # Do extrinsics change over time? (exo should be static, head should move)
    print("\n== extrinsics motion over the recording")
    ext_topics = [c["extrinsics"] for c in cfg["cameras"].values()]
    xyz: dict[str, list] = {t: [] for t in ext_topics}
    frames: dict[str, set] = {t: set() for t in ext_topics}
    for topic, _, proto in iter_messages(path, topics=ext_topics):
        for tr in proto.sensor_extrinsics:
            xyz[topic].append(list(tr.translation_meters_xyz))
            frames[topic].add((tr.source_frame_id, tr.destination_frame_id))
    for topic, pts in xyz.items():
        p = np.asarray(pts)
        span = np.ptp(p, axis=0) if len(p) else []
        print(f"  {topic:25s} n={len(p)} xyz range (m)={np.round(span, 4)} frames={frames[topic]}")

    # Contact sheet from exocam1 to find the ironing window.
    sheet_topic = cfg["cameras"]["exocam1"]["image"]
    tiles, next_t = [], 0.0
    for _, t, proto in iter_messages(path, topics=[sheet_topic]):
        ts = (t - t0) / 1e9
        if ts >= next_t:
            img = cv2.resize(decode_image(proto), (320, 180))
            cv2.putText(img, f"{ts:.0f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            tiles.append(img)
            next_t += SHEET_EVERY_S
    while len(tiles) % 5:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + 5]) for i in range(0, len(tiles), 5)]
    cv2.imwrite(str(OUT / "contact_sheet_exocam1.jpg"), np.vstack(rows))
    print(f"\n== wrote {OUT}/contact_sheet_exocam1.jpg ({len(tiles)} tiles)")


if __name__ == "__main__":
    main()
