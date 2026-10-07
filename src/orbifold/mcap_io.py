"""Read protobuf messages from the MCAP recording and decode images."""

from collections.abc import Iterable, Iterator
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

JPEG_MAGIC = b"\xff\xd8"
PNG_MAGIC = b"\x89PNG"


def summary(path: str | Path) -> dict:
    """Recording time span (ns) and per-topic message counts."""
    with open(path, "rb") as f:
        s = make_reader(f).get_summary()
    stats = s.statistics
    counts = {s.channels[cid].topic: n for cid, n in stats.channel_message_counts.items()}
    return {
        "start_ns": stats.message_start_time,
        "end_ns": stats.message_end_time,
        "counts": dict(sorted(counts.items())),
    }


def iter_messages(
    path: str | Path,
    topics: Iterable[str] | None = None,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> Iterator[tuple[str, int, object]]:
    """Yield (topic, log_time_ns, decoded protobuf) in log-time order."""
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, message, proto in reader.iter_decoded_messages(
            topics=list(topics) if topics else None,
            start_time=start_ns,
            end_time=end_ns,
            log_time_order=True,
        ):
            yield channel.topic, message.log_time, proto


def decode_image(msg) -> np.ndarray:
    """recording.protos.Image → BGR uint8 array (handles JPEG/PNG or raw pixels)."""
    data = msg.data
    if data[:2] == JPEG_MAGIC or data[:4] == PNG_MAGIC:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("cv2 could not decode compressed image")
        return img
    channels = len(data) // (msg.rows * msg.cols)
    return np.frombuffer(data, np.uint8).reshape(msg.rows, msg.cols, channels)
