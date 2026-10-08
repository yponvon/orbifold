#!/usr/bin/env bash
# Final assembly on the GPU box: best room + learned person + iron -> full-res video.
set -euo pipefail
export UV_NO_SYNC=1
ROOM=out/room_render
if [ -f out/room_render_sharp/DONE ]; then ROOM=out/room_render_sharp; fi
echo "room layer: $ROOM"
ls out/assets/human_gs/RENDER_DONE out/assets/iron/RENDER_DONE 2>/dev/null || true
uv run scripts/14_composite_layers.py --room "$ROOM" --iron "${IRON:-out/assets/iron}" 2>&1 | grep -v "findDecoder" || true
uv run scripts/13_progress_video.py --sim out/final --out out/final_video.mp4
uv run python - <<'PY'
import cv2
for cam in ("headcam", "exocam1", "exocam2"):
    r = cv2.cvtColor(cv2.imread(f"out/clip/frames/{cam}/0030.jpg"), cv2.COLOR_BGR2GRAY)
    o = cv2.cvtColor(cv2.imread(f"out/final/{cam}/0030.png"), cv2.COLOR_BGR2GRAY)
    print(f"{cam}: sharpness ours/original = {cv2.Laplacian(o, cv2.CV_64F).var() / cv2.Laplacian(r, cv2.CV_64F).var():.2f}")
PY
echo FINAL_DONE
