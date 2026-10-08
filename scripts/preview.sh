#!/usr/bin/env bash
# Quick preview on the GPU box: rebuild the scene, render the 3 cameras IN PARALLEL
# (every 3rd frame, 32 samples + OIDN denoise), composite.
# Finals: scripts/trackB_video.sh (256 spp, no denoiser, all frames).
set -euo pipefail
export UV_NO_SYNC=1
STEP=${STEP:-3}; SAMPLES=${SAMPLES:-32}; DENOISE=${DENOISE:-accurate}
uv run scripts/04_build_scene.py 2>&1 | grep -E "^saved|garments|key light|cloth|Error|Traceback" || true
rm -rf out/render
for cam in headcam exocam1 exocam2; do
  uv run scripts/05_render.py --scale 100 --samples "$SAMPLES" --denoise "$DENOISE" --step "$STEP" --cams "$cam" \
    > "out/render_$cam.log" 2>&1 &
done
wait
grep -h "^rendered\|Traceback" out/render_*.log
uv run scripts/06_compare.py
echo PREVIEW_DONE
