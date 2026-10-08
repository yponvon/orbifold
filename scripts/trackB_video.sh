#!/usr/bin/env bash
# Track B (all-CG person + CG iron) final video, on the GPU box:
#   04 build scene (BODY_FIT body) -> 05 render 3 cams in parallel, all frames, full res
#   -> composite over the shared splat room (out/room_render) -> ORIGINAL | OURS video.
# Never touches the Track A assets (out/assets/human_gs, out/assets/iron/render).
#   SAMPLES=256 DENOISE=none FRAMES="0 59" bash scripts/trackB_video.sh
# Writes out/progress/trackB_final.mp4 and out/progress/trackB_final_f30.jpg.
set -euo pipefail
export UV_NO_SYNC=1
export BODY_FIT=${BODY_FIT:-out/body/fit_legs.npz}
# The garments must come from the same fit (fit_X.npz <-> garments_X.npz).
G=${BODY_FIT/\/fit/\/garments}
if [ -z "${GARMENTS:-}" ] && [ -f "$G" ]; then export GARMENTS=$G; fi
echo "[trackB] body $BODY_FIT  garments ${GARMENTS:-out/body/garments.npz}"
SAMPLES=${SAMPLES:-256}        # no denoiser (it smears fabric/skin), so many samples
DENOISE=${DENOISE:-none}       # or "accurate" (OIDN, albedo+normal, ACCURATE prefilter)
FRAMES=${FRAMES:-"0 59"}
STEP=${STEP:-1}                # 2 = every 2nd frame (video plays at 15 fps, real time)
SKIP_BUILD=${SKIP_BUILD:-0}    # 1 = reuse out/scene_trackB.blend
R=out/trackB
mkdir -p "$R" out/progress

if [ "$SKIP_BUILD" != 1 ]; then
echo "[trackB] 04 build scene (BODY_FIT=$BODY_FIT)"
uv run scripts/04_build_scene.py > "$R/04.log" 2>&1 || { tail -30 "$R/04.log"; exit 1; }
grep -E "^saved|cg_iron|floor z|garment|cloth|hair|Error" "$R/04.log" || true
# Private copy (same directory, so relative texture paths still resolve): other agents
# may rebuild out/scene.blend while we render.
cp out/scene.blend out/scene_trackB.blend
fi

echo "[trackB] 05 render ($SAMPLES spp, denoise $DENOISE, frames $FRAMES)"
rm -rf "$R/render"
for cam in headcam exocam1 exocam2; do
  # shellcheck disable=SC2086
  uv run scripts/05_render.py --blend out/scene_trackB.blend --out "$R/render" --cams "$cam" \
    --scale 100 --samples "$SAMPLES" --denoise "$DENOISE" --frames $FRAMES --step "$STEP" \
    > "$R/render_$cam.log" 2>&1 &
done
wait
# The GPU is shared: a camera that ran out of memory in parallel is retried on its own.
for cam in headcam exocam1 exocam2; do
  if ! grep -q "^rendered" "$R/render_$cam.log"; then
    echo "[trackB] retrying $cam alone"
    # shellcheck disable=SC2086
    uv run scripts/05_render.py --blend out/scene_trackB.blend --out "$R/render" --cams "$cam" \
      --scale 100 --samples "$SAMPLES" --denoise "$DENOISE" --frames $FRAMES --step "$STEP" \
      > "$R/render_$cam.log" 2>&1
  fi
done
grep -h "^sharpness\|^rendered\|Traceback\|Error" "$R"/render_*.log

echo "[trackB] composite over out/room_render"
rm -rf "$R/sim"
uv run scripts/06_compare.py --fg "$R/render" --sim "$R/sim" --no-video
uv run scripts/15_sharpness_check.py --sim "$R/sim" --fg "$R/render" || true

echo "[trackB] video"
uv run scripts/13_progress_video.py --sim "$R/sim" --out out/progress/trackB_final.mp4 --crf 12 \
  --fps "$(python3 -c "print(30 / $STEP)")"
ffmpeg -loglevel error -y -i out/progress/trackB_final.mp4 -vf "select=eq(n\,$((30 / STEP)))" -frames:v 1 \
  -q:v 2 out/progress/trackB_final_f30.jpg
ls -la out/progress/trackB_final.mp4 out/progress/trackB_final_f30.jpg
echo TRACKB_DONE
