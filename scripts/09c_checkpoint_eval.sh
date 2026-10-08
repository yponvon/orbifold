#!/usr/bin/env bash
# Stage 09c (checkpoint eval): render headcam (60 clip frames, no refine) from a 08b milestone,
# measure room sharpness / PSNR, write the frame-30 ORIGINAL | OURS crop preview.
#   bash scripts/09c_checkpoint_eval.sh <step>
set -euo pipefail
step=$(printf "%06d" "$1")
ck=out/room_sharp/splats_${step}.pt
[ -f "$ck" ] || ck=out/room_sharp/splats.pt
ev=out/room_sharp/eval_${step}
UV_NO_SYNC=1 uv run scripts/09c_render_room_sharp.py --ckpt "$ck" --out "$ev" --cams headcam
UV_NO_SYNC=1 uv run scripts/09c_measure.py --ckpt "$ck" --after "$ev/headcam" --tag "$step" \
  | tee out/room_sharp/measure_${step}.txt
