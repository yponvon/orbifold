#!/usr/bin/env bash
# Save the latest comparison from the 5090 as a numbered round in out/progress/.
#   bash scripts/save_round.sh <round> "<what changed>"
set -euo pipefail
ROUND=$1; NOTE=$2
REMOTE="root@38.246.237.140"; PORT=33507
mkdir -p out/progress
rsync -az -e "ssh -o BatchMode=yes -p $PORT" "$REMOTE:orbifold/out/compare.mp4" "out/progress/round$ROUND.mp4"
ffmpeg -loglevel error -y -ss 1.0 -i "out/progress/round$ROUND.mp4" -frames:v 1 "out/progress/round$ROUND.jpg"
# Start / middle / end stills (real | sim, all cameras) for the reviewer agent.
mkdir -p "out/progress/round${ROUND}_frames"
for t in 0.0 1.0 1.95; do
  ffmpeg -loglevel error -y -ss "$t" -i "out/progress/round$ROUND.mp4" -frames:v 1 \
    "out/progress/round${ROUND}_frames/t${t}s.jpg"
done
SCORES=$(ssh -o BatchMode=yes -p $PORT "$REMOTE" 'grep "mean abs error" orbifold/out/round.log' 2>/dev/null \
  | awk '{printf "%s %s · ", $1, $5}')
[ -f out/progress/LOG.md ] || printf "# Progress log\n\nError = mean abs pixel difference /255 (lower is better).\n\n| Round | Video | What changed | Error | Your feedback |\n|---|---|---|---|---|\n" > out/progress/LOG.md
echo "| $ROUND | [round$ROUND.mp4](round$ROUND.mp4) | $NOTE | ${SCORES% · } |  |" >> out/progress/LOG.md
echo "saved round $ROUND -> out/progress/"
