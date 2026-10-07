#!/usr/bin/env bash
# Download the hackathon MCAP into data/ (1.1 GB). Requires network access.
set -euo pipefail
FOLDER="https://drive.google.com/drive/folders/14GwdeVu9BERb9Y8g4HBIxlNTjtD2R9aB"
mkdir -p data
if ls data/*.mcap >/dev/null 2>&1; then
  echo "MCAP already present: $(ls data/*.mcap)"
  exit 0
fi
uvx gdown --folder "$FOLDER" -O data/
ls -lh data/
