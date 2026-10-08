# Stages in run order. GPU stages (fit, room-*, env, scene, render) need the 5090 box.
# CLIP_S = clip length in seconds (2 s = 60 frames, the rendered clip).
CLIP_S ?= 2
SAMPLES ?= 64

.PHONY: setup data inspect test lint all \
	calib extract overlay fit colors albedo garments \
	room-data room-train room-render env scene render compare color-check

setup:
	uv sync
	uv run pre-commit install

data:
	bash scripts/00_download.sh

inspect:
	uv run scripts/01_inspect.py

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .

# --- Cameras and clip ---------------------------------------------------------------
calib:
	uv run scripts/01b_calibrate.py

extract:
	uv run scripts/02_extract.py --duration $(CLIP_S)

overlay:
	uv run scripts/03_overlay.py

# --- Person -------------------------------------------------------------------------
fit:
	uv run --extra body scripts/03b_fit_body.py

colors:
	uv run scripts/03c_sample_colors.py

# Optional closed loop after a render: 03c --sim, then rescale albedo; rerun scene/render.
albedo:
	uv run scripts/03c_sample_colors.py --sim
	uv run scripts/03d_calibrate_albedo.py

garments:
	uv run scripts/03e_garments.py

# --- Room ---------------------------------------------------------------------------
room-data:
	uv run --extra recon scripts/07_room_data.py --clip-seconds $(CLIP_S)

room-train:
	uv run --extra recon scripts/08_train_room.py

room-render:
	uv run --extra recon scripts/09_render_room.py

env:
	uv run --extra recon scripts/09b_env_map.py

# --- Scene, render, compare ---------------------------------------------------------
scene:
	uv run --extra render scripts/04_build_scene.py

render:
	uv run --extra render scripts/05_render.py --scale 100 --samples $(SAMPLES)

compare:
	uv run scripts/06_compare.py

color-check:
	uv run --extra recon scripts/10_color_check.py

all: calib extract fit colors garments room-data room-train room-render env scene render compare
