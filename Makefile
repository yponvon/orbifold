.PHONY: setup data inspect test lint all

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

# Stages 02-06 are added here as they land.
all: inspect
