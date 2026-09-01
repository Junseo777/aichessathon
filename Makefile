SHELL := /bin/bash

.PHONY: setup weights play arena zip gate

setup:
	uv sync

weights:
	uv run --group train python -m train.export_onnx --out weights

play:
	uv run python -m harness.play --white . --black baselines/greedy

arena:
	uv run python -m harness.arena --opponent baselines/greedy --games 20

zip:
	uv run python -m harness.package --include chessml

gate:
	uv run ruff check .
	uv run mypy
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000
