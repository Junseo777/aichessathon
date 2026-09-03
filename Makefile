SHELL := /bin/bash

.PHONY: setup weights baseline-hero play arena zip gate

setup:
	uv sync

weights:
	uv run --group train python -m train.export_onnx --out weights

baseline-hero:
	cp agent.py baselines/reference-hero/agent.py
	ln -sfn ../../chessml baselines/reference-hero/chessml
	uv run --group train python -m train.import_reference_hero --out baselines/reference-hero/weights

play:
	uv run python -m harness.play --white . --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --opponent baselines/greedy --games 20

zip:
	uv run python -m harness.package --include chessml

gate:
	uv run ruff check .
	uv run mypy
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000
