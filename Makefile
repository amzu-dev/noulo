.PHONY: install models start stop restart status serve test test-all benchmark compare openapi lint format

install:            ## install dependencies and make sure the bundled models are present
	uv sync
	-git lfs pull 2>/dev/null
	uv run noulo model download --missing

start:              ## background service + frontend (opens the browser)
	uv run noulo start

stop:
	uv run noulo stop

restart:
	uv run noulo restart

status:
	uv run noulo status

serve:              ## foreground, headless API (for systemd/containers)
	uv run noulo serve

test:               ## fast suite (skips the slow end-to-end test)
	uv run pytest -m "not slow"

test-all:           ## everything, including the real-server end-to-end test
	uv run pytest

benchmark:          ## accuracy, calibration, latency and RAM of the active model
	uv run noulo benchmark

compare:            ## download, tune and compare every catalog model
	uv run noulo benchmark --all --download --tune --compare benchmark/results/comparison.md

openapi:            ## regenerate openapi.json
	uv run noulo openapi --output openapi.json

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests
