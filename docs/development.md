# Development

## Setup

```bash
git lfs install && git clone <repo> noulo && cd noulo
uv sync                       # runtime + dev dependencies (pytest, hypothesis, ruff, qdrant, chroma)
uv run noulo model download --missing
```

## Tests

```bash
make test           # uv run pytest -m "not slow"      (~12 s)
make test-all       # includes the real-server end-to-end test
uv run pytest tests/test_engine.py -q
uv run pytest -k learning
```

| Marker | Meaning |
|---|---|
| `model` | Needs the bundled model in `models/`; skipped automatically if absent |
| `slow` | End-to-end: starts a real background server with the CLI and drives it through subprocesses |

Remote vector-store contract tests run when `NOULO_TEST_QDRANT_URL` / `NOULO_TEST_CHROMA_URL`
point at live servers.

What is covered:

| Area | Test files |
|---|---|
| Hard invariants (`0 ≤ noul, score ≤ 1`; choice ∈ supplied IDs) | `test_primitives.py`, `test_engine.py`, `test_acceptance.py` (Hypothesis property tests) |
| Model loading and failure | `test_model.py`, `test_engine.py` (load/warm-up failure, NaN output, LFS pointers) |
| Health/readiness, API contracts, validation, malformed input | `test_api.py`, `test_validation.py`, `test_api_learning.py` |
| Concurrency, queueing, graceful shutdown, model switching | `test_engine.py`, `test_api.py` |
| OpenAI-compatible backend (mock transport, no network) | `test_openai_backend.py` |
| Learning memory and every vector store (one contract suite) | `test_memory.py`, `test_vector_stores.py`, `test_engine_learning.py` |
| CLI, shell, background service, model picker | `test_cli.py`, `test_cli_models.py`, `test_shell.py`, `test_service.py` |
| OpenAPI validity and drift | `test_api.py`, `test_openapi_file.py` |
| Real model end to end (spec examples) | `test_acceptance.py`, `test_e2e.py` |

## Workflow

The code base was built test-first (red → green → refactor):

1. Write one failing test for the behaviour; run it and check it fails *for the right reason*.
2. Write the minimal code to pass it.
3. Refactor with the suite green.

Test doubles live in `tests/fakes.py` (`FakeBackend`, `FakeLoader`). Everything that touches
the OS or the network (process spawning, HTTP, the browser, prompts) is injected, so it can be
tested without mocks of internals.

## Lint and format

```bash
make lint      # ruff check
make format    # ruff format
```

## Regenerating artefacts

| Artefact | Command |
|---|---|
| `openapi.json` | `make openapi` (a test fails if it drifts from the live schema) |
| Tuned profiles for catalog models | `noulo benchmark --all --download --tune`, then copy `models/<id>/{profile,calibration}.json` to `src/noulo/profiles/<id>/` |
| Model comparison | `make compare` → `benchmark/results/comparison.md` |

## Extending noulo

- **New decision backend:** implement `DecisionBackend` (`inference/types.py`: `noul`,
  `choice`, `score`, `warmup`, `close`, `info`) and teach `ModelRegistry.load_backend` to build
  it from a `models.json` entry. The engine's invariant guards apply automatically.
- **New vector store:** implement `VectorStore` (`inference/stores/__init__.py`), add it to
  the parametrised fixture in `tests/test_vector_stores.py`, or load it without code changes
  via `NOULO_MEMORY_STORE=module:Class`.
- **New catalog model:** add a `CatalogEntry` in `registry.py`, run
  `noulo benchmark --model <id> --download --tune`, and ship its profile.
- **Dataset:** JSON Lines in `benchmark/data/` with a `split` of `calibration` or `test`.
  Never tune on `test`.
