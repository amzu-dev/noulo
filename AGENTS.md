# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, Cursor, Copilot and others) that install,
use or change this repository. Humans: start with [README.md](README.md).

## What noulo is

A local, CPU-first decision engine. You send text and get back one of three strict answers:

| Primitive | Returns |
|---|---|
| **Noul** | P(proposition is true given the input), 0.0–1.0; 0.5 means "the input doesn't say" |
| **Choice** | exactly one of the option IDs you supplied |
| **Score** | position on an ordered rubric, 0.0 (lowest level) – 1.0 (highest) |

It runs an 83 MB INT8 NLI model on ONNX Runtime. It isn't a chatbot and never generates text.

## Install (non-interactive)

```bash
uv sync                                  # Python 3.10+; uv picks 3.12 from .python-version
uv run noulo model download --missing    # fetches models the clone didn't bring (see below)
uv run noulo model list                  # the default model should show "installed"
```

Without uv: `python -m venv .venv && . .venv/bin/activate && pip install -e . && noulo model download --missing`
(then drop the `uv run` prefix below).

**Git LFS:** the bundled models are stored with Git LFS. If the clone ran without LFS, the
model files are small pointer files; `noulo model download --missing` detects that and
downloads the real weights (~110 MB from Hugging Face). You don't need `git lfs` for this.

## Run

```bash
uv run noulo start --headless            # background service on http://127.0.0.1:8787
curl -sf http://127.0.0.1:8787/health    # {"status":"ok","modelLoaded":true} once ready
uv run noulo status
uv run noulo stop
```

- `noulo start` waits until the model is loaded and exits non-zero (pointing at the log) if it
  can't start. It opens a browser only when a person is at a terminal.
- For a foreground process (CI, containers, supervisors): `uv run noulo serve`. Stop it with
  SIGTERM; it drains in-flight requests first.
- Another port: `--port 8790`. Other CLI commands then need `--url http://127.0.0.1:8790`
  (put it before the subcommand).

## Use

```bash
uv run noulo noul   -i "The invoice has remained unpaid for 120 days." -p "The customer has an overdue payment." --json
uv run noulo choice -i "I was charged twice." -q "Which department?" -o A=Billing -o B=Sales --json
uv run noulo score  -i "The production system is unavailable for every customer." \
                    -q "How severe is this incident?" -r insignificant,low,medium,high,critical --json
echo '{"type":"noul","input":"...","proposition":"..."}' | uv run noulo evaluate - --json
```

Over HTTP: `POST /api/v1/noul | /choice | /score | /evaluate` (see [docs/api.md](docs/api.md)
and [openapi.json](openapi.json)). In-process Python, with no server:
`from noulo import evaluate; evaluate({"type": "noul", "input": "...", "proposition": "..."})`.

**Exit codes:** `0` ok · `1` API/service error (message on stderr) · `2` usage/config error ·
`3` server not reachable (start it). Errors from the API are always
`{"error": {"code": "...", "message": "..."}}`.

### Don't run commands that wait for a person

These prompt for input. Without a terminal they exit with code 2 and name the alternative:

| Interactive | Non-interactive equivalent |
|---|---|
| `noulo` (interactive session) | the one-shot commands above, or pipe commands into it: `printf '/noul P\ninput\n' \| noulo` |
| `noulo model` (picker) | `noulo model list`, then `noulo model use <id>` |
| `/config` editor inside the session | `noulo config show`, `noulo config set <key> <value>` |

`noulo learning clear` refuses to run without `--yes` (exit 2), so pass `--yes` when you mean it.

`noulo model use <id>` downloads the model if needed, saves it in `.env` and restarts the
running service.

## Develop

```bash
uv run pytest -m "not slow"              # fast suite, ~35 s
uv run pytest                            # everything, incl. a real-server end-to-end test
uv run ruff check src tests && uv run ruff format src tests
uv run noulo openapi -o openapi.json     # after any API change (a test checks for drift)
```

- **Test-first is the house style:** write a failing test, check it fails for the right reason,
  write the minimal code, refactor. Every behaviour change needs a test.
- Tests marked `model` need the bundled model and skip if it's missing. Tests for optional
  models (LLMs, larger models) skip unless those are downloaded.
- **Numbers in the docs are measured, never estimated.** Use `noulo benchmark` (it tunes on the
  calibration split and reports on the test split) and copy the results.
- Code map: `src/noulo/api/` (FastAPI, validation, schemas) · `src/noulo/inference/` (engine,
  NLI and LLM backends, calibration, learning memory, vector stores, devices) ·
  `src/noulo/cli.py` + `src/noulo/repl/` (CLI and interactive session) · `src/noulo/registry.py`
  (models) · `src/noulo/benchmark/` · `tests/`. Architecture: [docs/architecture.md](docs/architecture.md).
- Configuration is `NOULO_*` environment variables or `.env`
  ([docs/configuration.md](docs/configuration.md)).

## Keep in mind

- **Security defaults:** it binds `127.0.0.1` only. Don't expose it on a network
  (`NOULO_ALLOW_NETWORK=true`) without also setting `NOULO_API_KEY`.
- **The learning memory stores the inputs it sees** in `data/` (on by default). Turn it off
  with `noulo learning off` when handling data that shouldn't be kept, and clear it with
  `noulo learning clear --yes`.
- **Only the two bundled models are committed** (`models/nli-deberta-v3-xsmall-int8`,
  `models/minilm-l6-v2-int8`). Downloaded models, `data/` and `.env` are git-ignored; keep
  them out of commits.
- **Accuracy is modest on hard tasks** (see Known limitations in the README). Check results
  that matter; `?diagnostics=true` / `--diagnostics` shows the probabilities.
