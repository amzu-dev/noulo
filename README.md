# noulo

**A small, local, CPU-only decision engine.** Send natural-language text, get back one
of three strict answers:

| Primitive | Question it answers | Output |
|---|---|---|
| **Noul** | Is this proposition true, given the input? | a probability `0.0`–`1.0` (0 = no, 0.5 = undetermined, 1 = yes) |
| **Choice** | Which of *your* options fits the input? | exactly one of the IDs you supplied |
| **Score** | Where does the input sit on your ordered rubric? | a value `0.0`–`1.0` (lowest → highest level) |

noulo is a semantic decision service, **not a chatbot**. It runs a quantised NLI
model (83 MB, INT8) with ONNX Runtime on the CPU and needs no GPU or cloud. It
exposes the same engine through a versioned REST API, a CLI, a Python module and a
local web frontend.

```bash
$ noulo noul -i "The invoice has remained unpaid for 120 days." -p "The customer has an overdue payment."
0.968
$ noulo choice -i "My subscription payment was taken twice." -q "Which department?" \
      -o A=Billing -o B="Technical Support" -o C=Sales
A
```

---

## Contents

- [Highlights](#highlights)
- [Quick start](#quick-start)
- [Using noulo](#using-noulo): [CLI](#cli) · [REST API](#rest-api) · [Python](#python) · [Frontend](#frontend)
- [Models](#models)
- [Learning from inputs](#learning-from-inputs)
- [Configuration](#configuration) · [Security defaults](#security-defaults)
- [Measured performance](#measured-performance)
- [Known limitations](#known-limitations)
- [Development](#development) · [Project layout](#project-layout)
- [Documentation](#documentation)

## Highlights

- **Fully local by default:** CPU-only ONNX Runtime and Hugging Face `tokenizers`, with no PyTorch. The bundled model is 83 MB. A running server uses about 500 MiB of RAM, about 300 MiB with the Fast & light model, or about 410 MiB in low-memory mode.
- **Strict outputs:** the model never writes free text. Noul and Score come from NLI probabilities, and Choice always returns one of your IDs. The hard invariants are enforced and tested.
- **Calibrated Noul:** a separate calibration layer (temperature, Platt or isotonic) is fitted on held-out data for every model.
- **Switchable models:** three small models, three larger 0.5–1 GB models (each shown with download size, quantisation and measured RAM), any OpenAI-compatible endpoint (OpenAI, Ollama, LM Studio, vLLM, llama.cpp), or your own ONNX model. You can switch at runtime without downtime, or via `noulo model`, which restarts the service.
- **Learning from inputs (RAG-style):** every case can be remembered, and feedback nudges future answers for similar inputs. It can be switched on or off live, and the vector store is pluggable: SQLite (default), Qdrant or Chroma (local or remote), or your own.
- **Four ways in, one engine:** REST (`/api/v1`, OpenAPI), a Claude Code-style interactive CLI, a Python module, and the web frontend.
- **GPU where it actually helps:** CUDA, DirectML or ROCm (and CoreML on Apple Silicon) through
  ONNX Runtime, with automatic CPU fallback (`NOULO_DEVICE=auto|gpu|cpu|...`). `auto` is set
  from measurements: CoreML was 12–100× *slower* than the CPU on an M1 Pro, so on Macs noulo
  stays on the CPU unless you ask otherwise. See [docs/gpu.md](docs/gpu.md).
- **Safe defaults:** binds to `127.0.0.1` only, has no wildcard CORS, supports an optional API key, and never exposes filesystem paths or secrets.
- **Measured, not assumed:** the `noulo benchmark` command reports accuracy, calibration, latency, RAM and cold start, and compares models.

## Quick start

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/) (or pip) and
[Git LFS](https://git-lfs.com) (the bundled model weights are stored with LFS).

```bash
git lfs install
git clone <your-repo-url> noulo && cd noulo
make install          # uv sync + fetch the bundled models if LFS didn't
uv run noulo start    # background service + frontend, opens http://127.0.0.1:8787/ui/
```

To skip `uv run` before each command, activate the virtual environment
(`source .venv/bin/activate`) or install the CLI globally with `uv tool install .`.

```bash
noulo status          # is it running? which model?
noulo                 # interactive session: type inputs, /model, /config, /learning, /help ...
noulo stop
```

With pip instead of uv: `python -m venv .venv && . .venv/bin/activate && pip install -e .`,
then `noulo model download --missing` and `noulo start`.

## Using noulo

### Interactive session

Run `noulo` on its own to open a Claude Code-style session. Type a sentence and noulo
evaluates it. Slash commands (with a completion menu as you type `/`) check or change
everything else.

```text
❯ /noul The customer has an overdue payment.
noul ❯ The invoice has been unpaid for 120 days.
  ● Yes       0.95  ━━━━━━━━━━━━━━━━━━━━━━━─
  57 ms · nli-deberta-v3-xsmall-int8
noul ❯ /good                      # teach: that was right
noul ❯ /model                     # arrow-key picker: size · quantisation · RAM · accuracy
noul ❯ /config                    # settings editor; /config set port 9000 also works
noul ❯ /learning off
```

There's a status bar at the bottom (server, model, learning, current mode), history with ↑/↓,
`/verbose` for probability bars, and `/json` for raw responses. See the
[interactive session docs](docs/cli.md#interactive-session).

### CLI

| Command | What it does |
|---|---|
| `noulo start [--headless] [--foreground] [--no-browser]` | Start the service in the background with the frontend (`--headless`: API only) |
| `noulo stop` · `restart` · `status` · `logs` | Manage the background service |
| `noulo serve` | Run the API in the foreground, headless (for systemd, containers, CI) |
| `noulo model` | **Pick a model** from 3 basic and 3 larger ones (size, quantisation, RAM and accuracy shown), or see how to plug in your own. A new model is downloaded, saved as the default and the service is restarted with the frontend |
| `noulo model list \| use <id> \| download \| add-endpoint \| add-onnx \| remove` | Model management |
| `noulo noul \| choice \| score \| evaluate` | Call the API (`--json` for full output incl. `recordId`, `--diagnostics` for probabilities) |
| `noulo learning on \| off \| status \| records \| clear --yes` | Learning memory controls (applied live **and** saved to `.env`) |
| `noulo feedback <recordId> <expected>` | Teach the correct outcome of a past evaluation |
| `noulo config show \| get \| set \| unset \| path` | View and edit settings in `.env` (validated, secrets redacted) |
| `noulo benchmark [--model id ...] [--tune] [--all --download --compare file.md]` | Measure models |
| `noulo openapi -o openapi.json` | Regenerate the OpenAPI document |
| `noulo` | Interactive session: type inputs to evaluate them, slash commands to check or change anything |

```bash
noulo score -i "The production system is unavailable for every customer." \
            -q "How severe is this incident?" -r insignificant,low,medium,high,critical
echo '{"type":"noul","input":"...","proposition":"..."}' | noulo evaluate -
noulo config set port 9000 && noulo restart
```

See [docs/cli.md](docs/cli.md) for every command and flag.

### REST API

Base URL `http://127.0.0.1:8787`. Interactive docs live at `/docs`, and the spec at `/openapi.json`
([committed copy](openapi.json)).

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | `{"status":"ok","modelLoaded":true}`: 200 only once the model is loaded |
| GET | `/api/v1/info` | name, version, model, quantization, capabilities, backend, learning |
| POST | `/api/v1/noul` | `{"input","proposition"}` → `{"type":"noul","value":0.97}` |
| POST | `/api/v1/choice` | `{"input","question","choices":[{"id","text"}]}` → `{"type":"choice","value":"A"}` |
| POST | `/api/v1/score` | `{"input","question","rubric":[...]}` → `{"type":"score","value":0.92}` |
| POST | `/api/v1/evaluate` | Any of the above with `"type"`; routed to the same implementation |
| GET · PUT | `/api/v1/models` · `/api/v1/models/active` | List models · switch the active model at runtime |
| POST · DELETE | `/api/v1/models` · `/api/v1/models/{id}` | Register · remove an OpenAI-compatible endpoint |
| GET · PUT | `/api/v1/learning` | Learning status · turn learning on/off |
| GET · DELETE | `/api/v1/learning/records` | Stored cases · delete all |
| POST | `/api/v1/feedback` | `{"recordId","expected"}` or `{"request":{...},"expected"}` |

Add `?diagnostics=true` to include probabilities and the learning influence. When learning
is on, responses carry an `X-Record-Id` header, and the body keeps the strict contract.
Errors always look like `{"error":{"code":"INVALID_REQUEST","message":"Noul requires a proposition."}}`.

A consuming app only needs HTTP:

```javascript
const response = await fetch("http://127.0.0.1:8787/api/v1/noul", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    input: "The invoice has been unpaid for four months.",
    proposition: "The customer has an overdue invoice.",
  }),
});
console.log((await response.json()).value); // 0.97
```

See [docs/api.md](docs/api.md) for the full reference and status codes, and
[docs/examples.md](docs/examples.md) for worked Noul, Choice and Score examples in every
interface.

### Python

The same engine runs in-process, with no server needed:

```python
from noulo import evaluate, Noulo

evaluate({"type": "noul", "input": "The invoice has remained unpaid for 120 days.",
          "proposition": "The customer has an overdue payment."})
# {'type': 'noul', 'value': 0.968}

with Noulo(model="nli-minilm2-l6-int8", learning_enabled=False) as engine:
    engine.choice("Charged twice", "Which team?", {"A": "Billing", "B": "Sales"})  # 'A'
    engine.score("Everything is down", "How severe?", ["low", "medium", "high"])   # 0.0-1.0
```

`aevaluate()` and `Noulo.aevaluate()` are the async variants.

### Frontend

`noulo start` serves a small local console at `http://127.0.0.1:8787/ui/`. It uses the same
REST API as every other client. It covers all three primitives, model switching, the learning
toggle, feedback, the memory browser and diagnostics, in light and dark themes.

## Models

`noulo model` (and `/model` in the session, or the frontend) lists every option with its
**download size, quantisation, measured RAM and accuracy**. The figures below are from the
[measured comparison](docs/model-comparison.md) on the held-out test split:

| Menu name | Model | Download | Quant | RAM (peak) | Noul | Choice | Score MAE |
|---|---|---|---|---|---|---|---|
| **Balanced ★** (bundled) | `nli-deberta-v3-xsmall-int8` | 87 MB | INT8 | 443 MB | **91%** | 67.5% | 0.205 |
| Fast & light | `nli-minilm2-l6-int8` | 83 MB | INT8 | 232 MB | 87% | 52.5% | 0.253 |
| Best at Choice | `zeroshot-deberta-v3-xsmall-int8` | 87 MB | INT8 | 333 MB | 87% | 70.0% | 0.244 |
| **Most accurate** (larger) | `zeroshot-deberta-v3-base-fp32` | 739 MB | FP32 | 1234 MB | 91% | **85.0%** | **0.115** |
| Larger NLI | `nli-deberta-v3-base-fp32` | 739 MB | FP32 | 1419 MB | 89% | 70.0% | 0.175 |
| BART (slow) | `nli-bart-large-fp16` | 816 MB | FP16 | 2010 MB | 91% | 70.0% | 0.205 |
| Qwen3 0.6B (LLM) | `qwen3-0.6b-q4f16` | 570 MB | INT4 | 2588 MB | 63.0% | 67.5% | 0.381 |
| Qwen2.5 0.5B (LLM) | `qwen2.5-0.5b-q4` | 786 MB | INT4 | 2113 MB | 67.0% | 55.0% | 0.334 |
| Gemma 3 1B (LLM) | `gemma-3-1b-q4` | 859 MB | INT4 | 1295 MB | 68.0% | 57.5% | 0.314 |
| LFM2 1.2B (LLM) | `lfm2-1.2b-q4` | 850 MB | INT4 | 856 MB | 85.0% | 87.5% | 0.221 |

The last group is **small open-weight LLMs** (Qwen, Gemma, Liquid LFM2) at 4-bit. noulo
reads their next-token probabilities for the answer labels in a single forward pass, so they
never generate text. Measured honestly: **LFM2 1.2B** has the best Choice accuracy of any model
(87.5%), but Qwen and Gemma score below the 83 MB default. All of them are ~100× slower per
call (0.4–0.9 s) and need 0.9–2.6 GB of RAM. Details are in
[docs/models.md](docs/models.md#small-llms-4-bit-05-09-gb).

Choosing a model that isn't installed downloads it once from Hugging Face, sets
`NOULO_MODEL`, and restarts the service with the frontend. Tuned templates, calibration and
measurements for every catalog model ship in the package, so a downloaded model is tuned
immediately. Two INT8 DeBERTa-*large* exports are kept as `experimental` and not offered,
because quantisation measurably damages them (see [docs/models.md](docs/models.md)).

**Plug in your own model** (the picker's last option prints these steps):

```bash
# Any OpenAI-compatible endpoint (OpenAI, Ollama, LM Studio, vLLM, llama.cpp)
noulo model add-endpoint --id local-llm --base-url http://127.0.0.1:11434/v1 --model qwen2.5:1.5b
noulo model use local-llm

# Your own ONNX NLI / zero-shot classifier (model.onnx + tokenizer.json + config.json)
noulo model add-onnx --id my-nli --path ./my-nli-int8 --quantization INT8
noulo benchmark --model my-nli --tune      # fit templates + Noul calibration on the calibration split
noulo model use my-nli
```

OpenAI-compatible backends never ask the model to *generate* a number or verdict. Options
are shown as labels, and noulo reads the log-probabilities of the single next token. See
[docs/models.md](docs/models.md), which covers model replacement, the catalog, calibration
and tuning.

## Learning from inputs

When learning is on (the default), each evaluation is stored with its task, input, embedding
and model output. Later requests for the **same task** (same proposition, the same question
and options, or the same rubric) look up **similar past inputs** and blend their outcomes into
the model's answer:

- **Weights:** feedback-verified cases count fully (`1.0`); past unverified outputs count a
  little (`0.25`).
- **Cap:** the total influence is capped (`0.9`) and grows with the amount of evidence.
- **Choice safety:** Choice only ever votes among the IDs you supplied in the current request.

```bash
noulo noul -i "My parcel never arrived." -p "The customer is satisfied." --json   # note recordId
noulo feedback <recordId> false        # similar future inputs now lean towards "no"
noulo learning off                     # stop recording and applying memory (live + saved)
```

Storage is pluggable: `NOULO_MEMORY_STORE=sqlite` (default), `qdrant`, `chroma`, or
`your.module:YourStore`. Use a file or directory for local stores, or an `http(s)://` URL
for remote servers. See [docs/learning.md](docs/learning.md).

## Configuration

Settings come from `NOULO_*` environment variables or `.env`. Every setting is documented in
[`.env.example`](.env.example) and [docs/configuration.md](docs/configuration.md).

```bash
noulo config show                 # effective settings (secrets redacted)
noulo config set model nli-minilm2-l6-int8
noulo config set cors_origins https://app.example.com
```

### Security defaults

- **Loopback only:** the server binds `127.0.0.1:8787`. Any other host is refused unless
  `NOULO_ALLOW_NETWORK=true`.
- **No authentication on loopback;** set `NOULO_API_KEY` to require
  `Authorization: Bearer <key>` on `/api/v1/*`. This is recommended whenever network access is on.
  Keys are compared in constant time.
- **CORS:** allowed for localhost origins only, plus any origins you list. Wildcard `*` is
  rejected. It can be disabled with `NOULO_CORS_ENABLED=false`.
- **Limits:** request body size, input and field lengths, and the number of options and
  levels are all capped (HTTP 413/422).
- **No path or secret exposure:** errors and `/info` never include filesystem paths. API keys
  come from env vars and are redacted in `config show`. `.env` and `models.json` are written
  with mode `600`.
- **Data stays local** unless you configure a remote model endpoint or a remote vector store.
  The learning memory stores inputs on disk: `noulo learning clear --yes` deletes them, and
  `NOULO_LEARNING_ENABLED=false` stops recording.

## Measured performance

These are the default model's numbers on the held-out **test split** of the bundled dataset
(`benchmark/data/`: 240 Noul, 80 Choice and 70 Score items). They were measured on an
**Apple M1 Pro** (10 cores, macOS 26, arm64) with `noulo benchmark`:

| Metric | `nli-deberta-v3-xsmall-int8` |
|---|---|
| Model size / quantisation | 83.2 MiB / INT8 |
| Server RAM (steady state, model + embedder + API) | ~500 MiB RSS (~410 MiB with `NOULO_LOW_MEMORY=true`) |
| Benchmark process peak RAM | ~330–430 MiB |
| Cold start (load + readiness check) | 0.5–0.7 s |
| Latency P50 / P95 (mixed primitives) | 7.9 ms / 31.9 ms |
| Throughput (sequential, 1 thread of work) | ~75 req/s |
| Noul accuracy (yes/no items) | 91.0% |
| Noul calibration (ECE / Brier) | 0.105 / 0.089 |
| Choice accuracy | 67.5% |
| Score MAE (normalised 0–1) | 0.205 |

The comparison of nine models and three quantisation levels (FP16, INT8, INT4) is in
[docs/model-comparison.md](docs/model-comparison.md). Re-measure on your hardware with
`make benchmark` / `make compare`.

**Supported platforms:** macOS, Linux and Windows on x86-64 and ARM64, anywhere ONNX Runtime
and `tokenizers` publish wheels (Python 3.10+). Development and all measurements above were
on macOS arm64. Linux and Windows are expected to work but were not measured here. On
Windows, `noulo stop` terminates the process instead of sending a graceful signal.

## Known limitations

- **Small models, modest accuracy on hard tasks.** Choice is about 67–70% and Score MAE about
  0.2 on a deliberately hard test set. Ordinal severity is the weakest area: a trivial incident
  can still score "high". Mitigations are feedback learning, `noulo benchmark --tune` on your
  own labelled data, or a larger or OpenAI-compatible model.
- **Small evaluation set.** 120 Noul, 40 Choice and 35 Score test items give wide confidence
  intervals (±5–15 points). Treat the comparison as indicative.
- **RAM sits right at the 500 MB target with the default model** (about 500 MiB RSS). Most of
  it is ONNX constant folding and the DeBERTa tokenizer. Use the Fast & light model (~300 MiB),
  `NOULO_LOW_MEMORY=true` (~410 MiB, ~3× slower), or learning off (−27 MiB) on tight machines.
- **English only;** inputs longer than about 512 tokens (roughly 2,000 characters) are truncated
  for the NLI models.
- **Calibration** is fitted on 100 labelled Noul items per model; ECE is about 0.1, not zero.
- **OpenAI-compatible endpoints** need `logprobs` support for graded probabilities. Otherwise
  noulo falls back to strict label parsing (0/0.5/1) and logs a warning. Remote endpoints
  send your text off the machine.
- **Learning memory:** unverified past outputs carry a small weight and can reinforce a
  model's own mistakes on near-duplicate inputs. Use feedback, or set
  `NOULO_MEMORY_OBSERVED_WEIGHT=0` for feedback-only learning.
- **Stack deviation from the original requirements:** the runtime is Python/FastAPI (at the
  project owner's request) rather than Node.js. The optional Tauri desktop wrapper was not
  built; the frontend is served by the API instead.

## Development

The project was built test-first. There are 840+ tests in total: unit, property-based
(Hypothesis), API contract, CLI, vector-store contract, real-model acceptance, and an
end-to-end test that drives a real background server through the CLI.

```bash
make test        # fast suite (~12 s)
make test-all    # + end-to-end test with a real server subprocess
make lint        # ruff
```

Tests that need the bundled model are marked `model` and skip automatically if it's missing.
Remote Qdrant/Chroma contract tests run when `NOULO_TEST_QDRANT_URL` / `NOULO_TEST_CHROMA_URL`
are set. See [docs/development.md](docs/development.md).

## Project layout

```
src/noulo/
  api/            server.py (app factory, auth, CORS, errors) · routes.py · validation.py · schemas.py
  inference/      engine.py · model.py (ONNX NLI) · nli_backend.py · openai_backend.py
                  noul.py · choice.py · score.py · calibration.py
                  memory.py · embedder.py · stores/ (sqlite, qdrant, chroma)
  benchmark/      run.py · tune.py · metrics.py · report.py · data.py
  profiles/       tuned templates + calibration for every catalog model
  ui/static/      frontend (plain HTML/CSS/JS)
  repl/           interactive session: session.py · app.py · prompter.py · completion.py · render.py
  cli.py · model_menu.py · service.py · registry.py · config.py · runtime.py · embedded.py
models/           bundled models (Git LFS) + downloaded ones
benchmark/data/   evaluation dataset (calibration + test splits)
docs/             documentation
tests/            test suite
openapi.json      generated API specification
```

## Documentation

| Guide | |
|---|---|
| [Getting started](docs/getting-started.md) | Install, first run, first requests |
| [Examples](docs/examples.md) | Noul, Choice and Score in the session, CLI, curl, JavaScript and Python |
| [CLI reference](docs/cli.md) | Every command, the interactive session, exit codes |
| [REST API](docs/api.md) | Endpoints, schemas, errors, auth, examples |
| [Configuration](docs/configuration.md) | Every `NOULO_*` setting, `.env`, security |
| [Models](docs/models.md) | Basic and larger models, switching, OpenAI endpoints, your own ONNX, calibration |
| [Learning](docs/learning.md) | How the memory works, feedback, vector stores |
| [Model comparison](docs/model-comparison.md) | Measured results for every catalog model and quantisation |
| [GPU and accelerators](docs/gpu.md) | CoreML, CUDA, DirectML, ROCm: setup and measured speed-ups |
| [Architecture](docs/architecture.md) | Design, lifecycle, concurrency, invariants |
| [Development](docs/development.md) | Tests, TDD workflow, extending noulo |

The bundled model weights are third-party and Apache-2.0 licensed; see
[models/README.md](models/README.md).
