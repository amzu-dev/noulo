# Architecture

**Ways in.** Every interface ends up in the same engine:

```mermaid
flowchart LR
  accTitle: Ways into noulo
  accDescr: Applications, the web frontend and the CLI call the FastAPI app over HTTP; Python code calls the DecisionEngine in-process. Both paths reach the same engine.

  APP["Any application"] -- HTTP --> API
  UI["Web frontend /ui"] -- HTTP --> API
  CLI["noulo CLI and session"] -- HTTP --> API
  API["FastAPI app<br/>auth · CORS · limits<br/>validation · errors"] --> ENG
  PY["Python<br/>noulo.evaluate()"] -- in-process --> ENG
  ENG["DecisionEngine<br/>lifecycle · concurrency<br/>switching · invariants"]

  classDef core fill:#282b23,stroke:#282b23,color:#f1f1e8
  class ENG core
```

**Inside the engine.** Calibration, the three primitives and the learning memory are shared;
only the decision backend changes with the model:

```mermaid
flowchart LR
  accTitle: Inside the DecisionEngine
  accDescr: The engine uses a calibration layer, the three primitives, a learning memory with an embedder and a vector store, and one decision backend: local NLI, local 4-bit LLM or an OpenAI-compatible endpoint.

  ENG["DecisionEngine"] --> CAL["Calibration<br/>Noul only"]
  ENG --> PRIM["Primitives<br/>noul · choice · score"]
  ENG --> MEM["LearningMemory"]
  ENG --> BE{{"DecisionBackend"}}
  MEM --> EMB["Embedder"]
  MEM --> VS[("VectorStore<br/>sqlite · qdrant<br/>chroma · custom")]
  BE --> NLI["NliBackend<br/>OnnxNliModel · CPU"]
  BE --> LLM["LlmBackend<br/>OnnxCausalLM · 4-bit LLMs"]
  BE --> OAI["OpenAIBackend<br/>chat completions + logprobs"]

  classDef core fill:#282b23,stroke:#282b23,color:#f1f1e8
  class ENG core
```

**One inference implementation.** The REST API, CLI, frontend and Python module all end up
in `DecisionEngine.evaluate()`, and the REST API and Python module share the same request
validation (`api/validation.py`). There is no second code path to drift.

## Request pipeline

```mermaid
flowchart TB
  accTitle: Request pipeline
  accDescr: A request is validated, gets a slot, is scored by the backend, calibrated, optionally blended with the learning memory, reduced to the primitive, checked against the invariants and recorded.

  V["1 · Validate<br/>schemas and limits"] --> S["2 · Acquire a slot<br/>bounded concurrency and queue"]
  S --> B["3 · Backend<br/>raw probabilities"]
  B --> C["4 · Calibrate<br/>Noul only"]
  C --> L["5 · Learn<br/>blend similar past cases"]
  L --> R["6 · Reduce<br/>a supplied ID or a 0–1 value"]
  R --> G["7 · Guard invariants"]
  G --> OUT(["Strict answer"])
  G -.-> REC[("8 · Record the case")]
  S -. queue full .-> BUSY["503 ENGINE_BUSY"]
  G -. invalid output .-> ERR["500 / 502, never returned"]

  classDef core fill:#282b23,stroke:#282b23,color:#f1f1e8
  class OUT core
```

1. **Validate** (`parse_request`): Pydantic schemas plus stable, human-readable messages and
   configurable limits. The same schemas generate the OpenAPI document.
2. **Acquire** a slot: bounded concurrency (`NOULO_MAX_CONCURRENCY`) and a bounded queue
   (`NOULO_MAX_QUEUE` → `503 ENGINE_BUSY`). The request also pins the *current* backend, so a
   model switch can't pull it out from under a running request.
3. **Backend** returns raw probabilities: a Noul scalar, or a distribution over options or levels.
4. **Calibrate** (Noul only).
5. **Learn** (optional): recall similar cases for the same task and blend them in.
6. **Reduce** to the primitive: Choice = argmax over the *supplied* IDs (ties go to the first);
   Score = expected level ÷ max level.
7. **Guard invariants:** non-finite or out-of-range values and wrong-length distributions
   raise `InferenceError` (HTTP 500/502). They're never returned.
8. **Record** the case in memory with the pre-blend model outcome. Learning failures are
   logged and never break inference.

## NLI formulation

| Primitive | Premise | Hypothesis | Value |
|---|---|---|---|
| Noul | input | proposition | `P(entail) + 0.5·P(neutral)` (3-class) or `P(entail)` (2-class) |
| Choice | input (or input + question) | template(question, option), per option | softmax over entailment (or entailment − contradiction) logits across options |
| Score | input (or input + question) | template(question, level), per level | same distribution → Σ p·level ÷ (n − 1) |

All candidate hypotheses for a request are scored in **one padded batch**. The template,
premise mode, method and Score temperature are per-model, tuned data (`profile.json`).

## Lifecycle

```mermaid
stateDiagram-v2
  accTitle: Engine lifecycle
  accDescr: The engine loads the model once and passes a readiness check before serving. A model switch loads the new model while the old one keeps serving. On shutdown it drains in-flight requests before releasing the model.

  [*] --> created
  created --> starting: start()
  starting --> ready: model loaded and checked
  starting --> failed: ModelLoadError
  ready --> switching: switch model
  switching --> ready: new model live
  ready --> stopping: noulo stop or SIGTERM
  stopping --> stopped: in-flight done or timeout
  stopped --> [*]
  failed --> [*]
```

**Startup** (uvicorn lifespan, before the socket is bound):
1. build the registry and engine from settings
2. load the model once (`ModelLoadError` → the process exits with a clear message)
3. readiness check: one tiny inference must produce finite output
4. load the calibration, and open the learning memory if enabled
5. bind the API; `/health` → `200 ok`
6. `noulo start` opens the frontend

**Shutdown** (SIGTERM / Ctrl+C / `noulo stop`):
1. uvicorn stops accepting connections
2. in-flight requests finish (new engine calls get `503 SHUTTING_DOWN`), up to
   `NOULO_SHUTDOWN_TIMEOUT`
3. the model session and memory store are released
4. the listener closes

**Model switch** (`PUT /api/v1/models/active`): the new model is loaded and warmed up
while the old one keeps serving. The swap is atomic, and the old backend is closed only once
its last in-flight request finishes. If loading fails, the old model stays active.

```mermaid
sequenceDiagram
  accTitle: Switching models at runtime
  accDescr: The new model loads and passes a readiness check while the current model keeps serving; after an atomic swap the old model is closed once its last request finishes.

  participant C as Client
  participant E as DecisionEngine
  participant O as Current model
  participant N as New model

  C->>E: PUT /api/v1/models/active
  E->>N: load and readiness check
  Note over O: keeps serving requests meanwhile
  N-->>E: ready
  E->>E: atomic swap
  E-->>C: 200 active model
  O-->>E: last in-flight request finishes
  E->>O: close()
  Note over E,N: if loading fails, the current model stays active
```

## Background service

`noulo start` spawns `python -m noulo.cli --env-file <abs .env> serve ...` detached, writes
`data/run/noulo.pid` (pid, host, port, UI mode), and waits for `/health`. `restart` reuses
the recorded address and UI mode. `stop` sends SIGTERM and escalates to SIGKILL only after
the timeout.

## Why these choices

| Decision | Reason |
|---|---|
| Encoder NLI models, not a generative LLM | Deterministic, calibratable probabilities; 80 MB instead of GBs; ~8 ms per call on CPU |
| ONNX Runtime + `tokenizers`, no PyTorch | Keeps a running server at ~300–500 MiB (model-dependent) and installs fast on every OS/arch |
| INT8 dynamic quantisation | Best measured size/speed/accuracy balance (see [model-comparison.md](model-comparison.md)) |
| Calibration as a separate JSON layer | Can be refitted or replaced without touching the model |
| Logprobs for remote LLMs, next-token probabilities for local LLMs | Honours "never ask a generative model for a number" while still supporting them; local LLMs need one forward pass, no generation loop |
| Learning as a probability blend | Works with any backend, is bounded by `max_influence`, can never invent an option |
| Pluggable vector store | SQLite with zero setup; Qdrant/Chroma/custom when you outgrow it |
| Python/FastAPI | Requested by the project owner (the original spec preferred Node.js) |
