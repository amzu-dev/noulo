# Local Lightweight Choice / Score / Noul AI Engine

## Objective

Build a lightweight, fully local AI decision engine that accepts
natural-language input and returns one of three primitives: **Choice**,
**Score**, or **Noul**.

The model and inference runtime must run locally on CPU-only, low-RAM
systems. The application must also expose a **local REST API** so any
desktop app, CLI, agent, browser app, mobile app on an explicitly
enabled local network, or other software can use the engine after it
starts.

This is a semantic decision engine, not a chatbot.

## Core goals

-   CPU-only; no GPU required.
-   Prefer total runtime RAM below 500 MB.
-   Prefer quantised model below 100 MB; prototype maximum target 250
    MB.
-   No cloud AI or external inference API.
-   Fast startup and inference.
-   JavaScript/Node.js preferred for runtime.
-   Python allowed for training, conversion and quantisation only.
-   Windows, macOS and Linux support; x86-64 required and ARM64 strongly
    preferred.
-   UI and API must share one inference implementation.
-   Strict outputs only: Choice ID, Score 0--1, or Noul 0--1.

## Architecture

``` text
Any application
      |
      | HTTP
      v
Local REST API (127.0.0.1:8787 by default)
      |
      v
Decision Engine
      |
      v
Quantised semantic/NLI model
      |
 +----+----+
 |    |    |
 v    v    v
Choice Score Noul
A/B/C 0-1  0-1
```

The inference engine must be independent of both the API and UI so it
can also be embedded directly as a JavaScript module.

## Preferred stack

-   Node.js
-   Fastify preferred; Express acceptable
-   ONNX Runtime Node.js, Transformers.js, or another lightweight CPU
    inference runtime
-   INT8 quantisation initially; investigate INT4 where practical
-   Small transformer encoder/NLI model
-   Optional Tauri UI using plain HTML/CSS/JavaScript

Do not use Next.js, cloud services, remote inference, Docker as a
runtime requirement, or a large generative LLM unless compact semantic
models prove inadequate.

The project should ideally run with:

``` bash
npm install
npm start
```

## Project structure

``` text
/src
  /api
    server.js
    routes.js
    validation.js
  /inference
    engine.js
    model.js
    choice.js
    score.js
    noul.js
    calibration.js
  /ui
  /models
  /benchmark
  /tests
/openapi.json
```

## API service

Start a local REST service when the engine starts.

Defaults:

``` text
host: 127.0.0.1
port: 8787
base URL: http://127.0.0.1:8787
```

The host and port must be configurable.

Never bind to `0.0.0.0` by default. Network/LAN exposure must require
explicit configuration.

Load the model once at startup and keep it resident. Never reload it per
request.

Support headless operation:

``` bash
decision-engine serve
```

or:

``` bash
npm start -- --headless
```

Headless mode starts the model, inference engine and REST API without
opening the UI.

## API versioning

Use versioned endpoints:

``` text
GET  /health
GET  /api/v1/info
POST /api/v1/evaluate
POST /api/v1/choice
POST /api/v1/score
POST /api/v1/noul
GET  /openapi.json
```

Do not silently introduce breaking changes to `/api/v1`.

## Health API

### GET /health

Example:

``` json
{
  "status": "ok",
  "modelLoaded": true
}
```

Only report ready after the model has loaded successfully.

### GET /api/v1/info

Example:

``` json
{
  "name": "local-decision-engine",
  "version": "0.1.0",
  "model": "MODEL_NAME",
  "quantization": "INT8",
  "capabilities": ["choice", "score", "noul"]
}
```

Do not expose local filesystem paths.

## Noul

Noul evaluates a proposition against input and returns:

``` text
P(proposition is true | input)
```

Output is always between 0.0 and 1.0.

Conceptually:

``` text
0.0 = strongly NO
0.5 = ambiguous/uncertain
1.0 = strongly YES
```

### POST /api/v1/noul

Request:

``` json
{
  "input": "I checked my account and you have taken the subscription payment twice.",
  "proposition": "The customer reports being charged more than once."
}
```

Response:

``` json
{
  "type": "noul",
  "value": 0.97
}
```

Noul must use semantic inference/NLI probabilities or logits. Do not ask
a generative model to generate YES/NO or a percentage.

It must ideally generalise to propositions not individually seen during
training.

Example:

``` text
Input:
"The invoice has remained unpaid for 120 days."

Proposition:
"The customer has an overdue payment."
```

The model should understand the semantic relationship between input and
proposition rather than memorising fixed labels.

## Choice

Choice selects exactly one supplied option.

### POST /api/v1/choice

Request:

``` json
{
  "input": "The customer says their subscription payment was taken twice.",
  "question": "Which department should handle this?",
  "choices": [
    {"id": "A", "text": "Billing"},
    {"id": "B", "text": "Technical Support"},
    {"id": "C", "text": "Sales"}
  ]
}
```

Response:

``` json
{
  "type": "choice",
  "value": "A"
}
```

Evaluate each choice semantically against the input/question and select
the highest-scoring valid option.

Internally probabilities may be retained for diagnostics, but the normal
public response contains only the selected supplied ID.

The model must never invent another choice.

## Score

Score evaluates input against an ordered rubric and returns a normalised
value from 0.0 to 1.0.

### POST /api/v1/score

Request:

``` json
{
  "input": "The production system is unavailable for every customer.",
  "question": "How severe is this incident?",
  "rubric": [
    "insignificant",
    "low",
    "medium",
    "high",
    "critical"
  ]
}
```

Response:

``` json
{
  "type": "score",
  "value": 0.92
}
```

Treat rubric entries as ordered semantic candidates. Obtain a
probability distribution over levels, calculate the probability-weighted
expected level, and normalise by the maximum level.

Do not ask a generative model to generate an arbitrary number.

## Unified evaluation API

### POST /api/v1/evaluate

Noul:

``` json
{
  "type": "noul",
  "input": "...",
  "proposition": "..."
}
```

Choice:

``` json
{
  "type": "choice",
  "input": "...",
  "question": "...",
  "choices": [
    {"id": "A", "text": "..."},
    {"id": "B", "text": "..."}
  ]
}
```

Score:

``` json
{
  "type": "score",
  "input": "...",
  "question": "...",
  "rubric": ["low", "medium", "high"]
}
```

Route internally to the same primitive implementations used by the
dedicated endpoints.

## Direct JavaScript API

Expose the inference engine as an importable JavaScript module as well
as HTTP.

Example:

``` javascript
const result = await evaluate({
  type: "noul",
  input: "The invoice has remained unpaid for 120 days.",
  proposition: "The customer has an overdue payment."
});
```

Result:

``` javascript
{
  type: "noul",
  value: 0.96
}
```

Do not duplicate inference logic between the JS module and REST API.

## Strict model output contract

The model itself must not return free-form text, explanations, markdown,
conversational output, or generated JSON.

Permitted semantic outputs are:

``` text
Choice: one supplied choice ID
Score:  0.0–1.0
Noul:   0.0–1.0
```

The application layer creates JSON responses.

## Validation

Validate requests before inference.

Reject:

-   missing or empty input
-   missing/empty Noul proposition
-   missing Choice question
-   no choices
-   duplicate Choice IDs
-   invalid or insufficient Score rubric
-   unsupported primitive
-   malformed JSON
-   input exceeding configured limits

Example error:

``` json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Noul requires a proposition."
  }
}
```

Use appropriate HTTP status codes.

## API security

Loopback-only operation requires no authentication by default.

If network access is explicitly enabled, support optional API-key
authentication:

``` http
Authorization: Bearer <local-api-key>
```

Configure CORS conservatively. Do not default to unrestricted `*`
origins.

## Model architecture

Start by evaluating compact transformer encoder/NLI models rather than
generative LLMs.

``` text
Input + Question/Proposition/Candidate
              |
              v
      Quantised Encoder
              |
              v
   Semantic representation
              |
       +------+------+
       |      |      |
       v      v      v
    Choice  Score  Noul
```

Investigate compact families such as MiniLM, TinyBERT, DistilBERT and
small NLI/semantic transformer models. Benchmark candidates rather than
assuming one model is best.

If compact encoders cannot achieve acceptable semantic accuracy,
benchmark a sub-1B quantised generative model as a fallback using a
lightweight CPU runtime and constrained decoding.

## Quantisation and targets

Evaluate FP16, INT8 and INT4 where supported.

Prefer INT8 initially when it offers the best
compatibility/accuracy/size balance.

Targets:

``` text
Preferred model size: <100 MB
Prototype maximum target: <250 MB
Preferred total RAM: <500 MB
GPU requirement: none
```

Measure rather than assume:

-   model size
-   peak RAM
-   cold startup time
-   warm latency
-   throughput
-   Choice accuracy
-   Score MAE
-   Noul accuracy
-   Noul calibration

## Calibration

Do not assume raw neural-network probabilities are calibrated real-world
probabilities.

For Noul especially, evaluate calibration on held-out data.

Investigate:

-   temperature scaling
-   Platt scaling
-   isotonic regression

Calibration must be a separate layer so it can be updated without
replacing the base model.

## Optional local UI

Provide a minimal local interface primarily for testing.

Preferred:

``` text
Tauri + HTML/CSS/JavaScript
```

It should support input, primitive selection, Choice options, Score
rubric, Noul proposition, Run, and result.

The UI should consume the same local REST API exposed to other
applications. Do not build a traditional full-stack web application.

## Service lifecycle

Startup:

1.  Start runtime initialization.
2.  Load the quantised model once.
3.  Run a lightweight readiness check.
4.  Start/bind the API.
5.  Mark `/health` ready.
6.  Optionally launch the UI.

Shutdown:

1.  Stop accepting new requests.
2.  Safely finish/cancel outstanding inference.
3.  Release model resources.
4.  Close the API listener.

Implement request queuing/concurrency control if the inference runtime
is not safe for simultaneous calls.

## External application example

A consuming app should need only HTTP:

``` javascript
const response = await fetch("http://127.0.0.1:8787/api/v1/noul", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({
    input: "The invoice has been unpaid for four months.",
    proposition: "The customer has an overdue invoice."
  })
});

const result = await response.json();
console.log(result.value);
```

No AI SDK should be required by consumers.

## OpenAPI

Generate and serve a valid OpenAPI specification at:

``` text
/openapi.json
```

Document all public endpoints and schemas so client SDKs can later be
generated for JavaScript, Python, Go, Java, C#, Swift, Kotlin and other
languages.

## Benchmark suite

Provide:

``` bash
npm run benchmark
```

Example report shape (numbers must be measured, not hard-coded):

``` text
Model: MODEL_NAME
Quantisation: INT8
Model size:             ...
Peak RAM:               ...
Cold startup:           ...
Choice accuracy:        ...
Noul accuracy:          ...
Noul calibration:       ...
Score MAE:              ...
P50 inference:          ...
P95 inference:          ...
```

## Automated tests

Test:

-   model loading
-   health/readiness
-   Noul range
-   Choice membership in supplied IDs
-   Score range
-   request validation
-   malformed input
-   API contracts
-   concurrency
-   model failure handling
-   graceful shutdown

Hard invariants:

``` text
0 <= Noul <= 1
0 <= Score <= 1
Choice must equal one supplied choice ID
```

## Implementation phases

### Phase 1: Noul + API

Build the smallest useful vertical slice:

``` text
input + proposition
       |
       v
tiny NLI model
       |
       v
logits/probability
       |
       v
calibration
       |
       v
Noul 0–1
       |
       v
POST /api/v1/noul
```

Benchmark accuracy, calibration, memory and latency.

### Phase 2: Choice

Add semantic Choice evaluation and `/api/v1/choice`.

### Phase 3: Score

Add ordered rubric evaluation and `/api/v1/score`.

### Phase 4: Unified API

Add `/api/v1/evaluate`, `/api/v1/info`, OpenAPI and direct JS module
API.

### Phase 5: Optimisation

Compare models and quantisation levels; optimise startup, memory and
inference.

### Phase 6: UI and packaging

Add the minimal optional UI and package for supported operating systems.

## Deliverables

Produce:

1.  Working local AI engine.
2.  Local versioned REST API.
3.  Direct JavaScript inference API.
4.  Noul implementation.
5.  Choice implementation.
6.  Score implementation.
7.  Quantised model.
8.  OpenAPI specification.
9.  Benchmark suite.
10. Automated tests.
11. Representative test/evaluation dataset.
12. Optional lightweight local UI.
13. README with setup, startup and API examples.
14. Model comparison report.

README must document model name, size, quantisation, RAM, supported
OS/architectures, measured CPU performance, model replacement procedure,
API endpoints, Choice/Score/Noul semantics, security defaults and known
limitations.

## Most Important Design Principle

Do not build a miniature ChatGPT.

Build a small, reusable local semantic decision service:

``` text
              Any Application
                    |
                    | HTTP
                    v
             Local API Layer
                    |
                    v
              Decision Engine
                    |
                    v
           Tiny Quantised Model
                    |
          +---------+---------+
          |         |         |
          v         v         v
       Choice     Score      Noul
        A/B/C      0–1        0–1
```

Prioritise semantic accuracy, deterministic output, low resource
consumption, predictable behaviour, API stability and easy embedding
over generative capability.
