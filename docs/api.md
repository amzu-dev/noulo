# REST API reference

- **Base URL:** `http://127.0.0.1:8787` (`NOULO_HOST` / `NOULO_PORT`)
- **Spec:** `GET /openapi.json` (OpenAPI 3.1; the committed copy is [`/openapi.json`](../openapi.json)).
  Interactive docs are at `/docs`. Use the spec to generate clients for JavaScript, Python, Go,
  Java, C#, Swift, Kotlin and other languages.
- **Versioning:** everything under `/api/v1` is stable; breaking changes would go to `/api/v2`.
- **Content type:** `application/json` in both directions.

## Decisions

### `POST /api/v1/noul`

P(proposition is true | input). `0.0` = strongly no, `0.5` = undetermined, `1.0` = strongly yes.

```json
// request
{"input": "I checked my account and you have taken the subscription payment twice.",
 "proposition": "The customer reports being charged more than once."}
// response 200
{"type": "noul", "value": 0.90}
```

With the default NLI model, the value is `P(entailment) + 0.5 · P(neutral)` from the model's
softmax, passed through the model's calibration layer. It is never generated text.

### `POST /api/v1/choice`

Selects exactly one supplied option.

```json
{"input": "The customer says their subscription payment was taken twice.",
 "question": "Which department should handle this?",
 "choices": [{"id": "A", "text": "Billing"},
             {"id": "B", "text": "Technical Support"},
             {"id": "C", "text": "Sales"}]}
```
```json
{"type": "choice", "value": "A"}
```

Every option is scored against the input and question, and the highest-probability option
wins. Ties go to the earliest supplied option. The answer is **always** one of your `id`s;
IDs can be any non-empty strings (`"billing"`, `"1"`, ...).

### `POST /api/v1/score`

Places the input on an ordered rubric (lowest first) and returns `0.0`–`1.0`.

```json
{"input": "The production system is unavailable for every customer.",
 "question": "How severe is this incident?",
 "rubric": ["insignificant", "low", "medium", "high", "critical"]}
```
```json
{"type": "score", "value": 0.77}
```

noulo gets a probability for each level, takes the probability-weighted expected level, and
divides by the highest level index. So a certain "insignificant" is `0.0` and a certain
"critical" is `1.0`. To get the nearest level back: `round(value * (len(rubric) - 1))`.

### `POST /api/v1/evaluate`

A single endpoint for all three primitives. Add `"type": "noul" | "choice" | "score"` to one
of the bodies above; it is routed to the same implementation as the dedicated endpoint.

```json
{"type": "noul", "input": "...", "proposition": "..."}
```

### Diagnostics and record ids

- `?diagnostics=true` adds a `diagnostics` object, for example:
  ```json
  {"type": "choice", "value": "A",
   "diagnostics": {"model": "nli-deberta-v3-xsmall-int8", "backend": "onnx-nli",
                   "probabilities": {"A": 0.94, "B": 0.04, "C": 0.02},
                   "learning": {"applied": false, "influence": 0.0, "matches": 0}}}
  ```
  Noul also includes `raw`, the uncalibrated probability.
- When learning is on, the response has an **`X-Record-Id`** header. The body stays strict.
  Use the id with `/api/v1/feedback`.

## Health and info

| | Response |
|---|---|
| `GET /health` | `200 {"status":"ok","modelLoaded":true}` once the model has loaded and passed its readiness check; otherwise `503 {"status":"<state>","modelLoaded":false}` where state is `created`, `starting`, `stopping`, `stopped` or `failed` |
| `GET /api/v1/info` | `{"name":"noulo","version":"0.1.0","model":"nli-deberta-v3-xsmall-int8","quantization":"INT8","capabilities":["choice","score","noul"],"backend":"onnx-nli","local":true,"learning":true}` |

`/health` and `/openapi.json` never require an API key.

## Models

| | |
|---|---|
| `GET /api/v1/models` | `{"active": "<id>", "models": [{"id","backend","model","quantization","local","installed","description"}]}` |
| `PUT /api/v1/models/active` `{"id": "nli-minilm2-l6-int8"}` | Load, warm up and atomically switch. In-flight requests finish on the old model, which is then released. Returns `{"active": {...}}`. Unknown id → `404 MODEL_NOT_FOUND`; not installed → `500 MODEL_LOAD_FAILED` (the current model stays active) |
| `POST /api/v1/models` `{"id","baseUrl","model","apiKeyEnv"?}` | Register an OpenAI-compatible endpoint (`201`) |
| `DELETE /api/v1/models/{id}` | Remove a registered endpoint (`204`; `409 MODEL_IN_USE` if it's active) |

A runtime switch doesn't change `.env`. For a persistent change, use `noulo model use <id>`
or set `NOULO_MODEL`.

## Learning

| | |
|---|---|
| `GET /api/v1/learning` | `{"enabled": true, "stats": {"records": 12, "verified": 3, "embedder": "minilm-l6-v2-int8"}}` |
| `PUT /api/v1/learning` `{"enabled": false}` | Turn recording and recall on/off at runtime (same response shape) |
| `GET /api/v1/learning/records?limit=50&offset=0&type=noul` | `{"records": [{"id","primitive","task","input","observedValue","observedChoice","verifiedValue","verifiedChoice","hits","modelId","createdAt","updatedAt"}]}` |
| `DELETE /api/v1/learning/records` | `{"deleted": 12}` |
| `POST /api/v1/feedback` | See below |

**Feedback by record id** (from `X-Record-Id`):

```json
{"recordId": "5f0c9a...", "expected": false}
```

**Feedback with a full request** ("teach", with no evaluation first):

```json
{"request": {"type": "choice", "input": "Charged twice", "question": "Which team?",
             "choices": [{"id": "A", "text": "Billing"}, {"id": "B", "text": "Sales"}]},
 "expected": "A"}
```

`expected` can be `true`/`false` or a number in `[0,1]` (Noul, Score), a supplied option id
(Choice), or a rubric level name (Score, with the `request` form). The response is
`{"recordId": "...", "verified": true}`. While learning is off, feedback returns
`409 LEARNING_DISABLED`.

## Errors

Every error has the same shape:

```json
{"error": {"code": "INVALID_REQUEST", "message": "Noul requires a proposition."}}
```

| HTTP | `code` | When |
|---|---|---|
| 400 | `MALFORMED_JSON` | Body isn't valid JSON |
| 401 | `UNAUTHORIZED` | `NOULO_API_KEY` is set and the bearer token is missing or wrong |
| 404 | `NOT_FOUND`, `MODEL_NOT_FOUND`, `RECORD_NOT_FOUND` | Unknown route, model or learning record |
| 405 | `METHOD_NOT_ALLOWED` | |
| 409 | `LEARNING_DISABLED`, `MODEL_IN_USE` | |
| 413 | `PAYLOAD_TOO_LARGE`, `INPUT_TOO_LARGE` | Body over `NOULO_MAX_BODY_BYTES`; a text field over its character limit |
| 422 | `INVALID_REQUEST`, `UNSUPPORTED_PRIMITIVE` | Validation failed (see below) or unknown `type` |
| 500 | `INFERENCE_ERROR`, `MODEL_LOAD_FAILED`, `INTERNAL_ERROR` | Local model failure (details are logged, not returned) |
| 502 | `UPSTREAM_ERROR` | A remote OpenAI-compatible endpoint failed |
| 503 | `MODEL_NOT_READY`, `ENGINE_BUSY` (+`Retry-After: 1`), `SHUTTING_DOWN` | |

Validation messages (these are stable, so clients can match on them):

| Condition | Message |
|---|---|
| missing/blank `input` | `Input is required.` |
| missing/blank Noul `proposition` | `Noul requires a proposition.` |
| missing Choice/Score `question` | `Choice requires a question.` / `Score requires a question.` |
| no choices | `Choice requires at least one choice.` |
| duplicate choice ids | `Choice IDs must be unique (duplicate: 'A').` |
| fewer than 2 rubric levels | `Score requires a rubric with at least two levels.` |
| duplicate rubric levels | `Score rubric levels must be unique (duplicate: 'low').` |
| unknown `type` on `/evaluate` | `Unsupported primitive. Use one of: choice, score, noul.` |

## Authentication, CORS, network

- Loopback-only by default and no authentication. With `NOULO_API_KEY=...`, every
  `/api/v1/*` call needs `Authorization: Bearer <key>`.
- CORS allows `http(s)://localhost|127.0.0.1|[::1]` on any port, plus `NOULO_CORS_ORIGINS`.
  It never allows `*`. The exposed headers are `X-Record-Id` and `Retry-After`.
- To serve a LAN: `NOULO_HOST=0.0.0.0 NOULO_ALLOW_NETWORK=true NOULO_API_KEY=...`.

## Client examples

**curl**

```bash
curl -s localhost:8787/api/v1/score -H 'Content-Type: application/json' -d '{
  "input": "The production system is unavailable for every customer.",
  "question": "How severe is this incident?",
  "rubric": ["insignificant","low","medium","high","critical"]}'
```

**JavaScript**

```javascript
const res = await fetch("http://127.0.0.1:8787/api/v1/choice", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    input: "The customer says their subscription payment was taken twice.",
    question: "Which department should handle this?",
    choices: [{ id: "A", text: "Billing" }, { id: "B", text: "Technical Support" }],
  }),
});
const { value } = await res.json(); // "A"
```

**Python (HTTP)**

```python
import httpx
r = httpx.post("http://127.0.0.1:8787/api/v1/evaluate",
               json={"type": "noul", "input": "The invoice has been unpaid for four months.",
                     "proposition": "The customer has an overdue invoice."})
r.raise_for_status()
print(r.json()["value"])
```

**Python (in-process, no server)**: see the [README](../README.md#python).

## Concurrency and lifecycle

- The model is loaded **once** at startup and stays resident.
- `NOULO_MAX_CONCURRENCY` inferences run at a time (default 1), and up to `NOULO_MAX_QUEUE`
  more wait. Beyond that the API returns `503 ENGINE_BUSY` with `Retry-After`.
- On shutdown (SIGTERM, Ctrl+C, `noulo stop`), noulo stops accepting requests, finishes
  in-flight inference (up to `NOULO_SHUTDOWN_TIMEOUT` seconds), releases the model and
  closes the listener.
