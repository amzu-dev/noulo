# Configuration

noulo reads settings from, in order of precedence:

1. command-line flags (`noulo start --port 9000`),
2. environment variables prefixed with `NOULO_`,
3. the `.env` file in the working directory (or `--env-file` / `$NOULO_ENV_FILE`),
4. built-in defaults.

The easiest way to edit `.env` is the CLI, which validates values and keeps your comments:

```bash
noulo config show                        # effective settings, secrets redacted
noulo config set port 9000               # "port", "NOULO_PORT" and "noulo_port" all work
noulo config set cors_origins "https://app.example.com, https://admin.example.com"
noulo config unset port
noulo restart                            # most settings apply on the next start
```

[`.env.example`](../.env.example) lists every setting with its default.

## Server

| Setting | Default | |
|---|---|---|
| `NOULO_HOST` | `127.0.0.1` | Bind address. Anything other than loopback needs `NOULO_ALLOW_NETWORK=true` |
| `NOULO_PORT` | `8787` | |
| `NOULO_ALLOW_NETWORK` | `false` | Explicit opt-in for LAN/public exposure |
| `NOULO_API_KEY` | unset | When set, `/api/v1/*` requires `Authorization: Bearer <key>` |
| `NOULO_UI_ENABLED` | `true` | Serve the frontend at `/ui/` (`noulo serve` and `start --headless` turn it off) |
| `NOULO_LOG_LEVEL` | `info` | `critical`, `error`, `warning`, `info`, `debug` |
| `NOULO_RUN_DIR` | `data/run` | Pidfile and log for the background service |
| `NOULO_SHUTDOWN_TIMEOUT` | `10` | Seconds to drain in-flight requests on stop |

## CORS

| Setting | Default | |
|---|---|---|
| `NOULO_CORS_ENABLED` | `true` | Set `false` to send no CORS headers at all |
| `NOULO_CORS_ALLOW_LOCALHOST` | `true` | Allow `http(s)://localhost`, `127.0.0.1`, `[::1]` on any port |
| `NOULO_CORS_ORIGINS` | empty | Extra allowed origins, comma-separated. `*` is rejected at startup |

## Models

| Setting | Default | |
|---|---|---|
| `NOULO_MODEL` | `nli-deberta-v3-xsmall-int8` | Active model id at startup (`noulo model list`) |
| `NOULO_MODELS_DIR` | `models` | Where local models live |
| `NOULO_MODELS_FILE` | `models.json` | Your OpenAI-compatible endpoints, custom ONNX models and embedders |
| `NOULO_THREADS` | unset | ONNX Runtime intra-op threads (unset = runtime default) |
| `NOULO_OPENAI_BASE_URL` · `NOULO_OPENAI_MODEL` · `NOULO_OPENAI_API_KEY` | unset | Shortcut that registers one endpoint as model id `openai` |

See [models.md](models.md) for the `models.json` format.

## Execution and limits

| Setting | Default | |
|---|---|---|
| `NOULO_MAX_CONCURRENCY` | `1` | Inferences running at the same time |
| `NOULO_MAX_QUEUE` | `32` | Requests allowed to wait; more get `503 ENGINE_BUSY` |
| `NOULO_MAX_BODY_BYTES` | `65536` | Larger bodies get `413 PAYLOAD_TOO_LARGE` |
| `NOULO_MAX_INPUT_CHARS` | `5000` | Longer `input` gets `413 INPUT_TOO_LARGE` |
| `NOULO_MAX_TEXT_CHARS` | `1000` | Limit for proposition, question, option text and rubric level |
| `NOULO_MAX_CHOICES` | `20` | |
| `NOULO_MAX_RUBRIC_LEVELS` | `11` | |

## Learning memory

| Setting | Default | |
|---|---|---|
| `NOULO_LEARNING_ENABLED` | `true` | Record cases and let similar past cases adjust results |
| `NOULO_EMBEDDER` | `minilm-l6-v2-int8` | Embedder for similarity search; `hashing` needs no model; an `embedders` entry in `models.json` can point at an OpenAI-compatible `/embeddings` API |
| `NOULO_MEMORY_STORE` | `sqlite` | `sqlite`, `qdrant`, `chroma`, or `package.module:ClassName` |
| `NOULO_MEMORY_LOCATION` | `data/memory.sqlite3` | File or directory for local stores; `http(s)://...` for remote servers; `:memory:` for tests |
| `NOULO_MEMORY_COLLECTION` | `noulo_memory` | Collection/table name (the embedder id is appended) |
| `NOULO_MEMORY_API_KEY` | unset | API key for remote vector databases |
| `NOULO_MEMORY_TOP_K` | `8` | Neighbours considered |
| `NOULO_MEMORY_MIN_SIMILARITY` | `0.80` | Cosine similarity a past input needs to count |
| `NOULO_MEMORY_FEEDBACK_WEIGHT` | `1.0` | Weight of verified (feedback) cases |
| `NOULO_MEMORY_OBSERVED_WEIGHT` | `0.25` | Weight of past unverified outputs (`0` = feedback-only learning) |
| `NOULO_MEMORY_MAX_INFLUENCE` | `0.9` | Upper bound on how far memory can move a result |
| `NOULO_MEMORY_PRIOR_STRENGTH` | `0.5` | Evidence needed before memory has much influence |

See [learning.md](learning.md) for how these interact.

## Security defaults

| Default | Why |
|---|---|
| Binds `127.0.0.1` only, and refuses other hosts without `NOULO_ALLOW_NETWORK=true` | Nothing is exposed to the network by accident |
| No auth on loopback; `NOULO_API_KEY` enables bearer auth (compared in constant time) | Zero friction locally; a key is required once you open up |
| CORS limited to localhost origins; `*` rejected | Other websites can't script your local engine |
| Bodies, text lengths, option and level counts are capped | Bounded memory and CPU per request |
| `/info`, errors and the model list never include filesystem paths or API keys | No local information leaks through the API |
| `.env` and `models.json` are written with mode `600` | They can hold API keys |
| The OpenAI API key is read from an env var (`apiKeyEnv`) where possible | Keeps secrets out of files |
| Local by default | Text only leaves the machine if you configure a remote model endpoint or a remote vector store |

**Exposing noulo on a LAN:**

```bash
noulo config set host 0.0.0.0
noulo config set allow_network true
noulo config set api_key "$(openssl rand -hex 24)"
noulo config set cors_origins https://your-app.example
noulo restart
```
