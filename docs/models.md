# Models

noulo separates **what** it answers (Noul, Choice, Score) from **which model** answers it.
A model is either:

- a **local ONNX NLI model** (`onnx-nli`) run on the CPU with ONNX Runtime, or
- an **OpenAI-compatible endpoint** (`openai`): OpenAI, Ollama, LM Studio, vLLM,
  llama.cpp server, or anything that implements `/v1/chat/completions`.

Every model goes through the same engine, validation, calibration, learning memory and
output invariants.

## The three curated models

| # | Id | Size | Noul acc | Choice acc | Score MAE | Server RAM | Pick it when |
|---|---|---|---|---|---|---|---|
| 1 | `nli-deberta-v3-xsmall-int8` (bundled) | 83 MB | **91.0%** | 67.5% | 0.205 | ~290 MiB | You want the best all-rounder (default) |
| 2 | `nli-minilm2-l6-int8` | 79 MB | 87.0% | 52.5% | 0.253 | less (232 MiB peak in benchmark) | RAM or start-up time matters most (0.2 s cold start, 5 ms/call) |
| 3 | `zeroshot-deberta-v3-xsmall-int8` | 83 MB | 87.0% | **70.0%** | 0.244 | similar to #1 | Choice-heavy workloads; best-calibrated Noul (ECE 0.092) |

These figures are from the held-out test split; the full table and method are in
[model-comparison.md](model-comparison.md).

## Switching models

| How | Persistent? | Downtime |
|---|---|---|
| `noulo model` (picker) / `noulo model use <id>` | yes, writes `NOULO_MODEL` to `.env` | brief restart of the managed service (or a live switch for a manually started server) |
| `PUT /api/v1/models/active {"id": "..."}` or the frontend's **Switch** button | no, runtime only | none: the new model loads and warms up while the old one keeps serving; in-flight requests finish on the old model |
| `NOULO_MODEL=<id>` in `.env`, then `noulo restart` | yes | restart |
| `noulo start --model <id>` | no, this run only | n/a |

If a catalog model isn't installed, `noulo model use` downloads it first into
`models/<id>/` (it's written atomically and records a sha256 manifest).

### Model replacement procedure (checklist)

1. `noulo model list`: find the id (`installed` column).
2. `noulo model use <id>`: download, persist, restart.
3. `noulo status`: confirm the model, backend and quantisation.
4. Optional: `noulo benchmark --model <id>` to measure it on your machine.
5. To go back: `noulo model use nli-deberta-v3-xsmall-int8`.

## Full catalog

`noulo model list` shows everything. Downloadable catalog models:

| Id | Source (Hugging Face) | Quant | Notes |
|---|---|---|---|
| `nli-deberta-v3-xsmall-int8` | Xenova/nli-deberta-v3-xsmall | INT8 | bundled default |
| `nli-deberta-v3-xsmall-fp16` | Xenova/nli-deberta-v3-xsmall | FP16 | reference; slower and larger |
| `nli-deberta-v3-xsmall-q4f16` | Xenova/nli-deberta-v3-xsmall | INT4 | 4-bit weights, fp16 embeddings (115 MB) |
| `nli-deberta-v3-small-int8` | Xenova/nli-deberta-v3-small | INT8 | larger (165 MB), not more accurate here |
| `nli-minilm2-l6-int8` | cross-encoder/nli-MiniLM2-L6-H768 | INT8 | arm64/x86-64-specific quantised files |
| `nli-mobilebert-int8` | Xenova/mobilebert-uncased-mnli | INT8 | 26 MB |
| `nli-distilbert-int8` | Xenova/distilbert-base-uncased-mnli | INT8 | |
| `zeroshot-xtremedistil-int8` | MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33 | INT8 | 13 MB, 116 MiB RAM; weak Noul (69%) |
| `zeroshot-deberta-v3-xsmall-int8` | MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33 | INT8 | 2-class zero-shot |
| `minilm-l6-v2-int8` | Xenova/all-MiniLM-L6-v2 | INT8 | the learning-memory embedder (bundled) |

## OpenAI-compatible endpoints

```bash
# Ollama running locally
noulo model add-endpoint --id ollama-qwen --base-url http://127.0.0.1:11434/v1 --model qwen2.5:1.5b
# OpenAI (the key is read from the environment at load time)
export OPENAI_API_KEY=sk-...
noulo model add-endpoint --id gpt --base-url https://api.openai.com/v1 --model gpt-4o-mini \
    --api-key-env OPENAI_API_KEY
noulo model use ollama-qwen
```

Or set `NOULO_OPENAI_BASE_URL`, `NOULO_OPENAI_MODEL` and `NOULO_OPENAI_API_KEY` to register
one endpoint as the id `openai`.

**How it decides without generating numbers.** noulo sends one short classification prompt
with `temperature=0`, `max_tokens=1`, `logprobs=true` and `top_logprobs=20`, then reads the
probability of each allowed **label token**:

| Primitive | Labels | Value |
|---|---|---|
| Noul | `yes` / `no` / `unknown` | `P(yes) + 0.5 · P(unknown)` |
| Choice | `A`, `B`, `C`, ... (mapped back to your IDs) | distribution over your options |
| Score | `A` (lowest) ... | distribution over levels → expected level |

Token variants (`" Yes"`, `"yes."`) are aggregated. If the server doesn't return logprobs,
noulo parses the single generated label strictly (one-hot, logged as a warning). Anything
else is an error. It never invents an option.

Remote endpoints send your text to that server; `GET /api/v1/info` reports
`"local": false` for them. `noulo benchmark --model <endpoint-id> --tune` fits a Noul
calibration for an endpoint too.

## Plug in your own ONNX model

Any sequence-classification NLI or zero-shot model with an `entailment` label works: 3-class
(entailment/neutral/contradiction) or 2-class (entailment/not_entailment).

```bash
pip install "optimum[onnxruntime]"            # only needed for conversion
optimum-cli export onnx --model MoritzLaurer/deberta-v3-base-zeroshot-v2.0 --task text-classification my-nli/
optimum-cli onnxruntime quantize --onnx_model my-nli/ --avx2 -o my-nli-int8/   # or --arm64
mv my-nli-int8/model_quantized.onnx my-nli-int8/model.onnx
cp my-nli/tokenizer.json my-nli/config.json my-nli-int8/
# my-nli-int8/ must contain model.onnx, tokenizer.json and config.json
noulo model add-onnx --id my-nli --path ./my-nli-int8 --quantization INT8
noulo benchmark --model my-nli --tune         # fits templates + calibration (calibration split)
noulo model use my-nli
```

`add-onnx` checks the files exist and records the absolute path in `models.json`. The path
is never exposed through the API.

## `models.json`

`noulo model add-endpoint` and `add-onnx` write this file (`NOULO_MODELS_FILE`, mode 600).
You can also edit it by hand:

```json
{
  "models": [
    {"id": "ollama-qwen", "backend": "openai", "baseUrl": "http://127.0.0.1:11434/v1",
     "model": "qwen2.5:1.5b"},
    {"id": "gpt", "backend": "openai", "baseUrl": "https://api.openai.com/v1",
     "model": "gpt-4o-mini", "apiKeyEnv": "OPENAI_API_KEY", "timeout": 30,
     "extraBody": {"seed": 0}},
    {"id": "my-nli", "backend": "onnx-nli", "path": "/abs/path/my-nli-int8",
     "quantization": "INT8", "profile": {"choice_template": "answer-is"}}
  ],
  "embedders": [
    {"id": "openai-embed", "backend": "openai", "baseUrl": "https://api.openai.com/v1",
     "model": "text-embedding-3-small", "apiKeyEnv": "OPENAI_API_KEY"}
  ]
}
```

## Calibration and tuning

Raw neural-network probabilities aren't real-world probabilities, so each model has two
small, replaceable files. Neither needs changes to the model itself:

- **`calibration.json`** maps the raw Noul probability to a calibrated one. The methods are
  `identity`, `temperature`, `platt` and `isotonic`. `--tune` picks the method with the lowest
  5-fold cross-validated log loss on the **calibration split** and fits it there. Example:
  `{"method": "platt", "a": 0.769, "b": 0.487}`.
- **`profile.json`** says how Choice/Score candidates become NLI hypotheses: the template
  (`question-answer`, `answer-is`, `level-only`, ...), the premise (`input` or
  `input+question`), the scoring method (`entailment` or `entailment-contradiction`), and the
  Score softmax temperature. `--tune` grid-searches these on the calibration split.

Tuned files for every catalog model ship in `src/noulo/profiles/<id>/`. A file in
`models/<id>/` overrides the packaged one; that's where `noulo benchmark --tune` writes.
The test split is never used for tuning.
