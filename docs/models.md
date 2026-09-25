# Models

noulo separates **what** it answers (Noul, Choice, Score) from **which model** answers it.
A model is either:

- a **local ONNX NLI model** (`onnx-nli`) run on the CPU with ONNX Runtime, or
- an **OpenAI-compatible endpoint** (`openai`): OpenAI, Ollama, LM Studio, vLLM,
  llama.cpp server, or anything that implements `/v1/chat/completions`.

Every model goes through the same engine, validation, calibration, learning memory and
output invariants.

## Choosing a model

`noulo model`, `/model` in the interactive session, and the frontend's model panel all list
the same options. Each shows the **download size**, **quantisation**, **measured RAM** (peak
while benchmarking) and measured accuracy on the held-out test split.

### Basic models (< 100 MB)

| Menu name | Id | Download | Quant | RAM (peak) | Noul | Choice | Score MAE | P50 |
|---|---|---|---|---|---|---|---|---|
| Balanced ★ (bundled) | `nli-deberta-v3-xsmall-int8` | 87 MB | INT8 | 443 MB | **91%** | 67.5% | 0.205 | 8 ms |
| Fast & light | `nli-minilm2-l6-int8` | 83 MB | INT8 | 232 MB | 87% | 52.5% | 0.253 | 5 ms |
| Best at Choice | `zeroshot-deberta-v3-xsmall-int8` | 87 MB | INT8 | 333 MB | 87% | 70.0% | 0.244 | 8 ms |

### Larger models (0.5–1 GB)

| Menu name | Id | Download | Quant | RAM (peak) | Noul | Choice | Score MAE | P50 |
|---|---|---|---|---|---|---|---|---|
| Most accurate | `zeroshot-deberta-v3-base-fp32` | 739 MB | FP32 (not quantised) | 1234 MB | 91% | **85.0%** | **0.115** | 29 ms |
| Larger NLI | `nli-deberta-v3-base-fp32` | 739 MB | FP32 (not quantised) | 1419 MB | 89% | 70.0% | 0.175 | 30 ms |
| BART (slow) | `nli-bart-large-fp16` | 816 MB | FP16 | 2010 MB | 91% | 70.0% | 0.205 | 182 ms |

Pick a larger model when accuracy matters more than footprint. **Most accurate** gets 85%
Choice accuracy (the small default gets 67.5%) and nearly halves the Score error, at about
3× the RAM and latency. All figures were measured on an Apple M1 Pro; see
[model-comparison.md](model-comparison.md).

### Small LLMs (4-bit, ~0.5–0.9 GB)

Prominent open-weight LLMs, run locally on ONNX Runtime with 4-bit (INT4) weights:

| Menu name | Id | Download | Quantisation | RAM (peak) | Noul | Choice | Score MAE | P50 | License |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3 0.6B | `qwen3-0.6b-q4f16` | 570 MB | INT4 (4-bit weights, FP16 activations) | 2588 MB | 63.0% | 67.5% | 0.381 | 551 ms | Alibaba, Apache-2.0 |
| Qwen2.5 0.5B | `qwen2.5-0.5b-q4` | 786 MB | INT4 (4-bit weights) | 2113 MB | 67.0% | 55.0% | 0.334 | 379 ms | Alibaba, Apache-2.0 |
| Gemma 3 1B | `gemma-3-1b-q4` | 859 MB | INT4 (4-bit weights) | 1295 MB | 68.0% | 57.5% | 0.314 | 813 ms | Google, Gemma terms of use |
| LFM2 1.2B | `lfm2-1.2b-q4` | 850 MB | INT4 (4-bit weights) | 856 MB | 85.0% | 87.5% | 0.221 | 929 ms | Liquid AI, LFM Open License |

**How an LLM answers without generating anything.** noulo renders the model's own chat
template around the same classification prompt the OpenAI-compatible backend uses. It runs
**one forward pass** and reads the next-token probability of each label: `yes`/`no`/`unknown`
for Noul, `A`, `B`, ... for Choice and Score, with case and leading-space variants folded
together. There's no generation loop, so outputs stay within the contract (0–1, supplied IDs
only). Noul calibration is fitted per model, like the NLI models. The trade-off is one
forward pass of a 0.6–1.2B model per request: 0.4–0.9 s per call on the CPU and 0.9–2.6 GB of
RAM.

**What the measurements say:** **LFM2 1.2B** is the only one worth choosing over the NLI
models: it has the best Choice accuracy of any model (87.5%) and good Noul accuracy (85%),
though its raw Noul probabilities are poorly calibrated (ECE 0.28). Qwen3 0.6B, Qwen2.5 0.5B
and Gemma 3 1B all score below the 83 MB default on Noul, Choice and Score. Small LLMs tend to
over-use "unknown" and have letter-position biases. They're offered because you asked for
them and because they may suit your inputs better. Measure on your own data with
`noulo benchmark --model <id>`.

MiniMax's open models are hundreds of billions of parameters, and MiniCPM and Llama 3.2 1B have
no ONNX export that fits this size range (Llama 3.2 1B's 4-bit export is 1.1 GB). You can run
any of them, or any GGUF model, through Ollama / llama.cpp as an
[OpenAI-compatible endpoint](#openai-compatible-endpoints), or register your own ONNX LLM export
(see [below](#plug-in-your-own-onnx-model)).

**Why no INT8 "large" model?** Two DeBERTa-v3-*large* INT8 exports (643 MB each) are in the
catalog as `experimental`. They're installable with `noulo model use <id>` but aren't offered
in the menus, because dynamic INT8 quantisation measurably damages them: Noul 77% and 64%,
below the small default, while using 1.0–1.4 GB of RAM.

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
| `zeroshot-deberta-v3-base-fp32` | MoritzLaurer/deberta-v3-base-zeroshot-v2.0 | FP32 | larger tier: most accurate (739 MB) |
| `nli-deberta-v3-base-fp32` | cross-encoder/nli-deberta-v3-base | FP32 | larger tier (739 MB) |
| `nli-bart-large-fp16` | Xenova/bart-large-mnli | FP16 | larger tier (816 MB) |
| `nli-deberta-v3-large-int8` | cross-encoder/nli-deberta-v3-large | INT8 | experimental: quantisation-damaged (643 MB) |
| `nli-deberta-v3-large-anli-int8` | MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli | INT8 | experimental: quantisation-damaged (643 MB) |
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

**Your own ONNX LLM.** Any decoder-only export with `model.onnx` (external weight files next
to it are fine), `tokenizer.json` and a `tokenizer_config.json` containing a chat template
works. For example, most `onnx-community/*` repos:

```bash
noulo model add-onnx --id my-llm --path ./my-llm-q4 --kind llm --quantization INT4
noulo benchmark --model my-llm --tune        # fits the Noul calibration
noulo model use my-llm
```

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
