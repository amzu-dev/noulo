# Model comparison

Fourteen catalog models (nine basic, five larger) and four precisions (FP32, FP16, INT8, INT4), measured with:

```bash
noulo benchmark --all --download --tune --compare benchmark/results/comparison.md
```

- **Hardware:** Apple M1 Pro (10 cores), macOS 26, arm64, CPU only, ONNX Runtime CPU
  execution provider.
- **Data:** the bundled dataset in `benchmark/data/`, written for this project across
  support, billing, IT incidents, HR, e-commerce, legal, travel, security and more. It
  includes paraphrase, negation, numeric, temporal and lexical-overlap traps.
- **Protocol:** every model is **tuned only on the calibration split** (hypothesis templates,
  scoring method and Score temperature by grid search; Noul calibration method by 5-fold CV
  log loss). It is then **measured on the held-out test split**, in a fresh subprocess per
  model.

| Split | Noul (yes/no/unknown) | Choice | Score |
|---|---|---|---|
| calibration | 50 / 50 / 20 | 40 | 35 |
| test | 50 / 50 / 20 | 40 | 35 |

## Results (test split)

| Model | Quant | Size | Peak RAM¹ | Cold start | Choice acc | Noul acc | Noul ECE | Score MAE | P50 | P95 |
|---|---|---|---|---|---|---|---|---|---|---|
| **nli-deberta-v3-xsmall-int8** (default) | INT8 | 83.2 MiB | 328–443 MiB | 0.54–0.72 s | 67.5% | **91.0%** | 0.105 | 0.205 | 8.0 ms | 32–37 ms |
| nli-deberta-v3-xsmall-fp16 | FP16 | 136.2 MiB | 453.7 MiB | 0.87 s | 67.5% | 89.0% | 0.120 | 0.227 | 36.0 ms | 84.7 ms |
| nli-deberta-v3-xsmall-q4f16 | INT4 | 115.2 MiB | 420.2 MiB | 0.47 s | 67.5% | 90.0% | 0.107 | **0.200** | 11.9 ms | 55.7 ms |
| nli-deberta-v3-small-int8 | INT8 | 164.5 MiB | 417.8 MiB | 0.96 s | 65.0% | 88.0% | 0.115 | 0.266 | 7.9 ms | 39.5 ms |
| **nli-minilm2-l6-int8** | INT8 | 79.0 MiB | 232.2 MiB | 0.19 s | 52.5% | 87.0% | 0.116 | 0.253 | 5.1 ms | 23.6 ms |
| nli-mobilebert-int8 | INT8 | 25.7 MiB | 155.0 MiB | 0.66 s | 47.5% | 84.0% | 0.149 | 0.276 | 8.4 ms | 26.1 ms |
| nli-distilbert-int8 | INT8 | 64.5 MiB | 193.5 MiB | 0.13 s | 47.5% | 79.0% | 0.170 | 0.290 | 5.2 ms | 23.8 ms |
| zeroshot-xtremedistil-int8 | INT8 | **12.5 MiB** | **116.5 MiB** | **0.09 s** | 62.5% | 69.0% | 0.104 | 0.251 | **1.6 ms** | **6.5 ms** |
| **zeroshot-deberta-v3-xsmall-int8** | INT8 | 83.2 MiB | 333.4 MiB | 0.59 s | **70.0%** | 87.0% | **0.092** | 0.244 | 8.0 ms | 25.6 ms |

### Larger models (0.5–1 GB downloads)

| Model | Quant | Size | Peak RAM¹ | Cold start | Choice acc | Noul acc | Noul ECE | Score MAE | P50 | P95 |
|---|---|---|---|---|---|---|---|---|---|---|
| **zeroshot-deberta-v3-base-fp32** | FP32 | 704 MiB | 1234 MiB | 1.88 s | **85.0%** | **91.0%** | **0.078** | **0.115** | 28.5 ms | 134 ms |
| nli-deberta-v3-base-fp32 | FP32 | 704 MiB | 1419 MiB | 1.66 s | 70.0% | 89.0% | 0.101 | 0.175 | 30.1 ms | 146 ms |
| nli-bart-large-fp16 | FP16 | 778 MiB | 2010 MiB | 3.40 s | 70.0% | 91.0% | 0.095 | 0.205 | 182 ms | 565 ms |
| nli-deberta-v3-large-int8 *(experimental)* | INT8 | 613 MiB | 1001 MiB | 2.45 s | 65.0% | 77.0% | 0.089 | 0.227 | 47.4 ms | 208 ms |
| nli-deberta-v3-large-anli-int8 *(experimental)* | INT8 | 613 MiB | 1435 MiB | 2.99 s | 57.5% | 64.0% | 0.071 | 0.271 | 53.7 ms | 178 ms |

### Small LLMs (4-bit, next-token scoring)

| Model | Quant | Size | Peak RAM¹ | Cold start | Choice acc | Noul acc | Noul ECE | Score MAE | P50 | P95 |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-0.6b-q4f16 | INT4 | 543 MiB | 2588 MiB | 3.56 s | 67.5% | 63.0% | 0.077 | 0.381 | 551 ms | 680 ms |
| qwen2.5-0.5b-q4 | INT4 | 750 MiB | 2113 MiB | 3.11 s | 55.0% | 67.0% | 0.117 | 0.334 | 379 ms | 496 ms |
| gemma-3-1b-q4 | INT4 | 820 MiB | 1295 MiB | 3.44 s | 57.5% | 68.0% | 0.179 | 0.314 | 813 ms | 1053 ms |
| lfm2-1.2b-q4 | INT4 | 811 MiB | 856 MiB | 3.42 s | 87.5% | 85.0% | 0.281 | 0.221 | 929 ms | 1113 ms |

¹ Peak RSS of the benchmark process, which loads the model and runs every test item,
including long inputs that grow ONNX Runtime's memory arena. A running server with the
default model, the learning embedder and the API measured about **500 MiB RSS** (end-to-end
test). Ranges show two separate runs.

### Server RAM

The steady-state RSS of `noulo serve` after a few requests (learning memory on unless noted):

| Configuration | Server RSS |
|---|---|
| Default model | 500 MiB |
| Default model, learning off | 473 MiB |
| Default model, `NOULO_LOW_MEMORY=true` | 405–414 MiB |
| Default model, low memory, learning off | 387 MiB |
| Fast & light (`nli-minilm2-l6-int8`) | 302 MiB |
| Most accurate (`zeroshot-deberta-v3-base-fp32`) | 829 MiB |

Where the default's ~500 MiB goes: the ONNX session takes ~290 MiB for an 87 MB file,
because constant folding re-materialises the INT8 embedding table as FP32. The DeBERTa-v3
tokenizer (128k-piece vocabulary) takes ~100 MiB, the learning embedder ~30–50 MiB, and
Python with the libraries ~55 MiB. **Low-memory mode** skips constant folding: −90 MiB, but
P50 latency goes from 7.4 ms to 20.6 ms and results shift slightly (Noul 89%, Choice 65% on
the test split), because different kernels are fused.

Latency is measured per call across all three primitives. Choice and Score score every
candidate in one batch, so they cost more than Noul.

## Findings

1. **INT8 is the right default.** For DeBERTa-v3-xsmall, INT8 matches FP16 accuracy (it's
   slightly better on Noul) while being 39% smaller and 4.5× faster at P50. The 4-bit
   `q4f16` export is *larger* than INT8 (its embedding matrix stays in fp16) and slower on
   this CPU, with no accuracy gain. INT4 isn't worth it for encoders this small.
2. **Bigger isn't better here.** DeBERTa-v3-small (2× the size) scored lower than xsmall
   on every task after tuning. The small test set limits how much weight this can carry, but
   there's no evidence it justifies the extra 80 MB.
3. **Choice and Score benefit most from tuning.** For the default model, tuning on the
   calibration split raised held-out Choice accuracy from 60.0% to 67.5% and cut Score MAE from
   0.237 to 0.205. It also improved Noul ECE from 0.125 to 0.105 via Platt scaling. Every model
   ended up with a different best template, which is why tuning is per model (`src/noulo/profiles/`).
4. **Zero-shot-trained models are better at Choice, 3-class NLI models at Noul.** The
   2-class zero-shot DeBERTa has the best Choice accuracy and calibration. The 3-class NLI
   DeBERTa is best at Noul, because its *neutral* class maps naturally to "undetermined" (0.5).
5. **The best larger model is an unquantised *base*, not a quantised *large*.**
   `zeroshot-deberta-v3-base-fp32` is the most accurate model overall: Choice 85% (+17.5
   points over the default), Score MAE 0.115 (−44%), best-calibrated Noul. It costs about 3×
   the RAM and latency. The DeBERTa-v3-**large** INT8 exports come out *worse* than the
   83 MB default (Noul 77% and 64%). Dynamic INT8 quantisation evidently damages these large
   models, so they are marked `experimental` and not offered in the menus. A large model at
   FP16/FP32 (1.7 GB or more) would exceed the 1 GB tier.
6. **Small LLMs aren't a shortcut to accuracy.** Scored by next-token probability, the 4-bit
   Qwen3 0.6B, Qwen2.5 0.5B and Gemma 3 1B land *below* the 83 MB NLI default on every task,
   at 50–100× the latency and 3–6× the RAM. Liquid's **LFM2 1.2B** is the exception on Choice
   (87.5%, the best of all models) with good Noul accuracy (85%), but it's poorly calibrated
   (ECE 0.28) and takes ~0.9 s per call.
7. **The tiny extreme:** xtremedistil (12.5 MB, 116 MiB, 1.6 ms) is attractive for very
   constrained devices, but 69% Noul accuracy is too low for a general default.

## Menu choices (`noulo model`)

| Tier | Model | Reason |
|---|---|---|
| basic | `nli-deberta-v3-xsmall-int8` (bundled) | Best small-model Noul, strong everywhere, under the 100 MB target |
| basic | `nli-minilm2-l6-int8` | Lowest RAM with Noul ≥ 85%; fastest cold start among the accurate models |
| basic | `zeroshot-deberta-v3-xsmall-int8` | Best small-model Choice accuracy and Noul calibration |
| larger | `zeroshot-deberta-v3-base-fp32` | Most accurate overall |
| larger | `nli-deberta-v3-base-fp32` | Strong 3-class NLI (neutral = "undetermined") |
| larger | `nli-bart-large-fp16` | The classic zero-shot model, as a reference point (FP16 is slow on CPU: 182 ms per call, 2 GB RAM) |

## Caveats

- The test split is small (100 binary Noul, 40 Choice, 35 Score items). A 95% confidence
  interval on Choice accuracy is roughly ±14 points, so differences of a few points between
  models are not significant.
- The numbers are from one machine; re-run `make compare` on your target hardware.
- A sub-1B generative model via an OpenAI-compatible server (e.g. Ollama) can be evaluated
  the same way: `noulo benchmark --model <endpoint-id> --tune`.
