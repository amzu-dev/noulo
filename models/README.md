# Models

`noulo` ships with two small, INT8-quantised ONNX models (stored with Git LFS):

| Directory | Purpose | Source | License |
|---|---|---|---|
| `nli-deberta-v3-xsmall-int8/` | Default decision model (NLI cross-encoder) | [Xenova/nli-deberta-v3-xsmall](https://huggingface.co/Xenova/nli-deberta-v3-xsmall), ONNX export of [cross-encoder/nli-deberta-v3-xsmall](https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall) | Apache-2.0 |
| `minilm-l6-v2-int8/` | Sentence embedder for the learning memory | [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2), ONNX export of [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Apache-2.0 |

Each directory contains `model.onnx`, `tokenizer.json`, `config.json` and a
`manifest.json` (source, quantisation, sha256, size).

Other catalog models are downloaded on demand into this folder with
`noulo model` / `noulo model download <id>`. Tuned hypothesis templates and
Noul calibration for every catalog model ship inside the package
(`src/noulo/profiles/`); a `profile.json` / `calibration.json` placed in a
model directory overrides them (that is what `noulo benchmark --tune` writes).

If you cloned without Git LFS, `model.onnx` is a small pointer file. Fix it with
`git lfs install && git lfs pull`, or re-download: `noulo model download`.

See [docs/models.md](../docs/models.md) for switching models and plugging in your own.
