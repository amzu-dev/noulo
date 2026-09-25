"""ONNX Runtime wrapper for a quantised NLI cross-encoder (CPU only).

Loads the model once; `predict_logits` scores (premise, hypothesis) pairs in a
single padded batch. No PyTorch — only onnxruntime, tokenizers and numpy.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .types import ModelLoadError

_CANONICAL_LABELS = ("entailment", "neutral", "contradiction", "not_entailment")


def ensure_real_model_file(path: Path) -> None:
    """Fail clearly when a clone without Git LFS left a pointer instead of weights."""
    with path.open("rb") as fh:
        if fh.read(40).startswith(b"version https://git-lfs"):
            raise ModelLoadError(
                f"Model file '{path.name}' is a Git LFS pointer; run `git lfs pull` "
                "(or `noulo model download`) to fetch the weights."
            )


def parse_nli_labels(id2label: dict) -> dict[str, int]:
    """Map canonical NLI label names to logit indices from a HF `id2label` dict."""
    labels: dict[str, int] = {}
    for index, name in id2label.items():
        key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
        if key in _CANONICAL_LABELS:
            labels[key] = int(index)
    if "entailment" not in labels:
        raise ModelLoadError("Model config has no 'entailment' label; not an NLI model.")
    return labels


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class OnnxNliModel:
    """A cross-encoder NLI model: (premise, hypothesis) -> label logits."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        max_length: int = 512,
        threads: int | None = None,
        low_memory: bool = False,
    ):
        """low_memory skips constant folding, which otherwise re-materialises INT8
        embedding tables as FP32 (~110 MiB less RAM for DeBERTa-xsmall, slightly slower)."""
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        for name in ("model.onnx", "tokenizer.json", "config.json"):
            if not (model_dir / name).exists():
                raise ModelLoadError(f"Model file '{name}' is missing.")

        ensure_real_model_file(model_dir / "model.onnx")
        try:
            config = json.loads((model_dir / "config.json").read_text())
            self.labels = parse_nli_labels(config.get("id2label", {}))

            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.log_severity_level = 3
            if threads:
                options.intra_op_num_threads = threads
            extra = {"disabled_optimizers": ["ConstantFolding"]} if low_memory else {}
            self._session = ort.InferenceSession(
                str(model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"], **extra
            )
            self._input_names = {i.name for i in self._session.get_inputs()}

            self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
            limit = min(max_length, int(config.get("max_position_embeddings", max_length)))
            self._tokenizer.enable_truncation(limit, strategy="longest_first")
            pad_id = int(config.get("pad_token_id") or 0)
            pad_token = self._tokenizer.id_to_token(pad_id) or "[PAD]"
            self._tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)
        except ModelLoadError:
            raise
        except Exception as exc:  # onnxruntime/tokenizers raise assorted types
            raise ModelLoadError(f"Failed to load NLI model: {type(exc).__name__}") from exc

    def predict_logits(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros((0, len(self.labels)), dtype=np.float32)
        encodings = self._tokenizer.encode_batch([(p, h) for p, h in pairs])
        feeds = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        feeds = {name: value for name, value in feeds.items() if name in self._input_names}
        (logits,) = self._session.run(None, feeds)[:1]
        return np.asarray(logits, dtype=np.float32)

    def predict_probs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        return _softmax(self.predict_logits(pairs))

    def close(self) -> None:
        self._session = None
