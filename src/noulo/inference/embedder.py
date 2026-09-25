"""Sentence embedders used by the learning memory for similarity retrieval.

Every embedder satisfies the :class:`~noulo.inference.types.Embedder`
protocol: ``embed(texts)`` returns a float32 array of shape ``(len(texts), dim)``
whose rows are L2-normalised, so a dot product is a cosine similarity.
"""

from __future__ import annotations

import json
import re
import zlib
from collections.abc import Sequence
from pathlib import Path

import httpx
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from noulo.inference.model import ensure_real_model_file
from noulo.inference.types import BackendError, ModelLoadError

_WORD_RE = re.compile(r"\w+")


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)


def _hash_features(text: str) -> list[str]:
    """Lowercased word unigrams plus boundary-marked character 3-grams per word."""
    features: list[str] = []
    for word in _WORD_RE.findall(text.lower()):
        features.append(f"w:{word}")
        padded = f" {word} "
        features.extend(f"c:{padded[i : i + 3]}" for i in range(len(padded) - 2))
    return features


class HashingEmbedder:
    """Dependency-free, deterministic bag-of-features embedder.

    Features are hashed with CRC32 (stable across processes, unlike ``hash()``)
    into ``dim`` buckets. Similar surface strings get a high cosine similarity;
    there is no semantic understanding. Used as a lightweight fallback and in tests.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive.")
        self.dim = dim
        self.id = f"hashing-{dim}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in _hash_features(text):
                out[row, zlib.crc32(feature.encode("utf-8")) % self.dim] += 1.0
        return _l2_normalise(out)

    def close(self) -> None:
        """Nothing to release; present for protocol conformance."""


def _require_file(model_dir: Path, name: str) -> Path:
    """Return ``model_dir / name`` or raise ModelLoadError naming only the file."""
    path = model_dir / name
    if not path.is_file():
        raise ModelLoadError(f"Embedding model file '{name}' is missing.")
    return path


class OnnxEmbedder:
    """Mean-pooled sentence embeddings from a local ONNX transformer (CPU only).

    Expects ``model_dir`` to contain ``model.onnx`` (first output is the
    token-level ``last_hidden_state``) and a Hugging Face ``tokenizer.json``;
    ``config.json`` is optional and only consulted for ``hidden_size``.
    """

    _FEEDABLE_INPUTS = ("input_ids", "attention_mask", "token_type_ids")

    def __init__(
        self,
        model_dir: Path,
        *,
        id: str,
        max_length: int = 256,
        threads: int | None = None,
    ) -> None:
        model_dir = Path(model_dir)
        model_path = _require_file(model_dir, "model.onnx")
        ensure_real_model_file(model_path)
        tokenizer_path = _require_file(model_dir, "tokenizer.json")
        self.id = id
        try:
            self._session: ort.InferenceSession | None = _load_session(model_path, threads)
        except Exception as exc:
            raise ModelLoadError(_load_failure("model.onnx", exc)) from exc
        try:
            self._tokenizer = _load_tokenizer(tokenizer_path, max_length)
        except Exception as exc:
            raise ModelLoadError(_load_failure("tokenizer.json", exc)) from exc
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._output_name = self._session.get_outputs()[0].name
        self.dim = _hidden_size(self._session, model_dir)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        session = self._session
        if session is None:
            raise BackendError(f"Embedder '{self.id}' is closed.")
        encodings = self._tokenizer.encode_batch(list(texts))
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feeds = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": mask,
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        feeds = {name: arr for name, arr in feeds.items() if name in self._input_names}
        hidden = session.run([self._output_name], feeds)[0]
        return _l2_normalise(_mean_pool(hidden, mask))

    def close(self) -> None:
        self._session = None


def _load_failure(name: str, exc: Exception) -> str:
    # Runtime error messages often embed absolute paths, so only the type is kept.
    return f"Failed to load embedding model file '{name}' ({type(exc).__name__})."


def _load_session(model_path: Path, threads: int | None) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is not None:
        options.intra_op_num_threads = threads
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _load_tokenizer(tokenizer_path: Path, max_length: int) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    padding = tokenizer.padding or {}
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(
        pad_id=padding.get("pad_id", 0), pad_token=padding.get("pad_token", "[PAD]")
    )
    return tokenizer


def _hidden_size(session: ort.InferenceSession, model_dir: Path) -> int:
    """Embedding width from the output shape, falling back to config.json."""
    last = session.get_outputs()[0].shape[-1]
    if isinstance(last, int) and last > 0:
        return last
    config_path = model_dir / "config.json"
    if config_path.is_file():
        size = json.loads(config_path.read_text(encoding="utf-8")).get("hidden_size")
        if isinstance(size, int) and size > 0:
            return size
    raise ModelLoadError("Cannot determine embedding dimension of 'model.onnx'.")


def _mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Average token vectors over real (non-padding) tokens."""
    weights = mask[..., None].astype(np.float32)
    summed = (hidden * weights).sum(axis=1)
    return summed / np.maximum(weights.sum(axis=1), 1e-9)


class OpenAIEmbedder:
    """Embeddings from an OpenAI-compatible ``POST {base_url}/embeddings`` endpoint."""

    def __init__(
        self,
        *,
        id: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.id = id
        self._url = base_url.rstrip("/") + "/embeddings"
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout = timeout
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._closed = False
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        """Embedding width; probes the endpoint once if no call has been made yet."""
        if self._dim is None:
            return self._fetch(["dimension probe"]).shape[1]
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalise(self._fetch(texts))

    def close(self) -> None:
        """Stop accepting calls; closes the HTTP client only if this object created it."""
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            self._client.close()

    def _fetch(self, texts: Sequence[str]) -> np.ndarray:
        """Raw (unnormalised) embeddings for a non-empty batch; records ``dim``."""
        payload = self._post({"model": self._model, "input": list(texts)})
        matrix = _parse_embeddings(payload, expected=len(texts))
        self._dim = matrix.shape[1]
        return matrix

    def _post(self, body: dict) -> object:
        """POST to the embeddings endpoint; errors never include headers or bodies."""
        if self._closed:
            raise BackendError(f"Embedder '{self.id}' is closed.")
        try:
            response = self._client.post(
                self._url, json=body, headers=self._headers, timeout=self._timeout
            )
        except httpx.HTTPError as exc:
            raise BackendError(f"Embedding request failed ({type(exc).__name__}).") from None
        if response.status_code != 200:
            raise BackendError(f"Embedding endpoint returned HTTP {response.status_code}.")
        try:
            return response.json()
        except ValueError:
            raise BackendError("Embedding endpoint returned invalid JSON.") from None


def _parse_embeddings(payload: object, *, expected: int) -> np.ndarray:
    """Extract ``data[i].embedding`` ordered by ``index`` as a finite float32 matrix."""
    try:
        items = sorted(payload["data"], key=lambda item: item["index"])  # type: ignore[index]
        matrix = np.array([item["embedding"] for item in items], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise BackendError(f"Malformed embeddings response ({type(exc).__name__}).") from None
    if matrix.ndim != 2 or matrix.shape[0] != expected or matrix.shape[1] == 0:
        raise BackendError(
            f"Malformed embeddings response: expected {expected} vectors, got shape {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise BackendError("Malformed embeddings response: non-finite values.")
    return matrix
