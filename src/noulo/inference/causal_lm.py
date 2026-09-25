"""A small causal LLM (Qwen, Gemma, LFM2, ...) on ONNX Runtime, used as a scorer.

Only the next-token distribution after a chat-formatted prompt is needed, so
one forward pass with empty KV caches is enough: no generation loop.

- ChatTemplate renders the model's own Jinja chat template (as transformers does).
- LabelTokens maps a label ("yes", "A") to the first token of its common
  spellings (" yes", "Yes", ...) and folds their probabilities together.
- build_feeds creates the session inputs for any decoder export, including empty
  past key/value tensors and architecture-specific states (e.g. LFM2 conv caches).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .devices import CPU_PLACEMENT, Placement, create_session
from .model import ensure_real_model_file
from .types import ModelLoadError

REQUIRED_FILES = ("model.onnx", "tokenizer.json", "tokenizer_config.json")
MAX_PROMPT_TOKENS = 4096


def _special(value: Any) -> str:
    return value.get("content", "") if isinstance(value, dict) else (value or "")


class ChatTemplate:
    def __init__(self, tokenizer_config: dict[str, Any]):
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        source = tokenizer_config.get("chat_template")
        if isinstance(source, list):
            named = {t.get("name"): t.get("template") for t in source}
            source = named.get("default") or next(iter(named.values()), None)
        if not source:
            raise ModelLoadError("Model has no chat template in tokenizer_config.json.")

        def raise_exception(message: str) -> None:
            raise ValueError(message)

        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        env.filters["tojson"] = lambda value, **kw: json.dumps(value, ensure_ascii=False)
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self._template = env.from_string(source)
        self._specials = {
            "bos_token": _special(tokenizer_config.get("bos_token")),
            "eos_token": _special(tokenizer_config.get("eos_token")),
        }

    def _render(self, messages: list[dict[str, str]]) -> str:
        return self._template.render(
            messages=messages, add_generation_prompt=True, enable_thinking=False, **self._specials
        )

    def render(self, messages: list[dict[str, str]]) -> str:
        try:
            return self._render(messages)
        except ValueError:
            if not messages or messages[0]["role"] != "system" or len(messages) < 2:
                raise
            # Templates without a system role: fold it into the first user turn.
            merged = {
                "role": messages[1]["role"],
                "content": f"{messages[0]['content']}\n\n{messages[1]['content']}",
            }
            return self._render([merged, *messages[2:]])


class LabelTokens:
    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer
        self._cache: dict[str, list[int]] = {}

    def ids(self, label: str) -> list[int]:
        if label not in self._cache:
            found: list[int] = []
            for variant in (
                label,
                f" {label}",
                label.capitalize(),
                f" {label.capitalize()}",
                label.upper(),
                f" {label.upper()}",
            ):
                encoded = self._tokenizer.encode(variant, add_special_tokens=False).ids
                if not encoded:
                    continue
                piece = self._tokenizer.decode([encoded[0]]).strip().lower()
                if piece and label.lower().startswith(piece) and encoded[0] not in found:
                    found.append(encoded[0])
            self._cache[label] = found
        return self._cache[label]

    def distribution(self, probs: np.ndarray, labels: Sequence[str]) -> tuple[list[float], float]:
        masses = [float(sum(probs[i] for i in self.ids(label))) for label in labels]
        total = sum(masses)
        if total <= 0:
            return [1.0 / len(labels)] * len(labels), 0.0
        return [m / total for m in masses], total


def build_feeds(inputs: Sequence[Any], ids: Sequence[int]) -> dict[str, np.ndarray]:
    """Session inputs for one forward pass over `ids` with empty caches."""
    n = len(ids)
    feeds: dict[str, np.ndarray] = {}
    for spec in inputs:
        if spec.name == "input_ids":
            feeds[spec.name] = np.array([list(ids)], dtype=np.int64)
        elif spec.name == "attention_mask":
            feeds[spec.name] = np.ones((1, n), dtype=np.int64)
        elif spec.name == "position_ids":
            feeds[spec.name] = np.arange(n, dtype=np.int64)[None, :]
        elif spec.name == "num_logits_to_keep":
            feeds[spec.name] = np.array(1, dtype=np.int64)
        else:  # past key/values or recurrent/conv state: empty or zero-initialised
            dtype = (
                np.float16
                if "float16" in spec.type
                else np.int64
                if "int64" in spec.type
                else np.float32
            )
            shape = [
                1
                if isinstance(d, str) and "batch" in d
                else 0
                if isinstance(d, str) or d is None
                else d
                for d in spec.shape
            ]
            feeds[spec.name] = np.zeros(shape, dtype=dtype)
    return feeds


class OnnxCausalLM:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        threads: int | None = None,
        low_memory: bool = False,
        placement: Placement | None = None,
    ):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        for name in REQUIRED_FILES:
            if not (model_dir / name).exists():
                raise ModelLoadError(f"Model file '{name}' is missing.")
        ensure_real_model_file(model_dir / "model.onnx")
        for data in model_dir.glob("*.onnx_data*"):
            ensure_real_model_file(data)
        try:
            options = ort.SessionOptions()
            options.log_severity_level = 3
            if threads:
                options.intra_op_num_threads = threads
            extra = {"disabled_optimizers": ["ConstantFolding"]} if low_memory else {}
            self._session, self.device = create_session(
                model_dir / "model.onnx", options, placement or CPU_PLACEMENT, **extra
            )
            self._inputs = self._session.get_inputs()
            self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
            config = json.loads((model_dir / "tokenizer_config.json").read_text())
        except Exception as exc:
            raise ModelLoadError(f"Failed to load LLM: {type(exc).__name__}") from exc
        self._template = ChatTemplate(config)
        self._labels = LabelTokens(self._tokenizer)

    def next_token_probs(self, text: str) -> np.ndarray:
        ids = self._tokenizer.encode(text, add_special_tokens=False).ids[-MAX_PROMPT_TOKENS:]
        (logits,) = self._session.run(["logits"], build_feeds(self._inputs, ids))
        last = np.asarray(logits[0, -1], dtype=np.float64)
        exp = np.exp(last - last.max())
        return exp / exp.sum()

    def label_distribution(
        self, messages: list[dict[str, str]], labels: Sequence[str]
    ) -> tuple[list[float], float]:
        probs = self.next_token_probs(self._template.render(messages))
        return self._labels.distribution(probs, labels)

    def close(self) -> None:
        self._session = None
