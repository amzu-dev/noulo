"""Model registry: which decision models and embedders exist, which are
installed, how to download them and how to turn an id into a live backend.

Sources, in order:
1. Built-in catalog of downloadable, quantised ONNX models (`CATALOG`).
2. User entries in `models.json` (OpenAI-compatible endpoints, custom ONNX dirs).
3. An `openai` entry synthesised from NOULO_OPENAI_* settings, when configured.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import is_loopback
from .inference.calibration import Calibrator, load_calibrator
from .inference.types import DecisionBackend, Embedder, ModelLoadError

HF_BASE = "https://huggingface.co"
PROFILES_DIR = Path(__file__).resolve().parent / "profiles"  # measured tuning per catalog model
REQUIRED_FILES = ("model.onnx", "tokenizer.json", "config.json")
LLM_REQUIRED_FILES = ("model.onnx", "tokenizer.json", "tokenizer_config.json")


def _required(kind: str) -> tuple[str, ...]:
    return LLM_REQUIRED_FILES if kind == "llm" else REQUIRED_FILES


def _backend_for(kind: str) -> str:
    return "onnx-llm" if kind == "llm" else "onnx-nli"


class UnknownModelError(ModelLoadError):
    """No model with this id is known to the registry."""


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    kind: str  # "nli" | "embedder"
    repo: str
    onnx_file: str | dict[str, str]  # a path, or {"arm64": path, "x86_64": path}
    quantization: str
    description: str
    size_mb: int  # download size of the ONNX file (Hugging Face listing)
    # "basic" (< 200 MB) | "large" (0.5-1 GB) | "experimental" (installable, not offered in menus)
    tier: str = "basic"
    label: str | None = None  # short name shown in model menus
    extra_files: tuple[str, ...] = ()  # e.g. external weights; saved under their own names
    profile: dict[str, Any] = field(default_factory=dict)

    def onnx_path(self) -> str:
        if isinstance(self.onnx_file, str):
            return self.onnx_file
        arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x86_64"
        return self.onnx_file[arch]


_ARCH_INT8 = {"arm64": "onnx/model_qint8_arm64.onnx", "x86_64": "onnx/model_quint8_avx2.onnx"}

CATALOG: tuple[CatalogEntry, ...] = (
    # --- basic: small and fast
    CatalogEntry(
        "nli-deberta-v3-xsmall-int8",
        "nli",
        "Xenova/nli-deberta-v3-xsmall",
        "onnx/model_quantized.onnx",
        "INT8",
        "DeBERTa-v3-xsmall NLI (3-class), default",
        87,
    ),
    CatalogEntry(
        "nli-deberta-v3-xsmall-fp16",
        "nli",
        "Xenova/nli-deberta-v3-xsmall",
        "onnx/model_fp16.onnx",
        "FP16",
        "DeBERTa-v3-xsmall NLI, FP16 reference",
        143,
    ),
    CatalogEntry(
        "nli-deberta-v3-xsmall-q4f16",
        "nli",
        "Xenova/nli-deberta-v3-xsmall",
        "onnx/model_q4f16.onnx",
        "INT4",
        "DeBERTa-v3-xsmall NLI, 4-bit weights",
        121,
    ),
    CatalogEntry(
        "nli-deberta-v3-small-int8",
        "nli",
        "Xenova/nli-deberta-v3-small",
        "onnx/model_quantized.onnx",
        "INT8",
        "DeBERTa-v3-small NLI",
        172,
    ),
    CatalogEntry(
        "nli-minilm2-l6-int8",
        "nli",
        "cross-encoder/nli-MiniLM2-L6-H768",
        {"arm64": "onnx/model_qint8_arm64.onnx", "x86_64": "onnx/model_quint8_avx2.onnx"},
        "INT8",
        "MiniLM2-L6-H768 NLI cross-encoder",
        83,
    ),
    CatalogEntry(
        "nli-mobilebert-int8",
        "nli",
        "Xenova/mobilebert-uncased-mnli",
        "onnx/model_quantized.onnx",
        "INT8",
        "MobileBERT MNLI (very small)",
        26,
    ),
    CatalogEntry(
        "nli-distilbert-int8",
        "nli",
        "Xenova/distilbert-base-uncased-mnli",
        "onnx/model_quantized.onnx",
        "INT8",
        "DistilBERT MNLI",
        68,
    ),
    CatalogEntry(
        "zeroshot-xtremedistil-int8",
        "nli",
        "MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33",
        "onnx/model_quantized.onnx",
        "INT8",
        "xtremedistil zero-shot (2-class, tiny)",
        13,
    ),
    CatalogEntry(
        "zeroshot-deberta-v3-xsmall-int8",
        "nli",
        "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
        "onnx/model_quantized.onnx",
        "INT8",
        "DeBERTa-v3-xsmall zero-shot (2-class)",
        87,
    ),
    # --- large: 0.5-1 GB downloads, more accurate, more RAM
    CatalogEntry(
        "nli-deberta-v3-large-int8",
        "nli",
        "cross-encoder/nli-deberta-v3-large",
        _ARCH_INT8,
        "INT8",
        "DeBERTa-v3-large NLI, INT8 (quantisation hurts it: measured below xsmall)",
        643,
        tier="experimental",
    ),
    CatalogEntry(
        "nli-deberta-v3-large-anli-int8",
        "nli",
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "onnx/model_quantized.onnx",
        "INT8",
        "DeBERTa-v3-large MNLI/FEVER/ANLI/WANLI, INT8 (quantisation hurts it)",
        643,
        tier="experimental",
    ),
    CatalogEntry(
        "zeroshot-deberta-v3-base-fp32",
        "nli",
        "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        "onnx/model.onnx",
        "FP32",
        "DeBERTa-v3-base zero-shot v2.0 (2-class, unquantised)",
        739,
        tier="large",
        label="Most accurate",
    ),
    CatalogEntry(
        "nli-deberta-v3-base-fp32",
        "nli",
        "cross-encoder/nli-deberta-v3-base",
        "onnx/model.onnx",
        "FP32",
        "DeBERTa-v3-base NLI cross-encoder (unquantised)",
        739,
        tier="large",
        label="Larger NLI",
    ),
    CatalogEntry(
        "nli-bart-large-fp16",
        "nli",
        "Xenova/bart-large-mnli",
        "onnx/model_fp16.onnx",
        "FP16",
        "BART-large MNLI (classic zero-shot model)",
        816,
        tier="large",
        label="BART (slow)",
    ),
    # --- llm: small instruction-tuned LLMs, 4-bit, scored by next-token probability
    CatalogEntry(
        "qwen3-0.6b-q4f16",
        "llm",
        "onnx-community/Qwen3-0.6B-ONNX",
        "onnx/model_q4f16.onnx",
        "INT4",
        "Qwen3 0.6B (Alibaba, Apache-2.0), 4-bit weights / FP16 activations",
        570,
        tier="llm",
        label="Qwen3 0.6B",
    ),
    CatalogEntry(
        "qwen2.5-0.5b-q4",
        "llm",
        "onnx-community/Qwen2.5-0.5B-Instruct",
        "onnx/model_q4.onnx",
        "INT4",
        "Qwen2.5 0.5B Instruct (Alibaba, Apache-2.0), 4-bit weights",
        786,
        tier="llm",
        label="Qwen2.5 0.5B",
    ),
    CatalogEntry(
        "gemma-3-1b-q4",
        "llm",
        "onnx-community/gemma-3-1b-it-ONNX",
        "onnx/model_q4.onnx",
        "INT4",
        "Gemma 3 1B instruct (Google, Gemma terms of use), 4-bit weights",
        859,
        tier="llm",
        label="Gemma 3 1B",
        extra_files=("onnx/model_q4.onnx_data",),
    ),
    CatalogEntry(
        "lfm2-1.2b-q4",
        "llm",
        "onnx-community/LFM2-1.2B-ONNX",
        "onnx/model_q4.onnx",
        "INT4",
        "LFM2 1.2B (Liquid AI, LFM Open License), 4-bit weights",
        850,
        tier="llm",
        label="LFM2 1.2B",
        extra_files=("onnx/model_q4.onnx_data",),
    ),
    # --- embedders
    CatalogEntry(
        "minilm-l6-v2-int8",
        "embedder",
        "Xenova/all-MiniLM-L6-v2",
        "onnx/model_quantized.onnx",
        "INT8",
        "all-MiniLM-L6-v2 sentence embedder (learning memory)",
        23,
    ),
)
_CATALOG_BY_ID = {e.id: e for e in CATALOG}


@dataclass(frozen=True)
class CuratedModel:
    id: str
    label: str
    summary: str


# Chosen from the measured comparison in docs/model-comparison.md.
BUNDLED_MODEL = "nli-deberta-v3-xsmall-int8"
CURATED_MODELS: tuple[CuratedModel, ...] = (
    CuratedModel(BUNDLED_MODEL, "Balanced ★", "best Noul accuracy; good Choice and Score; 83 MB"),
    CuratedModel(
        "nli-minilm2-l6-int8",
        "Fast & light",
        "about half the RAM, ~0.2 s cold start, 5 ms per call; 79 MB",
    ),
    CuratedModel(
        "zeroshot-deberta-v3-xsmall-int8",
        "Best at Choice",
        "highest Choice accuracy and best-calibrated Noul; 83 MB",
    ),
)


def _http_fetch(url: str, dest: Path, progress: Callable[[str], None] | None = None) -> None:
    import httpx

    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        done, next_mark = 0, 10
        with dest.open("wb") as fh:
            for chunk in response.iter_bytes(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                while progress and total and done * 100 >= next_mark * total and next_mark <= 100:
                    progress(f"  {dest.name}: {next_mark}% ({done / 1e6:.0f}/{total / 1e6:.0f} MB)")
                    next_mark += 10


def _present(path: Path) -> bool:
    """A real file: exists and isn't a Git LFS pointer left by a clone without LFS."""
    if not path.is_file():
        return False
    with path.open("rb") as fh:
        return not fh.read(40).startswith(b"version https://git-lfs")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"Invalid endpoint URL {base_url!r}; expected http(s)://host[:port]/v1")
    return base_url.rstrip("/")


class ModelRegistry:
    def __init__(
        self,
        models_dir: str | Path,
        models_file: str | Path | None = None,
        *,
        openai_base_url: str | None = None,
        openai_model: str | None = None,
        openai_api_key: str | None = None,
        threads: int | None = None,
        open_nli_model: Callable[..., Any] | None = None,
        open_llm: Callable[..., Any] | None = None,
        profiles_dir: str | Path = PROFILES_DIR,
        low_memory: bool = False,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
    ):
        from .inference.devices import DEVICES

        if device.lower() not in DEVICES:
            raise ValueError(f"Unknown device {device!r}; use one of: {', '.join(DEVICES)}.")
        self.models_dir = Path(models_dir)
        self.models_file = Path(models_file) if models_file else None
        self._threads = threads
        self._open_nli_model = open_nli_model
        self._open_llm = open_llm
        self._profiles_dir = Path(profiles_dir)
        self._low_memory = low_memory
        self._device = device.lower()
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._env_openai: dict[str, Any] | None = None
        if openai_base_url and openai_model:
            self._env_openai = {
                "id": "openai",
                "backend": "openai",
                "baseUrl": _check_base_url(openai_base_url),
                "model": openai_model,
                "apiKey": openai_api_key,
            }

    # ------------------------------------------------------------------ user entries

    def _user_file(self) -> dict[str, list[dict[str, Any]]]:
        if self.models_file is None or not self.models_file.is_file():
            return {"models": [], "embedders": []}
        data = json.loads(self.models_file.read_text() or "{}")
        return {
            "models": list(data.get("models", [])),
            "embedders": list(data.get("embedders", [])),
        }

    def _save_user_file(self, data: dict[str, list[dict[str, Any]]]) -> None:
        if self.models_file is None:
            raise ValueError("No models file configured; cannot persist model entries.")
        self.models_file.parent.mkdir(parents=True, exist_ok=True)
        self.models_file.write_text(json.dumps(data, indent=2) + "\n")
        os.chmod(self.models_file, 0o600)  # may hold API keys

    def _user_models(self) -> list[dict[str, Any]]:
        models = self._user_file()["models"]
        if self._env_openai and all(m.get("id") != "openai" for m in models):
            models.append(self._env_openai)
        return models

    def register_openai(
        self,
        *,
        id: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        if id in _CATALOG_BY_ID or any(m.get("id") == id for m in self._user_models()):
            raise ValueError(f"A model with id {id!r} already exists.")
        entry: dict[str, Any] = {
            "id": id,
            "backend": "openai",
            "baseUrl": _check_base_url(base_url),
            "model": model,
        }
        if api_key_env:
            entry["apiKeyEnv"] = api_key_env
        if api_key:
            entry["apiKey"] = api_key
        data = self._user_file()
        data["models"].append(entry)
        self._save_user_file(data)

    def register_onnx(
        self,
        *,
        id: str,
        path: str | Path,
        quantization: str = "unknown",
        description: str | None = None,
        kind: str = "nli",
    ) -> None:
        """Register your own ONNX model directory.

        kind="nli": model.onnx, tokenizer.json, config.json (with an entailment label).
        kind="llm": a causal LM export: model.onnx, tokenizer.json, tokenizer_config.json
        (with a chat template); external weight files next to model.onnx are fine.
        """
        if kind not in ("nli", "llm"):
            raise ValueError("kind must be 'nli' or 'llm'.")
        if id in _CATALOG_BY_ID or any(m.get("id") == id for m in self._user_models()):
            raise ValueError(f"A model with id {id!r} already exists.")
        model_dir = Path(path).expanduser().resolve()
        missing = [name for name in _required(kind) if not (model_dir / name).exists()]
        if missing:
            raise ValueError(f"Model directory is missing: {', '.join(missing)}.")
        entry: dict[str, Any] = {
            "id": id,
            "backend": _backend_for(kind),
            "path": str(model_dir),
            "quantization": quantization,
        }
        if description:
            entry["description"] = description
        data = self._user_file()
        data["models"].append(entry)
        self._save_user_file(data)

    def is_installed(self, model_id: str) -> bool:
        if model_id in _CATALOG_BY_ID:
            return self._installed(model_id)
        return any(m["id"] == model_id and m["installed"] for m in self.list())

    def is_downloadable(self, model_id: str) -> bool:
        return model_id in _CATALOG_BY_ID

    def unregister(self, model_id: str) -> None:
        data = self._user_file()
        remaining = [m for m in data["models"] if m.get("id") != model_id]
        if len(remaining) == len(data["models"]):
            raise UnknownModelError(f"No user-defined model {model_id!r}.")
        data["models"] = remaining
        self._save_user_file(data)

    # ------------------------------------------------------------------ listing

    def _installed(self, model_id: str) -> bool:
        entry = _CATALOG_BY_ID.get(model_id)
        kind = entry.kind if entry else "nli"
        extras = tuple(Path(f).name for f in entry.extra_files) if entry else ()
        model_dir = self.models_dir / model_id
        return all(_present(model_dir / name) for name in (*_required(kind), *extras))

    def list(self) -> list[dict[str, Any]]:
        """Public listing (no filesystem paths, no secrets)."""
        listing = [
            {
                "id": e.id,
                "backend": _backend_for(e.kind),
                "model": e.repo,
                "quantization": e.quantization,
                "local": True,
                "installed": self._installed(e.id),
                "description": e.description,
                "tier": e.tier,
                "label": e.label,
                "sizeMB": e.size_mb,
                **self._measured(e.id),
            }
            for e in CATALOG
            if e.kind in ("nli", "llm")
        ]
        for m in self._user_models():
            if m.get("backend") == "openai":
                listing.append(
                    {
                        "id": m["id"],
                        "backend": "openai",
                        "model": m["model"],
                        "quantization": "n/a",
                        "local": is_loopback(urlparse(m["baseUrl"]).hostname or ""),
                        "installed": True,
                        "description": m.get("description", "OpenAI-compatible endpoint"),
                    }
                )
            elif m.get("backend") in ("onnx-nli", "onnx-llm"):
                kind = "llm" if m["backend"] == "onnx-llm" else "nli"
                listing.append(
                    {
                        "id": m["id"],
                        "backend": m["backend"],
                        "model": m.get("model", m["id"]),
                        "quantization": m.get("quantization", "unknown"),
                        "local": True,
                        "installed": self._installed_path(Path(m["path"]), kind),
                        "description": m.get("description", f"Custom ONNX {kind.upper()} model"),
                    }
                )
        return listing

    def _measured(self, model_id: str) -> dict[str, Any]:
        """Benchmark results shipped for a model (RAM is measured, never estimated)."""
        data: dict[str, Any] = {}
        for source in (
            self._profiles_dir / model_id / "measured.json",
            self.models_dir / model_id / "measured.json",
        ):
            if source.exists():
                data.update(json.loads(source.read_text()))
        peak = data.get("peakRssBytes")
        return {
            "ramMB": round(peak / (1024 * 1024)) if peak else None,
            "noulAccuracy": data.get("noulAccuracy"),
            "choiceAccuracy": data.get("choiceAccuracy"),
            "scoreMae": data.get("scoreMae"),
            "p50Ms": data.get("p50Ms"),
        }

    @staticmethod
    def _installed_path(path: Path, kind: str = "nli") -> bool:
        return all(_present(path / name) for name in _required(kind))

    # ------------------------------------------------------------------ loading

    def _resolve_api_key(self, entry: dict[str, Any]) -> str | None:
        if entry.get("apiKeyEnv"):
            return os.environ.get(entry["apiKeyEnv"])
        return entry.get("apiKey")

    def _nli_backend(
        self, model_id: str, model_dir: Path, quantization: str, profile: dict[str, Any]
    ) -> DecisionBackend:
        from .inference.nli_backend import NliBackend

        if not self._installed_path(model_dir):
            hint = (
                f"run: noulo models download {model_id}"
                if model_id in _CATALOG_BY_ID
                else "check its files"
            )
            raise ModelLoadError(f"Model {model_id!r} is not installed; {hint}.")
        for profile_file in (
            self._profiles_dir / model_id / "profile.json",
            model_dir / "profile.json",
        ):  # packaged, then local override
            if profile_file.exists():
                profile = {**profile, **json.loads(profile_file.read_text())}
        opener = self._open_nli_model
        if opener is None:
            from .inference.model import OnnxNliModel

            opener = OnnxNliModel
        model = opener(
            model_dir,
            threads=self._threads,
            low_memory=self._low_memory,
            placement=self._placement(quantization),
        )
        return NliBackend(model, model_id=model_id, quantization=quantization, **profile)

    def _llm_backend(self, model_id: str, model_dir: Path, quantization: str) -> DecisionBackend:
        from .inference.llm_backend import LlmBackend

        installed = (
            self._installed(model_id)
            if model_id in _CATALOG_BY_ID
            else self._installed_path(model_dir, "llm")
        )
        if not installed:
            hint = (
                f"run: noulo model download {model_id}"
                if model_id in _CATALOG_BY_ID
                else "check its files"
            )
            raise ModelLoadError(f"Model {model_id!r} is not installed; {hint}.")
        opener = self._open_llm
        if opener is None:
            from .inference.causal_lm import OnnxCausalLM

            opener = OnnxCausalLM
        model = opener(
            model_dir,
            threads=self._threads,
            low_memory=self._low_memory,
            placement=self._placement(quantization),
        )
        return LlmBackend(model, model_id=model_id, quantization=quantization)

    def _placement(self, precision: str) -> Any:
        import onnxruntime as ort

        from .inference.devices import resolve

        return resolve(
            self._device,
            available=ort.get_available_providers(),
            precision=precision,
            cache_dir=self._cache_dir,
        )

    def load_backend(self, model_id: str) -> DecisionBackend:
        entry = _CATALOG_BY_ID.get(model_id)
        if entry is not None and entry.kind == "nli":
            return self._nli_backend(
                model_id, self.models_dir / model_id, entry.quantization, dict(entry.profile)
            )
        if entry is not None and entry.kind == "llm":
            return self._llm_backend(model_id, self.models_dir / model_id, entry.quantization)
        for m in self._user_models():
            if m.get("id") != model_id:
                continue
            if m.get("backend") == "openai":
                from .inference.openai_backend import OpenAIBackend

                return OpenAIBackend(
                    id=model_id,
                    base_url=m["baseUrl"],
                    model=m["model"],
                    api_key=self._resolve_api_key(m),
                    timeout=float(m.get("timeout", 30.0)),
                    extra_body=m.get("extraBody"),
                )
            if m.get("backend") == "onnx-llm":
                return self._llm_backend(
                    model_id, Path(m["path"]), m.get("quantization", "unknown")
                )
            if m.get("backend") == "onnx-nli":
                return self._nli_backend(
                    model_id,
                    Path(m["path"]),
                    m.get("quantization", "unknown"),
                    dict(m.get("profile", {})),
                )
        raise UnknownModelError(f"Unknown model {model_id!r}.")

    def load_calibrator(self, model_id: str) -> Calibrator:
        local = self.models_dir / model_id / "calibration.json"
        return load_calibrator(
            local if local.exists() else self._profiles_dir / model_id / "calibration.json"
        )

    def load_embedder(self, embedder_id: str) -> Embedder:
        from .inference.embedder import HashingEmbedder, OnnxEmbedder, OpenAIEmbedder

        if embedder_id == "hashing":
            return HashingEmbedder()
        entry = _CATALOG_BY_ID.get(embedder_id)
        if entry is not None and entry.kind == "embedder":
            model_dir = self.models_dir / embedder_id
            if not (model_dir / "model.onnx").exists():
                raise ModelLoadError(
                    f"Embedder {embedder_id!r} is not installed; "
                    f"run: noulo models download {embedder_id}."
                )
            return OnnxEmbedder(model_dir, id=embedder_id, threads=self._threads)
        for e in self._user_file()["embedders"]:
            if e.get("id") == embedder_id and e.get("backend") == "openai":
                return OpenAIEmbedder(
                    id=embedder_id,
                    base_url=_check_base_url(e["baseUrl"]),
                    model=e["model"],
                    api_key=self._resolve_api_key(e),
                )
        raise UnknownModelError(f"Unknown embedder {embedder_id!r}.")

    # ------------------------------------------------------------------ download

    def download(
        self,
        model_id: str,
        *,
        fetch: Callable[[str, Path], None] = _http_fetch,
        progress: Callable[[str], None] | None = None,
    ) -> Path:
        """Download a catalog model into models_dir/<id>/ (atomic)."""
        entry = _CATALOG_BY_ID.get(model_id)
        if entry is None:
            raise UnknownModelError(f"{model_id!r} is not a downloadable catalog model.")
        target = self.models_dir / model_id
        self.models_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{model_id}-", dir=self.models_dir))
        try:
            sources = {
                "model.onnx": entry.onnx_path(),
                "tokenizer.json": "tokenizer.json",
                "config.json": "config.json",
            }
            if entry.kind == "llm":
                sources["tokenizer_config.json"] = "tokenizer_config.json"
            for remote in entry.extra_files:
                sources[Path(remote).name] = remote
            for local_name, remote_path in sources.items():
                if progress:
                    progress(f"Downloading {entry.repo}/{remote_path}")
                url = f"{HF_BASE}/{entry.repo}/resolve/main/{remote_path}"
                if fetch is _http_fetch:
                    fetch(url, staging / local_name, progress)
                else:
                    fetch(url, staging / local_name)
            weights = ["model.onnx", *(Path(f).name for f in entry.extra_files)]
            manifest = {
                "id": entry.id,
                "kind": entry.kind,
                "source": f"{HF_BASE}/{entry.repo}",
                "file": entry.onnx_path(),
                "quantization": entry.quantization,
                "sha256": _sha256(staging / "model.onnx"),
                "sizeBytes": sum((staging / name).stat().st_size for name in weights),
            }
            if entry.extra_files:
                manifest["extraFiles"] = {name: _sha256(staging / name) for name in weights[1:]}
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if target.exists():
                for keep in ("calibration.json", "profile.json"):
                    if (target / keep).exists():
                        shutil.copy2(target / keep, staging / keep)
                shutil.rmtree(target)
            staging.rename(target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return target
