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
    profile: dict[str, Any] = field(default_factory=dict)

    def onnx_path(self) -> str:
        if isinstance(self.onnx_file, str):
            return self.onnx_file
        arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x86_64"
        return self.onnx_file[arch]


CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry(
        "nli-deberta-v3-xsmall-int8",
        "nli",
        "Xenova/nli-deberta-v3-xsmall",
        "onnx/model_quantized.onnx",
        "INT8",
        "DeBERTa-v3-xsmall NLI (3-class), default",
    ),
    CatalogEntry(
        "nli-deberta-v3-xsmall-fp16",
        "nli",
        "Xenova/nli-deberta-v3-xsmall",
        "onnx/model_fp16.onnx",
        "FP16",
        "DeBERTa-v3-xsmall NLI, FP16 reference",
    ),
    CatalogEntry(
        "nli-deberta-v3-xsmall-q4f16",
        "nli",
        "Xenova/nli-deberta-v3-xsmall",
        "onnx/model_q4f16.onnx",
        "INT4",
        "DeBERTa-v3-xsmall NLI, 4-bit weights",
    ),
    CatalogEntry(
        "nli-deberta-v3-small-int8",
        "nli",
        "Xenova/nli-deberta-v3-small",
        "onnx/model_quantized.onnx",
        "INT8",
        "DeBERTa-v3-small NLI (larger, more accurate)",
    ),
    CatalogEntry(
        "nli-minilm2-l6-int8",
        "nli",
        "cross-encoder/nli-MiniLM2-L6-H768",
        {"arm64": "onnx/model_qint8_arm64.onnx", "x86_64": "onnx/model_quint8_avx2.onnx"},
        "INT8",
        "MiniLM2-L6-H768 NLI cross-encoder",
    ),
    CatalogEntry(
        "nli-mobilebert-int8",
        "nli",
        "Xenova/mobilebert-uncased-mnli",
        "onnx/model_quantized.onnx",
        "INT8",
        "MobileBERT MNLI (very small)",
    ),
    CatalogEntry(
        "nli-distilbert-int8",
        "nli",
        "Xenova/distilbert-base-uncased-mnli",
        "onnx/model_quantized.onnx",
        "INT8",
        "DistilBERT MNLI",
    ),
    CatalogEntry(
        "zeroshot-xtremedistil-int8",
        "nli",
        "MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33",
        "onnx/model_quantized.onnx",
        "INT8",
        "xtremedistil zero-shot (2-class, tiny)",
    ),
    CatalogEntry(
        "zeroshot-deberta-v3-xsmall-int8",
        "nli",
        "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
        "onnx/model_quantized.onnx",
        "INT8",
        "DeBERTa-v3-xsmall zero-shot (2-class)",
    ),
    CatalogEntry(
        "minilm-l6-v2-int8",
        "embedder",
        "Xenova/all-MiniLM-L6-v2",
        "onnx/model_quantized.onnx",
        "INT8",
        "all-MiniLM-L6-v2 sentence embedder (learning memory)",
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
    CuratedModel(
        BUNDLED_MODEL, "Balanced (bundled)", "best Noul accuracy; good Choice and Score; 83 MB"
    ),
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


def _http_fetch(url: str, dest: Path) -> None:
    import httpx

    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
        response.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in response.iter_bytes(1 << 20):
                fh.write(chunk)


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
        profiles_dir: str | Path = PROFILES_DIR,
    ):
        self.models_dir = Path(models_dir)
        self.models_file = Path(models_file) if models_file else None
        self._threads = threads
        self._open_nli_model = open_nli_model
        self._profiles_dir = Path(profiles_dir)
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
        if self.models_file is None or not self.models_file.exists():
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
    ) -> None:
        """Register your own ONNX NLI model directory (model.onnx, tokenizer.json, config.json)."""
        if id in _CATALOG_BY_ID or any(m.get("id") == id for m in self._user_models()):
            raise ValueError(f"A model with id {id!r} already exists.")
        model_dir = Path(path).expanduser().resolve()
        missing = [name for name in REQUIRED_FILES if not (model_dir / name).exists()]
        if missing:
            raise ValueError(f"Model directory is missing: {', '.join(missing)}.")
        entry: dict[str, Any] = {
            "id": id,
            "backend": "onnx-nli",
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
        return all((self.models_dir / model_id / name).exists() for name in REQUIRED_FILES)

    def list(self) -> list[dict[str, Any]]:
        """Public listing (no filesystem paths, no secrets)."""
        listing = [
            {
                "id": e.id,
                "backend": "onnx-nli",
                "model": e.repo,
                "quantization": e.quantization,
                "local": True,
                "installed": self._installed(e.id),
                "description": e.description,
            }
            for e in CATALOG
            if e.kind == "nli"
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
            elif m.get("backend") == "onnx-nli":
                listing.append(
                    {
                        "id": m["id"],
                        "backend": "onnx-nli",
                        "model": m.get("model", m["id"]),
                        "quantization": m.get("quantization", "unknown"),
                        "local": True,
                        "installed": self._installed_path(Path(m["path"])),
                        "description": m.get("description", "Custom ONNX NLI model"),
                    }
                )
        return listing

    @staticmethod
    def _installed_path(path: Path) -> bool:
        return all((path / name).exists() for name in REQUIRED_FILES)

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
        model = opener(model_dir, threads=self._threads)
        return NliBackend(model, model_id=model_id, quantization=quantization, **profile)

    def load_backend(self, model_id: str) -> DecisionBackend:
        entry = _CATALOG_BY_ID.get(model_id)
        if entry is not None and entry.kind == "nli":
            return self._nli_backend(
                model_id, self.models_dir / model_id, entry.quantization, dict(entry.profile)
            )
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
            for local_name, remote_path in sources.items():
                if progress:
                    progress(f"Downloading {entry.repo}/{remote_path}")
                fetch(f"{HF_BASE}/{entry.repo}/resolve/main/{remote_path}", staging / local_name)
            onnx_bytes = (staging / "model.onnx").read_bytes()
            manifest = {
                "id": entry.id,
                "kind": entry.kind,
                "source": f"{HF_BASE}/{entry.repo}",
                "file": entry.onnx_path(),
                "quantization": entry.quantization,
                "sha256": hashlib.sha256(onnx_bytes).hexdigest(),
                "sizeBytes": len(onnx_bytes),
            }
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
