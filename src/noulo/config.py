"""Runtime configuration from environment variables (prefix `NOULO_`) and `.env`.

Secure defaults: bind to loopback only, CORS limited to localhost origins,
no wildcard origins, secrets never printed.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .api.validation import Limits

LOCALHOST_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
SECRET_FIELDS = ("api_key", "openai_api_key", "memory_api_key")


class ConfigError(ValueError):
    """Configuration is invalid or unsafe."""


def is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NOULO_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- server
    host: str = "127.0.0.1"
    port: int = 8787
    allow_network: bool = False
    api_key: SecretStr | None = None
    cors_enabled: bool = True
    cors_origins: str = ""
    cors_allow_localhost: bool = True
    ui_enabled: bool = True
    open_browser: bool = False
    run_dir: Path = Path("data/run")  # pidfile + log of the background service
    log_level: str = "info"

    # --- models
    models_dir: Path = Path("models")
    models_file: Path | None = Path("models.json")
    model: str = "nli-deberta-v3-xsmall-int8"
    threads: int | None = None
    low_memory: bool = False
    device: str = "auto"
    openai_base_url: str | None = None
    openai_model: str | None = None
    openai_api_key: SecretStr | None = None

    # --- execution
    max_concurrency: int = 1
    max_queue: int = 32
    shutdown_timeout: float = 10.0

    # --- request limits
    max_input_chars: int = 5000
    max_text_chars: int = 1000
    max_choices: int = 20
    max_rubric_levels: int = 11
    max_body_bytes: int = 65536

    # --- learning memory
    learning_enabled: bool = True
    embedder: str = "minilm-l6-v2-int8"
    memory_store: str = "sqlite"
    memory_location: str | None = "data/memory.sqlite3"
    memory_collection: str = "noulo_memory"
    memory_api_key: SecretStr | None = None
    memory_top_k: int = 8
    memory_min_similarity: float = 0.80
    memory_feedback_weight: float = 1.0
    memory_observed_weight: float = 0.0
    memory_max_influence: float = 0.9
    memory_prior_strength: float = 0.5

    @property
    def base_url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def cors_origin_regex(self) -> str | None:
        return LOCALHOST_ORIGIN_REGEX if self.cors_allow_localhost else None

    @property
    def limits(self) -> Limits:
        return Limits(
            max_input_chars=self.max_input_chars,
            max_text_chars=self.max_text_chars,
            max_choices=self.max_choices,
            max_rubric_levels=self.max_rubric_levels,
        )

    @field_validator("models_file", mode="before")
    @classmethod
    def _empty_means_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    def check(self) -> Settings:
        if not is_loopback(self.host) and not self.allow_network:
            raise ConfigError(
                f"Refusing to bind to non-loopback host {self.host!r}. "
                "Set NOULO_ALLOW_NETWORK=true to expose the API on the network."
            )
        if "*" in self.cors_origin_list:
            raise ConfigError("Wildcard CORS origins are not allowed; list origins explicitly.")
        return self

    def public_dict(self) -> dict:
        data = self.model_dump(mode="json")
        for name in SECRET_FIELDS:
            if data.get(name) is not None:
                data[name] = "***"
        return data


SETTING_HELP: dict[str, str] = {
    "host": "Bind address; non-loopback hosts also need allow_network",
    "port": "HTTP port of the API and frontend",
    "allow_network": "Allow binding a non-loopback (LAN/public) interface",
    "api_key": "Require 'Authorization: Bearer <key>' on /api/v1/*",
    "cors_enabled": "Send CORS headers at all",
    "cors_origins": "Extra allowed browser origins, comma-separated (never *)",
    "cors_allow_localhost": "Allow localhost/127.0.0.1 origins on any port",
    "ui_enabled": "Serve the frontend at /ui/",
    "open_browser": "Open the frontend in a browser when the server starts",
    "run_dir": "Where the background service keeps its pidfile and log",
    "log_level": "Server log level (critical, error, warning, info, debug)",
    "models_dir": "Directory holding local models",
    "models_file": "JSON file with your endpoints and custom models",
    "model": "Model loaded at startup (see /model)",
    "threads": "ONNX Runtime intra-op threads (empty = automatic)",
    "low_memory": "Trade some speed for ~25% less RAM (skips ONNX constant folding)",
    "device": "Where models run: auto, cpu, gpu, coreml, cuda, directml or rocm",
    "openai_base_url": "Shortcut OpenAI-compatible endpoint registered as model 'openai'",
    "openai_model": "Model name for the 'openai' shortcut endpoint",
    "openai_api_key": "API key for the 'openai' shortcut endpoint",
    "max_concurrency": "Inferences allowed to run at the same time",
    "max_queue": "Requests allowed to wait before 503 ENGINE_BUSY",
    "shutdown_timeout": "Seconds to finish in-flight requests on stop",
    "max_input_chars": "Maximum characters in 'input'",
    "max_text_chars": "Maximum characters in a proposition, question, option or level",
    "max_choices": "Maximum number of Choice options",
    "max_rubric_levels": "Maximum number of Score rubric levels",
    "max_body_bytes": "Maximum request body size in bytes",
    "learning_enabled": "Remember cases and let feedback adjust future answers",
    "embedder": "Sentence embedder used by the learning memory",
    "memory_store": "Vector store: sqlite, qdrant, chroma or module:Class",
    "memory_location": "File/dir for local stores or http(s):// URL for remote ones",
    "memory_collection": "Collection name (the embedder id is appended)",
    "memory_api_key": "API key for a remote vector database",
    "memory_top_k": "Past cases considered per request",
    "memory_min_similarity": "Similarity a past input needs to count (0-1)",
    "memory_feedback_weight": "Weight of cases confirmed by feedback",
    "memory_observed_weight": "Weight of past unverified answers (0 = feedback only)",
    "memory_max_influence": "Upper bound on how far memory can move a result (0-1)",
    "memory_prior_strength": "Evidence needed before memory has much influence",
}

# Settings the running server can apply without a restart (via the API).
LIVE_SETTINGS = ("model", "learning_enabled")
