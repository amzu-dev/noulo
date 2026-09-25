"""Runtime configuration from environment variables (prefix `NOULO_`) and `.env`.

Secure defaults: bind to loopback only, CORS limited to localhost origins,
no wildcard origins, secrets never printed.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path

from pydantic import SecretStr
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
    memory_observed_weight: float = 0.25
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
