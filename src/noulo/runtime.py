"""Wiring: build the registry, learning memory and engine from Settings."""

from __future__ import annotations

import re

from .config import Settings
from .inference.engine import DecisionEngine
from .inference.memory import LearningMemory, MemoryConfig
from .inference.stores import open_store
from .registry import ModelRegistry


def build_registry(settings: Settings) -> ModelRegistry:
    return ModelRegistry(
        settings.models_dir,
        settings.models_file,
        openai_base_url=settings.openai_base_url,
        openai_model=settings.openai_model,
        openai_api_key=settings.openai_api_key.get_secret_value()
        if settings.openai_api_key
        else None,
        threads=settings.threads,
    )


def build_memory(settings: Settings, registry: ModelRegistry) -> LearningMemory:
    """Open the configured vector store (sqlite/qdrant/chroma/custom) with its embedder.

    The collection name is suffixed with the embedder id so vectors of
    different widths never share a collection.
    """
    embedder = registry.load_embedder(settings.embedder)
    suffix = re.sub(r"[^A-Za-z0-9_]+", "_", embedder.id).strip("_")
    store = open_store(
        settings.memory_store,
        location=settings.memory_location,
        collection=f"{settings.memory_collection}_{suffix}",
        api_key=settings.memory_api_key.get_secret_value() if settings.memory_api_key else None,
        dim=embedder.dim,
    )
    config = MemoryConfig(
        top_k=settings.memory_top_k,
        min_similarity=settings.memory_min_similarity,
        feedback_weight=settings.memory_feedback_weight,
        observed_weight=settings.memory_observed_weight,
        max_influence=settings.memory_max_influence,
        prior_strength=settings.memory_prior_strength,
    )
    return LearningMemory(store, embedder, config)


def build_engine(settings: Settings, registry: ModelRegistry) -> DecisionEngine:
    return DecisionEngine(
        load_backend=registry.load_backend,
        model_id=settings.model,
        load_calibrator=registry.load_calibrator,
        open_memory=lambda: build_memory(settings, registry),
        learning=settings.learning_enabled,
        max_concurrency=settings.max_concurrency,
        max_queue=settings.max_queue,
    )
