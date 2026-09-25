from noulo.config import Settings
from noulo.inference.memory import LearningMemory
from noulo.runtime import build_engine, build_memory, build_registry
from tests.fakes import FakeBackend


class RecordingStore:
    """Custom store loaded via 'module:Class' — records how it was constructed."""

    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        RecordingStore.instances.append(self)

    def close(self):
        pass


def settings(tmp_path, **overrides):
    base = {
        "models_dir": tmp_path / "models",
        "models_file": None,
        "embedder": "hashing",
        "memory_location": str(tmp_path / "memory.sqlite3"),
    }
    return Settings(_env_file=None, **{**base, **overrides})


def test_sqlite_memory_is_the_default(tmp_path):
    memory = build_memory(settings(tmp_path), build_registry(settings(tmp_path)))
    assert isinstance(memory, LearningMemory)
    assert type(memory.store).__name__ == "SqliteVectorStore"
    memory.close()


def test_memory_uses_configured_blend_weights(tmp_path):
    cfg = settings(tmp_path, memory_top_k=3, memory_min_similarity=0.5, memory_max_influence=0.4)
    memory = build_memory(cfg, build_registry(cfg))
    assert (memory.config.top_k, memory.config.min_similarity, memory.config.max_influence) == (
        3,
        0.5,
        0.4,
    )
    memory.close()


def test_qdrant_local_store_can_be_selected(tmp_path):
    cfg = settings(tmp_path, memory_store="qdrant", memory_location=str(tmp_path / "qdrant"))
    memory = build_memory(cfg, build_registry(cfg))
    assert type(memory.store).__name__ == "QdrantVectorStore"
    memory.close()


def test_custom_store_receives_location_collection_and_key(tmp_path):
    cfg = settings(
        tmp_path,
        memory_store="tests.test_runtime:RecordingStore",
        memory_location="https://vectors.example",
        memory_api_key="k",
    )
    build_memory(cfg, build_registry(cfg))
    kwargs = RecordingStore.instances[-1].kwargs
    assert kwargs["location"] == "https://vectors.example" and kwargs["api_key"] == "k"
    assert kwargs["collection"] == "noulo_memory_hashing_256"


def test_engine_learning_follows_settings(tmp_path, monkeypatch):
    cfg = settings(tmp_path, learning_enabled=False)
    registry = build_registry(cfg)
    monkeypatch.setattr(registry, "load_backend", lambda _id: FakeBackend())
    engine = build_engine(cfg, registry)
    engine.start()
    assert engine.learning_enabled is False and engine.memory is None
    engine.set_learning(True)
    assert isinstance(engine.memory, LearningMemory)
    engine.shutdown()


def test_engine_passes_concurrency_settings(tmp_path):
    cfg = settings(tmp_path, max_concurrency=3, max_queue=5)
    engine = build_engine(cfg, build_registry(cfg))
    assert engine._max_concurrency == 3 and engine._max_pending == 8
