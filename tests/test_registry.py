import hashlib
import json

import numpy as np
import pytest

from noulo.inference.calibration import IdentityCalibrator, PlattCalibrator
from noulo.inference.nli_backend import NliBackend
from noulo.inference.openai_backend import OpenAIBackend
from noulo.inference.types import ModelLoadError
from noulo.registry import CATALOG, ModelRegistry, UnknownModelError


class RecordingNliModel:
    labels = {"entailment": 0, "neutral": 1, "contradiction": 2}

    def __init__(self):
        self.pairs = []

    def predict_logits(self, pairs):
        self.pairs.extend(pairs)
        return np.zeros((len(pairs), 3), dtype=np.float32)

    def close(self):
        pass


def install_fake(models_dir, model_id, **extra_files):
    d = models_dir / model_id
    d.mkdir(parents=True)
    for name in ("model.onnx", "tokenizer.json", "config.json"):
        (d / name).write_text("{}")
    for name, content in extra_files.items():
        (d / name).write_text(json.dumps(content))
    return d


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(
        tmp_path / "models",
        tmp_path / "models.json",
        open_nli_model=lambda *a, **k: RecordingNliModel(),
        profiles_dir=tmp_path / "no-profiles",
    )


# ---------------------------------------------------------------- catalog / listing


def test_catalog_contains_default_model_and_embedder():
    ids = {e.id for e in CATALOG}
    assert {"nli-deberta-v3-xsmall-int8", "minilm-l6-v2-int8"} <= ids


def test_listing_marks_installed_models(registry, tmp_path):
    install_fake(tmp_path / "models", "nli-mobilebert-int8")
    listing = {m["id"]: m for m in registry.list()}
    assert listing["nli-mobilebert-int8"]["installed"] is True
    assert listing["nli-deberta-v3-xsmall-int8"]["installed"] is False


def test_listing_never_exposes_paths_or_secrets(registry, tmp_path, monkeypatch):
    monkeypatch.setenv("MY_KEY", "sk-secret-value")
    registry.register_openai(
        id="remote", base_url="https://api.example.com/v1", model="m", api_key_env="MY_KEY"
    )
    text = json.dumps(registry.list())
    assert str(tmp_path) not in text
    assert "sk-secret-value" not in text and "MY_KEY" not in text


def test_listing_only_includes_decision_models_not_embedders(registry):
    assert all(m["backend"] in ("onnx-nli", "onnx-llm", "openai") for m in registry.list())


# ---------------------------------------------------------------- openai endpoints


def test_models_file_declares_openai_compatible_endpoints(tmp_path):
    models_file = tmp_path / "models.json"
    models_file.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "ollama-qwen",
                        "backend": "openai",
                        "baseUrl": "http://127.0.0.1:11434/v1",
                        "model": "qwen2.5:0.5b",
                    },
                ]
            }
        )
    )
    registry = ModelRegistry(tmp_path / "models", models_file)
    entry = {m["id"]: m for m in registry.list()}["ollama-qwen"]
    assert entry["backend"] == "openai" and entry["local"] is True and entry["installed"] is True


def test_env_configured_openai_endpoint_is_registered_as_openai(tmp_path):
    registry = ModelRegistry(
        tmp_path / "models",
        None,
        openai_base_url="https://api.openai.com/v1",
        openai_model="gpt-4o-mini",
        openai_api_key="sk-x",
    )
    entry = {m["id"]: m for m in registry.list()}["openai"]
    assert entry["model"] == "gpt-4o-mini" and entry["local"] is False


def test_load_openai_backend_resolves_api_key_from_env(registry, monkeypatch):
    monkeypatch.setenv("MY_KEY", "sk-abc")
    registry.register_openai(
        id="remote", base_url="https://api.example.com/v1", model="m", api_key_env="MY_KEY"
    )
    backend = registry.load_backend("remote")
    assert isinstance(backend, OpenAIBackend)
    assert backend.info.id == "remote" and backend.info.local is False
    backend.close()


def test_register_openai_persists_to_models_file(tmp_path):
    models_file = tmp_path / "models.json"
    ModelRegistry(tmp_path / "models", models_file).register_openai(
        id="lmstudio", base_url="http://localhost:1234/v1", model="phi"
    )
    reloaded = ModelRegistry(tmp_path / "models", models_file)
    assert "lmstudio" in {m["id"] for m in reloaded.list()}


def test_register_rejects_duplicate_ids_and_bad_urls(registry):
    with pytest.raises(ValueError):
        registry.register_openai(
            id="nli-deberta-v3-xsmall-int8", base_url="http://localhost:1/v1", model="m"
        )
    with pytest.raises(ValueError):
        registry.register_openai(id="bad", base_url="ftp://nope", model="m")


def test_unregister_removes_user_endpoint(registry):
    registry.register_openai(id="tmp", base_url="http://localhost:1/v1", model="m")
    registry.unregister("tmp")
    assert "tmp" not in {m["id"] for m in registry.list()}


# ---------------------------------------------------------------- loading


def test_unknown_model_raises_unknown_model_error(registry):
    with pytest.raises(UnknownModelError):
        registry.load_backend("does-not-exist")


def test_catalog_model_not_installed_explains_how_to_download(registry):
    with pytest.raises(ModelLoadError, match="noulo models download nli-mobilebert-int8"):
        registry.load_backend("nli-mobilebert-int8")


def test_load_installed_nli_model_builds_nli_backend(registry, tmp_path):
    install_fake(tmp_path / "models", "nli-mobilebert-int8")
    backend = registry.load_backend("nli-mobilebert-int8")
    assert isinstance(backend, NliBackend)
    assert backend.info.id == "nli-mobilebert-int8" and backend.info.quantization == "INT8"


def test_profile_file_overrides_candidate_strategy(tmp_path):
    model = RecordingNliModel()
    registry = ModelRegistry(tmp_path / "models", None, open_nli_model=lambda *a, **k: model)
    install_fake(
        tmp_path / "models",
        "nli-mobilebert-int8",
        **{"profile.json": {"score_template": "level-only"}},
    )
    registry.load_backend("nli-mobilebert-int8").score("input", "How bad?", ["low", "high"])
    assert [h for _, h in model.pairs] == ["low.", "high."]


def test_load_calibrator_reads_model_calibration_file(registry, tmp_path):
    install_fake(
        tmp_path / "models",
        "nli-mobilebert-int8",
        **{"calibration.json": {"method": "platt", "a": 1.0, "b": 0.5}},
    )
    assert isinstance(registry.load_calibrator("nli-mobilebert-int8"), PlattCalibrator)
    assert isinstance(registry.load_calibrator("nli-deberta-v3-xsmall-int8"), IdentityCalibrator)


# ---------------------------------------------------------------- embedders


def test_hashing_embedder_is_always_available(registry):
    embedder = registry.load_embedder("hashing")
    assert embedder.embed(["hello"]).shape[0] == 1


def test_unknown_embedder_raises(registry):
    with pytest.raises(UnknownModelError):
        registry.load_embedder("nope")


# ---------------------------------------------------------------- download


def fake_fetch(files):
    def fetch(url, dest):
        name = url.rsplit("/", 1)[-1]
        dest.write_bytes(files[name])

    return fetch


def test_download_writes_standard_layout_and_manifest(registry, tmp_path):
    files = {"model_quantized.onnx": b"onnx-bytes", "tokenizer.json": b"{}", "config.json": b"{}"}
    path = registry.download("nli-mobilebert-int8", fetch=fake_fetch(files))
    assert (path / "model.onnx").read_bytes() == b"onnx-bytes"
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["id"] == "nli-mobilebert-int8"
    assert manifest["quantization"] == "INT8"
    assert manifest["sha256"] == hashlib.sha256(b"onnx-bytes").hexdigest()
    assert manifest["sizeBytes"] == len(b"onnx-bytes")
    assert {m["id"]: m for m in registry.list()}["nli-mobilebert-int8"]["installed"]


def test_failed_download_leaves_no_partial_model(registry, tmp_path):
    def broken(url, dest):
        raise OSError("network down")

    with pytest.raises(OSError):
        registry.download("nli-mobilebert-int8", fetch=broken)
    assert not (tmp_path / "models" / "nli-mobilebert-int8").exists()


def test_download_unknown_or_remote_model_is_rejected(registry):
    with pytest.raises(UnknownModelError):
        registry.download("does-not-exist", fetch=fake_fetch({}))


# ---------------------------------------------------------------- your own ONNX model


def test_register_onnx_requires_standard_files(registry, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="model.onnx"):
        registry.register_onnx(id="mine", path=empty)


def test_register_onnx_lists_and_loads_custom_model(registry, tmp_path):
    custom = install_fake(tmp_path / "elsewhere", "mine")
    registry.register_onnx(id="mine", path=custom, quantization="INT8")
    entry = {m["id"]: m for m in registry.list()}["mine"]
    assert entry["installed"] and entry["quantization"] == "INT8"
    assert str(tmp_path) not in json.dumps(registry.list())  # path stays private
    assert registry.load_backend("mine").info.id == "mine"


def test_installed_and_downloadable_helpers(registry, tmp_path):
    install_fake(tmp_path / "models", "nli-mobilebert-int8")
    assert registry.is_installed("nli-mobilebert-int8")
    assert not registry.is_installed("nli-minilm2-l6-int8")
    assert registry.is_downloadable("nli-minilm2-l6-int8")
    assert not registry.is_downloadable("my-endpoint")


# ---------------------------------------------------------------- packaged tuning


@pytest.fixture
def profiles(tmp_path):
    d = tmp_path / "profiles" / "nli-mobilebert-int8"
    d.mkdir(parents=True)
    (d / "profile.json").write_text(json.dumps({"score_template": "level-only"}))
    (d / "calibration.json").write_text(json.dumps({"method": "platt", "a": 2.0, "b": 0.0}))
    return tmp_path / "profiles"


def test_packaged_profile_applies_when_model_dir_has_none(tmp_path, profiles):
    model = RecordingNliModel()
    registry = ModelRegistry(
        tmp_path / "models", None, open_nli_model=lambda *a, **k: model, profiles_dir=profiles
    )
    install_fake(tmp_path / "models", "nli-mobilebert-int8")
    registry.load_backend("nli-mobilebert-int8").score("x", "q?", ["low", "high"])
    assert [h for _, h in model.pairs] == ["low.", "high."]


def test_model_dir_profile_overrides_packaged_profile(tmp_path, profiles):
    model = RecordingNliModel()
    registry = ModelRegistry(
        tmp_path / "models", None, open_nli_model=lambda *a, **k: model, profiles_dir=profiles
    )
    install_fake(
        tmp_path / "models",
        "nli-mobilebert-int8",
        **{"profile.json": {"score_template": "answer-is"}},
    )
    registry.load_backend("nli-mobilebert-int8").score("x", "q?", ["low", "high"])
    assert model.pairs[0][1] == 'The answer to "q?" is low.'


def test_packaged_calibration_is_the_fallback(tmp_path, profiles):
    registry = ModelRegistry(tmp_path / "models", None, profiles_dir=profiles)
    assert registry.load_calibrator("nli-mobilebert-int8") == PlattCalibrator(a=2.0, b=0.0)


def test_every_curated_model_ships_tuned_profile_and_calibration():
    from noulo.registry import CURATED_MODELS, PROFILES_DIR

    for m in CURATED_MODELS:
        assert (PROFILES_DIR / m.id / "profile.json").exists(), m.id
        assert (PROFILES_DIR / m.id / "calibration.json").exists(), m.id


def test_is_installed_covers_catalog_embedders(registry, tmp_path):
    assert not registry.is_installed("minilm-l6-v2-int8")
    install_fake(tmp_path / "models", "minilm-l6-v2-int8")
    assert registry.is_installed("minilm-l6-v2-int8")


# ---------------------------------------------------------------- tiers, size, RAM


def test_catalog_has_larger_tier_between_half_and_one_gigabyte():
    large = [e for e in CATALOG if e.kind == "nli" and e.tier == "large"]
    assert len(large) >= 3
    assert all(500 <= e.size_mb <= 1024 for e in large)


def test_listing_reports_size_quantisation_tier_and_measured_ram(tmp_path):
    profiles = tmp_path / "profiles"
    (profiles / "nli-mobilebert-int8").mkdir(parents=True)
    (profiles / "nli-mobilebert-int8" / "measured.json").write_text(
        json.dumps({"peakRssBytes": 150 * 1024 * 1024, "noulAccuracy": 0.84})
    )
    registry = ModelRegistry(tmp_path / "models", None, profiles_dir=profiles)
    entry = {m["id"]: m for m in registry.list()}["nli-mobilebert-int8"]
    assert entry["sizeMB"] == 26 and entry["quantization"] == "INT8" and entry["tier"] == "basic"
    assert entry["ramMB"] == 150 and entry["noulAccuracy"] == 0.84
    unmeasured = {m["id"]: m for m in registry.list()}["nli-distilbert-int8"]
    assert unmeasured["ramMB"] is None


def test_quantisation_damaged_large_models_are_not_offered_in_menus():
    tiers = {e.id: e.tier for e in CATALOG}
    assert tiers["nli-deberta-v3-large-int8"] == "experimental"
    assert tiers["nli-deberta-v3-large-anli-int8"] == "experimental"
    assert tiers["zeroshot-deberta-v3-base-fp32"] == "large"


def test_catalog_entries_can_carry_a_menu_label(registry):
    entry = {m["id"]: m for m in registry.list()}["zeroshot-deberta-v3-base-fp32"]
    assert entry["label"] == "Most accurate"


def test_http_fetch_reports_download_progress(tmp_path, monkeypatch):
    import contextlib

    import httpx

    from noulo.registry import _http_fetch

    class FakeResponse:
        headers = {"content-length": str(10 * 1024 * 1024)}

        def raise_for_status(self):
            pass

        def iter_bytes(self, size):
            for _ in range(10):
                yield b"x" * 1024 * 1024

    @contextlib.contextmanager
    def fake_stream(method, url, **kwargs):
        yield FakeResponse()

    monkeypatch.setattr(httpx, "stream", fake_stream)
    messages = []
    _http_fetch("https://example/model.onnx", tmp_path / "model.onnx", progress=messages.append)
    assert (tmp_path / "model.onnx").stat().st_size == 10 * 1024 * 1024
    assert messages[0].startswith("  model.onnx: 10%")
    assert messages[-1].startswith("  model.onnx: 100%")
    assert len(messages) == 10


def test_registry_passes_low_memory_to_nli_models(tmp_path):
    opened = {}

    def opener(model_dir, **kwargs):
        opened.update(kwargs)
        return RecordingNliModel()

    registry = ModelRegistry(tmp_path / "models", None, open_nli_model=opener, low_memory=True)
    install_fake(tmp_path / "models", "nli-mobilebert-int8")
    registry.load_backend("nli-mobilebert-int8")
    assert opened["low_memory"] is True


# ---------------------------------------------------------------- local LLMs


LLM_IDS = ("qwen3-0.6b-q4f16", "qwen2.5-0.5b-q4", "gemma-3-1b-q4", "lfm2-1.2b-q4")


def test_catalog_offers_prominent_small_llms_highly_quantised():
    entries = {e.id: e for e in CATALOG}
    for model_id in LLM_IDS:
        entry = entries[model_id]
        assert entry.kind == "llm" and entry.tier == "llm"
        assert entry.quantization == "INT4" and 450 <= entry.size_mb <= 900


def test_llm_download_keeps_external_weight_file_names(registry, tmp_path):
    files = {
        "model_q4.onnx": b"graph",
        "model_q4.onnx_data": b"weights",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        "config.json": b"{}",
    }
    path = registry.download("gemma-3-1b-q4", fetch=fake_fetch(files))
    assert (path / "model.onnx").read_bytes() == b"graph"
    assert (path / "model_q4.onnx_data").read_bytes() == b"weights"
    assert (path / "tokenizer_config.json").exists()
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["sizeBytes"] == len(b"graph") + len(b"weights")


def install_fake_llm(models_dir, model_id, data_file=None):
    d = models_dir / model_id
    d.mkdir(parents=True)
    for name in ("model.onnx", "tokenizer.json", "tokenizer_config.json", "config.json"):
        (d / name).write_text("{}")
    if data_file:
        (d / data_file).write_text("{}")
    return d


def test_llm_listing_and_loading(tmp_path):
    from noulo.inference.llm_backend import LlmBackend

    class FakeLM:
        def label_distribution(self, messages, labels):
            return [1 / len(labels)] * len(labels), 1.0

        def close(self):
            pass

    registry = ModelRegistry(tmp_path / "models", None, open_llm=lambda *a, **k: FakeLM())
    assert not {m["id"]: m for m in registry.list()}["gemma-3-1b-q4"]["installed"]
    install_fake_llm(tmp_path / "models", "gemma-3-1b-q4", "model_q4.onnx_data")
    entry = {m["id"]: m for m in registry.list()}["gemma-3-1b-q4"]
    assert entry["installed"] and entry["backend"] == "onnx-llm" and entry["tier"] == "llm"
    backend = registry.load_backend("gemma-3-1b-q4")
    assert isinstance(backend, LlmBackend) and backend.info.quantization == "INT4"


def test_llm_without_its_external_weights_is_not_installed(tmp_path):
    registry = ModelRegistry(tmp_path / "models", None)
    install_fake_llm(tmp_path / "models", "gemma-3-1b-q4")  # data file missing
    assert not {m["id"]: m for m in registry.list()}["gemma-3-1b-q4"]["installed"]


def test_register_your_own_onnx_llm(tmp_path):
    from noulo.inference.llm_backend import LlmBackend

    class FakeLM:
        def close(self):
            pass

    registry = ModelRegistry(
        tmp_path / "models", tmp_path / "models.json", open_llm=lambda *a, **k: FakeLM()
    )
    custom = install_fake_llm(tmp_path / "elsewhere", "my-llm")
    registry.register_onnx(id="my-llm", path=custom, quantization="INT4", kind="llm")
    entry = {m["id"]: m for m in registry.list()}["my-llm"]
    assert entry["backend"] == "onnx-llm" and entry["installed"]
    assert isinstance(registry.load_backend("my-llm"), LlmBackend)


def test_register_onnx_llm_requires_chat_template_config(registry, tmp_path):
    custom = install_fake(tmp_path / "elsewhere", "nli-like")  # no tokenizer_config.json
    with pytest.raises(ValueError, match="tokenizer_config.json"):
        registry.register_onnx(id="x", path=custom, kind="llm")


# ---------------------------------------------------------------- devices


def test_registry_resolves_a_placement_per_model_and_passes_it_to_the_loader(tmp_path):
    seen = {}

    def opener(model_dir, **kwargs):
        seen.update(kwargs)
        model = RecordingNliModel()
        model.device = kwargs["placement"].device
        return model

    registry = ModelRegistry(tmp_path / "models", None, open_nli_model=opener, device="cpu")
    install_fake(tmp_path / "models", "nli-mobilebert-int8")
    backend = registry.load_backend("nli-mobilebert-int8")
    assert seen["placement"].device == "cpu"
    assert backend.info.device == "cpu"


def test_unknown_device_setting_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="device"):
        ModelRegistry(tmp_path / "models", None, device="tpu")
