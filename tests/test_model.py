import numpy as np
import pytest

from noulo.inference.model import OnnxNliModel, parse_nli_labels
from noulo.inference.types import ModelLoadError


def test_parse_three_class_labels():
    labels = parse_nli_labels({"0": "contradiction", "1": "entailment", "2": "neutral"})
    assert labels == {"entailment": 1, "neutral": 2, "contradiction": 0}


def test_parse_labels_is_case_insensitive():
    labels = parse_nli_labels({"0": "ENTAILMENT", "1": "NEUTRAL", "2": "CONTRADICTION"})
    assert labels == {"entailment": 0, "neutral": 1, "contradiction": 2}


def test_parse_two_class_zero_shot_labels():
    labels = parse_nli_labels({"0": "entailment", "1": "not_entailment"})
    assert labels == {"entailment": 0, "not_entailment": 1}


def test_parse_labels_requires_entailment():
    with pytest.raises(ModelLoadError, match="entailment"):
        parse_nli_labels({"0": "LABEL_0", "1": "LABEL_1"})


def test_missing_model_dir_raises_without_leaking_path(tmp_path):
    missing = tmp_path / "nowhere" / "secret-dir"
    with pytest.raises(ModelLoadError) as excinfo:
        OnnxNliModel(missing)
    assert str(tmp_path) not in str(excinfo.value)
    assert "secret-dir" not in str(excinfo.value)


def test_corrupt_model_file_raises_model_load_error(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"not an onnx graph")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "config.json").write_text('{"id2label": {"0": "entailment", "1": "neutral"}}')
    with pytest.raises(ModelLoadError):
        OnnxNliModel(tmp_path)


PREMISE = "I checked my account and you have taken the subscription payment twice."


@pytest.mark.model
def test_real_model_returns_one_logit_row_per_pair(real_nli_model):
    logits = real_nli_model.predict_logits(
        [(PREMISE, "The customer was charged twice."), (PREMISE, "The weather is nice.")]
    )
    assert logits.shape == (2, len(real_nli_model.labels))
    assert np.all(np.isfinite(logits))


@pytest.mark.model
def test_real_model_detects_entailment_and_contradiction(real_nli_model):
    probs = real_nli_model.predict_probs(
        [
            (PREMISE, "The customer reports being charged more than once."),
            (PREMISE, "The customer has never been charged."),
        ]
    )
    ent, con = real_nli_model.labels["entailment"], real_nli_model.labels["contradiction"]
    assert probs[0].argmax() == ent
    assert probs[1].argmax() == con
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.model
def test_real_model_batching_matches_single_predictions(real_nli_model):
    pairs = [
        (PREMISE, "Billing"),
        ("Short.", "A much longer hypothesis sentence that forces padding in the batch."),
    ]
    batched = real_nli_model.predict_logits(pairs)
    singles = np.vstack([real_nli_model.predict_logits([p]) for p in pairs])
    # Dynamic INT8 quantisation computes activation scales per batch, so padding
    # introduces small noise; order and meaning must be preserved.
    np.testing.assert_allclose(batched, singles, atol=0.15)
    np.testing.assert_array_equal(batched.argmax(axis=1), singles.argmax(axis=1))


@pytest.mark.model
def test_real_model_truncates_overlong_input(real_nli_model):
    long_input = "The server is down. " * 2000
    logits = real_nli_model.predict_logits([(long_input, "The server is unavailable.")])
    assert logits.shape[0] == 1


@pytest.mark.model
def test_real_model_is_deterministic(real_nli_model):
    pair = [(PREMISE, "The customer wants a refund.")]
    np.testing.assert_array_equal(
        real_nli_model.predict_logits(pair), real_nli_model.predict_logits(pair)
    )


@pytest.mark.model
def test_empty_batch_returns_empty_array(real_nli_model):
    assert real_nli_model.predict_logits([]).shape == (0, len(real_nli_model.labels))


def test_git_lfs_pointer_file_gives_actionable_error(tmp_path):
    (tmp_path / "model.onnx").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 91000000\n"
    )
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "config.json").write_text('{"id2label": {"0": "entailment", "1": "neutral"}}')
    with pytest.raises(ModelLoadError, match="git lfs pull"):
        OnnxNliModel(tmp_path)


def _capture_session(monkeypatch):
    import onnxruntime as ort

    seen = {}
    real = ort.InferenceSession

    def fake(path, options=None, providers=None, **kwargs):
        seen.update(kwargs)
        return real(path, options, providers=providers, **kwargs)

    monkeypatch.setattr(ort, "InferenceSession", fake)
    return seen


@pytest.mark.model
def test_low_memory_mode_disables_constant_folding(monkeypatch):
    from tests.conftest import DEFAULT_NLI_MODEL_DIR

    seen = _capture_session(monkeypatch)
    OnnxNliModel(DEFAULT_NLI_MODEL_DIR, low_memory=True).close()
    assert seen.get("disabled_optimizers") == ["ConstantFolding"]


@pytest.mark.model
def test_default_mode_keeps_all_optimisations(monkeypatch):
    from tests.conftest import DEFAULT_NLI_MODEL_DIR

    seen = _capture_session(monkeypatch)
    OnnxNliModel(DEFAULT_NLI_MODEL_DIR).close()
    assert not seen.get("disabled_optimizers")


def _coreml_available():
    import onnxruntime as ort

    return "CoreMLExecutionProvider" in ort.get_available_providers()


@pytest.mark.model
def test_models_run_on_cpu_by_default(real_nli_model):
    assert real_nli_model.device == "cpu"


@pytest.mark.model
@pytest.mark.skipif(not _coreml_available(), reason="CoreML not available")
def test_coreml_placement_matches_cpu_predictions(real_nli_model):
    from noulo.inference.devices import resolve
    from tests.conftest import DEFAULT_NLI_MODEL_DIR

    placement = resolve(
        "coreml", available=["CoreMLExecutionProvider", "CPUExecutionProvider"], precision="INT8"
    )
    gpu = OnnxNliModel(DEFAULT_NLI_MODEL_DIR, placement=placement)
    pairs = [
        (PREMISE, "The customer reports being charged more than once."),
        (PREMISE, "The customer has never been charged."),
    ]
    assert gpu.device == "coreml"
    np.testing.assert_array_equal(
        gpu.predict_logits(pairs).argmax(axis=1),
        real_nli_model.predict_logits(pairs).argmax(axis=1),
    )
    gpu.close()
