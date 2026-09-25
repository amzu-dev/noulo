import pytest

from noulo.inference.devices import Placement, resolve
from noulo.inference.types import ModelLoadError

CPU = "CPUExecutionProvider"
COREML = "CoreMLExecutionProvider"
CUDA = "CUDAExecutionProvider"
DML = "DmlExecutionProvider"


def names(placement: Placement) -> list[str]:
    return [p if isinstance(p, str) else p[0] for p in placement.providers]


def test_cpu_is_always_cpu():
    placement = resolve("cpu", available=[COREML, CPU], precision="FP32")
    assert placement.device == "cpu" and names(placement) == [CPU]


def test_explicit_accelerator_keeps_cpu_as_fallback_for_unsupported_ops():
    placement = resolve("coreml", available=[COREML, CPU], precision="INT8")
    assert placement.device == "coreml" and names(placement) == [COREML, CPU]


def test_explicit_accelerator_that_is_not_installed_explains_how_to_get_it():
    with pytest.raises(ModelLoadError, match="onnxruntime-gpu"):
        resolve("cuda", available=[CPU], precision="FP32")
    with pytest.raises(ModelLoadError, match="onnxruntime-directml"):
        resolve("directml", available=[CPU], precision="FP32")


def test_gpu_picks_the_best_available_accelerator():
    assert resolve("gpu", available=[COREML, CPU], precision="INT8").device == "coreml"
    assert resolve("gpu", available=[DML, CUDA, CPU], precision="INT8").device == "cuda"


def test_gpu_without_any_accelerator_falls_back_to_cpu_with_a_reason():
    placement = resolve("gpu", available=[CPU], precision="FP32")
    assert placement.device == "cpu" and "no GPU" in placement.reason


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="device"):
        resolve("tpu", available=[CPU], precision="FP32")


def test_coreml_uses_mlprogram_all_compute_units_and_a_compile_cache(tmp_path):
    placement = resolve("coreml", available=[COREML, CPU], precision="FP32", cache_dir=tmp_path)
    name, options = placement.providers[0]
    assert name == COREML
    assert options["ModelFormat"] == "MLProgram" and options["MLComputeUnits"] == "ALL"
    assert options["ModelCacheDirectory"] == str(tmp_path)


# ---------------------------------------------------------------- session creation


def test_create_session_uses_placement_providers(monkeypatch, tmp_path):
    import onnxruntime as ort

    from noulo.inference.devices import create_session

    seen = {}

    class FakeSession:
        def __init__(self, path, options=None, providers=None, **kwargs):
            seen["providers"], seen["kwargs"] = providers, kwargs

    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    placement = Placement("coreml", [(COREML, {"ModelFormat": "MLProgram"}), CPU])
    _, device = create_session(
        tmp_path / "m.onnx",
        ort.SessionOptions(),
        placement,
        disabled_optimizers=["ConstantFolding"],
    )
    assert device == "coreml" and seen["providers"][0][0] == COREML
    assert seen["kwargs"] == {"disabled_optimizers": ["ConstantFolding"]}


def test_create_session_falls_back_to_cpu_when_accelerator_fails(monkeypatch, tmp_path, caplog):
    import onnxruntime as ort

    from noulo.inference.devices import create_session

    class FlakySession:
        def __init__(self, path, options=None, providers=None, **kwargs):
            if providers != [CPU]:
                raise RuntimeError("accelerator init failed")
            self.providers = providers

    monkeypatch.setattr(ort, "InferenceSession", FlakySession)
    session, device = create_session(
        tmp_path / "m.onnx", ort.SessionOptions(), Placement("cuda", [CUDA, CPU])
    )
    assert device == "cpu" and session.providers == [CPU]
    assert "falling back to CPU" in caplog.text


def test_default_placement_is_cpu():
    from noulo.inference.devices import CPU_PLACEMENT

    assert CPU_PLACEMENT.device == "cpu" and CPU_PLACEMENT.providers == [CPU]


# ---------------------------------------------------------------- auto policy (measured)


def test_auto_skips_coreml_because_it_measured_slower_than_cpu():
    placement = resolve("auto", available=[COREML, CPU], precision="FP32")
    assert placement.device == "cpu" and "CoreML" in placement.reason


def test_auto_uses_a_discrete_gpu_for_float_models():
    assert resolve("auto", available=[CUDA, CPU], precision="FP32").device == "cuda"
    assert resolve("auto", available=[DML, CPU], precision="FP16").device == "directml"


def test_auto_keeps_quantised_models_on_cpu():
    placement = resolve("auto", available=[CUDA, CPU], precision="INT8")
    assert placement.device == "cpu" and "quantised" in placement.reason
    assert resolve("auto", available=[CUDA, CPU], precision="INT4").device == "cpu"


def test_auto_without_accelerators_is_cpu():
    assert resolve("auto", available=[CPU], precision="FP32").device == "cpu"
