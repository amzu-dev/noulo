"""Where models run: CPU or an ONNX Runtime accelerator (GPU / unified memory / NPU).

`resolve()` turns the NOULO_DEVICE setting into an ordered provider list. The
CPU provider is always last, so any operation the accelerator can't run
(common for INT8/INT4 kernels) falls back to the CPU instead of failing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import ModelLoadError

CPU = "CPUExecutionProvider"
ACCELERATORS = {  # preference order for "gpu" / "auto"
    "cuda": "CUDAExecutionProvider",  # NVIDIA (pip install onnxruntime-gpu)
    "rocm": "ROCMExecutionProvider",  # AMD (onnxruntime-rocm)
    "directml": "DmlExecutionProvider",  # any Windows GPU (onnxruntime-directml)
    "coreml": "CoreMLExecutionProvider",  # Apple Silicon GPU / Neural Engine (built in)
}
INSTALL_HINTS = {
    "cuda": "install onnxruntime-gpu in place of onnxruntime (NVIDIA driver + CUDA required)",
    "rocm": "install onnxruntime-rocm in place of onnxruntime",
    "directml": "install onnxruntime-directml in place of onnxruntime (Windows)",
    "coreml": "CoreML is only available on macOS",
}
DEVICES = ("auto", "cpu", "gpu", *ACCELERATORS)
# `auto` policy, from measurements (docs/gpu.md): CoreML on Apple Silicon was slower than the
# CPU for every model type tested (dynamic sequence lengths force CPU round-trips), so auto
# only uses discrete-GPU providers, and only for unquantised models.
AUTO_ACCELERATORS = ("cuda", "rocm", "directml")
FLOAT_PRECISIONS = ("FP32", "FP16")


@dataclass(frozen=True)
class Placement:
    device: str  # "cpu" or an ACCELERATORS key
    providers: list[Any] = field(default_factory=lambda: [CPU])
    reason: str = ""


def _providers(device: str, cache_dir: Path | None) -> list[Any]:
    name = ACCELERATORS[device]
    if device == "coreml":
        options = {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}
        if cache_dir is not None:
            options["ModelCacheDirectory"] = str(cache_dir)
        return [(name, options), CPU]
    return [name, CPU]


def resolve(
    device: str, *, available: Sequence[str], precision: str, cache_dir: Path | None = None
) -> Placement:
    device = device.lower()
    if device not in DEVICES:
        raise ValueError(f"Unknown device {device!r}; use one of: {', '.join(DEVICES)}.")
    if device == "cpu":
        return Placement("cpu", [CPU], "requested")
    if device in ACCELERATORS:
        if ACCELERATORS[device] not in available:
            raise ModelLoadError(f"Device {device!r} is not available: {INSTALL_HINTS[device]}.")
        return Placement(device, _providers(device, cache_dir), "requested")
    present = [d for d, name in ACCELERATORS.items() if name in available]
    if device == "gpu":
        if not present:
            return Placement("cpu", [CPU], "no GPU execution provider is installed")
        return Placement(present[0], _providers(present[0], cache_dir), "gpu: best available")
    # auto: only where it measured (or is known to be) faster than the CPU
    discrete = [d for d in present if d in AUTO_ACCELERATORS]
    if not discrete:
        reason = (
            "CoreML measured slower than the CPU for these models"
            if "coreml" in present
            else "no GPU execution provider is installed"
        )
        return Placement("cpu", [CPU], reason)
    if precision.upper() not in FLOAT_PRECISIONS:
        return Placement("cpu", [CPU], "quantised model: INT8/INT4 kernels run best on the CPU")
    return Placement(discrete[0], _providers(discrete[0], cache_dir), "auto: float model on GPU")


CPU_PLACEMENT = Placement("cpu", [CPU], "default")


def create_session(
    path: Path, options: Any, placement: Placement = CPU_PLACEMENT, **kwargs: Any
) -> tuple[Any, str]:
    """Create an InferenceSession on the placement's device; fall back to CPU on failure."""
    import logging

    import onnxruntime as ort

    try:
        return ort.InferenceSession(
            str(path), options, providers=placement.providers, **kwargs
        ), placement.device
    except Exception as exc:
        if placement.device == "cpu":
            raise
        logging.getLogger(__name__).warning(
            "Could not start %s on %s (%s); falling back to CPU.",
            Path(path).parent.name,
            placement.device,
            type(exc).__name__,
        )
        return ort.InferenceSession(str(path), options, providers=[CPU], **kwargs), "cpu"
