"""Real-model regression checks use the actual browser example definitions.

Node is optional for Python-only installations. No browser, npm dependencies or
remote models are needed; Node only evaluates the side-effect-free ES module.
The original crash example was reproduced RED in the browser before changing it.
"""

import itertools
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from noulo.embedded import Noulo
from noulo.inference.devices import CPU_PLACEMENT
from noulo.inference.engine import DecisionEngine
from noulo.inference.model import OnnxNliModel
from noulo.inference.nli_backend import NliBackend
from noulo.registry import PROFILES_DIR
from tests.conftest import DEFAULT_NLI_MODEL_DIR, DEFAULT_NLI_MODEL_ID, MODELS_DIR

NODE = shutil.which("node")
pytestmark = [pytest.mark.model, pytest.mark.skipif(NODE is None, reason="Node not installed")]


@pytest.fixture(scope="module")
def example():
    assert NODE is not None  # module is skipped above when Node is unavailable
    module = Path(__file__).resolve().parents[1] / "src/noulo/ui/static/examples.js"
    # Force ESM even on Node versions without automatic .js module detection.
    script = module.read_text() + "\nconsole.log(JSON.stringify(EXAMPLES.choice()));"
    result = subprocess.run(
        [NODE, "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def engine():
    # Construct the backend directly: the registry deliberately merges local
    # profile.json overrides, which must not alter this shipped-profile test.
    profile = json.loads((PROFILES_DIR / DEFAULT_NLI_MODEL_ID / "profile.json").read_text())
    backend = NliBackend(
        OnnxNliModel(DEFAULT_NLI_MODEL_DIR, placement=CPU_PLACEMENT),
        model_id=DEFAULT_NLI_MODEL_ID,
        quantization="INT8",
        **profile,
    )
    decision = DecisionEngine(load_backend=lambda _id: backend, model_id=DEFAULT_NLI_MODEL_ID)
    with Noulo(
        engine=decision,
        _env_file=None,
        model=DEFAULT_NLI_MODEL_ID,
        device="cpu",
        models_dir=MODELS_DIR,
        models_file=None,
        threads=None,
        low_memory=False,
        learning_enabled=False,
    ) as engine:
        yield engine


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("NOULO_MODEL", "not-the-bundled-model"),
        ("NOULO_DEVICE", "not-the-test-device"),
        ("NOULO_MODELS_DIR", "not-the-bundled-model-directory"),
        ("NOULO_MODELS_FILE", "not-the-test-models.json"),
        ("NOULO_THREADS", "2"),
        ("NOULO_LOW_MEMORY", "true"),
        (None, None),  # local profile override alone
    ],
)
def test_engine_fixture_ignores_external_model_configuration(monkeypatch, name, value):
    for variable in (
        "NOULO_MODEL",
        "NOULO_DEVICE",
        "NOULO_MODELS_DIR",
        "NOULO_MODELS_FILE",
        "NOULO_THREADS",
        "NOULO_LOW_MEMORY",
    ):
        monkeypatch.delenv(variable, raising=False)
    if name is not None:
        monkeypatch.setenv(name, value)

    # Simulate a local calibration override without touching downloaded files.
    local_profile = DEFAULT_NLI_MODEL_DIR / "profile.json"
    exists, read_text = Path.exists, Path.read_text
    monkeypatch.setattr(Path, "exists", lambda p: p.resolve() == local_profile or exists(p))
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda p, *a, **kw: (
            '{"choice_premise": "input+question"}'
            if p.resolve() == local_profile
            else read_text(p, *a, **kw)
        ),
    )
    fixture = engine.__wrapped__()
    try:
        actual = next(fixture)
        profile = json.loads((PROFILES_DIR / DEFAULT_NLI_MODEL_ID / "profile.json").read_text())
        assert actual.engine._slot.backend._choice.premise == profile["choice_premise"]
        assert actual.engine.model_info.id == DEFAULT_NLI_MODEL_ID
        assert actual.engine.model_info.device == "cpu"
        assert actual.settings.model == DEFAULT_NLI_MODEL_ID
        assert actual.settings.device == "cpu"
        assert actual.settings.models_dir == MODELS_DIR
        assert actual.settings.models_file is None
        assert actual.settings.threads is None
        assert actual.settings.low_memory is False
    finally:
        fixture.close()


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("You charged my card twice. Please refund the duplicate payment.", "A"),
        ("The app crashes every time I open settings.", "B"),
        ("I would like a quote for 100 licences.", "C"),
    ],
)
def test_ui_descriptive_choices_route_known_failures_in_any_order(
    engine, example, order, text, expected
):
    options = [example["options"][i] for i in order]
    assert engine.choice(text, example["question"], options) == expected
