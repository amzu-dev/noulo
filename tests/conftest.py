from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DEFAULT_NLI_MODEL_ID = "nli-deberta-v3-xsmall-int8"
DEFAULT_NLI_MODEL_DIR = MODELS_DIR / DEFAULT_NLI_MODEL_ID


def model_available(model_dir: Path = DEFAULT_NLI_MODEL_DIR) -> bool:
    return (model_dir / "model.onnx").exists() and (model_dir / "tokenizer.json").exists()


def pytest_collection_modifyitems(config, items):
    if model_available():
        return
    skip = pytest.mark.skip(reason=f"model not downloaded ({DEFAULT_NLI_MODEL_ID})")
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def real_nli_model():
    from noulo.inference.model import OnnxNliModel

    model = OnnxNliModel(DEFAULT_NLI_MODEL_DIR)
    yield model
    model.close()
