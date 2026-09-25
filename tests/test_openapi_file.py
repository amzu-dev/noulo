import io
import json
from pathlib import Path

from openapi_spec_validator import validate

from noulo.cli import main

ROOT = Path(__file__).resolve().parent.parent


def generate(tmp_path) -> dict:
    target = tmp_path / "openapi.json"
    assert main(["openapi", "--output", str(target)], stdout=io.StringIO()) == 0
    return json.loads(target.read_text())


def test_openapi_command_writes_a_valid_spec_without_loading_a_model(tmp_path):
    spec = generate(tmp_path)
    validate(spec)
    assert spec["info"]["title"] == "noulo"


def test_committed_openapi_json_matches_the_live_schema(tmp_path):
    committed = json.loads((ROOT / "openapi.json").read_text())
    assert committed == generate(tmp_path), "run: noulo openapi --output openapi.json"
