import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from noulo.inference.embedder import HashingEmbedder, OnnxEmbedder, OpenAIEmbedder
from noulo.inference.types import BackendError, Embedder, ModelLoadError


def test_hashing_embedder_returns_unit_float32_rows():
    emb = HashingEmbedder(dim=64)
    out = emb.embed(["hello world", "another sentence"])
    assert out.shape == (2, 64)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)


def test_hashing_embedder_id_and_determinism():
    a, b = HashingEmbedder(dim=128), HashingEmbedder(dim=128)
    assert a.id == "hashing-128"
    assert HashingEmbedder().dim == 256
    np.testing.assert_array_equal(a.embed(["Same text"]), b.embed(["Same text"]))


def _cos(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v))


def test_hashing_embedder_character_trigrams_link_word_variants():
    emb = HashingEmbedder()
    single, plural = emb.embed(["overdue", "overdues"])
    assert _cos(single, plural) > 0.6


def test_hashing_embedder_similar_strings_score_higher_than_unrelated():
    emb = HashingEmbedder()
    base, near, far = emb.embed(
        [
            "The invoice is overdue.",
            "the invoice is  OVERDUE",
            "Sunny weather at the beach today",
        ]
    )
    assert _cos(base, near) > 0.95
    assert _cos(base, far) < 0.5


def test_hashing_embedder_handles_empty_inputs():
    emb = HashingEmbedder(dim=32)
    assert emb.embed([]).shape == (0, 32)
    blank = emb.embed([""])
    assert blank.shape == (1, 32)
    assert np.all(np.isfinite(blank))


def test_hashing_embedder_rejects_non_positive_dim():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)


def test_hashing_embedder_satisfies_protocol_and_close_is_idempotent():
    emb = HashingEmbedder()
    assert isinstance(emb, Embedder)
    emb.close()
    emb.close()


# --- OnnxEmbedder ---------------------------------------------------------


def test_onnx_embedder_missing_model_raises_without_leaking_path(tmp_path):
    with pytest.raises(ModelLoadError) as excinfo:
        OnnxEmbedder(tmp_path, id="missing")
    message = str(excinfo.value)
    assert "model.onnx" in message
    assert str(tmp_path) not in message


def test_onnx_embedder_corrupt_model_raises_without_leaking_path(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"definitely not an onnx graph")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ModelLoadError) as excinfo:
        OnnxEmbedder(tmp_path, id="corrupt")
    assert str(tmp_path) not in str(excinfo.value)


MINILM_DIR = Path(__file__).resolve().parent.parent / "models" / "minilm-l6-v2-int8"
needs_minilm = pytest.mark.skipif(
    not (MINILM_DIR / "model.onnx").is_file(), reason="MiniLM ONNX model not downloaded"
)


@pytest.fixture(scope="module")
def minilm():
    emb = OnnxEmbedder(MINILM_DIR, id="minilm-l6-v2-int8", threads=1)
    yield emb
    emb.close()


@pytest.mark.model
@needs_minilm
def test_onnx_embedder_returns_unit_float32_rows(minilm):
    out = minilm.embed(["The invoice is overdue.", "Hello"])
    assert minilm.id == "minilm-l6-v2-int8"
    assert minilm.dim == 384
    assert out.shape == (2, 384)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)


@pytest.mark.model
@needs_minilm
def test_onnx_embedder_captures_meaning(minilm):
    invoice, payment, weather = minilm.embed(
        [
            "The invoice is overdue.",
            "Payment for the bill is late.",
            "The weather is sunny today.",
        ]
    )
    assert _cos(invoice, payment) > _cos(invoice, weather)


@pytest.mark.model
@needs_minilm
def test_onnx_embedder_empty_input(minilm):
    out = minilm.embed([])
    assert out.shape == (0, 384)
    assert out.dtype == np.float32


@pytest.mark.model
@needs_minilm
def test_onnx_embedder_padding_does_not_change_embeddings(minilm):
    # The int8 graph uses dynamic activation quantisation, whose scale depends on
    # the whole batch, so results are near-identical rather than bit-identical.
    # Without attention-mask pooling the cosine here drops far below 0.98.
    short = "Short text."
    alone = minilm.embed([short])[0]
    batched = minilm.embed([short, "A much longer sentence " * 10])[0]
    assert _cos(alone, batched) > 0.98


@pytest.mark.model
@needs_minilm
def test_onnx_embedder_truncates_to_max_length():
    emb = OnnxEmbedder(MINILM_DIR, id="minilm-short", max_length=8, threads=1)
    try:
        head, longer = emb.embed(
            ["alpha beta gamma delta epsilon zeta", "alpha beta gamma delta epsilon zeta eta theta"]
        )
        assert _cos(head, longer) > 0.99
    finally:
        emb.close()


@pytest.mark.model
@needs_minilm
def test_onnx_embedder_accepts_very_long_input(minilm):
    out = minilm.embed(["word " * 2000])
    assert out.shape == (1, 384)


@pytest.mark.model
@needs_minilm
def test_onnx_embedder_close_is_idempotent_and_blocks_use():
    emb = OnnxEmbedder(MINILM_DIR, id="minilm", threads=1)
    assert isinstance(emb, Embedder)
    emb.close()
    emb.close()
    with pytest.raises(BackendError):
        emb.embed(["after close"])


# --- OpenAIEmbedder -------------------------------------------------------

SECRET = "sk-test-secret-key"


def _embeddings_server(requests: list[httpx.Request], *, reverse: bool = False):
    """Fake /embeddings endpoint: text of length n -> vector [n, 0, 0] ... unnormalised."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        data = [
            {"object": "embedding", "index": i, "embedding": [float(len(t)), 1.0, 0.0]}
            for i, t in enumerate(body["input"])
        ]
        if reverse:
            data.reverse()
        return httpx.Response(200, json={"object": "list", "data": data, "model": body["model"]})

    return httpx.MockTransport(handler)


def _openai(transport: httpx.MockTransport, **kwargs) -> OpenAIEmbedder:
    return OpenAIEmbedder(
        id="remote-embed",
        base_url="http://llm.local/v1",
        model="text-embedding-3-small",
        api_key=SECRET,
        client=httpx.Client(transport=transport),
        **kwargs,
    )


def test_openai_embedder_posts_inputs_and_orders_by_index():
    requests: list[httpx.Request] = []
    emb = _openai(_embeddings_server(requests, reverse=True))
    out = emb.embed(["a", "abc"])

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "http://llm.local/v1/embeddings"
    assert request.headers["Authorization"] == f"Bearer {SECRET}"
    assert json.loads(request.content) == {"model": "text-embedding-3-small", "input": ["a", "abc"]}

    assert out.dtype == np.float32
    assert out.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)
    expected = np.array([[1, 1, 0], [3, 1, 0]], dtype=np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def _raising_transport(exc: Exception) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return httpx.MockTransport(handler)


def _fixed_response(status: int, content: bytes) -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(status, content=content))


def _error_chain_text(exc: BaseException) -> str:
    parts = []
    current: BaseException | None = exc
    while current is not None:
        parts.append(f"{current!s} {current!r}")
        current = current.__cause__ or current.__context__
    return " ".join(parts)


def _json_response(payload: object) -> httpx.MockTransport:
    return _fixed_response(200, json.dumps(payload).encode())


@pytest.mark.parametrize(
    ("transport", "texts"),
    [
        pytest.param(
            _fixed_response(401, json.dumps({"error": f"Incorrect API key {SECRET}"}).encode()),
            ["x"],
            id="http-401",
        ),
        pytest.param(_fixed_response(500, b"upstream exploded"), ["x"], id="http-500"),
        pytest.param(
            _raising_transport(httpx.ConnectError("connection refused")), ["x"], id="connect"
        ),
        pytest.param(_raising_transport(httpx.ReadTimeout("timed out")), ["x"], id="timeout"),
        pytest.param(_fixed_response(200, b"<html>not json</html>"), ["x"], id="not-json"),
        pytest.param(_json_response({"no": "data"}), ["x"], id="missing-data"),
        pytest.param(_json_response({"data": [{"index": 0}]}), ["x"], id="missing-embedding"),
        pytest.param(
            _json_response({"data": [{"index": 0, "embedding": ["x"]}]}), ["x"], id="non-numeric"
        ),
        pytest.param(_json_response({"data": []}), ["x"], id="wrong-count"),
        pytest.param(
            _json_response(
                {"data": [{"index": 0, "embedding": [1.0, 2.0]}, {"index": 1, "embedding": [1.0]}]}
            ),
            ["x", "y"],
            id="ragged",
        ),
        pytest.param(
            _fixed_response(200, b'{"data": [{"index": 0, "embedding": [NaN, 1.0]}]}'),
            ["x"],
            id="non-finite",
        ),
    ],
)
def test_openai_embedder_failures_raise_backend_error_without_key(transport, texts):
    emb = _openai(transport)
    with pytest.raises(BackendError) as excinfo:
        emb.embed(texts)
    assert SECRET not in _error_chain_text(excinfo.value)


def test_openai_embedder_dim_is_probed_lazily_once():
    requests: list[httpx.Request] = []
    emb = _openai(_embeddings_server(requests))
    assert requests == []
    assert emb.dim == 3
    assert emb.dim == 3
    assert len(requests) == 1


def test_openai_embedder_dim_known_after_first_embed():
    requests: list[httpx.Request] = []
    emb = _openai(_embeddings_server(requests))
    emb.embed(["hello"])
    assert emb.dim == 3
    assert len(requests) == 1


def test_openai_embedder_empty_input_has_zero_rows():
    requests: list[httpx.Request] = []
    emb = _openai(_embeddings_server(requests))
    emb.embed(["warm"])
    out = emb.embed([])
    assert out.shape == (0, 3)
    assert out.dtype == np.float32
    assert len(requests) == 1


def test_openai_embedder_without_key_sends_no_authorization():
    requests: list[httpx.Request] = []
    emb = OpenAIEmbedder(
        id="local",
        base_url="http://localhost:11434/v1/",
        model="nomic-embed-text",
        client=httpx.Client(transport=_embeddings_server(requests)),
    )
    emb.embed(["x"])
    assert "Authorization" not in requests[0].headers
    assert str(requests[0].url) == "http://localhost:11434/v1/embeddings"


def test_openai_embedder_close_is_idempotent_and_blocks_use():
    emb = OpenAIEmbedder(id="remote", base_url="http://127.0.0.1:9/v1", model="m", api_key=SECRET)
    assert isinstance(emb, Embedder)
    emb.close()
    emb.close()
    with pytest.raises(BackendError) as excinfo:
        emb.embed(["after close"])
    assert SECRET not in _error_chain_text(excinfo.value)


def test_openai_embedder_close_leaves_injected_client_open():
    client = httpx.Client(transport=_embeddings_server([]))
    emb = OpenAIEmbedder(id="remote", base_url="http://llm.local/v1", model="m", client=client)
    emb.close()
    assert not client.is_closed
    client.close()


def test_onnx_embedder_rejects_git_lfs_pointer(tmp_path):
    from noulo.inference.embedder import OnnxEmbedder
    from noulo.inference.types import ModelLoadError

    (tmp_path / "model.onnx").write_text("version https://git-lfs.github.com/spec/v1\n")
    (tmp_path / "tokenizer.json").write_text("{}")
    with pytest.raises(ModelLoadError, match="git lfs pull"):
        OnnxEmbedder(tmp_path, id="x")
