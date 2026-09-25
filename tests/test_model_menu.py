from noulo.model_menu import OWN_MODEL, facts, menu_items
from noulo.registry import CURATED_MODELS

LISTING = [
    {
        "id": CURATED_MODELS[0].id,
        "sizeMB": 87,
        "quantization": "INT8",
        "ramMB": 443,
        "noulAccuracy": 0.91,
        "choiceAccuracy": 0.675,
        "installed": True,
        "tier": "basic",
    },
    {
        "id": "big-one",
        "sizeMB": 739,
        "quantization": "FP32",
        "ramMB": 1234,
        "installed": False,
        "tier": "large",
        "backend": "onnx-nli",
    },
    {"id": "my-llm", "backend": "openai", "installed": True, "description": "Ollama", "tier": None},
]


def test_facts_show_size_quantisation_ram_accuracy_and_install_state():
    text = facts(LISTING[0])
    assert text == "87 MB INT8 · RAM 443 MB · Noul 91% · Choice 68% · ✓"
    assert facts(LISTING[1]).endswith("↓") and "RAM 1234 MB" in facts(LISTING[1])


def test_menu_orders_basic_then_larger_then_yours_then_plugin():
    items = menu_items(LISTING, active=CURATED_MODELS[0].id)
    values = [value for value, _, _ in items]
    assert values[:3] == [m.id for m in CURATED_MODELS]
    assert values.index("big-one") > 2 and values.index("my-llm") > values.index("big-one")
    assert values[-1] == OWN_MODEL
    assert "(current)" in items[0][1]


def test_menu_labels_fit_the_label_column():
    from noulo.registry import CATALOG

    labels = [c.label for c in CURATED_MODELS] + [e.label for e in CATALOG if e.label]
    assert all(len(label) <= 14 for label in labels)
