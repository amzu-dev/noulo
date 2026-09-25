import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from noulo.inference.causal_lm import ChatTemplate, LabelTokens, OnnxCausalLM, build_feeds
from noulo.inference.types import ModelLoadError

ROOT = Path(__file__).resolve().parent.parent
LLM_DIR = ROOT / "models" / "qwen3-0.6b-q4f16"
CHATML = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}"
    "<think>\n\n</think>\n\n{% endif %}"
    "{% endif %}"
)
MSGS = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]

# ---------------------------------------------------------------- chat templates


def test_chat_template_renders_generation_prompt_without_thinking():
    text = ChatTemplate({"chat_template": CHATML}).render(MSGS)
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "<|im_start|>system\nS<|im_end|>" in text


def test_chat_template_supplies_bos_token_given_as_dict():
    template = ChatTemplate(
        {
            "chat_template": "{{ bos_token }}{{ messages[0].content }}",
            "bos_token": {"content": "<bos>"},
        }
    )
    assert template.render(MSGS) == "<bos>S"


def test_chat_template_list_form_uses_default_entry():
    config = {
        "chat_template": [
            {"name": "tool_use", "template": "tools"},
            {"name": "default", "template": "default:{{ messages|length }}"},
        ]
    }
    assert ChatTemplate(config).render(MSGS) == "default:2"


def test_templates_without_system_role_get_it_folded_into_the_user_turn():
    strict = (
        "{% if messages[0].role == 'system' %}{{ raise_exception('System role not supported') }}"
        "{% endif %}{% for m in messages %}[{{ m.role }}]{{ m.content }}{% endfor %}"
    )
    assert ChatTemplate({"chat_template": strict}).render(MSGS) == "[user]S\n\nU"


def test_missing_chat_template_is_a_load_error():
    with pytest.raises(ModelLoadError, match="chat template"):
        ChatTemplate({})


# ---------------------------------------------------------------- label tokens


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer.from_file(str(ROOT / "models" / "minilm-l6-v2-int8" / "tokenizer.json"))


def test_label_tokens_collect_case_and_space_variants(tokenizer):
    ids = LabelTokens(tokenizer).ids("yes")
    assert ids and all(tokenizer.decode([i]).strip().lower() == "yes" for i in ids)


def test_label_distribution_sums_variant_mass_and_renormalises(tokenizer):
    labels = LabelTokens(tokenizer)
    probs = np.zeros(tokenizer.get_vocab_size())
    probs[labels.ids("yes")[0]] = 0.3
    probs[labels.ids("no")[0]] = 0.1
    dist, mass = labels.distribution(probs, ["yes", "no"])
    assert dist == pytest.approx([0.75, 0.25]) and mass == pytest.approx(0.4)


def test_zero_label_mass_gives_uniform_distribution(tokenizer):
    dist, mass = LabelTokens(tokenizer).distribution(
        np.zeros(tokenizer.get_vocab_size()), ["A", "B"]
    )
    assert dist == [0.5, 0.5] and mass == 0.0


# ---------------------------------------------------------------- session inputs

Input = namedtuple("Input", "name type shape")


def test_build_feeds_creates_empty_caches_for_any_architecture():
    inputs = [
        Input("input_ids", "tensor(int64)", ["batch_size", "sequence_length"]),
        Input("attention_mask", "tensor(int64)", ["batch_size", "past_sequence_length + 1"]),
        Input("position_ids", "tensor(int64)", ["batch_size", "sequence_length"]),
        Input(
            "past_key_values.0.key",
            "tensor(float16)",
            ["batch_size", 8, "past_sequence_length", 64],
        ),
        Input("past_conv.0", "tensor(float)", ["batch_size", 2048, 3]),
        Input("num_logits_to_keep", "tensor(int64)", []),
    ]
    feeds = build_feeds(inputs, [5, 6, 7])
    assert feeds["input_ids"].tolist() == [[5, 6, 7]]
    assert feeds["attention_mask"].shape == (1, 3) and feeds["position_ids"].tolist() == [[0, 1, 2]]
    assert feeds["past_key_values.0.key"].shape == (1, 8, 0, 64)
    assert feeds["past_key_values.0.key"].dtype == np.float16
    assert feeds["past_conv.0"].shape == (1, 2048, 3) and not feeds["past_conv.0"].any()
    assert int(feeds["num_logits_to_keep"]) == 1


# ---------------------------------------------------------------- loading


def test_missing_files_raise_load_error_without_paths(tmp_path):
    with pytest.raises(ModelLoadError) as excinfo:
        OnnxCausalLM(tmp_path / "nowhere")
    assert str(tmp_path) not in str(excinfo.value)


needs_llm = pytest.mark.skipif(not (LLM_DIR / "model.onnx").exists(), reason="LLM not downloaded")


@needs_llm
def test_real_llm_answers_with_labels():
    lm = OnnxCausalLM(LLM_DIR)
    dist, mass = lm.label_distribution(
        [{"role": "user", "content": 'Is the sky blue on a clear day? Reply "yes" or "no".'}],
        ["yes", "no"],
    )
    assert mass > 0.5 and dist[0] > dist[1]
    lm.close()


@needs_llm
def test_real_llm_chat_template_comes_from_the_model():
    config = json.loads((LLM_DIR / "tokenizer_config.json").read_text())
    assert "chat_template" in config


@needs_llm
def test_real_llm_accepts_a_placement_and_reports_its_device():
    from noulo.inference.devices import CPU_PLACEMENT

    lm = OnnxCausalLM(LLM_DIR, placement=CPU_PLACEMENT)
    assert lm.device == "cpu"
    lm.close()
