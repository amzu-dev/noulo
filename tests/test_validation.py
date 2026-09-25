import pytest

from noulo.api.validation import (
    ChoiceRequest,
    Limits,
    NoulRequest,
    RequestError,
    ScoreRequest,
    parse_request,
)

NOUL = {"input": "The invoice is unpaid.", "proposition": "The invoice is overdue."}
CHOICE = {
    "input": "Charged twice.",
    "question": "Which department?",
    "choices": [{"id": "A", "text": "Billing"}, {"id": "B", "text": "Sales"}],
}
SCORE = {"input": "Everything is down.", "question": "How severe?", "rubric": ["low", "high"]}


def error_of(primitive, payload, limits=None) -> RequestError:
    with pytest.raises(RequestError) as excinfo:
        parse_request(primitive, payload, limits or Limits())
    return excinfo.value


# ------------------------------------------------------------------ happy paths


def test_parses_noul_request():
    req = parse_request("noul", NOUL, Limits())
    assert isinstance(req, NoulRequest)
    assert req.proposition == "The invoice is overdue."


def test_parses_choice_request_preserving_option_order():
    req = parse_request("choice", CHOICE, Limits())
    assert isinstance(req, ChoiceRequest)
    assert [c.id for c in req.choices] == ["A", "B"]


def test_parses_score_request():
    req = parse_request("score", SCORE, Limits())
    assert isinstance(req, ScoreRequest)
    assert req.rubric == ["low", "high"]


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"type": "noul", **NOUL}, NoulRequest),
        ({"type": "choice", **CHOICE}, ChoiceRequest),
        ({"type": "score", **SCORE}, ScoreRequest),
    ],
)
def test_evaluate_dispatches_on_type(payload, expected):
    assert isinstance(parse_request("evaluate", payload, Limits()), expected)


def test_strings_are_stripped():
    req = parse_request("noul", {"input": "  hi  ", "proposition": " p "}, Limits())
    assert (req.input, req.proposition) == ("hi", "p")


def test_dedicated_endpoint_tolerates_matching_type_field():
    assert isinstance(parse_request("noul", {"type": "noul", **NOUL}, Limits()), NoulRequest)


# ------------------------------------------------------------------ rejections


@pytest.mark.parametrize("value", [None, "", "   "])
def test_rejects_missing_or_empty_input(value):
    payload = {**NOUL, "input": value} if value is not None else {"proposition": "p"}
    err = error_of("noul", payload)
    assert (err.status, err.code) == (422, "INVALID_REQUEST")
    assert err.message == "Input is required."


def test_rejects_missing_noul_proposition_with_spec_message():
    err = error_of("noul", {"input": "x"})
    assert err.message == "Noul requires a proposition."


def test_rejects_empty_noul_proposition():
    assert error_of("noul", {"input": "x", "proposition": " "}).message == (
        "Noul requires a proposition."
    )


def test_rejects_missing_choice_question():
    payload = {k: v for k, v in CHOICE.items() if k != "question"}
    assert error_of("choice", payload).message == "Choice requires a question."


@pytest.mark.parametrize("choices", [None, []])
def test_rejects_no_choices(choices):
    payload = (
        {**CHOICE, "choices": choices}
        if choices is not None
        else {k: v for k, v in CHOICE.items() if k != "choices"}
    )
    assert error_of("choice", payload).message == "Choice requires at least one choice."


def test_rejects_duplicate_choice_ids():
    payload = {**CHOICE, "choices": [{"id": "A", "text": "x"}, {"id": "A", "text": "y"}]}
    assert error_of("choice", payload).message == "Choice IDs must be unique (duplicate: 'A')."


def test_rejects_choice_with_empty_id_or_text():
    payload = {**CHOICE, "choices": [{"id": "", "text": "x"}]}
    assert error_of("choice", payload).code == "INVALID_REQUEST"
    payload = {**CHOICE, "choices": [{"id": "A", "text": " "}]}
    assert error_of("choice", payload).code == "INVALID_REQUEST"


def test_rejects_non_string_choice_id():
    payload = {**CHOICE, "choices": [{"id": 1, "text": "x"}]}
    assert error_of("choice", payload).code == "INVALID_REQUEST"


@pytest.mark.parametrize("rubric", [[], ["only"], ["low", ""], ["low", "low"], "low,high"])
def test_rejects_invalid_or_insufficient_rubric(rubric):
    err = error_of("score", {**SCORE, "rubric": rubric})
    assert err.code == "INVALID_REQUEST"
    assert "rubric" in err.message.lower()


def test_rejects_missing_score_question():
    payload = {k: v for k, v in SCORE.items() if k != "question"}
    assert error_of("score", payload).message == "Score requires a question."


@pytest.mark.parametrize("kind", ["magic", None, 42])
def test_evaluate_rejects_unsupported_primitive(kind):
    payload = {**NOUL} if kind is None else {"type": kind, **NOUL}
    err = error_of("evaluate", payload)
    assert (err.status, err.code) == (422, "UNSUPPORTED_PRIMITIVE")
    assert "choice, score, noul" in err.message


def test_rejects_non_object_body():
    err = error_of("noul", ["not", "an", "object"])
    assert err.code == "INVALID_REQUEST"


def test_rejects_input_over_character_limit_with_413():
    err = error_of("noul", {**NOUL, "input": "x" * 101}, Limits(max_input_chars=100))
    assert (err.status, err.code) == (413, "INPUT_TOO_LARGE")


def test_rejects_too_many_choices():
    choices = [{"id": str(i), "text": f"option {i}"} for i in range(4)]
    err = error_of("choice", {**CHOICE, "choices": choices}, Limits(max_choices=3))
    assert err.code == "INVALID_REQUEST"


def test_rejects_too_many_rubric_levels():
    err = error_of("score", {**SCORE, "rubric": list("abcd")}, Limits(max_rubric_levels=3))
    assert err.code == "INVALID_REQUEST"


def test_rejects_overlong_proposition():
    err = error_of("noul", {**NOUL, "proposition": "p" * 51}, Limits(max_text_chars=50))
    assert err.status == 413


def test_error_serialises_to_spec_shape():
    err = error_of("noul", {"input": "x"})
    assert err.to_dict() == {
        "error": {"code": "INVALID_REQUEST", "message": "Noul requires a proposition."}
    }
