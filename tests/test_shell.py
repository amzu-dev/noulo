import io

from noulo.shell import Shell


def make_shell(lines, dispatch_log):
    out = io.StringIO()
    feed = iter(lines)

    def ask(prompt):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    def dispatch(argv):
        dispatch_log.append(argv)
        return 0

    return Shell(dispatch=dispatch, ask=ask, stdout=out), out


def test_slash_commands_dispatch_to_cli():
    log = []
    shell, _ = make_shell(["/model", "/learning off", "/status", "/quit"], log)
    shell.loop()
    assert log == [["model"], ["learning", "off"], ["status"]]


def test_help_lists_commands():
    shell, out = make_shell(["/help", "/quit"], [])
    shell.loop()
    text = out.getvalue()
    for command in (
        "/model",
        "/start",
        "/stop",
        "/restart",
        "/learning",
        "/noul",
        "/choice",
        "/score",
    ):
        assert command in text


def test_noul_prompts_for_missing_fields():
    log = []
    shell, _ = make_shell(["/noul", "Invoice unpaid 120 days", "Payment is overdue", "/quit"], log)
    shell.loop()
    assert log == [
        ["noul", "--input", "Invoice unpaid 120 days", "--proposition", "Payment is overdue"]
    ]


def test_choice_prompts_for_options():
    log = []
    shell, _ = make_shell(
        ["/choice", "Charged twice", "Which team?", "A=Billing, B=Sales", "/quit"], log
    )
    shell.loop()
    assert log == [
        [
            "choice",
            "--input",
            "Charged twice",
            "--question",
            "Which team?",
            "--option",
            "A=Billing",
            "--option",
            "B=Sales",
        ]
    ]


def test_score_prompts_for_rubric():
    log = []
    shell, _ = make_shell(["/score", "All down", "How severe?", "low, medium, high", "/quit"], log)
    shell.loop()
    assert log == [
        ["score", "--input", "All down", "--question", "How severe?", "--rubric", "low,medium,high"]
    ]


def test_plain_text_suggests_help_and_unknown_commands_are_reported():
    shell, out = make_shell(["hello", "/nope", "/quit"], [])
    shell.loop()
    assert "/help" in out.getvalue() and "Unknown command" in out.getvalue()


def test_eof_exits_cleanly():
    shell, _ = make_shell([], [])
    assert shell.loop() == 0
