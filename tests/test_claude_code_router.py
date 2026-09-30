"""The Claude Code routing plugin in examples/claude-code-router (a stdlib-only script)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PLUGIN = Path(__file__).resolve().parent.parent / "examples" / "claude-code-router"
_spec = importlib.util.spec_from_file_location(
    "noulo_router", PLUGIN / "scripts" / "noulo_router.py"
)
router = importlib.util.module_from_spec(_spec)
sys.modules["noulo_router"] = router
_spec.loader.exec_module(router)


@pytest.fixture
def policy():
    return router.load_policy(PLUGIN / "router.json")


def signals(task="debug", confidence=0.9, large=0.1, security=0.1):
    return router.Signals(
        task=task, confidence=confidence, large=large, security=security, record_ids={}
    )


# ---------------------------------------------------------------- policy


@pytest.mark.parametrize(
    ("task", "tier"),
    [
        ("explain", "light"),
        ("edit", "light"),
        ("chore", "light"),
        ("refactor", "standard"),
        ("feature", "deep"),
        ("debug", "deep"),
        ("design", "deep"),
        ("review", "deep"),
    ],
)
def test_task_type_sets_the_tier(policy, task, tier):
    assert policy.decide(signals(task=task)) == tier


def test_a_deep_task_that_needs_substantial_work_goes_to_max(policy):
    assert policy.decide(signals(task="design", large=policy.max_at)) == "max"


def test_substantial_work_only_raises_deep_tasks(policy):
    assert policy.decide(signals(task="explain", large=0.99)) == "light"
    assert policy.decide(signals(task="refactor", large=0.99)) == "standard"


def test_security_work_is_at_least_deep(policy):
    assert policy.decide(signals(task="edit", security=policy.security_at)) == "deep"
    assert policy.decide(signals(task="edit", security=policy.security_at - 0.01)) == "light"


def test_an_unsure_task_type_falls_back_to_standard(policy):
    unsure = policy.min_confidence - 0.01
    assert policy.decide(signals(task="edit", confidence=unsure)) == "standard"
    assert policy.decide(signals(task="debug", confidence=unsure)) == "standard"


def test_every_tier_names_a_model_and_an_effort(policy):
    assert list(policy.tiers) == ["light", "standard", "deep", "max"]
    assert policy.tiers["max"] == {"model": "fable", "effort": "xhigh"}
    assert set(policy.task_tiers) == set(policy.tasks)


# ---------------------------------------------------------------- a fake noulo server


class FakeNoulo:
    """A tiny HTTP server that answers like noulo and records what it was sent."""

    def __init__(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.requests = []
        self.task = ("debug", 0.82)
        self.noul = {}  # proposition -> value
        self.status = 200
        self.healthy = True
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, code, body, headers=()):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in headers:
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                fake.requests.append({"method": "GET", "path": self.path, "headers": self.headers})
                if self.path == "/health" and fake.healthy:
                    self._reply(200, {"status": "ok", "modelLoaded": True})
                else:
                    self._reply(503, {"status": "starting", "modelLoaded": False})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                n = len(fake.requests)
                fake.requests.append(
                    {"method": "POST", "path": self.path, "headers": self.headers, "body": body}
                )
                if fake.status == 409:
                    err = {"error": {"code": "LEARNING_DISABLED", "message": "Learning is off."}}
                    return self._reply(409, err)
                if fake.status != 200:
                    err = {"error": {"code": "ENGINE_ERROR", "message": "boom"}}
                    return self._reply(fake.status, err)
                if self.path == "/api/v1/feedback":
                    return self._reply(200, {"recordId": body.get("recordId"), "verified": True})
                if self.path == "/api/v1/learning/import":
                    return self._reply(200, {"imported": len(body["items"]), "failed": []})
                rid = [("X-Record-Id", f"rec-{n}")]
                if body["type"] == "choice":
                    task, p = fake.task
                    probs = {c["id"]: (p if c["id"] == task else 0.01) for c in body["choices"]}
                    diag = {"probabilities": probs}
                    return self._reply(
                        200, {"type": "choice", "value": task, "diagnostics": diag}, rid
                    )
                value = fake.noul.get(body["proposition"], 0.05)
                return self._reply(200, {"type": "noul", "value": value}, rid)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self.thread.start()

    def posts(self, path="/api/v1/evaluate"):
        return [r for r in self.requests if r["method"] == "POST" and r["path"].startswith(path)]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def noulo():
    fake = FakeNoulo()
    yield fake
    fake.close()


def closed_port_url():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


# ---------------------------------------------------------------- client and classifier


def test_classify_asks_noulo_for_task_type_size_and_security(noulo, policy):
    noulo.task = ("review", 0.7)
    noulo.noul = {policy.large_work: 0.6, policy.security: 0.95}
    client = router.NouloClient(noulo.url)

    s = router.classify(client, "Audit the login code for timing attacks", policy)

    assert (s.task, s.confidence, s.large, s.security) == ("review", 0.7, 0.6, 0.95)
    choice, large, security = (r["body"] for r in noulo.posts())
    assert choice["question"] == policy.question
    assert [c["id"] for c in choice["choices"]] == list(policy.tasks)
    assert (large["proposition"], security["proposition"]) == (policy.large_work, policy.security)
    assert all(r["path"] == "/api/v1/evaluate?diagnostics=true" for r in noulo.posts())


def test_classify_keeps_the_record_ids_for_feedback(noulo, policy):
    s = router.classify(router.NouloClient(noulo.url), "Fix the flaky test", policy)
    assert s.record_ids == {"task": "rec-0", "large": "rec-1", "security": "rec-2"}


def test_the_api_key_is_sent_as_a_bearer_token(noulo, policy):
    router.classify(router.NouloClient(noulo.url, api_key="s3cret"), "Fix it properly", policy)
    assert noulo.posts()[0]["headers"]["Authorization"] == "Bearer s3cret"


def test_nothing_listening_raises_unavailable(policy):
    client = router.NouloClient(closed_port_url())
    with pytest.raises(router.NouloUnavailable):
        router.classify(client, "Fix the flaky test", policy)


def test_an_error_response_raises_unavailable(noulo, policy):
    noulo.status = 500
    with pytest.raises(router.NouloUnavailable, match="boom"):
        router.classify(router.NouloClient(noulo.url), "Fix the flaky test", policy)


def test_healthy_reflects_the_health_endpoint(noulo):
    assert router.NouloClient(noulo.url).healthy()
    noulo.healthy = False
    assert not router.NouloClient(noulo.url).healthy()
    assert not router.NouloClient(closed_port_url()).healthy()


def test_noulo_only_sees_the_start_and_end_of_a_long_prompt(noulo, policy):
    # Latency grows with input length (≈6 s for 1,500 characters on the base model).
    prompt = "Fix this crash:\n" + "traceback line\n" * 500 + "KeyError: 'payload'"
    router.classify(router.NouloClient(noulo.url), prompt, policy)
    for request in noulo.posts():
        assert len(request["body"]["input"]) <= 460
        assert request["body"]["input"].startswith("Fix this crash:")
        assert request["body"]["input"].endswith("KeyError: 'payload'")


def test_long_prompts_keep_their_start_and_end():
    text = "A" * 2000 + "B" * 1000
    trimmed = router.trim(text, head=1200, tail=300)
    assert trimmed.startswith("A" * 1200) and trimmed.endswith("B" * 300)
    assert len(trimmed) < 1600
    assert router.trim("short prompt") == "short prompt"


# ---------------------------------------------------------------- settings and state


def test_settings_default_to_auto_mode_on_the_router_port(tmp_path):
    s = router.Settings.from_env(
        {"CLAUDE_PLUGIN_ROOT": str(PLUGIN), "CLAUDE_PLUGIN_DATA": str(tmp_path)}
    )
    assert (s.mode, s.url, s.api_key) == ("auto", "http://127.0.0.1:8790", None)
    assert s.state_file == tmp_path / "state.json"
    assert s.policy.tiers["deep"]["model"] == "opus"


def test_settings_come_from_the_plugin_options(tmp_path):
    env = {
        "CLAUDE_PLUGIN_ROOT": str(PLUGIN),
        "CLAUDE_PLUGIN_DATA": str(tmp_path),
        "CLAUDE_PLUGIN_OPTION_MODE": "suggest",
        "CLAUDE_PLUGIN_OPTION_URL": "http://127.0.0.1:9999/",
        "NOULO_ROUTER_API_KEY": "k",
    }
    s = router.Settings.from_env(env)
    assert (s.mode, s.url, s.api_key) == ("suggest", "http://127.0.0.1:9999", "k")


def test_an_unknown_mode_means_auto(tmp_path):
    env = {"CLAUDE_PLUGIN_DATA": str(tmp_path), "CLAUDE_PLUGIN_OPTION_MODE": "sometimes"}
    assert router.Settings.from_env(env).mode == "auto"


def test_state_is_kept_per_session_across_runs(tmp_path):
    router.StateStore(tmp_path / "state.json").update("s1", tier="deep")
    router.StateStore(tmp_path / "state.json").update("s2", tier="light")
    store = router.StateStore(tmp_path / "state.json")
    assert store.get("s1")["tier"] == "deep"
    assert store.get("s2")["tier"] == "light"
    assert store.get("unknown") == {}


def test_state_keeps_only_recent_sessions(tmp_path):
    store = router.StateStore(tmp_path / "state.json", max_sessions=3)
    for i in range(5):
        store.update(f"s{i}", tier="light")
    kept = json.loads((tmp_path / "state.json").read_text())
    assert sorted(kept) == ["s2", "s3", "s4"]


# ---------------------------------------------------------------- the UserPromptSubmit hook


@pytest.fixture
def settings(noulo, tmp_path):
    def make(mode="auto", url=None):
        env = {
            "CLAUDE_PLUGIN_ROOT": str(PLUGIN),
            "CLAUDE_PLUGIN_DATA": str(tmp_path),
            "CLAUDE_PLUGIN_OPTION_MODE": mode,
            "CLAUDE_PLUGIN_OPTION_URL": url or noulo.url,
        }
        return router.Settings.from_env(env)

    return make


def prompt_event(prompt, session="sess-1"):
    return {"session_id": session, "hook_event_name": "UserPromptSubmit", "prompt": prompt}


def test_auto_mode_tells_claude_when_to_delegate(noulo, settings):
    noulo.task = ("debug", 0.82)
    out = router.route_prompt(
        prompt_event("The upload test fails with a KeyError, fix it"), settings()
    )

    context = out["hookSpecificOutput"]["additionalContext"]
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "`noulo-router:deep-agent`" in context
    assert "haiku < sonnet < opus < fable" in context
    assert "weaker model than opus" in context
    assert out["systemMessage"] == "noulo-router → deep · opus · high (debug, 82%)"


def test_a_light_task_is_handled_inline_with_a_hint(noulo, settings):
    noulo.task = ("edit", 0.64)
    out = router.route_prompt(prompt_event("Fix the typo in the README heading"), settings())

    assert "hookSpecificOutput" not in out
    assert out["systemMessage"].startswith("noulo-router → light · haiku · low (edit, 64%)")
    assert "/noulo-router:light" in out["systemMessage"]


def test_suggest_mode_only_names_the_command(noulo, settings):
    noulo.task = ("design", 0.9)
    noulo.noul = {router.load_policy(PLUGIN / "router.json").large_work: 0.7}
    out = router.route_prompt(prompt_event("Plan a multi-tenant architecture"), settings("suggest"))

    assert "hookSpecificOutput" not in out
    assert "max · fable · xhigh" in out["systemMessage"]
    assert "/noulo-router:max" in out["systemMessage"]


def test_off_mode_does_nothing(noulo, settings):
    assert router.route_prompt(prompt_event("Fix the flaky test please"), settings("off")) is None
    assert noulo.requests == []


def test_slash_commands_are_not_routed(noulo, settings):
    assert router.route_prompt(prompt_event("/noulo-router:deep fix it"), settings()) is None
    assert noulo.requests == []


@pytest.mark.parametrize(
    "prompt",
    [
        "<task-notification>\n<task-id>a9ce</task-id>\n<status>completed</status>\n</task-notification>",
        "<system-reminder>The user has not replied yet.</system-reminder>",
    ],
)
def test_messages_claude_code_generates_are_not_routed(noulo, settings, prompt):
    assert router.route_prompt(prompt_event(prompt), settings()) is None
    assert noulo.requests == []


def test_a_slow_noulo_is_reported_as_slow_not_missing(noulo, settings):
    s = settings()
    s.timeout = 0.0
    out = router.route_prompt(prompt_event("Fix the flaky upload test"), s)
    assert "took too long" in out["systemMessage"]
    assert "isn't reachable" not in out["systemMessage"]


def test_a_short_follow_up_reuses_the_sessions_last_tier(noulo, settings):
    s = settings()
    router.route_prompt(prompt_event("The upload test fails with a KeyError, fix it"), s)
    calls = len(noulo.requests)

    out = router.route_prompt(prompt_event("yes, go ahead"), s)

    assert len(noulo.requests) == calls
    assert out["systemMessage"] == "noulo-router → deep · opus · high (follow-up)"
    assert "`noulo-router:deep-agent`" in out["hookSpecificOutput"]["additionalContext"]


def test_a_short_first_prompt_is_not_routed(noulo, settings):
    assert router.route_prompt(prompt_event("hello"), settings()) is None
    assert noulo.requests == []


def test_the_decision_is_remembered_for_why_and_correct(noulo, settings):
    s = settings()
    router.route_prompt(prompt_event("The upload test fails with a KeyError, fix it"), s)

    last = router.StateStore(s.state_file).get("sess-1")["last"]
    assert last["tier"] == "deep"
    assert last["task"] == "debug"
    assert last["record_ids"]["task"] == "rec-0"
    assert last["url"] == noulo.url
    assert last["prompt"].startswith("The upload test fails")


def test_unreachable_noulo_warns_once_per_session_and_lets_the_prompt_through(settings):
    s = settings(url=closed_port_url())
    first = router.route_prompt(prompt_event("Fix the flaky upload test"), s)
    second = router.route_prompt(prompt_event("Fix the flaky upload test"), s)

    assert "isn't reachable" in first["systemMessage"]
    assert "--port" in first["systemMessage"]
    assert "hookSpecificOutput" not in first
    assert second is None


# ---------------------------------------------------------------- the SessionStart hook


def start_event(session="sess-1"):
    return {"session_id": session, "hook_event_name": "SessionStart", "source": "startup"}


def test_session_start_is_quiet_when_noulo_is_up(noulo, settings):
    assert router.session_start(start_event(), settings()) is None


def test_session_start_warns_when_noulo_is_down(settings):
    s = settings(url=closed_port_url())
    out = router.session_start(start_event(), s)

    assert "isn't reachable" in out["systemMessage"]
    assert router.route_prompt(prompt_event("Fix the flaky upload test"), s) is None


def test_session_start_is_quiet_when_routing_is_off(settings):
    assert router.session_start(start_event(), settings("off", url=closed_port_url())) is None


# ---------------------------------------------------------------- command line


def run(args, env, stdin=""):
    import io

    out = io.StringIO()
    code = router.main(args, stdin=io.StringIO(stdin), stdout=out, env=env)
    return code, out.getvalue()


@pytest.fixture
def env(noulo, tmp_path):
    return {
        "CLAUDE_PLUGIN_ROOT": str(PLUGIN),
        "CLAUDE_PLUGIN_DATA": str(tmp_path),
        "CLAUDE_PLUGIN_OPTION_URL": noulo.url,
    }


def test_the_prompt_hook_reads_the_event_and_prints_json(env):
    event = json.dumps(prompt_event("The upload test fails with a KeyError, fix it"))
    code, out = run(["hook", "prompt"], env, event)
    assert code == 0
    assert json.loads(out)["systemMessage"].startswith("noulo-router → deep")


def test_the_hook_prints_nothing_when_there_is_nothing_to_say(env):
    assert run(["hook", "prompt"], env, json.dumps(prompt_event("/help"))) == (0, "")


@pytest.mark.parametrize("stdin", ["", "not json", "[]", '{"prompt": 42}'])
def test_the_hook_never_blocks_a_prompt(env, stdin):
    assert run(["hook", "prompt"], env, stdin) == (0, "")


def test_the_session_start_hook_warns_when_noulo_is_down(env):
    env["CLAUDE_PLUGIN_OPTION_URL"] = closed_port_url()
    code, out = run(["hook", "session-start"], env, json.dumps(start_event()))
    assert code == 0
    assert "isn't reachable" in json.loads(out)["systemMessage"]


def test_why_explains_the_last_decision(env, tmp_path):
    run(["hook", "prompt"], env, json.dumps(prompt_event("The upload test fails with a KeyError")))
    code, out = run(["why", "--session", "sess-1", "--data", str(tmp_path)], {})
    assert code == 0
    assert "deep · opus · high" in out
    assert "debug (82%)" in out
    assert "The upload test fails with a KeyError" in out


def test_why_without_a_decision_says_so(tmp_path):
    code, out = run(["why", "--session", "nope", "--data", str(tmp_path)], {})
    assert code == 0
    assert "No routing decision" in out


def test_correct_teaches_noulo_the_right_task_type(noulo, env, tmp_path):
    run(["hook", "prompt"], env, json.dumps(prompt_event("The upload test fails with a KeyError")))
    code, out = run(["correct", "--session", "sess-1", "--data", str(tmp_path), "feature"], {})

    assert code == 0
    assert noulo.posts("/api/v1/feedback")[-1]["body"] == {
        "recordId": "rec-0",
        "expected": "feature",
    }
    assert "feature" in out and "deep" in out


def test_correct_can_mark_work_as_large_and_security_related(noulo, env, tmp_path):
    run(["hook", "prompt"], env, json.dumps(prompt_event("The upload test fails with a KeyError")))
    run(["correct", "--session", "sess-1", "--data", str(tmp_path), "large", "no-security"], {})

    bodies = [r["body"] for r in noulo.posts("/api/v1/feedback")]
    assert bodies == [
        {"recordId": "rec-1", "expected": True},
        {"recordId": "rec-2", "expected": False},
    ]


def test_correct_rejects_unknown_words(env, tmp_path):
    run(["hook", "prompt"], env, json.dumps(prompt_event("The upload test fails with a KeyError")))
    code, out = run(["correct", "--session", "sess-1", "--data", str(tmp_path), "urgent"], {})
    assert code == 2
    assert "explain" in out and "large" in out


def test_correct_explains_that_learning_must_be_on(noulo, env, tmp_path):
    run(["hook", "prompt"], env, json.dumps(prompt_event("The upload test fails with a KeyError")))
    noulo.status = 409
    code, out = run(["correct", "--session", "sess-1", "--data", str(tmp_path), "feature"], {})
    assert code == 1
    assert "Learning is off" in out


def labelled(tmp_path, rows):
    path = tmp_path / "labelled.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


ROWS = [
    {"prompt": "Fix the crash", "task": "debug", "complexity": "complex", "tier": "deep"},
    {"prompt": "Fix the typo", "task": "edit", "complexity": "trivial", "tier": "light"},
    {"prompt": "Tidy the tests", "task": "refactor", "complexity": "moderate", "tier": "standard"},
]


def test_teach_imports_labelled_prompts_as_verified_examples(noulo, tmp_path):
    code, out = run(["teach", str(labelled(tmp_path, ROWS)), "--url", noulo.url], {})

    items = noulo.posts("/api/v1/learning/import")[0]["body"]["items"]
    assert code == 0
    assert [i["expected"] for i in items if i["type"] == "choice"] == ["debug", "edit", "refactor"]
    policy = router.load_policy(PLUGIN / "router.json")
    large = [
        (i["input"], i["expected"]) for i in items if i.get("proposition") == policy.large_work
    ]
    assert large == [("Fix the crash", True), ("Fix the typo", False)]  # moderate isn't taught
    assert "Taught 5 examples" in out


def test_eval_reports_task_and_tier_accuracy(noulo, tmp_path):
    noulo.task = ("debug", 0.9)  # the fake always says debug
    code, out = run(["eval", str(labelled(tmp_path, ROWS)), "--url", noulo.url, "--json"], {})

    report = json.loads(out)
    assert code == 0
    assert report["items"] == 3
    assert report["task_accuracy"] == pytest.approx(1 / 3)
    assert report["tier_accuracy"] == pytest.approx(1 / 3)
    assert report["confusion"]["edit"] == {"debug": 1}
    assert report["latency_ms"]["p50"] >= 0


# ---------------------------------------------------------------- the plugin's files


def frontmatter(path):
    text = path.read_text()
    assert text.startswith("---\n"), path
    head, body = text[4:].split("\n---\n", 1)
    fields = dict(line.split(": ", 1) for line in head.splitlines() if ": " in line)
    return fields, body


def test_the_manifest_offers_mode_and_url_options():
    manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
    assert manifest["name"] == "noulo-router"
    mode = manifest["userConfig"]["mode"]
    assert (mode["options"], mode["default"]) == (list(router.MODES), "auto")
    assert manifest["userConfig"]["url"]["default"] == router.DEFAULT_URL


def test_the_hooks_run_the_script():
    hooks = json.loads((PLUGIN / "hooks" / "hooks.json").read_text())["hooks"]
    script = '"${CLAUDE_PLUGIN_ROOT}/scripts/noulo_router.py"'
    for event, kind in (("UserPromptSubmit", "prompt"), ("SessionStart", "session-start")):
        (hook,) = hooks[event][0]["hooks"]
        assert hook["command"] == f"python3 {script} hook {kind}"
        assert hook["timeout"] >= 5


def test_each_stronger_tier_has_a_subagent_with_its_model_and_effort(policy):
    agents = {p.stem for p in (PLUGIN / "agents").glob("*.md")}
    delegable = [t for t in policy.tiers if policy.tiers[t]["model"] != router.MODEL_ORDER[0]]
    assert agents == {f"{t}-agent" for t in delegable}
    for tier in delegable:
        fields, _ = frontmatter(PLUGIN / "agents" / f"{tier}-agent.md")
        assert fields["name"] == f"{tier}-agent"
        assert (fields["model"], fields["effort"]) == tuple(policy.tiers[tier].values())


def test_each_tier_has_a_slash_command_with_its_model_and_effort(policy):
    for tier, spec in policy.tiers.items():
        fields, body = frontmatter(PLUGIN / "skills" / tier / "SKILL.md")
        assert (fields["model"], fields["effort"]) == (spec["model"], spec["effort"])
        # Claude invoking these itself doesn't switch the model, so only people may run them.
        assert fields["disable-model-invocation"] == "true"
        assert "$ARGUMENTS" in body


def test_the_repo_is_a_marketplace_for_the_plugin():
    root = PLUGIN.parent.parent
    market = json.loads((root / ".claude-plugin" / "marketplace.json").read_text())
    (entry,) = market["plugins"]
    assert entry["name"] == "noulo-router"
    assert (root / entry["source"]).resolve() == PLUGIN


def test_the_eval_prompts_are_never_taught():
    seed = {r["prompt"] for r in router.read_labelled(PLUGIN / "data" / "seed.jsonl")}
    held_out = {r["prompt"] for r in router.read_labelled(PLUGIN / "data" / "eval.jsonl")}
    assert seed and held_out
    assert not seed & held_out


# ---------------------------------------------------------------- the event log (for status)


def events(settings):
    path = settings.events_file
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_each_routed_prompt_is_logged_with_its_session_and_transcript(noulo, settings):
    s = settings()
    event = prompt_event("The upload test fails with a KeyError, fix it")
    event.update(transcript_path="/tmp/t/sess-1.jsonl", cwd="/work/proj")
    router.route_prompt(event, s)

    (logged,) = events(s)
    assert logged["event"] == "route"
    assert logged["outcome"] == "routed"
    assert (logged["session_id"], logged["transcript_path"]) == ("sess-1", "/tmp/t/sess-1.jsonl")
    assert (logged["tier"], logged["model"], logged["effort"]) == ("deep", "opus", "high")
    assert (logged["task"], logged["confidence"]) == ("debug", 0.82)
    assert logged["cwd"] == "/work/proj"
    assert logged["noulo_ms"] >= 0
    assert logged["prompt"].startswith("The upload test fails")
    assert logged["ts"].endswith("Z")


def test_follow_ups_and_unreachable_noulo_are_logged(noulo, settings):
    s = settings()
    router.route_prompt(prompt_event("The upload test fails with a KeyError, fix it"), s)
    router.route_prompt(prompt_event("yes, go ahead"), s)
    down = settings(url=closed_port_url())
    router.route_prompt(prompt_event("Fix the flaky upload test", session="sess-2"), down)

    assert [e["outcome"] for e in events(s)] == ["routed", "follow-up", "unreachable"]
    assert events(s)[1]["tier"] == "deep"


def test_skipped_prompts_are_not_logged(noulo, settings):
    s = settings()
    router.route_prompt(prompt_event("/noulo-router:why"), s)
    router.route_prompt(prompt_event("<task-notification>done</task-notification>"), s)
    assert events(s) == []


def test_session_start_is_logged(noulo, settings):
    s = settings()
    event = start_event()
    event.update(transcript_path="/tmp/t/sess-1.jsonl", cwd="/work/proj")
    router.session_start(event, s)

    (logged,) = events(s)
    assert (logged["event"], logged["session_id"], logged["noulo_healthy"]) == (
        "session",
        "sess-1",
        True,
    )
    assert logged["transcript_path"] == "/tmp/t/sess-1.jsonl"


def test_the_event_log_keeps_the_newest_events_when_it_grows(tmp_path):
    log = router.EventLog(tmp_path / "events.jsonl", max_bytes=2000)
    for i in range(100):
        log.append({"event": "route", "n": i})
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert (tmp_path / "events.jsonl").stat().st_size <= 2000
    assert json.loads(lines[-1])["n"] == 99
    assert all(json.loads(line) for line in lines)


# ---------------------------------------------------------------- /noulo-router:status


def assistant(ts, msg_id, model, inp=2, write=100, read=1000, out=50, stop="end_turn"):
    usage = {
        "input_tokens": inp,
        "cache_creation_input_tokens": write,
        "cache_read_input_tokens": read,
        "output_tokens": out,
    }
    message = {"id": msg_id, "model": model, "usage": usage, "stop_reason": stop}
    return {"type": "assistant", "timestamp": ts, "message": message}


def write_session(tmp_path, session="sess-1", main=(), agents=None, partial=""):
    """A transcript laid out like Claude Code's: <session>.jsonl plus <session>/subagents/."""
    main_path = tmp_path / f"{session}.jsonl"
    main_path.write_text("".join(json.dumps(line) + "\n" for line in main) + partial)
    for agent_id, (agent_type, lines) in (agents or {}).items():
        folder = tmp_path / session / "subagents"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"agent-{agent_id}.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in lines)
        )
        (folder / f"agent-{agent_id}.meta.json").write_text(json.dumps({"agentType": agent_type}))
    return main_path


def test_calls_are_read_from_the_main_transcript_and_its_subagents(tmp_path):
    path = write_session(
        tmp_path,
        main=[
            assistant("2026-09-30T10:00:01.000Z", "m1", "claude-sonnet-5", out=40),
            assistant("2026-09-30T10:00:01.200Z", "m1", "claude-sonnet-5", out=40),  # same call
            {"type": "user", "timestamp": "2026-09-30T10:00:02.000Z", "message": {"content": "hi"}},
            assistant("2026-09-30T10:00:03.000Z", "m2", "<synthetic>", out=0),
        ],
        agents={
            "a1": (
                "noulo-router:deep-agent",
                [
                    assistant("2026-09-30T10:00:05.000Z", "s1", "claude-opus-5-5", out=300),
                ],
            )
        },
        partial='{"type": "assistant", "timestamp": "2026-09-30T10:00:0',  # still being written
    )
    calls = router.read_calls(path)

    assert [(c.model, c.agent, c.output) for c in calls] == [
        ("claude-sonnet-5", "main", 40),
        ("claude-opus-5-5", "noulo-router:deep-agent", 300),
    ]
    assert (calls[0].input, calls[0].cache_write, calls[0].cache_read) == (2, 100, 1000)


@pytest.mark.parametrize(
    ("model_id", "pretty"),
    [
        ("claude-opus-5-5", "opus 5.5"),
        ("claude-sonnet-5", "sonnet 5"),
        ("claude-haiku-4-5-20251001", "haiku 4.5"),
        ("claude-fable-5-1", "fable 5.1"),
        ("gpt-9", "gpt-9"),
    ],
)
def test_model_ids_are_shown_short(model_id, pretty):
    assert router.pretty_model(model_id) == pretty


@pytest.fixture
def routed_session(noulo, settings, tmp_path):
    """Two routed prompts in one session, with the calls Claude made after each."""
    s = settings()
    path = write_session(
        tmp_path,
        main=[
            assistant("2026-09-30T10:00:05.000Z", "m1", "claude-sonnet-5", out=500),
            assistant("2026-09-30T10:00:20.000Z", "m2", "claude-sonnet-5", out=100),
            assistant("2026-09-30T10:05:05.000Z", "m3", "claude-sonnet-5", out=30),
        ],
        agents={
            "a1": (
                "noulo-router:deep-agent",
                [
                    assistant("2026-09-30T10:00:10.000Z", "s1", "claude-opus-5-5", out=1200),
                    assistant("2026-09-30T10:00:15.000Z", "s2", "claude-opus-5-5", out=800),
                ],
            )
        },
    )
    log = router.EventLog(s.events_file)
    common = {
        "event": "route",
        "session_id": "sess-1",
        "transcript_path": str(path),
        "cwd": "/work/calc",
        "mode": "auto",
        "outcome": "routed",
        "noulo_ms": 400,
    }
    log.append(
        {
            **common,
            "ts": "2026-09-30T10:00:00.000Z",
            "prompt": "test_add fails, fix it",
            "tier": "deep",
            "model": "opus",
            "effort": "high",
            "task": "debug",
            "confidence": 0.64,
        }
    )
    log.append(
        {
            **common,
            "ts": "2026-09-30T10:05:00.000Z",
            "prompt": "Fix the typo",
            "tier": "light",
            "model": "haiku",
            "effort": "low",
            "task": "edit",
            "confidence": 0.92,
        }
    )
    return s, path


NOW = router.parse_ts("2026-09-30T10:06:00.000Z")


def test_status_totals_tokens_per_model_and_where_it_ran(routed_session):
    s, _ = routed_session
    report = router.status_report(s, session_id="sess-1", now=NOW)

    lines = report.splitlines()
    sonnet = next(line for line in lines if line.strip().startswith("sonnet 5 · main"))
    opus = next(line for line in lines if line.strip().startswith("opus 5.5 · deep-agent"))
    total = next(line for line in lines if line.strip().startswith("total"))
    assert sonnet.split()[-5:] == ["3", "6", "300", "3,000", "630"]
    assert opus.split()[-5:] == ["2", "4", "200", "2,000", "2,000"]
    assert total.split()[-5:] == ["5", "10", "500", "5,000", "2,630"]


def test_status_shows_which_models_ran_after_each_prompt(routed_session):
    s, _ = routed_session
    report = router.status_report(s, session_id="sess-1", now=NOW)

    lines = report.splitlines()
    typo = next(i for i, line in enumerate(lines) if "Fix the typo" in line)
    bug = next(i for i, line in enumerate(lines) if "test_add fails, fix it" in line)
    assert typo < bug  # newest first
    assert "light · haiku · low" in lines[typo]
    assert "sonnet 5 main ×1 · 30 out" in lines[typo + 1]
    assert "deep · opus · high  (debug 64%)" in lines[bug]
    assert "sonnet 5 main ×2 · 600 out" in lines[bug + 1]
    assert "opus 5.5 deep-agent ×2 · 2,000 out" in lines[bug + 1]


def test_status_shows_noulos_health_model_and_speed(routed_session, noulo):
    s, _ = routed_session
    report = router.status_report(s, session_id="sess-1", now=NOW)
    noulo_line = next(line for line in report.splitlines() if line.startswith("noulo "))
    assert "up" in noulo_line
    assert "2 prompts routed" in noulo_line
    assert "p50 400 ms" in noulo_line


def test_status_says_when_noulo_is_down(settings):
    report = router.status_report(settings(url=closed_port_url()), session_id="none", now=NOW)
    assert "not reachable" in report
    assert "No routed prompts" in report


def test_status_all_covers_every_session_in_the_window(routed_session, tmp_path):
    s, _ = routed_session
    old = write_session(
        tmp_path,
        session="old",
        main=[assistant("2026-09-28T09:00:00.000Z", "o1", "claude-opus-5-5", out=999)],
    )
    router.EventLog(s.events_file).append(
        {
            "event": "route",
            "ts": "2026-09-28T09:00:00.000Z",
            "session_id": "old",
            "transcript_path": str(old),
            "cwd": "/work/old",
            "outcome": "routed",
            "tier": "deep",
            "model": "opus",
            "effort": "high",
            "task": "design",
            "confidence": 0.9,
            "prompt": "Old one",
        }
    )

    report = router.status_report(s, session_id=None, now=NOW, hours=24)
    assert "all sessions" in report
    assert "calc" in report  # the project name
    assert "Old one" not in report and "999" not in report

    week = router.status_report(s, session_id=None, now=NOW, hours=24 * 7)
    assert "Old one" in week


def test_typing_the_status_command_shows_the_report_without_calling_claude(routed_session):
    s, path = routed_session
    event = prompt_event("/noulo-router:status")
    event["transcript_path"] = str(path)
    out = router.route_prompt(event, s)

    assert out["decision"] == "block"  # Claude never sees the prompt, so it costs no tokens
    assert "noulo-router status" in out["reason"]
    assert "opus 5.5 · deep-agent" in out["reason"]


def test_the_status_command_works_with_routing_off(routed_session, settings):
    s, _ = routed_session
    s.mode = "off"
    out = router.route_prompt(prompt_event("/noulo-router:status all"), s)
    assert "all sessions" in out["reason"]


def test_the_status_command_line_prints_the_report(routed_session, tmp_path):
    s, _ = routed_session
    code, out = run(["status", "--session", "sess-1", "--data", str(tmp_path)], {})
    assert code == 0
    assert "noulo-router status" in out


def test_status_watch_redraws_the_report(routed_session, tmp_path):
    code, out = run(
        ["status", "--all", "--data", str(tmp_path), "--watch", "--interval", "0", "--count", "2"],
        {},
    )
    assert code == 0
    assert out.count("noulo-router status") == 2
    assert "\x1b[2J" in out  # clears the screen between redraws


def test_one_routed_prompt_reads_naturally(noulo, settings):
    s = settings()
    router.route_prompt(prompt_event("The upload test fails with a KeyError, fix it"), s)
    report = router.status_report(s, session_id="sess-1")
    assert "1 prompt routed" in report


def test_a_calls_final_line_has_its_real_output_count(tmp_path):
    # While a call streams, Claude Code writes a line per content block with a provisional count.
    path = write_session(
        tmp_path,
        main=[
            assistant("2026-09-30T10:00:01.000Z", "m1", "claude-opus-5-5", out=8, stop=None),
            assistant(
                "2026-09-30T10:00:02.000Z", "m1", "claude-opus-5-5", out=301, stop="tool_use"
            ),
        ],
    )
    (call,) = router.read_calls(path)
    assert (call.output, call.final) == (301, True)


def test_background_subagent_output_is_shown_as_a_minimum(noulo, settings, tmp_path):
    # Claude Code never writes a final line for a background subagent's calls.
    s = settings()
    path = write_session(
        tmp_path,
        main=[assistant("2026-09-30T10:00:01.000Z", "m1", "claude-sonnet-5-5", out=500)],
        agents={
            "a1": (
                "noulo-router:deep-agent",
                [
                    assistant(
                        "2026-09-30T10:00:05.000Z", "s1", "claude-opus-5-5", out=16, stop=None
                    ),
                    assistant(
                        "2026-09-30T10:00:09.000Z", "s2", "claude-opus-5-5", out=5, stop=None
                    ),
                ],
            )
        },
    )
    router.EventLog(s.events_file).append(
        {
            "event": "route",
            "ts": "2026-09-30T10:00:00.000Z",
            "session_id": "sess-1",
            "transcript_path": str(path),
            "outcome": "routed",
            "tier": "deep",
            "model": "opus",
            "effort": "high",
            "task": "debug",
            "confidence": 0.64,
            "prompt": "Fix it",
            "noulo_ms": 400,
        }
    )

    report = router.status_report(s, session_id="sess-1", now=NOW)
    opus = next(line for line in report.splitlines() if line.strip().startswith("opus 5.5 · deep"))
    total = next(line for line in report.splitlines() if line.strip().startswith("total"))
    assert opus.split()[-1] == "21+"
    assert total.split()[-1] == "521+"
    assert "opus 5.5 deep-agent ×2 · 21+ out" in report
    assert "+ 2 calls have no final output count yet" in report


def test_the_command_line_finds_the_plugins_data_directory(tmp_path):
    import os

    data = tmp_path / ".claude" / "plugins" / "data"
    for name, age in (("noulo-router-inline", 100), ("noulo-router-noulo", 0)):
        (data / name).mkdir(parents=True)
        (data / name / "events.jsonl").write_text("{}\n")
        os.utime(data / name / "events.jsonl", (1_900_000_000 - age, 1_900_000_000 - age))

    settings = router.Settings.from_env({"HOME": str(tmp_path)})
    assert settings.state_file.parent == data / "noulo-router-noulo"


def test_without_plugin_data_the_command_line_uses_its_own_directory(tmp_path):
    settings = router.Settings.from_env({"HOME": str(tmp_path)})
    assert settings.state_file.parent == tmp_path / ".claude" / "noulo-router"


def test_the_command_line_ignores_another_plugins_data_directory(routed_session, tmp_path):
    # A shell inside Claude Code can inherit CLAUDE_PLUGIN_DATA from a different plugin.
    s, _ = routed_session
    home = tmp_path / "home"
    ours = home / ".claude" / "plugins" / "data" / "noulo-router-noulo"
    ours.mkdir(parents=True)
    (ours / "events.jsonl").write_text(s.events_file.read_text())
    other = home / ".claude" / "plugins" / "data" / "codex-openai-codex"
    other.mkdir(parents=True)

    env = {"HOME": str(home), "CLAUDE_PLUGIN_DATA": str(other)}
    code, out = run(["status", "all", "--hours", "100000"], env)
    assert code == 0
    assert "2 prompts routed" in out or "not reachable" in out
    assert "test_add fails, fix it" in out


# ---------------------------------------------------------------- why and correct via the hook


def test_typing_why_shows_the_last_decision_without_calling_claude(noulo, settings):
    s = settings()
    router.route_prompt(prompt_event("The upload test fails with a KeyError, fix it"), s)
    out = router.route_prompt(prompt_event("/noulo-router:why"), s)

    assert out["decision"] == "block"
    assert "Last routing decision: deep · opus · high" in out["reason"]


def test_typing_correct_teaches_noulo_without_calling_claude(noulo, settings):
    s = settings()
    router.route_prompt(prompt_event("The upload test fails with a KeyError, fix it"), s)
    out = router.route_prompt(prompt_event("/noulo-router:correct feature large"), s)

    assert out["decision"] == "block"
    assert "noulo learned: feature, large" in out["reason"]
    bodies = [r["body"] for r in noulo.posts("/api/v1/feedback")]
    assert bodies == [
        {"recordId": "rec-0", "expected": "feature"},
        {"recordId": "rec-1", "expected": True},
    ]


def test_a_mistyped_correction_explains_itself(noulo, settings):
    s = settings()
    router.route_prompt(prompt_event("The upload test fails with a KeyError, fix it"), s)
    out = router.route_prompt(prompt_event("/noulo-router:correct urgent"), s)
    assert "one of explain, edit" in out["reason"]
    assert noulo.posts("/api/v1/feedback") == []


def test_why_and_correct_work_with_routing_off(noulo, settings):
    s = settings("off")
    assert (
        "No routing decision" in router.route_prompt(prompt_event("/noulo-router:why"), s)["reason"]
    )


def test_tier_commands_still_go_to_claude(noulo, settings):
    for command in (
        "/noulo-router:deep fix it",
        "/noulo-router:light what is this",
        "/noulo-router:whyever",
    ):
        assert router.route_prompt(prompt_event(command), settings()) is None


# ---------------------------------------------------------------- /noulo-router:disable and :enable


def test_disable_stops_routing_in_every_session_until_enabled(noulo, settings):
    out = router.route_prompt(prompt_event("/noulo-router:disable"), settings())
    assert out["decision"] == "block"
    assert "Routing is off" in out["reason"]

    # Settings are read afresh for each hook run, like Claude Code does.
    assert router.route_prompt(prompt_event("Fix the flaky upload test"), settings()) is None
    other = prompt_event("Fix the flaky upload test", session="sess-2")
    assert router.route_prompt(other, settings()) is None
    assert noulo.posts() == []

    out = router.route_prompt(prompt_event("/noulo-router:enable"), settings())
    assert "Routing is on" in out["reason"]
    assert router.route_prompt(prompt_event("Fix the flaky upload test"), settings())
    assert len(noulo.posts()) == 3


def test_while_disabled_session_start_is_quiet(settings):
    down = closed_port_url()
    router.route_prompt(prompt_event("/noulo-router:disable"), settings(url=down))
    assert router.session_start(start_event(), settings(url=down)) is None


def test_while_disabled_the_other_commands_still_work(noulo, settings):
    router.route_prompt(prompt_event("The upload test fails with a KeyError, fix it"), settings())
    router.route_prompt(prompt_event("/noulo-router:disable"), settings())

    assert (
        "deep · opus · high"
        in router.route_prompt(prompt_event("/noulo-router:why"), settings())["reason"]
    )
    status = router.route_prompt(prompt_event("/noulo-router:status"), settings())["reason"]
    assert "routing  off" in status and "/noulo-router:enable" in status


def test_status_shows_that_routing_is_on_and_its_mode(noulo, settings):
    status = router.route_prompt(prompt_event("/noulo-router:status"), settings("suggest"))[
        "reason"
    ]
    assert "routing  on · mode suggest" in status


def test_enable_explains_when_the_mode_option_is_off(noulo, settings):
    out = router.route_prompt(prompt_event("/noulo-router:enable"), settings("off"))
    assert "mode option is off" in out["reason"]


def test_disabling_twice_says_so(noulo, settings):
    router.route_prompt(prompt_event("/noulo-router:disable"), settings())
    out = router.route_prompt(prompt_event("/noulo-router:disable"), settings())
    assert "already off" in out["reason"]


@pytest.mark.parametrize("name", ["disable", "enable"])
def test_disable_and_enable_work_from_the_command_line(name, tmp_path):
    code, out = run([name, "--data", str(tmp_path)], {})
    assert code == 0 and "Routing is" in out


@pytest.mark.parametrize("name", ["status", "why", "correct", "disable", "enable"])
def test_the_hook_alone_runs_the_plugins_commands(name):
    # Claude Code runs a typed skill's !`command` lines before UserPromptSubmit hooks, so a
    # fallback line there would run each command twice (twice the feedback, for correct).
    fields, body = frontmatter(PLUGIN / "skills" / name / "SKILL.md")
    assert fields["disable-model-invocation"] == "true"
    assert "!`" not in body
    assert "hook" in body


def test_corrections_read_naturally(noulo, settings):
    s = settings()
    router.route_prompt(prompt_event("Fix the typo in the README heading please"), s)
    out = router.route_prompt(prompt_event("/noulo-router:correct edit"), s)
    assert "an edit task" in out["reason"]
