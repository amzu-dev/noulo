#!/usr/bin/env python3
"""noulo-router: route Claude Code prompts to a model tier chosen by noulo.

Standard library only, so it runs with any Python 3.10+ without installing anything.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

TIER_ORDER = ("light", "standard", "deep", "max")
MODEL_ORDER = ("haiku", "sonnet", "opus", "fable")  # weakest → strongest
MODES = ("auto", "suggest", "off")
DEFAULT_URL = "http://127.0.0.1:8790"
PLUGIN_ROOT = Path(__file__).resolve().parent.parent
# The plugin's own commands, answered by the hook so they never call (or cost) Claude.
COMMAND = re.compile(r"\s*/noulo-router:(status|why|correct|disable|enable)(?=\s|$)(.*)", re.DOTALL)
GENERATED = re.compile(r"\s*<[a-z][a-z0-9_-]*[\s>]")  # messages Claude Code wraps in a tag


class NouloUnavailable(Exception):
    """noulo couldn't be reached, answered with an error, or took too long."""


class NouloTooSlow(NouloUnavailable):
    """noulo didn't answer within the time budget."""


class NouloClient:
    """Minimal noulo REST client. `timeout` is the total budget for everything this client does."""

    def __init__(self, url: str, api_key: str | None = None, timeout: float = 3.0):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.deadline = time.monotonic() + timeout

    def _request(self, method: str, path: str, body: dict | None = None):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise NouloTooSlow("noulo took too long")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=remaining) as resp:
                return json.loads(resp.read() or b"{}"), resp.headers
        except urllib.error.HTTPError as e:
            try:
                message = json.loads(e.read())["error"]["message"]
            except (ValueError, KeyError, TypeError):
                message = f"HTTP {e.code}"
            raise NouloUnavailable(message) from e
        except (OSError, ValueError) as e:  # URLError, timeouts, bad JSON
            reason = getattr(e, "reason", e)
            if isinstance(e, TimeoutError) or isinstance(reason, TimeoutError):
                raise NouloTooSlow("noulo took too long") from e
            raise NouloUnavailable(str(reason)) from e

    def evaluate(self, body: dict) -> tuple[dict, str | None]:
        result, headers = self._request("POST", "/api/v1/evaluate?diagnostics=true", body)
        return result, headers.get("X-Record-Id")

    def feedback(self, record_id: str, expected) -> None:
        self._request("POST", "/api/v1/feedback", {"recordId": record_id, "expected": expected})

    def teach(self, items: list[dict]) -> dict:
        result, _ = self._request("POST", "/api/v1/learning/import", {"items": items})
        return result

    def info(self) -> dict:
        return self._request("GET", "/api/v1/info")[0]

    def healthy(self) -> bool:
        try:
            result, _ = self._request("GET", "/health")
        except NouloUnavailable:
            return False
        return bool(result.get("modelLoaded"))


@dataclass
class Signals:
    """What noulo said about one prompt."""

    task: str
    confidence: float
    large: float
    security: float
    record_ids: dict[str, str] = field(default_factory=dict)


@dataclass
class Policy:
    question: str
    tasks: dict[str, str]
    large_work: str
    security: str
    task_tiers: dict[str, str]
    tiers: dict[str, dict[str, str]]
    min_confidence: float
    fallback_tier: str
    max_at: float
    security_at: float
    followup_max_words: int

    def decide(self, s: Signals) -> str:
        if s.confidence < self.min_confidence:
            tier = self.fallback_tier
        else:
            tier = self.task_tiers[s.task]
            if tier == "deep" and s.large >= self.max_at:
                tier = "max"
        if s.security >= self.security_at and TIER_ORDER.index(tier) < TIER_ORDER.index("deep"):
            tier = "deep"
        return tier


def load_policy(path: Path) -> Policy:
    return Policy(**json.loads(Path(path).read_text(encoding="utf-8")))


def trim(text: str, head: int = 300, tail: int = 150) -> str:
    """Keep the start and end of long prompts, where the request usually is.

    noulo's latency grows with input length (about 6 s for 1,500 characters on the base model),
    and a pasted log in the middle doesn't help to tell what kind of task it is.
    The defaults keep a long prompt around 2 s on the base model.
    """
    if len(text) <= head + tail:
        return text
    return f"{text[:head]}\n…\n{text[-tail:]}"


def classify(client: NouloClient, prompt: str, policy: Policy) -> Signals:
    text = trim(prompt)
    choices = [{"id": k, "text": v} for k, v in policy.tasks.items()]
    task, task_id = client.evaluate(
        {"type": "choice", "input": text, "question": policy.question, "choices": choices}
    )
    large, large_id = client.evaluate(
        {"type": "noul", "input": text, "proposition": policy.large_work}
    )
    security, security_id = client.evaluate(
        {"type": "noul", "input": text, "proposition": policy.security}
    )
    ids = {"task": task_id, "large": large_id, "security": security_id}
    return Signals(
        task=task["value"],
        confidence=task["diagnostics"]["probabilities"][task["value"]],
        large=large["value"],
        security=security["value"],
        record_ids={k: v for k, v in ids.items() if v},
    )


# ---------------------------------------------------------------- settings and state


@dataclass
class Settings:
    mode: str
    url: str
    api_key: str | None
    state_file: Path
    policy: Policy
    timeout: float = 4.0  # total noulo budget per prompt; the hook itself times out at 10 s
    disabled: bool = False  # /noulo-router:disable, until /noulo-router:enable

    @property
    def routing_on(self) -> bool:
        return self.mode != "off" and not self.disabled

    @property
    def disabled_file(self) -> Path:
        return self.state_file.parent / "disabled"

    @property
    def events_file(self) -> Path:
        return self.state_file.parent / "events.jsonl"

    @classmethod
    def from_env(cls, env) -> Settings:
        """Plugin options arrive as CLAUDE_PLUGIN_OPTION_<KEY>; the API key is a plain env var."""
        root = Path(env.get("CLAUDE_PLUGIN_ROOT") or PLUGIN_ROOT)
        home = Path(env.get("HOME") or Path.home())
        data = Path(env.get("CLAUDE_PLUGIN_DATA") or find_data_dir(home))
        mode = (env.get("CLAUDE_PLUGIN_OPTION_MODE") or "auto").strip().lower()
        return cls(
            mode=mode if mode in MODES else "auto",
            url=(env.get("CLAUDE_PLUGIN_OPTION_URL") or DEFAULT_URL).strip().rstrip("/"),
            api_key=env.get("NOULO_ROUTER_API_KEY") or None,
            state_file=data / "state.json",
            policy=load_policy(root / "router.json"),
            disabled=(data / "disabled").exists(),
        )


def find_data_dir(home: Path) -> Path:
    """Outside a hook: the plugin data directory used most recently (marketplace or inline)."""
    logs = (home / ".claude" / "plugins" / "data").glob("noulo-router-*/events.jsonl")
    newest = max(logs, key=lambda p: p.stat().st_mtime, default=None)
    return newest.parent if newest else home / ".claude" / "noulo-router"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class EventLog:
    """Append-only JSON Lines log for the status page; keeps the newest events when it grows."""

    def __init__(self, path: Path, max_bytes: int = 2_000_000):
        self.path = Path(path)
        self.max_bytes = max_bytes

    def append(self, event: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": now_iso(), **event}) + "\n")
        if self.path.stat().st_size > self.max_bytes:
            lines = self.path.read_text(encoding="utf-8").splitlines(keepends=True)
            kept, size = [], 0
            for line in reversed(lines):
                size += len(line.encode())
                if size > self.max_bytes // 2:
                    break
                kept.append(line)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text("".join(reversed(kept)), encoding="utf-8")
            os.replace(tmp, self.path)


class StateStore:
    """Per-session state in one JSON file: the last decision and whether we've warned."""

    def __init__(self, path: Path, max_sessions: int = 200):
        self.path = Path(path)
        self.max_sessions = max_sessions

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def get(self, session_id: str) -> dict:
        return self._load().get(session_id, {})

    def update(self, session_id: str, **fields) -> None:
        data = self._load()
        entry = data.pop(session_id, {})  # re-inserted last, so dict order is recency
        entry.update(fields, updated=time.time())
        data[session_id] = entry
        while len(data) > self.max_sessions:
            data.pop(next(iter(data)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)


# ---------------------------------------------------------------- hooks


def _label(tier: str, why: str, policy: Policy) -> str:
    t = policy.tiers[tier]
    return f"{tier} · {t['model']} · {t['effort']} ({why})"


def _delegation_context(tier: str, why: str, policy: Policy) -> str:
    model = policy.tiers[tier]["model"]
    effort = policy.tiers[tier]["effort"]
    subject = (
        "This short message continues the previous request, which noulo routed"
        if why == "follow-up"
        else f"noulo classified this request as {why} and routed it"
    )
    return (
        f"noulo-router: {subject} to the {tier} tier: model {model}, effort {effort}.\n"
        f"Model strength order: {' < '.join(MODEL_ORDER)}.\n"
        f"If you are running on a weaker model than {model}, delegate this request to the "
        f"`noulo-router:{tier}-agent` subagent with the Agent tool: pass the user's request word "
        "for word plus the context from this conversation it needs (relevant files, decisions, "
        "constraints), then report its result. If you are running on "
        f"{model} or a stronger model, handle the request yourself and don't delegate."
    )


def render(tier: str, why: str, settings: Settings) -> dict:
    """The hook output for a routing decision, depending on the mode."""
    policy = settings.policy
    model = policy.tiers[tier]["model"]
    label = _label(tier, why, policy)
    if settings.mode == "suggest":
        return {
            "systemMessage": f"noulo-router suggests {label}. "
            f"Send /noulo-router:{tier} <your prompt> to run this turn on {model}."
        }
    if model == MODEL_ORDER[0]:  # nothing weaker to delegate from
        return {
            "systemMessage": f"noulo-router → {label}. Runs on your session model; "
            f"/noulo-router:{tier} <your prompt> runs a turn on {model}."
        }
    return {
        "systemMessage": f"noulo-router → {label}",
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": _delegation_context(tier, why, policy),
        },
    }


def start_command(url: str) -> str:
    port = urllib.parse.urlsplit(url).port or 8790
    return (
        "NOULO_MODEL=zeroshot-deberta-v3-base-fp32 NOULO_RUN_DIR=data/router "
        f"NOULO_MEMORY_LOCATION=data/router/memory.sqlite3 noulo start --headless --port {port}"
    )


def _unreachable(settings: Settings, reason: NouloUnavailable | str) -> dict:
    if isinstance(reason, NouloTooSlow):
        return {
            "systemMessage": f"noulo-router: noulo at {settings.url} took too long to answer, "
            "so this prompt wasn't routed."
        }
    return {
        "systemMessage": f"noulo-router: noulo isn't reachable at {settings.url} ({reason}), so "
        "prompts aren't being routed. Start the router's instance from your noulo clone: "
        + start_command(settings.url)
    }


def route_prompt(event: dict, settings: Settings) -> dict | None:
    """UserPromptSubmit: classify the prompt with noulo and say where it should run."""
    prompt = event.get("prompt") or event.get("user_input") or ""
    command = COMMAND.match(prompt) if isinstance(prompt, str) else None
    if command:
        # Blocking the prompt shows the reason in the terminal without sending it to Claude.
        return {"decision": "block", "reason": run_command(*command.groups(), event, settings)}
    if not settings.routing_on:
        return None
    if not isinstance(prompt, str) or not prompt.strip() or prompt.lstrip().startswith("/"):
        return None
    if GENERATED.match(prompt):  # e.g. a subagent's <task-notification>, not something typed
        return None
    prompt = prompt.strip()
    session_id = event.get("session_id", "")
    store = StateStore(settings.state_file)
    session = store.get(session_id)
    log = EventLog(settings.events_file)
    base = {
        "event": "route",
        "session_id": session_id,
        "transcript_path": event.get("transcript_path"),
        "cwd": event.get("cwd"),
        "mode": settings.mode,
        "prompt": prompt[:200],
    }

    def tier_fields(tier: str) -> dict:
        return {"tier": tier, **settings.policy.tiers[tier]}

    if len(prompt.split()) <= settings.policy.followup_max_words:
        last = session.get("last")
        if not last:
            return None
        log.append({**base, "outcome": "follow-up", **tier_fields(last["tier"])})
        return render(last["tier"], "follow-up", settings)

    started = time.perf_counter()
    try:
        s = classify(
            NouloClient(settings.url, settings.api_key, settings.timeout), prompt, settings.policy
        )
    except NouloUnavailable as e:
        log.append({**base, "outcome": "slow" if isinstance(e, NouloTooSlow) else "unreachable"})
        if session.get("warned"):
            return None
        store.update(session_id, warned=True)
        return _unreachable(settings, e)
    noulo_ms = round((time.perf_counter() - started) * 1000)

    tier = settings.policy.decide(s)
    log.append(
        {
            **base,
            "outcome": "routed",
            **tier_fields(tier),
            "task": s.task,
            "confidence": s.confidence,
            "large": s.large,
            "security": s.security,
            "noulo_ms": noulo_ms,
        }
    )
    store.update(
        session_id,
        warned=False,
        last={
            "tier": tier,
            "task": s.task,
            "confidence": s.confidence,
            "large": s.large,
            "security": s.security,
            "record_ids": s.record_ids,
            "url": settings.url,
            "prompt": prompt[:200],
        },
    )
    return render(tier, f"{s.task}, {round(s.confidence * 100)}%", settings)


def run_command(name: str, args: str, event: dict, settings: Settings) -> str:
    words = args.split()
    session_id = event.get("session_id", "")
    if name == "status":
        return status_report(
            settings,
            session_id=None if "all" in words else session_id,
            transcript_path=event.get("transcript_path"),
        )
    if name == "why":
        return why(settings, session_id)
    if name in ("disable", "enable"):
        return set_enabled(settings, name == "enable")
    return correct(settings, session_id, words, settings.api_key)[1]


def set_enabled(settings: Settings, on: bool) -> str:
    """Turn routing on or off for every session; the other commands keep working either way."""
    if not on:
        if settings.disabled:
            return "Routing is already off. /noulo-router:enable turns it back on."
        settings.disabled_file.parent.mkdir(parents=True, exist_ok=True)
        settings.disabled_file.write_text(now_iso() + "\n", encoding="utf-8")
        settings.disabled = True
        return (
            "Routing is off in every session until you run /noulo-router:enable. "
            "/noulo-router:status, why and correct still work."
        )
    settings.disabled_file.unlink(missing_ok=True)
    settings.disabled = False
    if settings.mode == "off":
        return (
            "Routing is on, but the plugin's mode option is off, so prompts still aren't routed. "
            "Change Routing mode for noulo-router in /config."
        )
    return f"Routing is on (mode {settings.mode})."


def session_start(event: dict, settings: Settings) -> dict | None:
    """SessionStart: warn (never start anything) when the router's noulo isn't ready."""
    if not settings.routing_on:
        return None
    healthy = NouloClient(settings.url, settings.api_key, settings.timeout).healthy()
    EventLog(settings.events_file).append(
        {
            "event": "session",
            "session_id": event.get("session_id", ""),
            "transcript_path": event.get("transcript_path"),
            "cwd": event.get("cwd"),
            "noulo_healthy": healthy,
        }
    )
    if healthy:
        return None
    StateStore(settings.state_file).update(event.get("session_id", ""), warned=True)
    return _unreachable(settings, "no healthy answer from /health")


# ---------------------------------------------------------------- status: models, times, tokens


@dataclass
class Call:
    """One Claude API call, read from a Claude Code transcript."""

    ts: datetime
    model: str
    agent: str  # "main", or the subagent type such as "noulo-router:deep-agent"
    input: int
    cache_write: int
    cache_read: int
    output: int
    final: bool = True  # False: Claude Code hasn't written the call's final output count


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _calls_in(path: Path, agent: str) -> list[Call]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    text = text[: text.rfind("\n") + 1]  # the last line may still be being written
    calls: dict[str, Call] = {}  # one API call is written as one line per content block
    for line in text.splitlines():
        if '"assistant"' not in line or '"usage"' not in line:
            continue
        try:
            entry = json.loads(line)
            message = entry["message"]
            usage = message["usage"]
            model = message.get("model") or ""
            if entry.get("type") != "assistant" or not usage or model.startswith("<"):
                continue
            call = Call(
                ts=parse_ts(entry["timestamp"]),
                model=model,
                agent=agent,
                input=int(usage.get("input_tokens") or 0),
                cache_write=int(usage.get("cache_creation_input_tokens") or 0),
                cache_read=int(usage.get("cache_read_input_tokens") or 0),
                output=int(usage.get("output_tokens") or 0),
                final=bool(message.get("stop_reason")),
            )
            key = message.get("id") or entry.get("uuid")
            if key in calls:  # a later line of the same call: keep the first time, largest counts
                seen = calls[key]
                call.ts = seen.ts
                call.output = max(call.output, seen.output)
                call.final = call.final or seen.final
            calls[key] = call
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return list(calls.values())


def read_calls(transcript: Path) -> list[Call]:
    """Calls in a session's transcript and in its subagents' transcripts, oldest first."""
    transcript = Path(transcript)
    calls = _calls_in(transcript, "main")
    for path in sorted((transcript.parent / transcript.stem / "subagents").glob("agent-*.jsonl")):
        try:
            meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
            agent = meta.get("agentType") or "subagent"
        except (OSError, ValueError, AttributeError):
            agent = "subagent"
        calls += _calls_in(path, agent)
    return sorted(calls, key=lambda c: c.ts)


def pretty_model(model_id: str) -> str:
    m = re.fullmatch(r"claude-([a-z]+)-(\d+)(?:-(\d{1,2}))?(?:-\d{8})?", model_id)
    if not m:
        return model_id
    family, major, minor = m.groups()
    return f"{family} {major}.{minor}" if minor else f"{family} {major}"


def _where(agent: str) -> str:
    return agent.rsplit(":", 1)[-1]  # "noulo-router:deep-agent" → "deep-agent"


def _read_events(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    events = []
    for line in lines:
        try:
            event = json.loads(line)
            event["_ts"] = parse_ts(event["ts"])
            events.append(event)
        except (ValueError, KeyError, TypeError):
            continue
    return events


def _when(ts: datetime, now: datetime) -> str:
    local = ts.astimezone()
    return local.strftime("%H:%M" if local.date() == now.astimezone().date() else "%b %d %H:%M")


def _noulo_line(settings: Settings, routed: list[dict]) -> str:
    client = NouloClient(settings.url, settings.api_key, timeout=1.5)
    if not client.healthy():
        return f"noulo    not reachable at {settings.url}"
    try:
        model = client.info().get("model", "?")
    except NouloUnavailable:
        model = "?"
    count = f"{len(routed)} prompt{'' if len(routed) == 1 else 's'} routed"
    line = f"noulo    up · {model} · {settings.url} · {count}"
    times = sorted(e["noulo_ms"] for e in routed if isinstance(e.get("noulo_ms"), int))
    if times:
        line += f" · p50 {times[len(times) // 2]} ms"
    return line


def status_report(
    settings: Settings,
    session_id: str | None = None,
    now: datetime | None = None,
    hours: float = 24,
    transcript_path: str | None = None,
) -> str:
    """Which Claude models ran when, and their tokens: one session, or all sessions lately."""
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(hours=hours)
    events = _read_events(settings.events_file)
    if session_id:
        events = [e for e in events if e.get("session_id") == session_id]
    else:
        events = [e for e in events if e["_ts"] >= since]

    transcripts = {
        e["session_id"]: e["transcript_path"] for e in events if e.get("transcript_path")
    }
    if session_id and transcript_path:
        transcripts[session_id] = transcript_path
    calls: dict[str, list[Call]] = {}
    for sid, path in transcripts.items():
        found = read_calls(Path(path))
        calls[sid] = found if session_id else [c for c in found if c.ts >= since]

    scope = "this session" if session_id else f"all sessions · last {hours:g} h"
    out = [f"noulo-router status · {scope} · updated {now.astimezone().strftime('%H:%M:%S')}"]
    routes = [e for e in events if e.get("event") == "route"]
    if settings.disabled:
        out.append("routing  off · /noulo-router:enable turns it back on")
    elif settings.mode == "off":
        out.append("routing  off · the mode option is off in /config")
    else:
        out.append(f"routing  on · mode {settings.mode}")
    out.append(_noulo_line(settings, [e for e in routes if e.get("outcome") == "routed"]))

    # Tokens per model and where it ran.
    rows: dict[tuple[str, str], list[int]] = {}
    for c in (c for found in calls.values() for c in found):
        row = rows.setdefault((pretty_model(c.model), _where(c.agent)), [0, 0, 0, 0, 0, 0])
        for i, n in enumerate((1, c.input, c.cache_write, c.cache_read, c.output, not c.final)):
            row[i] += n
    out.append("")
    header = f"{'Claude API calls':<30}{'calls':>6}{'input':>10}{'cache write':>13}"
    out.append(f"{header}{'cache read':>13}{'output':>10}")
    if not rows:
        out.append("  no Claude calls yet")
    for (model, where), row in sorted(rows.items(), key=lambda kv: -kv[1][4]):
        out.append(_row(f"{model} · {where}", row))
    if len(rows) > 1:
        out.append(_row("total", [sum(col) for col in zip(*rows.values())]))
    provisional = sum(row[5] for row in rows.values())
    if provisional:
        calls_have = "1 call has" if provisional == 1 else f"{provisional} calls have"
        out.append(f"  + {calls_have} no final output count yet, so output is at least this.")
        out.append(
            "    (Still running, or run by a background subagent: Claude Code doesn't record"
        )
        out.append("    a background subagent's final count.)")

    # Each prompt, with the calls that followed it until the session's next prompt.
    out.append("")
    if not routes:
        out.append("No routed prompts yet.")
        return "\n".join(out)
    out.append("Prompts, newest first")
    for event in sorted(routes, key=lambda e: e["_ts"], reverse=True)[:10]:
        out.extend(
            _prompt_lines(event, routes, calls.get(event.get("session_id"), []), now, session_id)
        )
    return "\n".join(out)


def _tokens(n: int, provisional: int) -> str:
    return f"{n:,}+" if provisional else f"{n:,}"


def _row(label: str, row: list[int]) -> str:
    calls, inp, write, read, output, provisional = row
    shown = _tokens(output, provisional)
    return f"  {label:<28}{calls:>6,}{inp:>10,}{write:>13,}{read:>13,}{shown:>10}"


def _prompt_lines(event, routes, calls, now, session_id) -> list[str]:
    text = event.get("prompt", "")
    text = text if len(text) <= 60 else text[:59] + "…"
    project = "" if session_id else f"[{Path(event.get('cwd') or '?').name}] "
    outcome = event.get("outcome")
    if outcome in ("routed", "follow-up"):
        why = (
            f"({event['task']} {event['confidence']:.0%})" if outcome == "routed" else "(follow-up)"
        )
        what = f"{event['tier']} · {event['model']} · {event['effort']}  {why}"
    else:
        what = "not routed: noulo " + ("too slow" if outcome == "slow" else "unreachable")
    lines = [f"  {_when(event['_ts'], now):>12}  {project}{what}  {text}"]

    later = [e["_ts"] for e in routes if e.get("session_id") == event.get("session_id")]
    end = min((t for t in later if t > event["_ts"]), default=None)
    groups: dict[tuple[str, str], list[int]] = {}
    for c in calls:
        if c.ts >= event["_ts"] and (end is None or c.ts < end):
            g = groups.setdefault((pretty_model(c.model), _where(c.agent)), [0, 0, 0])
            g[0] += 1
            g[1] += c.output
            g[2] += not c.final
    used = "   ".join(f"{m} {w} ×{n} · {_tokens(o, p)} out" for (m, w), (n, o, p) in groups.items())
    lines.append(f"  {'':>12}  → {used or 'no Claude calls yet'}")
    return lines


# ---------------------------------------------------------------- why, correct, teach, eval


def why(settings: Settings, session_id: str) -> str:
    last = StateStore(settings.state_file).get(session_id).get("last")
    if not last:
        return "No routing decision in this session yet."
    policy = settings.policy
    tier = policy.tiers[last["tier"]]
    return "\n".join(
        [
            f"Last routing decision: {last['tier']} · {tier['model']} · {tier['effort']}",
            f"  prompt:      {last['prompt']}",
            f"  task type:   {last['task']} ({round(last['confidence'] * 100)}%)",
            f"  large work:  {last['large']:.0%}   security: {last['security']:.0%}",
            f"  noulo:       {last['url']}",
            "Wrong? /noulo-router:correct <task type> [large|small] [security|no-security]",
            f"Task types: {', '.join(policy.tasks)}",
        ]
    )


def correct(
    settings: Settings, session_id: str, words: list[str], api_key: str | None
) -> tuple[int, str]:
    """Send verified feedback for the last decision's evaluations."""
    policy = settings.policy
    valid = [*policy.tasks, "large", "small", "security", "no-security"]
    unknown = [w for w in words if w not in valid]
    if not words or unknown:
        return 2, f"Say what the request was: one of {', '.join(valid)}."
    last = StateStore(settings.state_file).get(session_id).get("last")
    if not last:
        return 1, "No routing decision in this session to correct."

    feedback = {}  # record kind -> expected
    for w in words:
        if w in policy.tasks:
            feedback["task"] = w
        else:
            kind = "security" if "security" in w else "large"
            feedback[kind] = w in ("large", "security")
    ids = last.get("record_ids", {})
    if any(kind not in ids for kind in feedback):
        return (
            1,
            "noulo didn't record this decision (its learning was off), so it can't be corrected.",
        )

    client = NouloClient(last["url"], api_key, timeout=10)
    try:
        for kind, expected in feedback.items():
            client.feedback(ids[kind], expected)
    except NouloUnavailable as e:
        return (
            1,
            f"noulo couldn't learn from this: {e} Learning must be on for the router's instance.",
        )

    fixed = Signals(
        task=feedback.get("task", last["task"]),
        confidence=1.0 if "task" in feedback else last["confidence"],
        large=float(feedback["large"]) if "large" in feedback else last["large"],
        security=float(feedback["security"]) if "security" in feedback else last["security"],
    )
    tier = policy.decide(fixed)
    return 0, (
        f"noulo learned: {', '.join(words)}. With that, this request is "
        f"{'an' if fixed.task[0] in 'aeiou' else 'a'} {fixed.task} task "
        f"for the {tier} tier ({policy.tiers[tier]['model']} · {policy.tiers[tier]['effort']}). "
        "noulo uses verified feedback for similar prompts from now on."
    )


LARGE_LABELS = {"trivial": False, "simple": False, "complex": True, "very complex": True}


def read_labelled(path: Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def teaching_items(rows: list[dict], policy: Policy) -> list[dict]:
    """Labelled prompts → noulo import items: the task type, and large work where it's clear."""
    choices = [{"id": k, "text": v} for k, v in policy.tasks.items()]
    items = []
    for r in rows:
        text = trim(r["prompt"])
        items.append(
            {
                "type": "choice",
                "input": text,
                "question": policy.question,
                "choices": choices,
                "expected": r["task"],
            }
        )
    for r in rows:
        if r.get("complexity") in LARGE_LABELS:
            items.append(
                {
                    "type": "noul",
                    "input": trim(r["prompt"]),
                    "proposition": policy.large_work,
                    "expected": LARGE_LABELS[r["complexity"]],
                }
            )
    return items


def evaluate_file(client_factory, rows: list[dict], policy: Policy) -> dict:
    task_ok = tier_ok = 0
    confusion: dict[str, dict[str, int]] = {}
    latencies = []
    for r in rows:
        started = time.perf_counter()
        s = classify(client_factory(), r["prompt"], policy)
        latencies.append((time.perf_counter() - started) * 1000)
        task_ok += s.task == r["task"]
        tier_ok += policy.decide(s) == r["tier"]
        if s.task != r["task"]:
            row = confusion.setdefault(r["task"], {})
            row[s.task] = row.get(s.task, 0) + 1
    latencies.sort()
    n = len(rows)
    return {
        "items": n,
        "task_accuracy": task_ok / n,
        "tier_accuracy": tier_ok / n,
        "confusion": confusion,
        "latency_ms": {
            "p50": round(latencies[n // 2]),
            "p95": round(latencies[max(0, -(-n * 95 // 100) - 1)]),
            "max": round(latencies[-1]),
        },
    }


# ---------------------------------------------------------------- entry point


def _cli_env(env, data: str | None) -> dict:
    """The environment for a command-line run: find this plugin's data unless --data says."""
    if data:
        return {**env, "CLAUDE_PLUGIN_DATA": data}
    inherited = env.get("CLAUDE_PLUGIN_DATA", "")
    if Path(inherited).name.startswith("noulo-router"):
        return dict(env)
    # A shell inside Claude Code can inherit another plugin's CLAUDE_PLUGIN_DATA.
    home = Path(env.get("HOME") or Path.home())
    return {**env, "CLAUDE_PLUGIN_DATA": str(find_data_dir(home))}


def _hook(kind: str, stdin, stdout, env) -> int:
    """Hooks never fail: any problem means the prompt goes through unrouted."""
    try:
        event = json.loads(stdin.read() or "{}")
        if not isinstance(event, dict):
            return 0
        settings = Settings.from_env(env)
        out = route_prompt(event, settings) if kind == "prompt" else session_start(event, settings)
        if out:
            stdout.write(json.dumps(out))
    except Exception:  # fail open: never block or break a prompt
        pass
    return 0


def main(argv=None, stdin=sys.stdin, stdout=sys.stdout, env=os.environ) -> int:
    parser = argparse.ArgumentParser(prog="noulo_router.py", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    hook = sub.add_parser("hook", help="run as a Claude Code hook (reads the event on stdin)")
    hook.add_argument("kind", choices=["prompt", "session-start"])
    for name in ("why", "correct"):
        p = sub.add_parser(name)
        p.add_argument("--session", required=True)
        p.add_argument("--data", help="the plugin data directory")
        if name == "correct":
            p.add_argument("words", nargs="*")
    status = sub.add_parser("status", help="which Claude models ran when, and their tokens")
    status.add_argument("scope", nargs="?", choices=["all"], help="all sessions, not just one")
    status.add_argument("--all", action="store_true", help="same as the 'all' scope")
    status.add_argument("--session", help="the session to report on")
    status.add_argument("--hours", type=float, default=24, help="window for all sessions")
    status.add_argument("--data", help="the plugin data directory")
    status.add_argument("--watch", action="store_true", help="redraw until Ctrl+C")
    status.add_argument("--interval", type=float, default=2.0, help="seconds between redraws")
    status.add_argument("--count", type=int, help=argparse.SUPPRESS)
    for name in ("disable", "enable"):
        p = sub.add_parser(name, help=f"{name} routing in every session")
        p.add_argument("--data", help="the plugin data directory")
    for name in ("teach", "eval"):
        p = sub.add_parser(name, help=f"{name} from a JSON Lines file of labelled prompts")
        p.add_argument("file", type=Path)
        p.add_argument("--url", default=DEFAULT_URL)
        if name == "eval":
            p.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "hook":
        return _hook(args.kind, stdin, stdout, env)

    api_key = env.get("NOULO_ROUTER_API_KEY") or None
    if args.command in ("disable", "enable"):
        settings = Settings.from_env(_cli_env(env, args.data))
        stdout.write(set_enabled(settings, args.command == "enable") + "\n")
        return 0
    if args.command == "status":
        settings = Settings.from_env(_cli_env(env, args.data))
        session = None if (args.all or args.scope == "all") else args.session
        if not args.watch:
            stdout.write(status_report(settings, session_id=session, hours=args.hours) + "\n")
            return 0
        drawn = 0
        try:
            while args.count is None or drawn < args.count:
                report = status_report(settings, session_id=session, hours=args.hours)
                stdout.write("\x1b[2J\x1b[H" + report + "\n")
                stdout.flush()
                drawn += 1
                if args.count is None or drawn < args.count:
                    time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
        return 0
    if args.command in ("why", "correct"):
        settings = Settings.from_env(_cli_env(env, args.data))
        if args.command == "why":
            stdout.write(why(settings, args.session) + "\n")
            return 0
        code, message = correct(settings, args.session, args.words, api_key)
        stdout.write(message + "\n")
        return code

    policy = load_policy(Path(env.get("CLAUDE_PLUGIN_ROOT") or PLUGIN_ROOT) / "router.json")
    rows = read_labelled(args.file)
    try:
        if args.command == "teach":
            result = NouloClient(args.url, api_key, timeout=120).teach(teaching_items(rows, policy))
            stdout.write(f"Taught {result['imported']} examples from {len(rows)} prompts.\n")
            for failure in result.get("failed", []):
                stdout.write(f"  item {failure['index']}: {failure['message']}\n")
            return 0
        report = evaluate_file(lambda: NouloClient(args.url, api_key, timeout=30), rows, policy)
    except NouloUnavailable as e:
        stdout.write(f"noulo at {args.url}: {e}\n")
        return 1
    try:
        report["model"] = NouloClient(args.url, api_key, timeout=5).info().get("model")
    except NouloUnavailable:
        report["model"] = None
    if args.json:
        stdout.write(json.dumps(report, indent=2) + "\n")
    else:
        lat = report["latency_ms"]
        stdout.write(
            f"{report['items']} prompts · model {report['model']}\n"
            f"task type accuracy {report['task_accuracy']:.1%} · "
            f"tier accuracy {report['tier_accuracy']:.1%}\n"
            f"latency per prompt p50 {lat['p50']} ms · p95 {lat['p95']} ms · max {lat['max']} ms\n"
        )
        for gold, preds in sorted(report["confusion"].items()):
            stdout.write(f"  {gold} → {preds}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
