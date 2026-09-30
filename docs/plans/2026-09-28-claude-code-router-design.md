# Design: noulo-router, a Claude Code plugin that routes tasks with noulo

Date: 2026-09-28 · Status: implemented · Branch: `feat/claude-code-router`

## Goal

An example use case for noulo: a Claude Code plugin that works out what kind of task each prompt
is and moves the work to a suitable Claude model and effort level. **noulo makes the decision**;
the plugin only applies it.

## What Claude Code allows (verified, Claude Code v2.1.283)

The mechanism was checked with throwaway plugins in headless (`claude -p`) and interactive (tmux)
sessions before any code was written. Models were read from `modelUsage` and the transcript;
effort from the `effort` field that `PreToolUse`, `SubagentStop` and `Stop` hooks receive.

| Mechanism | Model changes? | Effort changes? |
|---|---|---|
| A hook's output | No: there is no `model` or `effort` field, and hooks can't rewrite the prompt | No |
| Editing `model` / `effortLevel` in settings mid-session | No, not picked up until restart | No |
| Claude invokes a skill whose frontmatter sets `model` / `effort` | **No.** The switch is recorded in the transcript but the session model keeps answering (Haiku and Fable targets, headless and interactive) | **No**, it stayed `xhigh` |
| The user types that skill as `/plugin:skill <prompt>` | **Yes**, for that turn, with full chat history | Documented yes; only the model was measured |
| Claude delegates to a subagent whose frontmatter sets `model` / `effort` | **Yes** (Sonnet and Haiku ran) | **Yes** (`low` inside the subagent). Haiku 4.5 reports no effort; it doesn't support the parameter |

Other facts the design relies on:

- `UserPromptSubmit` receives the text as `prompt`. It can add `additionalContext` (Claude sees
  it) and a `systemMessage` (the user sees it). Plain stdout is also added as context.
- The `SessionStart` input didn't include a `model` field, so the hook can't reliably know the
  session model. Claude does know its own model, including a per-turn override.
- Each model has its own prompt cache. A subagent starts its own cache and leaves the parent's
  untouched. Changing the main session's model forces an uncached re-read of the conversation.
- Delegation isn't free: the session model still reads the prompt, writes the brief and relays
  the result. In the spike, delegating a trivial task cost more Opus output than doing it
  directly. Delegation pays off when it moves **up** to a stronger model.

## Decisions (agreed with the user)

1. **Mechanism:** automatic delegation to tier subagents, plus tier slash commands for switching
   inline with full history, plus a `suggest` mode. Mode is a plugin option: `auto` (default),
   `suggest`, `off`.
2. **Delegate only to stronger models.** If the tier's model is stronger than the model Claude is
   running on, Claude hands the task to that tier's subagent. Otherwise Claude does the work
   itself, and the user sees a one-line note naming the slash command that would switch inline.
3. **Tiers use model aliases** so they always resolve to the latest version:

   | Tier | Model | Effort | Delegation target |
   |---|---|---|---|
   | light | `haiku` | `low` | none (nothing is weaker than Haiku) |
   | standard | `sonnet` | `medium` | `noulo-router:standard-agent` |
   | deep | `opus` | `high` | `noulo-router:deep-agent` |
   | max | `fable` | `xhigh` | `noulo-router:max-agent` |

4. **Features are deep work:** new features start at `deep`, like debugging, design and review.
5. **Separate noulo instance** for routing, on port `8790` with its own memory file and run
   directory, so prompts don't mix with other noulo data.
6. **SessionStart only warns** when the router's noulo isn't reachable; it never starts it.
7. Default mode is `auto`.

## How a prompt is routed

```
prompt ─▶ UserPromptSubmit hook (python3, stdlib only)
            │  skip: mode off, starts with "/", starts with a tag (<task-notification> …)
            │  short follow-up (≤ 3 words): reuse the session's last tier
            ▼
          noulo on 127.0.0.1:8790 (POST /api/v1/evaluate?diagnostics=true), first 300 + last 150 chars
            Choice  task type: explain · edit · chore · feature · debug · refactor · design · review
            Noul    "The request needs substantial work across many files or a deep investigation."
            Noul    "The request involves security."
            ▼
          policy (router.json) ─▶ tier
            ▼
          auto:    additionalContext = tier, model order, "delegate to <tier>-agent if stronger"
                   systemMessage     = "noulo-router → deep · opus · high (debug, 64%)"
          suggest: systemMessage only, naming /noulo-router:<tier>
```

### Policy (in `router.json`, chosen on the calibration split)

- Task type → tier: explain, edit, chore → `light`; refactor → `standard`; feature, debug,
  design, review → `deep`.
- A deep task whose large-work probability is ≥ `max_at` (0.5) goes to `max`.
- A security probability ≥ `security_at` (0.8) makes the tier at least `deep`.
- A task-type probability below `min_confidence` (0.2) means the `standard` tier.

### Failure behaviour

- noulo unreachable or erroring: the prompt goes through unrouted, with one warning per session
  and the command that starts the router's instance. SessionStart gives the same warning.
- noulo slower than the 4 s budget: the prompt goes through unrouted, with a "took too long"
  note. The hook itself times out at 10 s.
- Any other problem (bad input, bad state file): the hook prints nothing and exits 0.

## Changes made during implementation, and why

| Planned | Built | Evidence (calibration split unless noted) |
|---|---|---|
| A Score for complexity moves tiers up and down | Dropped | Trivial and very complex prompts got the same average score on all three models tried |
| A small-work Noul moves tiers down | Dropped | No threshold improved tier accuracy: trivial prompts were already `light` |
| Router on noulo's default xsmall model | `zeroshot-deberta-v3-base-fp32` recommended | Task type 35/48 vs 20/48 on calibration; 70.0% vs 52.5% held-out |
| One Noul per task type | One Choice | Best on the base model (35/48 vs 34/48); rewording options only made it worse |
| First 1,200 + last 300 characters | First 300 + last 150 | 1,500 characters took 5.6–6.4 s on the base model; 450 take 1.5–2.1 s |
| (not planned) | Skip messages that start with a tag | A subagent's `<task-notification>` arrives as a user prompt and was being routed |
| `/noulo-router:mode` command | `userConfig` options in `/config` | Claude Code shows plugin options there and passes them to hooks |

## Measured result (held-out, 40 prompts, Apple M1 Pro)

| Router model | Task type | Tier | Latency p50 / p95 |
|---|---|---|---|
| `zeroshot-deberta-v3-base-fp32` | 70.0% | 65.0% | 414 / 514 ms |
| `nli-deberta-v3-xsmall-int8` | 52.5% | 32.5% | 146 / 237 ms |

With the base model, 10 of the 14 wrong tiers were lower than intended. In `auto` mode Claude
then just does the task on the session model. Teaching the calibration set changed neither
number: the held-out prompts aren't paraphrases of taught ones. One verified correction fixed
the corrected prompt and a paraphrase of it.

## Components

```
examples/claude-code-router/          ← plugin root
  .claude-plugin/plugin.json          ← manifest + userConfig options: mode (auto/suggest/off), url
  hooks/hooks.json                    ← SessionStart (health warning), UserPromptSubmit (route)
  scripts/noulo_router.py             ← stdlib only: client, policy, hooks, why/correct/eval CLI
  router.json                         ← policy thresholds and task → tier table
  agents/{standard,deep,max}-agent.md ← delegation targets (model + effort frontmatter)
  skills/{light,standard,deep,max}/   ← user-typed inline switch for one turn
  skills/why/, skills/correct/        ← last decision; verified feedback to noulo
  data/seed.jsonl                     ← labelled prompts to teach (`noulo teach`)
  data/eval.jsonl                     ← held-out labelled prompts, never taught
  README.md                           ← the use case walkthrough
.claude-plugin/marketplace.json       ← repo root: `/plugin marketplace add amzu-dev/noulo`
```

State (last decision per session, record ids for feedback, "warned" flags) lives in
`${CLAUDE_PLUGIN_DATA}`. The optional `NOULO_ROUTER_API_KEY` environment variable is sent as a
bearer token; the default loopback instance needs none.

### Learning loop

`/noulo-router:correct <task-type> [complexity]` sends verified feedback for the last decision
(`POST /api/v1/feedback` with the `X-Record-Id` of each evaluation), so similar prompts route
better next time. `data/seed.jsonl` bootstraps the memory; it must never contain eval prompts.

## Testing and measurement

- Test-first with pytest in `tests/test_claude_code_router.py` (64 tests): policy, skip rules,
  hook output for each mode, fail-open, a fake noulo HTTP server for the client, feedback,
  teach and eval, and checks on the plugin files (manifest, hooks, agent and skill frontmatter,
  marketplace source, no overlap between taught and held-out prompts).
- End to end with real Claude Code and noulo: delegation from a Sonnet session to Opus, no
  delegation from an Opus session, a light task inline, `/noulo-router:why`,
  `/noulo-router:correct`, and a typed `/noulo-router:deep`.
- `claude plugin validate` passes for the plugin and the marketplace.

## Addendum 2026-09-30: `/noulo-router:status`

The user asked for current, live data on which models run when and how many tokens they use.
A local web page was started, then replaced at the user's suggestion by a slash command.

- **Zero-token slash command (verified in v2.1.284):** the `UserPromptSubmit` hook recognises
  `/noulo-router:status [all]` and returns `{"decision": "block", "reason": <report>}`. Claude Code
  shows the reason in the terminal and never sends the prompt to a model: `modelUsage` was empty
  headless, and the transcript gained no assistant entries interactively. The skill file makes the
  command show in the `/` menu, and runs the same report through `!` if the hook fails.
- **Data:** the hooks append routing decisions and session starts to `events.jsonl` (session,
  transcript path, cwd, tier, task, noulo latency). The report reads each session's transcript
  and `<session>/subagents/agent-*.jsonl`, with the agent type from `agent-*.meta.json`. One API
  call is written as several lines sharing a message id, so calls are counted by id.
- **Accuracy:** headless, the four token columns matched Claude Code's `modelUsage` exactly for
  both models. For a **background** subagent, Claude Code writes only the provisional usage from the
  start of each stream, so its output count is too low: the report marks such calls (no
  `stop_reason`) with `+` and says why.
- **Live view:** `noulo_router.py status [all] --watch` redraws every 2 s in a terminal.

## Addendum 2026-09-30: every plugin command through the hook, and disable/enable

- `/noulo-router:why`, `:correct`, `:disable` and `:enable` are answered by the
  `UserPromptSubmit` hook like `:status`, so none of them calls Claude. Verified in a real session:
  no new assistant entries in the transcript, and `correct` sent exactly one feedback request.
- **Ordering found while testing:** Claude Code runs a typed skill's `` !`command` `` lines
  *before* `UserPromptSubmit` hooks. With a fallback `!` line in the skill, `disable` ran twice
  (the hook then reported "already off") and `correct` would send duplicate feedback. The
  command skills therefore contain no `!` lines. If the hook ever fails, Claude reads the skill
  and replies with one sentence saying that the hook isn't working.
- **No model override on that fallback.** A turn switched to a cheaper model re-reads the whole
  conversation without the prompt cache. At current prices (per MTok: Haiku 4.5 $1.25 cache
  write, $5 output; Opus 5.5 $0.20 cache read, $20 output) a one-sentence Haiku turn costs about
  $0.025 even at the start of a session (~20K tokens of system prompt and tools), against about
  $0.01 for the same turn on a warm Opus 5.5 session, and the gap grows with the conversation.
  Haiku's 200K context would also make that turn fail in long sessions.
- `/noulo-router:disable` writes a `disabled` file in the plugin data directory. It stops routing
  and the SessionStart warning in every session until `/noulo-router:enable`. The `mode` option in
  `/config` still applies on top.

## Out of scope

- Switching the main session's model automatically (Claude Code doesn't allow it, see above).
- Auto-starting noulo.
- Routing prompts typed inside subagents.
