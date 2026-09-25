# CLI reference

```text
noulo [--url URL] [--api-key KEY] [--env-file FILE] COMMAND ...
```

| Global option | Meaning |
|---|---|
| `--url` | Server to talk to (default: from `NOULO_HOST`/`NOULO_PORT`, or `$NOULO_URL`) |
| `--api-key` | Bearer key, when the server has `NOULO_API_KEY` set (default: from settings) |
| `--env-file` | Settings file (default `./.env`, or `$NOULO_ENV_FILE`) |

**Exit codes:** `0` success · `1` API/service error (message on stderr) · `2` usage or
configuration error · `3` server not reachable.

Running `noulo` with no command opens the [interactive session](#interactive-session).

---

## Service

### `noulo start`

Start the background service (with the frontend). When run from a terminal it also opens
the browser; from scripts and agents it doesn't.

| Flag | |
|---|---|
| `--headless` | API only; no frontend, no browser |
| `--foreground` | Run in this terminal instead of the background |
| `--no-browser` | Don't open the browser |
| `--host`, `--port` | Bind address (non-loopback also needs `--allow-network`) |
| `--model ID` | Model for this run (the default comes from `NOULO_MODEL`) |
| `--learning` / `--no-learning` | Override `NOULO_LEARNING_ENABLED` |
| `--threads N`, `--max-concurrency N`, `--log-level L` | Runtime tuning |
| `--allow-network` | Permit binding a non-loopback host |

The service writes its pidfile and log to `NOULO_RUN_DIR` (default `data/run/`). It waits
until `/health` reports ready, and fails with a pointer to the log if the model can't load.

### `noulo stop` · `noulo restart` · `noulo status` · `noulo logs [-n LINES]`

`stop` sends a graceful shutdown: it stops accepting requests, lets in-flight inference
finish, releases the model, and closes the listener. `restart` keeps the host, port and
frontend mode, and re-reads `.env`.

### `noulo serve`

Run the API in the foreground, headless, for process managers (systemd, launchd, Docker, CI).
It accepts the same flags as `start`, plus `--ui` to also serve the frontend.

---

## Decisions

All decision commands call the running server. Add `--json` for the full response
(including `recordId` when learning is on), or `--diagnostics` for probabilities and the
learning influence.

```bash
noulo noul    -i INPUT -p PROPOSITION
noulo choice  -i INPUT -q QUESTION -o ID=TEXT [-o ID=TEXT ...]
noulo score   -i INPUT -q QUESTION -r level1,level2,... [-r more ...]
noulo evaluate [FILE|-]              # a JSON /api/v1/evaluate request
```

`-i -` reads the input from stdin: `cat ticket.txt | noulo noul -i - -p "The customer is angry."`

---

## Models: `noulo model` (alias `noulo models`)

| Command | |
|---|---|
| `noulo model` | Picker: 3 basic models, 3 larger (0.5-1 GB) models, your registered models, and "plug in your own" instructions. Each row shows download size, quantisation, measured RAM and accuracy |
| `noulo model list` | All known models (`*` = active), install state |
| `noulo model use ID` | Download if needed → save `NOULO_MODEL` → restart the managed service (or switch a manually started server live) |
| `noulo model download [ID ...] [--missing]` | Download catalog models (default: current model + embedder) |
| `noulo model add-endpoint --id ID --base-url URL --model NAME [--api-key-env VAR]` | Register an OpenAI-compatible endpoint |
| `noulo model add-onnx --id ID --path DIR [--quantization Q] [--kind nli\|llm]` | Register your own ONNX NLI model, or (`--kind llm`) a causal LLM export with a chat template |
| `noulo model remove ID` | Remove a registered endpoint or custom model |

When you choose a model in the picker:

1. If it isn't installed, it's downloaded (with its packaged tuning).
2. `NOULO_MODEL=<id>` is written to `.env`.
3. If `noulo start` is running, the service restarts with the new model and the frontend.
   If a server you started some other way is running, it switches live through the API.
   Otherwise you're offered to start it.

---

## Learning

| Command | |
|---|---|
| `noulo learning status` | Enabled flag and memory stats |
| `noulo learning on` / `off` | Apply live (if running) **and** save `NOULO_LEARNING_ENABLED` |
| `noulo learning records [--limit N] [--type noul\|choice\|score]` | Stored cases |
| `noulo learning clear --yes` | Delete all stored cases |
| `noulo feedback RECORD_ID EXPECTED` | Correct outcome: `true`/`false` or `0.0–1.0` for Noul/Score, an option ID for Choice |
| `noulo teach FILE [--dry-run]` | Teach every labelled example in a `.jsonl`/`.json` file; bad lines are reported by line number (exit 1). See [learning.md](learning.md#teach-from-a-file) |

---

## Configuration: `noulo config`

| Command | |
|---|---|
| `noulo config show [--json]` | Effective settings (secrets shown as `***`) |
| `noulo config get KEY` | One setting (`port`, `NOULO_PORT` and `noulo_port` all work) |
| `noulo config set KEY VALUE` | Validate and write to `.env` (comments and other lines kept; file mode 600) |
| `noulo config unset KEY` | Remove from `.env` (falls back to the default) |
| `noulo config path` | Which `.env` file is used |

Most settings apply on the next start. Use `noulo restart` after changing them.

---

## Other commands

| Command | |
|---|---|
| `noulo health` | Prints `ok` (exit 0) or the not-ready state (exit 1) |
| `noulo info [--json]` | Engine, version, model, backend, quantisation, learning |
| `noulo openapi [-o FILE]` | Write the OpenAPI spec without loading a model |
| `noulo benchmark [...]` | See below |

### `noulo benchmark`

```bash
noulo benchmark                                   # active model, test split
noulo benchmark --model a --model b               # several models
noulo benchmark --model my-nli --tune             # tune templates + calibration first (calibration split)
noulo benchmark --all --download --tune --compare report.md
```

| Flag | |
|---|---|
| `--model ID` (repeatable) · `--all` | Which models |
| `--download` | Download missing catalog models |
| `--tune` | Fit hypothesis templates/strategies and Noul calibration on the **calibration split**, then write `profile.json` + `calibration.json` into the model dir |
| `--split test\|calibration` | Which split to report on (default `test`) |
| `--data-dir DIR` | Dataset directory (default `benchmark/data`) |
| `--out DIR` | JSON results (default `benchmark/results`) |
| `--compare FILE` | Write a markdown comparison table |

Each model is measured in a fresh subprocess, so peak RAM and cold start are real.

---

## Interactive session

Run `noulo` with no command. You get a Claude Code-style session: a status banner, an input
line with history (↑/↓, saved to `~/.noulo/history`), a completion menu that appears as soon
as you type `/`, and a status bar at the bottom.

```text
╭─ ✻ noulo 0.1.0a1 - local decision engine ─────╮
│  server    ● ready  http://127.0.0.1:8787     │
│  model     nli-deberta-v3-xsmall-int8 (INT8)  │
│  learning  on                                 │
╰───────────────────────────────────────────────╯
❯ /noul The customer has an overdue payment.
  Noul mode - every line you type is checked against: “The customer has an overdue payment.”
noul ❯ The invoice has been unpaid for 120 days.
  ● Yes       0.95  ━━━━━━━━━━━━━━━━━━━━━━━─
  57 ms · nli-deberta-v3-xsmall-int8
noul ❯ /choice Which department should handle this?
  Options (A=Billing, B=Sales - or just Billing, Sales): A=Billing, B=Technical Support, C=Sales
choice ❯ I was charged twice for my subscription.
  → A  Billing
choice ❯ /good
  ✔ Thanks - remembered as correct. Similar inputs will lean this way.
```

If the server isn't running, the session offers to start it.

**Simple requests.** Anything that doesn't start with `/` is evaluated as the input for the
current mode. With no mode yet, you're asked what to check (Noul, Choice or Score) and for
the proposition, question, options or rubric. The mode then sticks, so you can keep typing
inputs.

**Slash commands**

| Group | Command | |
|---|---|---|
| Decide | `/noul [proposition]` | Check whether a statement is true for each input |
| | `/choice [question]` | Pick one of your options for each input (asks for `A=Billing, B=Sales` or `Billing, Sales`) |
| | `/score [question]` | Place each input on a rubric (asks for `low, medium, high`) |
| | `/context` | Show the current mode and its question, options or rubric |
| | `/proposition`, `/question`, `/options`, `/rubric` | Change one part of the current mode |
| Teach | `/good` · `/bad` | The last answer was right or wrong (`/bad` asks for the right option or level) |
| | `/correct <answer>` | Give the right answer: `true`/`false`/`0.8`, an option id or text, or a level name |
| | `/teach <file>` | Teach every labelled example in a `.jsonl` file |
| | `/learning [on\|off]` | Show or switch learning (applied live and saved) |
| | `/memory [clear]` | Show or clear remembered cases |
| Configure | `/model` | Arrow-key model picker with size, quantisation, measured RAM and accuracy per model |
| | `/model <id>` · `/model list` | Switch directly · list everything |
| | `/config` | Interactive settings editor: pick a setting with ↑/↓, type the new value |
| | `/config show` · `get <key>` · `set <key> <value>` · `unset <key>` | Direct forms (validated; secrets are masked) |
| Service | `/status` · `/start` · `/stop` · `/restart` · `/logs` · `/open` | Manage the background service; `/open` opens the frontend |
| View | `/verbose` | Show probability bars, the raw Noul value and learning influence |
| | `/json` | Print raw JSON responses (including `recordId`) |
| | `/clear` · `/help` · `/exit` | |

After `/config set` or `/config`, noulo offers to restart the service so the change takes
effect. `model` and `learning_enabled` apply live without a restart.

**Keys:** `Tab` moves through the completion menu (`Enter` takes the highlighted one), `↑/↓` browse history, `Ctrl+C` clears the line,
`Ctrl+D` or `/exit` quits. In menus: `↑/↓` or `j/k` move, `1`-`9` jump, `Enter` selects,
`Esc` cancels.

**Scripting:** when stdin isn't a terminal, the session reads one line at a time, so you
can pipe a script:

```bash
printf '/noul The customer is unhappy.\nMy order never arrived.\nThanks, all good!\n' | noulo
```
