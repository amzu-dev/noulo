# Getting started

## 1. Install

You need Python 3.10+, [uv](https://docs.astral.sh/uv/) (recommended) or pip, and
[Git LFS](https://git-lfs.com) for the bundled model weights.

```bash
git lfs install                 # once per machine
git clone <your-repo-url> noulo
cd noulo
make install                    # = uv sync + git lfs pull + noulo model download --missing
```

`make install` is safe to re-run. If you cloned without LFS, the last step downloads the
two bundled models (about 110 MB) from Hugging Face instead.

Without `make`:

```bash
uv sync
uv run noulo model download --missing
```

With pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                       # add ".[qdrant]" or ".[chroma]" for those vector stores
noulo model download --missing
```

> The examples below write `noulo ...`. With uv, either prefix commands with `uv run`,
> activate `.venv`, or install the CLI globally with `uv tool install .`.

## 2. Start the service

```bash
noulo start
```

This loads the model once, runs a readiness check, binds `127.0.0.1:8787`, starts in the
background and opens the frontend at <http://127.0.0.1:8787/ui/>.

```text
Starting noulo with model nli-deberta-v3-xsmall-int8 ...
noulo is running at http://127.0.0.1:8787 (pid 60351)
Frontend: http://127.0.0.1:8787/ui/
Logs: /path/to/noulo/data/run/noulo.log
```

| Want | Command |
|---|---|
| API only, no frontend | `noulo start --headless` |
| Run in this terminal (Ctrl+C to stop) | `noulo start --foreground` or `noulo serve` (headless) |
| Check it | `noulo status` · `noulo health` · `curl localhost:8787/health` |
| Stop / restart | `noulo stop` · `noulo restart` |
| Follow the log | `noulo logs -n 100` |

## 3. Ask it something

```bash
noulo noul -i "I checked my account and you have taken the subscription payment twice." \
           -p "The customer reports being charged more than once."
# 0.899

noulo choice -i "The customer says their subscription payment was taken twice." \
             -q "Which department should handle this?" \
             -o A=Billing -o "B=Technical Support" -o C=Sales
# A

noulo score -i "The production system is unavailable for every customer." \
            -q "How severe is this incident?" -r insignificant,low,medium,high,critical
# 0.766
```

The same requests over HTTP:

```bash
curl -s localhost:8787/api/v1/noul -H 'Content-Type: application/json' \
  -d '{"input":"The invoice has remained unpaid for 120 days.","proposition":"The customer has an overdue payment."}'
# {"type":"noul","value":0.968...}
```

And from Python, in-process with no server:

```python
from noulo import evaluate
evaluate({"type": "choice", "input": "Charged twice", "question": "Which team?",
          "choices": [{"id": "A", "text": "Billing"}, {"id": "B", "text": "Sales"}]})
# {'type': 'choice', 'value': 'A'}
```

## 4. Use the interactive shell

Run `noulo` with no arguments:

```text
noulo 0.1.0 - local decision engine. Type /help for commands, /quit to exit.
noulo> /noul
  Input: The invoice has remained unpaid for 120 days.
  Proposition: The customer has an overdue payment.
0.968
noulo> /model
```

## 5. Pick a model

```bash
noulo model
```

```text
Current model: nli-deberta-v3-xsmall-int8

Choose a model:
  1) Balanced (bundled)   nli-deberta-v3-xsmall-int8         [installed]  <- current
  2) Fast & light         nli-minilm2-l6-int8                [downloads when selected]
  3) Best at Choice       zeroshot-deberta-v3-xsmall-int8    [downloads when selected]
  4) Plug in your own model (OpenAI-compatible endpoint or ONNX)
Select 1-4 (Enter keeps the current model):
```

A model that isn't installed is downloaded and saved as `NOULO_MODEL` in `.env`, and the
running service restarts with it and the frontend. Option 4 explains how to use an
OpenAI-compatible endpoint or your own ONNX model. See [models.md](models.md).

## 6. Teach it

Learning is on by default. Get the record id of an evaluation and send the correct answer:

```bash
noulo noul -i "My parcel never arrived." -p "The customer is satisfied." --json
# { "type": "noul", "value": 0.07, "recordId": "5f0c..." }
noulo feedback 5f0c... false
```

Similar future inputs for the same proposition now lean towards your answer. See
[learning.md](learning.md).

## Next steps

- [CLI reference](cli.md)
- [REST API reference](api.md)
- [Configuration](configuration.md)
