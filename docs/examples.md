# Examples: Noul, Choice and Score

Every example below was run against a local `noulo` server with the bundled default model
(`nli-deberta-v3-xsmall-int8`, CPU). The outputs shown are real. Each primitive is shown in
the interactive session, CLI, curl, JavaScript and Python (over HTTP, and in-process with no
server).

Start the server first: `noulo start` (or `noulo serve`).

---

## Noul: is this statement true for the input?

Returns `P(proposition is true | input)`: `1.0` = yes, `0.5` = the input doesn't say,
`0.0` = no.

| Input | Proposition | Result |
|---|---|---|
| The invoice has remained unpaid for 120 days. | The customer has an overdue payment. | **0.96** (yes) |
| I've tried resetting my password three times and still can't log in. | The user is locked out of their account. | **0.95** (yes) |
| All payments are up to date. | The customer has an overdue payment. | **0.03** (no) |

**Interactive session**

```text
❯ /noul The customer has an overdue payment.
noul ❯ The invoice has remained unpaid for 120 days.
  ● Yes       0.96  ━━━━━━━━━━━━━━━━━━━━━━━─
noul ❯ All payments are up to date.
  ● No        0.03  ━───────────────────────
```

**CLI**

```bash
noulo noul -i "The invoice has remained unpaid for 120 days." \
           -p "The customer has an overdue payment."
# 0.9635032952370379
```

**curl**

```bash
curl -s http://127.0.0.1:8787/api/v1/noul -H 'Content-Type: application/json' -d '{
  "input": "The invoice has remained unpaid for 120 days.",
  "proposition": "The customer has an overdue payment."
}'
# {"type":"noul","value":0.9635032952370379}
```

**JavaScript**

```javascript
async function noul(input, proposition) {
  const res = await fetch("http://127.0.0.1:8787/api/v1/noul", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input, proposition }),
  });
  if (!res.ok) throw new Error((await res.json()).error.message);
  return (await res.json()).value;
}

const p = await noul(
  "I've tried resetting my password three times and still can't log in.",
  "The user is locked out of their account.",
);
if (p > 0.8) escalateToIdentityTeam(); // p ≈ 0.95
```

**Python (HTTP)**

```python
import httpx

r = httpx.post("http://127.0.0.1:8787/api/v1/noul", json={
    "input": "All payments are up to date.",
    "proposition": "The customer has an overdue payment.",
})
print(r.json()["value"])   # 0.03
```

**Python (in-process)**

```python
from noulo import Noulo

with Noulo() as engine:
    overdue = engine.noul("The invoice has remained unpaid for 120 days.",
                          "The customer has an overdue payment.")   # 0.96
```

**Using the value.** Treat it as a probability: pick thresholds for your use case
(e.g. act at ≥ 0.8, ask a human between 0.35 and 0.8). Values near 0.5 mean the input
doesn't settle the question.

---

## Choice: which of your options fits?

Returns exactly one of the IDs you supplied. IDs are any strings, so use whatever your code
already understands.

| Input | Question | Options | Result |
|---|---|---|---|
| The customer says their subscription payment was taken twice. | Which department should handle this? | `A` Billing · `B` Technical Support · `C` Sales | **A** |
| Can you tell me when my order will arrive? | What does the customer want? | `track_order` · `cancel_order` · `refund` · `complaint` | **track_order** |

**Interactive session**

```text
❯ /choice Which department should handle this?
  Options (A=Billing, B=Sales - or just Billing, Sales): A=Billing, B=Technical Support, C=Sales
choice ❯ The customer says their subscription payment was taken twice.
  → A  Billing
choice ❯ /verbose
choice ❯ My app crashes every time I open settings.
  → A  Billing
    A Billing                    ━━━━━━━━━━────────────── 0.43
    B Technical Support          ━━━━━━━━──────────────── 0.32
    C Sales                      ━━━━━━────────────────── 0.25
choice ❯ /correct B
  ✔ Thanks - remembered as 'B'. Similar inputs will lean this way.
```

That second answer is wrong, and the probabilities show it: 0.43 / 0.32 / 0.25 is a model
that isn't sure. Low, spread-out probabilities are the moment to teach it: `/correct B`
stores the right answer, and similar inputs lean towards it from then on (see
[learning.md](learning.md)).

**CLI**

```bash
noulo choice -i "Can you tell me when my order will arrive?" \
             -q "What does the customer want?" \
             -o track_order="Track an order" -o cancel_order="Cancel an order" \
             -o refund="Get a refund" -o complaint="Make a complaint"
# track_order

# with probabilities
noulo choice ... --diagnostics
# "probabilities": {"track_order": 0.955, "cancel_order": 0.016, "refund": 0.017, "complaint": 0.012}
```

**curl**

```bash
curl -s http://127.0.0.1:8787/api/v1/choice -H 'Content-Type: application/json' -d '{
  "input": "The customer says their subscription payment was taken twice.",
  "question": "Which department should handle this?",
  "choices": [
    {"id": "A", "text": "Billing"},
    {"id": "B", "text": "Technical Support"},
    {"id": "C", "text": "Sales"}
  ]
}'
# {"type":"choice","value":"A"}
```

**JavaScript**: route support tickets.

```javascript
const QUEUES = [
  { id: "billing", text: "Billing and payments" },
  { id: "tech", text: "Technical problem or bug" },
  { id: "account", text: "Login or account access" },
  { id: "other", text: "Anything else" },
];

async function route(ticketText) {
  const res = await fetch("http://127.0.0.1:8787/api/v1/choice", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input: ticketText, question: "Which queue should handle this ticket?",
                           choices: QUEUES }),
  });
  return (await res.json()).value; // always one of the ids above
}
```

**Python (HTTP)**

```python
import httpx

r = httpx.post("http://127.0.0.1:8787/api/v1/choice", json={
    "input": "Can you tell me when my order will arrive?",
    "question": "What does the customer want?",
    "choices": [{"id": "track_order", "text": "Track an order"},
                {"id": "cancel_order", "text": "Cancel an order"},
                {"id": "refund", "text": "Get a refund"},
                {"id": "complaint", "text": "Make a complaint"}],
})
print(r.json()["value"])   # track_order
```

**Python (in-process)**

```python
from noulo import Noulo

with Noulo() as engine:
    queue = engine.choice("The customer says their subscription payment was taken twice.",
                          "Which department should handle this?",
                          {"A": "Billing", "B": "Technical Support", "C": "Sales"})   # 'A'
```

**Tips.** Write options as short, distinct descriptions ("Get a refund", not "refund"). Add
a catch-all option ("Anything else") when inputs may fit none of them. The answer is always
one of your IDs, never an invented one.

---

## Score: where does the input sit on your rubric?

Returns `0.0`–`1.0`: the probability-weighted level divided by the highest level. The
nearest level is `rubric[round(value * (len(rubric) - 1))]`.

| Input | Question | Rubric (lowest → highest) | Result |
|---|---|---|---|
| The production system is unavailable for every customer. | How severe is this incident? | insignificant, low, medium, high, critical | **0.77** (high) |
| Absolutely love it, best purchase I've made this year! | How does the customer feel? | very negative, negative, neutral, positive, very positive | **0.98** (very positive) |
| The delivery was late and the box arrived damaged. | How does the customer feel? | *(same)* | **0.14** (very negative) |

**Interactive session**

```text
❯ /score How severe is this incident?
  Rubric, lowest to highest (low, medium, high): insignificant, low, medium, high, critical
score ❯ The production system is unavailable for every customer.
  ◆ 0.77  high  ━━━━━━━━━━━━━━━━━━──────
```

**CLI**

```bash
noulo score -i "The delivery was late and the box arrived damaged." \
            -q "How does the customer feel?" \
            -r "very negative,negative,neutral,positive,very positive"
# 0.14
```

**curl**

```bash
curl -s http://127.0.0.1:8787/api/v1/score -H 'Content-Type: application/json' -d '{
  "input": "The production system is unavailable for every customer.",
  "question": "How severe is this incident?",
  "rubric": ["insignificant", "low", "medium", "high", "critical"]
}'
# {"type":"score","value":0.765652124071898}
```

**JavaScript**: turn the value back into a level.

```javascript
const rubric = ["insignificant", "low", "medium", "high", "critical"];
const res = await fetch("http://127.0.0.1:8787/api/v1/score", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ input: incidentText, question: "How severe is this incident?", rubric }),
});
const { value } = await res.json();
const level = rubric[Math.round(value * (rubric.length - 1))]; // "high"
```

**Python (HTTP)**

```python
import httpx

rubric = ["very negative", "negative", "neutral", "positive", "very positive"]
r = httpx.post("http://127.0.0.1:8787/api/v1/score", json={
    "input": "Absolutely love it, best purchase I've made this year!",
    "question": "How does the customer feel?",
    "rubric": rubric,
})
value = r.json()["value"]                              # 0.98
print(rubric[round(value * (len(rubric) - 1))])        # very positive
```

**Python (in-process)**

```python
from noulo import Noulo

with Noulo() as engine:
    severity = engine.score("The production system is unavailable for every customer.",
                            "How severe is this incident?",
                            ["insignificant", "low", "medium", "high", "critical"])   # 0.77
```

**Tips (these matter for Score).**

- **Phrase the question neutrally, and never put a rubric word in it.** "How positive is
  the customer?" makes the default model lean towards "positive" for *every* input: a
  late, damaged delivery scored 0.72. "How does the customer feel?" gives 0.14 for the same
  input.
- Use short, clearly ordered levels, lowest first.
- **Mixed feelings are hard for small models:** "The product is okay but the delivery took
  forever" scores ~0.74. For nuanced scoring use the **Most accurate** model (Score error
  0.115 vs 0.205), or teach it with `/correct` or `POST /api/v1/feedback`.

---

## The unified endpoint

All three primitives are also available through one endpoint, selected by `type`:

```bash
curl -s http://127.0.0.1:8787/api/v1/evaluate -H 'Content-Type: application/json' -d '{
  "type": "choice",
  "input": "The customer says their subscription payment was taken twice.",
  "question": "Which department should handle this?",
  "choices": [{"id": "A", "text": "Billing"}, {"id": "B", "text": "Technical Support"}]
}'
```

```python
from noulo import evaluate

evaluate({"type": "noul", "input": "...", "proposition": "..."})
evaluate({"type": "choice", "input": "...", "question": "...", "choices": [...]})
evaluate({"type": "score", "input": "...", "question": "...", "rubric": [...]})
```

## Teaching it (learning)

With learning on, every response carries an `X-Record-Id` header (`--json` in the CLI shows
it as `recordId`). Send the right answer back and similar inputs lean that way:

```bash
curl -s http://127.0.0.1:8787/api/v1/feedback -H 'Content-Type: application/json' \
     -d '{"recordId": "86a207f6ecc5...", "expected": "refund"}'
```

In the interactive session, just type `/good`, `/bad` or `/correct <answer>` after a result.
