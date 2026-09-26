# Improving decision accuracy

Strict JSON and a passing test suite do not guarantee correct predictions. Evaluate each
model on the tasks you actually intend to automate. Never interpret the largest Choice
probability as a guarantee: even a high-probability answer can be wrong.

## Changes in this accuracy pass

- The bundled xsmall Choice profile now places the question only in the hypothesis,
  not in the input evidence as well. This strategy was selected on the existing
  **calibration split**, before measuring its held-out result.
- The frontend's Choice example uses descriptions of responsibilities rather than bare
  department names. IDs remain A (billing), B (technical support), C (sales).
- Learning now uses **verified feedback only by default**. Unverified evaluations are
  still recorded, but do not affect later answers unless you explicitly set
  `NOULO_MEMORY_OBSERVED_WEIGHT` above zero. Existing explicit overrides are retained.
- Benchmark JSON includes Choice item-level predictions and counts, so failures can be
  audited rather than hidden behind a headline accuracy number. Inputs are not copied
  into prediction details; use the dataset item ID to locate them.
- Benchmark datasets can supply a `group` identifier for paraphrases of the same case.
  A group spanning calibration and test splits is rejected before filtering the data.
  This is an author-supplied grouping check, not automatic semantic duplicate detection.
- Calibration/custom-data runs no longer overwrite the generic model-menu measurements.

## Local evidence, not an accuracy guarantee

Measurements were made on an Apple M4, macOS arm64, CPU, with learning disabled.
The original dataset and its splits were not changed or used as teaching memory.

| Model / configuration | Choice test result |
|---|---:|
| xsmall INT8, previous `answer-is` + `input+question` profile | 25 / 40 (62.5%) |
| xsmall INT8, calibration-selected `answer-is` + `input` profile | 27 / 40 (67.5%) |
| zero-shot DeBERTa base FP32, existing shipped profile | 34 / 40 (85.0%) |

The selected xsmall strategy scored 75% on the calibration split. The exploratory
calibration-only search compared the existing templates plus topic/request variants,
input versus input+question premises, and entailment, entailment-minus-contradiction
and normalized-entailment scoring. None of the new variants beat the selected existing
strategy, so no extra production scoring methods were added.

This is a small benchmark: the xsmall improvement is only two additional correct answers,
not evidence of a statistically established gain across arbitrary domains. The previous
repository comparison was measured on different hardware/configuration; its historical
numbers should not be substituted for this local before/after comparison.

The stronger FP32 model also measured 91% binary Noul accuracy and Score MAE 0.115 in
this local run. It remains **optional**, rather than silently making the default download
and runtime much larger. Download size is approximately 739 MB; memory use must be checked
on the target deployment.

## Write choices that explain the distinction

Short labels such as `Billing`, `Technical Support` and `Sales` can be ambiguous to the
small NLI model. Keep machine-readable IDs, but describe the category in `text`:

```json
{
  "type": "choice",
  "input": "The app crashes every time I open settings.",
  "question": "Which department should handle this?",
  "choices": [
    {"id": "billing", "text": "Resolving an existing charge, payment, invoice or refund"},
    {"id": "technical", "text": "Fixing a software crash, error or malfunction"},
    {"id": "sales", "text": "Getting a quote or purchasing new products or additional licences"}
  ]
}
```

In the default model, the crash example above and `I would like a quote for 100 licences.`
were wrongly routed to Billing with bare labels. The descriptive choices route them to
technical support and sales respectively. A duplicate-charge example still routes to
billing. Regression tests exercise all six orderings of these three options with learning
disabled and the actual frontend example data.

These are **known regression cases**, not an independent accuracy estimate. Descriptions
may introduce new errors elsewhere; validate them on unseen examples from your own task.
Changing question/option text also changes the memory task key, so teach using the final
wording rather than expecting feedback for the old labels to transfer.

For Score, define what each ordered rubric level means in your application. Do not assume
that a weighted-average score will always round to the policy label you intended.

## Reproduce model comparisons

From the repository root:

```bash
uv run noulo model download --missing
uv run noulo model download zeroshot-deberta-v3-base-fp32
uv run noulo benchmark \
  --model nli-deberta-v3-xsmall-int8 \
  --model zeroshot-deberta-v3-base-fp32 \
  --out benchmark/results/accuracy \
  --compare benchmark/results/accuracy/comparison.md
```

Benchmarking builds an engine without learning memory. It does not change the running
server's model. The `--out` value is a **directory**. Inspect the generated per-model JSON
alongside aggregate results. A local `models/<id>/profile.json` overrides the shipped
profile; previous `--tune` runs may have created one, so check that before comparing.

To opt into the larger model persistently (downloads it if needed and restarts a running
service):

```bash
uv run noulo model use zeroshot-deberta-v3-base-fp32
# To return to the lightweight default:
uv run noulo model use nli-deberta-v3-xsmall-int8
```

Use `--tune --data-dir PATH` only on a labelled dataset with distinct calibration and test
splits. Tuning writes local profile/calibration overrides; it does not train model weights.
Keep related paraphrases within one split, for example:

```json
{"id":"billing-01","group":"duplicate-charge-01","split":"calibration","input":"..."}
{"id":"billing-02","group":"duplicate-charge-01","split":"calibration","input":"..."}
```

These snippets show metadata only; actual examples still need the primitive's fields and
labels. Group checking is per dataset file. Missing groups remain supported for existing
datasets, and do not imply they have been checked for semantic leakage.

## What remains to do for production

1. Collect representative, consented and appropriately redacted real examples. Include
   ambiguous inputs, mixed intents, negation and confusing category boundaries.
2. Freeze an untouched test split; choose wording/model/settings using calibration or a
   separate validation split. Repeatedly optimizing against a test set makes it validation
   data and requires a new held-out test set.
3. Measure verified-learning gains separately on unseen examples. Do not teach test rows
   and then call their re-evaluation held-out accuracy.
4. Establish an application-level review/escalation policy with measured coverage and
   error rates. Noulo's Choice contract still returns exactly one supplied ID; this change
   does **not** add automatic abstention or claim calibrated Choice confidence.
5. Only consider fine-tuning after these inexpensive experiments establish a remaining
   task-specific gap. No fine-tuning or per-primitive model routing is included here.
