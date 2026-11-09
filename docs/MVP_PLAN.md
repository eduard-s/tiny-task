# MVP plan

## First user and outcome

The first user is a Python developer who uses an LLM to select one label from a stable set.
Examples include support queues, intent routing, and document categories.

The outcome is a local candidate with an explicit review route and evidence for an acceptance decision.

## Scope and current state

| Deliverable | State | Acceptance evidence |
| --- | --- | --- |
| Python API and CLI | Implemented | Train, evaluate, predict, and demo commands |
| TF-IDF baseline | Implemented | Actual training and numeric export parity tests |
| SetFit backend | Implemented | Actual fine-tuning with changed encoder hashes |
| Separate data splits | Implemented | ID, group, and normalized-text overlap tests |
| Abstention policy | Implemented | Calibration-only threshold selection |
| Release decision | Implemented | Accuracy interval, coverage, and unknown acceptance |
| CSV inbox demo | Implemented | Local example with review output |
| Public benchmark | Implemented | One CLINC150 development cohort |
| PyPI distribution | Implemented | Wheel, source archive, installation tests, and tag-based release workflow |
| Live teacher comparison | Planned | Pinned provider, cost record, and gold labels |
| Multilingual evaluation | Planned | MASSIVE with grouped translations |
| ONNX service | Planned | Export parity and measured deployment benefit |
| External adoption | Not established | Three proposed user trials |

## Stage 1: source alpha

1. Train a candidate from known labels.
2. Select thresholds from separate calibration examples.
3. Preserve the candidate with a file manifest.
4. Evaluate separate known and unknown test examples.
5. Route a local CSV inbox to candidate labels or review.
6. Publish positive and negative results with the recipe.

This repository implements Stage 1.
The default benchmark gate rejects both current candidates.
A rejected result does not block the alpha package release. It blocks approval of that candidate for the selected criteria.

## Stage 2: independent task evaluation

Target five independent classification tasks from at least three teams.
Reserve the final test set before threshold changes.

Collect at least 100 reviewed test requests per label where the application permits that count.
Include unsupported requests from the actual input stream.
Keep each conversation, paraphrase group, and translated example in one split.

Use three fixed training seeds and the same data budget for each backend.
Freeze the acceptance criteria before the final run.

## Proposed release criteria for a candidate

The default gate uses these criteria:

- At least 30 accepted test requests.
- At least 30 unknown test requests.
- At least one test request for each training label.
- Accepted accuracy has a Wilson 95% lower bound of at least 90%.
- Known requests have coverage of at least 40%.
- Unknown acceptance has a Wilson 95% upper bound of at most 10%.

The counts are minimum evidence rules. They are not enough to establish every per-label accuracy rate.
For a real deployment, select stricter counts and criteria from the error cost.

The gate describes the configured test sample. It does not certify future requests or a changed input distribution.

## Stage 3: teacher and cost comparison

Capture a live teacher baseline with a fixed prompt, schema, model version, and provider date.
Measure correctness against reviewed labels, not teacher agreement alone.
Include parse failures, timeouts, and abstentions.

Compare quality at matched known coverage.
Measure p50 and p95 latency, memory, training time, and total cost.
Include review or fallback cost in the cost comparison.

Do not publish a savings percentage before this run exists.

## Stage 4: user trial

Give three teams the CSV workflow and one training recipe.
Target first local predictions within 15 minutes after dependency installation.
Record obstacles and repeat use after one week.

Expand adapters or service support only after repeated requests from these users.
