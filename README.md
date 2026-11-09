# TinyTask

[![CI](https://github.com/eduard-s/tiny-task/actions/workflows/ci.yml/badge.svg)](https://github.com/eduard-s/tiny-task/actions/workflows/ci.yml)

TinyTask trains a local text classifier and measures which requests it can accept.
Use it for repeated classification tasks, such as support queues or intent routing.

The question is practical: can this candidate handle enough requests at your target accuracy?
TinyTask returns `accept` or `review`, then evaluates the saved candidate against separate test data.

This is an alpha Python package. The proposed PyPI name is `tiny-task-py`.
Install from the repository until the first registry release.

## Try the support inbox

Use Python 3.11 or later. Run these commands:

```sh
git clone https://github.com/eduard-s/tiny-task.git
cd tiny-task
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/tiny-task demo --output artifacts/demo
```

Open `artifacts/demo/index.html`. The demo trains a local classifier without an API key or model download.
It routes cancellation, refund, delivery, and password messages.
A hamster-haiku request goes to review.

The example data is authored. Its small test set returns `needs_data`, even when the example messages route correctly.

Use the saved candidate on your own CSV inbox:

```sh
.venv/bin/tiny-task predict --model artifacts/demo/model \
  --input examples/inbox.csv --output artifacts/predictions.jsonl
```

The CSV needs `id` and `text` columns. JSONL uses the same fields.
The output keeps the ID, proposed label, decision, scores, and reason.
It does not change the inbox or send messages to an external service.

## Why choose TinyTask

- Get training, threshold selection, evaluation, and export through one small API.
- Keep training, calibration, and test data separate.
- Refuse known ID, group, and normalized-text overlap between splits.
- Send unsupported requests to review instead of forcing a label.
- Read accepted errors, coverage, unknown acceptance, and uncertainty intervals in one report.
- Bind the report to the saved model with file hashes.

The default backend uses TF-IDF and logistic regression.
The optional backend does actual SetFit fine-tuning, then preserves the encoder and numeric classifier head.

TinyTask reuses existing learning methods. Its contribution is the decision workflow around a trained candidate.
Read the [comparison](https://github.com/eduard-s/tiny-task/blob/main/docs/COMPARISON.md) before a novelty claim.

## Train your task

Prepare three JSONL files. Keep related examples in one split.
Each labeled row uses this format:

```json
{"id":"train-1","text":"Cancel my subscription","label":"cancel","group_id":"conversation-1","label_source":"human"}
```

Use `__unknown__` for unsupported requests in calibration and test data.
Training data needs at least two known labels.
Teacher labels can train the model. Calibration and test data need reviewed or dataset labels.

```sh
.venv/bin/tiny-task train \
  --train examples/support/train.jsonl \
  --calibration examples/support/calibration.jsonl \
  --min-calibration-accepted 8 --output artifacts/my-task

.venv/bin/tiny-task evaluate --model artifacts/my-task \
  --test examples/support/test.jsonl --output artifacts/evaluation.json
```

Evaluation exits with `0` for `ready`, `1` for `reject`, and `3` for `needs_data`.
Execution errors use code `2`. The supplied small example returns code `3`.

Use a matching successful report to enforce its criteria during model load:

```sh
.venv/bin/tiny-task predict --model artifacts/my-task \
  --release-report artifacts/evaluation.json --text "Please cancel my subscription"
```

The command refuses a rejected, incomplete, or mismatched report.
Without `--release-report`, prediction uses a candidate without release approval.

## Python API

```python
from tiny_task import PolicyConfig, TinyTask, read_records

task = TinyTask.fit(
    read_records("examples/support/train.jsonl"),
    read_records("examples/support/calibration.jsonl"),
    policy_config=PolicyConfig(min_calibration_accepted=8),
)
task.save("artifacts/api-model")
report = task.evaluate(read_records("examples/support/test.jsonl"))
task.write_report("artifacts/api-report.json", report)
print(task.predict("Write a haiku about a sleepy hamster"))
```

## Optional SetFit training

```sh
.venv/bin/python -m pip install -e '.[neural]'
.venv/bin/tiny-task train --backend setfit --allow-download \
  --train examples/support/train.jsonl \
  --calibration examples/support/calibration.jsonl \
  --min-calibration-accepted 8 --output artifacts/neural-task
```

The first checkpoint download needs `--allow-download`.
The checkpoint revision is fixed. Later inference uses local files on the CPU.
SetFit can select a training accelerator. The benchmark records the actual training device.

## Evidence and limits

The [development benchmark](https://github.com/eduard-s/tiny-task/blob/main/benchmarks/README.md) uses a pinned CLINC150 cohort.
It compares three acceptance policies on each trained classifier.
The SetFit run includes changed encoder hashes and a training history.

Both current candidates fail the default release criteria. The report preserves that result.
The result does not establish broad superiority over SetFit or Prompt2Model.

The first scope is English, one label per request, and short text.
Near-duplicate paraphrases can cross a split without an explicit group ID. Model scores are not calibrated probabilities of correctness.
Wilson intervals describe sample uncertainty under independent sampling assumptions. They do not establish future accuracy after a distribution change.

No live LLM baseline or measured API savings are included yet.
TinyTask does not generate training labels, invoke a fallback model, or do business actions.

## Documents

- [Prompt2Model and SetFit comparison](https://github.com/eduard-s/tiny-task/blob/main/docs/COMPARISON.md)
- [MVP plan and acceptance criteria](https://github.com/eduard-s/tiny-task/blob/main/docs/MVP_PLAN.md)
- [Python stack and architecture](https://github.com/eduard-s/tiny-task/blob/main/docs/ARCHITECTURE.md)
- [Datasets and partition rules](https://github.com/eduard-s/tiny-task/blob/main/docs/DATASETS.md)
- [Benchmark protocol and results](https://github.com/eduard-s/tiny-task/blob/main/benchmarks/README.md)
- [API and artifact contract](https://github.com/eduard-s/tiny-task/blob/main/docs/API.md)
- [Demo and user trial](https://github.com/eduard-s/tiny-task/blob/main/docs/DEMO.md)
- [Development](https://github.com/eduard-s/tiny-task/blob/main/CONTRIBUTING.md)
- [Prepared PyPI release](https://github.com/eduard-s/tiny-task/blob/main/PUBLISHING.md)

MIT. Read the [third-party notices](https://github.com/eduard-s/tiny-task/blob/main/THIRD_PARTY.md).
