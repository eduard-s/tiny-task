# API and artifact contract

## Records

`Record` contains `id`, `text`, `label`, optional `group_id`, and `label_source`.
The allowed label sources are `human`, `dataset`, `authored`, and `teacher`.
The default is `human`.

Each text needs 1 to 20,000 characters.
Each ID must be unique within its split.
Keep IDs unique across all splits.
The reserved label `__unknown__` appears only in calibration and test data.

Training needs at least two known labels.
Calibration needs every known label.
Calibration and test data need at most one record for each known group.
Clustered evaluation with several records per group is later work.
Unreviewed teacher labels cannot supply calibration or test truth.

## Training

```python
from tiny_task import PolicyConfig, TinyTask, TrainConfig, read_records

task = TinyTask.fit(
    read_records("train.jsonl"),
    read_records("calibration.jsonl"),
    config=TrainConfig(backend="tfidf", seed=42),
    policy_config=PolicyConfig(
        target_accuracy=0.90,
        max_unknown_acceptance=0.10,
        min_calibration_accepted=20,
    ),
)
task.save("candidate")
```

`TrainConfig.backend` accepts `tfidf` or `setfit`.
SetFit needs the `neural` dependencies.
The initial model download needs `allow_download=True`.
`neural_steps` limits contrastive training steps.

Calibration targets select empirical thresholds. They do not guarantee the same test or future accuracy.
If no feasible threshold pair exists, the policy sends every request to review.

## Prediction

```python
candidate = TinyTask.load("candidate")
one = candidate.predict("Please cancel my subscription")
many = candidate.predict(["Reset my password", "Tell me a joke"])
```

One string returns one result. A list returns a list of results.
An empty list returns an empty list.

| Field | Meaning |
| --- | --- |
| `label` | Accepted label, or `null` for review |
| `proposed_label` | Best class before abstention |
| `decision` | `accept` or `review` |
| `class_probability` | Classifier output for the proposed label |
| `support_similarity` | Cosine similarity with that class's training prototype |
| `reason` | Acceptance or review reason |

Review reasons include low class score, low training support, zero support, truncated neural input, and no feasible calibration policy.
The scores do not estimate a calibrated probability that the decision is correct.

Candidate prediction does not mean release approval.
Use `label` only when `decision` is `accept`. Keep `proposed_label` for diagnosis.
The application owns its routing, review queue, and fallback policy.

## Evaluation

```python
from tiny_task import GateConfig

report = candidate.evaluate(read_records("test.jsonl"), gate=GateConfig())
candidate.write_report("evaluation.json", report)
```

Save the candidate before evaluation.
The report binds the saved manifest identity, test digest, criteria, counts, intervals, predictions, and runtime.
It includes known coverage and unknown acceptance separately.

Accepted accuracy counts every accepted unknown request as an error.
Raw known accuracy and macro F1 use known requests before abstention.
A zero denominator produces `null`.

`GateConfig` controls the targets and minimum sample counts.
The status is `needs_data` when a required sample count is absent or too small.
Otherwise, failed criteria produce `reject`.
All criteria must pass for `ready`.

The intervals are two-sided Wilson 95% intervals.
They assume independent sampled requests. Related requests and repeated threshold changes can invalidate that interpretation.

## Approval during load

```python
approved = TinyTask.load("candidate", release_report="evaluation.json")
```

This call needs a matching artifact identity and passing recorded criteria.
It recomputes the gate decision from the report metrics.
It does not independently audit the reported labels or certify the report author.

## Files

The candidate directory must be new.
Existing model directories and report files are not overwritten.
Reports must stay outside the model directory.

Core model state uses JSON arrays and a vocabulary.
The neural encoder uses local Sentence Transformers files and safetensors.
The manifest covers the model files and rejects symbolic links or changed file hashes.

Keep exported models private when their vocabulary or learned parameters contain information from confidential training data.

## CLI exits

| Code | Meaning |
| --- | --- |
| `0` | Command completed, or evaluation returned `ready` |
| `1` | Evaluation returned `reject` |
| `2` | Input, dependency, artifact, or execution error |
| `3` | Evaluation returned `needs_data` |
