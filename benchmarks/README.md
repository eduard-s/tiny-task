# Public development benchmark

## Question

Does the acceptance policy reduce accepted unknown requests while keeping useful known coverage?
Does the saved candidate satisfy the selected release criteria?

The current run uses one CLINC150 development cohort and one seed.
It does not establish broad accuracy, adoption, or superiority over another training framework.

## Source and partitions

The source is [CLINC150](https://github.com/clinc/oos-eval), under CC BY 3.0.
The revision and file hash are fixed in `prepare.py`.
Read [CLINC_LICENSE.txt](CLINC_LICENSE.txt) and the [dataset protocol](../docs/DATASETS.md).

The cohort contains 200 training, 400 calibration, and 1,450 test requests.
Ten related intents form the supported task.
Five other banking intents form the related unknown group.
The original unsupported test split forms the broader unknown group.

The test set contains 300 known, 150 related unknown, and 1,000 broader unknown requests.
Calibration contains 200 known and 200 unknown requests.
The different unknown share is a distribution stress case.
Do not interpret it as a measured customer traffic distribution.

## Compared policies

Each backend trains once. All three policies use the same trained encoder, features, and classifier head.

| Policy | Rule |
| --- | --- |
| `always_predict` | Accept every proposed label. |
| `confidence_only` | Select a class-score threshold from calibration data. |
| `tinytask` | Select class-score and training-support thresholds from calibration data. |

The confidence-only control uses the same grid, targets, and minimum accepted count.
The TinyTask policy also needs positive training support.
Each test prediction stays in the metric denominators.

SetFit training uses a fixed MiniLM revision, 20 contrastive steps, and the trained logistic head.
The export copies that head's numeric coefficients.
The separate numeric export tests compare class scores with scikit-learn.

This is an acceptance-policy comparison with an actual SetFit backend.
It is not an independent comparison against every SetFit recipe or default setting.
Prompt2Model generation, a live LLM teacher, and ONNX deployment are not run.

## Results

Read [results.md](results.md) for the compact table.
Read [results-tfidf.json](results-tfidf.json) and [results-setfit.json](results-setfit.json) for the complete summaries.

The JSON includes selected record IDs, source hashes, package versions, thresholds, counts, intervals, and model identity.
The neural record includes training steps, loss history, device, and encoder hashes before and after fine-tuning.
Model weights stay outside Git.

Both current candidates fail the default release gate.
The TF-IDF support threshold adds no measured benefit over the confidence-only control on this run.
The SetFit support threshold reduces unknown acceptance, but its accepted accuracy lower bound still misses the target.

## Reproduce the runs

From the repository root, run these commands in a new environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m benchmarks.run --allow-download --output artifacts/clinc-tfidf

.venv/bin/python -m pip install -e '.[neural]'
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false .venv/bin/python -m benchmarks.run \
  --backend setfit --allow-download --output artifacts/clinc-setfit

TINYTASK_NEURAL_MODEL=artifacts/clinc-setfit/model \
  .venv/bin/python -m pytest tests/test_neural.py
```

The output directories must be new.
`--allow-download` permits the public source and the fixed checkpoint downloads.
Omit it after the source and checkpoint exist locally.

Training can use an accelerator selected by SetFit. Inference uses the CPU.
The stored run used an Apple MPS training device.
Other hardware can produce different weights, timings, and boundary decisions.

The CI workflow has an optional manual neural run.
Ordinary CI makes no dataset or model download.

## Metrics and interpretation

Accepted accuracy includes unknown requests as errors when a policy accepts them.
Known accuracy and macro F1 use known requests before abstention.
Known coverage measures the share of supported requests accepted by the policy.
Unknown acceptance is reported for both unknown groups and their aggregate.

The gate uses the Wilson lower bound for accepted accuracy and upper bound for unknown acceptance.
An empty accepted set has no accuracy estimate and cannot pass.
Class scores are not calibrated probabilities of correctness.

## Timing limits

The summary reports training plus calibration time and 100 warm single-request inference samples.
The neural run records the actual training device.
The two runs use different Python environments on one Mac.
Treat these timings as descriptive measurements, not a controlled speed comparison.

## Next comparison

Use three fixed seeds and five independent tasks before a general claim.
Compare policies at matched known coverage and report paired task-level differences.
Include a live teacher with a fixed prompt and reviewed gold labels.
Include teacher failures, fallback cost, and review cost.

For multilingual work, keep translations together and report each language separately.
Add clustered intervals before pooled claims over related records.
