# Development

Use Python 3.12 for type analysis.
CI runs the core tests on Python 3.11 through 3.14.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src/tiny_task
.venv/bin/pytest --cov=tiny_task --cov-report=term-missing
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
.venv/bin/python scripts/examine_distribution.py
```

## Behavior changes

Add a failing case before a change to threshold selection, data partitions, or report meaning.
Preserve old benchmark reports when behavior changes.
Do not change release targets after a result to produce a passing status.

## Dataset contributions

Record the source, license, revision, and hashes.
Keep related records in one split.
Separate authored examples from independent data.
Keep confidential records and model weights out of Git.

## Neural tests

The core suite makes no model download.
The optional neural run needs explicit source and checkpoint downloads.
Use the benchmark recipe and the neural integration tests after that run.

## Reports and claims

Preserve rejected candidates, unknown requests, and execution errors.
Report known coverage beside accepted accuracy.
Describe only the measured cohort and the actual model backend.
Do not claim live teacher savings from a local-only run.
