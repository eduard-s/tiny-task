# PyPI release procedure

The distribution name is `tiny-task-py`.
The import is `tiny_task`. The command is `tiny-task`.

The `Publish Python` workflow examines distributions before publication.
A manual workflow run examines the package without an upload.
A matching version tag starts the PyPI upload.

## Trusted publisher

The PyPI publisher uses this GitHub binding:

| Field | Value |
| --- | --- |
| PyPI project | `tiny-task-py` |
| GitHub owner | `eduard-s` |
| GitHub repository | `tiny-task` |
| Workflow filename | `publish-python.yml` |
| GitHub environment | `pypi` |

For the first release, create this pending publisher in the PyPI account.
Read the [PyPI pending publisher guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
The workflow uses GitHub OIDC. It does not need a stored PyPI API token.
Only the upload job gets `id-token: write` permission.

## Examine the local distributions

Run these commands from the repository:

```sh
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy src/tiny_task
.venv/bin/python -m pytest --cov=tiny_task --cov-report=term-missing
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
.venv/bin/python scripts/examine_distribution.py
python3 -m venv .venv-wheel
.venv-wheel/bin/python -m pip install --force-reinstall dist/*.whl
.venv-wheel/bin/python scripts/wheel_smoke.py
```

The wheel test trains and loads a candidate with only runtime dependencies.
The source archive includes the code, tests, plans, example data, and benchmark recipes.
It excludes model weights, caches, environments, and local artifacts.
The optional neural tests need the neural dependencies and a trained local model.
The core release workflow does not download neural weights or public datasets.

## Publish a release

1. Update the version in `pyproject.toml` and `src/tiny_task/__init__.py`.
2. Update the changelog.
3. Push the release commit to `main`.
4. Examine the successful CI run at that commit.
5. Make sure that the PyPI publisher matches the binding in this document.
6. Create the matching version tag.
7. Push the tag.
8. Examine the `Publish Python` workflow result.
9. Install the public package in a new environment.
10. Run the public installation through the wheel test.

For version `0.1.0`, use these commands at the examined commit:

```sh
git tag -a v0.1.0 -m 'Release TinyTask 0.1.0'
git push origin v0.1.0
```

The workflow refuses a tag that differs from the package version.
PyPI does not allow replacement of an uploaded distribution.
Use a new version for a later package change.

## Examine the public installation

```sh
python3 -m venv .venv-published
.venv-published/bin/python -m pip install --no-cache-dir \
  --index-url https://pypi.org/simple 'tiny-task-py==0.1.0'
.venv-published/bin/python -m pip check
.venv-published/bin/tiny-task --version
.venv-published/bin/python scripts/wheel_smoke.py
```

The final command uses the installed package and authored demo data.
It does not need example files from the source archive.
