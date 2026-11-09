# Prepared PyPI release

The proposed distribution name is `tiny-task-py`.
The import is `tiny_task`. The command is `tiny-task`.

The repository prepares version `0.1.0` for publication.
It does not publish to PyPI automatically.
A public registry lookup does not reserve a package name.

## Examine the local distributions

```sh
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
.venv/bin/python scripts/examine_distribution.py
python3 -m venv .venv-wheel
.venv-wheel/bin/python -m pip install dist/*.whl
.venv-wheel/bin/python scripts/wheel_smoke.py
```

The wheel test trains and loads a candidate with only runtime dependencies.
The source archive includes the code, tests, plans, example data, and benchmark recipes.
It excludes model weights, caches, environments, and local artifacts.

## First registry publication

1. Examine the final package name in the PyPI account.
2. Create a pending trusted publisher for `eduard-s/tiny-task`.
3. Select a dedicated publication workflow and environment.
4. Add that workflow after the publisher configuration exists.
5. Examine CI and distribution contents at the release revision.
6. Publish the matching version tag.
7. Install the public package in a new environment.
8. Run the public installation through the wheel test.

Read the [PyPI pending publisher guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
No publication tag or automatic upload workflow is part of this preparation task.
