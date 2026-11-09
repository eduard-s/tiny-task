import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.prepare import prepare


def test_no_implicit_dataset_download(tmp_path):
    with pytest.raises(ValueError, match="absent"):
        prepare(tmp_path / "missing.json")


def test_wrong_public_source_hash_is_rejected(tmp_path):
    path = tmp_path / "source.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        prepare(path)


def test_published_summaries_match_the_current_recipe():
    for backend in ("tfidf", "setfit"):
        report = json.loads(Path(f"benchmarks/results-{backend}.json").read_text())
        for name, expected in report["source_hashes"].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
        for policy in report["results"].values():
            metrics = policy["metrics"]
            assert metrics["accepted_correct"] + metrics["accepted_errors"] == metrics["accepted"]
            assert metrics["in_scope"] + metrics["unknown"] == metrics["records"]
