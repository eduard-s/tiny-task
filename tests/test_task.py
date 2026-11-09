import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from tiny_task import Record, TinyTask
from tiny_task.backends import NumericHead, TrainConfig


def test_actual_fit_routes_and_abstains(task):
    assert task.predict("Please cancel my subscription")["label"] == "cancel"
    assert task.predict("Write a haiku about a sleepy hamster")["label"] is None


def test_model_serialization_preserves_scores(task, tmp_path):
    before = task.predict(["Please refund my payment", "Reset my password"])
    task.save(tmp_path / "copy")
    after = TinyTask.load(tmp_path / "copy").predict(
        ["Please refund my payment", "Reset my password"]
    )
    assert before == after
    assert not list((tmp_path / "copy").rglob("*.pkl"))


@pytest.mark.parametrize("classes", [2, 3])
def test_numeric_export_matches_sklearn_probabilities(classes):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(90, 5))
    labels = [f"label-{i % classes}" for i in range(90)]
    classifier = LogisticRegression(C=4, max_iter=1000).fit(x, labels)
    head = NumericHead(
        list(classifier.classes_), classifier.coef_, classifier.intercept_, np.zeros((classes, 5))
    )
    predicted, probabilities, _ = head.score(x)
    assert list(predicted) == list(classifier.predict(x))
    assert np.allclose(probabilities, classifier.predict_proba(x).max(axis=1), atol=1e-12)


@pytest.mark.parametrize("text", ["", " ", 123, "x" * 20001])
def test_invalid_request(task, text):
    with pytest.raises(ValueError):
        task.predict([text])


def test_empty_request_batch(task):
    assert task.predict([]) == []


def test_untyped_request_container_is_rejected(task):
    with pytest.raises(ValueError, match="text string"):
        task.predict({"text": "cancel"})


def test_non_object_manifest_is_rejected(task):
    (task.directory / "manifest.json").write_text("[]")
    with pytest.raises(ValueError, match="format"):
        TinyTask.load(task.directory)


def test_save_preserves_existing_directory(task, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        task.save(output)
    assert (output / "keep.txt").read_text() == "keep"


def test_model_tamper_is_detected(task):
    (task.directory / "model.json").write_text("{}")
    with pytest.raises(ValueError, match="hashes"):
        TinyTask.load(task.directory)


def test_model_symlink_is_rejected(task):
    (task.directory / "extra").symlink_to(task.directory / "model.json")
    with pytest.raises(ValueError, match="symbolic"):
        TinyTask.load(task.directory)


def test_manifest_path_escape_is_rejected(task):
    (task.directory / "manifest.json").write_text(
        json.dumps({"format_version": 1, "files": {"../x": "x"}})
    )
    with pytest.raises(ValueError, match="unsafe"):
        TinyTask.load(task.directory)


@pytest.mark.parametrize("split", ["train", "calibration"])
def test_test_data_cannot_reuse_fit_data(task, splits, split):
    with pytest.raises(ValueError, match="leakage"):
        task.evaluate(splits[split])


def test_gold_teacher_labels_cannot_be_used_as_truth(task):
    with pytest.raises(ValueError, match="teacher"):
        task.evaluate([Record("new", "new text", "cancel", label_source="teacher")])


def test_evaluation_records_misses_and_unknowns(task, splits):
    report = task.evaluate(splits["test"])
    assert report["test_records"] == 36
    assert report["metrics"]["unknown"] == 12
    assert report["gate"]["status"] == "needs_data"
    assert len(report["predictions"]) == 36


def test_release_report_must_match_artifact(task, splits, tmp_path):
    report = task.evaluate(splits["test"])
    output = tmp_path / "report.json"
    task.write_report(output, report)
    with pytest.raises(ValueError, match="criteria"):
        TinyTask.load(task.directory, release_report=output)
    report["artifact_id"] = "different"
    output.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="does not match"):
        TinyTask.load(task.directory, release_report=output)


def test_report_cannot_overwrite_model_or_existing_file(task, splits, tmp_path):
    report = task.evaluate(splits["test"])
    with pytest.raises(ValueError):
        task.write_report(task.directory / "model.json", report)
    output = tmp_path / "report.json"
    output.write_text("keep")
    with pytest.raises(FileExistsError):
        task.write_report(output, report)
    assert output.read_text() == "keep"


def test_unsaved_model_cannot_claim_bound_evaluation(splits):
    task = TinyTask.fit(splits["train"], splits["calibration"])
    with pytest.raises(ValueError, match="Save"):
        task.evaluate(splits["test"])


@pytest.mark.parametrize(
    "config", [lambda: TrainConfig(backend="bad"), lambda: TrainConfig(neural_steps=0)]
)
def test_bad_train_config(config):
    with pytest.raises(ValueError):
        config()
