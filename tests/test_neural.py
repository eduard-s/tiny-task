"""Examine a real trained neural artifact after an explicit benchmark run."""

import os

import numpy as np
import pytest

from tiny_task import TinyTask

pytestmark = pytest.mark.skipif(
    not os.environ.get("TINYTASK_NEURAL_MODEL"), reason="The optional neural artifact is absent."
)


def test_actual_encoder_training_and_export(tmp_path, monkeypatch):
    def refuse_network(*args, **kwargs):
        raise AssertionError("A saved neural artifact must load without a network request.")

    monkeypatch.setattr("requests.sessions.Session.request", refuse_network)
    model = TinyTask.load(os.environ["TINYTASK_NEURAL_MODEL"])
    record = model.backend.training_record
    assert record["steps"] > 0
    assert record["initial_encoder_sha256"] != record["trained_encoder_sha256"]
    assert str(model.backend.encoder.device) == "cpu"
    texts = ["My debit card was declined", "Tell me a story about a penguin"]
    before = model.predict(texts)
    model.save(tmp_path / "model")
    after = TinyTask.load(tmp_path / "model").predict(texts)
    assert [r["decision"] for r in before] == [r["decision"] for r in after]
    assert np.allclose(
        [r["class_probability"] for r in before], [r["class_probability"] for r in after], atol=1e-7
    )


def test_truncated_neural_input_needs_review():
    model = TinyTask.load(os.environ["TINYTASK_NEURAL_MODEL"])
    row = model.predict("My card was declined. " * 100)
    assert row["decision"] == "review"
    assert row["reason"] == "truncated_input"
