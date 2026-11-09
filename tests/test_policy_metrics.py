import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tiny_task import UNKNOWN, GateConfig, PolicyConfig, Record
from tiny_task.metrics import measure, release_gate, wilson
from tiny_task.policy import select_policy


@given(st.integers(1, 10000), st.floats(0, 1, allow_nan=False))
def test_wilson_contains_empirical_rate(total, fraction):
    successes = int(total * fraction)
    lower, upper = wilson(successes, total)
    assert -1e-15 <= lower <= successes / total + 1e-15
    assert successes / total - 1e-15 <= upper <= 1 + 1e-15


def test_zero_denominator_has_no_estimate():
    assert wilson(0, 0) is None


def test_selection_rejects_unknown_and_wrong_rows():
    records = [Record("1", "a", "a"), Record("2", "b", "b"), Record("3", "unknown", UNKNOWN)]
    policy = select_policy(
        records,
        ["a", "b", "a"],
        [0.95, 0.90, 0.8],
        [0.8, 0.7, 0.01],
        PolicyConfig(target_accuracy=1, min_calibration_accepted=2),
    )
    assert policy.selected
    assert list(policy.accepts([0.95, 0.90, 0.8], [0.8, 0.7, 0.01])) == [True, True, False]


def test_no_feasible_threshold_means_review():
    policy = select_policy(
        [Record("a", "x", "b")],
        ["a"],
        [1.0],
        [1.0],
        PolicyConfig(target_accuracy=1, min_calibration_accepted=1),
    )
    assert not policy.selected
    assert not policy.accepts([1.0], [1.0])[0]


def test_metrics_keep_unknown_and_abstention_denominators():
    records = [Record("1", "a", "a"), Record("2", "b", "b"), Record("3", "u", UNKNOWN)]
    m = measure(records, ["a", "a", "a"], [True, False, True], ["a", "b"])
    assert m["accepted_accuracy"] == 0.5
    assert m["coverage"] == 2 / 3
    assert m["in_scope_coverage"] == 0.5
    assert m["unknown_acceptance"] == 1
    assert m["raw_in_scope_accuracy"] == 0.5


def test_gate_requires_evidence_even_with_perfect_observed_accuracy():
    records = [Record("1", "x", "a"), Record("2", "u", UNKNOWN)]
    m = measure(records, ["a", "a"], [True, False], ["a", "b"])
    assert release_gate(m, GateConfig())["status"] == "needs_data"


def test_gate_ready_and_reject():
    records = [Record(str(i), f"known {i}", "a") for i in range(200)]
    records += [Record(str(i + 200), f"unknown {i}", UNKNOWN) for i in range(100)]
    predicted = ["a"] * 300
    accepted = [True] * 200 + [False] * 100
    metrics = measure(records, predicted, accepted, ["a"])
    assert release_gate(metrics, GateConfig())["status"] == "ready"
    assert release_gate(metrics, GateConfig(target_accuracy=0.999))["status"] == "reject"


@pytest.mark.parametrize(
    "config",
    [
        lambda: PolicyConfig(target_accuracy=float("nan")),
        lambda: PolicyConfig(min_calibration_accepted=0),
        lambda: GateConfig(min_unknown=0),
        lambda: GateConfig(max_unknown_acceptance=1.1),
    ],
)
def test_invalid_config(config):
    with pytest.raises(ValueError):
        config()


def test_review_everything_does_not_pass():
    records = [Record("a", "text", "a"), Record("b", "unknown", UNKNOWN)]
    metrics = measure(records, ["a", "a"], np.zeros(2, dtype=bool), ["a"])
    assert metrics["accepted_accuracy"] is None
    assert release_gate(metrics, GateConfig())["status"] != "ready"
