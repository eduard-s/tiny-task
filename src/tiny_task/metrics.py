"""Measure accepted errors, coverage, and unsupported requests."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import f1_score

from .data import UNKNOWN, Record


def wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


@dataclass(frozen=True)
class GateConfig:
    target_accuracy: float = 0.90
    min_in_scope_coverage: float = 0.40
    max_unknown_acceptance: float = 0.10
    min_accepted: int = 30
    min_unknown: int = 30
    min_per_label: int = 1

    def __post_init__(self):
        rates = (self.target_accuracy, self.min_in_scope_coverage, self.max_unknown_acceptance)
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in rates):
            raise ValueError("Gate rates must be finite and between zero and one.")
        if min(self.min_accepted, self.min_unknown, self.min_per_label) < 1:
            raise ValueError("Gate sample counts must be positive.")


def measure(records: list[Record], predicted, accepted, labels: list[str]) -> dict:
    expected = np.asarray([r.label for r in records])
    predicted = np.asarray(predicted)
    accepted = np.asarray(accepted, dtype=bool)
    known = expected != UNKNOWN
    unknown = ~known
    correct = predicted == expected
    n = len(records)
    count = int(accepted.sum())
    successes = int((correct & accepted).sum())
    unknown_count = int(unknown.sum())
    unknown_accepted = int((unknown & accepted).sum())
    return {
        "records": n,
        "in_scope": int(known.sum()),
        "unknown": unknown_count,
        "accepted": count,
        "accepted_correct": successes,
        "accepted_errors": count - successes,
        "coverage": count / n,
        "in_scope_coverage": float((known & accepted).sum() / known.sum()) if known.any() else None,
        "accepted_accuracy": successes / count if count else None,
        "accepted_accuracy_wilson95": wilson(successes, count),
        "unknown_accepted": unknown_accepted,
        "unknown_acceptance": unknown_accepted / unknown_count if unknown_count else None,
        "unknown_acceptance_wilson95": wilson(unknown_accepted, unknown_count),
        "raw_in_scope_accuracy": float(correct[known].mean()) if known.any() else None,
        "raw_macro_f1": float(
            f1_score(
                expected[known], predicted[known], labels=labels, average="macro", zero_division=0
            )
        )
        if known.any()
        else None,
        "per_label": {
            label: {
                "test_rows": int((expected == label).sum()),
                "accepted": int(((expected == label) & accepted).sum()),
                "accepted_correct": int(((expected == label) & accepted & correct).sum()),
            }
            for label in labels
        },
    }


def release_gate(metrics: dict, config: GateConfig) -> dict:
    reasons = []
    insufficient = []
    if metrics["accepted"] < config.min_accepted:
        insufficient.append("Too few accepted test records.")
    if metrics["unknown"] < config.min_unknown:
        insufficient.append("Too few unknown test records.")
    if any(v["test_rows"] < config.min_per_label for v in metrics["per_label"].values()):
        insufficient.append("A training label has too few test records.")
    accuracy_interval = metrics["accepted_accuracy_wilson95"]
    unknown_interval = metrics["unknown_acceptance_wilson95"]
    if accuracy_interval is None or accuracy_interval[0] < config.target_accuracy:
        reasons.append("The accepted accuracy lower bound misses the target.")
    if unknown_interval is None or unknown_interval[1] > config.max_unknown_acceptance:
        reasons.append("The unknown acceptance upper bound exceeds the limit.")
    coverage = metrics["in_scope_coverage"]
    if coverage is None or coverage < config.min_in_scope_coverage:
        reasons.append("The in-scope coverage misses the target.")
    status = "needs_data" if insufficient else ("reject" if reasons else "ready")
    return {"status": status, "reasons": insufficient + reasons, "criteria": asdict(config)}
