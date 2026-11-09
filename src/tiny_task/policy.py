"""Select abstention thresholds using calibration data only."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .data import UNKNOWN, Record


@dataclass(frozen=True)
class PolicyConfig:
    target_accuracy: float = 0.90
    max_unknown_acceptance: float = 0.10
    min_calibration_accepted: int = 20

    def __post_init__(self):
        for value in (self.target_accuracy, self.max_unknown_acceptance):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Policy rates must be finite and between zero and one.")
        if self.min_calibration_accepted < 1:
            raise ValueError("min_calibration_accepted must be positive.")


@dataclass(frozen=True)
class Policy:
    probability_threshold: float
    similarity_threshold: float
    selected: bool
    calibration_accepted: int
    calibration_accuracy: float | None
    calibration_unknown_acceptance: float | None
    config: dict

    def accepts(self, probabilities, similarities):
        return (
            (np.asarray(probabilities) >= self.probability_threshold)
            & (np.asarray(similarities) >= self.similarity_threshold)
            & (np.asarray(similarities) > 0)
            & self.selected
        )


def select_policy(records: list[Record], labels, probabilities, similarities, config: PolicyConfig):
    expected = np.asarray([r.label for r in records])
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    similarities = np.asarray(similarities)
    known = expected != UNKNOWN
    unknown = ~known
    best = None
    confidence_grid = np.unique(np.r_[0.0, np.quantile(probabilities, np.linspace(0, 1, 21))])
    similarity_grid = np.unique(np.r_[-1.0, np.quantile(similarities, np.linspace(0, 1, 21))])
    for confidence in confidence_grid:
        for similarity in similarity_grid:
            accepted = (
                (probabilities >= confidence) & (similarities >= similarity) & (similarities > 0)
            )
            count = int(accepted.sum())
            if count < config.min_calibration_accepted:
                continue
            accuracy = float(((labels == expected) & accepted).sum() / count)
            unknown_rate = (
                float((accepted & unknown).sum() / unknown.sum()) if unknown.any() else None
            )
            if accuracy < config.target_accuracy:
                continue
            if unknown_rate is not None and unknown_rate > config.max_unknown_acceptance:
                continue
            rank = (int((accepted & known).sum()), accuracy, -float(confidence), -float(similarity))
            if best is None or rank > best[0]:
                best = (rank, confidence, similarity, count, accuracy, unknown_rate)
    if best is None:
        return Policy(1.0, 1.0, False, 0, None, None, asdict(config))
    _, confidence, similarity, count, accuracy, unknown_rate = best
    return Policy(
        float(confidence), float(similarity), True, count, accuracy, unknown_rate, asdict(config)
    )
