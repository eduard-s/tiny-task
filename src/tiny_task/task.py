"""Fit, preserve, and evaluate a local classification candidate."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
from dataclasses import asdict
from pathlib import Path, PurePosixPath

import numpy as np

from . import __version__
from .backends import BACKENDS, TrainConfig, write_json
from .data import (
    UNKNOWN,
    Record,
    digest,
    examine_overlap,
    examine_records,
    examine_splits,
    fingerprints,
    records_digest,
)
from .metrics import GateConfig, measure, release_gate
from .policy import Policy, PolicyConfig, select_policy


def _file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _artifact_files(directory: Path) -> dict[str, str]:
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Model artifacts cannot contain symbolic links.")
        if path.is_file() and path.relative_to(directory).as_posix() != "manifest.json":
            files[path.relative_to(directory).as_posix()] = _file_hash(path)
    return files


class TinyTask:
    """A trained candidate with an explicit abstention policy."""

    def __init__(self, backend, policy: Policy, provenance: dict):
        self.backend = backend
        self.policy = policy
        self.provenance = provenance
        self.artifact_id: str | None = None
        self.directory: Path | None = None

    @classmethod
    def fit(
        cls,
        train: list[Record],
        calibration: list[Record],
        *,
        config: TrainConfig | None = None,
        policy_config: PolicyConfig | None = None,
    ):
        config = config or TrainConfig()
        policy_config = policy_config or PolicyConfig()
        labels = examine_splits(train, calibration)
        started = time.perf_counter()
        backend = BACKENDS[config.backend].fit(train, labels, config)
        predicted, probability, similarity = backend.score([r.text for r in calibration])
        policy = select_policy(calibration, predicted, probability, similarity, policy_config)
        return cls(
            backend,
            policy,
            {
                "train_config": asdict(config),
                "train_rows": len(train),
                "calibration_rows": len(calibration),
                "train_digest": records_digest(train),
                "calibration_digest": records_digest(calibration),
                "train_fingerprints": fingerprints(train),
                "calibration_fingerprints": fingerprints(calibration),
                "label_sources": sorted({r.label_source for r in train}),
                "training_and_selection_seconds": time.perf_counter() - started,
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "scipy", "scikit-learn")
                },
                "python": platform.python_version(),
                "tiny_task_version": __version__,
            },
        )

    def predict(self, texts: str | list[str]):
        if not isinstance(texts, (str, list)):
            raise ValueError("Prediction needs a text string or a list of text strings.")
        single = isinstance(texts, str)
        batch = [texts] if single else texts
        if not batch:
            return []
        for text in batch:
            if not isinstance(text, str) or not text.strip() or len(text) > 20000:
                raise ValueError("Each request needs 1 to 20,000 characters of text.")
        labels, probabilities, similarities = self.backend.score(batch)
        accepted = self.policy.accepts(probabilities, similarities)
        truncated = (
            self.backend.truncated(batch)
            if hasattr(self.backend, "truncated")
            else [False] * len(batch)
        )
        accepted &= ~np.asarray(truncated)
        predictions = []
        for label, probability, similarity, accept, was_truncated in zip(
            labels, probabilities, similarities, accepted, truncated, strict=True
        ):
            reason = "accepted"
            if not self.policy.selected:
                reason = "no_calibration_policy"
            elif was_truncated:
                reason = "truncated_input"
            elif similarity <= 0:
                reason = "no_training_support"
            elif probability < self.policy.probability_threshold:
                reason = "low_class_probability"
            elif similarity < self.policy.similarity_threshold:
                reason = "low_training_support"
            predictions.append(
                {
                    "label": str(label) if accept else None,
                    "proposed_label": str(label),
                    "decision": "accept" if accept else "review",
                    "class_probability": float(probability),
                    "support_similarity": float(similarity),
                    "reason": reason,
                }
            )
        return predictions[0] if single else predictions

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        try:
            backend_state = self.backend.export(directory)
            write_json(
                directory / "model.json",
                {
                    "format_version": 1,
                    "backend": self.backend.name,
                    "backend_state": backend_state,
                    "policy": asdict(self.policy),
                    "provenance": self.provenance,
                },
            )
            files = _artifact_files(directory)
            manifest = {"format_version": 1, "files": files}
            write_json(directory / "manifest.json", manifest)
        except Exception:
            shutil.rmtree(directory)
            raise
        self.artifact_id = digest(manifest)
        self.directory = directory.resolve()
        return directory

    @classmethod
    def load(cls, directory: str | Path, *, release_report: str | Path | None = None):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
            raise ValueError("Unsupported artifact format.")
        for name in manifest["files"]:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("The artifact manifest contains an unsafe path.")
        if _artifact_files(directory) != manifest["files"]:
            raise ValueError("Artifact hashes or file names differ from the manifest.")
        state = json.loads((directory / "model.json").read_text())
        if (
            not isinstance(state, dict)
            or state.get("format_version") != 1
            or state.get("backend") not in BACKENDS
        ):
            raise ValueError("Unsupported model format or backend.")
        policy = Policy(**state["policy"])
        if (
            not np.isfinite([policy.probability_threshold, policy.similarity_threshold]).all()
            or not 0 <= policy.probability_threshold <= 1
            or not -1 <= policy.similarity_threshold <= 1
            or not isinstance(policy.selected, bool)
        ):
            raise ValueError("Invalid abstention thresholds.")
        task = cls(
            BACKENDS[state["backend"]].load(state["backend_state"], directory),
            policy,
            state["provenance"],
        )
        task.artifact_id = digest(manifest)
        task.directory = directory.resolve()
        if release_report is not None:
            report = json.loads(Path(release_report).read_text())
            if report.get("artifact_id") != task.artifact_id or report.get("format_version") != 1:
                raise ValueError("The release report does not match this artifact.")
            computed = release_gate(report["metrics"], GateConfig(**report["gate"]["criteria"]))
            if computed["status"] != "ready" or report["gate"] != computed:
                raise ValueError("The candidate did not pass its recorded release criteria.")
        return task

    def evaluate(self, test: list[Record], *, gate: GateConfig | None = None) -> dict:
        if self.artifact_id is None:
            raise ValueError("Save the candidate before evaluation to bind its artifact identity.")
        examine_records(test, gold=True)
        test_fingerprints = fingerprints(test)
        for split in ("train", "calibration"):
            examine_overlap(self.provenance[f"{split}_fingerprints"], test_fingerprints)
        if not {r.label for r in test} <= set(self.backend.head.labels) | {UNKNOWN}:
            raise ValueError(
                "Test data contains an unknown class without the reserved unknown label."
            )
        started = time.perf_counter()
        rows = self.predict([r.text for r in test])
        elapsed = time.perf_counter() - started
        metrics = measure(
            test,
            [r["proposed_label"] for r in rows],
            [r["decision"] == "accept" for r in rows],
            self.backend.head.labels,
        )
        return {
            "format_version": 1,
            "artifact_id": self.artifact_id,
            "test_digest": records_digest(test),
            "test_records": len(test),
            "metrics": metrics,
            "gate": release_gate(metrics, gate or GateConfig()),
            "policy": asdict(self.policy),
            "provenance": self.provenance,
            "runtime": {"batch_seconds": elapsed, "records_per_second": len(test) / elapsed},
            "predictions": [
                dict(id=r.id, expected_label=r.label, **p) for r, p in zip(test, rows, strict=True)
            ],
        }

    def write_report(self, path: str | Path, report: dict):
        path = Path(path)
        if self.directory is not None and path.resolve().is_relative_to(self.directory):
            raise ValueError("Reports must stay outside the model artifact.")
        if path.exists():
            raise FileExistsError("The report output already exists.")
        write_json(path, report)
