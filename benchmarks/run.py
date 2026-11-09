"""Compare acceptance policies on the same trained classifier."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tiny_task import GateConfig, PolicyConfig, TinyTask, TrainConfig
from tiny_task.backends import write_json
from tiny_task.metrics import measure, release_gate
from tiny_task.policy import select_policy

from .prepare import prepare


def run(source, output, *, backend="tfidf", allow_download=False, seed=42, neural_steps=20):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    splits, strata, provenance = prepare(source, allow_download=allow_download, seed=seed)
    task = TinyTask.fit(
        splits["train"],
        splits["calibration"],
        config=TrainConfig(
            backend=backend, seed=seed, neural_steps=neural_steps, allow_download=allow_download
        ),
    )
    task.save(output / "model")
    task = TinyTask.load(output / "model")
    report = task.evaluate(splits["test"])
    task.write_report(output / "evaluation.json", report)
    cal_labels, cal_prob, cal_support = task.backend.score([r.text for r in splits["calibration"]])
    confidence_policy = select_policy(
        splits["calibration"], cal_labels, cal_prob, np.ones(len(cal_prob)), PolicyConfig()
    )
    test = splits["test"]
    labels, probabilities, support = task.backend.score([r.text for r in test])
    policies = {
        "always_predict": np.ones(len(test), dtype=bool),
        "confidence_only": confidence_policy.accepts(probabilities, np.ones(len(test))),
        "tinytask": task.policy.accepts(probabilities, support),
    }
    results = {}
    for name, accepted in policies.items():
        metrics = measure(test, labels, accepted, task.backend.head.labels)
        unknown_strata = {}
        for stratum in ("near_unknown", "far_unknown"):
            mask = np.asarray([strata[r.id] == stratum for r in test])
            unknown_strata[stratum] = {
                "records": int(mask.sum()),
                "accepted": int((mask & accepted).sum()),
                "acceptance": float(accepted[mask].mean()) if mask.any() else None,
            }
        results[name] = {
            "metrics": metrics,
            "unknown_strata": unknown_strata,
            "gate": release_gate(metrics, GateConfig()),
        }
    timings = []
    for record in test[:100]:
        started = time.perf_counter()
        task.predict(record.text)
        timings.append(time.perf_counter() - started)
    summary = {
        "format_version": 1,
        "cohort": "development",
        "backend": backend,
        "dataset": provenance,
        "artifact_id": task.artifact_id,
        "train_config": task.provenance["train_config"],
        "training_seconds": task.provenance["training_and_selection_seconds"],
        "backend_training_record": getattr(task.backend, "training_record", None),
        "policy": asdict(task.policy),
        "confidence_policy": asdict(confidence_policy),
        "results": results,
        "latency": {
            "unit": "seconds",
            "warm_samples": len(timings),
            "p50": float(np.median(timings)),
            "p95": float(np.quantile(timings, 0.95)),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "machine": platform.machine(),
            **task.provenance["versions"],
        },
        "unrun_baselines": ["Prompt2Model workflow", "live LLM teacher", "ONNX deployment"],
        "source_hashes": {
            path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(
                [*Path("src/tiny_task").glob("*.py"), *Path("benchmarks").glob("*.py")]
            )
        },
    }
    if backend == "setfit":
        summary["environment"].update(
            {
                name: importlib.metadata.version(name)
                for name in ("setfit", "sentence-transformers", "transformers", "torch")
            }
        )
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/data/clinc150.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["tfidf", "setfit"], default="tfidf")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neural-steps", type=int, default=20)
    args = parser.parse_args()
    result = run(
        args.source,
        args.output,
        backend=args.backend,
        allow_download=args.allow_download,
        seed=args.seed,
        neural_steps=args.neural_steps,
    )
    print(json.dumps({name: row["gate"] for name, row in result["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
