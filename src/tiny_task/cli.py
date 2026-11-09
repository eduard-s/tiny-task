"""Command interface for local classification and release evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from . import __version__
from .backends import TrainConfig
from .data import read_records
from .demo import run_demo
from .metrics import GateConfig
from .policy import PolicyConfig
from .task import TinyTask


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train a candidate and select abstention thresholds.")
    train.add_argument("--train", required=True, type=Path)
    train.add_argument("--calibration", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--backend", choices=["tfidf", "setfit"], default="tfidf")
    train.add_argument("--allow-download", action="store_true")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--neural-steps", type=int, default=20)
    train.add_argument("--target-accuracy", type=float, default=0.90)
    train.add_argument("--max-unknown-acceptance", type=float, default=0.10)
    train.add_argument("--min-calibration-accepted", type=int, default=20)
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate a saved candidate on separate test data."
    )
    evaluate.add_argument("--model", required=True, type=Path)
    evaluate.add_argument("--test", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--target-accuracy", type=float, default=0.90)
    evaluate.add_argument("--min-coverage", type=float, default=0.40)
    evaluate.add_argument("--max-unknown-acceptance", type=float, default=0.10)
    evaluate.add_argument("--min-accepted", type=int, default=30)
    evaluate.add_argument("--min-unknown", type=int, default=30)
    predict = commands.add_parser("predict", help="Classify text or a local inbox file.")
    predict.add_argument("--model", required=True, type=Path)
    predict.add_argument("--release-report", type=Path)
    source = predict.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--input", type=Path)
    predict.add_argument("--output", type=Path)
    demo = commands.add_parser("demo", help="Make a local support inbox example.")
    demo.add_argument("--output", required=True, type=Path)
    demo.add_argument("--backend", choices=["tfidf", "setfit"], default="tfidf")
    demo.add_argument("--allow-download", action="store_true")
    return root


def _requests(path: Path):
    with path.open(encoding="utf-8", newline="") as source:
        rows = (
            list(csv.DictReader(source))
            if path.suffix.lower() == ".csv"
            else [json.loads(line) for line in source if line.strip()]
        )
    if not rows or any(
        not isinstance(row, dict)
        or not isinstance(row.get("id"), str)
        or not row["id"].strip()
        or not isinstance(row.get("text"), str)
        for row in rows
    ):
        raise ValueError("Inbox rows need string id and text fields.")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Inbox IDs must be unique.")
    return rows


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "train":
            task = TinyTask.fit(
                read_records(args.train),
                read_records(args.calibration),
                config=TrainConfig(
                    backend=args.backend,
                    seed=args.seed,
                    neural_steps=args.neural_steps,
                    allow_download=args.allow_download,
                ),
                policy_config=PolicyConfig(
                    target_accuracy=args.target_accuracy,
                    max_unknown_acceptance=args.max_unknown_acceptance,
                    min_calibration_accepted=args.min_calibration_accepted,
                ),
            )
            task.save(args.output)
            print(
                json.dumps(
                    {"artifact_id": task.artifact_id, "policy_selected": task.policy.selected}
                )
            )
        elif args.command == "evaluate":
            task = TinyTask.load(args.model)
            report = task.evaluate(
                read_records(args.test),
                gate=GateConfig(
                    target_accuracy=args.target_accuracy,
                    min_in_scope_coverage=args.min_coverage,
                    max_unknown_acceptance=args.max_unknown_acceptance,
                    min_accepted=args.min_accepted,
                    min_unknown=args.min_unknown,
                ),
            )
            task.write_report(args.output, report)
            print(json.dumps(report["gate"]))
            return {"ready": 0, "reject": 1, "needs_data": 3}[report["gate"]["status"]]
        elif args.command == "predict":
            task = TinyTask.load(args.model, release_report=args.release_report)
            if args.text is not None:
                output = json.dumps(task.predict(args.text), ensure_ascii=False) + "\n"
            else:
                rows = _requests(args.input)
                predictions = task.predict([row["text"] for row in rows])
                output = "".join(
                    json.dumps(dict(id=row["id"], **prediction), ensure_ascii=False) + "\n"
                    for row, prediction in zip(rows, predictions, strict=True)
                )
            if args.output:
                with args.output.open("x", encoding="utf-8") as target:
                    target.write(output)
            else:
                print(output, end="")
        elif args.command == "demo":
            print(
                json.dumps(
                    run_demo(args.output, backend=args.backend, allow_download=args.allow_download),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return 0
    except (OSError, ValueError, TypeError, KeyError, ImportError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2
