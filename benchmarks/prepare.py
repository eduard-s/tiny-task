"""Prepare a pinned CLINC150 development cohort."""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from pathlib import Path

from tiny_task.data import UNKNOWN, Record, normalize

REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
URL = f"https://raw.githubusercontent.com/clinc/oos-eval/{REVISION}/data/data_full.json"
SOURCE_SHA256 = "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"
LABELS = [
    "account_blocked",
    "card_declined",
    "damaged_card",
    "expiration_date",
    "freeze_account",
    "new_card",
    "pin_change",
    "replacement_card_duration",
    "report_fraud",
    "report_lost_card",
]
NEAR_UNKNOWN = ["balance", "credit_limit", "pay_bill", "transactions", "transfer"]


def prepare(
    source: Path, *, allow_download: bool = False, seed: int = 42, train_per_class: int = 20
):
    if not source.exists():
        if not allow_download:
            raise ValueError(
                "The source is absent. Use --allow-download or supply the pinned file."
            )
        source.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            source.write_bytes(response.read())
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("The CLINC150 source hash differs from the pinned source.")
    data = json.loads(raw)
    splits = {"train": [], "calibration": [], "test": []}
    strata = {}
    removed = []
    selected_train = set()
    rng = random.Random(seed)
    for label in LABELS:
        candidates = [i for i, (_, observed) in enumerate(data["train"]) if observed == label]
        selected_train.update(rng.sample(candidates, train_per_class))
    seen_text = set()
    for split, origin in (("train", "train"), ("calibration", "val"), ("test", "test")):
        sources = (
            [origin]
            if split == "train"
            else [origin, "oos_val" if split == "calibration" else "oos_test"]
        )
        for original_split in sources:
            for index, (text, original_label) in enumerate(data[original_split]):
                if split == "train" and index not in selected_train:
                    continue
                known = original_label in LABELS
                near = original_label in NEAR_UNKNOWN
                far = original_split.startswith("oos_")
                if not (known or near or far):
                    continue
                record_id = f"clinc-{original_split}-{index}"
                if normalize(text) in seen_text:
                    removed.append(record_id)
                    continue
                seen_text.add(normalize(text))
                splits[split].append(
                    Record(
                        record_id,
                        text,
                        original_label if known else UNKNOWN,
                        label_source="dataset",
                    )
                )
                strata[record_id] = (
                    "known" if known else ("near_unknown" if near else "far_unknown")
                )
    return (
        splits,
        strata,
        {
            "dataset": "CLINC150 development support slice",
            "url": URL,
            "revision": REVISION,
            "sha256": SOURCE_SHA256,
            "license": "CC-BY-3.0",
            "seed": seed,
            "known_labels": LABELS,
            "near_unknown_labels": NEAR_UNKNOWN,
            "train_per_class_requested": train_per_class,
            "split_rows": {name: len(rows) for name, rows in splits.items()},
            "removed_normalized_duplicates": removed,
            "train_ids": [r.id for r in splits["train"]],
            "calibration_ids": [r.id for r in splits["calibration"]],
            "test_ids": [r.id for r in splits["test"]],
        },
    )
