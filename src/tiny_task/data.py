"""Read labeled records and reject known split leakage."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

UNKNOWN = "__unknown__"
SOURCES = {"human", "dataset", "authored", "teacher"}


@dataclass(frozen=True)
class Record:
    id: str
    text: str
    label: str
    group_id: str | None = None
    label_source: str = "human"

    def __post_init__(self):
        for field in ("id", "text", "label"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Record {field} must be a nonempty string.")
        if len(self.text) > 20000:
            raise ValueError("A record must contain at most 20,000 characters.")
        if self.group_id is not None and (
            not isinstance(self.group_id, str) or not self.group_id.strip()
        ):
            raise ValueError("group_id must be a nonempty string or null.")
        if self.label_source not in SOURCES:
            raise ValueError(f"Unknown label source: {self.label_source}")


def digest(value) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def read_records(path: str | Path) -> list[Record]:
    records = []
    with Path(path).open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("Each line must be a JSON object.")
                records.append(Record(**value))
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid record at line {number}: {error}") from error
    if not records:
        raise ValueError("The data file contains no records.")
    return records


def fingerprints(records: list[Record]) -> dict[str, list[str]]:
    return {
        "id": sorted({digest(r.id) for r in records}),
        "text": sorted({digest(normalize(r.text)) for r in records}),
        "group": sorted({digest(r.group_id) for r in records if r.group_id is not None}),
    }


def examine_records(records: list[Record], *, gold: bool = False) -> None:
    if not records:
        raise ValueError("Each split needs records.")
    if len({r.id for r in records}) != len(records):
        raise ValueError("Record IDs must be unique within each split.")
    groups = [r.group_id for r in records if r.group_id is not None]
    if gold and len(set(groups)) != len(groups):
        raise ValueError("Calibration and test need at most one record per known group.")
    text_labels: dict[str, str] = {}
    for record in records:
        key = normalize(record.text)
        if key in text_labels and text_labels[key] != record.label:
            raise ValueError("Identical normalized text has conflicting labels.")
        if key in text_labels:
            raise ValueError("Each split must contain unique normalized text.")
        text_labels[key] = record.label
        if gold and record.label_source == "teacher":
            raise ValueError("Calibration and test labels cannot use unreviewed teacher labels.")


def examine_overlap(left: dict[str, list[str]], right: dict[str, list[str]]) -> None:
    for kind in ("id", "text", "group"):
        if set(left[kind]) & set(right[kind]):
            raise ValueError(f"Split leakage detected through {kind} overlap.")


def examine_splits(train: list[Record], calibration: list[Record]) -> list[str]:
    examine_records(train)
    examine_records(calibration, gold=True)
    labels = sorted({r.label for r in train})
    if UNKNOWN in labels or len(labels) < 2:
        raise ValueError("Training needs at least two known labels and no unknown rows.")
    if not {r.label for r in calibration} <= set(labels) | {UNKNOWN}:
        raise ValueError("Calibration contains a label absent from training.")
    if not set(labels) <= {r.label for r in calibration}:
        raise ValueError("Calibration needs at least one row for every training label.")
    examine_overlap(fingerprints(train), fingerprints(calibration))
    return labels


def records_digest(records: list[Record]) -> str:
    return digest([asdict(r) for r in records])
