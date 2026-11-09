import json

import pytest

from tiny_task import UNKNOWN, Record, read_records
from tiny_task.data import examine_records, examine_splits, normalize, records_digest


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", ""),
        ("text", " "),
        ("label", 4),
        ("group_id", ""),
        ("label_source", "unexamined"),
        ("text", "x" * 20001),
    ],
)
def test_invalid_record(field, value):
    row = dict(id="a", text="message", label="label")
    row[field] = value
    with pytest.raises(ValueError):
        Record(**row)


@pytest.mark.parametrize(
    "line", ["[]", "{bad", '{"id":"a"}', '{"id":"a","text":"x","label":"y","extra":1}']
)
def test_bad_jsonl(tmp_path, line):
    path = tmp_path / "data.jsonl"
    path.write_text(line)
    with pytest.raises(ValueError, match="line 1"):
        read_records(path)


def test_empty_file(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("\n")
    with pytest.raises(ValueError, match="no records"):
        read_records(path)


def test_round_trip(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("\n" + json.dumps(dict(id="a", text="Привет", label="greet")) + "\n")
    assert read_records(path) == [Record("a", "Привет", "greet")]


@pytest.mark.parametrize("kind", ["id", "text", "group_id"])
def test_cross_split_leakage(kind):
    train = [
        Record("a", "cancel subscription", "cancel", "g1"),
        Record("b", "password reset", "password", "g2"),
    ]
    row = dict(id="c", text="end my plan", label="cancel", group_id="g3")
    row[kind] = {"id": "a", "text": "  CANCEL  SUBSCRIPTION ", "group_id": "g1"}[kind]
    calibration = [Record(**row), Record("d", "forgot my login", "password", "g4")]
    with pytest.raises(ValueError, match="leakage"):
        examine_splits(train, calibration)


def test_compatibility_unicode_normalization():
    assert normalize("ＡＢＣ\tHi") == "abc hi"


@pytest.mark.parametrize(
    "records",
    [
        [Record("a", "same", "x"), Record("a", "different", "x")],
        [Record("a", "same", "x"), Record("b", "SAME", "x")],
        [Record("a", "same", "x"), Record("b", "SAME", "y")],
    ],
)
def test_duplicate_rows_cannot_inflate_counts(records):
    with pytest.raises(ValueError):
        examine_records(records)


def test_teacher_labels_need_review_for_evaluation():
    with pytest.raises(ValueError, match="teacher"):
        examine_records([Record("a", "text", "x", label_source="teacher")], gold=True)


def test_training_rejects_unknown(splits):
    with pytest.raises(ValueError, match="no unknown"):
        examine_splits(
            splits["train"] + [Record("xx", "new unknown", UNKNOWN)], splits["calibration"]
        )


def test_missing_calibration_class(splits):
    with pytest.raises(ValueError, match="every training label"):
        examine_splits(splits["train"], [r for r in splits["calibration"] if r.label != "cancel"])


def test_records_hash_includes_labels():
    assert records_digest([Record("a", "text", "x")]) != records_digest([Record("a", "text", "y")])


def test_related_gold_rows_cannot_inflate_independent_sample_counts():
    with pytest.raises(ValueError, match="one record"):
        examine_records(
            [Record("a", "one text", "x", "group"), Record("b", "another text", "x", "group")],
            gold=True,
        )
