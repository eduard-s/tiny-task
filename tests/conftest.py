import pytest

from tiny_task import PolicyConfig, TinyTask
from tiny_task.demo import demo_records


@pytest.fixture
def splits():
    return demo_records()


@pytest.fixture
def task(tmp_path, splits):
    candidate = TinyTask.fit(
        splits["train"],
        splits["calibration"],
        policy_config=PolicyConfig(min_calibration_accepted=8),
    )
    candidate.save(tmp_path / "model")
    return TinyTask.load(tmp_path / "model")
