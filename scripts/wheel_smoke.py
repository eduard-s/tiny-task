"""Exercise the installed wheel with a local training workflow."""

from pathlib import Path
from tempfile import TemporaryDirectory

from tiny_task import TinyTask
from tiny_task.demo import run_demo

with TemporaryDirectory(prefix="tinytask-wheel-") as temporary:
    root = Path(temporary) / "demo"
    result = run_demo(root)
    assert result["gate"] == "needs_data", result
    task = TinyTask.load(root / "model")
    assert task.predict("Please cancel my subscription")["label"] == "cancel"
    assert task.predict("Write a haiku about a sleepy hamster")["decision"] == "review"
print(
    "The installed wheel trains, exports, reloads, routes a ticket, and reviews an unknown request."
)
