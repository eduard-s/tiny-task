"""Local text classification with explicit abstention and evaluation."""

__version__ = "0.1.0"

from .backends import TrainConfig
from .data import UNKNOWN, Record, read_records
from .metrics import GateConfig
from .policy import PolicyConfig
from .task import TinyTask

__all__ = [
    "UNKNOWN",
    "GateConfig",
    "PolicyConfig",
    "Record",
    "TinyTask",
    "TrainConfig",
    "read_records",
]
