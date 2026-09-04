"""The eight gates that decide whether a regenerated texture ships.

Two halves, deliberately separate:

* :mod:`~topotexgen.gates.metrics` measures — pure functions over arrays and
  images, no thresholds, no verdicts. Every one is testable on synthetic input
  and none of them needs a GPU.
* :mod:`~topotexgen.gates.verdict` decides — pure functions over a dict of
  measurements plus a threshold set loaded from ``configs/gates.yaml``.

Keeping them apart is what makes the thresholds auditable: a re-calibration
changes a YAML file with an evidence line, and the decision can be replayed
over stored measurements without re-rendering anything.
"""
from topotexgen.gates.thresholds import Thresholds, load_thresholds
from topotexgen.gates.verdict import GateResult, verdict

__all__ = ["GateResult", "Thresholds", "load_thresholds", "verdict"]
