from __future__ import annotations

from math import inf, isfinite
from typing import Callable

from .model import ConstraintChecker, State, is_valid, objective


def eta_priority_objective(state: State, checker: ConstraintChecker | None = None) -> float:
    if not is_valid(state, checker):
        return -inf
    for package in state:
        if (isinstance(package.eta, bool) or not isinstance(package.eta, (int, float))
                or not isfinite(package.eta)):
            raise ValueError(f"package {package.id}: ETA must be a finite number")
    return float(sum(package.value / (1 + max(0, package.eta)) for package in state.inside_packages))


OBJECTIVES = {"total_value": objective, "eta_priority": eta_priority_objective}
OBJECTIVE_LABELS = {"total_value": "Objective (total value)", "eta_priority": "Skor prioritas ETA"}


def get_objective(name: str) -> Callable[[State, ConstraintChecker | None], float]:
    if not isinstance(name, str) or name not in OBJECTIVES:
        raise ValueError(f"unknown objective: {name}; choose total_value or eta_priority")
    return OBJECTIVES[name]
