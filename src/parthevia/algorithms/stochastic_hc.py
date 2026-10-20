from __future__ import annotations

import random

from ..model import ConstraintChecker, State, objective
from ..neighbors import random_neighbor
from ..result import SearchResult


def stochastic_hill_climbing(
    initial_state: State,
    rng: random.Random | None = None,
    max_iterations: int = 1000,
) -> SearchResult:
    rng = rng or random.Random()
    checker = ConstraintChecker(initial_state.truck)
    current = initial_state
    history = [current.total_value]
    improvements = 0
    for _ in range(max_iterations):
        neighbor = random_neighbor(current, checker, rng)
        if neighbor is not None and neighbor.total_value > current.total_value:
            current = neighbor
            improvements += 1
        history.append(current.total_value)

    return SearchResult(
        algorithm="stochastic_hill_climbing",
        initial_state=initial_state,
        final_state=current,
        initial_objective=objective(initial_state, checker),
        final_objective=objective(current, checker),
        iterations=max_iterations,
        history=history,
        params={"max_iterations": max_iterations},
        extra={"improvements": improvements},
    )
