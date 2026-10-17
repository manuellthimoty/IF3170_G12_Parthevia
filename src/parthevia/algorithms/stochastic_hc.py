from __future__ import annotations

import random

from ..model import ConstraintChecker, State, objective
from ..neighbors import enumerate_neighbors, random_neighbor
from ..result import SearchResult


def _uphill_neighbors(state: State, rng: random.Random, neighborhood: str, samples: int) -> list[State]:
    checker = ConstraintChecker(state.truck)
    current_value = state.total_value
    if neighborhood == "full":
        candidates = enumerate_neighbors(state, checker)
    elif neighborhood == "sample":
        candidates = [neighbor for neighbor in (random_neighbor(state, checker, rng) for _ in range(samples)) if neighbor]
    else:
        raise ValueError("neighborhood must be 'sample' or 'full'")
    return [neighbor for neighbor in candidates if neighbor.total_value > current_value]


def stochastic_hill_climbing(
    initial_state: State,
    rng: random.Random | None = None,
    max_iterations: int = 1000,
    neighborhood: str = "sample",
    samples: int = 100,
) -> SearchResult:
    rng = rng or random.Random()
    current = initial_state
    history = [current.total_value]
    iteration = 0
    while iteration < max_iterations:
        uphill = _uphill_neighbors(current, rng, neighborhood, samples)
        if not uphill:
            break
        current = rng.choice(uphill)
        iteration += 1
        history.append(current.total_value)

    checker = ConstraintChecker(initial_state.truck)
    return SearchResult(
        algorithm="stochastic_hill_climbing",
        initial_state=initial_state,
        final_state=current,
        initial_objective=objective(initial_state, checker),
        final_objective=objective(current, checker),
        iterations=iteration,
        history=history,
        params={"max_iterations": max_iterations, "neighborhood": neighborhood, "samples": samples},
        extra={"stopped_at_local_optimum": iteration < max_iterations},
    )
